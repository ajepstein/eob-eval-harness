"""Run the task suite against one or more adapters and print a summary table.

Usage:
    python scripts/run_eval.py --adapter anthropic --adapter openai --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from harness.adapters import available_adapters, get_adapter
from harness.cache import ResponseCache
from harness.runner import run_tasks
from harness.scorers.fields import FieldScorer
from harness.scorers.schema import SchemaScorer
from harness.tasks import load_tasks
from harness.types import RunSummary


async def main_async(args: argparse.Namespace) -> list[RunSummary]:
    tasks = load_tasks(args.tasks, categories=args.category, limit=args.limit)
    cache = ResponseCache(enabled=not args.no_cache)

    summaries = []
    for name in args.adapter:
        adapter = get_adapter(name, model_alias=args.model)
        summary = await run_tasks(
            tasks,
            adapter,
            prompt_name=args.prompt,
            concurrency=args.concurrency,
            cache=cache,
            max_tokens=args.max_tokens,
            scorers=[SchemaScorer(), FieldScorer()],
        )
        summaries.append(summary)
    return summaries


def render(summaries: list[RunSummary], console: Console) -> None:
    table = Table(title="Eval run")
    table.add_column("adapter", no_wrap=True)
    table.add_column("model", no_wrap=True)
    table.add_column("ok", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("cached", justify="right")
    table.add_column("schema", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("cost $", justify="right", no_wrap=True)
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("wall s", justify="right")

    for s in summaries:
        table.add_row(
            s.adapter_name,
            s.model_id,
            str(s.succeeded),
            str(s.failed),
            str(s.cached),
            f"{s.schema_pass_rate:.2f}",
            f"{s.mean_f1:.3f}",
            f"{s.total_cost_usd:.6f}",
            f"{s.latency_p50_ms:.0f}",
            f"{s.latency_p95_ms:.0f}",
            f"{s.wall_clock_seconds:.1f}",
        )

    console.print(table)
    render_categories(summaries, console)

    for s in summaries:
        for r in s.results:
            if r.error:
                console.print(f"[red]{s.adapter_name} {r.task_id}: {r.error}[/red]")


def render_categories(summaries: list[RunSummary], console: Console) -> None:
    """Mean F1 per category, one column per adapter."""
    categories = sorted(
        {c for s in summaries for c in s.mean_f1_by_category}
    )
    if not categories:
        return

    table = Table(title="Mean F1 by category")
    table.add_column("category", no_wrap=True)
    for s in summaries:
        table.add_column(s.adapter_name, justify="right")

    for category in categories:
        row = [category]
        for s in summaries:
            value = s.mean_f1_by_category.get(category)
            row.append("—" if value is None else f"{value:.3f}")
        table.add_row(*row)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        action="append",
        default=None,
        help=f"Adapter to run; repeatable. Valid: {', '.join(available_adapters())}",
    )
    parser.add_argument("--tasks", default="tasks/", help="Task directory")
    parser.add_argument("--category", action="append", default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=3, help="Max tasks to run (default 3)")
    parser.add_argument("--prompt", default="extract_v1", help="Prompt template name")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--model", default=None, help="Override MODELS alias")
    parser.add_argument("--no-cache", action="store_true", help="Disable the response cache")
    args = parser.parse_args()

    if not args.adapter:
        args.adapter = ["anthropic"]

    summaries = asyncio.run(main_async(args))
    render(summaries, Console())


if __name__ == "__main__":
    main()
