"""Run the task suite against one or more adapters and print a summary table.

Usage:
    python scripts/run_eval.py --adapter anthropic --adapter openai --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from harness.adapters import available_adapters, get_adapter
from harness.cache import ResponseCache
from harness.report import render_category_table, render_diff, render_run_table
from harness.runner import run_tasks
from harness.scorers.fields import FieldScorer
from harness.scorers.judge import JudgeScorer
from harness.scorers.schema import SchemaScorer
from harness.store import (
    DEFAULT_DB_PATH,
    compare_runs,
    list_runs,
    load_run,
    rescore_run,
    save_run,
)
from harness.tasks import load_tasks
from harness.types import RunSummary, Task


async def main_async(args: argparse.Namespace, tasks: list[Task]) -> list[RunSummary]:
    cache = ResponseCache(enabled=not args.no_cache)

    scorers = [SchemaScorer(), FieldScorer()]
    if args.judge:
        scorers.append(
            JudgeScorer(
                adapter=get_adapter(args.judge, model_alias=args.judge_model),
                only_near_misses=not args.judge_all,
                cache=cache,
            )
        )

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
            scorers=scorers,
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
    table.add_column("judge F1", justify="right")
    table.add_column("cost $", justify="right", no_wrap=True)
    table.add_column("judge $", justify="right", no_wrap=True)
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
            f"{s.mean_judge_f1:.3f}" if s.judge_prompt_hash else "—",
            f"{s.total_cost_usd:.6f}",
            f"{s.judge_cost_usd:.6f}" if s.judge_prompt_hash else "—",
            f"{s.latency_p50_ms:.0f}",
            f"{s.latency_p95_ms:.0f}",
            f"{s.wall_clock_seconds:.1f}",
        )

    console.print(table)
    for s in summaries:
        if s.judge_prompt_hash:
            console.print(
                f"  [dim]judge: {s.judge_calls} call(s), rubric "
                f"{s.judge_prompt_hash}[/dim]"
            )
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


def cmd_list_runs(args: argparse.Namespace, console: Console) -> None:
    runs = list_runs(limit=args.limit_runs, adapter=args.adapter_filter, db_path=args.db)
    if not runs:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return
    render_run_table([m.run_id for m in runs], db_path=args.db, console=console)


def cmd_show(args: argparse.Namespace, console: Console) -> None:
    render_run_table(args.show, db_path=args.db, console=console)
    render_category_table(args.show, db_path=args.db, console=console)


def cmd_compare(args: argparse.Namespace, console: Console) -> None:
    run_a, run_b = args.compare
    render_diff(compare_runs(run_a, run_b, db_path=args.db), console=console)


def cmd_rescore(args: argparse.Namespace, console: Console) -> None:
    """Re-run scorers over stored response text. No API calls, no cost."""
    tasks = load_tasks(args.tasks)
    before = load_run(args.rescore, db_path=args.db).meta
    record = rescore_run(
        args.rescore, tasks, [SchemaScorer(), FieldScorer()], db_path=args.db
    )
    after = record.meta

    console.print(
        f"Rescored [bold]{after.run_id[:8]}[/bold] "
        f"({len(record.results)} results, no API calls)"
    )
    console.print(
        f"  schema {before.schema_pass_rate:.3f} → {after.schema_pass_rate:.3f}   "
        f"F1 {before.mean_f1:.4f} → {after.mean_f1:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run, persist, and compare LLM extraction evals."
    )
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
    parser.add_argument(
        "--judge",
        default=None,
        metavar="ADAPTER",
        help="Adjudicate near-misses with this adapter as judge",
    )
    parser.add_argument(
        "--judge-all",
        action="store_true",
        help="Judge every mismatched field, not only near-misses",
    )
    parser.add_argument("--judge-model", default=None, help="MODELS alias for the judge")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite store path")
    parser.add_argument(
        "--no-save", action="store_true", help="Run without persisting to the store"
    )

    store = parser.add_argument_group("stored runs")
    store.add_argument("--list-runs", action="store_true", help="List recorded runs")
    store.add_argument(
        "--limit-runs", type=int, default=20, help="Rows for --list-runs (default 20)"
    )
    store.add_argument(
        "--adapter-filter", default=None, help="Filter --list-runs by adapter"
    )
    store.add_argument("--show", nargs="+", metavar="RUN_ID", help="Render stored runs")
    store.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Diff two runs"
    )
    store.add_argument(
        "--rescore",
        metavar="RUN_ID",
        help="Re-run scorers over stored text (no API calls)",
    )

    args = parser.parse_args()
    console = Console()

    # Python's default warning format dumps the offending source line, which
    # is noise for a warning the user is meant to act on rather than debug.
    warnings.formatwarning = lambda msg, *a, **kw: f"warning: {msg}\n"

    # Read-only store commands short-circuit before anything can spend money.
    if args.list_runs:
        return cmd_list_runs(args, console)
    if args.show:
        return cmd_show(args, console)
    if args.compare:
        return cmd_compare(args, console)
    if args.rescore:
        return cmd_rescore(args, console)

    if not args.adapter:
        args.adapter = ["anthropic"]

    tasks = load_tasks(args.tasks, categories=args.category, limit=args.limit)
    summaries = asyncio.run(main_async(args, tasks))
    render(summaries, console)

    if not args.no_save:
        console.print()
        for summary in summaries:
            run_id = save_run(summary, tasks, db_path=args.db)
            console.print(f"  saved {summary.adapter_name} run [bold]{run_id[:8]}[/bold]")


if __name__ == "__main__":
    main()
