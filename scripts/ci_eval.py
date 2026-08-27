"""Run the eval in CI and enforce the quality gates.

Two modes, and the default matters. A CI job that spends money on every push
and fails whenever a provider rate-limits is a job that gets disabled inside
a week, after which it protects nothing.

**Cached (default).** Replays committed responses from fixtures/cache. Fast,
free, deterministic, needs no secrets. It catches the failure that actually
matters day to day: a prompt or scorer change that breaks things. The
adapter is one that raises if called, so a cache miss is loud rather than a
silent bill.

**Live (--live).** Real calls against the providers. Catches drift that the
cache cannot — a silently updated model, a changed default. Prints its
estimated cost and requires the flag.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from harness.adapters import get_adapter
from harness.adapters.base import FatalError
from harness.cache import ResponseCache
from harness.gates import GateConfigError, GateReport, evaluate_gates
from harness.runner import run_tasks
from harness.scorers.fields import FieldScorer
from harness.scorers.schema import SchemaScorer
from harness.store import DEFAULT_DB_PATH, save_run
from harness.tasks import load_tasks
from harness.types import ModelResponse

FIXTURE_CACHE = "fixtures/cache"
EST_COST_PER_CALL = 0.004


class CacheMiss(FatalError):
    """A task had no committed response and live calls are disabled."""


class ReplayOnlyAdapter:
    """Serves nothing. Every response must come from the committed cache.

    Making the adapter raise is what turns "CI should not call the network"
    from an intention into a guarantee — a cache miss cannot quietly become
    a paid request.
    """

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.attempts: list[str] = []

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse:
        self.attempts.append(prompt[:80])
        raise CacheMiss(
            f"no committed response for this {self.name} request. The prompt, "
            f"model, or task set has changed since fixtures/cache was built. "
            f"Regenerate it locally and commit, or run with --live."
        )


def render_gates(report: GateReport, console: Console) -> None:
    table = Table(title=f"Quality gates — run {report.run_id[:8]}")
    table.add_column("metric", no_wrap=True)
    table.add_column("rule", no_wrap=True)
    table.add_column("observed", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("result", no_wrap=True)

    for result in report.results:
        if result.skipped_reason:
            status = "[yellow]skipped[/yellow]"
        elif result.passed:
            status = "[green]pass[/green]"
        else:
            status = "[red]FAIL[/red]"
        table.add_row(
            result.metric,
            result.bound,
            "—" if result.observed is None else f"{result.observed:.4f}",
            "—" if result.threshold is None else f"{result.threshold:.4f}",
            status,
        )
    console.print(table)

    for result in report.results:
        if result.skipped_reason:
            console.print(f"  [yellow]{result.metric}: {result.skipped_reason}[/yellow]")


def explain(report: GateReport, console: Console, limit: int = 10) -> None:
    """Name the tasks driving each failure.

    "mean_f1 dropped 4 points" is not actionable. "mean_f1 dropped 4 points,
    driven by 6 missing_field tasks now hallucinating an NPI" is a bug report.
    """
    failing = [r for r in report.results if not r.passed and not r.skipped_reason]
    if not failing:
        return
    for result in failing:
        console.print(f"\n[bold red]{result.metric} ({result.bound}) failed[/bold red]")
        if result.difference is not None:
            console.print(
                f"  paired difference {result.difference.point:+.4f} "
                f"[{result.difference.low:+.4f}, {result.difference.high:+.4f}]"
                f"  ({'distinguishable from zero' if result.significant else 'within noise'})"
            )
        if not result.driving_tasks:
            console.print(
                f"  observed {result.observed:.4f} against threshold "
                f"{result.threshold:.4f} — an absolute threshold, so no "
                f"per-task attribution applies."
            )
            continue
        table = Table(title=f"tasks driving the {result.metric} regression")
        table.add_column("task", no_wrap=True)
        table.add_column("baseline", justify="right")
        table.add_column("current", justify="right")
        table.add_column("delta", justify="right")
        for row in result.driving_tasks[:limit]:
            table.add_row(
                row["task_id"], f"{row['baseline']:.3f}",
                f"{row['current']:.3f}", f"{row['delta']:+.3f}",
            )
        console.print(table)
        if len(result.driving_tasks) > limit:
            console.print(f"  [dim]... and {len(result.driving_tasks) - limit} more[/dim]")


async def _run(args, tasks, console) -> list:
    cache = ResponseCache(cache_dir=args.cache, enabled=True)
    summaries = []
    for name in args.adapter:
        if args.live:
            adapter = get_adapter(name)
        else:
            # Model id must match what the cache was built with, so resolve
            # it from the registry without constructing a live client.
            from harness.adapters import _DEFAULT_MODEL_ALIAS
            from harness.config import MODELS

            adapter = ReplayOnlyAdapter(name, MODELS[_DEFAULT_MODEL_ALIAS[name]])
        summaries.append(
            await run_tasks(
                tasks, adapter, concurrency=args.concurrency, cache=cache,
                scorers=[SchemaScorer(), FieldScorer()],
            )
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", default=None)
    parser.add_argument("--tasks", default="tasks/")
    parser.add_argument("--config", default="eval_gates.yaml")
    parser.add_argument("--cache", default=FIXTURE_CACHE)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--explain", action="store_true",
                        help="Name the tasks driving each failing gate")
    parser.add_argument("--live", action="store_true",
                        help="Make real API calls instead of replaying the cache")
    parser.add_argument("--markdown", default=None,
                        help="Also write the gate table as markdown (for PR comments)")
    args = parser.parse_args()

    console = Console()
    if not args.adapter:
        args.adapter = ["anthropic", "openai"]

    tasks = load_tasks(args.tasks)

    if args.live:
        calls = len(tasks) * len(args.adapter)
        console.print(
            f"[yellow]--live: about {calls} API calls "
            f"(~${calls * EST_COST_PER_CALL:.2f})[/yellow]"
        )
    else:
        console.print(
            f"[dim]replaying {args.cache} — no network, no secrets[/dim]"
        )

    summaries = asyncio.run(_run(args, tasks, console))

    misses = [
        r for s in summaries for r in s.results
        if r.error and "no committed response" in r.error
    ]
    if misses:
        console.print(
            f"\n[red]{len(misses)} task(s) had no committed response.[/red] "
            "The prompt, model, or task set has changed since fixtures/cache "
            "was built. Regenerate it and commit, or run with --live."
        )
        return 1

    try:
        reports = []
        for summary in summaries:
            run_id = save_run(summary, tasks, db_path=args.db)
            reports.append(evaluate_gates(run_id, args.config, db_path=args.db))
    except GateConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    failed = False
    for report in reports:
        render_gates(report, console)
        if args.explain:
            explain(report, console)
        failed = failed or not report.passed

    if args.markdown:
        Path(args.markdown).write_text(_markdown(reports))

    console.print(
        f"\n[bold]{'GATES FAILED' if failed else 'all gates passed'}[/bold]"
    )
    return 1 if failed else 0


def _markdown(reports: list[GateReport]) -> str:
    lines = ["## Eval gates", ""]
    for report in reports:
        lines.append(f"### run `{report.run_id[:8]}`")
        lines.append("")
        lines.append("| metric | rule | observed | threshold | result |")
        lines.append("|---|---|---:|---:|---|")
        for r in report.results:
            status = "skipped" if r.skipped_reason else ("pass" if r.passed else "**FAIL**")
            observed = "—" if r.observed is None else f"{r.observed:.4f}"
            threshold = "—" if r.threshold is None else f"{r.threshold:.4f}"
            lines.append(f"| {r.metric} | {r.bound} | {observed} | {threshold} | {status} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
