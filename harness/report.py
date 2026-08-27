"""Terminal reporting over stored runs.

Reads from the store, never from in-memory results. That constraint is the
point: a report you can only produce in the same process that did the run
is not a report, it is a print statement. Everything here works equally on
a run from five minutes ago and one from last week.

Deliberately terminal-only. The HTML report arrives in Week 3B and will
read from this same store; adding plots here would just be work to throw
away.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness.stats import (
    FrontierPoint,
    bootstrap_ci,
    describe_difference,
    mcnemar,
    minimum_detectable_effect,
    paired_bootstrap_diff,
    pareto_frontier,
)
from harness.store import DEFAULT_DB_PATH, load_run
from harness.types import RunDiff, RunRecord

SHORT_ID = 8


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[min(int(pct) - 1, len(quantiles) - 1)]


def per_task_scores(record: RunRecord, scorer: str = "fields") -> dict[str, float]:
    """One value per task — the unit the bootstrap resamples.

    Deliberately not per field: the eight fields of a task come from one
    document and fail together, so treating them as independent would give
    intervals far too narrow.
    """
    out: dict[str, float] = {}
    for result in record.results:
        value = next((s.value for s in result.scores if s.scorer == scorer), None)
        if value is not None:
            out[result.task_id] = value
    return out


def _latencies(record: RunRecord) -> list[float]:
    """Latencies of calls that actually hit the network.

    Cached results replay the original call's latency, so including them
    would report a number that describes neither the cache (which is
    instant) nor the provider (which is not).
    """
    return [r.latency_ms for r in record.results if not r.cached and r.error is None]


def render_run_table(
    run_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    console: Console | None = None,
) -> None:
    """One row per run: accuracy, cost, latency, failures."""
    console = console or Console()
    records = [load_run(run_id, db_path) for run_id in run_ids]

    table = Table(title="Runs")
    table.add_column("run", no_wrap=True)
    table.add_column("adapter", no_wrap=True)
    table.add_column("model", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("schema", justify="right")
    table.add_column("F1 [95% CI]", justify="right", no_wrap=True)
    table.add_column("cost $", justify="right", no_wrap=True)
    table.add_column("$/task", justify="right", no_wrap=True)
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("wall s", justify="right")
    table.add_column("fail", justify="right")

    for record in records:
        meta = record.meta
        latencies = _latencies(record)
        per_task = meta.total_cost_usd / meta.task_count if meta.task_count else 0.0
        table.add_row(
            meta.run_id[:SHORT_ID],
            meta.adapter,
            meta.model_id,
            str(meta.task_count),
            f"{meta.schema_pass_rate:.2f}",
            str(bootstrap_ci(list(per_task_scores(record).values()), seed=0)),
            f"{meta.total_cost_usd:.6f}",
            f"{per_task:.6f}",
            f"{_percentile(latencies, 50):.0f}",
            f"{_percentile(latencies, 95):.0f}",
            f"{meta.wall_seconds:.1f}",
            f"[red]{meta.failures}[/red]" if meta.failures else "0",
        )

    console.print(table)
    _print_provenance(records, console)


def _print_provenance(records: list[RunRecord], console: Console) -> None:
    """Run ids (for --compare) plus the commit each was recorded against."""
    console.print()
    for record in records:
        meta = record.meta
        commit = (meta.git_commit or "unknown")[:8]
        dirty = " [yellow](dirty tree — not reproducible)[/yellow]" if meta.git_dirty else ""
        console.print(
            f"  [bold]{meta.run_id[:SHORT_ID]}[/bold]  {meta.created_at[:19]}  "
            f"{meta.adapter}  commit {commit}  prompt {meta.prompt_name}"
            f"@{meta.prompt_hash}{dirty}"
        )


def render_category_table(
    run_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    console: Console | None = None,
    scorer: str = "fields",
) -> None:
    """Categories down the side, runs across the top."""
    console = console or Console()
    records = [load_run(run_id, db_path) for run_id in run_ids]

    per_run: list[dict[str, list[float]]] = []
    for record in records:
        buckets: dict[str, list[float]] = {}
        for result in record.results:
            value = next((s.value for s in result.scores if s.scorer == scorer), None)
            if value is not None:
                buckets.setdefault(result.category, []).append(value)
        per_run.append(buckets)

    categories = sorted({c for buckets in per_run for c in buckets})
    if not categories:
        return

    table = Table(title=f"Mean {scorer} score by category")
    table.add_column("category", no_wrap=True)
    for record in records:
        table.add_column(
            f"{record.meta.adapter}\n{record.meta.run_id[:SHORT_ID]}", justify="right"
        )

    for category in categories:
        row = [category]
        for buckets in per_run:
            values = buckets.get(category)
            row.append(f"{sum(values) / len(values):.3f}" if values else "—")
        table.add_row(*row)

    # A count row makes a suspiciously clean category legible as "n=2"
    # rather than being read as a real result.
    counts = ["[dim]n[/dim]"]
    for buckets in per_run:
        counts.append(f"[dim]{sum(len(v) for v in buckets.values())}[/dim]")
    table.add_section()
    table.add_row(*counts)

    console.print(table)


def render_diff(diff: RunDiff, console: Console | None = None) -> None:
    """Per-task changes between two runs."""
    console = console or Console()

    console.print(
        f"\n[bold]{diff.run_id_a[:SHORT_ID]} → {diff.run_id_b[:SHORT_ID]}[/bold]  "
        f"(scorer: {diff.scorer})"
    )
    direction = "no change"
    if diff.mean_delta > 0:
        direction = f"[green]+{diff.mean_delta:.4f}[/green]"
    elif diff.mean_delta < 0:
        direction = f"[red]{diff.mean_delta:.4f}[/red]"
    console.print(
        f"  mean {diff.mean_a:.4f} → {diff.mean_b:.4f}   delta {direction}"
    )
    console.print(
        f"  {len(diff.regressed)} regressed, {len(diff.improved)} improved, "
        f"{diff.unchanged} unchanged"
    )

    if diff.only_in_a or diff.only_in_b:
        console.print(
            f"  [yellow]only in A: {len(diff.only_in_a)}, "
            f"only in B: {len(diff.only_in_b)}[/yellow]"
        )

    if not diff.regressed and not diff.improved:
        return

    table = Table(title="Changed tasks")
    table.add_column("task", no_wrap=True)
    table.add_column("category", no_wrap=True)
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("delta", justify="right")

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    for entry in diff.regressed + diff.improved:
        colour = "red" if entry.delta < 0 else "green"
        table.add_row(
            entry.task_id,
            entry.category,
            fmt(entry.value_a),
            fmt(entry.value_b),
            f"[{colour}]{entry.delta:+.3f}[/{colour}]",
        )

    console.print(table)


def render_frontier(
    run_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    console: Console | None = None,
) -> None:
    """Cost per task against quality, with dominated runs flagged.

    This is the view a buyer actually wants. "Which model is best" is rarely
    the question; "which models are worth their price for this workload" is.
    A run that costs more *and* scores worse is never the right choice, and
    naming it is more useful than ranking everything.
    """
    console = console or Console()
    records = [load_run(run_id, db_path) for run_id in run_ids]

    points = []
    for record in records:
        scores = list(per_task_scores(record).values())
        if not scores:
            continue
        meta = record.meta
        points.append(
            FrontierPoint(
                label=f"{meta.adapter} ({meta.run_id[:SHORT_ID]})",
                cost_per_task=(
                    meta.total_cost_usd / meta.task_count if meta.task_count else 0.0
                ),
                quality=meta.mean_f1,
                quality_interval=bootstrap_ci(scores, seed=0),
            )
        )

    if not points:
        console.print("[yellow]No scored runs to compare.[/yellow]")
        return

    table = Table(title="Cost / quality frontier")
    table.add_column("run", no_wrap=True)
    table.add_column("$/task", justify="right", no_wrap=True)
    table.add_column("mean F1 [95% CI]", justify="right", no_wrap=True)
    table.add_column("frontier", no_wrap=True)

    for point in sorted(pareto_frontier(points), key=lambda p: p.cost_per_task):
        if point.pareto_optimal:
            status = "[green]optimal[/green]"
        else:
            status = f"[dim]dominated by {', '.join(point.dominated_by)}[/dim]"
        table.add_row(
            point.label,
            f"{point.cost_per_task:.6f}",
            str(point.quality_interval),
            status,
        )

    console.print(table)
    console.print(
        "[dim]A run is dominated when another is no more expensive and no "
        "worse. Overlapping intervals mean the quality ordering is not "
        "established.[/dim]"
    )


def render_paired_comparison(
    run_id_a: str,
    run_id_b: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    console: Console | None = None,
) -> None:
    """Paired difference with an interval and a verdict spelled out in words."""
    console = console or Console()
    record_a = load_run(run_id_a, db_path)
    record_b = load_run(run_id_b, db_path)

    scores_a = per_task_scores(record_a)
    scores_b = per_task_scores(record_b)
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        console.print("[yellow]No tasks in common between these runs.[/yellow]")
        return

    a_values = [scores_a[t] for t in shared]
    b_values = [scores_b[t] for t in shared]
    label_a = f"{record_a.meta.adapter} ({record_a.meta.run_id[:SHORT_ID]})"
    label_b = f"{record_b.meta.adapter} ({record_b.meta.run_id[:SHORT_ID]})"

    diff = paired_bootstrap_diff(a_values, b_values, seed=0)
    perfect = mcnemar(
        [v == 1.0 for v in a_values], [v == 1.0 for v in b_values]
    )

    table = Table(title="Paired comparison", show_header=False)
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_row("tasks compared", str(len(shared)))
    table.add_row(label_a, f"{statistics.fmean(a_values):.3f}")
    table.add_row(label_b, f"{statistics.fmean(b_values):.3f}")
    table.add_row("difference", str(diff))
    table.add_row(
        "McNemar (task fully correct)",
        f"b={perfect.b} c={perfect.c}  exact p={perfect.p_exact:.3f}",
    )
    console.print(table)

    # Said in words, because a reader who skims the number and misses the
    # interval is exactly who this is for.
    console.print(f"\n  [bold]{describe_difference(diff, label_a, label_b)}[/bold]")
    console.print(f"  [dim]on fully-correct tasks: {perfect.verdict}[/dim]")


def render_mde(
    run_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    console: Console | None = None,
) -> None:
    """How small a difference this suite could resolve, at its current size."""
    console = console or Console()
    records = [load_run(run_id, db_path) for run_id in run_ids]
    if not records:
        console.print("[yellow]No runs to size.[/yellow]")
        return

    n = max(r.meta.task_count for r in records)
    baseline = statistics.fmean([r.meta.mean_f1 for r in records])

    table = Table(title=f"Minimum detectable effect (n={n}, power 0.80)")
    table.add_column("baseline", justify="right")
    table.add_column("smallest resolvable difference", justify="right")
    for candidate in sorted({round(baseline, 2), 0.50, 0.80, 0.90, 0.95}):
        marker = "  <- this suite" if abs(candidate - round(baseline, 2)) < 1e-9 else ""
        table.add_row(f"{candidate:.2f}", f"{minimum_detectable_effect(n, candidate):.3f}{marker}")
    console.print(table)
    console.print(
        "[dim]The effect is hardest to resolve near a baseline of 0.50, where "
        "binomial variance peaks, and easiest at the extremes.[/dim]"
    )
