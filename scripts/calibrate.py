"""Report whether the judge can be trusted, and publish the answer either way.

    python scripts/calibrate.py --label-set <id> [--bias-tests] [--dry-run]

Prints agreement with its interval, the human ceiling it should be read
against, a per-category breakdown, the confusion matrix, bias-test results,
and the highest-disagreement items with full context for hand review.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness.adapters import get_adapter
from harness.bias_tests import human_drift, length_bias, position_bias
from harness.calibration import agreement, disagreements
from harness.labeling import list_label_sets, load_label_set, load_labels
from harness.prompts import judge_prompt_hash
from harness.store import DEFAULT_DB_PATH, _session, save_calibration

# Rough per-call cost used only for the --dry-run estimate.
EST_COST_PER_CALL = 0.004


def _judge_context(label_set, db_path):
    """Judge verdicts, categories, and documents for the labelled items."""
    verdicts, categories, model_ids = {}, {}, set()
    with _session(db_path) as conn:
        for item in label_set.items:
            row = conn.execute(
                """SELECT verdict, judge_model_id, judge_prompt_hash
                   FROM judge_calls WHERE run_id=? AND task_id=? AND field=?""",
                (item.run_id, item.task_id, item.field),
            ).fetchone()
            if row:
                verdicts[(item.run_id, item.task_id, item.field)] = row["verdict"]
                model_ids.add(row["judge_model_id"])
            cat = conn.execute(
                "SELECT category FROM results WHERE run_id=? AND task_id=?",
                (item.run_id, item.task_id),
            ).fetchone()
            if cat:
                categories[(item.run_id, item.task_id)] = cat["category"]
    return verdicts, categories, model_ids


def _documents() -> dict[str, str]:
    docs = {}
    for path in sorted(Path("tasks").glob("**/*.yaml")):
        raw = yaml.safe_load(path.read_text())
        docs[raw.get("id")] = raw.get("input", "")
    return docs


def _double_labelled(label_set, labels) -> list[tuple[str, str]]:
    by_item = {row["item_id"]: row for row in labels}
    by_key: dict[str, list[str]] = {}
    for item in label_set.items:
        row = by_item.get(item.item_id)
        if row:
            by_key.setdefault(item.item_key, []).append(row["verdict"])
    return [(v[0], v[1]) for v in by_key.values() if len(v) == 2]


def _fmt(value: float | None, places: int = 3) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.{places}f}"


def render(report, console: Console) -> None:
    console.print()
    if report.n == 0:
        console.print(
            Panel(
                "[yellow]No labelled items to calibrate against yet.[/yellow]\n\n"
                "Run [bold]python scripts/label.py[/bold] first. Until labels "
                "exist there is nothing to measure the judge against, and any "
                "number printed here would be meaningless.",
                title="Calibration",
            )
        )
        return

    head = Table(title="Judge agreement with human labels", show_header=False)
    head.add_column(style="bold", no_wrap=True)
    head.add_column()
    head.add_row("items compared", str(report.n))
    head.add_row("raw agreement", _fmt(report.raw_agreement))
    head.add_row(
        "Cohen's kappa",
        f"{_fmt(report.kappa)}  [{_fmt(report.kappa_ci[0])}, "
        f"{_fmt(report.kappa_ci[1])}]  [bold]{report.band}[/bold]",
    )
    ceiling = _fmt(report.human_ceiling_kappa)
    head.add_row(
        "human ceiling",
        f"{ceiling}  (self-agreement over {report.human_ceiling_n} repeated items)",
    )
    console.print(head)

    if report.undefined_reason:
        console.print(f"[yellow]kappa undefined: {report.undefined_reason}[/yellow]")
    if report.skew_warning:
        console.print(f"[yellow]{report.skew_warning}[/yellow]")
    for note in report.notes:
        console.print(f"[dim]note: {note}[/dim]")

    if report.excluded:
        console.print(
            "\n[dim]excluded from the primary metric: "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.excluded.items()))
            + "[/dim]"
        )

    matrix = Table(title="Confusion matrix (human x judge)")
    matrix.add_column("human \\ judge", no_wrap=True)
    matrix.add_column("equivalent", justify="right")
    matrix.add_column("different", justify="right")
    for human in ("equivalent", "different"):
        matrix.add_row(
            human,
            str(report.confusion.get((human, "equivalent"), 0)),
            str(report.confusion.get((human, "different"), 0)),
        )
    console.print(matrix)

    if report.per_category:
        cats = Table(title="Per-category kappa (exploratory)")
        cats.add_column("category", no_wrap=True)
        cats.add_column("n", justify="right")
        cats.add_column("raw", justify="right")
        cats.add_column("kappa", justify="right")
        for name, result in report.per_category.items():
            cats.add_row(
                name, str(result.n), _fmt(result.raw_agreement), _fmt(result.kappa)
            )
        console.print(cats)
        console.print(
            "[dim]Judges fail unevenly. A single aggregate hides which "
            "category is carrying the error.[/dim]"
        )


def render_bias(results: list, console: Console) -> None:
    if not results:
        return
    table = Table(title="Bias tests")
    table.add_column("test", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("statistic", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("result", no_wrap=True)
    for r in results:
        table.add_row(
            r.name, str(r.n), _fmt(r.statistic, 2), _fmt(r.threshold, 2),
            "[red]FIRED[/red]" if r.fired else "[green]ok[/green]",
        )
    console.print(table)
    for r in results:
        if r.interpretation:
            console.print(f"  [dim]{r.name}: {r.interpretation}[/dim]")


def render_disagreements(rows: list[dict], docs: dict, console: Console) -> None:
    if not rows:
        console.print("\n[green]No human/judge disagreements to review.[/green]")
        return
    console.print(
        f"\n[bold]{len(rows)} highest-disagreement items[/bold] — read these. "
        "Either the judge is wrong, you were wrong, or the rubric is "
        "ambiguous; the third is the most useful outcome because it is fixable."
    )
    for row in rows:
        console.print(
            f"\n  [bold]{row['task_id']} / {row['field']}[/bold]  "
            f"({row.get('seconds', 0):.0f}s to label)"
        )
        console.print(f"    you said   : {row['verdict']}")
        console.print(f"    judge said : {row['judge_verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", default=None)
    parser.add_argument("--bias-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate bias-test API cost without running")
    parser.add_argument("--judge", default="anthropic", help="Adapter for bias tests")
    parser.add_argument("--sample", type=int, default=25,
                        help="Items to use for the position-bias test")
    parser.add_argument("--save", action="store_true", help="Persist the calibration")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    console = Console()
    sets = list_label_sets(args.db)
    if not sets:
        console.print("[red]No label sets. Build one with scripts/label.py --new.[/red]")
        return 1

    label_set = load_label_set(args.label_set or sets[0]["id"], args.db)
    labels = load_labels(label_set.id, args.db)
    verdicts, categories, model_ids = _judge_context(label_set, args.db)

    report = agreement(
        labels, verdicts, categories=categories,
        double_labelled=_double_labelled(label_set, labels),
    )

    console.print(
        f"[bold]Label set[/bold] {label_set.id[:8]}   "
        f"{len(labels)}/{len(label_set.items)} labelled   "
        f"rubric {judge_prompt_hash()}"
    )
    render(report, console)

    bias_results = []
    if args.bias_tests or args.dry_run:
        rows = [
            {**row, "predicted": next(
                (i.predicted for i in label_set.items
                 if (i.run_id, i.task_id, i.field)
                 == (row["run_id"], row["task_id"], row["field"])), ""),
             "expected": next(
                (i.expected for i in label_set.items
                 if (i.run_id, i.task_id, i.field)
                 == (row["run_id"], row["task_id"], row["field"])), "")}
            for row in labels
        ]
        sample = rows[: args.sample]
        calls = len(sample) * 2
        console.print(
            f"\n[bold]Bias tests[/bold]: position bias needs {calls} API calls "
            f"(~${calls * EST_COST_PER_CALL:.2f}). "
            "Length and drift tests are offline."
        )
        if args.dry_run:
            console.print("[yellow]--dry-run: not calling the API.[/yellow]")
            return 0

        bias_results = [length_bias(rows), human_drift(labels)]
        if sample:
            bias_results.insert(
                0,
                asyncio.run(
                    position_bias(get_adapter(args.judge), sample, _documents())
                ),
            )
        render_bias(bias_results, console)

    render_disagreements(disagreements(labels, verdicts), _documents(), console)

    if args.save and report.n:
        calibration_id = save_calibration(
            report, label_set.id,
            judge_model_id=next(iter(model_ids), "unknown"),
            judge_prompt_hash=judge_prompt_hash(),
            bias_results={r.name: r.statistic for r in bias_results},
            labelers=[row["labeler"] for row in labels],
            db_path=args.db,
        )
        console.print(f"\nSaved calibration [bold]{calibration_id[:8]}[/bold]")
        from harness.calibration import calibration_is_usable
        from harness.store import find_calibration

        usable, why_not = calibration_is_usable(
            find_calibration(judge_prompt_hash(), db_path=args.db)
        )
        if not usable:
            console.print(
                f"  [yellow]This calibration will not be treated as "
                f"establishing reliability: {why_not}.[/yellow]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
