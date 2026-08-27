"""Labelling progress. Deliberately reports no accuracy information.

Seeing how you are tracking against the judge mid-session would bias every
remaining label, and there is no way to un-bias them afterwards. This shows
how far through you are and how your own verdicts are distributed — nothing
about whether they agree with anything.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from harness.labeling import list_label_sets, load_label_set, load_labels
from harness.store import DEFAULT_DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", default=None)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    console = Console()
    sets = list_label_sets(args.db)
    if not sets:
        console.print("[yellow]No label sets yet.[/yellow]")
        return 0

    set_id = args.label_set or sets[0]["id"]
    label_set = load_label_set(set_id, args.db)
    labels = load_labels(label_set.id, args.db)

    total = len(label_set.items)
    done = len(labels)
    console.print(
        f"\nLabel set [bold]{label_set.id[:8]}[/bold]  seed {label_set.seed}"
    )
    console.print(f"  {done} / {total} labelled  ({done / total * 100:.0f}%)\n")

    counts = Counter(row["verdict"] for row in labels)
    table = Table(title="Your verdicts")
    table.add_column("verdict", no_wrap=True)
    table.add_column("n", justify="right")
    for verdict in ("equivalent", "different", "unsure", "bad_task"):
        table.add_row(verdict, str(counts.get(verdict, 0)))
    console.print(table)

    if labels:
        seconds = [row["seconds"] for row in labels]
        console.print(f"  median {statistics.median(seconds):.1f}s per label")
        # Both tails are worth revisiting: sub-3s labels may be reflexive,
        # and very slow ones mark genuinely hard cases.
        console.print(
            f"  {sum(1 for s in seconds if s < 3)} under 3s, "
            f"{sum(1 for s in seconds if s > 90)} over 90s"
        )
    if counts.get("unsure"):
        frac = counts["unsure"] / done
        console.print(f"\n  unsure rate {frac:.0%}")
        if frac > 0.10:
            console.print(
                "  [yellow]Above 10% — the rubric is likely underspecified. "
                "Consider revising it to judge_v2 before calibrating.[/yellow]"
            )
    if counts.get("bad_task"):
        console.print(
            f"  [magenta]{counts['bad_task']} answer-key problems flagged — "
            f"fix the YAML and --rescore affected runs.[/magenta]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
