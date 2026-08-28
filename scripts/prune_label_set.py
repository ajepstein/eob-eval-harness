"""Remove items whose verdict SCHEMA.md already settles.

A label set should contain only questions a person can answer better than
the schema can. Items settled by a documented convention — a CPT modifier,
a member-id dependent code — are removed here so that:

  * nobody spends time relabelling a lookup,
  * the calibration is computed over judgment rather than over rubric
    compliance, which would inflate kappa without evidence, and
  * verdicts already recorded against an undefined or misread rule stop
    contaminating the population.

Removing an item also removes any label on it, which is the point: those
labels are what the convention supersedes. Everything is reported before
anything is deleted, and the default is a dry run.

Usage:
    python scripts/prune_label_set.py                    # dry run, shows counts
    python scripts/prune_label_set.py --apply            # asks, then deletes
    python scripts/prune_label_set.py --apply --yes      # no prompt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from harness.conventions import settled_verdict
from harness.labeling import list_label_sets, load_label_set
from harness.store import DEFAULT_DB_PATH, _session


def settled_items(label_set) -> list[tuple[object, str]]:
    """(item, verdict) for every item a convention already decides."""
    out = []
    for item in label_set.items:
        verdict = settled_verdict(item.field, item.expected, item.predicted)
        if verdict is not None:
            out.append((item, verdict))
    return out


def _labelled_ids(conn, label_set_id: str) -> set[int]:
    return {
        r["item_id"]
        for r in conn.execute(
            "SELECT item_id FROM human_labels WHERE label_set_id = ?",
            (label_set_id,),
        )
    }


def prune(label_set_id: str, item_ids: list[int], db_path) -> tuple[int, int]:
    """Delete the given items and their labels. Returns (items, labels)."""
    if not item_ids:
        return (0, 0)
    placeholders = ",".join("?" * len(item_ids))
    with _session(db_path) as conn:
        labels = conn.execute(
            f"SELECT COUNT(*) FROM human_labels WHERE item_id IN ({placeholders})",
            item_ids,
        ).fetchone()[0]
        conn.execute(
            f"DELETE FROM human_labels WHERE item_id IN ({placeholders})", item_ids
        )
        conn.execute(
            f"DELETE FROM label_items WHERE id IN ({placeholders})", item_ids
        )
    return (len(item_ids), labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", default=None)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete. Without this, nothing is written.")
    parser.add_argument("--yes", action="store_true", help="Skip the prompt")
    args = parser.parse_args()

    console = Console()
    sets = list_label_sets(args.db)
    if not sets:
        console.print("[red]No label sets.[/red]")
        return 1

    label_set = load_label_set(args.label_set or sets[0]["id"], args.db)
    settled = settled_items(label_set)
    with _session(args.db) as conn:
        labelled = _labelled_ids(conn, label_set.id)

    if not settled:
        console.print("[green]Nothing is settled by a convention — "
                      "every item still needs a person.[/green]")
        return 0

    table = Table(title=f"Settled by SCHEMA.md — label set {label_set.id[:8]}")
    table.add_column("field", no_wrap=True)
    table.add_column("verdict", no_wrap=True)
    table.add_column("items", justify="right")
    table.add_column("of those, labelled", justify="right")

    counts: Counter = Counter()
    labelled_counts: Counter = Counter()
    for item, verdict in settled:
        counts[(item.field, verdict)] += 1
        if item.item_id in labelled:
            labelled_counts[(item.field, verdict)] += 1
    for (field, verdict), n in sorted(counts.items()):
        table.add_row(field, verdict, str(n), str(labelled_counts[(field, verdict)]))
    console.print(table)

    remaining = len(label_set.items) - len(settled)
    remaining_labelled = len(labelled) - sum(labelled_counts.values())
    console.print(
        f"\n  removing {len(settled)} items "
        f"({sum(labelled_counts.values())} of them already labelled)\n"
        f"  leaving  {remaining} judgment items, "
        f"{remaining_labelled} labelled, {remaining - remaining_labelled} to go"
    )
    if remaining < 30:
        console.print(
            f"  [yellow]warning: {remaining} items is below the "
            f"calibration floor of 30[/yellow]"
        )

    if not args.apply:
        console.print("\n[dim]dry run — rerun with --apply to delete[/dim]")
        return 0

    if not args.yes:
        console.print("\n[bold]Delete these items and their labels?[/bold] [y/N]")
        try:
            if input("> ").strip().lower() != "y":
                console.print("Nothing removed.")
                return 0
        except (EOFError, KeyboardInterrupt):
            console.print("\nNothing removed.")
            return 0

    items, labels = prune(label_set.id, [i.item_id for i, _ in settled], args.db)
    console.print(f"[green]Removed {items} items and {labels} labels.[/green]")
    console.print("[dim]Re-export with scripts/export_labels.py[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
