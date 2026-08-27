"""Blind terminal labelling loop.

You see the document, the field, and two values. You do not see which value
the judge preferred, which model produced the prediction, or which adapter
ran — those are absent from the label_items table entirely, so this script
could not show them if it tried.

Verdicts:
  e  equivalent   the two values name the same thing
  d  different    they do not
  u  unsure       genuinely cannot decide
  b  bad task     the reference value itself is wrong
  s  skip         come back to it
  q  save and quit

`b` matters: labelling reliably surfaces answer-key errors that
verify_tasks.py missed. Those go back into the YAML, and any run scored
before the fix should be --rescore'd.

Progress is shown; running agreement is not. Seeing how you are doing
against the judge mid-session would bias every remaining label, and there
is no way to un-bias them afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness.labeling import (
    build_label_set,
    labelled_item_ids,
    list_label_sets,
    load_label_set,
    save_label,
)
from harness.store import DEFAULT_DB_PATH, _session, list_runs

VERDICTS = {"e": "equivalent", "d": "different", "u": "unsure", "b": "bad_task"}


def _document(run_id: str, task_id: str, db_path: str) -> str:
    """The source document, pulled from the task file via the stored run."""
    import yaml

    for path in sorted(Path("tasks").glob("**/*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if raw.get("id") == task_id:
            return raw.get("input", "")
    return "(document not found)"


def render(item, index: int, total: int, document: str, console: Console) -> None:
    console.clear()
    console.print(
        Panel(document, title=f"Source document — {item.task_id}", height=28)
    )
    table = Table(show_header=False, box=None)
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_row("field", item.field)
    table.add_row("value A", item.expected or "(absent)")
    table.add_row("value B", item.predicted or "(absent)")
    console.print(table)
    console.print(
        f"\n[bold]{index}/{total}[/bold]   "
        "[green]e[/green]=equivalent  [red]d[/red]=different  "
        "[yellow]u[/yellow]=unsure  [magenta]b[/magenta]=bad task  "
        "s=skip  q=save and quit"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", action="store_true", help="Build a new label set")
    parser.add_argument("--n", type=int, default=200, help="Target sample size")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--double-label-frac", type=float, default=0.15)
    parser.add_argument("--label-set", default=None, help="Resume this label set")
    parser.add_argument("--labeler", default="self", help="Who is labelling")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--list", action="store_true", help="List label sets")
    args = parser.parse_args()

    console = Console()

    if args.list:
        sets = list_label_sets(args.db)
        if not sets:
            console.print("[yellow]No label sets yet. Use --new.[/yellow]")
            return 0
        for s in sets:
            done = len(labelled_item_ids(s["id"], args.db))
            console.print(
                f"  {s['id'][:8]}  seed={s['seed']}  n={s['requested_n']}  "
                f"labelled={done}  {s['created_at'][:19]}"
            )
        return 0

    if args.new:
        run_ids = [m.run_id for m in list_runs(limit=50, db_path=args.db)]
        if not run_ids:
            console.print("[red]No runs in the store. Run an eval with --judge first.[/red]")
            return 1
        label_set = build_label_set(
            run_ids, n=args.n, seed=args.seed,
            double_label_frac=args.double_label_frac, db_path=args.db,
        )
        console.print(
            f"Built label set [bold]{label_set.id[:8]}[/bold] with "
            f"{len(label_set.items)} items (seed {label_set.seed})."
        )
    elif args.label_set:
        label_set = load_label_set(args.label_set, args.db)
    else:
        sets = list_label_sets(args.db)
        if not sets:
            console.print("[yellow]No label set yet. Use --new.[/yellow]")
            return 1
        label_set = load_label_set(sets[0]["id"], args.db)

    done = labelled_item_ids(label_set.id, args.db)
    remaining = [i for i in label_set.items if i.item_id not in done]
    if not remaining:
        console.print("[green]This label set is fully labelled.[/green]")
        return 0

    console.print(
        f"{len(done)} already labelled, {len(remaining)} to go. Press enter to start."
    )
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return 0

    labelled = 0
    for offset, item in enumerate(remaining):
        document = _document(item.run_id, item.task_id, args.db)
        render(item, len(done) + offset + 1, len(label_set.items), document, console)

        started = time.monotonic()
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nStopping.")
            break
        elapsed = time.monotonic() - started

        if answer == "q":
            break
        if answer == "s":
            continue
        if answer not in VERDICTS:
            continue

        # Saved immediately: you will not finish in one sitting, and a lost
        # session would mean re-labelling items you have already seen.
        save_label(
            label_set.id, item, VERDICTS[answer], elapsed,
            labeler=args.labeler, db_path=args.db,
        )
        labelled += 1

    console.print(f"\nLabelled {labelled} this session. Resume with:")
    console.print(f"  python scripts/label.py --label-set {label_set.id[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
