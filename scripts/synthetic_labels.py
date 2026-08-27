"""Generate clearly-marked synthetic labels to exercise the pipeline.

These are NOT ground truth and cannot become a calibration. They are written
under labeler="synthetic", and `calibration_is_usable` refuses any
calibration computed from them — a kappa derived from these measures one
model agreeing with another, which is the exact assertion the calibration
layer exists to avoid making.

Their only purpose is to prove the plumbing works end to end: sampling,
labelling, agreement, bias tests, the report banner, and the gate.

The verdicts follow a stated rule set rather than a coin flip, so the
resulting verdict distribution and disagreement pattern resemble what a real
labelling session would produce:

  * a name written in a different order, or shortened, names the same person
  * a procedure code with a modifier appended is the same procedure
  * an identifier differing only in spacing or punctuation is the same one
  * an identifier carrying an extra component the document labels as a
    *different* field (certificate vs contract) is a different identifier
  * a payer's legal entity and its trading name are the same payer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from harness.labeling import labelled_item_ids, load_label_set, save_label
from harness.store import DEFAULT_DB_PATH

LABELER = "synthetic"


def _squash(text: str) -> str:
    return re.sub(r"[\s,.\-/]", "", (text or "")).casefold()


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[\s,./-]+", (text or "").casefold()) if t}


def decide(field: str, expected: str, predicted: str) -> str:
    """Apply the rule set above to one pair. Returns a verdict."""
    a, b = expected or "", predicted or ""
    sa, sb = _squash(a), _squash(b)

    if sa == sb:
        # Differs only in spacing or punctuation.
        return "equivalent"

    if field == "cpt_codes":
        # Modifiers appended to a code leave the procedure identity intact.
        base = lambda s: {c.split("-")[0] for c in re.findall(r"[0-9A-Za-z]+(?:-[A-Z0-9]+)?", s)}
        return "equivalent" if base(a) == base(b) else "different"

    if field == "patient_name":
        ta, tb = _tokens(a), _tokens(b)
        # Reordering or dropping a middle name keeps the same person; a
        # different surname does not.
        if ta == tb or ta <= tb or tb <= ta:
            return "equivalent"
        return "different"

    if field == "payer_name":
        ta, tb = _tokens(a), _tokens(b)
        # A trading name is a prefix of the legal entity, or vice versa.
        return "equivalent" if ta <= tb or tb <= ta else "different"

    if field == "member_id":
        # An extra trailing component is a *different* identifier: the
        # documents label these as certificate or individual references
        # distinct from the contract number.
        if sa.startswith(sb) or sb.startswith(sa):
            return "different"
        return "different"

    return "different"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", required=True)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    console = Console()
    console.print(
        "[yellow]Generating SYNTHETIC labels. These are not ground truth and "
        "cannot establish judge reliability.[/yellow]"
    )

    label_set = load_label_set(args.label_set, args.db)
    done = labelled_item_ids(label_set.id, args.db)
    todo = [i for i in label_set.items if i.item_id not in done]
    if args.limit:
        todo = todo[: args.limit]

    counts = {"equivalent": 0, "different": 0}
    for item in todo:
        verdict = decide(item.field, item.expected, item.predicted)
        counts[verdict] += 1
        save_label(
            label_set.id, item, verdict, seconds=0.0,
            labeler=LABELER, db_path=args.db,
        )

    console.print(
        f"Wrote {len(todo)} synthetic labels "
        f"({counts['equivalent']} equivalent, {counts['different']} different). "
        f"{len(done)} pre-existing human label(s) left untouched."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
