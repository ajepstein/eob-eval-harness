"""Export and restore the hand-labelled data as reviewable JSON.

Human labels are the only irreplaceable thing in the store. Runs replay from
the cache, scores recompute, calibrations derive from labels plus judge
calls — but a verdict typed by a person does not come back. ``*.db`` is
gitignored, correctly, because everything else in it is derived; the effect
is that the labels live on exactly one disk unless they are exported.

The output is deterministic — sorted keys, items ordered by position — so
the file diffs cleanly and a reviewer can see which verdicts changed between
two commits rather than a blob that differs on every write.

Labels are keyed by ``position`` rather than by row id. Row ids are
AUTOINCREMENT and mean nothing outside the database that issued them;
position is stable and is what ``UNIQUE(label_set_id, position)`` already
guarantees. Restore therefore works into an empty database without
inheriting id collisions from wherever the file came from.

Carries no judge data, deliberately. Blinding is enforced by the schema —
``label_items`` has no verdict column — and a backup that embedded one would
reintroduce through the back door exactly what that shape prevents.

Usage:
    python scripts/export_labels.py                       # -> labels/labels.json
    python scripts/export_labels.py --out FILE
    python scripts/export_labels.py --restore FILE --db fresh.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.store import DEFAULT_DB_PATH, _session, init_db

DEFAULT_OUT = Path("labels/labels.json")
FORMAT_VERSION = 1


def build_export(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Read every label set, its items, and its human labels."""
    sets = []
    with _session(db_path) as conn:
        for s in conn.execute("SELECT * FROM label_sets ORDER BY created_at"):
            items = [
                {
                    "position": r["position"],
                    "item_key": r["item_key"],
                    "run_id": r["run_id"],
                    "task_id": r["task_id"],
                    "field": r["field"],
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                    "pass_number": r["pass_number"],
                }
                for r in conn.execute(
                    "SELECT * FROM label_items WHERE label_set_id=? ORDER BY position",
                    (s["id"],),
                )
            ]
            # Joined through label_items so a label is addressed by position.
            # The run_id/task_id/field on human_labels are denormalised copies
            # of the item's and are reconstructed on restore rather than stored
            # twice here.
            labels = [
                {
                    "position": r["position"],
                    "verdict": r["verdict"],
                    "seconds": r["seconds"],
                    "labeled_at": r["labeled_at"],
                    "labeler": r["labeler"],
                }
                for r in conn.execute(
                    """SELECT li.position, h.verdict, h.seconds, h.labeled_at,
                              h.labeler
                         FROM human_labels h
                         JOIN label_items li ON li.id = h.item_id
                        WHERE h.label_set_id = ?
                        ORDER BY li.position""",
                    (s["id"],),
                )
            ]
            sets.append(
                {
                    "id": s["id"],
                    "created_at": s["created_at"],
                    "seed": s["seed"],
                    "requested_n": s["requested_n"],
                    "double_label_frac": s["double_label_frac"],
                    "run_ids": json.loads(s["run_ids_json"]),
                    "items": items,
                    "labels": labels,
                }
            )
    return {"format_version": FORMAT_VERSION, "label_sets": sets}


def write_export(export: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
    return out


def restore_export(export: dict, db_path: str | Path) -> tuple[int, int]:
    """Write an export into a database. Returns (items, labels) restored.

    Refuses to touch a label set that already exists rather than merging
    into it: a half-overwritten label set is harder to detect than a
    refusal, and the labels it would silently replace are the data this
    script exists to protect.
    """
    version = export.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {version!r}; this script writes "
            f"and reads version {FORMAT_VERSION}"
        )

    init_db(db_path)
    items_written = labels_written = 0
    with _session(db_path) as conn:
        for s in export["label_sets"]:
            exists = conn.execute(
                "SELECT 1 FROM label_sets WHERE id=?", (s["id"],)
            ).fetchone()
            if exists:
                raise ValueError(
                    f"Label set {s['id'][:8]} already exists in {db_path}. "
                    f"Restore into a fresh database, or delete that set first."
                )

            conn.execute(
                """INSERT INTO label_sets
                     (id, created_at, seed, requested_n, double_label_frac,
                      run_ids_json)
                   VALUES (?,?,?,?,?,?)""",
                (
                    s["id"], s["created_at"], s["seed"], s["requested_n"],
                    s["double_label_frac"], json.dumps(s["run_ids"]),
                ),
            )

            # position -> the id SQLite assigns here, so labels reattach
            # without carrying ids across databases.
            item_ids: dict[int, int] = {}
            by_position: dict[int, dict] = {}
            for item in s["items"]:
                cur = conn.execute(
                    """INSERT INTO label_items
                         (label_set_id, position, item_key, run_id, task_id,
                          field, expected, predicted, pass_number)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        s["id"], item["position"], item["item_key"],
                        item["run_id"], item["task_id"], item["field"],
                        item["expected"], item["predicted"], item["pass_number"],
                    ),
                )
                item_ids[item["position"]] = cur.lastrowid
                by_position[item["position"]] = item
                items_written += 1

            for label in s["labels"]:
                item = by_position[label["position"]]
                conn.execute(
                    """INSERT INTO human_labels
                         (label_set_id, item_id, run_id, task_id, field,
                          verdict, seconds, labeled_at, pass_number, labeler)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        s["id"], item_ids[label["position"]], item["run_id"],
                        item["task_id"], item["field"], label["verdict"],
                        label["seconds"], label["labeled_at"],
                        item["pass_number"], label["labeler"],
                    ),
                )
                labels_written += 1
    return items_written, labels_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Export destination")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--restore", metavar="FILE",
        help="Read this export into --db instead of writing one",
    )
    args = parser.parse_args()

    if args.restore:
        export = json.loads(Path(args.restore).read_text())
        items, labels = restore_export(export, args.db)
        print(f"Restored {items} items and {labels} labels into {args.db}")
        return 0

    export = build_export(args.db)
    out = write_export(export, args.out)
    total_items = sum(len(s["items"]) for s in export["label_sets"])
    total_labels = sum(len(s["labels"]) for s in export["label_sets"])
    print(
        f"Wrote {out} — {len(export['label_sets'])} label set(s), "
        f"{total_items} items, {total_labels} labels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
