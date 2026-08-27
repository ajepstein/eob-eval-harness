"""Sampling and storage for blind human labelling.

Week 2C measures the judge against these labels, so everything here exists
to keep them uncontaminated.

**Blinding is structural.** The judge's verdict is read while *sampling* —
it has to be, to stratify — and is then discarded. It is never written to
``label_items``, so the labelling UI cannot display it even by accident.
The guarantee is a property of the schema, not of anyone remembering.

**Stratification.** Sampling uniformly from a suite where most fields match
would produce a queue of trivial cases that says nothing about the judge's
discrimination. Items are drawn proportionally across (category, verdict)
strata so both verdicts survive into the sample.

**Double labelling.** A fraction of items appear twice, far apart in the
queue. Agreement with yourself is the ceiling on any agreement the judge
could achieve: a judge at kappa 0.72 against a human ceiling of 0.78 is a
very different result from 0.72 against a ceiling of 0.95.
"""

from __future__ import annotations

import json
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.calibration import NON_HUMAN_LABELERS
from harness.store import DEFAULT_DB_PATH, _resolve_run_id, _session, init_db

# Minimum queue distance between the two passes of a double-labelled item.
# Close repeats measure short-term memory rather than genuine consistency.
MIN_REPEAT_GAP = 20


@dataclass(frozen=True)
class LabelItem:
    """One thing to label.

    Carries no judge verdict, by design — see the module docstring.
    """

    item_id: int
    position: int
    item_key: str
    run_id: str
    task_id: str
    field: str
    expected: str | None
    predicted: str | None
    pass_number: int


@dataclass(frozen=True)
class LabelSet:
    id: str
    seed: int
    requested_n: int
    double_label_frac: float
    run_ids: list[str]
    items: list[LabelItem]


def _candidates(conn, run_ids: list[str]) -> list[dict]:
    """Judged fields, with the category needed for stratification."""
    rows = []
    for run_id in run_ids:
        for row in conn.execute(
            """SELECT j.run_id, j.task_id, j.field, j.expected, j.predicted,
                      j.verdict, r.category
               FROM judge_calls j
               JOIN results r ON r.run_id = j.run_id AND r.task_id = j.task_id
               WHERE j.run_id = ?""",
            (run_id,),
        ).fetchall():
            rows.append(dict(row))
    return rows


def _stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Draw n items proportionally across (category, verdict) strata.

    Proportional allocation with a guaranteed floor of one per stratum, so a
    small but meaningful stratum — a handful of "different" verdicts among
    many "equivalent" ones — cannot be rounded out of the sample entirely.
    """
    if n >= len(rows):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(row["category"], row["verdict"])].append(row)

    for bucket in strata.values():
        rng.shuffle(bucket)

    total = len(rows)
    allocation: dict[tuple, int] = {}
    for key, bucket in strata.items():
        allocation[key] = min(len(bucket), max(1, round(n * len(bucket) / total)))

    # Proportional rounding rarely lands exactly on n, so correct the drift
    # rather than silently returning the wrong sample size.
    #
    # Both loops check that they made progress. Without that, a request for
    # fewer items than there are strata spins forever: every stratum sits at
    # its floor of one, the total stays above n, and nothing is decrementable.
    order = sorted(strata, key=lambda k: -len(strata[k]))

    while sum(allocation.values()) > n:
        reduced = False
        for key in order:
            if sum(allocation.values()) <= n:
                break
            if allocation[key] > 1:
                allocation[key] -= 1
                reduced = True
        if not reduced:
            # More strata than requested items. One per stratum is no longer
            # possible, so drop whole strata smallest-first, keeping the
            # largest represented.
            for key in sorted(order, key=lambda k: len(strata[k])):
                if sum(allocation.values()) <= n:
                    break
                allocation.pop(key, None)
            break

    while sum(allocation.values()) < n:
        grew = False
        for key in order:
            if sum(allocation.values()) >= n:
                break
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                grew = True
        if not grew:
            # Every stratum is exhausted; n exceeds what the population holds.
            break

    picked: list[dict] = []
    for key, count in allocation.items():
        picked.extend(strata[key][:count])
    rng.shuffle(picked)
    return picked


def _interleave_repeats(
    picked: list[dict], repeats: list[dict], rng: random.Random
) -> list[tuple[dict, int]]:
    """Place second passes at least MIN_REPEAT_GAP positions after the first."""
    queue: list[tuple[dict, int]] = [(row, 1) for row in picked]
    for row in repeats:
        first = next(i for i, (r, p) in enumerate(queue) if r is row and p == 1)
        earliest = first + MIN_REPEAT_GAP + 1
        if earliest >= len(queue):
            queue.append((row, 2))
        else:
            queue.insert(rng.randint(earliest, len(queue)), (row, 2))
    return queue


def build_label_set(
    run_ids: list[str],
    n: int = 200,
    seed: int = 0,
    double_label_frac: float = 0.15,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> LabelSet:
    """Sample a reproducible, stratified, blind label set."""
    init_db(db_path)
    rng = random.Random(seed)
    set_id = str(uuid.uuid4())

    with _session(db_path) as conn:
        resolved = [_resolve_run_id(conn, r) for r in run_ids]
        rows = _candidates(conn, resolved)
        if not rows:
            raise ValueError(
                "No judged fields found in those runs. Run with --judge first — "
                "there is nothing to label until the judge has produced verdicts."
            )

        picked = _stratified_sample(rows, n, rng)

        repeat_count = int(round(len(picked) * double_label_frac))
        repeats = picked[:repeat_count] if repeat_count else []
        queue = _interleave_repeats(picked, repeats, rng)

        conn.execute(
            """INSERT INTO label_sets (id, created_at, seed, requested_n,
                                       double_label_frac, run_ids_json)
               VALUES (?,?,?,?,?,?)""",
            (
                set_id,
                datetime.now(timezone.utc).isoformat(),
                seed,
                n,
                double_label_frac,
                json.dumps(resolved),
            ),
        )
        for position, (row, pass_number) in enumerate(queue):
            # NOTE: row["verdict"] is deliberately not written.
            conn.execute(
                """INSERT INTO label_items (label_set_id, position, item_key,
                                            run_id, task_id, field, expected,
                                            predicted, pass_number)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    set_id,
                    position,
                    f"{row['run_id']}:{row['task_id']}:{row['field']}",
                    row["run_id"],
                    row["task_id"],
                    row["field"],
                    row["expected"],
                    row["predicted"],
                    pass_number,
                ),
            )

    return load_label_set(set_id, db_path)


def _resolve_set_id(conn, set_id: str) -> str:
    """Accept a full label-set id or an unambiguous prefix.

    Every public function that takes a set id goes through this. Without it
    the accessors disagree: some resolve prefixes and others match exactly,
    so a short id silently returns no labels rather than raising — which
    reads as "nothing has been labelled" and is the worst possible way to be
    wrong about that.
    """
    row = conn.execute(
        "SELECT id FROM label_sets WHERE id = ? OR id LIKE ? || '%'",
        (set_id, set_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"No label set matching {set_id!r}")
    return row["id"]


def load_label_set(set_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> LabelSet:
    with _session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM label_sets WHERE id = ? OR id LIKE ? || '%'",
            (set_id, set_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"No label set matching {set_id!r}")
        items = [
            LabelItem(
                item_id=r["id"], position=r["position"], item_key=r["item_key"],
                run_id=r["run_id"], task_id=r["task_id"], field=r["field"],
                expected=r["expected"], predicted=r["predicted"],
                pass_number=r["pass_number"],
            )
            for r in conn.execute(
                "SELECT * FROM label_items WHERE label_set_id = ? ORDER BY position",
                (row["id"],),
            ).fetchall()
        ]
    return LabelSet(
        id=row["id"], seed=row["seed"], requested_n=row["requested_n"],
        double_label_frac=row["double_label_frac"],
        run_ids=json.loads(row["run_ids_json"]), items=items,
    )


def save_label(
    label_set_id: str, item: LabelItem, verdict: str, seconds: float,
    labeler: str = "self", db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Record one verdict. Written immediately so a quit never loses work."""
    with _session(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO human_labels
                 (label_set_id, item_id, run_id, task_id, field, verdict,
                  seconds, labeled_at, pass_number, labeler)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                label_set_id, item.item_id, item.run_id, item.task_id, item.field,
                verdict, seconds, datetime.now(timezone.utc).isoformat(),
                item.pass_number, labeler,
            ),
        )


def labelled_item_ids(label_set_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> set[int]:
    with _session(db_path) as conn:
        full_id = _resolve_set_id(conn, label_set_id)
        return {
            r["item_id"]
            for r in conn.execute(
                "SELECT item_id FROM human_labels WHERE label_set_id = ?",
                (full_id,),
            ).fetchall()
        }


def load_labels(label_set_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    with _session(db_path) as conn:
        full_id = _resolve_set_id(conn, label_set_id)
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM human_labels WHERE label_set_id = ? ORDER BY id",
                (full_id,),
            ).fetchall()
        ]


def list_label_sets(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    if not Path(db_path).exists():
        return []
    with _session(db_path) as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM label_sets ORDER BY created_at DESC"
            ).fetchall()
        ]


def clear_labels(
    label_set_id: str,
    labelers: frozenset[str] | set[str] = NON_HUMAN_LABELERS,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[int, int]:
    """Delete labels from the named labellers. Returns (removed, kept).

    Defaults to the non-human labellers only. Human labels are real work and
    are never touched by this — clearing a synthetic run must not destroy
    verdicts somebody actually sat and made.

    Calibrations computed from the removed labels are dropped alongside
    them. Leaving one behind would be harmless (the guard already refuses
    synthetic calibrations) but confusing: the store would carry a record
    whose inputs no longer exist.
    """
    if not labelers:
        return 0, 0

    placeholders = ",".join("?" for _ in labelers)
    names = sorted(labelers)

    with _session(db_path) as conn:
        full_id = _resolve_set_id(conn, label_set_id)

        removed = conn.execute(
            f"SELECT COUNT(*) AS n FROM human_labels "
            f"WHERE label_set_id = ? AND labeler IN ({placeholders})",
            (full_id, *names),
        ).fetchone()["n"]
        conn.execute(
            f"DELETE FROM human_labels "
            f"WHERE label_set_id = ? AND labeler IN ({placeholders})",
            (full_id, *names),
        )
        kept = conn.execute(
            "SELECT COUNT(*) AS n FROM human_labels WHERE label_set_id = ?",
            (full_id,),
        ).fetchone()["n"]

        # Drop calibrations whose inputs have just been deleted.
        stale = []
        for row in conn.execute(
            "SELECT id, labelers_json FROM calibrations WHERE label_set_id = ?",
            (full_id,),
        ).fetchall():
            try:
                contributors = set(json.loads(row["labelers_json"] or "[]"))
            except (TypeError, ValueError):
                contributors = set()
            if contributors & set(names):
                stale.append(row["id"])
        for calibration_id in stale:
            conn.execute("DELETE FROM calibrations WHERE id = ?", (calibration_id,))

    return removed, kept
