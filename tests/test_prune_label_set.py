"""Pruning removes settled items and their labels, and nothing else."""

import sqlite3
from pathlib import Path

import pytest

from harness.labeling import load_label_set
from harness.store import init_db
from scripts.prune_label_set import prune, settled_items


@pytest.fixture
def label_db(tmp_path: Path) -> Path:
    path = tmp_path / "labels.db"
    init_db(path)
    rows = [
        # field, expected, predicted, labelled?
        ("cpt_codes", "['99214']", "['99214-25']", True),    # settled
        ("cpt_codes", "['99214']", "['99215']", False),      # judgment
        ("member_id", "EG-441002", "EG-441002-A", True),     # settled
        ("patient_name", "DOE, JANE", "Jane Doe", True),     # judgment
        ("patient_name", "SMITH, AL", "Al Smith", False),    # judgment
    ]
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO label_sets (id, created_at, seed, requested_n,
                                       double_label_frac, run_ids_json)
               VALUES ('s1','2026-01-01T00:00:00Z',0,5,0.0,'[\"run1\"]')"""
        )
        for position, (field, exp, pred, labelled) in enumerate(rows):
            cur = conn.execute(
                """INSERT INTO label_items (label_set_id, position, item_key,
                                            run_id, task_id, field, expected,
                                            predicted, pass_number)
                   VALUES ('s1',?,?, 'run1',?,?,?,?,1)""",
                (position, f"k{position}", f"t{position}", field, exp, pred),
            )
            if labelled:
                conn.execute(
                    """INSERT INTO human_labels (label_set_id, item_id, run_id,
                                                 task_id, field, verdict, seconds,
                                                 labeled_at, pass_number, labeler)
                       VALUES ('s1',?, 'run1',?,?, 'equivalent',2.0,
                               '2026-01-01T00:00:00Z',1,'self')""",
                    (cur.lastrowid, f"t{position}", field),
                )
    return path


def _counts(path: Path) -> tuple[int, int]:
    with sqlite3.connect(path) as conn:
        return (
            conn.execute("SELECT COUNT(*) FROM label_items").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0],
        )


def test_only_settled_items_are_identified(label_db: Path):
    settled = settled_items(load_label_set("s1", label_db))

    assert sorted(i.field for i, _ in settled) == ["cpt_codes", "member_id"]
    assert {v for _, v in settled} == {"different"}


def test_prune_removes_the_items_and_their_labels(label_db: Path):
    label_set = load_label_set("s1", label_db)
    ids = [i.item_id for i, _ in settled_items(label_set)]

    items, labels = prune("s1", ids, label_db)

    assert (items, labels) == (2, 2)
    assert _counts(label_db) == (3, 1)


def test_prune_leaves_judgment_items_untouched(label_db: Path):
    label_set = load_label_set("s1", label_db)
    prune("s1", [i.item_id for i, _ in settled_items(label_set)], label_db)

    remaining = load_label_set("s1", label_db)

    assert sorted(i.field for i in remaining.items) == [
        "cpt_codes", "patient_name", "patient_name"
    ]
    # The surviving cpt_codes item is the genuine disagreement, not a modifier.
    kept = next(i for i in remaining.items if i.field == "cpt_codes")
    assert kept.predicted == "['99215']"


def test_prune_on_an_empty_list_writes_nothing(label_db: Path):
    before = _counts(label_db)

    assert prune("s1", [], label_db) == (0, 0)
    assert _counts(label_db) == before
