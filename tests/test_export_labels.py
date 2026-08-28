"""The label export is a backup, so the test that matters is restore."""

import json
import sqlite3
from pathlib import Path

import pytest

from harness.labeling import build_label_set, load_labels, save_label
from harness.store import init_db
from scripts.export_labels import FORMAT_VERSION, build_export, restore_export

JUDGE_REASON = "SENTINEL-judge-rationale-must-not-be-exported"
JUDGE_MODEL = "SENTINEL-judge-model"


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """One run, judged fields, and a few human labels."""
    path = tmp_path / "labels.db"
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO runs (id, created_at, git_commit, git_dirty, adapter,
                                 model_id, prompt_name, prompt_hash, task_count,
                                 total_cost_usd, total_tokens_in, total_tokens_out,
                                 wall_seconds, harness_version)
               VALUES ('run1','2026-01-01T00:00:00Z',NULL,0,'anthropic','m',
                       'extract_v1','ph',40,0.1,10,10,1.0,'0.1.0')"""
        )
        for n in range(40):
            verdict = "equivalent" if n % 4 else "different"
            conn.execute(
                """INSERT INTO results (run_id, task_id, category, difficulty,
                                        edge_case, response_text, tokens_in,
                                        tokens_out, latency_ms, cost_usd,
                                        cached, error)
                   VALUES ('run1',?,'hard','hard',1,'{}',1,1,1.0,0.0,0,NULL)""",
                (f"t{n}",),
            )
            conn.execute(
                """INSERT INTO judge_calls (run_id, task_id, field, expected,
                                            predicted, verdict, reason, cost_usd,
                                            judge_model_id, judge_prompt_hash,
                                            created_at)
                   VALUES ('run1',?,'member_id',?,?,?,?,0.001,?,'jh',
                           '2026-01-01T00:00:00Z')""",
                (f"t{n}", f"exp{n}", f"pred{n}", verdict, JUDGE_REASON, JUDGE_MODEL),
            )
    return path


@pytest.fixture
def labelled_db(seeded_db: Path) -> Path:
    label_set = build_label_set(["run1"], n=20, seed=3, db_path=seeded_db)
    for i, item in enumerate(label_set.items[:8]):
        save_label(
            label_set.id, item,
            "different" if i % 3 == 0 else "equivalent",
            1.5 + i, db_path=seeded_db,
        )
    return seeded_db


def _json(export: dict) -> str:
    return json.dumps(export, indent=2, sort_keys=True)


# --- the backup actually restores -------------------------------------------


def test_export_restores_byte_identically(labelled_db: Path, tmp_path: Path):
    # The only property that makes this a backup rather than a report.
    original = build_export(labelled_db)
    fresh = tmp_path / "restored.db"

    restore_export(original, fresh)

    assert _json(build_export(fresh)) == _json(original)


def test_restored_verdicts_match_by_position(labelled_db: Path, tmp_path: Path):
    original = build_export(labelled_db)
    fresh = tmp_path / "restored.db"
    restore_export(original, fresh)

    before = {(r["task_id"], r["field"], r["verdict"]) for r in
              load_labels(original["label_sets"][0]["id"], labelled_db)}
    after = {(r["task_id"], r["field"], r["verdict"]) for r in
             load_labels(original["label_sets"][0]["id"], fresh)}

    assert before == after


def test_restore_survives_a_database_that_assigns_different_row_ids(
    labelled_db: Path, tmp_path: Path
):
    # Labels are keyed by position precisely so a restore does not depend on
    # inheriting AUTOINCREMENT ids. Occupy the target's id space first, so
    # the ids it hands out cannot coincide with the exported ones.
    export = build_export(labelled_db)
    target = tmp_path / "occupied.db"
    init_db(target)
    with sqlite3.connect(target) as conn:
        conn.execute(
            """INSERT INTO label_sets (id, created_at, seed, requested_n,
                                       double_label_frac, run_ids_json)
               VALUES ('decoy','2026-01-01T00:00:00Z',0,5,0.0,'[]')"""
        )
        for position in range(500):
            conn.execute(
                """INSERT INTO label_items (label_set_id, position, item_key,
                                            run_id, task_id, field, expected,
                                            predicted, pass_number)
                   VALUES ('decoy',?,?,'r','t','member_id','e','p',1)""",
                (position, f"decoy:{position}"),
            )

    restore_export(export, target)

    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT li.position, h.verdict
                 FROM human_labels h JOIN label_items li ON li.id = h.item_id
                WHERE h.label_set_id = ?
                ORDER BY li.position""",
            (export["label_sets"][0]["id"],),
        ).fetchall()
        lowest = conn.execute(
            "SELECT MIN(id) FROM label_items WHERE label_set_id = ?",
            (export["label_sets"][0]["id"],),
        ).fetchone()[0]

    # The premise: the restored rows really did get different ids.
    assert lowest > 500

    expected = [(l["position"], l["verdict"])
                for l in export["label_sets"][0]["labels"]]
    assert [(r["position"], r["verdict"]) for r in rows] == expected


# --- blinding ---------------------------------------------------------------


def test_export_carries_no_judge_data(labelled_db: Path):
    # A backup that embedded the judge's verdict would reintroduce, through
    # the file, exactly what label_items has no column for.
    blob = _json(build_export(labelled_db))

    assert JUDGE_REASON not in blob
    assert JUDGE_MODEL not in blob
    assert "judge" not in blob.lower()


# --- refusals ---------------------------------------------------------------


def test_restore_refuses_to_overwrite_an_existing_label_set(labelled_db: Path):
    # Silently merging into a live set could replace hand-typed verdicts,
    # which is the one thing this file exists to prevent losing.
    export = build_export(labelled_db)

    with pytest.raises(ValueError, match="already exists"):
        restore_export(export, labelled_db)


def test_restore_rejects_an_unknown_format_version(labelled_db: Path, tmp_path: Path):
    export = build_export(labelled_db)
    export["format_version"] = FORMAT_VERSION + 1

    with pytest.raises(ValueError, match="format_version"):
        restore_export(export, tmp_path / "x.db")


def test_export_is_deterministic(labelled_db: Path):
    # Committed output has to diff on verdicts changing, not on dict order.
    assert _json(build_export(labelled_db)) == _json(build_export(labelled_db))
