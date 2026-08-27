"""The gate that makes calibration load-bearing rather than advisory."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from harness.calibration import AgreementReport
from harness.store import find_calibration, init_db, save_calibration

RUBRIC = "abc123def456"


def _report(n: int = 20) -> AgreementReport:
    return AgreementReport(
        n=n, raw_agreement=0.9, kappa=0.61, kappa_ci=(0.42, 0.78),
        band="usable with caveats", confusion={("equivalent", "equivalent"): n},
        per_category={}, excluded={}, human_ceiling_kappa=0.78, human_ceiling_n=12,
    )


def test_no_calibration_for_an_unmeasured_rubric(tmp_path: Path):
    db = tmp_path / "c.db"
    init_db(db)

    assert find_calibration(RUBRIC, db_path=db) is None


def test_calibration_round_trips(tmp_path: Path):
    db = tmp_path / "c.db"
    init_db(db)

    save_calibration(_report(), "labelset1", "claude-sonnet-5", RUBRIC, db_path=db)
    found = find_calibration(RUBRIC, db_path=db)

    assert found is not None
    assert found["kappa"] == pytest.approx(0.61)
    assert found["kappa_ci_low"] == pytest.approx(0.42)
    assert found["human_ceiling_kappa"] == pytest.approx(0.78)
    assert found["band"] == "usable with caveats"


def test_calibration_is_keyed_to_the_rubric_hash(tmp_path: Path):
    # Editing the rubric invalidates the calibration: verdicts produced
    # under a new rubric are not comparable to the ones measured.
    db = tmp_path / "c.db"
    init_db(db)
    save_calibration(_report(), "ls", "m", RUBRIC, db_path=db)

    assert find_calibration(RUBRIC, db_path=db) is not None
    assert find_calibration("a-different-rubric", db_path=db) is None


def test_undefined_kappa_is_stored_as_null_not_zero(tmp_path: Path):
    # Storing NaN as 0.0 would later read as "measured, and terrible"
    # rather than "not measurable".
    db = tmp_path / "c.db"
    init_db(db)
    report = AgreementReport(
        n=5, raw_agreement=1.0, kappa=float("nan"),
        kappa_ci=(float("nan"), float("nan")), band="undefined",
        confusion={}, per_category={}, excluded={},
    )
    save_calibration(report, "ls", "m", RUBRIC, db_path=db)

    found = find_calibration(RUBRIC, db_path=db)
    assert found["kappa"] is None
    assert found["kappa_ci_low"] is None


def test_most_recent_calibration_wins(tmp_path: Path):
    db = tmp_path / "c.db"
    init_db(db)
    save_calibration(_report(n=10), "ls1", "m", RUBRIC, db_path=db)
    save_calibration(_report(n=99), "ls2", "m", RUBRIC, db_path=db)

    assert find_calibration(RUBRIC, db_path=db)["n"] == 99


def test_find_calibration_migrates_a_store_written_before_the_table(tmp_path: Path):
    # A store written by an earlier version of *this* schema has the full
    # runs table but no calibrations table. It must migrate, not fail.
    db = tmp_path / "old.db"
    init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE calibrations")

    assert find_calibration(RUBRIC, db_path=db) is None  # migrates silently

    save_calibration(_report(), "ls", "m", RUBRIC, db_path=db)
    assert find_calibration(RUBRIC, db_path=db) is not None


def test_find_calibration_on_an_unmigratable_store_reports_uncalibrated(tmp_path: Path):
    # A store too different to migrate must read as uncalibrated rather than
    # crash: the gate then refuses, which is the safe direction.
    db = tmp_path / "foreign.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE runs (id TEXT PRIMARY KEY)")

    assert find_calibration(RUBRIC, db_path=db) is None


def test_find_calibration_on_a_missing_database_returns_none(tmp_path: Path):
    assert find_calibration(RUBRIC, db_path=tmp_path / "nope.db") is None


# --- the CLI gate ------------------------------------------------------------


def _run_eval(args: list[str], db: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--db", str(db), *args],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(db.parent)},
    )


def test_judge_is_refused_without_calibration(tmp_path: Path):
    db = tmp_path / "gate.db"
    init_db(db)

    done = _run_eval(["--adapter", "anthropic", "--limit", "1",
                      "--judge", "anthropic"], db)

    combined = done.stdout + done.stderr
    assert "No calibration exists" in combined
    assert "--uncalibrated" in combined


def test_running_without_a_judge_needs_no_calibration(tmp_path: Path):
    # The gate must only apply to judge-adjusted scores; a plain
    # deterministic run is unaffected by it.
    db = tmp_path / "gate.db"
    init_db(db)

    done = _run_eval(["--list-runs"], db)

    assert "No calibration" not in (done.stdout + done.stderr)
