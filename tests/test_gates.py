"""Gate evaluation, config validation, and the significance guard."""

import subprocess
import sys
from pathlib import Path

import pytest

from harness.calibration import (
    MIN_CALIBRATION_N,
    AgreementReport,
    calibration_is_usable,
)
from harness.gates import (
    GateConfigError,
    evaluate_gates,
    load_config,
    metrics_for,
)
from harness.store import init_db, load_run, save_calibration, save_run
from harness.types import ModelResponse, RunSummary, Score, Task, TaskResult

CONFIG = """
baseline_run_id: null
gates:
  - metric: mean_f1
    min: 0.80
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "g.db"
    init_db(path)
    return path


def _cfg(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "gates.yaml"
    path.write_text(text)
    return path


def _save(db: Path, f1s: list[float], adapter="anthropic", cost=0.01,
          dirty=False, failures=0, hallucinations=0, nullables=0,
          latency=100.0) -> str:
    tasks, results = [], []
    for i, f1 in enumerate(f1s):
        tasks.append(Task(id=f"eob-{i:03d}", category="clean", difficulty="easy",
                          edge_case=False, input="doc", expected={}))
        fields = {"patient_name": "tp" if f1 == 1.0 else "fp_fn_wrong"}
        for j in range(nullables):
            fields[f"n{j}"] = "fp_hallucinated" if j < hallucinations else "tn"
        results.append(TaskResult(
            task_id=f"eob-{i:03d}",
            response=ModelResponse(text="{}", model_id="m", tokens_in=1, tokens_out=1,
                                   latency_ms=latency, cost_usd=cost / max(1, len(f1s)),
                                   finish_reason="end_turn", raw={}),
            scores=[
                Score(scorer="schema", value=1.0, passed=True, detail={}),
                Score(scorer="fields", value=f1, passed=f1 == 1.0,
                      detail={"f1": f1, "fields": fields}),
            ],
        ))
    for i in range(failures):
        tasks.append(Task(id=f"err-{i}", category="clean", difficulty="easy",
                          edge_case=False, input="d", expected={}))
        results.append(TaskResult(task_id=f"err-{i}", response=None, scores=[],
                                  error="FatalError: boom"))

    summary = RunSummary(
        results=results, adapter_name=adapter, model_id="m",
        prompt_name="extract_v1", prompt_hash="ph", total_cost_usd=cost,
        total_tokens_in=10, total_tokens_out=10, wall_clock_seconds=1.0,
        succeeded=len(f1s), failed=failures, cached=0,
        latency_p50_ms=latency, latency_p95_ms=latency,
    )
    import harness.store as store_module

    original = store_module._git_info
    store_module._git_info = lambda: ("abc1234", dirty)
    try:
        return save_run(summary, tasks, db_path=db)
    finally:
        store_module._git_info = original


# --- config validation -------------------------------------------------------


def test_valid_config_loads(tmp_path: Path):
    baseline, gates = load_config(_cfg(tmp_path, CONFIG))

    assert baseline is None
    assert gates[0].metric == "mean_f1"
    assert gates[0].bound == "min"


def test_missing_config_fails_loudly(tmp_path: Path):
    with pytest.raises(GateConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_unknown_metric_fails_loudly(tmp_path: Path):
    text = "gates:\n  - metric: vibes\n    min: 0.5\n"
    with pytest.raises(GateConfigError, match="unknown metric"):
        load_config(_cfg(tmp_path, text))


def test_gate_without_a_bound_fails_loudly(tmp_path: Path):
    text = "gates:\n  - metric: mean_f1\n"
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_config(_cfg(tmp_path, text))


def test_gate_with_two_bounds_fails_loudly(tmp_path: Path):
    text = "gates:\n  - metric: mean_f1\n    min: 0.5\n    max: 0.9\n"
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_config(_cfg(tmp_path, text))


def test_non_numeric_threshold_fails_loudly(tmp_path: Path):
    text = "gates:\n  - metric: mean_f1\n    min: high\n"
    with pytest.raises(GateConfigError, match="must be a number"):
        load_config(_cfg(tmp_path, text))


def test_require_significant_on_an_absolute_gate_fails_loudly(tmp_path: Path):
    # It has no meaning there, so accepting it would mislead.
    text = "gates:\n  - metric: mean_f1\n    min: 0.5\n    require_significant: true\n"
    with pytest.raises(GateConfigError, match="only applies to"):
        load_config(_cfg(tmp_path, text))


def test_malformed_yaml_fails_loudly(tmp_path: Path):
    with pytest.raises(GateConfigError, match="invalid YAML"):
        load_config(_cfg(tmp_path, "gates: [unclosed\n"))


def test_unknown_top_level_key_fails_loudly(tmp_path: Path):
    with pytest.raises(GateConfigError, match="unknown top-level"):
        load_config(_cfg(tmp_path, "gates:\n  - metric: mean_f1\n    min: 0.5\nfoo: 1\n"))


# --- metric extraction -------------------------------------------------------


def test_metrics_cover_every_gateable_name(db: Path):
    run = _save(db, [1.0, 0.5], nullables=2, hallucinations=1)
    values = metrics_for(load_run(run, db))

    assert set(values) >= {"schema_pass_rate", "mean_f1", "cost_per_task",
                           "p95_latency_ms", "hallucination_rate"}


def test_hallucination_rate_is_measured_against_nullable_cases(db: Path):
    # 2 tasks x 4 nullable fields = 8 nullable cases, 1 hallucinated each = 2
    run = _save(db, [1.0, 1.0], nullables=4, hallucinations=1)

    assert metrics_for(load_run(run, db))["hallucination_rate"] == pytest.approx(2 / 8)


# --- absolute gates ----------------------------------------------------------


def test_min_gate_passes_and_fails(db: Path, tmp_path: Path):
    good = _save(db, [1.0, 0.9])
    bad = _save(db, [0.5, 0.5])
    config = _cfg(tmp_path, CONFIG)

    assert evaluate_gates(good, config, db_path=db).passed
    assert not evaluate_gates(bad, config, db_path=db).passed


def test_max_gate_passes_and_fails(db: Path, tmp_path: Path):
    text = "gates:\n  - metric: cost_per_task\n    max: 0.001\n"
    config = _cfg(tmp_path, text)
    cheap = _save(db, [1.0] * 10, cost=0.001)
    dear = _save(db, [1.0] * 10, cost=1.0)

    assert evaluate_gates(cheap, config, db_path=db).passed
    assert not evaluate_gates(dear, config, db_path=db).passed


def test_exit_code_is_one_on_failure_zero_on_pass(db: Path, tmp_path: Path):
    config = _cfg(tmp_path, CONFIG)

    assert evaluate_gates(_save(db, [1.0]), config, db_path=db).exit_code == 0
    assert evaluate_gates(_save(db, [0.1]), config, db_path=db).exit_code == 1


# --- regression gates --------------------------------------------------------


REGRESSION = """
baseline_run_id: "{baseline}"
gates:
  - metric: mean_f1
    max_regression_vs_baseline: 0.03
    require_significant: {significant}
"""


def test_regression_gate_is_skipped_without_a_baseline(db: Path, tmp_path: Path):
    config = _cfg(tmp_path, "gates:\n  - metric: mean_f1\n"
                            "    max_regression_vs_baseline: 0.03\n")
    report = evaluate_gates(_save(db, [1.0]), config, db_path=db)

    assert report.results[0].skipped_reason
    assert report.passed  # a skipped gate does not fail the build


def test_a_real_regression_fails(db: Path, tmp_path: Path):
    baseline = _save(db, [1.0] * 40)
    degraded = _save(db, [0.5] * 40)
    config = _cfg(tmp_path, REGRESSION.format(baseline=baseline, significant="true"))

    report = evaluate_gates(degraded, config, db_path=db)

    assert not report.passed
    assert report.results[0].significant


def test_require_significant_suppresses_a_noise_sized_regression(db: Path, tmp_path: Path):
    # A tiny inconsistent wobble: larger than the threshold on the point
    # estimate for some tasks, but the interval spans zero.
    baseline = _save(db, [1.0, 0.0] * 20)
    noisy = _save(db, [0.0, 1.0] * 20)
    config = _cfg(tmp_path, REGRESSION.format(baseline=baseline, significant="true"))

    report = evaluate_gates(noisy, config, db_path=db)

    assert report.passed
    assert not report.results[0].significant


def test_without_require_significant_the_same_wobble_would_fail(db: Path, tmp_path: Path):
    baseline = _save(db, [1.0] * 30)
    slightly_worse = _save(db, [0.90] * 30)
    strict = _cfg(tmp_path, REGRESSION.format(baseline=baseline, significant="false"))

    assert not evaluate_gates(slightly_worse, strict, db_path=db).passed


def test_regression_gate_names_the_driving_tasks(db: Path, tmp_path: Path):
    baseline = _save(db, [1.0] * 10)
    degraded = _save(db, [1.0, 1.0, 0.2, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0])
    config = _cfg(tmp_path, REGRESSION.format(baseline=baseline, significant="false"))

    driving = evaluate_gates(degraded, config, db_path=db).results[0].driving_tasks

    assert [d["task_id"] for d in driving[:2]] == ["eob-004", "eob-002"]
    assert driving[0]["delta"] < 0


def test_improvement_does_not_trip_a_regression_gate(db: Path, tmp_path: Path):
    baseline = _save(db, [0.5] * 30)
    better = _save(db, [0.95] * 30)
    config = _cfg(tmp_path, REGRESSION.format(baseline=baseline, significant="false"))

    assert evaluate_gates(better, config, db_path=db).passed


# --- calibration threshold ---------------------------------------------------


def test_a_tiny_calibration_does_not_count_as_calibrated():
    usable, why = calibration_is_usable({"n": 2, "kappa": 0.0})

    assert not usable
    assert "only 2" in why


def test_a_sufficient_calibration_counts():
    usable, why = calibration_is_usable({"n": MIN_CALIBRATION_N, "kappa": 0.6})

    assert usable and why is None


def test_an_undefined_kappa_does_not_count():
    usable, why = calibration_is_usable({"n": 100, "kappa": None})

    assert not usable
    assert "undefined kappa" in why


def test_absent_calibration_does_not_count():
    usable, why = calibration_is_usable(None)

    assert not usable


# --- set_baseline ------------------------------------------------------------


def _set_baseline(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/set_baseline.py", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_set_baseline_refuses_a_dirty_run(db: Path, tmp_path: Path):
    run = _save(db, [1.0] * 5, dirty=True)
    config = _cfg(tmp_path, CONFIG)

    done = _set_baseline([run, "--db", str(db), "--config", str(config),
                          "--allow-uncalibrated"])

    assert done.returncode == 1
    assert "dirty working tree" in done.stdout


def test_set_baseline_refuses_a_run_with_failures(db: Path, tmp_path: Path):
    run = _save(db, [1.0] * 5, failures=2)
    config = _cfg(tmp_path, CONFIG)

    done = _set_baseline([run, "--db", str(db), "--config", str(config),
                          "--allow-uncalibrated"])

    assert done.returncode == 1
    assert "failed in this run" in done.stdout


def test_set_baseline_refuses_an_uncalibrated_run(db: Path, tmp_path: Path):
    run = _save(db, [1.0] * 5)
    config = _cfg(tmp_path, CONFIG)

    done = _set_baseline([run, "--db", str(db), "--config", str(config)])

    assert done.returncode == 1
    assert "not calibrated" in done.stdout


def test_set_baseline_accepts_a_clean_run_and_records_provenance(db: Path, tmp_path: Path):
    run = _save(db, [1.0] * 5)
    config = _cfg(tmp_path, CONFIG)

    done = _set_baseline([run, "--db", str(db), "--config", str(config),
                          "--allow-uncalibrated"])

    assert done.returncode == 0
    text = config.read_text()
    assert run in text
    assert "baseline promoted by" in text


def test_promoting_twice_does_not_duplicate_the_provenance_comment(db: Path, tmp_path: Path):
    config = _cfg(tmp_path, CONFIG)
    for _ in range(2):
        run = _save(db, [1.0] * 5)
        _set_baseline([run, "--db", str(db), "--config", str(config),
                       "--allow-uncalibrated"])

    assert config.read_text().count("baseline promoted by") == 1


# --- synthetic labels can never establish calibration ------------------------


def test_synthetic_labels_do_not_establish_calibration():
    # The whole point of the guard: a kappa derived from model-generated
    # labels measures one model agreeing with another.
    usable, why = calibration_is_usable(
        {"n": 100, "kappa": 0.8, "labelers_json": '["synthetic"]'}
    )

    assert not usable
    assert "synthetic labels" in why


def test_a_single_synthetic_labeller_taints_a_mixed_set():
    # Mixing a few real labels in must not launder the rest.
    usable, _ = calibration_is_usable(
        {"n": 100, "kappa": 0.8, "labelers_json": '["self", "synthetic"]'}
    )

    assert not usable


def test_human_labels_still_establish_calibration():
    usable, why = calibration_is_usable(
        {"n": 100, "kappa": 0.8, "labelers_json": '["self", "second-rater"]'}
    )

    assert usable and why is None


def test_absent_or_malformed_labeller_record_does_not_crash():
    for value in (None, "", "not json", "[]"):
        usable, _ = calibration_is_usable(
            {"n": 100, "kappa": 0.8, "labelers_json": value}
        )
        assert usable is True


def test_labellers_round_trip_through_the_store(tmp_path: Path):
    from harness.calibration import AgreementReport
    from harness.store import find_calibration

    db = tmp_path / "lab.db"
    init_db(db)
    report = AgreementReport(
        n=50, raw_agreement=0.8, kappa=0.5, kappa_ci=(0.3, 0.7),
        band="weak", confusion={}, per_category={}, excluded={},
    )
    save_calibration(report, "ls", "m", "rubric-x",
                     labelers=["synthetic", "synthetic", "self"], db_path=db)

    stored = find_calibration("rubric-x", db_path=db)

    assert stored["labelers_json"] == '["self", "synthetic"]'
    assert not calibration_is_usable(stored)[0]


def test_migration_adds_the_labeller_column_to_an_older_store(tmp_path: Path):
    # A store written before the column existed must gain it rather than
    # failing, and CREATE TABLE IF NOT EXISTS will not do that on its own.
    import sqlite3

    db = tmp_path / "old.db"
    init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE calibrations DROP COLUMN labelers_json")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calibrations)")}
        assert "labelers_json" not in cols

    init_db(db)

    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calibrations)")}
    assert "labelers_json" in cols
