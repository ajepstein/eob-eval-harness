import json
import sqlite3
from pathlib import Path

import pytest

import harness.store as store_module
from harness.scorers.fields import FieldScorer
from harness.store import (
    AmbiguousRunId,
    RunNotFound,
    compare_runs,
    init_db,
    list_runs,
    load_run,
    rescore_run,
    save_run,
)
from harness.types import ModelResponse, RunSummary, Score, Task, TaskResult

PERFECT = {
    "patient_name": "Jane Doe",
    "date_of_service": "2026-03-14",
    "provider_npi": None,
    "payer_name": "Northstar Health",
    "member_id": "NS-88213",
    "cpt_codes": ["99213"],
    "billed_amount": 340.00,
    "patient_responsibility": 45.00,
}


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _task(task_id: str, category: str = "clean") -> Task:
    return Task(
        id=task_id,
        category=category,
        difficulty="easy",
        edge_case=False,
        input=f"doc {task_id}",
        expected=dict(PERFECT),
    )


def _response(text: str | None = None) -> ModelResponse:
    return ModelResponse(
        text=text if text is not None else json.dumps(PERFECT),
        model_id="claude-sonnet-5",
        tokens_in=100,
        tokens_out=50,
        latency_ms=1234.5,
        cost_usd=0.001,
        finish_reason="end_turn",
        raw={"id": "msg_1"},
    )


def _summary(
    results: list[TaskResult], adapter: str = "anthropic", cost: float = 0.003
) -> RunSummary:
    return RunSummary(
        results=results,
        adapter_name=adapter,
        model_id="claude-sonnet-5",
        prompt_name="extract_v1",
        prompt_hash="abc123def456",
        total_cost_usd=cost,
        total_tokens_in=300,
        total_tokens_out=150,
        wall_clock_seconds=3.5,
        succeeded=len([r for r in results if r.error is None]),
        failed=len([r for r in results if r.error is not None]),
        cached=0,
        latency_p50_ms=1234.5,
        latency_p95_ms=1500.0,
    )


def _result(task_id: str, f1: float = 1.0, error: str | None = None) -> TaskResult:
    if error is not None:
        return TaskResult(task_id=task_id, response=None, scores=[], error=error)
    return TaskResult(
        task_id=task_id,
        response=_response(),
        scores=[
            Score(scorer="schema", value=1.0, passed=True, detail={"reason": "ok"}),
            Score(scorer="fields", value=f1, passed=f1 == 1.0, detail={"f1": f1}),
        ],
    )


# --- round trip -------------------------------------------------------------


def test_run_round_trips(db: Path):
    tasks = [_task("t1"), _task("t2")]
    summary = _summary([_result("t1"), _result("t2", f1=0.5)])

    run_id = save_run(summary, tasks, db_path=db)
    record = load_run(run_id, db_path=db)

    assert record.meta.run_id == run_id
    assert record.meta.adapter == "anthropic"
    assert record.meta.model_id == "claude-sonnet-5"
    assert record.meta.prompt_hash == "abc123def456"
    assert record.meta.task_count == 2
    assert record.meta.total_cost_usd == pytest.approx(0.003)
    assert [r.task_id for r in record.results] == ["t1", "t2"]


def test_raw_response_text_is_stored(db: Path):
    # The whole basis of --rescore: generations are paid for once.
    run_id = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    record = load_run(run_id, db_path=db)

    assert json.loads(record.results[0].response_text) == PERFECT


def test_scores_round_trip_with_detail(db: Path):
    run_id = save_run(_summary([_result("t1", f1=0.75)]), [_task("t1")], db_path=db)

    scores = load_run(run_id, db_path=db).results[0].scores

    assert [s.scorer for s in scores] == ["schema", "fields"]
    assert scores[1].value == pytest.approx(0.75)
    assert scores[1].detail == {"f1": 0.75}


def test_task_metadata_is_denormalized_onto_results(db: Path):
    run_id = save_run(
        _summary([_result("t1")]), [_task("t1", category="hard")], db_path=db
    )

    result = load_run(run_id, db_path=db).results[0]

    assert result.category == "hard"
    assert result.difficulty == "easy"


def test_aggregates_are_recomputed_from_scores(db: Path):
    summary = _summary([_result("t1", f1=1.0), _result("t2", f1=0.5)])
    run_id = save_run(summary, [_task("t1"), _task("t2")], db_path=db)

    meta = load_run(run_id, db_path=db).meta

    assert meta.mean_f1 == pytest.approx(0.75)
    assert meta.schema_pass_rate == pytest.approx(1.0)


def test_failed_task_is_recorded_and_counted(db: Path):
    summary = _summary([_result("t1"), _result("t2", error="FatalError: nope")])
    run_id = save_run(summary, [_task("t1"), _task("t2")], db_path=db)

    record = load_run(run_id, db_path=db)
    failed = next(r for r in record.results if r.task_id == "t2")

    assert failed.error == "FatalError: nope"
    assert failed.response_text is None
    assert record.meta.failures == 1


# --- init_db ----------------------------------------------------------------


def test_init_db_is_idempotent(tmp_path: Path):
    path = tmp_path / "idem.db"
    init_db(path)
    init_db(path)
    init_db(path)

    save_run(_summary([_result("t1")]), [_task("t1")], db_path=path)
    assert len(list_runs(db_path=path)) == 1


def test_init_db_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "deeper" / "runs.db"
    init_db(path)

    assert path.exists()


# --- list_runs --------------------------------------------------------------


def test_list_runs_orders_newest_first(db: Path):
    first = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)
    second = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    assert [m.run_id for m in list_runs(db_path=db)] == [second, first]


def test_list_runs_filters_by_adapter(db: Path):
    save_run(_summary([_result("t1")], adapter="anthropic"), [_task("t1")], db_path=db)
    openai_id = save_run(
        _summary([_result("t1")], adapter="openai"), [_task("t1")], db_path=db
    )

    runs = list_runs(adapter="openai", db_path=db)

    assert [m.run_id for m in runs] == [openai_id]


def test_list_runs_respects_limit(db: Path):
    for _ in range(5):
        save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    assert len(list_runs(limit=2, db_path=db)) == 2


def test_list_runs_on_missing_database_returns_empty(tmp_path: Path):
    assert list_runs(db_path=tmp_path / "does-not-exist.db") == []


# --- git provenance ---------------------------------------------------------


def test_dirty_git_tree_is_recorded(db: Path, monkeypatch: pytest.MonkeyPatch):
    # A run recorded against modified tracked files is not reproducible, and
    # that has to be visible in the record rather than discovered later.
    monkeypatch.setattr(store_module, "_git_info", lambda: ("deadbeef", True))

    run_id = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)
    meta = load_run(run_id, db_path=db).meta

    assert meta.git_dirty is True
    assert meta.git_commit == "deadbeef"


def test_clean_git_tree_is_recorded(db: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(store_module, "_git_info", lambda: ("cafe1234", False))

    run_id = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    assert load_run(run_id, db_path=db).meta.git_dirty is False


def test_git_info_survives_a_missing_git_binary(monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(store_module.subprocess, "run", boom)

    assert store_module._git_info() == (None, False)


# --- id resolution ----------------------------------------------------------


def test_run_can_be_loaded_by_unique_prefix(db: Path):
    run_id = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    assert load_run(run_id[:8], db_path=db).meta.run_id == run_id


def test_unknown_run_id_raises(db: Path):
    with pytest.raises(RunNotFound):
        load_run("nosuchrun", db_path=db)


def test_ambiguous_prefix_raises(db: Path, monkeypatch: pytest.MonkeyPatch):
    ids = iter(["abc11111-0000", "abc22222-0000"])
    monkeypatch.setattr(store_module.uuid, "uuid4", lambda: next(ids))

    save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)
    save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    with pytest.raises(AmbiguousRunId):
        load_run("abc", db_path=db)


# --- compare_runs -----------------------------------------------------------


def test_compare_runs_identifies_regression_and_improvement(db: Path):
    tasks = [_task("t1"), _task("t2"), _task("t3")]
    run_a = save_run(
        _summary([_result("t1", 1.0), _result("t2", 0.5), _result("t3", 0.8)]),
        tasks,
        db_path=db,
    )
    run_b = save_run(
        _summary([_result("t1", 0.4), _result("t2", 0.9), _result("t3", 0.8)]),
        tasks,
        db_path=db,
    )

    diff = compare_runs(run_a, run_b, db_path=db)

    assert [d.task_id for d in diff.regressed] == ["t1"]
    assert diff.regressed[0].delta == pytest.approx(-0.6)
    assert [d.task_id for d in diff.improved] == ["t2"]
    assert diff.improved[0].delta == pytest.approx(0.4)
    assert diff.unchanged == 1


def test_compare_runs_computes_aggregate_delta(db: Path):
    tasks = [_task("t1"), _task("t2")]
    run_a = save_run(_summary([_result("t1", 1.0), _result("t2", 1.0)]), tasks, db_path=db)
    run_b = save_run(_summary([_result("t1", 0.5), _result("t2", 0.5)]), tasks, db_path=db)

    diff = compare_runs(run_a, run_b, db_path=db)

    assert diff.mean_a == pytest.approx(1.0)
    assert diff.mean_b == pytest.approx(0.5)
    assert diff.mean_delta == pytest.approx(-0.5)


def test_comparing_a_run_against_itself_is_zero_delta(db: Path):
    tasks = [_task("t1"), _task("t2")]
    run_id = save_run(_summary([_result("t1", 1.0), _result("t2", 0.6)]), tasks, db_path=db)

    diff = compare_runs(run_id, run_id, db_path=db)

    assert diff.regressed == []
    assert diff.improved == []
    assert diff.mean_delta == pytest.approx(0.0)


def test_compare_runs_reports_tasks_present_in_only_one_run(db: Path):
    run_a = save_run(
        _summary([_result("t1"), _result("t2")]), [_task("t1"), _task("t2")], db_path=db
    )
    run_b = save_run(
        _summary([_result("t2"), _result("t3")]), [_task("t2"), _task("t3")], db_path=db
    )

    diff = compare_runs(run_a, run_b, db_path=db)

    assert diff.only_in_a == ["t1"]
    assert diff.only_in_b == ["t3"]


def test_compare_runs_treats_a_lost_score_as_a_regression(db: Path):
    # t1 errored in run B, so it has no field score. It must not vanish
    # from the comparison as though nothing changed.
    tasks = [_task("t1")]
    run_a = save_run(_summary([_result("t1", 1.0)]), tasks, db_path=db)
    run_b = save_run(_summary([_result("t1", error="FatalError: x")]), tasks, db_path=db)

    diff = compare_runs(run_a, run_b, db_path=db)

    assert [d.task_id for d in diff.regressed] == ["t1"]
    assert diff.regressed[0].delta == pytest.approx(-1.0)


# --- rescore ----------------------------------------------------------------


def test_rescore_recomputes_scores_from_stored_text(db: Path):
    # Save with a deliberately wrong stored score, then rescore with the
    # real scorer and confirm the stored value is corrected.
    tasks = [_task("t1")]
    run_id = save_run(_summary([_result("t1", f1=0.123)]), tasks, db_path=db)
    assert load_run(run_id, db_path=db).meta.mean_f1 == pytest.approx(0.123)

    record = rescore_run(run_id, tasks, [FieldScorer()], db_path=db)

    # Stored text is a perfect prediction, so the corrected F1 is 1.0.
    assert record.meta.mean_f1 == pytest.approx(1.0)
    assert load_run(run_id, db_path=db).meta.mean_f1 == pytest.approx(1.0)


def test_rescore_replaces_rather_than_appends_scores(db: Path):
    tasks = [_task("t1")]
    run_id = save_run(_summary([_result("t1")]), tasks, db_path=db)

    rescore_run(run_id, tasks, [FieldScorer()], db_path=db)
    rescore_run(run_id, tasks, [FieldScorer()], db_path=db)

    scores = load_run(run_id, db_path=db).results[0].scores
    assert [s.scorer for s in scores] == ["fields"]


def test_rescore_makes_no_network_calls(db: Path, monkeypatch: pytest.MonkeyPatch):
    # The point of --rescore is that it is free. Any adapter construction
    # would mean a provider SDK got involved.
    import harness.adapters as adapters

    def explode(*args, **kwargs):
        raise AssertionError("rescore must not construct an adapter")

    monkeypatch.setattr(adapters, "get_adapter", explode)

    tasks = [_task("t1")]
    run_id = save_run(_summary([_result("t1")]), tasks, db_path=db)

    rescore_run(run_id, tasks, [FieldScorer()], db_path=db)  # must not raise


def test_rescore_leaves_errored_results_alone(db: Path):
    tasks = [_task("t1"), _task("t2")]
    summary = _summary([_result("t1"), _result("t2", error="FatalError: x")])
    run_id = save_run(summary, tasks, db_path=db)

    record = rescore_run(run_id, tasks, [FieldScorer()], db_path=db)

    errored = next(r for r in record.results if r.task_id == "t2")
    assert errored.scores == []
    assert errored.error == "FatalError: x"


# --- schema integrity -------------------------------------------------------


def test_deleting_a_run_cascades_to_results_and_scores(db: Path):
    run_id = save_run(_summary([_result("t1")]), [_task("t1")], db_path=db)

    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
