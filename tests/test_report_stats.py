"""Report integration: intervals, frontier, and paired verdicts over the store."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from harness.report import (
    per_task_scores,
    render_frontier,
    render_mde,
    render_paired_comparison,
    render_run_table,
)
from harness.store import init_db, save_run
from harness.types import ModelResponse, RunSummary, Score, Task, TaskResult


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "s.db"
    init_db(path)
    return path


def _task(i: int, category: str = "clean") -> Task:
    return Task(id=f"t{i}", category=category, difficulty="easy",
                edge_case=False, input=f"doc {i}", expected={})


def _result(i: int, f1: float) -> TaskResult:
    return TaskResult(
        task_id=f"t{i}",
        response=ModelResponse(text="{}", model_id="m", tokens_in=10, tokens_out=5,
                               latency_ms=100.0, cost_usd=0.001,
                               finish_reason="end_turn", raw={}),
        scores=[Score(scorer="fields", value=f1, passed=f1 == 1.0, detail={"f1": f1})],
    )


def _save(db: Path, adapter: str, f1s: list[float], cost: float) -> str:
    tasks = [_task(i) for i in range(len(f1s))]
    summary = RunSummary(
        results=[_result(i, f) for i, f in enumerate(f1s)],
        adapter_name=adapter, model_id=f"{adapter}-model",
        prompt_name="extract_v1", prompt_hash="ph",
        total_cost_usd=cost, total_tokens_in=100, total_tokens_out=50,
        wall_clock_seconds=1.0, succeeded=len(f1s), failed=0, cached=0,
        latency_p50_ms=100.0, latency_p95_ms=120.0,
    )
    return save_run(summary, tasks, db_path=db)


def _capture(fn, *args, **kwargs) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=160, force_terminal=False)
    fn(*args, console=console, **kwargs)
    return buffer.getvalue()


def test_per_task_scores_is_one_value_per_task(db: Path):
    from harness.store import load_run

    run_id = _save(db, "anthropic", [1.0, 0.5, 0.75], 0.003)
    scores = per_task_scores(load_run(run_id, db))

    assert scores == {"t0": 1.0, "t1": 0.5, "t2": 0.75}


def test_run_table_shows_a_confidence_interval(db: Path):
    run_id = _save(db, "anthropic", [1.0, 0.8, 0.9, 0.7] * 5, 0.02)

    text = _capture(render_run_table, [run_id], db_path=db)

    assert "95% CI" in text
    assert "[" in text and "]" in text


def test_frontier_flags_a_dominated_run(db: Path):
    cheap = _save(db, "openai", [0.9] * 10, cost=0.001)      # same quality
    dear = _save(db, "anthropic", [0.9] * 10, cost=0.010)    # costlier

    text = _capture(render_frontier, [cheap, dear], db_path=db)

    assert "optimal" in text
    assert "dominated by" in text


def test_frontier_keeps_both_ends_of_a_real_tradeoff(db: Path):
    cheap = _save(db, "openai", [0.70] * 10, cost=0.001)
    dear = _save(db, "anthropic", [0.95] * 10, cost=0.010)

    text = _capture(render_frontier, [cheap, dear], db_path=db)

    assert text.count("optimal") >= 2
    assert "dominated by" not in text


def test_identical_runs_are_reported_as_indistinguishable(db: Path):
    # The property Week 3A exists for: comparing equivalent runs must not
    # imply a winner.
    scores = [1.0, 0.8, 0.9, 0.7, 0.85] * 4
    a = _save(db, "anthropic", scores, 0.02)
    b = _save(db, "openai", list(scores), 0.02)

    text = _capture(render_paired_comparison, a, b, db_path=db)

    assert "not distinguishable from zero" in text


def test_a_real_difference_is_named(db: Path):
    a = _save(db, "anthropic", [0.5] * 30, 0.02)
    b = _save(db, "openai", [0.95] * 30, 0.02)

    text = _capture(render_paired_comparison, a, b, db_path=db)

    assert "scores higher" in text
    assert "not distinguishable" not in text


def test_comparison_with_no_shared_tasks_says_so(db: Path):
    a = _save(db, "anthropic", [1.0], 0.001)
    # A run whose task ids do not overlap.
    tasks = [Task(id="zzz", category="clean", difficulty="easy",
                  edge_case=False, input="d", expected={})]
    summary = RunSummary(
        results=[TaskResult(task_id="zzz", response=None, scores=[], error="x")],
        adapter_name="openai", model_id="m", prompt_name="p", prompt_hash="h",
        total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0,
        wall_clock_seconds=0.0, succeeded=0, failed=1, cached=0,
        latency_p50_ms=0.0, latency_p95_ms=0.0,
    )
    b = save_run(summary, tasks, db_path=db)

    text = _capture(render_paired_comparison, a, b, db_path=db)

    assert "No tasks in common" in text


def test_mde_marks_the_suite_baseline(db: Path):
    _save(db, "anthropic", [0.9] * 20, 0.02)

    text = _capture(render_mde, [m for m in [_latest(db)]], db_path=db)

    assert "this suite" in text
    assert "n=20" in text


def _latest(db: Path) -> str:
    from harness.store import list_runs

    return list_runs(limit=1, db_path=db)[0].run_id
