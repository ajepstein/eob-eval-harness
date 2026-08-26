import asyncio
import json
from pathlib import Path

import pytest

import harness.runner as runner_module
from harness.cache import ResponseCache
from harness.runner import run_tasks
from harness.scorers.fields import FieldScorer
from harness.scorers.schema import SchemaScorer
from harness.types import Task
from tests.fakes import FakeAdapter

_PERFECT = {
    "patient_name": "Jane Doe",
    "date_of_service": "2026-03-14",
    "provider_npi": None,
    "payer_name": "Northstar Health",
    "member_id": "NS-88213",
    "cpt_codes": ["99213"],
    "billed_amount": 340.00,
    "patient_responsibility": 45.00,
}


@pytest.fixture(autouse=True)
def _zero_wait(monkeypatch: pytest.MonkeyPatch):
    # No real backoff in tests — keeps the suite fast regardless of retry count.
    monkeypatch.setattr(runner_module, "_wait_strategy", lambda retry_state: 0.0)


def _task(task_id: str, input_text: str | None = None) -> Task:
    return Task(
        id=task_id,
        category="clean",
        difficulty="easy",
        edge_case=False,
        input=input_text if input_text is not None else f"document for {task_id}",
        expected={},
    )


def _scored_task(task_id: str, category: str = "clean") -> Task:
    """A task whose expected block matches _PERFECT, for scoring tests."""
    return Task(
        id=task_id,
        category=category,
        difficulty="easy",
        edge_case=False,
        input=f"document for {task_id}",
        expected=dict(_PERFECT),
    )


def test_results_returned_in_input_order_regardless_of_completion_order():
    # t1 sleeps longest, so it finishes last even though it's first in input order.
    delays = {"marker-t1": 0.03, "marker-t2": 0.0, "marker-t3": 0.01}

    class VariableDelayAdapter(FakeAdapter):
        async def complete(self, prompt, *, max_tokens=2000, temperature=0.0):
            for marker, delay in delays.items():
                if marker in prompt:
                    self.sleep_seconds = delay
                    break
            return await super().complete(prompt, max_tokens=max_tokens, temperature=temperature)

    fake = VariableDelayAdapter()
    tasks = [_task("t1", "marker-t1"), _task("t2", "marker-t2"), _task("t3", "marker-t3")]

    summary = asyncio.run(run_tasks(tasks, fake, concurrency=5, cache=None))

    assert [r.task_id for r in summary.results] == ["t1", "t2", "t3"]


def test_concurrency_cap_is_respected():
    tasks = [_task(f"t{i}") for i in range(8)]
    fake = FakeAdapter(sleep_seconds=0.01)

    summary = asyncio.run(run_tasks(tasks, fake, concurrency=3, cache=None))

    assert fake.max_in_flight <= 3
    assert summary.succeeded == 8


def test_rate_limited_is_retried_and_eventually_succeeds():
    tasks = [_task("t1")]
    fake = FakeAdapter(rate_limited_times=2)

    summary = asyncio.run(run_tasks(tasks, fake, concurrency=1, cache=None))

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.results[0].error is None
    assert fake.call_count == 3


def test_fatal_error_is_not_retried_and_run_continues():
    tasks = [_task("t1", "break-me"), _task("t2", "ok document")]
    fake = FakeAdapter(fatal_if_prompt_contains="break-me")

    summary = asyncio.run(run_tasks(tasks, fake, concurrency=2, cache=None))

    assert summary.failed == 1
    assert summary.succeeded == 1
    t1_result = next(r for r in summary.results if r.task_id == "t1")
    t2_result = next(r for r in summary.results if r.task_id == "t2")
    assert t1_result.error is not None
    assert "FatalError" in t1_result.error
    assert t1_result.response is None
    assert t2_result.error is None
    assert t2_result.response is not None
    # A fatal error must never be retried.
    assert fake.call_count == 2


def test_cache_hit_produces_no_adapter_call_and_sets_cached(tmp_path: Path):
    tasks = [_task("t1")]
    fake = FakeAdapter()
    cache = ResponseCache(cache_dir=tmp_path)

    first = asyncio.run(run_tasks(tasks, fake, concurrency=1, cache=cache))
    assert first.results[0].cached is False
    assert fake.call_count == 1

    second = asyncio.run(run_tasks(tasks, fake, concurrency=1, cache=cache))
    assert second.results[0].cached is True
    assert fake.call_count == 1  # no new adapter call on the cache hit


def test_cost_and_token_totals_aggregate_correctly():
    tasks = [_task("t1"), _task("t2"), _task("t3")]
    fake = FakeAdapter(cost_usd=0.01)

    summary = asyncio.run(run_tasks(tasks, fake, concurrency=3, cache=None))

    assert summary.total_cost_usd == pytest.approx(0.03)
    assert summary.total_tokens_in == 30
    assert summary.total_tokens_out == 15


def test_scorers_populate_task_result_scores():
    payload = json.dumps(_PERFECT)
    fake = FakeAdapter(text=payload)

    summary = asyncio.run(
        run_tasks(
            [_scored_task("t1")],
            fake,
            concurrency=1,
            cache=None,
            scorers=[SchemaScorer(), FieldScorer()],
        )
    )

    scores = summary.results[0].scores
    assert [s.scorer for s in scores] == ["schema", "fields"]
    assert summary.schema_pass_rate == pytest.approx(1.0)
    assert summary.mean_f1 == pytest.approx(1.0)


def test_cache_hit_still_recomputes_scores(tmp_path: Path):
    # The payoff of scoring outside the adapter: a scorer change takes
    # effect on replay without re-paying for generations.
    payload = json.dumps(_PERFECT)
    fake = FakeAdapter(text=payload)
    cache = ResponseCache(cache_dir=tmp_path)
    tasks = [_scored_task("t1")]

    first = asyncio.run(
        run_tasks(tasks, fake, concurrency=1, cache=cache, scorers=[FieldScorer()])
    )
    second = asyncio.run(
        run_tasks(tasks, fake, concurrency=1, cache=cache, scorers=[FieldScorer()])
    )

    assert fake.call_count == 1  # no second network call
    assert second.results[0].cached is True
    assert second.results[0].scores  # but scores were still produced
    assert second.mean_f1 == pytest.approx(first.mean_f1)


def test_scores_break_down_by_category():
    perfect = json.dumps(_PERFECT)
    fake = FakeAdapter(text=perfect)
    tasks = [
        _scored_task("t1", category="clean"),
        _scored_task("t2", category="hard"),
    ]

    summary = asyncio.run(
        run_tasks(tasks, fake, concurrency=2, cache=None, scorers=[FieldScorer()])
    )

    assert set(summary.mean_f1_by_category) == {"clean", "hard"}
    assert summary.mean_f1_by_category["clean"] == pytest.approx(1.0)


def test_no_scorers_leaves_scores_empty_and_aggregates_zero():
    fake = FakeAdapter()

    summary = asyncio.run(run_tasks([_task("t1")], fake, concurrency=1, cache=None))

    assert summary.results[0].scores == []
    assert summary.schema_pass_rate == 0.0
    assert summary.mean_f1 == 0.0


def test_malformed_model_output_scores_zero_without_failing_the_task():
    fake = FakeAdapter(text="I was unable to extract that.")

    summary = asyncio.run(
        run_tasks(
            [_scored_task("t1")],
            fake,
            concurrency=1,
            cache=None,
            scorers=[SchemaScorer(), FieldScorer()],
        )
    )

    # The call succeeded; the *output* was unusable. Those are different
    # failures and the summary must not conflate them.
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.schema_pass_rate == 0.0
    assert summary.mean_f1 == 0.0


def test_latency_percentiles_exclude_cached_results(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path)
    fake = FakeAdapter()

    # Warm the cache for t1 only.
    asyncio.run(run_tasks([_task("t1")], fake, concurrency=1, cache=cache))

    tasks = [_task("t1"), _task("t2"), _task("t3")]
    summary = asyncio.run(run_tasks(tasks, fake, concurrency=3, cache=cache))

    non_cached = [r for r in summary.results if not r.cached]
    assert len(non_cached) == 2
    # All fake responses report latency_ms=1.0, so both percentiles land there.
    assert summary.latency_p50_ms == pytest.approx(1.0)
    assert summary.latency_p95_ms == pytest.approx(1.0)
