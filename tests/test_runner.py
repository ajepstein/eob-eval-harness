import asyncio
from pathlib import Path

import pytest

import harness.runner as runner_module
from harness.cache import ResponseCache
from harness.runner import run_tasks
from harness.types import Task
from tests.fakes import FakeAdapter


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
