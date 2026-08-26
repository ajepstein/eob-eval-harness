"""Concurrent task runner.

Imports only `adapters.base` — never a concrete adapter or a provider SDK.
The cache lives here rather than inside the adapters, so adapters stay
unaware that caching exists at all.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from harness.adapters.base import Adapter, RateLimited, TransientError
from harness.cache import ResponseCache, make_cache_key
from harness.prompts import load_prompt, prompt_hash
from harness.scorers.base import (
    Scorer,
    is_async_scorer,
    score_result,
    score_result_async,
)
from harness.types import ModelResponse, RunSummary, Task, TaskResult

_MAX_ATTEMPTS = 4
_base_wait = wait_exponential_jitter(initial=1, max=30, jitter=2)


def _wait_strategy(retry_state: RetryCallState) -> float:
    """Exponential backoff with jitter, but honor a provider's retry-after."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return float(retry_after)
    return _base_wait(retry_state)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[min(int(pct) - 1, len(quantiles) - 1)]


async def _complete_with_retries(
    adapter: Adapter, prompt: str, max_tokens: int
) -> ModelResponse:
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((RateLimited, TransientError)),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=lambda rs: _wait_strategy(rs),
        reraise=True,
    ):
        with attempt:
            return await adapter.complete(prompt, max_tokens=max_tokens)
    raise AssertionError("unreachable")


def _score_of(result: TaskResult, scorer_name: str) -> float | None:
    """The value of one named score on a result, or None if absent."""
    for score in result.scores:
        if score.scorer == scorer_name:
            return score.value
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _by_category(
    tasks: list[Task], results: list[TaskResult], scorer_name: str
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for task, result in zip(tasks, results):
        value = _score_of(result, scorer_name)
        if value is not None:
            buckets.setdefault(task.category, []).append(value)
    return {category: _mean(values) for category, values in sorted(buckets.items())}


async def run_tasks(
    tasks: list[Task],
    adapter: Adapter,
    prompt_name: str = "extract_v1",
    concurrency: int = 5,
    cache: ResponseCache | None = None,
    max_tokens: int = 2000,
    scorers: list[Scorer] | None = None,
) -> RunSummary:
    template = load_prompt(prompt_name)
    template_hash = prompt_hash(prompt_name)
    params = {"max_tokens": max_tokens}
    semaphore = asyncio.Semaphore(concurrency)
    all_scorers = list(scorers or [])
    sync_scorers = [s for s in all_scorers if not is_async_scorer(s)]
    async_scorers = [s for s in all_scorers if is_async_scorer(s)]

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("${task.fields[cost]:.6f}"),
        TimeElapsedColumn(),
    )
    bar = progress.add_task(f"{adapter.name}", total=len(tasks), cost=0.0)
    running_cost = 0.0

    async def run_one(task: Task) -> TaskResult:
        nonlocal running_cost
        prompt = template.replace("{document}", task.input)
        key = make_cache_key(prompt, adapter.model_id, params, template_hash)

        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                progress.update(bar, advance=1, cost=running_cost)
                # Scoring is recomputed on cache hits: it is local and free,
                # and a scorer fix must take effect without re-paying for
                # generations.
                scores = score_result(task, hit, sync_scorers)
                if async_scorers:
                    async with semaphore:
                        scores += await score_result_async(task, hit, async_scorers)
                return TaskResult(
                    task_id=task.id, response=hit, scores=scores, cached=True
                )

        # Only the network call is bounded by the semaphore, so the cap
        # actually limits in-flight requests rather than task creation.
        async with semaphore:
            response = await _complete_with_retries(adapter, prompt, max_tokens)

        if cache is not None:
            cache.set(key, response)
        running_cost += response.cost_usd
        progress.update(bar, advance=1, cost=running_cost)
        scores = score_result(task, response, sync_scorers)
        if async_scorers:
            # Judge calls share the extraction semaphore so the cap bounds
            # total in-flight network calls, not just extraction calls.
            async with semaphore:
                scores += await score_result_async(task, response, async_scorers)
        return TaskResult(
            task_id=task.id, response=response, scores=scores, cached=False
        )

    start = time.monotonic()
    with progress:
        gathered = await asyncio.gather(
            *(run_one(t) for t in tasks), return_exceptions=True
        )
    wall_clock_seconds = time.monotonic() - start

    # gather preserves input order, so results line up with `tasks`.
    results: list[TaskResult] = []
    for task, outcome in zip(tasks, gathered):
        if isinstance(outcome, BaseException):
            error = f"{type(outcome).__name__}: {outcome}"
            results.append(
                TaskResult(task_id=task.id, response=None, scores=[], error=error)
            )
        else:
            results.append(outcome)

    succeeded = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    cached_count = sum(1 for r in results if r.cached)
    total_cost = sum(r.response.cost_usd for r in results if r.response and not r.cached)
    total_in = sum(r.response.tokens_in for r in results if r.response)
    total_out = sum(r.response.tokens_out for r in results if r.response)
    latencies = [r.response.latency_ms for r in results if r.response and not r.cached]

    schema_values = [v for r in results if (v := _score_of(r, "schema")) is not None]
    f1_values = [v for r in results if (v := _score_of(r, "fields")) is not None]
    judge_values = [v for r in results if (v := _score_of(r, "judge")) is not None]

    judge_details = [
        s.detail for r in results for s in r.scores if s.scorer == "judge"
    ]
    judge_cost = sum(d.get("judge_cost_usd", 0.0) for d in judge_details)
    judge_calls = sum(d.get("judge_calls", 0) for d in judge_details)
    judge_hash = next(
        (d.get("judge_prompt_hash") for d in judge_details if d.get("judge_prompt_hash")),
        None,
    )

    return RunSummary(
        results=results,
        adapter_name=adapter.name,
        model_id=adapter.model_id,
        prompt_name=prompt_name,
        prompt_hash=template_hash,
        total_cost_usd=total_cost,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        wall_clock_seconds=wall_clock_seconds,
        succeeded=succeeded,
        failed=failed,
        cached=cached_count,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        schema_pass_rate=_mean(schema_values),
        mean_f1=_mean(f1_values),
        schema_pass_rate_by_category=_by_category(tasks, results, "schema"),
        mean_f1_by_category=_by_category(tasks, results, "fields"),
        judge_cost_usd=judge_cost,
        judge_calls=judge_calls,
        mean_judge_f1=_mean(judge_values),
        judge_prompt_hash=judge_hash,
    )
