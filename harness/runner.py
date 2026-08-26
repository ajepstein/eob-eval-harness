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


async def run_tasks(
    tasks: list[Task],
    adapter: Adapter,
    prompt_name: str = "extract_v1",
    concurrency: int = 5,
    cache: ResponseCache | None = None,
    max_tokens: int = 2000,
) -> RunSummary:
    template = load_prompt(prompt_name)
    template_hash = prompt_hash(prompt_name)
    params = {"max_tokens": max_tokens}
    semaphore = asyncio.Semaphore(concurrency)

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
                return TaskResult(task_id=task.id, response=hit, scores=[], cached=True)

        # Only the network call is bounded by the semaphore, so the cap
        # actually limits in-flight requests rather than task creation.
        async with semaphore:
            response = await _complete_with_retries(adapter, prompt, max_tokens)

        if cache is not None:
            cache.set(key, response)
        running_cost += response.cost_usd
        progress.update(bar, advance=1, cost=running_cost)
        return TaskResult(task_id=task.id, response=response, scores=[], cached=False)

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
    )
