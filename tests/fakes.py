"""A configurable in-memory Adapter for testing the runner without network."""

from __future__ import annotations

import asyncio

from harness.adapters.base import FatalError, RateLimited
from harness.types import ModelResponse


class FakeAdapter:
    name = "fake"

    def __init__(
        self,
        model_id: str = "fake-model-1",
        text: str = '{"ok": true}',
        finish_reason: str = "end_turn",
        sleep_seconds: float = 0.0,
        rate_limited_times: int = 0,
        retry_after: float | None = None,
        fatal: bool = False,
        fatal_if_prompt_contains: str | None = None,
        cost_usd: float = 0.0001,
    ):
        self.model_id = model_id
        self.text = text
        self.finish_reason = finish_reason
        self.sleep_seconds = sleep_seconds
        self.rate_limited_times = rate_limited_times
        self.retry_after = retry_after
        self.fatal = fatal
        self.fatal_if_prompt_contains = fatal_if_prompt_contains
        self.cost_usd = cost_usd

        self.call_count = 0
        self._attempts_by_prompt: dict[str, int] = {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.call_count += 1
        try:
            if self.sleep_seconds:
                await asyncio.sleep(self.sleep_seconds)

            if self.fatal or (
                self.fatal_if_prompt_contains and self.fatal_if_prompt_contains in prompt
            ):
                raise FatalError("fake fatal error")

            attempt = self._attempts_by_prompt.get(prompt, 0)
            self._attempts_by_prompt[prompt] = attempt + 1
            if attempt < self.rate_limited_times:
                raise RateLimited("fake rate limit", retry_after=self.retry_after)

            return ModelResponse(
                text=self.text,
                model_id=self.model_id,
                tokens_in=10,
                tokens_out=5,
                latency_ms=1.0,
                cost_usd=self.cost_usd,
                finish_reason=self.finish_reason,
                raw={},
            )
        finally:
            self.in_flight -= 1
