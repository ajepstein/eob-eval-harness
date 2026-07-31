"""Anthropic API adapter."""

from __future__ import annotations

import time

import anthropic

from harness.adapters.base import FatalError, RateLimited, TransientError
from harness.config import compute_cost, get_api_key
from harness.types import ModelResponse


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = anthropic.AsyncAnthropic(api_key=get_api_key())

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse:
        # Claude Sonnet 5 (and the 4.7/4.8/Fable 5 family) reject a non-default
        # temperature with a 400. 0.0 is our harness default, not the API's
        # own default, so only forward it when the caller asked for something
        # other than the harness default.
        kwargs = {} if temperature == 0.0 else {"temperature": temperature}

        start = time.monotonic()
        try:
            response = await self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except anthropic.RateLimitError as exc:
            raise RateLimited(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise TransientError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise FatalError(str(exc)) from exc
        latency_ms = (time.monotonic() - start) * 1000

        cost_usd = compute_cost(response.model, response.usage.input_tokens, response.usage.output_tokens)
        text = next((b.text for b in response.content if b.type == "text"), "")

        return ModelResponse(
            text=text,
            model_id=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            finish_reason=response.stop_reason,
            raw=response.model_dump(),
        )
