"""OpenAI API adapter.

OpenAI's Responses API names things differently from Anthropic's Messages
API — usage.input_tokens/output_tokens instead of tokens_in/tokens_out on
usage, a `status` field instead of `stop_reason`, and a nested
`incomplete_details.reason` instead of a single truncation value. This file
normalizes all of that onto the same vocabulary Anthropic's adapter uses
(finish_reason values like "end_turn" / "max_tokens" / "refusal") so nothing
upstream — the cache's truncation check included — can tell which provider
answered.
"""

from __future__ import annotations

import time

import openai

from harness.adapters.base import FatalError, RateLimited, TransientError, retry_after_seconds
from harness.config import compute_cost, get_openai_api_key
from harness.types import ModelResponse

# OpenAI's `status` -> our finish_reason vocabulary (matches Anthropic's
# stop_reason values so cache.py's truncation check works for either
# provider).
_STATUS_TO_FINISH_REASON = {
    "completed": "end_turn",
}
_INCOMPLETE_REASON_TO_FINISH_REASON = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}


def _finish_reason(response) -> str:
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        return _INCOMPLETE_REASON_TO_FINISH_REASON.get(reason, "incomplete")
    return _STATUS_TO_FINISH_REASON.get(response.status, response.status)


class OpenAIAdapter:
    name = "openai"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = openai.AsyncOpenAI(api_key=get_openai_api_key())

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse:
        # GPT-5.6 rejects temperature unless reasoning effort is "none". 0.0
        # is our harness default, not something the caller explicitly asked
        # for, so only forward it when it differs (same rule as the
        # Anthropic adapter).
        kwargs = {} if temperature == 0.0 else {"temperature": temperature}

        start = time.monotonic()
        try:
            response = await self._client.responses.create(
                model=self.model_id,
                input=prompt,
                max_output_tokens=max_tokens,
                **kwargs,
            )
        except openai.RateLimitError as exc:
            raise RateLimited(str(exc), retry_after=retry_after_seconds(exc)) from exc
        except openai.APIConnectionError as exc:
            raise TransientError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise FatalError(str(exc)) from exc
        latency_ms = (time.monotonic() - start) * 1000

        cost_usd = compute_cost(response.model, response.usage.input_tokens, response.usage.output_tokens)

        return ModelResponse(
            text=response.output_text,
            model_id=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            finish_reason=_finish_reason(response),
            raw=response.to_dict(mode="json"),
        )
