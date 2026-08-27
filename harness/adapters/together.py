"""Together adapter — open-weights models behind the same contract.

Included to put a genuinely different price point on the cost/quality
frontier. Two hosted frontier models clustered at similar prices cannot
answer "is the cheap option good enough", which is the question the frontier
exists for.

**A third token vocabulary.** Anthropic reports `usage.input_tokens` /
`output_tokens`; OpenAI's Responses API reports `input_tokens` /
`output_tokens` on a differently-shaped object; Together, following the
older Chat Completions convention, reports `usage.prompt_tokens` /
`completion_tokens` and puts the text at `choices[0].message.content` with
the stop reason at `choices[0].finish_reason`. Absorbing that is the entire
job of this file — nothing upstream should be able to tell which of the
three answered.
"""

from __future__ import annotations

import time

import together

from harness.adapters.base import (
    FatalError,
    RateLimited,
    TransientError,
    retry_after_seconds,
)
from harness.config import compute_cost, get_together_api_key
from harness.types import ModelResponse

# Chat-Completions stop reasons -> the vocabulary the rest of the harness
# uses (which follows Anthropic's, since that adapter came first). Mapping
# "length" onto "max_tokens" is what lets cache.py's truncation check work
# for this provider without knowing it exists.
_FINISH_REASONS = {
    "stop": "end_turn",
    "eos": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


class TogetherAdapter:
    name = "together"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = together.AsyncTogether(api_key=get_together_api_key())

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse:
        # Unlike the frontier models, open-weights routes do accept a
        # temperature, so the harness default is forwarded as given.
        start = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except together.RateLimitError as exc:
            raise RateLimited(str(exc), retry_after=retry_after_seconds(exc)) from exc
        except together.APIConnectionError as exc:
            raise TransientError(str(exc)) from exc
        except together.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise FatalError(str(exc)) from exc
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        raw_reason = getattr(choice, "finish_reason", None)
        # finish_reason may arrive as an enum on this SDK; normalise via its
        # value before mapping.
        raw_reason = getattr(raw_reason, "value", raw_reason)

        usage = response.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        return ModelResponse(
            text=text,
            model_id=response.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=compute_cost(response.model, tokens_in, tokens_out),
            finish_reason=_FINISH_REASONS.get(raw_reason, raw_reason or "unknown"),
            raw=response.model_dump(mode="json"),
        )
