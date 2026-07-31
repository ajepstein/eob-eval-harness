import asyncio

import pytest

from harness.adapters.anthropic import AnthropicAdapter
from harness.config import MODELS


@pytest.mark.live
def test_anthropic_adapter_returns_populated_response():
    adapter = AnthropicAdapter(model_id=MODELS["sonnet"])
    response = asyncio.run(adapter.complete("Say the single word: hello", max_tokens=20))

    assert response.text
    assert response.model_id
    assert response.tokens_in > 0
    assert response.tokens_out > 0
    assert response.cost_usd > 0
    assert response.finish_reason in {"end_turn", "max_tokens"}
    assert isinstance(response.raw, dict)
