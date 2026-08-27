"""Together adapter: a third token vocabulary, normalised to the same shape."""

import asyncio
from types import SimpleNamespace

import pytest

from harness.adapters.base import FatalError, RateLimited, TransientError
from harness.adapters.together import _FINISH_REASONS, TogetherAdapter
from harness.config import PRICING


class _Response:
    """Mimics the Together SDK's ChatCompletion shape."""

    def __init__(self, text="ok", finish="stop", prompt_tokens=100,
                 completion_tokens=20, model="deepseek-ai/DeepSeek-V3.1"):
        self.model = model
        self.choices = [
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=text),
            )
        ]
        # Chat-Completions naming: prompt_/completion_, not input_/output_.
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def model_dump(self, mode="json"):
        return {"model": self.model, "object": "chat.completion"}


def _adapter(monkeypatch, response=None, error=None) -> TogetherAdapter:
    monkeypatch.setattr(
        "harness.adapters.together.get_together_api_key", lambda: "test-key"
    )
    monkeypatch.setattr(
        "harness.adapters.together.together.AsyncTogether", lambda api_key: object()
    )
    adapter = TogetherAdapter(model_id="deepseek-ai/DeepSeek-V3.1")

    async def fake_create(**kwargs):
        if error is not None:
            raise error
        fake_create.kwargs = kwargs
        return response or _Response()

    adapter._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    adapter._create = fake_create
    return adapter


def test_prompt_and_completion_tokens_are_normalised(monkeypatch):
    # The whole job of this adapter: a third naming convention in, the same
    # ModelResponse out.
    adapter = _adapter(monkeypatch, _Response(prompt_tokens=250, completion_tokens=40))

    result = asyncio.run(adapter.complete("hi"))

    assert result.tokens_in == 250
    assert result.tokens_out == 40


def test_text_is_read_from_the_choice_message(monkeypatch):
    adapter = _adapter(monkeypatch, _Response(text='{"a": 1}'))

    assert asyncio.run(adapter.complete("hi")).text == '{"a": 1}'


def test_model_id_comes_from_the_response_not_the_request(monkeypatch):
    # Routes can resolve to a different served model; record what ran.
    adapter = _adapter(monkeypatch, _Response(model="deepseek-ai/DeepSeek-V3.1"))

    assert asyncio.run(adapter.complete("hi")).model_id == "deepseek-ai/DeepSeek-V3.1"


def test_cost_is_computed_from_the_response_token_counts(monkeypatch):
    adapter = _adapter(monkeypatch, _Response(prompt_tokens=1_000_000,
                                              completion_tokens=1_000_000))
    price = PRICING["deepseek-ai/DeepSeek-V3.1"]

    cost = asyncio.run(adapter.complete("hi")).cost_usd

    assert cost == pytest.approx(price["input_per_mtok"] + price["output_per_mtok"])


@pytest.mark.parametrize(
    "raw,expected",
    [("stop", "end_turn"), ("eos", "end_turn"), ("length", "max_tokens"),
     ("tool_calls", "tool_use"), ("content_filter", "refusal")],
)
def test_finish_reasons_map_onto_the_shared_vocabulary(monkeypatch, raw, expected):
    adapter = _adapter(monkeypatch, _Response(finish=raw))

    assert asyncio.run(adapter.complete("hi")).finish_reason == expected


def test_truncation_maps_to_the_value_the_cache_checks_for(monkeypatch):
    # cache.py refuses to store finish_reason == "max_tokens". That check
    # must work for this provider without knowing it exists.
    from harness.cache import _TRUNCATED_FINISH_REASONS

    assert _FINISH_REASONS["length"] in _TRUNCATED_FINISH_REASONS


def test_unknown_finish_reason_is_passed_through_not_guessed(monkeypatch):
    adapter = _adapter(monkeypatch, _Response(finish="something_new"))

    assert asyncio.run(adapter.complete("hi")).finish_reason == "something_new"


def test_raw_payload_is_a_dict(monkeypatch):
    adapter = _adapter(monkeypatch, _Response())

    assert isinstance(asyncio.run(adapter.complete("hi")).raw, dict)


def test_text_is_returned_verbatim_without_parsing(monkeypatch):
    fenced = '```json\n{"a": 1}\n```'
    adapter = _adapter(monkeypatch, _Response(text=fenced))

    assert asyncio.run(adapter.complete("hi")).text == fenced


def test_empty_choices_does_not_crash(monkeypatch):
    response = _Response()
    response.choices = []
    adapter = _adapter(monkeypatch, response)

    assert asyncio.run(adapter.complete("hi")).text == ""


# --- error taxonomy ----------------------------------------------------------


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


def test_rate_limit_maps_to_rate_limited(monkeypatch):
    import together

    err = together.RateLimitError.__new__(together.RateLimitError)
    Exception.__init__(err, "slow down")
    adapter = _adapter(monkeypatch, error=err)

    with pytest.raises(RateLimited):
        asyncio.run(adapter.complete("hi"))


def test_connection_error_maps_to_transient(monkeypatch):
    import together

    err = together.APIConnectionError.__new__(together.APIConnectionError)
    Exception.__init__(err, "network")
    adapter = _adapter(monkeypatch, error=err)

    with pytest.raises(TransientError):
        asyncio.run(adapter.complete("hi"))


def test_server_error_is_transient_and_client_error_is_fatal(monkeypatch):
    import together

    for code, expected in ((503, TransientError), (400, FatalError)):
        err = together.APIStatusError.__new__(together.APIStatusError)
        Exception.__init__(err, f"status {code}")
        err.status_code = code
        adapter = _adapter(monkeypatch, error=err)
        with pytest.raises(expected):
            asyncio.run(adapter.complete("hi"))


# --- the contract all three adapters share -----------------------------------


def test_all_three_adapters_satisfy_the_same_contract():
    # "Swappable" means exactly this: the same attributes and the same
    # coroutine signature, with nothing provider-shaped leaking out.
    import inspect

    from harness.adapters.anthropic import AnthropicAdapter
    from harness.adapters.openai import OpenAIAdapter

    for cls in (AnthropicAdapter, OpenAIAdapter, TogetherAdapter):
        assert isinstance(cls.name, str) and cls.name
        assert inspect.iscoroutinefunction(cls.complete)
        params = inspect.signature(cls.complete).parameters
        assert set(params) == {"self", "prompt", "max_tokens", "temperature"}
        assert params["max_tokens"].default == 2000
        assert params["temperature"].default == 0.0


def test_every_registered_adapter_has_pricing():
    from harness.adapters import _DEFAULT_MODEL_ALIAS
    from harness.config import MODELS

    for alias in _DEFAULT_MODEL_ALIAS.values():
        assert MODELS[alias] in PRICING, f"{alias} has no pricing entry"
