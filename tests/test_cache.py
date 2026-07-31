from pathlib import Path

from harness.cache import ResponseCache, make_cache_key
from harness.types import ModelResponse


def _response(finish_reason: str = "end_turn") -> ModelResponse:
    return ModelResponse(
        text="hello",
        model_id="claude-sonnet-5",
        tokens_in=10,
        tokens_out=5,
        latency_ms=123.4,
        cost_usd=0.0001,
        finish_reason=finish_reason,
        raw={"id": "msg_123", "nested": {"a": 1}},
    )


def test_cache_key_changes_when_prompt_template_changes():
    key_a = make_cache_key("prompt text", "claude-sonnet-5", {}, "hash_a")
    key_b = make_cache_key("prompt text", "claude-sonnet-5", {}, "hash_b")

    assert key_a != key_b


def test_cache_key_stable_across_dict_ordering():
    key_a = make_cache_key(
        "prompt text", "claude-sonnet-5", {"max_tokens": 2000, "temperature": 0.0}, "hash"
    )
    key_b = make_cache_key(
        "prompt text", "claude-sonnet-5", {"temperature": 0.0, "max_tokens": 2000}, "hash"
    )

    assert key_a == key_b


def test_get_returns_none_on_miss(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path)

    assert cache.get("nonexistent") is None


def test_set_then_get_round_trips_response(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path)
    response = _response()

    cache.set("key1", response)
    result = cache.get("key1")

    assert result == response


def test_enabled_false_disables_get_and_set(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path, enabled=False)
    response = _response()

    cache.set("key1", response)

    assert cache.get("key1") is None
    assert not (tmp_path / "key1.json").exists()


def test_truncated_finish_reason_not_cached(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path)
    response = _response(finish_reason="max_tokens")

    cache.set("key1", response)

    assert cache.get("key1") is None
    assert not (tmp_path / "key1.json").exists()
