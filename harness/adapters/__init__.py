"""Registry so callers can select an adapter by provider name."""

from __future__ import annotations

from harness.adapters.base import Adapter
from harness.config import MODELS

# provider name -> default MODELS alias for that provider.
_DEFAULT_MODEL_ALIAS = {
    "anthropic": "sonnet",
    "openai": "gpt",
    "together": "oss",
}


def available_adapters() -> list[str]:
    return sorted(_DEFAULT_MODEL_ALIAS)


def get_adapter(name: str, model_alias: str | None = None) -> Adapter:
    """Construct an adapter by provider name.

    Concrete adapter modules (and the provider SDKs they import) are
    imported lazily inside this function, so selecting "anthropic" never
    requires an OpenAI key/install to be present, and vice versa.
    """
    if name not in _DEFAULT_MODEL_ALIAS:
        raise ValueError(
            f"Unknown adapter {name!r}. Valid adapters: {', '.join(available_adapters())}"
        )

    alias = model_alias or _DEFAULT_MODEL_ALIAS[name]
    model_id = MODELS[alias]

    if name == "anthropic":
        from harness.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter(model_id=model_id)

    if name == "openai":
        from harness.adapters.openai import OpenAIAdapter

        return OpenAIAdapter(model_id=model_id)

    if name == "together":
        from harness.adapters.together import TogetherAdapter

        return TogetherAdapter(model_id=model_id)

    raise AssertionError(f"unreachable: {name!r}")
