"""Environment configuration, model registry, and pricing table."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"{var_name} is not set. Copy .env.example to .env and fill "
            f"in your key, or export {var_name} in your shell."
        )
    return value


def get_api_key() -> str:
    """Return ANTHROPIC_API_KEY, raising a clear error if it isn't set."""
    return _require_env("ANTHROPIC_API_KEY")


def get_openai_api_key() -> str:
    """Return OPENAI_API_KEY, raising a clear error if it isn't set."""
    return _require_env("OPENAI_API_KEY")


def get_together_api_key() -> str:
    """Return TOGETHER_API_KEY, raising a clear error if it isn't set."""
    return _require_env("TOGETHER_API_KEY")


# Short alias -> pinned full model version string.
MODELS: dict[str, str] = {
    "sonnet": "claude-sonnet-5",
    "gpt": "gpt-5.6-terra",
    # Open-weights, served by Together. Deliberately a different class from
    # the two hosted frontier models so the cost/quality frontier has a
    # genuinely different price point on it rather than three points in a
    # cluster.
    "oss": "deepseek-ai/DeepSeek-V3.1",
}

# Checked 2026-06-24 against https://platform.claude.com/docs/en/about-claude/models/overview.md
# and https://platform.claude.com/docs/en/pricing.md
#
# Claude Sonnet 5 has an introductory price of $2.00 / $10.00 per MTok
# (input/output) through 2026-08-31; standard pricing of $3.00 / $15.00 per
# MTok applies after that date. The values below reflect the introductory
# price, which is in effect as of today (2026-07-31) — update after
# 2026-08-31.
#
# Checked 2026-08-26 against https://developers.openai.com/api/docs/pricing
# and https://developers.openai.com/api/docs/models
#
# GPT-5.6 Terra ("balanced" tier, comparable in role to Sonnet) short-context
# standard pricing: $2.00 / $12.00 per MTok (input/output). OpenAI also
# publishes a higher "long context" rate above a request-size threshold and
# batch/flex discounts — not modeled here, only the standard short-context
# rate used by this harness's requests.
#
# Checked 2026-08-27 against https://www.together.ai/pricing
#
# DeepSeek V3.1 is Together's current flagship open-weights route at
# $0.60 / $1.70 per MTok — roughly a third the input price and a sixth the
# output price of the hosted frontier models above, which is the point of
# including it.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},
    "gpt-5.6-terra": {"input_per_mtok": 2.00, "output_per_mtok": 12.00},
    "deepseek-ai/DeepSeek-V3.1": {"input_per_mtok": 0.60, "output_per_mtok": 1.70},
}


def compute_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """USD cost of a request, given the model that actually ran and its token counts."""
    pricing = PRICING[model_id]
    return (
        tokens_in * pricing["input_per_mtok"] / 1_000_000
        + tokens_out * pricing["output_per_mtok"] / 1_000_000
    )
