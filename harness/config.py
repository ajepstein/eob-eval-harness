"""Environment configuration, model registry, and pricing table."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Return ANTHROPIC_API_KEY, raising a clear error if it isn't set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key, or export ANTHROPIC_API_KEY in your shell."
        )
    return api_key


# Short alias -> pinned full model version string.
MODELS: dict[str, str] = {
    "sonnet": "claude-sonnet-5",
}

# Checked 2026-06-24 against https://platform.claude.com/docs/en/about-claude/models/overview.md
# and https://platform.claude.com/docs/en/pricing.md
#
# Claude Sonnet 5 has an introductory price of $2.00 / $10.00 per MTok
# (input/output) through 2026-08-31; standard pricing of $3.00 / $15.00 per
# MTok applies after that date. The values below reflect the introductory
# price, which is in effect as of today (2026-07-31) — update after
# 2026-08-31.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},
}


def compute_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """USD cost of a request, given the model that actually ran and its token counts."""
    pricing = PRICING[model_id]
    return (
        tokens_in * pricing["input_per_mtok"] / 1_000_000
        + tokens_out * pricing["output_per_mtok"] / 1_000_000
    )
