"""Loading of prompt templates from the prompts/ directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (e.g. 'extract_v1') from prompts/<name>.txt."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text()


def prompt_hash(name: str) -> str:
    """First 12 hex chars of the sha256 of the prompt template's exact contents."""
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()[:12]
