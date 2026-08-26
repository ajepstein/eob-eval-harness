"""Pulling a JSON object out of raw model text.

The extraction prompt asks for bare JSON. Models do not always comply —
they add a preamble, wrap the object in markdown fences, or append an
explanation. This module recovers the object through a fixed ladder of
*structural* steps and reports which rung it landed on.

The method is a diagnostic worth keeping: a model that consistently needs
the brace-matching fallback is behaving differently from one that returns
clean JSON, even when both end up scoring the same.

**Structural recovery only.** Nothing here repairs malformed JSON — no
quote fixing, no trailing-comma stripping, no key quoting. Silently
correcting a model's output measures this module's cleverness rather than
the model's compliance, and inflates every score downstream.
"""

from __future__ import annotations

import json
import re

# ```json ... ``` or ``` ... ```, non-greedy so the first complete fence wins.
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

Method = str  # one of: "direct" | "fenced" | "braces" | "failed"


def _loads_dict(text: str) -> dict | None:
    """json.loads, but only a JSON *object* counts as success."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _outermost_object(text: str) -> str | None:
    """Return the substring from the first '{' to its matching '}'.

    Brace counting is string-aware: braces inside a JSON string value (and
    escaped quotes within it) must not affect the depth, or a value like
    "{not json}" would truncate the span at the wrong place.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json(text: str) -> tuple[dict | None, str]:
    """Recover a JSON object from model output.

    Returns ``(parsed, method)`` where method is one of ``"direct"``,
    ``"fenced"``, ``"braces"``, or ``"failed"``. On failure ``parsed`` is
    ``None``.

    Valid JSON that isn't an object (a bare list, string, or number) does
    not satisfy the schema and is reported as ``"failed"``.
    """
    if not text or not text.strip():
        return None, "failed"

    # 1. The whole string is the object.
    parsed = _loads_dict(text.strip())
    if parsed is not None:
        return parsed, "direct"

    # 2. Inside a markdown fence. Try each fence in order; a model may emit
    #    a prose fence before the JSON one.
    for fenced in _FENCE.findall(text):
        parsed = _loads_dict(fenced.strip())
        if parsed is not None:
            return parsed, "fenced"

    # 3. The outermost balanced {...} anywhere in the text.
    span = _outermost_object(text)
    if span is not None:
        parsed = _loads_dict(span)
        if parsed is not None:
            return parsed, "braces"

    return None, "failed"
