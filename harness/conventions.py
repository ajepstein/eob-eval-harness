"""Verdicts that SCHEMA.md already determines, without a human.

Some near-misses are not judgment calls. Once SCHEMA.md says that
``cpt_codes`` holds bare procedure codes, ``['99214']`` against
``['99214-25']`` has a correct answer that follows from the document alone —
asking a person to label it adds no information the schema does not already
contain, and asking a judge to rule on it measures whether the rubric was
read, not whether the judge can judge.

Both effects showed up in the data. Structurally identical pairs drew
opposite verdicts twenty seconds apart, and a rubric stating the convention
would have scored every one of them correctly without exercising judgment.
So these items are separated out: verified directly against the convention,
and kept out of the population that calibration is computed over.

The predicate is deliberately narrow. It settles only the two patterns
SCHEMA.md names explicitly, and returns None for everything else —
including anything it cannot parse. A false "settled" silently removes a
real disagreement from the evidence, which is the more expensive mistake.
"""

from __future__ import annotations

import ast
import re

# A dependent code: a hyphen then a short alphanumeric run, at the end.
_DEPENDENT_CODE = re.compile(r"^-[A-Za-z0-9]{1,3}$")

# A CPT modifier: two characters after the code's hyphen.
_MODIFIER = re.compile(r"^([0-9]{5})-([A-Za-z0-9]{2})$")


def _as_list(value) -> list[str] | None:
    """Coerce a stored cpt_codes value back to a list, or give up."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return [str(v) for v in parsed] if isinstance(parsed, list) else None


def _strip_modifiers(codes: list[str]) -> list[str] | None:
    """Bare codes, or None if any element is not a code or code+modifier."""
    out = []
    for code in codes:
        text = str(code).strip()
        if re.fullmatch(r"[0-9]{5}", text):
            out.append(text)
            continue
        match = _MODIFIER.fullmatch(text)
        if not match:
            return None
        out.append(match.group(1))
    return out


def settled_verdict(field: str, expected, predicted) -> str | None:
    """The verdict SCHEMA.md determines, or None if judgment is required.

    Returns only ``"different"`` today: both settled patterns are cases
    where the model returned something the schema excludes. A convention
    that settled a pair as equivalent would belong here too.
    """
    if expected is None or predicted is None:
        return None

    if field == "cpt_codes":
        exp, pred = _as_list(expected), _as_list(predicted)
        if exp is None or pred is None or exp == pred:
            return None
        bare = _strip_modifiers(pred)
        # Settled only when stripping modifiers is exactly what reconciles
        # them. If they still differ, the disagreement is about something
        # else and a human should see it.
        if bare is not None and bare == _strip_modifiers(exp) == exp:
            return "different"
        return None

    if field == "member_id":
        exp, pred = str(expected).strip(), str(predicted).strip()
        if exp == pred or not pred.startswith(exp):
            return None
        if _DEPENDENT_CODE.fullmatch(pred[len(exp):]):
            return "different"
        return None

    return None
