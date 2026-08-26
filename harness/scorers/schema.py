"""Binary pass/fail on structural validity.

The cheapest signal in the harness and often the most diagnostic: a model
that cannot hold an 8-field schema is disqualified before accuracy is worth
measuring.

Passing requires all of:

1. JSON was recoverable from the response text
2. it is an object
3. all 8 schema keys are present
4. no keys beyond those 8
5. every value has a plausible type for its field

On (5) the rule differs between scalars and the code list, deliberately:

- **Scalar fields** accept their type *or null*. Null is how a model says
  "absent"; whether absence is *correct* is a content question, and
  ``fields.py`` scores it under the counting rules.
- **``cpt_codes`` must be a list.** Null there means the model failed to
  produce the container type at all, not that it judged a value missing.
  SCHEMA.md marks it non-nullable-but-possibly-empty, so ``[]`` is the way
  to say "no codes found".

Amounts must be JSON numbers. ``"$340.00"`` is a string, and the prompt
asks explicitly for bare numbers — accepting it here would let a model
ignore the output contract for free.
"""

from __future__ import annotations

from harness.extract import extract_json
from harness.normalize import FIELD_NORMALIZERS
from harness.types import ModelResponse, Score, Task

SCHEMA_FIELDS: frozenset[str] = frozenset(FIELD_NORMALIZERS)

_STRING_FIELDS = {
    "patient_name",
    "date_of_service",
    "provider_npi",
    "payer_name",
    "member_id",
}
_NUMERIC_FIELDS = {"billed_amount", "patient_responsibility"}


def _type_error(key: str, value: object) -> str | None:
    """Return a failure reason if `value` is an implausible type for `key`."""
    if key in _STRING_FIELDS:
        if value is None or isinstance(value, str):
            return None
        return f"{key} must be a string or null, got {type(value).__name__}"

    if key in _NUMERIC_FIELDS:
        # bool is a subclass of int in Python; True is not a valid amount.
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return None
        return f"{key} must be a number or null, got {type(value).__name__}"

    if key == "cpt_codes":
        if not isinstance(value, list):
            return f"cpt_codes must be a list, got {type(value).__name__}"
        for item in value:
            if not isinstance(item, str):
                return f"cpt_codes items must be strings, got {type(item).__name__}"
        return None

    return None


class SchemaScorer:
    name = "schema"

    def score(self, task: Task, response: ModelResponse) -> Score:
        parsed, method = extract_json(response.text)

        detail: dict[str, object] = {"extraction_method": method}

        if parsed is None:
            detail["reason"] = "json_extraction_failed"
            return Score(scorer=self.name, value=0.0, passed=False, detail=detail)

        present = set(parsed)
        missing = sorted(SCHEMA_FIELDS - present)
        extra = sorted(present - SCHEMA_FIELDS)

        if missing:
            detail["reason"] = f"missing keys: {', '.join(missing)}"
            detail["missing_keys"] = missing
            return Score(scorer=self.name, value=0.0, passed=False, detail=detail)

        if extra:
            detail["reason"] = f"unexpected keys: {', '.join(extra)}"
            detail["extra_keys"] = extra
            return Score(scorer=self.name, value=0.0, passed=False, detail=detail)

        for key in sorted(SCHEMA_FIELDS):
            reason = _type_error(key, parsed[key])
            if reason is not None:
                detail["reason"] = reason
                return Score(scorer=self.name, value=0.0, passed=False, detail=detail)

        detail["reason"] = "ok"
        return Score(scorer=self.name, value=1.0, passed=True, detail=detail)
