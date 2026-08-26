"""Normalization rules for comparing extracted values.

The single source of truth for what "equal" means in this harness. Day 4's
deterministic scorer and Week 2's LLM judge must agree on these rules, so
every normalization decision lives here and nowhere else.

Two invariants hold for every function in this module:

1. **Total.** Each returns a normalized value or ``None``. None of them
   raise, whatever garbage a model produces.
2. **Idempotent.** ``f(f(x)) == f(x)``. The scorer normalizes both the
   expected and the predicted value, and expected values are already
   partly normalized in the task YAML.

Absence is normalized, not penalized. Models express "this field isn't in
the document" inconsistently — ``null``, ``"N/A"``, ``""``, ``"not found"``
— and scoring those as wrong would measure formatting compliance rather
than extraction quality. ``_NULL_EQUIVALENTS`` holds the full set.

Currency uses ``Decimal``, never ``float``: binary floating point makes
``0.1 + 0.2 != 0.3``, and a cent of drift in a comparison is a silent
scoring error.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

# The 8 EOB fields, in SCHEMA.md order. This is the canonical list: the task
# loader validates expected blocks against it and the schema scorer checks
# response keys against it, so adding a field is a one-line change here
# rather than a hunt through three modules.
SCHEMA_FIELDS: tuple[str, ...] = (
    "patient_name",
    "date_of_service",
    "provider_npi",
    "payer_name",
    "member_id",
    "cpt_codes",
    "billed_amount",
    "patient_responsibility",
)

# Fields SCHEMA.md marks nullable — the document may genuinely not contain
# them. The other six are always present in a well-formed EOB, so a null
# there is a miss rather than a correct abstention.
NULLABLE_FIELDS: frozenset[str] = frozenset({"provider_npi", "member_id"})

# Spellings that all mean "this field is absent from the document".
# Compared against the casefolded, whitespace-collapsed string form.
#
# This set is deliberately wider than the handful of spellings seen so far.
# Scoring "not provided" as a hallucination would measure a model's
# formatting compliance rather than whether it correctly declined to invent
# a value, and that is the distinction the missing_field tasks exist to
# test. The cost of the wider set is that a field whose genuine value is
# literally one of these strings would read as absent — no real EOB field
# (a name, date, NPI, payer, member id, or amount) can take these values,
# so the trade is one-sided here. `test_null_equivalents_are_pinned` fixes
# the exact set so any future widening is a deliberate, reviewed change.
_NULL_EQUIVALENTS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "not found",
    "not provided",
    "not specified",
    "not available",
    "unknown",
    "-",
    "--",
}

_WHITESPACE = re.compile(r"\s+")

# Tried in order. ISO first because that's what the prompt asks for.
#
# NOTE ON AMBIGUITY: "03/04/2026" is March 4th in US convention and April
# 3rd in European convention, and nothing in the string itself resolves it.
# These are US health-insurance documents, so slash-separated dates are read
# US-style (month first). A European-ordered source document would need an
# unambiguous day (>12) to parse correctly here — that limitation is real
# and is why %d/%m/%Y sits after %m/%d/%Y rather than being absent.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%B %d %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %b %Y",
    "%Y/%m/%d",
)


def _is_null(value: Any) -> bool:
    """True when `value` means absence under any of its accepted spellings."""
    if value is None:
        return True
    if isinstance(value, str):
        return _WHITESPACE.sub(" ", value).strip().casefold() in _NULL_EQUIVALENTS
    return False


def norm_string(v: Any) -> str | None:
    """Casefold and collapse internal whitespace, per SCHEMA.md."""
    if _is_null(v):
        return None
    if isinstance(v, bool):
        return None
    text = str(v)
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if not collapsed:
        return None
    return collapsed.casefold()


def norm_currency(v: Any) -> Decimal | None:
    """Strip currency symbols and thousands separators, return a Decimal.

    Accepts a bare number (the prompt asks for this) or a formatted string
    like "$1,250.00". Parenthesized negatives — "(45.00)" — are accounting
    notation and are read as negative.
    """
    if _is_null(v):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, int):
        return Decimal(v)
    if isinstance(v, float):
        # str() first: Decimal(0.1) captures the binary representation error,
        # Decimal(str(0.1)) does not.
        return Decimal(str(v))

    text = str(v).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    cleaned = re.sub(r"[^\d.\-]", "", text)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def norm_date(v: Any) -> str | None:
    """Parse any recognized date format and return ISO ``YYYY-MM-DD``."""
    if _is_null(v):
        return None
    if isinstance(v, bool):
        return None

    text = _WHITESPACE.sub(" ", str(v)).strip()
    if not text:
        return None

    # Tolerate an ISO datetime ("2026-03-14T00:00:00Z") by taking the date.
    if "T" in text:
        text = text.split("T", 1)[0]

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def norm_npi(v: Any) -> str | None:
    """Digits only. Returns None unless exactly 10 digits remain.

    An NPI that isn't 10 digits isn't an NPI, so a malformed one normalizes
    to None rather than to a wrong-but-comparable string.
    """
    if _is_null(v):
        return None
    if isinstance(v, bool):
        return None
    digits = re.sub(r"\D", "", str(v))
    return digits if len(digits) == 10 else None


def norm_codes(v: Any) -> list[str] | None:
    """Normalize a CPT code list, preserving document order.

    Only a genuine list normalizes. A model that returns "99213, 85025" as
    a single string has not produced the requested type, and splitting it
    here would silently repair the output and inflate the score — the schema
    scorer reports that as a type error instead.
    """
    if v is None:
        return None
    if not isinstance(v, list):
        return None

    codes: list[str] = []
    for item in v:
        if _is_null(item):
            continue
        code = _WHITESPACE.sub("", str(item)).strip().casefold()
        if code:
            codes.append(code)
    return codes


FIELD_NORMALIZERS: dict[str, Callable[[Any], Any]] = {
    "patient_name": norm_string,
    "date_of_service": norm_date,
    "provider_npi": norm_npi,
    "payer_name": norm_string,
    "member_id": norm_string,
    "cpt_codes": norm_codes,
    "billed_amount": norm_currency,
    "patient_responsibility": norm_currency,
}
