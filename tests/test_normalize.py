from decimal import Decimal

import pytest

from harness.normalize import (
    FIELD_NORMALIZERS,
    norm_codes,
    norm_currency,
    norm_date,
    norm_npi,
    norm_string,
)


# --- currency ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (340.00, Decimal("340.00")),
        (340, Decimal("340")),
        ("340.00", Decimal("340.00")),
        ("$340.00", Decimal("340.00")),
        ("$1,250.00", Decimal("1250.00")),
        ("1,250.00", Decimal("1250.00")),
        ("USD 1,250.00", Decimal("1250.00")),
        ("(45.00)", Decimal("-45.00")),
        ("-45.00", Decimal("-45.00")),
        (Decimal("12.34"), Decimal("12.34")),
    ],
)
def test_norm_currency_formats(raw, expected):
    assert norm_currency(raw) == expected


def test_norm_currency_avoids_binary_float_error():
    # The whole reason this returns Decimal: 0.1 + 0.2 != 0.3 in float.
    assert norm_currency(0.1) + norm_currency(0.2) == norm_currency(0.3)


def test_norm_currency_garbage_returns_none():
    assert norm_currency("abc") is None
    assert norm_currency("$") is None


# --- dates ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "2026-03-14",
        "03/14/2026",
        "03-14-2026",
        "March 14, 2026",
        "March 14 2026",
        "14 March 2026",
        "Mar 14, 2026",
        "2026/03/14",
        "2026-03-14T00:00:00Z",
    ],
)
def test_norm_date_formats_all_reach_iso(raw):
    assert norm_date(raw) == "2026-03-14"


def test_norm_date_unparseable_returns_none():
    assert norm_date("sometime last spring") is None
    assert norm_date("13/45/2026") is None


# --- NPI --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["1234567890", "123-456-7890", "123 456 7890", " 1234567890 ", "NPI 1234567890"],
)
def test_norm_npi_strips_to_digits(raw):
    assert norm_npi(raw) == "1234567890"


@pytest.mark.parametrize("raw", ["12345", "12345678901", "abcdefghij"])
def test_norm_npi_wrong_length_returns_none(raw):
    assert norm_npi(raw) is None


# --- null equivalence -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "null", "NULL", "None", "N/A", "n/a", "not found", "Not Found", "-"],
)
def test_null_equivalents_normalize_to_none(raw):
    # Every normalizer must agree on what absence looks like, or the scorer
    # would count the same missing field differently per field type.
    assert norm_string(raw) is None
    assert norm_currency(raw) is None
    assert norm_date(raw) is None
    assert norm_npi(raw) is None


# --- strings ----------------------------------------------------------------


def test_norm_string_casefolds_and_collapses_whitespace():
    assert norm_string("  Jane   Doe  ") == "jane doe"
    assert norm_string("JANE DOE") == norm_string("jane doe")
    assert norm_string("Northstar\tHealth\nPlan") == "northstar health plan"


# --- codes ------------------------------------------------------------------


def test_norm_codes_preserves_order():
    assert norm_codes(["99213", "85025"]) == ["99213", "85025"]
    # Order is meaningful per SCHEMA.md, so the reverse must not compare equal.
    assert norm_codes(["85025", "99213"]) != norm_codes(["99213", "85025"])


def test_norm_codes_strips_each_element():
    assert norm_codes([" 99213 ", "85025\n"]) == ["99213", "85025"]


def test_norm_codes_empty_list_is_not_null():
    # SCHEMA.md: cpt_codes is non-nullable but may be empty. An empty list
    # is a real answer and must stay distinguishable from a missing field.
    assert norm_codes([]) == []
    assert norm_codes([]) is not None


def test_norm_codes_non_list_returns_none():
    # Splitting "99213, 85025" here would silently repair malformed output.
    assert norm_codes("99213, 85025") is None
    assert norm_codes(99213) is None


# --- idempotence ------------------------------------------------------------


@pytest.mark.parametrize(
    "func,raw",
    [
        (norm_string, "  Jane   Doe "),
        (norm_currency, "$1,250.00"),
        (norm_date, "03/14/2026"),
        (norm_npi, "123-456-7890"),
        (norm_codes, [" 99213 ", "85025"]),
    ],
)
def test_normalizing_twice_equals_normalizing_once(func, raw):
    once = func(raw)
    assert func(once) == once


# --- registry ---------------------------------------------------------------


def test_field_normalizers_cover_all_eight_schema_fields():
    assert set(FIELD_NORMALIZERS) == {
        "patient_name",
        "date_of_service",
        "provider_npi",
        "payer_name",
        "member_id",
        "cpt_codes",
        "billed_amount",
        "patient_responsibility",
    }


def test_normalizers_never_raise_on_hostile_input():
    hostile = [None, "", [], {}, 0, -1, True, False, "🙃", object()]
    for func in FIELD_NORMALIZERS.values():
        for value in hostile:
            func(value)  # must not raise
