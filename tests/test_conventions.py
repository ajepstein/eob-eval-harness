"""What the schema settles, and — more importantly — what it does not.

A false "settled" deletes a real disagreement from the evidence and nobody
sees it again. The refusals below matter more than the acceptances.
"""

import pytest

from harness.conventions import settled_verdict


# --- settled ----------------------------------------------------------------


def test_modifier_only_difference_is_settled():
    assert settled_verdict(
        "cpt_codes", "['99214', '93000']", "['99214-25', '93000-26']"
    ) == "different"


def test_settled_on_real_lists_as_well_as_stored_text():
    assert settled_verdict(
        "cpt_codes", ["99214"], ["99214-25"]
    ) == "different"


def test_dependent_code_is_settled():
    assert settled_verdict("member_id", "EG-441002", "EG-441002-A") == "different"
    assert settled_verdict("member_id", "ga-88014", "ga-88014-01") == "different"


# --- refusals: the expensive mistakes ---------------------------------------


def test_genuinely_different_codes_are_not_settled():
    # Stripping modifiers does not reconcile these, so a person should see it.
    assert settled_verdict(
        "cpt_codes", "['99214', '93000']", "['99215-25', '93010-26']"
    ) is None


def test_a_reordered_list_is_not_settled():
    # SCHEMA.md makes order meaningful; that is a real disagreement.
    assert settled_verdict(
        "cpt_codes", "['99214', '93000']", "['93000', '99214']"
    ) is None


def test_an_extra_code_is_not_settled():
    assert settled_verdict(
        "cpt_codes", "['99214']", "['99214-25', '93000-26']"
    ) is None


def test_an_expected_value_carrying_a_modifier_is_not_settled():
    # The convention says the key holds bare codes. A key that does not is a
    # data problem, not something to quietly resolve against the model.
    assert settled_verdict(
        "cpt_codes", "['99214-25']", "['99214-25']"
    ) is None


def test_member_id_whitespace_is_not_settled():
    # Handled by norm_member_id, so it never reaches a label set. If one
    # does appear, it is not this predicate's call.
    assert settled_verdict("member_id", "PM 4471 2039", "PM44712039") is None


def test_an_unrelated_member_id_is_not_settled():
    assert settled_verdict("member_id", "EG-441002", "XX-999999") is None


def test_a_long_suffix_is_not_settled():
    # Dependent codes are short. A longer tail is something else.
    assert settled_verdict("member_id", "EG-441002", "EG-441002-SPOUSE") is None


def test_judgment_fields_are_never_settled():
    for field in ("patient_name", "payer_name", "date_of_service"):
        assert settled_verdict(field, "a", "b") is None


def test_unparseable_values_are_not_settled():
    assert settled_verdict("cpt_codes", "not a list", "['99214-25']") is None
    assert settled_verdict("cpt_codes", None, "['99214-25']") is None


def test_identical_values_are_not_settled():
    # Not a near-miss at all; nothing to decide.
    assert settled_verdict("cpt_codes", "['99214']", "['99214']") is None
    assert settled_verdict("member_id", "EG-441002", "EG-441002") is None
