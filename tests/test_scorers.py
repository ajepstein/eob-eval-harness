import json

import pytest

from harness.scorers.base import score_result
from harness.scorers.fields import FieldScorer, classify, prf1
from harness.scorers.schema import SchemaScorer
from harness.types import ModelResponse, Score, Task

# 7 non-null fields, 1 genuinely-absent nullable field (provider_npi).
EXPECTED = {
    "patient_name": "Jane Doe",
    "date_of_service": "2026-03-14",
    "provider_npi": None,
    "payer_name": "Northstar Health",
    "member_id": "NS-88213",
    "cpt_codes": ["99213", "85025"],
    "billed_amount": 340.00,
    "patient_responsibility": 45.00,
}


def _task(expected: dict | None = None) -> Task:
    return Task(
        id="t1",
        category="clean",
        difficulty="easy",
        edge_case=False,
        input="<document>",
        expected=dict(EXPECTED if expected is None else expected),
    )


def _response(payload: dict | str) -> ModelResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelResponse(
        text=text,
        model_id="fake-1",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1.0,
        cost_usd=0.0,
        finish_reason="end_turn",
        raw={},
    )


def _prediction(**overrides) -> dict:
    """A perfect prediction, with named fields replaced."""
    payload = dict(EXPECTED)
    payload.update(overrides)
    return payload


# --- field scorer: the counting rules ---------------------------------------


def test_perfect_prediction_scores_one():
    # TP=7, FP=0, FN=0 (provider_npi is TN, excluded).
    # P = 7/7 = 1.0, R = 7/7 = 1.0, F1 = 1.0
    score = FieldScorer().score(_task(), _response(_prediction()))

    assert score.value == pytest.approx(1.0)
    assert score.passed is True
    assert score.detail["fields"]["provider_npi"] == "tn"


def test_one_wrong_field_counts_as_both_fp_and_fn():
    # patient_name wrong: TP=6, FP=1, FN=1.
    # P = 6/(6+1) = 6/7, R = 6/(6+1) = 6/7
    # F1 = 2 * (6/7)(6/7) / (6/7 + 6/7) = 6/7 = 0.857142...
    score = FieldScorer().score(
        _task(), _response(_prediction(patient_name="John Smith"))
    )

    assert score.value == pytest.approx(6 / 7)
    assert score.detail["fields"]["patient_name"] == "fp_fn_wrong"
    assert score.detail["tp"] == 6
    assert score.detail["fp"] == 1
    assert score.detail["fn"] == 1


def test_hallucinated_value_where_expected_is_null_is_penalized():
    # provider_npi invented: TP=7, FP=1, FN=0.
    # P = 7/(7+1) = 0.875, R = 7/(7+0) = 1.0
    # F1 = 2 * 0.875 * 1.0 / 1.875 = 14/15 = 0.93333...
    score = FieldScorer().score(
        _task(), _response(_prediction(provider_npi="1234567890"))
    )

    assert score.value == pytest.approx(14 / 15)
    assert score.detail["fields"]["provider_npi"] == "fp_hallucinated"
    assert score.passed is False


def test_missed_field_counts_as_false_negative_only():
    # member_id dropped: TP=6, FP=0, FN=1.
    # P = 6/6 = 1.0, R = 6/7
    # F1 = 2 * 1.0 * (6/7) / (1 + 6/7) = (12/7)/(13/7) = 12/13 = 0.923076...
    score = FieldScorer().score(_task(), _response(_prediction(member_id=None)))

    assert score.value == pytest.approx(12 / 13)
    assert score.detail["fields"]["member_id"] == "fn_missed"


def test_correctly_null_field_does_not_inflate_score():
    # Everything missed except the correctly-null provider_npi.
    # TP=0, FP=0, FN=7 -> P = 0.0, R = 0/7 = 0.0, F1 = 0.0
    # The one true negative must not rescue this.
    prediction = {key: None for key in EXPECTED}
    prediction["cpt_codes"] = []  # empty list, not null, so schema type holds

    score = FieldScorer().score(_task(), _response(prediction))

    assert score.value == pytest.approx(0.0)
    assert score.detail["fields"]["provider_npi"] == "tn"


def test_normalization_differences_do_not_count_as_errors():
    # Same answers, all differently formatted.
    score = FieldScorer().score(
        _task(),
        _response(
            _prediction(
                patient_name="  JANE   DOE ",
                date_of_service="03/14/2026",
                payer_name="northstar health",
                billed_amount="$340.00",
                patient_responsibility="45",
            )
        ),
    )

    assert score.value == pytest.approx(1.0)


def test_cpt_code_order_matters():
    # SCHEMA.md requires document order, so a reordered list is wrong.
    score = FieldScorer().score(
        _task(), _response(_prediction(cpt_codes=["85025", "99213"]))
    )

    assert score.detail["fields"]["cpt_codes"] == "fp_fn_wrong"


def test_null_spellings_are_treated_as_absence_not_as_wrong_answers():
    # "N/A" for a genuinely-absent field is a correct answer, not a
    # hallucination — it just isn't JSON null.
    score = FieldScorer().score(_task(), _response(_prediction(provider_npi="N/A")))

    assert score.detail["fields"]["provider_npi"] == "tn"
    assert score.value == pytest.approx(1.0)


def test_malformed_json_scores_zero_without_raising():
    score = FieldScorer().score(_task(), _response("I could not extract that."))

    assert score.value == 0.0
    assert score.passed is False
    assert score.detail["extraction_failed"] is True


def test_empty_cpt_list_is_a_real_answer_not_a_null():
    task = _task({**EXPECTED, "cpt_codes": []})
    score = FieldScorer().score(task, _response(_prediction(cpt_codes=[])))

    # [] is falsy; a truthiness check here would misclassify it as null.
    assert score.detail["fields"]["cpt_codes"] == "tp"


def test_zero_amount_is_a_real_answer_not_a_null():
    task = _task({**EXPECTED, "patient_responsibility": 0.00})
    score = FieldScorer().score(task, _response(_prediction(patient_responsibility=0)))

    assert score.detail["fields"]["patient_responsibility"] == "tp"


# --- counting primitives ----------------------------------------------------


@pytest.mark.parametrize(
    "expected,predicted,outcome",
    [
        ("a", "a", "tp"),
        ("a", "b", "fp_fn_wrong"),
        ("a", None, "fn_missed"),
        (None, None, "tn"),
        (None, "a", "fp_hallucinated"),
    ],
)
def test_classify_matches_the_documented_rules(expected, predicted, outcome):
    assert classify(expected, predicted) == outcome


def test_prf1_on_hand_computed_counts():
    # tp=6, fp=1, fn=1 -> P = 6/7, R = 6/7, F1 = 6/7
    precision, recall, f1 = prf1(6, 1, 1)
    assert (precision, recall, f1) == pytest.approx((6 / 7, 6 / 7, 6 / 7))


def test_prf1_with_nothing_to_score_is_vacuously_perfect():
    assert prf1(0, 0, 0) == (1.0, 1.0, 1.0)


# --- schema scorer ----------------------------------------------------------


def test_schema_passes_on_well_formed_output():
    score = SchemaScorer().score(_task(), _response(_prediction()))

    assert score.passed is True
    assert score.value == 1.0
    assert score.detail["extraction_method"] == "direct"


def test_schema_records_the_extraction_method():
    score = SchemaScorer().score(
        _task(), _response("Here you go:\n" + json.dumps(_prediction()))
    )

    assert score.passed is True
    assert score.detail["extraction_method"] == "braces"


def test_schema_fails_on_missing_key():
    payload = _prediction()
    del payload["member_id"]

    score = SchemaScorer().score(_task(), _response(payload))

    assert score.passed is False
    assert score.detail["missing_keys"] == ["member_id"]


def test_schema_fails_on_extra_key():
    score = SchemaScorer().score(
        _task(), _response(_prediction(confidence="high"))
    )

    assert score.passed is False
    assert score.detail["extra_keys"] == ["confidence"]


def test_schema_fails_on_stringified_amount():
    # The prompt asks for bare numbers; "$340.00" ignores the output contract.
    score = SchemaScorer().score(
        _task(), _response(_prediction(billed_amount="$340.00"))
    )

    assert score.passed is False
    assert "billed_amount" in score.detail["reason"]


def test_schema_fails_when_cpt_codes_is_not_a_list():
    score = SchemaScorer().score(
        _task(), _response(_prediction(cpt_codes="99213, 85025"))
    )

    assert score.passed is False
    assert "cpt_codes" in score.detail["reason"]


def test_schema_allows_null_for_nullable_scalar():
    score = SchemaScorer().score(_task(), _response(_prediction(provider_npi=None)))

    assert score.passed is True


def test_schema_fails_on_unparseable_text():
    score = SchemaScorer().score(_task(), _response("no json here"))

    assert score.passed is False
    assert score.detail["reason"] == "json_extraction_failed"


def test_schema_rejects_boolean_as_amount():
    # bool is a subclass of int in Python; True must not pass as a number.
    score = SchemaScorer().score(_task(), _response(_prediction(billed_amount=True)))

    assert score.passed is False


# --- score_result harness ---------------------------------------------------


def test_score_result_runs_every_scorer():
    scores = score_result(
        _task(), _response(_prediction()), [SchemaScorer(), FieldScorer()]
    )

    assert [s.scorer for s in scores] == ["schema", "fields"]


def test_a_scorer_that_throws_is_captured_not_propagated():
    class ExplodingScorer:
        name = "exploding"

        def score(self, task, response):
            raise RuntimeError("boom")

    scores = score_result(
        _task(), _response(_prediction()), [ExplodingScorer(), FieldScorer()]
    )

    assert scores[0].scorer == "exploding"
    assert scores[0].passed is False
    assert "RuntimeError: boom" in scores[0].detail["scorer_error"]
    # The run continues: the healthy scorer still produced a real score.
    assert scores[1].value == pytest.approx(1.0)


def test_score_result_returns_score_objects():
    scores = score_result(_task(), _response(_prediction()), [FieldScorer()])

    assert all(isinstance(s, Score) for s in scores)
