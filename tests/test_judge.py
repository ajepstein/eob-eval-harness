import asyncio
import json
from pathlib import Path

import pytest

from harness.adapters.base import FatalError
from harness.cache import ResponseCache
from harness.prompts import judge_prompt_hash
from harness.scorers.judge import JudgeScorer
from harness.types import ModelResponse, Task

EXPECTED = {
    "patient_name": "Jane Doe",
    "date_of_service": "2026-03-14",
    "provider_npi": None,
    "payer_name": "Northstar Health",
    "member_id": "NS-88213",
    "cpt_codes": ["99213"],
    "billed_amount": 340.00,
    "patient_responsibility": 45.00,
}


class VerdictAdapter:
    """Returns a canned judge verdict; records every prompt it was sent."""

    name = "fake-judge"

    def __init__(self, verdict="equivalent", reason="canned", text=None, fatal=False):
        self.model_id = "fake-judge-1"
        self._verdict, self._reason = verdict, reason
        self._text, self.fatal = text, fatal
        self.prompts: list[str] = []

    async def complete(self, prompt, *, max_tokens=2000, temperature=0.0):
        self.prompts.append(prompt)
        if self.fatal:
            raise FatalError("judge exploded")
        body = self._text
        if body is None:
            body = json.dumps({"verdict": self._verdict, "reason": self._reason})
        return ModelResponse(
            text=body, model_id=self.model_id, tokens_in=10, tokens_out=5,
            latency_ms=1.0, cost_usd=0.002, finish_reason="end_turn", raw={},
        )

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _task(expected=None) -> Task:
    return Task(
        id="t1", category="clean", difficulty="easy", edge_case=False,
        input="NORTHSTAR HEALTH PLAN OF OHIO\nMember: Jane Doe",
        expected=dict(EXPECTED if expected is None else expected),
    )


def _response(**overrides) -> ModelResponse:
    payload = dict(EXPECTED)
    payload.update(overrides)
    return ModelResponse(
        text=json.dumps(payload), model_id="m", tokens_in=1, tokens_out=1,
        latency_ms=1.0, cost_usd=0.0, finish_reason="end_turn", raw={},
    )


def _score(adapter, response, **kwargs):
    return asyncio.run(JudgeScorer(adapter, **kwargs).score_async(_task(), response))


# --- what gets judged --------------------------------------------------------


def test_near_miss_field_is_judged():
    adapter = VerdictAdapter(verdict="equivalent")

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert adapter.calls == 1
    assert score.detail["fields"]["payer_name"]["judged"] is True


def test_exact_match_is_not_judged():
    adapter = VerdictAdapter()

    score = _score(adapter, _response())

    assert adapter.calls == 0
    assert all(not f["judged"] for f in score.detail["fields"].values())


def test_field_with_a_null_side_is_not_judged_by_default():
    # provider_npi is expected null; a hallucinated value is a mismatch but
    # not a near-miss — "is X the same as nothing?" is not the question.
    adapter = VerdictAdapter()

    score = _score(adapter, _response(provider_npi="1234567890"))

    assert adapter.calls == 0
    assert score.detail["fields"]["provider_npi"]["judged"] is False


def test_missing_value_is_not_judged_by_default():
    adapter = VerdictAdapter()

    _score(adapter, _response(member_id=None))

    assert adapter.calls == 0


def test_judge_all_judges_every_mismatched_field():
    adapter = VerdictAdapter(verdict="different")

    score = _score(
        adapter,
        _response(provider_npi="1234567890", member_id=None, payer_name="Other Co"),
        only_near_misses=False,
    )

    assert adapter.calls == 3
    judged = {f for f, e in score.detail["fields"].items() if e["judged"]}
    assert judged == {"provider_npi", "member_id", "payer_name"}


# --- effect on the score -----------------------------------------------------


def test_equivalent_raises_f1_to_one():
    # One near-miss out of 7 non-null fields: raw F1 = 6/7, rescued to 1.0.
    adapter = VerdictAdapter(verdict="equivalent")

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["raw_f1"] == pytest.approx(6 / 7)
    assert score.value == pytest.approx(1.0)


def test_different_leaves_f1_unchanged():
    adapter = VerdictAdapter(verdict="different")

    score = _score(adapter, _response(payer_name="Completely Other Payer"))

    assert score.value == pytest.approx(score.detail["raw_f1"])
    assert score.value == pytest.approx(6 / 7)


def test_both_raw_and_adjusted_f1_are_always_reported():
    # The gap between them is the size of the semantic-equivalence tail.
    adapter = VerdictAdapter(verdict="equivalent")

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert "raw_f1" in score.detail and "f1" in score.detail
    assert score.detail["f1"] > score.detail["raw_f1"]


# --- failure falls back, never forward ---------------------------------------


def test_unparseable_verdict_falls_back_to_deterministic():
    adapter = VerdictAdapter(text="I think they're basically the same, honestly.")

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["fields"]["payer_name"]["verdict"] == "judge_failed"
    assert score.value == pytest.approx(score.detail["raw_f1"])


def test_verdict_outside_the_allowed_set_falls_back():
    adapter = VerdictAdapter(text=json.dumps({"verdict": "maybe", "reason": "unsure"}))

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["fields"]["payer_name"]["verdict"] == "judge_failed"
    assert score.value == pytest.approx(6 / 7)


def test_adapter_error_falls_back_rather_than_raising():
    adapter = VerdictAdapter(fatal=True)

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["fields"]["payer_name"]["verdict"] == "judge_failed"
    assert score.value == pytest.approx(6 / 7)


# --- blinding ----------------------------------------------------------------


def test_judge_prompt_does_not_reveal_the_deterministic_verdict():
    # If the judge could see what the automated comparison concluded, its
    # agreement would measure suggestibility rather than judgement. The
    # rubric's own guidance uses words like "wrong" in the abstract; what
    # must not appear is a claim about *this* field's outcome.
    adapter = VerdictAdapter()

    _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    prompt = adapter.prompts[0].casefold()

    # Internal outcome labels must never reach the judge.
    for label in ("fp_fn_wrong", "fp_hallucinated", "fn_missed", "true positive"):
        assert label not in prompt, f"prompt leaks outcome label {label!r}"

    # Nor may either value be asserted as the right one.
    for claim in (
        "the correct answer",
        "ground truth",
        "the right answer",
        "is correct",
        "an automated comparison found",
        "differs from the reference",
    ):
        assert claim not in prompt, f"prompt anchors the judge with {claim!r}"


def test_judge_prompt_contains_document_field_and_both_values():
    adapter = VerdictAdapter()

    _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    prompt = adapter.prompts[0]
    assert "NORTHSTAR HEALTH PLAN OF OHIO" in prompt
    assert "payer_name" in prompt
    assert "northstar health plan of ohio" in prompt.casefold()


# --- cost and caching --------------------------------------------------------


def test_judge_cost_is_tracked_separately():
    adapter = VerdictAdapter()

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["judge_cost_usd"] == pytest.approx(0.002)
    assert score.detail["judge_calls"] == 1


def test_judge_calls_are_cached(tmp_path: Path):
    cache = ResponseCache(cache_dir=tmp_path)
    adapter = VerdictAdapter()
    response = _response(payer_name="Northstar Health Plan of Ohio")

    _score(adapter, response, cache=cache)
    second = _score(adapter, response, cache=cache)

    assert adapter.calls == 1  # the second run was served from cache
    assert second.detail["fields"]["payer_name"]["cached"] is True
    assert second.detail["judge_cost_usd"] == 0.0


def test_judge_cache_key_changes_when_the_rubric_changes(tmp_path: Path):
    # Verdicts produced under a different rubric are not comparable, so a
    # rubric edit must force re-judging rather than silently reusing them.
    cache = ResponseCache(cache_dir=tmp_path)
    adapter = VerdictAdapter()
    response = _response(payer_name="Northstar Health Plan of Ohio")

    _score(adapter, response, cache=cache)

    scorer = JudgeScorer(adapter, cache=cache)
    scorer.prompt_hash = "different-rubric-hash"
    asyncio.run(scorer.score_async(_task(), response))

    assert adapter.calls == 2  # cache miss forced a fresh judgement


def test_rubric_hash_is_recorded_on_the_score():
    adapter = VerdictAdapter()

    score = _score(adapter, _response(payer_name="Northstar Health Plan of Ohio"))

    assert score.detail["judge_prompt_hash"] == judge_prompt_hash("judge_v1")
    assert score.detail["judge_model_id"] == "fake-judge-1"


# --- degenerate input --------------------------------------------------------


def test_unextractable_response_is_not_judged():
    adapter = VerdictAdapter()

    score = asyncio.run(
        JudgeScorer(adapter).score_async(
            _task(),
            ModelResponse(text="no json", model_id="m", tokens_in=1, tokens_out=1,
                          latency_ms=1.0, cost_usd=0.0, finish_reason="end_turn", raw={}),
        )
    )

    assert adapter.calls == 0
    assert score.value == 0.0
    assert score.detail["extraction_failed"] is True
