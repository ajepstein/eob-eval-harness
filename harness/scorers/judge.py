"""LLM-as-judge for the semantic equivalence tail.

Most extraction fields have a checkable right answer and ``fields.py``
already scores them correctly. This judge exists for one narrow job: the
cases where exact match after normalization says *wrong* but a domain
expert would say *right* — ``"Northstar Health"`` versus ``"Northstar
Health Plan of Ohio"``, ``"J. Doe"`` versus ``"Jane Doe"``.

**Why only near-misses.** A judge applied to every field is expensive,
slow, and noisier than the string comparison it replaces — it would
introduce disagreement on the 96% of fields that already match exactly.
Applied only to fields the deterministic scorer already flagged, it is
cheap and can only move fields that were going to score zero anyway.

**The judge never sees the deterministic verdict.** It is told which field
is disputed and shown both values, but not which one the automated
comparison preferred, and not that the reference value is "correct". If it
could see that, it would anchor on it and its agreement rate would measure
suggestibility rather than judgement.

**Failure falls back, never forward.** An unparseable verdict, a network
error, a refusal — all resolve to the deterministic verdict, never to
``equivalent``. The safe direction for a broken judge is "no change".
"""

from __future__ import annotations

import asyncio

from harness.adapters.base import Adapter, AdapterError
from harness.cache import ResponseCache, make_cache_key
from harness.extract import extract_json
from harness.normalize import FIELD_NORMALIZERS
from harness.prompts import judge_prompt_hash, load_prompt
from harness.scorers.fields import (
    FP_FN_WRONG,
    FP_HALLUCINATED,
    FN_MISSED,
    TP,
    _COUNTS,
    classify,
    prf1,
)
from harness.types import ModelResponse, Score, Task

EQUIVALENT = "equivalent"
DIFFERENT = "different"
JUDGE_FAILED = "judge_failed"

# Outcomes eligible for judging. The near-miss filter keeps only the first:
# a disagreement where both sides actually have a value. The other two
# involve a null on one side, where "are these the same thing?" is not the
# question being asked.
_NEAR_MISS = {FP_FN_WRONG}
_ANY_MISMATCH = {FP_FN_WRONG, FN_MISSED, FP_HALLUCINATED}

_ABSENT = "(absent — the document does not contain this field)"


def _render(value: object) -> str:
    return _ABSENT if value is None else str(value)


class JudgeScorer:
    name = "judge"

    def __init__(
        self,
        adapter: Adapter,
        prompt_name: str = "judge_v1",
        only_near_misses: bool = True,
        cache: ResponseCache | None = None,
        max_tokens: int = 512,
    ):
        self.adapter = adapter
        self.prompt_name = prompt_name
        self.only_near_misses = only_near_misses
        self.cache = cache
        self.max_tokens = max_tokens
        self._template = load_prompt(prompt_name)
        self.prompt_hash = judge_prompt_hash(prompt_name)

    # --- one contested field ------------------------------------------------

    def _build_prompt(self, task: Task, field: str, expected, predicted) -> str:
        return (
            self._template.replace("{document}", task.input)
            .replace("{field}", field)
            .replace("{expected}", _render(expected))
            .replace("{predicted}", _render(predicted))
        )

    async def _judge_field(
        self, task: Task, field: str, expected, predicted
    ) -> dict:
        prompt = self._build_prompt(task, field, expected, predicted)
        params = {"max_tokens": self.max_tokens, "judge_prompt": self.prompt_hash}
        key = make_cache_key(prompt, self.adapter.model_id, params, self.prompt_hash)

        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            response, cost, cached_hit = cached, 0.0, True
        else:
            try:
                response = await self.adapter.complete(prompt, max_tokens=self.max_tokens)
            except AdapterError as exc:
                # Falls back to the deterministic verdict, never to equivalent.
                return {
                    "verdict": JUDGE_FAILED,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "cost_usd": 0.0,
                    "cached": False,
                }
            if self.cache:
                self.cache.set(key, response)
            cost, cached_hit = response.cost_usd, False

        parsed, _ = extract_json(response.text)
        verdict = (parsed or {}).get("verdict")
        if verdict not in (EQUIVALENT, DIFFERENT):
            return {
                "verdict": JUDGE_FAILED,
                "reason": f"unparseable verdict: {response.text[:120]!r}",
                "cost_usd": cost,
                "cached": cached_hit,
            }

        return {
            "verdict": verdict,
            "reason": str((parsed or {}).get("reason", ""))[:500],
            "cost_usd": cost,
            "cached": cached_hit,
        }

    # --- the whole task -----------------------------------------------------

    async def score_async(self, task: Task, response: ModelResponse) -> Score:
        parsed, method = extract_json(response.text)
        eligible = _NEAR_MISS if self.only_near_misses else _ANY_MISMATCH

        if parsed is None:
            # Nothing to adjudicate: no field survived extraction.
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                detail={
                    "extraction_failed": True,
                    "raw_f1": 0.0,
                    "f1": 0.0,
                    "judge_calls": 0,
                    "judge_cost_usd": 0.0,
                    "judge_model_id": self.adapter.model_id,
                    "judge_prompt_hash": self.prompt_hash,
                    "fields": {},
                },
            )

        outcomes: dict[str, str] = {}
        values: dict[str, tuple] = {}
        contested: list[str] = []

        for field, normalizer in FIELD_NORMALIZERS.items():
            expected = normalizer(task.expected.get(field))
            predicted = normalizer(parsed.get(field))
            outcome = classify(expected, predicted)
            outcomes[field] = outcome
            values[field] = (expected, predicted)
            if outcome in eligible:
                contested.append(field)

        verdicts = await asyncio.gather(
            *(
                self._judge_field(task, field, *values[field])
                for field in contested
            )
        )

        detail_fields: dict[str, dict] = {}
        adjusted = dict(outcomes)
        judge_cost = 0.0

        for field in outcomes:
            entry: dict = {"outcome": outcomes[field], "judged": False}
            if field in contested:
                result = verdicts[contested.index(field)]
                judge_cost += result["cost_usd"]
                entry.update(
                    judged=True,
                    verdict=result["verdict"],
                    reason=result["reason"],
                    cached=result["cached"],
                    expected=_render(values[field][0]),
                    predicted=_render(values[field][1]),
                    cost_usd=result["cost_usd"],
                )
                if result["verdict"] == EQUIVALENT:
                    adjusted[field] = TP
            detail_fields[field] = entry

        def f1_of(outcome_map: dict[str, str]) -> float:
            tp = fp = fn = 0
            for outcome in outcome_map.values():
                d_tp, d_fp, d_fn = _COUNTS[outcome]
                tp += d_tp
                fp += d_fp
                fn += d_fn
            return prf1(tp, fp, fn)[2]

        raw_f1 = f1_of(outcomes)
        adjusted_f1 = f1_of(adjusted)

        return Score(
            scorer=self.name,
            value=adjusted_f1,
            passed=adjusted_f1 == 1.0,
            detail={
                "extraction_method": method,
                "extraction_failed": False,
                # Both numbers are always reported. The gap between them is
                # exactly the size of the semantic-equivalence tail, which is
                # the thing worth watching.
                "raw_f1": raw_f1,
                "f1": adjusted_f1,
                "judge_calls": len(contested),
                "judge_cost_usd": judge_cost,
                "judge_model_id": self.adapter.model_id,
                "judge_prompt_hash": self.prompt_hash,
                "only_near_misses": self.only_near_misses,
                "fields": detail_fields,
            },
        )
