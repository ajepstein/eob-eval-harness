"""Per-field comparison producing micro-averaged precision, recall, and F1.

Values are compared *after* normalization (``harness.normalize``), so
casing, whitespace, currency formatting, and date layout never count as
errors. What remains is extraction quality.

Counting rules
--------------

Each of the 8 schema fields lands in exactly one bucket:

===========================  ==========================  ====================
expected                     predicted                   outcome
===========================  ==========================  ====================
non-null                     equal after normalization   ``tp``
non-null                     non-null but different      ``fp_fn_wrong``
                                                         (**both** FP and FN)
non-null                     null                        ``fn_missed``
null                         null                        ``tn`` (excluded)
null                         non-null                    ``fp_hallucinated``
===========================  ==========================  ====================

Two of these deserve their reasoning stated:

**A wrong value counts twice.** It is a bad prediction (false positive) and
simultaneously a miss of the real value (false negative). Counting it once
would make a confidently wrong model score the same as a model that
correctly abstained, which is backwards — a plausible wrong NPI on a claim
form is worse than a blank one.

**A hallucination is a false positive.** Expected-null means the document
genuinely does not contain that field, so any value is invented. This is
the case the ``missing_field`` tasks exist to create, and it is the number
most worth watching in a document-extraction system.

True negatives are excluded from the micro-average. Including them would
let a model inflate its score by correctly leaving nullable fields empty,
which requires no extraction at all.

Micro-averaging (pooling counts across fields, then computing the ratio
once) is deliberate over macro-averaging: it weights every field decision
equally rather than every field *type* equally, so a model cannot offset a
systematic failure on one field by being right about seven others in tasks
where that field is absent anyway.
"""

from __future__ import annotations

from harness.extract import extract_json
from harness.normalize import FIELD_NORMALIZERS
from harness.types import ModelResponse, Score, Task

# Outcome labels used in Score.detail.
TP = "tp"
FP_FN_WRONG = "fp_fn_wrong"
FN_MISSED = "fn_missed"
TN = "tn"
FP_HALLUCINATED = "fp_hallucinated"

# outcome -> (true positives, false positives, false negatives)
_COUNTS: dict[str, tuple[int, int, int]] = {
    TP: (1, 0, 0),
    FP_FN_WRONG: (0, 1, 1),
    FN_MISSED: (0, 0, 1),
    TN: (0, 0, 0),
    FP_HALLUCINATED: (0, 1, 0),
}


def classify(expected: object, predicted: object) -> str:
    """Bucket one already-normalized (expected, predicted) pair.

    Both arguments must already be normalized. Comparisons use ``is None``
    rather than truthiness because ``[]`` (no CPT codes found) and
    ``Decimal("0.00")`` (a genuine zero amount) are falsy but are real,
    non-null answers.
    """
    if expected is None:
        return TN if predicted is None else FP_HALLUCINATED
    if predicted is None:
        return FN_MISSED
    return TP if expected == predicted else FP_FN_WRONG


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Micro precision, recall, F1 from pooled counts."""
    if tp + fp + fn == 0:
        # Nothing to find and nothing wrongly found — vacuously perfect.
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


class FieldScorer:
    name = "fields"

    def score(self, task: Task, response: ModelResponse) -> Score:
        parsed, method = extract_json(response.text)

        if parsed is None:
            # No JSON means no field survived extraction. Report zero rather
            # than raising — a malformed response is a result, not a crash.
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                detail={
                    "extraction_method": method,
                    "extraction_failed": True,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            )

        outcomes: dict[str, str] = {}
        tp = fp = fn = 0

        for field, normalizer in FIELD_NORMALIZERS.items():
            expected = normalizer(task.expected.get(field))
            predicted = normalizer(parsed.get(field))

            outcome = classify(expected, predicted)
            outcomes[field] = outcome

            d_tp, d_fp, d_fn = _COUNTS[outcome]
            tp += d_tp
            fp += d_fp
            fn += d_fn

        precision, recall, f1 = prf1(tp, fp, fn)

        return Score(
            scorer=self.name,
            value=f1,
            passed=f1 == 1.0,
            detail={
                "extraction_method": method,
                "extraction_failed": False,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fields": outcomes,
            },
        )
