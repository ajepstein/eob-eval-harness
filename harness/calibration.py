"""Measuring whether the judge can be trusted.

This is the week the project's thesis rests on. A harness that reports
judge-adjusted scores without reporting judge reliability is asserting
something it has not measured, so everything here exists to put a number
and an interval on that assertion — including when the number is bad.

Three things are reported together, and they only mean something together:

**Kappa**, not raw agreement. Raw agreement is inflated by whichever class
dominates; on a 93/7 split a judge that answered "equivalent" every single
time would score 93% and be worthless.

**Its confidence interval.** With ~100 items the interval is wide, and a
point estimate quoted alone invites a conclusion the data cannot support.

**The human ceiling**, from double-labelled items. A judge at kappa 0.72
against a human ceiling of 0.78 is close to the limit of what any judge
could achieve on this task; the same 0.72 against a ceiling of 0.95 means
something quite different. Reporting the judge's number without the ceiling
makes the first case look like a failure.

Kappa is also unstable when the marginals are skewed — the "kappa paradox",
where high agreement coexists with near-zero kappa. That is detected and
stated rather than glossed, because it is the situation this suite is
actually in.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Verdicts that are real decisions. `unsure` and `bad_task` are excluded
# from the primary metric — they are not disagreements with the judge, they
# are statements that the item could not be judged or that the answer key
# is wrong. Their counts are reported, because a high unsure rate is itself
# a finding about the rubric.
DECISIVE = ("equivalent", "different")

# Below this many labelled items a calibration is not a measurement. Kappa on
# a handful of items is dominated by noise — two items can produce a tidy
# 0.0 or 1.0 that means nothing — and a number that *looks* measured is worse
# than an admitted absence, because it silently unblocks every gate that
# depends on it. Consumers treat anything below this as uncalibrated.
MIN_CALIBRATION_N = 30


def calibration_is_usable(calibration: dict | None) -> tuple[bool, str | None]:
    """Whether a stored calibration may be relied on, and why not if not."""
    if calibration is None:
        return False, "no calibration exists for this rubric"
    n = calibration.get("n") or 0
    if n < MIN_CALIBRATION_N:
        return False, (
            f"a calibration exists but rests on only {n} labelled item(s); "
            f"at least {MIN_CALIBRATION_N} are needed before kappa is a "
            f"measurement rather than noise"
        )
    if calibration.get("kappa") is None:
        return False, (
            "the stored calibration has an undefined kappa (degenerate "
            "marginals), so it establishes no reliability"
        )
    return True, None

# Landis & Koch bands, with the honest gloss rather than the bare label.
_BANDS = (
    (0.80, "strong"),
    (0.60, "usable with caveats"),
    (0.40, "weak"),
    (-1.0, "close to useless"),
)


def kappa_band(value: float) -> str:
    if value != value:  # NaN
        return "undefined"
    for threshold, label in _BANDS:
        if value >= threshold:
            return label
    return "close to useless"


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    raw_agreement: float
    n: int
    confusion: dict[tuple[str, str], int]
    undefined_reason: str | None = None
    skew_warning: str | None = None


def cohens_kappa(pairs: list[tuple[str, str]]) -> KappaResult:
    """Cohen's kappa for paired categorical judgements.

    `pairs` is [(human_verdict, judge_verdict), ...] over decisive items.
    """
    n = len(pairs)
    confusion: dict[tuple[str, str], int] = Counter(pairs)
    if n == 0:
        return KappaResult(
            kappa=float("nan"), raw_agreement=float("nan"), n=0,
            confusion={}, undefined_reason="no decisive items to compare",
        )

    observed = sum(c for (a, b), c in confusion.items() if a == b) / n

    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    human = Counter(a for a, _ in pairs)
    judge = Counter(b for _, b in pairs)
    expected = sum((human[l] / n) * (judge[l] / n) for l in labels)

    if math.isclose(expected, 1.0):
        # Both raters used a single class throughout. Agreement is total but
        # chance agreement is also total, so kappa is 0/0 — genuinely
        # undefined rather than perfect.
        return KappaResult(
            kappa=float("nan"), raw_agreement=observed, n=n, confusion=dict(confusion),
            undefined_reason=(
                "both raters used only one category, so chance agreement is "
                "100% and kappa is undefined (0/0). Raw agreement is still "
                "meaningful; kappa is not."
            ),
        )

    value = (observed - expected) / (1 - expected)

    # The kappa paradox: with skewed marginals, kappa is depressed and
    # unstable even when raw agreement is high.
    skew = None
    majority = max(human.values()) / n
    if majority >= 0.85:
        skew = (
            f"{majority:.0%} of human labels fall in one class. Kappa is "
            f"unstable at this imbalance — small changes in the minority "
            f"class move it a long way, and it will read low even when raw "
            f"agreement is high. Read it alongside raw agreement, not instead."
        )

    return KappaResult(
        kappa=value, raw_agreement=observed, n=n,
        confusion=dict(confusion), skew_warning=skew,
    )


def bootstrap_kappa_ci(
    pairs: list[tuple[str, str]],
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for kappa. Seeded, so it reproduces.

    Resamples *items*, which is the unit that was sampled in the first
    place. Degenerate resamples (where kappa is undefined) are discarded
    rather than counted as zero, which would bias the interval downward.
    """
    if len(pairs) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        result = cohens_kappa(sample)
        if result.kappa == result.kappa:  # not NaN
            values.append(result.kappa)

    if not values:
        return (float("nan"), float("nan"))

    values.sort()
    lo = values[int((alpha / 2) * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return (lo, hi)


@dataclass(frozen=True)
class AgreementReport:
    n: int
    raw_agreement: float
    kappa: float
    kappa_ci: tuple[float, float]
    band: str
    confusion: dict[tuple[str, str], int]
    per_category: dict[str, KappaResult]
    excluded: dict[str, int]
    human_ceiling_kappa: float | None = None
    human_ceiling_n: int = 0
    undefined_reason: str | None = None
    skew_warning: str | None = None
    notes: list[str] = field(default_factory=list)


def intra_rater_kappa(
    double_labelled: list[tuple[str, str]]
) -> tuple[float | None, int]:
    """Self-agreement from items labelled twice — the ceiling for any judge.

    No human labels perfectly consistently, so this bounds what agreement
    the judge could possibly have reached.
    """
    decisive = [
        (a, b) for a, b in double_labelled if a in DECISIVE and b in DECISIVE
    ]
    if len(decisive) < 2:
        return None, len(decisive)
    result = cohens_kappa(decisive)
    return (result.kappa if result.kappa == result.kappa else None), len(decisive)


def agreement(
    labels: list[dict],
    verdicts: dict[tuple[str, str, str], str],
    categories: dict[tuple[str, str], str] | None = None,
    double_labelled: list[tuple[str, str]] | None = None,
    seed: int = 0,
    bootstrap_iterations: int = 10_000,
    min_category_n: int = 10,
) -> AgreementReport:
    """Compare human labels against judge verdicts.

    `labels`      rows from human_labels
    `verdicts`    (run_id, task_id, field) -> judge verdict
    `categories`  (run_id, task_id) -> category, for the per-category split
    """
    categories = categories or {}
    pairs: list[tuple[str, str]] = []
    per_category_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    excluded: Counter[str] = Counter()

    # Only the first pass of a double-labelled item enters the primary
    # metric; counting both would weight those items twice.
    seen: set[tuple[str, str, str]] = set()

    for row in labels:
        key = (row["run_id"], row["task_id"], row["field"])
        if row["verdict"] not in DECISIVE:
            excluded[row["verdict"]] += 1
            continue
        if row.get("pass_number", 1) != 1 or key in seen:
            continue
        judge_verdict = verdicts.get(key)
        if judge_verdict is None:
            excluded["no_judge_verdict"] += 1
            continue
        seen.add(key)
        pairs.append((row["verdict"], judge_verdict))
        category = categories.get((row["run_id"], row["task_id"]))
        if category:
            per_category_pairs[category].append((row["verdict"], judge_verdict))

    overall = cohens_kappa(pairs)
    ci = bootstrap_kappa_ci(pairs, iterations=bootstrap_iterations, seed=seed)

    notes: list[str] = []
    per_category: dict[str, KappaResult] = {}
    for category, cat_pairs in sorted(per_category_pairs.items()):
        per_category[category] = cohens_kappa(cat_pairs)
        if len(cat_pairs) < min_category_n:
            notes.append(
                f"{category}: only {len(cat_pairs)} items — treat this "
                f"category's kappa as indicative, not a measurement."
            )

    ceiling, ceiling_n = intra_rater_kappa(double_labelled or [])
    if ceiling is not None and overall.kappa == overall.kappa:
        if overall.kappa > ceiling:
            notes.append(
                "The judge's kappa exceeds the human self-agreement ceiling. "
                "That usually means the ceiling is measured on too few items "
                "to be reliable, not that the judge is superhuman."
            )

    return AgreementReport(
        n=overall.n,
        raw_agreement=overall.raw_agreement,
        kappa=overall.kappa,
        kappa_ci=ci,
        band=kappa_band(overall.kappa),
        confusion=overall.confusion,
        per_category=per_category,
        excluded=dict(excluded),
        human_ceiling_kappa=ceiling,
        human_ceiling_n=ceiling_n,
        undefined_reason=overall.undefined_reason,
        skew_warning=overall.skew_warning,
        notes=notes,
    )


def disagreements(
    labels: list[dict],
    verdicts: dict[tuple[str, str, str], str],
    limit: int = 15,
) -> list[dict]:
    """Items where human and judge differ, for hand review.

    These are the rows worth reading. Three things turn up in them: the
    judge is wrong, you were wrong, or the rubric is genuinely ambiguous —
    and the third is the most useful outcome, because it is fixable.
    """
    out = []
    for row in labels:
        key = (row["run_id"], row["task_id"], row["field"])
        judge_verdict = verdicts.get(key)
        if judge_verdict is None or row["verdict"] not in DECISIVE:
            continue
        if row["verdict"] != judge_verdict:
            out.append({**row, "judge_verdict": judge_verdict})
    # Slowest labels first: time spent is a decent proxy for how genuinely
    # difficult the item was, and the hard ones are what repay reading.
    out.sort(key=lambda r: -r.get("seconds", 0.0))
    return out[:limit]
