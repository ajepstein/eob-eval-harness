"""Confidence intervals, paired tests, and power for the eval suite.

Everything before this week reported point estimates. With ~100 tasks most
visible differences between models are noise, and the harness should be
able to say so rather than leaving a reader to infer a winner from the
third decimal place.

Two choices here do most of the work:

**Bootstrap over tasks, never over fields.** The eight fields within a task
are drawn from the same document and fail together — a model that misreads
the header gets the payer and the member id wrong at once. Resampling
fields as if independent would treat ~800 correlated observations as 800
independent ones and produce intervals far too narrow, which is exactly the
error that makes a null result look like a finding.

**Paired tests throughout.** Every model sees identical tasks, so the
task-to-task variation that dominates the raw spread cancels. A paired test
on the same data is substantially more powerful than an unpaired one, and
using the unpaired version here would be throwing away the design.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Sequence

_NORMAL = NormalDist()


# --- bootstrap ---------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    @property
    def spans_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for a statistic over per-task values.

    `values` must be one number per *task*, not per field.
    """
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    point = statistic(values)
    if n == 1:
        return Interval(point, point, point, 1)

    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(sample))
    draws.sort()
    lo = draws[int((alpha / 2) * iterations)]
    hi = draws[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return Interval(point, lo, hi, n)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Interval for mean(b) - mean(a), resampling *task indices* together.

    Resampling the pair keeps each model's score on a given task attached to
    the other's, which is what makes the comparison paired. Drawing the two
    independently would reintroduce the between-task variance the pairing
    exists to remove.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    point = statistics.fmean(b) - statistics.fmean(a)
    if n == 1:
        return Interval(point, point, point, 1)

    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            statistics.fmean([b[i] for i in idx]) - statistics.fmean([a[i] for i in idx])
        )
    draws.sort()
    lo = draws[int((alpha / 2) * iterations)]
    hi = draws[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return Interval(point, lo, hi, n)


# --- McNemar -----------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    b: int  # a correct, b wrong
    c: int  # a wrong, b correct
    statistic: float
    p_value: float
    p_exact: float
    n_discordant: int

    @property
    def verdict(self) -> str:
        if self.n_discordant == 0:
            return "the two models never differed on any task"
        if self.p_value > 0.05:
            return "difference not distinguishable from zero"
        return "difference unlikely to be chance"


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Paired test for binary outcomes (schema pass, per-field correct).

    Only the discordant pairs carry information: tasks both models got right,
    or both got wrong, say nothing about which is better. Reports the
    continuity-corrected chi-square alongside an exact binomial p-value,
    because with the handful of discordant pairs this suite produces the
    chi-square approximation is not trustworthy and the exact test is.
    """
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")

    b_count = sum(1 for x, y in zip(a, b) if x and not y)
    c_count = sum(1 for x, y in zip(a, b) if not x and y)
    n = b_count + c_count

    if n == 0:
        return McNemarResult(0, 0, 0.0, 1.0, 1.0, 0)

    # Edwards' continuity correction.
    statistic = (abs(b_count - c_count) - 1) ** 2 / n if n else 0.0
    p_chi = _chi2_sf_1df(statistic)

    # Exact two-sided binomial against p=0.5.
    k = min(b_count, c_count)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    p_exact = min(1.0, 2 * tail)

    return McNemarResult(b_count, c_count, statistic, p_chi, p_exact, n)


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 df.

    For 1 df this is exactly 2*(1 - Phi(sqrt(x))), so no gamma function is
    needed and the result is exact rather than approximated.
    """
    if x <= 0:
        return 1.0
    return 2.0 * (1.0 - _NORMAL.cdf(math.sqrt(x)))


# --- minimum detectable effect ----------------------------------------------


def minimum_detectable_effect(
    n: int, baseline: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest difference the suite can resolve, in absolute proportion.

    Answers "is my suite big enough?" in one number.

    NOTE ON DIRECTION: the effect is *largest* near a baseline of 0.5 and
    smallest at the extremes, because the binomial variance p(1-p) peaks at
    0.5. A suite whose models sit at 0.95 can resolve smaller differences
    than one whose models sit at 0.50 — there is simply less variance to see
    through. This is the opposite of the intuition that mid-range accuracy
    leaves "more room to improve".
    """
    if n <= 0 or not (0.0 < baseline < 1.0):
        return float("nan")
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = _NORMAL.inv_cdf(power)
    return (z_alpha + z_beta) * math.sqrt(2 * baseline * (1 - baseline) / n)


def required_n(
    effect: float, baseline: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Tasks needed to resolve a given difference — the inverse of the above."""
    if effect <= 0 or not (0.0 < baseline < 1.0):
        return 0
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = _NORMAL.inv_cdf(power)
    return math.ceil(2 * baseline * (1 - baseline) * ((z_alpha + z_beta) / effect) ** 2)


# --- multiple comparisons ----------------------------------------------------


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Three models across six categories is eighteen comparisons; at alpha
    0.05 roughly one will look significant by chance alone. Holm controls
    the family-wise error rate while being uniformly more powerful than
    plain Bonferroni.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        # Step-down: enforce monotonicity so an adjusted value can never be
        # smaller than one for a more significant raw p.
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


# --- cost/quality frontier ---------------------------------------------------


@dataclass(frozen=True)
class FrontierPoint:
    label: str
    cost_per_task: float
    quality: float
    quality_interval: Interval | None = None
    pareto_optimal: bool = False
    dominated_by: list[str] = field(default_factory=list)


def pareto_frontier(points: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    """Flag the points no alternative beats on both cost and quality.

    The question a buyer actually asks is not "which model is best" but
    "which models are worth their price". A dominated point — costlier *and*
    worse — is never the right choice, and saying so is more useful than a
    leaderboard.
    """
    out: list[FrontierPoint] = []
    for point in points:
        dominators = [
            other.label
            for other in points
            if other is not point
            and other.cost_per_task <= point.cost_per_task
            and other.quality >= point.quality
            and (
                other.cost_per_task < point.cost_per_task
                or other.quality > point.quality
            )
        ]
        out.append(
            FrontierPoint(
                label=point.label,
                cost_per_task=point.cost_per_task,
                quality=point.quality,
                quality_interval=point.quality_interval,
                pareto_optimal=not dominators,
                dominated_by=dominators,
            )
        )
    return out


# --- plain-language reporting ------------------------------------------------


def describe_difference(interval: Interval, label_a: str, label_b: str) -> str:
    """State the comparison in words.

    A reader who skims the number and misses the interval is the exact
    reader this harness exists to protect, so the verdict is spelled out
    rather than left to be inferred from bracket positions.
    """
    if interval.n == 0 or interval.point != interval.point:
        return "no comparable tasks"
    if interval.spans_zero:
        return (
            f"difference between {label_a} and {label_b} is not "
            f"distinguishable from zero at n={interval.n}"
        )
    better, worse = (label_b, label_a) if interval.point > 0 else (label_a, label_b)
    return (
        f"{better} scores higher than {worse} by {abs(interval.point):.3f} "
        f"[{abs(interval.high):.3f}, {abs(interval.low):.3f}] at n={interval.n}"
    )
