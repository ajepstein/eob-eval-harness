import math
import random
import statistics

import pytest

from harness.stats import (
    FrontierPoint,
    bootstrap_ci,
    describe_difference,
    holm_correction,
    mcnemar,
    minimum_detectable_effect,
    paired_bootstrap_diff,
    pareto_frontier,
    required_n,
)


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_ci_matches_the_analytic_interval_on_normal_data():
    rng = random.Random(11)
    values = [rng.gauss(0.80, 0.10) for _ in range(400)]

    boot = bootstrap_ci(values, iterations=4000, seed=0)

    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    assert boot.low == pytest.approx(mean - 1.96 * se, abs=0.01)
    assert boot.high == pytest.approx(mean + 1.96 * se, abs=0.01)


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    values = [0.2, 0.9, 0.5, 0.7, 0.4, 0.8]

    assert bootstrap_ci(values, iterations=500, seed=3) == bootstrap_ci(
        values, iterations=500, seed=3
    )


def test_bootstrap_interval_narrows_as_n_grows():
    rng = random.Random(5)
    small = [rng.gauss(0.8, 0.1) for _ in range(25)]
    large = [rng.gauss(0.8, 0.1) for _ in range(400)]

    assert bootstrap_ci(large, iterations=1000, seed=0).width < bootstrap_ci(
        small, iterations=1000, seed=0
    ).width


def test_bootstrap_on_identical_values_has_zero_width():
    interval = bootstrap_ci([0.75] * 40, iterations=500, seed=0)

    assert interval.width == pytest.approx(0.0)
    assert interval.point == pytest.approx(0.75)


def test_bootstrap_on_empty_input_is_nan_not_a_crash():
    interval = bootstrap_ci([], iterations=100, seed=0)

    assert math.isnan(interval.point)
    assert interval.n == 0


# --- paired comparison -------------------------------------------------------


def test_paired_bootstrap_on_identical_runs_contains_zero():
    # Comparing a run against itself must never find a difference.
    rng = random.Random(7)
    values = [rng.random() for _ in range(60)]

    interval = paired_bootstrap_diff(values, list(values), iterations=1000, seed=0)

    assert interval.point == pytest.approx(0.0)
    assert interval.spans_zero


def test_paired_bootstrap_detects_a_consistent_shift():
    rng = random.Random(9)
    a = [rng.gauss(0.70, 0.05) for _ in range(120)]
    b = [x + 0.20 for x in a]

    interval = paired_bootstrap_diff(a, b, iterations=1000, seed=0)

    assert interval.point == pytest.approx(0.20, abs=0.01)
    assert not interval.spans_zero


def test_paired_bootstrap_keeps_pairs_together():
    # If indices were drawn independently for each side, the perfectly
    # correlated pairing below would produce a much wider interval.
    a = [0.0, 1.0] * 40
    b = [0.1, 1.1] * 40

    interval = paired_bootstrap_diff(a, b, iterations=1000, seed=0)

    assert interval.width < 0.01
    assert interval.point == pytest.approx(0.1)


def test_paired_comparison_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_diff([0.1, 0.2], [0.1])


# --- McNemar -----------------------------------------------------------------


def test_mcnemar_matches_a_hand_computed_2x2():
    # b = 10 (a right, b wrong), c = 2 (a wrong, b right), n = 12
    # chi-square with continuity correction = (|10-2| - 1)^2 / 12
    #                                       = 49 / 12 = 4.08333...
    a = [True] * 10 + [False] * 2 + [True] * 5
    b = [False] * 10 + [True] * 2 + [True] * 5

    result = mcnemar(a, b)

    assert result.b == 10
    assert result.c == 2
    assert result.statistic == pytest.approx(49 / 12)


def test_mcnemar_ignores_concordant_pairs():
    # Tasks both models got right say nothing about which is better.
    a = [True] * 50 + [True, False]
    b = [True] * 50 + [False, True]

    result = mcnemar(a, b)

    assert result.n_discordant == 2


def test_mcnemar_with_no_discordant_pairs_finds_no_difference():
    a = b = [True, False, True, True]

    result = mcnemar(a, b)

    assert result.p_value == 1.0
    assert "never differed" in result.verdict


def test_mcnemar_exact_p_on_a_hand_computed_case():
    # b=5, c=0, n=5. Two-sided exact = 2 * (1/2)^5 = 0.0625
    a = [True] * 5 + [True] * 5
    b = [False] * 5 + [True] * 5

    result = mcnemar(a, b)

    assert result.p_exact == pytest.approx(0.0625)


def test_mcnemar_verdict_is_plain_language():
    a = [True, False] * 20
    b = [True, False] * 20

    assert "never differed" in mcnemar(a, b).verdict


# --- minimum detectable effect ----------------------------------------------


def test_mde_decreases_as_n_increases():
    assert minimum_detectable_effect(400, 0.8) < minimum_detectable_effect(100, 0.8)
    assert minimum_detectable_effect(100, 0.8) < minimum_detectable_effect(25, 0.8)


def test_mde_is_largest_near_a_baseline_of_one_half():
    # NOTE: the build plan states MDE "decreases ... as baseline accuracy
    # approaches 0.5". That is backwards. Binomial variance p(1-p) peaks at
    # 0.5, so that is where the effect is hardest to resolve, not easiest.
    # A suite whose models sit at 0.95 can resolve smaller differences than
    # one sitting at 0.50 — there is less variance to see through.
    at_half = minimum_detectable_effect(100, 0.50)
    at_high = minimum_detectable_effect(100, 0.95)
    at_low = minimum_detectable_effect(100, 0.05)

    assert at_half > at_high
    assert at_half > at_low
    assert at_high == pytest.approx(at_low)  # symmetric about 0.5


def test_mde_and_required_n_are_inverses():
    n = 150
    effect = minimum_detectable_effect(n, 0.8)

    assert required_n(effect, 0.8) == pytest.approx(n, abs=1)


def test_mde_on_degenerate_input_is_nan():
    assert math.isnan(minimum_detectable_effect(0, 0.8))
    assert math.isnan(minimum_detectable_effect(100, 1.0))


# --- Holm --------------------------------------------------------------------


def test_holm_matches_a_worked_example():
    # p = [.01, .02, .03, .04, .05], m = 5
    #   rank 0: 5*.01 = .05
    #   rank 1: 4*.02 = .08
    #   rank 2: 3*.03 = .09
    #   rank 3: 2*.04 = .08 -> held at .09 by monotonicity
    #   rank 4: 1*.05 = .05 -> held at .09
    adjusted = holm_correction([0.01, 0.02, 0.03, 0.04, 0.05])

    assert adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])


def test_holm_is_monotonic():
    adjusted = holm_correction([0.001, 0.4, 0.02, 0.9, 0.03])
    ordered = [a for _, a in sorted(zip([0.001, 0.4, 0.02, 0.9, 0.03], adjusted))]

    assert ordered == sorted(ordered)


def test_holm_caps_at_one():
    assert all(a <= 1.0 for a in holm_correction([0.9, 0.95, 0.99]))


def test_holm_is_less_conservative_than_bonferroni():
    raw = [0.01, 0.02, 0.03]
    holm = holm_correction(raw)
    bonferroni = [min(1.0, 3 * p) for p in raw]

    assert all(h <= b for h, b in zip(holm, bonferroni))
    assert any(h < b for h, b in zip(holm, bonferroni))


def test_holm_on_empty_input():
    assert holm_correction([]) == []


# --- Pareto frontier ---------------------------------------------------------


def test_pareto_identifies_a_dominated_point():
    points = [
        FrontierPoint("cheap-good", cost_per_task=0.001, quality=0.90),
        FrontierPoint("dear-worse", cost_per_task=0.010, quality=0.85),  # dominated
        FrontierPoint("dear-best", cost_per_task=0.020, quality=0.95),
    ]

    result = {p.label: p for p in pareto_frontier(points)}

    assert result["cheap-good"].pareto_optimal
    assert result["dear-best"].pareto_optimal
    assert not result["dear-worse"].pareto_optimal
    assert "cheap-good" in result["dear-worse"].dominated_by


def test_pareto_keeps_both_ends_of_a_genuine_tradeoff():
    points = [
        FrontierPoint("cheap", cost_per_task=0.001, quality=0.80),
        FrontierPoint("expensive", cost_per_task=0.020, quality=0.95),
    ]

    assert all(p.pareto_optimal for p in pareto_frontier(points))


def test_pareto_marks_an_equal_but_costlier_point_as_dominated():
    points = [
        FrontierPoint("a", cost_per_task=0.001, quality=0.90),
        FrontierPoint("b", cost_per_task=0.005, quality=0.90),
    ]

    result = {p.label: p for p in pareto_frontier(points)}

    assert result["a"].pareto_optimal
    assert not result["b"].pareto_optimal


def test_pareto_on_a_single_point():
    result = pareto_frontier([FrontierPoint("only", 0.001, 0.9)])

    assert result[0].pareto_optimal


# --- plain-language verdicts -------------------------------------------------


def test_difference_spanning_zero_is_described_as_indistinguishable():
    rng = random.Random(3)
    values = [rng.random() for _ in range(50)]
    interval = paired_bootstrap_diff(values, list(values), iterations=500, seed=0)

    text = describe_difference(interval, "anthropic", "openai")

    assert "not distinguishable from zero" in text
    assert "n=50" in text


def test_a_real_difference_names_the_better_model():
    a = [0.5] * 60
    b = [0.9] * 60
    interval = paired_bootstrap_diff(a, b, iterations=500, seed=0)

    text = describe_difference(interval, "anthropic", "openai")

    assert "openai scores higher" in text
