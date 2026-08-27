import math

import pytest

from harness.bias_tests import human_drift, length_bias, self_preference
from harness.calibration import (
    agreement,
    bootstrap_kappa_ci,
    cohens_kappa,
    disagreements,
    intra_rater_kappa,
    kappa_band,
)

EQ, DIFF = "equivalent", "different"


# --- kappa, hand-computed ----------------------------------------------------


def test_kappa_on_a_hand_computed_2x2():
    # human:  eq, eq, diff, diff      judge: eq, diff, diff, diff
    # observed agreement = 3/4 = 0.75
    # human marginals  eq .5   diff .5
    # judge marginals  eq .25  diff .75
    # expected = .5*.25 + .5*.75 = 0.5
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
    result = cohens_kappa([(EQ, EQ), (EQ, DIFF), (DIFF, DIFF), (DIFF, DIFF)])

    assert result.raw_agreement == pytest.approx(0.75)
    assert result.kappa == pytest.approx(0.5)


def test_perfect_agreement_with_mixed_classes_is_kappa_one():
    # observed = 1.0; expected = .5*.5 + .5*.5 = 0.5; kappa = (1-.5)/(1-.5) = 1
    result = cohens_kappa([(EQ, EQ), (EQ, EQ), (DIFF, DIFF), (DIFF, DIFF)])

    assert result.kappa == pytest.approx(1.0)


def test_degenerate_all_one_class_makes_kappa_undefined():
    # Every item equivalent for both raters: observed = 1.0 but expected is
    # also 1.0, so kappa is 0/0. Reporting 1.0 here would claim perfect
    # reliability from a rater that never made a distinction.
    result = cohens_kappa([(EQ, EQ)] * 10)

    assert math.isnan(result.kappa)
    assert result.raw_agreement == pytest.approx(1.0)
    assert "undefined" in result.undefined_reason


def test_chance_level_agreement_is_kappa_zero():
    # human eq .5 / diff .5, judge eq .5 / diff .5, observed .5
    # expected = .25 + .25 = .5  ->  kappa = 0
    result = cohens_kappa([(EQ, EQ), (EQ, DIFF), (DIFF, EQ), (DIFF, DIFF)])

    assert result.kappa == pytest.approx(0.0)


def test_worse_than_chance_is_negative():
    result = cohens_kappa([(EQ, DIFF), (EQ, DIFF), (DIFF, EQ), (DIFF, EQ)])

    assert result.kappa < 0


def test_empty_input_is_undefined_not_a_crash():
    result = cohens_kappa([])

    assert math.isnan(result.kappa)
    assert result.n == 0


# --- the kappa paradox -------------------------------------------------------


def test_skewed_marginals_are_flagged():
    # 95% one class: high raw agreement, unstable kappa. This is the
    # situation the real label set is in, so it must be surfaced.
    pairs = [(EQ, EQ)] * 19 + [(DIFF, DIFF)]
    result = cohens_kappa(pairs)

    assert result.raw_agreement == pytest.approx(1.0)
    assert result.skew_warning is not None
    assert "unstable" in result.skew_warning


def test_balanced_marginals_are_not_flagged():
    result = cohens_kappa([(EQ, EQ)] * 10 + [(DIFF, DIFF)] * 10)

    assert result.skew_warning is None


def test_high_raw_agreement_can_still_give_low_kappa():
    # The paradox itself: 90% agreement, near-zero kappa.
    pairs = [(EQ, EQ)] * 18 + [(EQ, DIFF), (DIFF, EQ)]
    result = cohens_kappa(pairs)

    assert result.raw_agreement == pytest.approx(0.9)
    assert result.kappa < 0.2


# --- bands -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(0.95, "strong"), (0.70, "usable with caveats"),
     (0.50, "weak"), (0.10, "close to useless")],
)
def test_kappa_bands(value, expected):
    assert kappa_band(value) == expected


def test_undefined_kappa_has_no_band():
    assert kappa_band(float("nan")) == "undefined"


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    pairs = [(EQ, EQ)] * 15 + [(DIFF, DIFF)] * 10 + [(EQ, DIFF)] * 5
    a = bootstrap_kappa_ci(pairs, iterations=500, seed=1)
    b = bootstrap_kappa_ci(pairs, iterations=500, seed=1)

    assert a == b


def test_bootstrap_interval_brackets_the_point_estimate():
    pairs = [(EQ, EQ)] * 15 + [(DIFF, DIFF)] * 10 + [(EQ, DIFF)] * 5
    point = cohens_kappa(pairs).kappa
    lo, hi = bootstrap_kappa_ci(pairs, iterations=1000, seed=0)

    assert lo <= point <= hi


def test_bootstrap_interval_narrows_as_n_grows():
    small = [(EQ, EQ), (DIFF, DIFF)] * 10
    large = [(EQ, EQ), (DIFF, DIFF)] * 200
    small = small + [(EQ, DIFF)] * 4
    large = large + [(EQ, DIFF)] * 80

    lo_s, hi_s = bootstrap_kappa_ci(small, iterations=500, seed=0)
    lo_l, hi_l = bootstrap_kappa_ci(large, iterations=500, seed=0)

    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_on_a_single_item_is_undefined():
    lo, hi = bootstrap_kappa_ci([(EQ, EQ)], iterations=10, seed=0)

    assert math.isnan(lo) and math.isnan(hi)


# --- human ceiling -----------------------------------------------------------


def test_intra_rater_kappa_from_repeated_items():
    ceiling, n = intra_rater_kappa([(EQ, EQ), (EQ, EQ), (DIFF, DIFF), (DIFF, DIFF)])

    assert ceiling == pytest.approx(1.0)
    assert n == 4


def test_intra_rater_ignores_non_decisive_verdicts():
    ceiling, n = intra_rater_kappa(
        [(EQ, EQ), (DIFF, DIFF), ("unsure", EQ), (EQ, "bad_task")]
    )

    assert n == 2


def test_intra_rater_with_too_few_items_returns_none():
    ceiling, n = intra_rater_kappa([(EQ, EQ)])

    assert ceiling is None


# --- the full report ---------------------------------------------------------


def _labels(rows):
    return [
        {"run_id": "r", "task_id": t, "field": f, "verdict": v,
         "pass_number": 1, "seconds": s}
        for t, f, v, s in rows
    ]


def test_agreement_excludes_unsure_and_bad_task_but_counts_them():
    labels = _labels([
        ("t1", "member_id", EQ, 5.0),
        ("t2", "member_id", DIFF, 5.0),
        ("t3", "member_id", "unsure", 5.0),
        ("t4", "member_id", "bad_task", 5.0),
    ])
    verdicts = {("r", f"t{i}", "member_id"): EQ for i in range(1, 5)}

    report = agreement(labels, verdicts, bootstrap_iterations=100)

    assert report.n == 2
    assert report.excluded == {"unsure": 1, "bad_task": 1}


def test_agreement_reports_items_with_no_judge_verdict():
    labels = _labels([("t1", "member_id", EQ, 1.0)])

    report = agreement(labels, {}, bootstrap_iterations=100)

    assert report.n == 0
    assert report.excluded["no_judge_verdict"] == 1


def test_per_category_breakdown_is_computed():
    labels = _labels([
        ("t1", "f", EQ, 1.0), ("t2", "f", EQ, 1.0),
        ("t3", "f", DIFF, 1.0), ("t4", "f", DIFF, 1.0),
    ])
    verdicts = {("r", t, "f"): (EQ if t in ("t1", "t2") else DIFF)
                for t in ("t1", "t2", "t3", "t4")}
    categories = {("r", "t1"): "hard", ("r", "t2"): "hard",
                  ("r", "t3"): "name_variance", ("r", "t4"): "name_variance"}

    report = agreement(labels, verdicts, categories=categories,
                       bootstrap_iterations=100)

    assert set(report.per_category) == {"hard", "name_variance"}


def test_thin_category_is_flagged_rather_than_crashing():
    labels = _labels([("t1", "f", EQ, 1.0), ("t2", "f", DIFF, 1.0)])
    verdicts = {("r", "t1", "f"): EQ, ("r", "t2", "f"): DIFF}
    categories = {("r", "t1"): "hard", ("r", "t2"): "hard"}

    report = agreement(labels, verdicts, categories=categories,
                       bootstrap_iterations=100, min_category_n=10)

    assert any("indicative" in note for note in report.notes)


def test_second_pass_of_a_repeated_item_does_not_double_count():
    labels = [
        {"run_id": "r", "task_id": "t1", "field": "f", "verdict": EQ,
         "pass_number": 1, "seconds": 1.0},
        {"run_id": "r", "task_id": "t1", "field": "f", "verdict": DIFF,
         "pass_number": 2, "seconds": 1.0},
    ]
    report = agreement(labels, {("r", "t1", "f"): EQ}, bootstrap_iterations=100)

    assert report.n == 1


def test_disagreements_are_returned_slowest_first():
    labels = _labels([
        ("t1", "f", EQ, 5.0), ("t2", "f", DIFF, 90.0), ("t3", "f", EQ, 2.0),
    ])
    verdicts = {("r", "t1", "f"): DIFF, ("r", "t2", "f"): EQ,
                ("r", "t3", "f"): DIFF}

    rows = disagreements(labels, verdicts)

    assert [r["task_id"] for r in rows] == ["t2", "t1", "t3"]


# --- bias tests --------------------------------------------------------------


def test_length_bias_fires_on_a_prefers_longer_judge():
    items = (
        [{"verdict": EQ, "predicted": "x" * 80} for _ in range(10)]
        + [{"verdict": DIFF, "predicted": "x"} for _ in range(10)]
    )
    result = length_bias(items)

    assert result.fired
    assert result.statistic > 0.30


def test_length_bias_does_not_fire_when_length_is_unrelated():
    items = [
        {"verdict": EQ if i % 2 else DIFF, "predicted": "x" * (10 + i % 3)}
        for i in range(20)
    ]
    result = length_bias(items)

    assert not result.fired


def test_length_bias_with_too_few_items_is_inconclusive():
    result = length_bias([{"verdict": EQ, "predicted": "x"}])

    assert not result.fired
    assert "Too few" in result.interpretation


def test_self_preference_fires_when_a_judge_favours_its_own_family():
    rows = {
        "anthropic": (
            [{"source_family": "anthropic", "verdict": EQ} for _ in range(10)]
            + [{"source_family": "openai", "verdict": DIFF} for _ in range(10)]
        )
    }
    result = self_preference(rows)

    assert result.fired
    assert result.statistic == pytest.approx(1.0)


def test_self_preference_needs_both_families():
    rows = {"anthropic": [{"source_family": "anthropic", "verdict": EQ}]}
    result = self_preference(rows)

    assert not result.fired
    assert "two families" in result.interpretation


def test_human_drift_fires_when_verdicts_change_over_the_session():
    labels = [
        {"verdict": EQ if i < 15 else DIFF, "seconds": 5.0} for i in range(30)
    ]
    result = human_drift(labels)

    assert result.fired


def test_human_drift_does_not_fire_on_a_steady_session():
    labels = [
        {"verdict": EQ if i % 2 else DIFF, "seconds": 5.0} for i in range(30)
    ]
    result = human_drift(labels)

    assert not result.fired
