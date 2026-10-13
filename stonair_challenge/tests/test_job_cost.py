import statistics

import pytest

from stonair import die_cost, job_cost, units
from stonair.estimate import op_c, params
from stonair.evidence import process_evidence


def _ev():
    return {r["evidence_id"]: r for r in process_evidence() if r["topic"] == "laser_job_cost"}


def test_rate_range_from_evidence_matches_assumptions():
    ev = _ev()
    rows = [ev[i] for i in job_cost.RATE_EVIDENCE]
    r = job_cost.rate_range()
    assert r["low"] == min(x["normalized_low"] for x in rows)
    assert r["high"] == max(x["normalized_high"] for x in rows)
    assert r["central"] == statistics.median((x["normalized_low"] + x["normalized_high"]) / 2 for x in rows)
    for level in ("low", "central", "high"):    # assumptions.csv must carry the computed values
        assert params("OP-C", level)["laser_rate_USD_h"] == pytest.approx(r[level])


def test_copper_price_derivation_matches_assumptions():
    s = job_cost.sheet_price_per_lb()
    assert s["full_sheet_036x120in"] == pytest.approx(420.71 / (30 * 1.0512))
    assert s["small_sheet_12x24in"] == pytest.approx(94.46 / 4.7232)
    assert params("OP-C", "low")["copper_sheet_USD_lb"] == pytest.approx(s["full_sheet_036x120in"], rel=1e-3)
    assert params("OP-C", "central")["copper_sheet_USD_lb"] == pytest.approx(s["small_sheet_12x24in"], rel=1e-3)
    assert s["spot_floor"] == pytest.approx(6.69)
    assert s["spot_floor"] < s["full_sheet_036x120in"] < s["small_sheet_12x24in"]


def test_no_checked_online_service_offers_0_3_mm_copper():
    rows = job_cost.thin_copper_availability()
    assert len(rows) == 3 and not any(r["offers_0.3_mm"] for r in rows)
    assert min(r["thinnest_copper_mm"] for r in rows) == pytest.approx(0.40)


def test_central_job_price_hand_calculation():
    c = op_c(params("OP-C"))
    h = c["in_cycle_time_h"] + c["handling_time_h"]
    lb = c["input_material_g"] / 453.59237
    expected = 50 + h * 130 + lb * 20.00 * 1.15
    j = job_cost.job_price("central")
    assert j["order_price_USD"] == pytest.approx(expected, rel=1e-6)
    assert not j["minimum_binds"]
    assert j["price_per_good_part_USD"] * 200 == pytest.approx(j["order_price_USD"])


def test_minimum_charge_binds_when_larger():
    j = job_cost.job_price("high")
    assert j["minimum_binds"] and j["order_price_USD"] == 750


def test_slower_cutting_raises_price():
    assert job_cost.job_price("central", cut_case="unfavourable")["order_price_USD"] > \
        job_cost.job_price("central")["order_price_USD"]


def test_break_even_lower_bound():
    die = die_cost.top_down("pierce_cutoff")["central_USD"]
    m = job_cost.job_price("central")["marginal_per_part_USD"]
    q = die_cost.break_even_parts(die, m, 0.0)
    assert q == pytest.approx(die / m)
    assert q > 200 * 10          # more than ten 200-part orders before the die pays back


def test_mass_units():
    assert units.convert(1, "lb", "g") == pytest.approx(453.59237)
    assert units.convert(1.0512, "lb/ft2", "lb/ft2") == pytest.approx(1.0512)
