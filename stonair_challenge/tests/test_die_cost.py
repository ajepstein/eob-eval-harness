import statistics

import pytest

from stonair import die_cost
from stonair.estimate import params
from stonair.evidence import process_evidence


def _ev():
    return {r["evidence_id"]: r for r in process_evidence() if r["topic"] == "die_cost"}


def test_top_down_is_min_median_max_of_cited_ranges():
    ev = _ev()
    ids = die_cost.DIE_CLASSES["pierce_cutoff"]["evidence"]
    r = die_cost.top_down("pierce_cutoff")
    assert r["low_USD"] == min(ev[i]["normalized_low"] for i in ids) == 1000
    assert r["high_USD"] == max(ev[i]["normalized_high"] for i in ids) == 10000
    assert r["central_USD"] == statistics.median(
        (ev[i]["normalized_low"] + ev[i]["normalized_high"]) / 2 for i in ids) == 4000
    assert r["evidence_ids"].split(";") == ids


def test_compound_die_costs_more_than_simple_die():
    simple, compound = die_cost.top_down("pierce_cutoff"), die_cost.top_down("compound_blank")
    assert compound["central_USD"] > simple["central_USD"]
    assert compound["high_USD"] > simple["high_USD"]


def test_bottom_up_hand_calculation():
    # central: die set 723.67 + components 600 + 50 h x 29.82 USD/h x 3.0
    assert die_cost.bottom_up_level("central") == pytest.approx(723.67 + 600 + 50 * 29.82 * 3.0)
    assert die_cost.bottom_up_level("low") < die_cost.bottom_up_level("central") < die_cost.bottom_up_level("high")


def test_methods_overlap():
    td, bu = die_cost.top_down("pierce_cutoff"), die_cost.bottom_up()
    assert bu["low_USD"] < td["high_USD"] and td["low_USD"] < bu["high_USD"]


def test_die_set_scenario_traces_to_catalog_evidence():
    ev = _ev()
    assert params("OP-B", "low")["die_set_USD"] == ev["D-10"]["normalized_low"]
    assert params("OP-B", "central")["die_set_USD"] == ev["D-11"]["normalized_low"]
    assert params("OP-B", "high")["die_set_USD"] == ev["D-11"]["normalized_high"]


def test_per_part_and_break_even():
    assert die_cost.per_part(4000, 200, 1) == 20
    assert die_cost.per_part(4000, 200, 10) == 2
    with pytest.raises(ValueError):
        die_cost.per_part(4000, 0, 1)
    ev = _ev()["D-13"]   # laser-vendor busbar example: $2,500 die, $0.25 laser vs $0.15 stamped
    assert die_cost.break_even_parts(2500, ev["normalized_high"], ev["normalized_low"]) == pytest.approx(25000)
    assert die_cost.break_even_parts(2500, 0.15, 0.25) is None


def test_currency_units():
    from stonair import units
    assert units.convert(29.82, "USD/h", "USD/h") == 29.82
    with pytest.raises(units.UnitError):
        units.convert(1, "USD", "kWh")
