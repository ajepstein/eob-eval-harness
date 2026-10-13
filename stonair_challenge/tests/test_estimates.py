import itertools
import math

import pytest

from stonair.estimate import (envelope, load_part, nest_count, op_a, op_b, params, press_motor_ratings,
                              started_parts)

FLOWS_A = ["retained_good_parts_g", "rejected_parts_g", "removed_inside_part_holes_g",
           "skeleton_between_parts_g", "sheet_remainder_g"]
FLOWS_B = ["retained_good_parts_g", "rejected_parts_g", "tryout_parts_g", "slugs_g", "strip_web_and_ends_g"]


@pytest.mark.parametrize("level,sheet", list(itertools.product(("low", "central", "high"),
                                                               ("1000x2000", "1219x2438", "1250x2500"))))
def test_op_a_mass_balance_closes(level, sheet):
    r = op_a(params("OP-A", level, {"sheet_size_mm": sheet}))
    assert sum(r[k] for k in FLOWS_A) == pytest.approx(r["input_material_g"], rel=1e-12)
    assert all(r[k] >= -1e-9 for k in FLOWS_A)       # no negative flow means the nest is physically possible
    assert r["retained_good_parts_g"] < r["input_material_g"]


@pytest.mark.parametrize("level,route", list(itertools.product(("low", "central", "high"),
                                                               ("pierce_cutoff", "compound_blank"))))
def test_op_b_mass_balance_closes(level, route):
    r = op_b(params("OP-B", level, {"route": route}), motor_kW=5.5)
    assert sum(r[k] for k in FLOWS_B) == pytest.approx(r["input_material_g"], rel=1e-12)
    assert all(r[k] >= -1e-9 for k in FLOWS_B)


def test_retention_separate_from_rejection():
    r = op_a(params("OP-A", overrides={"reject_rate": 0.05}))
    assert r["started_parts"] == 106 and r["good_parts"] == 100
    assert r["rejected_parts_g"] == pytest.approx(6 * r["part_net_mass_g"])
    assert r["retained_good_parts_g"] == pytest.approx(100 * r["part_net_mass_g"])


def test_started_parts():
    assert started_parts(100, 0) == 100
    assert started_parts(100, 0.02) == 103      # 100/0.98 = 102.04 -> 103
    assert started_parts(200, 0.05) == 211      # 210.5 -> 211
    with pytest.raises(ValueError):
        started_parts(100, 1.0)


def test_nest_count_hand_case():
    # 1219 x 2438, margin 10, gap 5, part 172 x 124: (2418+5)//177=13, (1199+5)//129=9 -> 117
    assert nest_count(2438, 1219, 172, 124, 5, 10) == 117
    assert nest_count(100, 100, 172, 124, 5, 10) == 0


def test_part_mass_from_volume_and_density():
    r = op_a(params("OP-A"))
    assert r["part_net_mass_g"] == pytest.approx(42583.366377849015 * 2.7e-3)
    b = op_b(params("OP-B"), 5.5)
    assert b["part_net_mass_g"] == pytest.approx(374.47251166716813 * 8.89e-3)


def test_op_a_energy_hand_calculation():
    p = params("OP-A")
    part = load_part("P002")
    v = 9 * 1000 / 60  # mm/s
    t = 592 / v + 4 * math.pi * 3.4 / (v * 0.35) + 5 * 0.3 + 5 * 0.4
    r = op_a(p)
    assert r["cut_time_per_part_s"] == pytest.approx(t)
    assert r["energy_cutting_kWh"] == pytest.approx(103 * t / 3600 * 12)
    assert r["energy_handling_standby_kWh"] == pytest.approx(1 * 5 / 60 * 5)
    assert r["energy_per_good_part_kWh"] * 100 == pytest.approx(r["energy_total_kWh"])
    assert part.outer_perimeter_mm == 592


def test_op_a_energy_is_monotonic():
    base = params("OP-A")
    e = lambda **kw: op_a({**base, **kw})["energy_total_kWh"]
    assert e(processing_power_kW=18) > e(processing_power_kW=12) > e(processing_power_kW=10)
    assert e(contour_speed_m_min=4) > e(contour_speed_m_min=9) > e(contour_speed_m_min=16.5)


def test_op_b_force_and_work_hand_calculation():
    r = op_b(params("OP-B"), 5.5)
    L = 12 + math.pi * 5.5
    assert r["shear_length_per_stroke_mm"] == pytest.approx(L)
    assert r["required_force_kN"] == pytest.approx(L * 0.3 * 190 * 1.2 / 1000)
    assert r["work_per_stroke_J"] == pytest.approx(0.5 * r["required_force_kN"] * 1000 * 0.3e-3)
    c = op_b(params("OP-B", overrides={"route": "compound_blank"}), 5.5)
    assert c["shear_length_per_stroke_mm"] == pytest.approx(2 * (12 + 106) + math.pi * 5.5)


def test_op_b_idle_dominates_and_differs_from_naive():
    r = op_b(params("OP-B"), 5.5)
    assert r["energy_idle_kWh"] > 100 * r["energy_deformation_kWh"]
    assert r["energy_total_kWh"] != pytest.approx(r["naive_motor_rating_x_run_time_kWh"])


def test_capacity_gate_inputs_are_far_below_press_ratings():
    from stonair.estimate import required_force_bounds
    assert required_force_bounds()["required_force_kN_worst"] < 0.15 * 200   # smallest candidate is 200 kN


def test_envelopes_are_ordered():
    for op, kw in (("OP-A", None), ("OP-B", press_motor_ratings()["B1"])):
        e = envelope(op, kw)
        assert e["favourable"]["energy_total_kWh"] < e["central"]["energy_total_kWh"] < \
            e["unfavourable"]["energy_total_kWh"]
