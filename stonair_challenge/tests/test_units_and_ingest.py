import math

import pytest

from stonair import units
from stonair.ingest import load_bom, load_instances, reconcile
from stonair.step_geom import fix_mojibake, part_geometry
from stonair.ingest import RAW


# ---------------------------------------------------------------- unit conversion
@pytest.mark.parametrize("value,unit,target,expected", [
    (0.2, "cm", "mm", 2.0),
    (300, "µm", "mm", 0.3),     # U+00B5 micro sign, as in bom.csv
    (300, "μm", "mm", 0.3),     # U+03BC Greek mu
    (0.320, "in", "mm", 8.128),
    (6.7, "hp", "kW", 4.99619),
    (3000, "W", "kW", 3.0),
    (9, "m/min", "mm/s", 150.0),
    (25, "US ton", "kN", 222.411),
    (21.8, "ksi", "MPa", 150.306),
    (0.321, "lb/in3", "g/cm3", 8.885),
    (1800, "kJ", "kWh", 0.5),
])
def test_conversions(value, unit, target, expected):
    assert units.convert(value, unit, target) == pytest.approx(expected, rel=1e-4)


def test_kva_is_not_kw():
    with pytest.raises(units.UnitError):
        units.convert(58, "kVA", "kW")


def test_unknown_unit_raises():
    with pytest.raises(units.UnitError):
        units.to_base(1, "furlong")


def test_blank_is_none_not_zero():
    assert units.parse_blank("") is None
    assert units.parse_blank("  ") is None
    assert units.parse_blank(None) is None
    assert units.convert(None, "cm", "mm") is None
    assert units.parse_blank("0") == 0.0


def test_bom_micro_sign_is_present_and_handled():
    raw = (RAW / "bom.csv").read_bytes()
    assert "µm".encode() in raw
    assert load_bom()["P007"].thickness_mm == pytest.approx(0.3)


# ---------------------------------------------------------------- ingest / reconciliation
def test_blank_bom_cells_stay_missing():
    bom = load_bom()
    assert bom["P001"].thickness_mm is None
    assert bom["P002"].purchased_unit_mass_g is None
    assert bom["P005"].purchased_unit_mass_g == 69.0


def test_mixed_thickness_units_normalise():
    bom = load_bom()
    assert bom["P002"].thickness_stated_unit == "cm"
    assert bom["P002"].thickness_mm == pytest.approx(2.0)
    assert all(bom[p].thickness_mm == pytest.approx(0.3) for p in ("P006", "P007", "P008"))


def test_quantity_basis_per_parent_times_modules():
    bom = load_bom()
    assert bom["P002"].qty_total_demand == 100
    assert bom["P007"].qty_total_demand == 200
    assert bom["P005"].qty_total_demand == 2400
    assert sum(b.qty_per_parent for b in bom.values()) == len(load_instances()["instances"]) == 53


def test_reconciliation_has_no_failures():
    checks = reconcile()
    fails = [c for c in checks if c.status == "fail"]
    assert not fails, fails
    ids = {c.check_id for c in checks}
    assert {"C01", "C03", "C04", "C06b", "C06c", "C06d", "C07"} <= ids


def test_reconciliation_flags_known_issues():
    checks = {(c.check_id, c.part_id): c for c in reconcile()}
    assert checks[("C05", "P009")].status == "flag"           # double-encoded ×
    assert checks[("C05", "P007")].status == "flag"           # name differs between BOM and STEP
    assert checks[("C06d", "P008")].status == "flag"          # non-circular cut-outs, not a prism with holes
    assert checks[("C10", "P007-001")].status == "flag"       # collector not touching cell


def test_mojibake_repair():
    broken = "M3 ×".encode("utf-8").decode("latin-1").encode("utf-8").decode("latin-1")
    fixed, rounds = fix_mojibake(broken)
    assert fixed == "M3 ×" and rounds == 2
    assert fix_mojibake("plain") == ("plain", 0)


# ---------------------------------------------------------------- STEP geometry
def test_p002_geometry_from_step():
    g = part_geometry(RAW / "CAD" / "parts" / "P002.step")
    assert g.unit == "mm"
    assert g.extents == (172.0, 124.0, 2.0)
    assert [(h.x, h.y) for h in g.holes] == [(6, 6), (6, 118), (166, 6), (166, 118)]
    assert all(h.diameter == pytest.approx(3.4) for h in g.holes)
    assert g.prism_volume() == pytest.approx(load_bom()["P002"].net_volume_mm3, rel=1e-12)


def test_p007_geometry_from_step():
    g = part_geometry(RAW / "CAD" / "parts" / "P007.step")
    assert g.extents == (12.0, 106.0, 0.3)
    assert len(g.holes) == 1 and g.holes[0].diameter == pytest.approx(5.5)
    assert g.prism_volume() == pytest.approx(load_bom()["P007"].net_volume_mm3, rel=1e-12)
    assert g.outer_perimeter() + g.hole_perimeter() == pytest.approx(2 * 118 + math.pi * 5.5)
