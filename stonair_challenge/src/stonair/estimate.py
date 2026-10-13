"""Scoped material and electricity estimates for the selected operations.

OP-A  P002 top cover: fiber-laser cutting of the outer contour and 4 x Ø3.4
      holes from 2 mm 6061-T6 sheet.
OP-B  P007 end collector: pierce Ø5.5 + cut-off (or compound blank) from
      0.3 mm C11000 strip on a gap-frame press.
OP-C  P007 end collector, alternative route: fiber-laser cutting from 0.3 mm
      C11000 sheet (no die). Same model as OP-A with copper inputs.

All quantities carry explicit units in their names. Inputs come from three
places only: CAD (geometry), the evidence tables (sourced values) and
data/assumptions.csv (labelled assumptions and scenarios).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from . import units
from .evidence import machine_evidence, process_evidence
from .ingest import MODULES_REQUIRED, RAW, load_bom
from .step_geom import part_geometry

ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = ROOT / "data" / "assumptions.csv"
LEVELS = ("low", "central", "high")


def load_assumptions(path: Path = ASSUMPTIONS) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["parameter"] + "@" + r["operation"]: r for r in csv.DictReader(f)}


def params(op: str, level: str = "central", overrides: dict | None = None,
           path: Path = ASSUMPTIONS) -> dict:
    """Parameter set for an operation at one level, with optional overrides."""
    out = {}
    for key, r in load_assumptions(path).items():
        name, o = key.split("@")
        if o != op:
            continue
        v = r[level]
        try:
            out[name] = float(v)
        except ValueError:
            out[name] = v
    out.update(overrides or {})
    return out


def _density(evidence_id: str) -> float:
    row = next(r for r in process_evidence() if r["evidence_id"] == evidence_id)
    return row["normalized_low"]  # g/cm3


@dataclass
class Part:
    part_id: str
    qty_good: int
    net_volume_mm3: float
    thickness_mm: float
    x_mm: float
    y_mm: float
    outer_perimeter_mm: float
    hole_perimeter_mm: float
    hole_volume_mm3: float
    n_holes: int


def load_part(pid: str, raw: Path = RAW) -> Part:
    b = load_bom(raw)[pid]
    g = part_geometry(raw / "CAD" / "parts" / f"{pid}.step")
    x, y, t = g.extents
    return Part(pid, int(b.qty_per_parent * MODULES_REQUIRED), b.net_volume_mm3, b.thickness_mm, x, y,
                g.outer_perimeter(), g.hole_perimeter(),
                sum(math.pi * h.radius ** 2 * t for h in g.holes), len(g.holes))


def started_parts(good: int, reject_rate: float) -> int:
    if not 0 <= reject_rate < 1:
        raise ValueError("reject_rate must be in [0, 1)")
    return math.ceil(good / (1 - reject_rate) - 1e-9)


def nest_count(sheet_l: float, sheet_w: float, px: float, py: float, gap: float, margin: float) -> int:
    """Rectangular grid nest, best of two orientations."""
    def fit(a, b):
        ua, ub = sheet_l - 2 * margin, sheet_w - 2 * margin
        return max(0, math.floor((ua + gap) / (a + gap))) * max(0, math.floor((ub + gap) / (b + gap)))
    return max(fit(px, py), fit(py, px))


def _sheet(dim: str) -> tuple[float, float]:
    a, b = (float(v) for v in dim.lower().split("x"))
    return max(a, b), min(a, b)


# --------------------------------------------------------------------------- OP-A / OP-C
def op_a(p: dict, part: Part | None = None, density_g_cm3: float | None = None) -> dict:
    return laser_cut("OP-A", p, part or load_part("P002"), (density_g_cm3 or _density("M-01")))


def op_c(p: dict, part: Part | None = None, density_g_cm3: float | None = None) -> dict:
    return laser_cut("OP-C", p, part or load_part("P007"), (density_g_cm3 or _density("M-03")))


def laser_cut(op: str, p: dict, part: Part, density_g_cm3: float) -> dict:
    """Sheet nesting + cut-time x power model shared by the laser operations."""
    rho = density_g_cm3 * 1e-3  # g/mm3
    t = part.thickness_mm
    good = part.qty_good
    started = started_parts(good, p["reject_rate"])
    sl, sw = _sheet(p["sheet_size_mm"])
    per_sheet = nest_count(sl, sw, part.x_mm, part.y_mm, p["part_gap_mm"], p["edge_margin_mm"])
    if per_sheet == 0:
        raise ValueError("part does not fit the sheet")
    sheets = math.ceil(started / per_sheet)
    m_part = part.net_volume_mm3 * rho
    input_g = sheets * sl * sw * t * rho
    pitch_area = (part.x_mm + p["part_gap_mm"]) * (part.y_mm + p["part_gap_mm"])
    flows = {
        "retained_good_parts_g": good * m_part,
        "rejected_parts_g": (started - good) * m_part,
        "removed_inside_part_holes_g": started * part.hole_volume_mm3 * rho,
        "skeleton_between_parts_g": started * (pitch_area - part.x_mm * part.y_mm) * t * rho,
    }
    flows["sheet_remainder_g"] = input_g - sum(flows.values())
    kerf_g = started * (part.outer_perimeter_mm + part.hole_perimeter_mm) * p["kerf_width_mm"] * t * rho

    v = units.convert(p["contour_speed_m_min"], "m/min", "mm/s")
    n_feat = 1 + part.n_holes
    t_part_s = (part.outer_perimeter_mm / v
                + part.hole_perimeter_mm / (v * p["small_hole_speed_factor"])
                + n_feat * p["pierce_time_s"] + n_feat * p["move_time_per_feature_s"])
    t_cycle_h = started * t_part_s / 3600
    t_handle_h = sheets * p["sheet_handling_min_per_sheet"] / 60
    e_cut = t_cycle_h * p["processing_power_kW"]
    e_handle = t_handle_h * p["standby_power_kW"]
    return {
        "operation": op, "part_id": part.part_id, "good_parts": good, "started_parts": started,
        "parts_per_sheet": per_sheet, "sheets": sheets, "sheet_size_mm": p["sheet_size_mm"],
        "part_net_mass_g": m_part, "input_material_g": input_g, **flows,
        "kerf_vaporised_g (subset of holes+skeleton)": kerf_g,
        "material_utilisation_good_over_input": flows["retained_good_parts_g"] / input_g,
        # Remainder is reusable offcut; this ratio ignores it (nest-allocated basis).
        "material_utilisation_good_over_nest_used": flows["retained_good_parts_g"] / (input_g - flows["sheet_remainder_g"]),
        "cut_time_per_part_s": t_part_s, "in_cycle_time_h": t_cycle_h, "handling_time_h": t_handle_h,
        "energy_cutting_kWh": e_cut, "energy_handling_standby_kWh": e_handle,
        "energy_total_kWh": e_cut + e_handle, "energy_per_good_part_kWh": (e_cut + e_handle) / good,
    }


# --------------------------------------------------------------------------- OP-B
def op_b(p: dict, motor_kW: float, part: Part | None = None, density_g_cm3: float | None = None) -> dict:
    part = part or load_part("P007")
    rho = (density_g_cm3 or _density("M-03")) * 1e-3
    t = part.thickness_mm
    good = part.qty_good
    started = started_parts(good, p["reject_rate"])
    strokes = started + int(p["tryout_parts"])
    long_side, short_side = max(part.x_mm, part.y_mm), min(part.x_mm, part.y_mm)
    hole_perim = part.hole_perimeter_mm
    if p["route"] == "pierce_cutoff":
        strip_w, pitch = short_side, long_side
        shear_len = short_side + hole_perim  # one cut-off line + pierce per stroke
    elif p["route"] == "compound_blank":
        strip_w, pitch = long_side + 2 * p["web_mm"], short_side + p["web_mm"]
        shear_len = 2 * (long_side + short_side) + hole_perim
    else:
        raise ValueError(p["route"])
    strip_len = strokes * pitch + p["strip_end_scrap_mm"]
    input_g = strip_w * strip_len * t * rho
    m_part = part.net_volume_mm3 * rho
    flows = {
        "retained_good_parts_g": good * m_part,
        "rejected_parts_g": (started - good) * m_part,
        "tryout_parts_g": p["tryout_parts"] * m_part,
        "slugs_g": strokes * part.hole_volume_mm3 * rho,
    }
    flows["strip_web_and_ends_g"] = input_g - sum(flows.values())

    force_kN = shear_len * t * p["shear_strength_MPa"] * p["force_factor"] / 1000  # mm*mm*N/mm2 = N
    work_J = p["penetration_energy_factor"] * force_kN * 1000 * t / 1000
    run_min = strokes / p["parts_per_min"]
    on_min = run_min + p["motor_on_setup_min"]
    e_idle = motor_kW * p["idle_fraction_of_motor_rating"] * on_min / 60
    e_work = strokes * work_J / p["drive_efficiency"] / 3.6e6
    return {
        "operation": "OP-B", "part_id": part.part_id, "route": p["route"], "good_parts": good,
        "started_parts": started, "strokes": strokes, "strip_width_mm": strip_w, "pitch_mm": pitch,
        "strip_length_m": strip_len / 1000, "part_net_mass_g": m_part, "input_material_g": input_g, **flows,
        "material_utilisation_good_over_input": flows["retained_good_parts_g"] / input_g,
        "shear_length_per_stroke_mm": shear_len, "required_force_kN": force_kN, "work_per_stroke_J": work_J,
        "motor_rating_kW": motor_kW, "run_time_min": run_min, "motor_on_time_min": on_min,
        "energy_idle_kWh": e_idle, "energy_deformation_kWh": e_work,
        "energy_total_kWh": e_idle + e_work, "energy_per_good_part_kWh": (e_idle + e_work) / good,
        "naive_motor_rating_x_run_time_kWh": motor_kW * run_min / 60,
    }


def required_force_bounds() -> dict:
    """Worst-case force over both routes and the shear/force-factor range (for the capacity gate)."""
    hi = params("OP-B", "high")
    worst = 0.0
    for route in ("pierce_cutoff", "compound_blank"):
        r = op_b({**hi, "route": route}, motor_kW=1.0)
        worst = max(worst, r["required_force_kN"])
    return {"required_force_kN_worst": worst}


def press_motor_ratings() -> dict[str, float]:
    """Sourced main-motor ratings (kW) by candidate, applicable configs only."""
    out = {}
    for r in machine_evidence():
        if r["operation"] == "OP-B" and r["power_type"] == "motor_rating" and r["normalized_low"] is not None:
            out[r["candidate_id"]] = r["normalized_low"]
    return out


# --------------------------------------------------------------------------- sensitivity
NUMERIC_A = ["part_gap_mm", "edge_margin_mm", "kerf_width_mm", "contour_speed_m_min", "small_hole_speed_factor",
             "pierce_time_s", "move_time_per_feature_s", "processing_power_kW", "standby_power_kW",
             "sheet_handling_min_per_sheet", "reject_rate"]
NUMERIC_C = NUMERIC_A
NUMERIC_B = ["web_mm", "strip_end_scrap_mm", "tryout_parts", "reject_rate", "shear_strength_MPa", "force_factor",
             "penetration_energy_factor", "drive_efficiency", "idle_fraction_of_motor_rating", "parts_per_min",
             "motor_on_setup_min"]
# For these, the 'high' column is the favourable end for energy (faster = less time).
_INVERSE = {"contour_speed_m_min", "small_hole_speed_factor", "parts_per_min", "drive_efficiency"}


def _runner(op: str, motor_kW: float | None):
    return {"OP-A": op_a, "OP-C": op_c}.get(op) or (lambda p: op_b(p, motor_kW))


def sensitivity(op: str, motor_kW: float | None = None) -> list[dict]:
    base = params(op, "central")
    run = _runner(op, motor_kW)
    ref = run(base)
    rows = []
    for name in {"OP-A": NUMERIC_A, "OP-B": NUMERIC_B, "OP-C": NUMERIC_C}[op]:
        for level in ("low", "high"):
            v = params(op, level)[name]
            r = run({**base, name: v})
            rows.append({"operation": op, "parameter": name, "level": level, "value": v,
                         "energy_total_kWh": r["energy_total_kWh"],
                         "delta_energy_pct": 100 * (r["energy_total_kWh"] / ref["energy_total_kWh"] - 1),
                         "input_material_g": r["input_material_g"],
                         "delta_input_pct": 100 * (r["input_material_g"] / ref["input_material_g"] - 1)})
    cat = {"OP-A": ("sheet_size_mm", ("1000x2000", "1219x2438", "1250x2500")),
           "OP-B": ("route", ("pierce_cutoff", "compound_blank")),
           "OP-C": ("sheet_size_mm", ("600x600", "600x1500", "1000x2000"))}[op]
    for v in cat[1]:
        r = run({**base, cat[0]: v})
        rows.append({"operation": op, "parameter": cat[0], "level": "scenario", "value": v,
                     "energy_total_kWh": r["energy_total_kWh"],
                     "delta_energy_pct": 100 * (r["energy_total_kWh"] / ref["energy_total_kWh"] - 1),
                     "input_material_g": r["input_material_g"],
                     "delta_input_pct": 100 * (r["input_material_g"] / ref["input_material_g"] - 1)})
    return rows


def envelope(op: str, motor_kW: float | None = None) -> dict:
    """All parameters at their energy-favourable vs unfavourable ends together."""
    lo, hi = params(op, "low"), params(op, "high")
    fav = {k: (hi[k] if k in _INVERSE else lo[k]) for k in lo if isinstance(lo[k], float)}
    unf = {k: (lo[k] if k in _INVERSE else hi[k]) for k in lo if isinstance(lo[k], float)}
    base = params(op, "central")
    run = _runner(op, motor_kW)
    return {"favourable": run({**base, **fav}), "central": run(base), "unfavourable": run({**base, **unf})}
