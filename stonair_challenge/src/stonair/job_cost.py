"""Laser job-shop price for P007 (OP-C), built bottom-up.

No instant quote could be obtained: the online services need a CAD upload,
and none of the three checked lists 0.3 mm copper (J-01, J-02, J-04). The
price is therefore assembled from published shop parameters:

    price = max(minimum charge,
                setup fee + machine hours x laser rate + sheet mass x copper price x markup)

Machine hours and sheet mass come from the OP-C model (estimate.op_c), so
the job price and the energy estimate describe the same cutting job.
Values are nominal USD as published; none of this is a quote.
"""
from __future__ import annotations

import statistics

from . import units
from .estimate import envelope, op_c, params
from .evidence import process_evidence

RATE_EVIDENCE = ["J-06", "J-07", "J-08", "J-09", "J-10"]


def _ev() -> dict[str, dict]:
    return {r["evidence_id"]: r for r in process_evidence() if r["topic"] == "laser_job_cost"}


def rate_range() -> dict:
    """Laser hourly rate: min of lows, median of midpoints, max of highs (USD/h)."""
    ev = _ev()
    rows = [ev[i] for i in RATE_EVIDENCE]
    return {"low": min(r["normalized_low"] for r in rows),
            "central": statistics.median((r["normalized_low"] + r["normalized_high"]) / 2 for r in rows),
            "high": max(r["normalized_high"] for r in rows)}


def sheet_price_per_lb() -> dict:
    """Copper sheet USD/lb derived from listed price and weight (J-17/J-18 full sheet, J-19/J-20 small sheet)."""
    ev = _ev()
    full_sheet_ft2 = 36 * 120 / 144  # the listed 36 x 120 in sheet
    return {"full_sheet_036x120in": ev["J-17"]["normalized_low"] / (float(ev["J-18"]["value_low"]) * full_sheet_ft2),
            "small_sheet_12x24in": ev["J-19"]["normalized_low"] / float(ev["J-20"]["value_low"]),
            "spot_floor": ev["J-21"]["normalized_low"] * units.convert(1, "lb", "g")}


def thin_copper_availability() -> list[dict]:
    """Thinnest copper each checked online service lists, vs P007's 0.3 mm."""
    ev = _ev()
    out = []
    for eid, service in (("J-01", "SendCutSend"), ("J-02", "OSH Cut"), ("J-04", "Ponoko")):
        t = ev[eid]["normalized_low"]
        out.append({"service": service, "thinnest_copper_mm": t, "offers_0.3_mm": t <= 0.3 + 1e-9,
                    "evidence_id": eid})
    return out


def job_price(level: str = "central", overrides: dict | None = None, cut_case: str = "central") -> dict:
    """Price inputs at `level`; the cutting job (time, sheet) at `cut_case` of the OP-C envelope."""
    p = params("OP-C", level, overrides)
    c = op_c(params("OP-C")) if cut_case == "central" else envelope("OP-C")[cut_case]
    machine_h = c["in_cycle_time_h"] + c["handling_time_h"]
    sheet_lb = units.convert(c["input_material_g"], "g", "lb")
    machine_USD = machine_h * p["laser_rate_USD_h"]
    material_USD = sheet_lb * p["copper_sheet_USD_lb"] * p["material_markup"]
    build_up = p["setup_fee_USD"] + machine_USD + material_USD
    price = max(p["minimum_charge_USD"], build_up)
    good = c["good_parts"]
    return {"level": level, "cut_case": cut_case, "good_parts": good, "machine_h": machine_h, "sheet_lb": sheet_lb,
            "laser_rate_USD_h": p["laser_rate_USD_h"], "setup_fee_USD": p["setup_fee_USD"],
            "minimum_charge_USD": p["minimum_charge_USD"], "machine_USD": machine_USD,
            "material_USD": material_USD, "build_up_USD": build_up, "minimum_binds": price > build_up,
            "order_price_USD": price, "price_per_good_part_USD": price / good,
            "marginal_per_part_USD": (machine_USD + material_USD) / good}
