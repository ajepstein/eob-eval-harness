"""Die cost estimate for the P007 press route (OP-B).

Two independent methods, reported side by side rather than blended:

* top-down: published price ranges for the die class (vendor and supplier
  pages, evidence D-01..D-09). low = smallest published low, high = largest
  published high, central = median of the range midpoints;
* bottom-up: catalog die set (D-10, D-11) + components + toolmaker hours x
  shop rate, where shop rate = sourced wage (D-12) x a burden multiplier
  (assumption B-15). Built only for the pierce + cut-off die.

All values are nominal USD as published (mixed US / China suppliers, no FX or
inflation adjustment). None of this is a quote.
"""
from __future__ import annotations

import statistics

from .estimate import params
from .evidence import process_evidence

# Which evidence rows describe which die class.
DIE_CLASSES = {
    "pierce_cutoff": {
        "label": "simple die (single-station / 2-station pierce + cut-off strip die)",
        "evidence": ["D-01", "D-03", "D-06", "D-07", "D-08", "D-09"],
    },
    "compound_blank": {
        "label": "compound blanking die (outline + hole in one stroke)",
        "evidence": ["D-02", "D-04", "D-05"],
    },
}


def _ev() -> dict[str, dict]:
    return {r["evidence_id"]: r for r in process_evidence() if r["topic"] == "die_cost"}


def top_down(route: str) -> dict:
    ev = _ev()
    rows = [ev[i] for i in DIE_CLASSES[route]["evidence"]]
    lows = [r["normalized_low"] for r in rows]
    highs = [r["normalized_high"] for r in rows]
    mids = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
    return {"route": route, "method": "top_down", "die_class": DIE_CLASSES[route]["label"],
            "low_USD": min(lows), "central_USD": statistics.median(mids), "high_USD": max(highs),
            "n_sources": len({r["source_id"] for r in rows}), "evidence_ids": ";".join(r["evidence_id"] for r in rows)}


def bottom_up_level(level: str) -> float:
    p = params("OP-B", level)
    wage = _ev()["D-12"]["normalized_low"]
    return p["die_set_USD"] + p["die_components_USD"] + p["toolmaker_hours"] * wage * p["shop_rate_burden_multiplier"]


def bottom_up() -> dict:
    return {"route": "pierce_cutoff", "method": "bottom_up",
            "die_class": DIE_CLASSES["pierce_cutoff"]["label"],
            "low_USD": bottom_up_level("low"), "central_USD": bottom_up_level("central"),
            "high_USD": bottom_up_level("high"), "n_sources": 3, "evidence_ids": "D-10;D-11;D-12"}


def per_part(die_USD: float, good_parts_per_order: int, orders: float) -> float:
    if good_parts_per_order <= 0 or orders <= 0:
        raise ValueError("parts and orders must be positive")
    return die_USD / (good_parts_per_order * orders)


def break_even_parts(die_USD: float, c_laser_USD: float, c_press_USD: float) -> float | None:
    """Q* = die / (c_laser - c_press). None when the press is not cheaper per part."""
    gap = c_laser_USD - c_press_USD
    return die_USD / gap if gap > 0 else None


def summary(good_parts_per_order: int = 200) -> list[dict]:
    rows = [top_down("pierce_cutoff"), bottom_up(), top_down("compound_blank")]
    orders = params("OP-B")["orders_amortised"], params("OP-B", "high")["orders_amortised"]
    for r in rows:
        r["central_USD_per_good_part_1_order"] = per_part(r["central_USD"], good_parts_per_order, 1)
        r["central_USD_per_good_part_%d_orders" % orders[0]] = per_part(r["central_USD"], good_parts_per_order, orders[0])
        r["central_USD_per_good_part_%d_orders" % orders[1]] = per_part(r["central_USD"], good_parts_per_order, orders[1])
    return rows
