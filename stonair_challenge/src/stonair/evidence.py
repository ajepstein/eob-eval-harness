"""Evidence loading, unit normalisation and attribution checks."""
from __future__ import annotations

import csv
from pathlib import Path

from . import units

EVID = Path(__file__).resolve().parents[2] / "data" / "evidence"

POWER_TYPES = {
    "rated_laser_output": "optical output of the laser source; not electrical draw",
    "motor_rating": "nameplate rating of the main drive motor; not the power drawn",
    "installed_connected_load": "supply/connection rating (often kVA); sizes the feeder, not consumption",
    "max_operating_power": "maximum measured/declared draw in operation",
    "average_operating_power": "average draw over a defined production cycle",
    "operating_power_range": "range of operating draw, basis unstated",
    "operating_power_unspecified": "a 'consumption' figure whose basis (installed vs average) is not stated",
    "energy_per_cycle": "energy per stroke/part",
    "relative_claim": "percentage claim without an absolute baseline; not a power value",
    "n/a": "not a power parameter",
}
# What each power type can legitimately be used for in an estimate.
ENERGY_EVIDENCE_SCORE = {
    "energy_per_cycle": 4, "average_operating_power": 3, "max_operating_power": 2,
    "operating_power_range": 1.5, "installed_connected_load": 1, "motor_rating": 1,
    "operating_power_unspecified": 0.5, "rated_laser_output": 0, "relative_claim": 0, "n/a": 0,
}
TIER_SCORE = {
    "OEM": 3, "OEM_document_distributor_hosted": 2.5, "OEM_resale_listing": 1.5,
    "distributor": 1.5, "directory": 1.5, "peer_reviewed_research": 2, "materials_database": 2,
    "resale_listing": 1, "third_party": 0.5, "supplied_by_stonair": 3,
}
NOT_SCOPED = "not the scoped config"


def read(name: str, base: Path = EVID) -> list[dict[str, str]]:
    with open(base / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(s: str) -> float | None:
    return units.parse_blank(s)


def normalise_row(row: dict[str, str]) -> dict:
    """Add normalized_low/high + normalized_unit. Non-numeric rows stay blank."""
    out = dict(row)
    unit = row["original_unit"]
    lo, hi = _num(row["value_low"]), _num(row["value_high"])
    if unit == "text" or lo is None:
        out.update(normalized_low=None, normalized_high=None, normalized_unit=None)
        return out
    dim = units.dimension(unit)
    base = units.BASE_UNIT[dim]
    out.update(normalized_low=units.to_base(lo, unit), normalized_high=units.to_base(hi, unit),
               normalized_unit=base)
    return out


def machine_evidence(base: Path = EVID) -> list[dict]:
    return [normalise_row(r) for r in read("machine_evidence_raw.csv", base)]


def process_evidence(base: Path = EVID) -> list[dict]:
    return [normalise_row(r) for r in read("process_material_evidence.csv", base)]


def config_applicable(row: dict) -> bool:
    return NOT_SCOPED not in row["configuration"]


def validate(base: Path = EVID) -> list[str]:
    """Return a list of attribution / consistency problems (empty = clean)."""
    problems = []
    sources = {s["source_id"]: s for s in read("sources.csv", base)}
    cands = {c["candidate_id"]: c for c in read("machine_candidates.csv", base)}
    searches = {s["search_id"] for s in read("search_log.csv", base)}
    for s in sources.values():
        if s["source_tier"] not in TIER_SCORE:
            problems.append(f"{s['source_id']}: unknown tier {s['source_tier']}")
        if not s["url"].startswith("https://"):
            problems.append(f"{s['source_id']}: url not https")
        if not s["access_date"]:
            problems.append(f"{s['source_id']}: no access date")
    seen = set()
    for r in read("machine_evidence_raw.csv", base):
        eid = r["evidence_id"]
        if eid in seen:
            problems.append(f"{eid}: duplicate id")
        seen.add(eid)
        if r["source_id"] not in sources:
            problems.append(f"{eid}: source {r['source_id']} not in sources.csv")
        c = cands.get(r["candidate_id"])
        if c is None:
            problems.append(f"{eid}: candidate {r['candidate_id']} not in machine_candidates.csv")
        elif r["manufacturer"] != c["manufacturer"] or r["model"] not in (
                [c["model"]] + [a for a in c["model_aliases"].split(";") if a]):
            problems.append(f"{eid}: machine identity {r['manufacturer']} {r['model']} != candidate "
                            f"{c['manufacturer']} {c['model']}")
        for f in ("excerpt", "evidence_location", "applicability", "uncertainty", "configuration"):
            if not r[f].strip():
                problems.append(f"{eid}: empty {f}")
        if r["power_type"] not in POWER_TYPES:
            problems.append(f"{eid}: unknown power_type {r['power_type']}")
        unit = r["original_unit"]
        if unit != "text":
            dim = units.dimension(unit)
            if dim in ("power", "apparent_power") and r["power_type"] == "n/a":
                problems.append(f"{eid}: power value without a power_type")
            if dim == "apparent_power" and r["power_type"] not in ("installed_connected_load",):
                problems.append(f"{eid}: kVA must be classed as installed/connected load")
            if r["power_type"] == "rated_laser_output" and dim != "power":
                problems.append(f"{eid}: rated laser output not in a power unit")
            lo, hi = _num(r["value_low"]), _num(r["value_high"])
            if lo is None or hi is None:
                problems.append(f"{eid}: numeric unit but blank value")
            # every number in value_low must appear in the original text (anti-transcription check)
            elif not _appears(lo, r["original_value"]):
                problems.append(f"{eid}: value_low {lo} not found in original_value {r['original_value']!r}")
    for r in read("process_material_evidence.csv", base):
        if r["source_id"] not in sources:
            problems.append(f"{r['evidence_id']}: source {r['source_id']} not in sources.csv")
    for r in read("rejected_leads.csv", base):
        if r["search_id"] not in searches:
            problems.append(f"rejected lead {r['lead']!r}: search {r['search_id']} not in search_log.csv")
    return problems


def _appears(x: float, text: str) -> bool:
    import re
    nums = [float(n) for n in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))]
    return any(abs(n - x) < 1e-9 for n in nums)
