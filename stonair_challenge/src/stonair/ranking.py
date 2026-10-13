"""Compatibility gates and ranking of machine candidates.

Gates decide eligibility (pass / pass_inferred / unknown / fail). Ranking
then orders eligible candidates by evidence usefulness, not by brand:

    score = 2 x energy-evidence score + source-tier score + scale fit

within status bands (eligible > eligible_unverified > secondary evidence only).

Energy evidence counts double because the deliverable is an energy estimate;
an average operating power for the scoped configuration is worth more than a
connection rating, which is worth more than nothing.
"""
from __future__ import annotations

from .estimate import load_part, params, required_force_bounds
from .evidence import (ENERGY_EVIDENCE_SCORE, TIER_SCORE, config_applicable, machine_evidence, read)


# Fully verified gates outrank unverified gates, which outrank resale-only support.
STATUS_ORDER = {"eligible": 0, "eligible_unverified": 1, "eligible_secondary_evidence_only": 2}


def _rows(cid: str, ev: list[dict]) -> list[dict]:
    return [r for r in ev if r["candidate_id"] == cid]


def _find(rows, *words):
    return [r for r in rows if all(w in r["parameter"].lower() for w in words) and r["normalized_low"] is not None]


def _gate_value(found, pick):
    if not found:
        return None, None
    best = pick(found)
    return best, best["evidence_id"]


def rank() -> list[dict]:
    ev = machine_evidence()
    sources = {s["source_id"]: s for s in read("sources.csv")}
    cands = read("machine_candidates.csv")
    p002, p007 = load_part("P002"), load_part("P007")
    f_req = required_force_bounds()["required_force_kN_worst"]
    die = sorted((float(v) for v in params("OP-B", "high")["die_footprint_mm"].split("x")), reverse=True)
    sheet_central = sorted((float(v) for v in params("OP-A")["sheet_size_mm"].split("x")), reverse=True)
    out = []
    for c in cands:
        cid, op = c["candidate_id"], c["operation"]
        rows = [r for r in _rows(cid, ev) if config_applicable(r)]
        gates: dict[str, str] = {}
        refs: list[str] = []
        if op == "OP-A":
            th, ref = _gate_value(_find(rows, "thickness", "alumin"), lambda f: max(f, key=lambda r: r["normalized_high"]))
            if th is None:
                gates["material_thickness"] = "unknown"
            else:
                ok = th["normalized_high"] >= p002.thickness_mm
                tier = sources[th["source_id"]]["source_tier"]
                gates["material_thickness"] = ("pass" if TIER_SCORE[tier] >= 1.5 else "pass_resale_only") if ok else "fail"
                refs.append(ref)
            area = [r for r in rows if any(k in r["parameter"] for k in ("working range", "axis travel", "sheet size"))]
            if area:
                a = sorted((area[0]["normalized_low"], area[0]["normalized_high"]), reverse=True)
                gates["envelope"] = "pass" if a[0] >= sheet_central[0] and a[1] >= sheet_central[1] else "fail"
                refs.append(area[0]["evidence_id"])
            elif any(k in c["model"] for k in ("3015", "1530")):
                gates["envelope"] = "pass_inferred"  # model designation implies a 3000 x 1500 mm bed
            else:
                gates["envelope"] = "unknown"
            gates["tooling"] = "pass"  # no part-specific tooling for laser cutting
        else:
            cap = _find(rows, "capacity") + _find(rows, "press force")
            if cap:
                gates["force_capacity"] = "pass" if cap[0]["normalized_low"] >= f_req else "fail"
                refs.append(cap[0]["evidence_id"])
            else:
                gates["force_capacity"] = "unknown"
            bol = _find(rows, "bolster")
            if bol:
                b = sorted((bol[0]["normalized_low"], bol[0]["normalized_high"]), reverse=True)
                gates["bolster"] = "pass" if b[0] >= die[0] and b[1] >= die[1] else "fail"
                refs.append(bol[0]["evidence_id"])
            else:
                gates["bolster"] = "unknown"
            gates["tooling"] = "requires_custom_die"
        # Only numeric, normalised values count: an uninterpreted text field is not evidence of power.
        energy_rows = [r for r in rows if r["power_type"] != "n/a" and r["normalized_low"] is not None]
        best_energy = max((ENERGY_EVIDENCE_SCORE[r["power_type"]] for r in energy_rows), default=0)
        best_energy_type = max(energy_rows, key=lambda r: ENERGY_EVIDENCE_SCORE[r["power_type"]])["power_type"] \
            if energy_rows else "none"
        tiers = [sources[r["source_id"]]["source_tier"] for r in rows]
        tier_score = max((TIER_SCORE[t] for t in tiers), default=0)
        oem = any(t in ("OEM", "OEM_document_distributor_hosted") for t in tiers)
        vals = set(gates.values()) - {"requires_custom_die"}
        if "fail" in vals:
            status = "excluded_incompatible"
        elif not rows or tier_score < 1:
            status = "excluded_insufficient_evidence"
        elif "unknown" in vals:
            status = "eligible_unverified"
        elif "pass_resale_only" in vals or not oem:
            status = "eligible_secondary_evidence_only"
        else:
            status = "eligible"
        score = 2 * best_energy + tier_score + float(c["scale_fit"])
        out.append({
            "operation": op, "candidate_id": cid, "manufacturer": c["manufacturer"], "model": c["model"],
            "configuration": c["configuration"], "status": status,
            **{f"gate_{k}": v for k, v in gates.items()},
            "best_energy_evidence": best_energy_type, "energy_score": best_energy,
            "best_source_tier_score": tier_score, "has_oem_evidence": oem, "scale_fit": float(c["scale_fit"]),
            "score": score if status.startswith("eligible") else None,
            "gate_evidence_ids": ";".join(refs), "scale_note": c["scale_note"], "tooling_note": c["tooling_note"],
        })
    for op in ("OP-A", "OP-B"):
        el = sorted((r for r in out if r["operation"] == op and r["score"] is not None),
                    key=lambda r: (STATUS_ORDER[r["status"]], -r["score"], r["candidate_id"]))
        for i, r in enumerate(el, 1):
            r["rank"] = i
    for r in out:
        r.setdefault("rank", None)
    return out
