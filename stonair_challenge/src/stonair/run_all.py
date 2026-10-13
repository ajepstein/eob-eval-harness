"""Regenerate every output in outputs/ from data/. Usage: python -m stonair.run_all"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .estimate import envelope, op_a, op_b, params, press_motor_ratings, sensitivity
from .evidence import machine_evidence, process_evidence, validate
from .ingest import DATA_DICTIONARY, normalised_bom_rows, reconcile
from .ranking import rank

OUT = Path(__file__).resolve().parents[2] / "outputs"


def _write(name: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else _fmt(r.get(k))) for k in keys})


def _fmt(v):
    return round(v, 6) if isinstance(v, float) else v


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    problems = validate()
    if problems:
        raise SystemExit("evidence validation failed:\n" + "\n".join(problems))

    _write("normalized_bom.csv", normalised_bom_rows())
    _write("data_dictionary.csv", [dict(zip(("field", "source_field", "meaning", "unit", "quantity_basis",
                                              "ambiguity"), d)) for d in DATA_DICTIONARY])
    checks = [asdict(c) for c in reconcile()]
    _write("reconciliation_checks.csv", checks)
    _write("machine_evidence.csv", machine_evidence())
    _write("process_material_evidence.csv", process_evidence())
    ranking = rank()
    _write("candidate_ranking.csv", sorted(ranking, key=lambda r: (r["operation"], r["rank"] or 99)))

    motors = press_motor_ratings()
    top_b = next(r["candidate_id"] for r in ranking if r["operation"] == "OP-B" and r["rank"] == 1)
    a_c = op_a(params("OP-A"))
    b_c = op_b(params("OP-B"), motors[top_b])
    _write("estimates_central.csv", [{"basis": "OP-A central, sourced 3 kW power scenario (A1/A2 evidence)", **a_c},
                                     {"basis": f"OP-B central, motor rating of rank-1 press {top_b}", **b_c}])
    by_cand = []
    for cid, kw in sorted(motors.items()):
        r = op_b(params("OP-B"), kw)
        by_cand.append({"candidate_id": cid, "motor_rating_kW": kw,
                        "note": "servo: idle-fraction assumption is for flywheel presses and likely overstates"
                                if cid == "B3" else "",
                        "energy_total_kWh": r["energy_total_kWh"], "energy_per_good_part_kWh": r["energy_per_good_part_kWh"],
                        "naive_motor_rating_x_run_time_kWh": r["naive_motor_rating_x_run_time_kWh"]})
    _write("op_b_energy_by_candidate.csv", by_cand)
    _write("sensitivity.csv", sensitivity("OP-A") + sensitivity("OP-B", motors[top_b]))
    env = []
    for op, kw in (("OP-A", None), ("OP-B", motors[top_b])):
        for case, r in envelope(op, kw).items():
            env.append({"operation": op, "case": case, "energy_total_kWh": r["energy_total_kWh"],
                        "energy_per_good_part_kWh": r["energy_per_good_part_kWh"],
                        "input_material_g": r["input_material_g"], "started_parts": r["started_parts"]})
    _write("envelopes.csv", env)

    summary = {
        "reconciliation": {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "flag", "fail", "unresolved")},
        "candidates": {op: {s: sum(1 for r in ranking if r["operation"] == op and r["status"] == s)
                            for s in sorted({r["status"] for r in ranking})} for op in ("OP-A", "OP-B")},
        "op_a_central": {k: a_c[k] for k in ("started_parts", "sheets", "input_material_g", "retained_good_parts_g",
                                             "energy_total_kWh", "energy_per_good_part_kWh")},
        "op_b_central": {k: b_c[k] for k in ("started_parts", "strokes", "input_material_g", "retained_good_parts_g",
                                             "required_force_kN", "energy_total_kWh", "energy_per_good_part_kWh")},
        "envelopes_kWh": {f"{e['operation']}:{e['case']}": e["energy_total_kWh"] for e in env},
        "scope_note": "Energy covers only the two selected operations; it is not module energy.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda v: round(v, 4)))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=lambda v: round(v, 4)))
