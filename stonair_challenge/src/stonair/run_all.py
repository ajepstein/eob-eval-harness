"""Regenerate every output in outputs/ from data/. Usage: python -m stonair.run_all"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from . import die_cost, job_cost
from .estimate import envelope, op_a, op_b, op_c, params, press_motor_ratings, sensitivity
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
    c_c = op_c(params("OP-C"))
    _write("estimates_central.csv", [{"basis": "OP-A central, sourced 3 kW power scenario (A1/A2 evidence)", **a_c},
                                     {"basis": f"OP-B central, motor rating of rank-1 press {top_b}", **b_c},
                                     {"basis": "OP-C central, sourced 3-4 kW power scenario (C1/C2 evidence)", **c_c}])
    by_cand = []
    for cid, kw in sorted(motors.items()):
        r = op_b(params("OP-B"), kw)
        by_cand.append({"candidate_id": cid, "motor_rating_kW": kw,
                        "note": "servo: idle-fraction assumption is for flywheel presses and likely overstates"
                                if cid == "B3" else "",
                        "energy_total_kWh": r["energy_total_kWh"], "energy_per_good_part_kWh": r["energy_per_good_part_kWh"],
                        "naive_motor_rating_x_run_time_kWh": r["naive_motor_rating_x_run_time_kWh"]})
    _write("op_b_energy_by_candidate.csv", by_cand)
    _write("sensitivity.csv", sensitivity("OP-A") + sensitivity("OP-B", motors[top_b]) + sensitivity("OP-C"))
    env = []
    for op, kw in (("OP-A", None), ("OP-B", motors[top_b]), ("OP-C", None)):
        for case, r in envelope(op, kw).items():
            env.append({"operation": op, "case": case, "energy_total_kWh": r["energy_total_kWh"],
                        "energy_per_good_part_kWh": r["energy_per_good_part_kWh"],
                        "input_material_g": r["input_material_g"], "started_parts": r["started_parts"]})
    _write("envelopes.csv", env)
    dies = die_cost.summary(b_c["good_parts"])
    _write("die_cost.csv", dies)
    die_td = dies[0]
    jobs = [job_cost.job_price(l) for l in ("low", "central", "high")]
    jobs.append(job_cost.job_price("central", cut_case="unfavourable"))
    _write("laser_job_cost.csv", jobs)
    _write("laser_service_thickness.csv", job_cost.thin_copper_availability())
    job_c = jobs[1]
    # Lower bound: treats the press's own per-part running cost as zero, so the true break-even is later.
    be = die_cost.break_even_parts(die_td["central_USD"], job_c["marginal_per_part_USD"], 0.0)
    p007 = []
    for route, r in (("OP-B press pierce + cut-off", b_c), ("OP-C fiber laser", c_c)):
        e = {x["case"]: x["energy_total_kWh"] for x in env if x["operation"] == r["operation"]}
        p007.append({"route": route, "good_parts": r["good_parts"], "input_material_g": r["input_material_g"],
                     "retained_good_parts_g": r["retained_good_parts_g"],
                     "reusable_remainder_g": r.get("sheet_remainder_g", 0.0),
                     "scrap_g": r["input_material_g"] - r["retained_good_parts_g"] - r.get("sheet_remainder_g", 0.0),
                     "energy_central_kWh": r["energy_total_kWh"], "energy_favourable_kWh": e["favourable"],
                     "energy_unfavourable_kWh": e["unfavourable"],
                     "part_tooling": "custom pierce/cut-off die" if r["operation"] == "OP-B" else "none",
                     "die_cost_USD_low_central_high": (f"{die_td['low_USD']:.0f} / {die_td['central_USD']:.0f} / "
                                                       f"{die_td['high_USD']:.0f}") if r["operation"] == "OP-B" else "0",
                     "die_USD_per_good_part_this_order": die_td["central_USD_per_good_part_1_order"]
                     if r["operation"] == "OP-B" else 0.0,
                     "laser_job_price_USD_low_central_high": "" if r["operation"] == "OP-B" else
                     " / ".join(f"{j['order_price_USD']:.0f}" for j in jobs[:3]),
                     "break_even_parts_lower_bound": be if r["operation"] == "OP-B" else ""})
    _write("p007_route_comparison.csv", p007)

    summary = {
        "reconciliation": {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "flag", "fail", "unresolved")},
        "candidates": {op: {s: sum(1 for r in ranking if r["operation"] == op and r["status"] == s)
                            for s in sorted({r["status"] for r in ranking})} for op in ("OP-A", "OP-B", "OP-C")},
        "op_a_central": {k: a_c[k] for k in ("started_parts", "sheets", "input_material_g", "retained_good_parts_g",
                                             "energy_total_kWh", "energy_per_good_part_kWh")},
        "op_b_central": {k: b_c[k] for k in ("started_parts", "strokes", "input_material_g", "retained_good_parts_g",
                                             "required_force_kN", "energy_total_kWh", "energy_per_good_part_kWh")},
        "op_c_central": {k: c_c[k] for k in ("started_parts", "sheets", "sheet_size_mm", "input_material_g",
                                             "retained_good_parts_g", "sheet_remainder_g", "energy_total_kWh",
                                             "energy_per_good_part_kWh")},
        "p007_die_cost_USD": {d["method"] + ":" + d["route"]: [d["low_USD"], d["central_USD"], d["high_USD"]]
                              for d in dies},
        "p007_laser_job_price_USD": {j["level"] + ":" + j["cut_case"]: j["order_price_USD"] for j in jobs},
        "p007_break_even_parts_lower_bound": be,
        "envelopes_kWh": {f"{e['operation']}:{e['case']}": e["energy_total_kWh"] for e in env},
        "scope_note": "Energy covers only the selected operations (OP-B and OP-C are alternative routes for the same part); it is not module energy.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda v: round(v, 4)))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=lambda v: round(v, 4)))
