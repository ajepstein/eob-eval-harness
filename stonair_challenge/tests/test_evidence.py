import csv
import shutil

import pytest

from stonair import evidence
from stonair.estimate import params
from stonair.evidence import machine_evidence, read, validate
from stonair.ranking import rank


def test_committed_evidence_validates():
    assert validate() == []


@pytest.fixture
def evid_copy(tmp_path):
    shutil.copytree(evidence.EVID, tmp_path / "ev")
    return tmp_path / "ev"


def _edit(path, eid, **changes):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        if r.get("evidence_id") == eid:
            r.update(changes)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("changes,expect", [
    ({"source_id": "SRC-NOPE"}, "not in sources.csv"),
    ({"candidate_id": "A99"}, "not in machine_candidates.csv"),
    ({"model": "TruLaser 3030 fiber"}, "machine identity"),
    ({"value_low": "11", "value_high": "11"}, "not found in original_value"),
    ({"excerpt": ""}, "empty excerpt"),
    ({"power_type": "n/a"}, "power value without a power_type"),
])
def test_attribution_tampering_is_caught(evid_copy, changes, expect):
    _edit(evid_copy / "machine_evidence_raw.csv", "E-A1-02", **changes)
    probs = validate(evid_copy)
    assert any(expect in p for p in probs), probs


def test_kva_cannot_be_labelled_operating_power(evid_copy):
    _edit(evid_copy / "machine_evidence_raw.csv", "E-A4-01", power_type="average_operating_power")
    assert any("kVA must be classed" in p for p in validate(evid_copy))


def test_normalisation_keeps_original_and_converts():
    ev = {r["evidence_id"]: r for r in machine_evidence()}
    r = ev["E-A4-02"]
    assert (r["original_value"], r["original_unit"]) == ("0.320", "in")
    assert r["normalized_low"] == pytest.approx(8.128) and r["normalized_unit"] == "mm"
    assert ev["E-A4-01"]["normalized_unit"] == "kVA"          # stays apparent power
    assert ev["E-B3-04"]["normalized_low"] == pytest.approx(4.996, rel=1e-3)
    assert ev["E-B5-03"]["normalized_low"] is None             # uninterpreted text stays blank


def test_power_types_are_not_interchangeable():
    ev = machine_evidence()
    a1 = [r for r in ev if r["candidate_id"] == "A1" and r["normalized_unit"] == "kW"]
    kinds = {r["power_type"] for r in a1}
    # the 3 kW optical output must never be confused with the 12 kW average electrical input
    assert {"rated_laser_output", "average_operating_power"} <= kinds
    assert all(r["normalized_low"] < 5 for r in a1 if r["power_type"] == "rated_laser_output")


def test_sourced_assumptions_trace_to_evidence():
    """Central/low/high values labelled 'sourced_scenario' must exist in the evidence tables."""
    ev = machine_evidence() + evidence.process_evidence()
    vals = {(round(v, 6)) for r in ev for v in (r["normalized_low"], r["normalized_high"]) if v is not None}
    raw_vals = {round(float(x), 6) for r in ev for x in (r["value_low"], r["value_high"]) if x not in ("", None)}
    p = {lvl: params("OP-A", lvl) for lvl in ("low", "central", "high")}
    assert p["central"]["processing_power_kW"] in raw_vals          # 12 kW (E-A1-02, E-A2-01)
    assert p["low"]["processing_power_kW"] in raw_vals               # 10 kW (E-A1-01)
    assert p["high"]["processing_power_kW"] in raw_vals              # 18 kW (E-A2-02)
    assert p["central"]["contour_speed_m_min"] in raw_vals           # 9 m/min (P-01)
    b = {lvl: params("OP-B", lvl) for lvl in ("low", "high")}
    assert b["low"]["shear_strength_MPa"] in raw_vals and b["high"]["shear_strength_MPa"] in raw_vals
    assert vals  # normalisation produced values


def test_candidate_counts_and_distinct_machines():
    cands = read("machine_candidates.csv")
    for op in ("OP-A", "OP-B"):
        models = {(c["manufacturer"], c["model"]) for c in cands if c["operation"] == op}
        assert len(models) >= 5, op


def test_ranking_rules():
    r = {x["candidate_id"]: x for x in rank()}
    assert r["A8"]["status"] == "excluded_insufficient_evidence" and r["A8"]["rank"] is None
    assert r["B5"]["best_energy_evidence"] == "none"                 # text motor field earns nothing
    for op in ("OP-A", "OP-B"):
        ranks = sorted(x["rank"] for x in r.values() if x["operation"] == op and x["rank"])
        assert ranks == list(range(1, len(ranks) + 1))
    ev_ids = {e["evidence_id"] for e in machine_evidence()}
    for x in r.values():
        assert all(i in ev_ids for i in x["gate_evidence_ids"].split(";") if i)
    assert r["A1"]["rank"] == 1                                       # only eligible with OEM average power
