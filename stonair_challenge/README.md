# Stonair challenge: equipment discovery and scoped resource estimate

This folder is self-contained, and nothing in the parent repository is used. Read `REPORT.md` first; it is the two-page findings report.

## Run

```bash
cd stonair_challenge
python -m pip install -r requirements.txt      # openpyxl (xlsx check), pytest
PYTHONPATH=src python -m stonair.run_all       # regenerates outputs/ from data/
python -m pytest                               # 81 tests
```

Python ≥ 3.10 and the standard library only, apart from `openpyxl` for the csv/xlsx consistency check. There is no network access at run time: all evidence is committed as data.

## Layout

| Path | What it is |
|---|---|
| `data/raw/` | Unmodified candidate pack: BOM csv/xlsx, cad_parts.csv, cad_instances.json, START_HERE, STEP assembly and parts |
| `data/evidence/search_log.csv` | Every query, with tool, mode, date, results considered and outcome (S00 records the failed direct-retrieval test) |
| `data/evidence/sources.csv` | Source registry: URL, title, publisher, **tier** (OEM / OEM document hosted by distributor / distributor / directory / resale / third party / research), access date and method |
| `data/evidence/machine_candidates.csv` | One row per machine/configuration candidate (8 laser for P002, 5 press and 6 laser for P007) |
| `data/evidence/machine_evidence_raw.csv` | **The machine_evidence dataset**: identity, configuration, parameter, power type, original value and unit, source, location, excerpt, applicability, uncertainty |
| `data/evidence/process_material_evidence.csv` | Cutting speeds, densities, shear strengths, stock sizes, press-energy literature |
| `data/evidence/rejected_leads.csv` | Leads excluded before candidacy, with reason codes |
| `data/assumptions.csv` | Every non-sourced input (low / central / high), marked `assumption`, `sourced_scenario` or `scenario` |
| `docs/process_routes.md` | Routes and alternatives for P002 and P007: one operation for P002, and two alternative routes estimated for P007 (press OP-B, laser OP-C) |
| `src/stonair/` | `units` (conversion), `step_geom` (minimal STEP reader), `ingest` (normalisation and reconciliation), `evidence` (normalisation and attribution validation), `ranking` (gates and ranking), `estimate` (material and energy), `die_cost` (press-route die cost), `run_all` |
| `outputs/` | Generated: `normalized_bom.csv`, `data_dictionary.csv`, `reconciliation_checks.csv`, `machine_evidence.csv` (with normalised values), `candidate_ranking.csv`, `estimates_central.csv`, `op_b_energy_by_candidate.csv`, `p007_route_comparison.csv`, `die_cost.csv`, `sensitivity.csv`, `envelopes.csv`, `summary.json` |
| `tests/` | Unit conversion, ingest/reconciliation, STEP geometry, evidence attribution (including tamper tests), ranking rules, mass balance and hand-calculated energy |

## Conventions

- Blank means not supplied, never zero, and it propagates as `None`.
- Power is never one number. `power_type` takes one of these values: `rated_laser_output`, `motor_rating`, `installed_connected_load` (kVA is kept as kVA and cannot be converted to kW), `max_operating_power`, `average_operating_power`, `operating_power_range`, `operating_power_unspecified`, `energy_per_cycle` or `relative_claim`.
- Every machine number is tied to one manufacturer + model + configuration. Rows for a different configuration are marked "(not the scoped config)" and are excluded from gates and estimates.
- Material flows are reported separately: retained (good parts), rejected parts, try-out, removed (holes or slugs), skeleton or strip waste, and sheet remainder. The flows must sum to the input, and a test enforces this.

## Limitations of this session

Only a web *search* tool worked. Page and PDF retrieval failed: WebFetch DNS errors, and the container's egress proxy returned 403. Every excerpt is therefore a search-engine summary of the page, and `primary_document_opened = no` throughout. To lift this, allow the source hosts (trumpf.com, salvagnini*, aida-global.com, brudereramericas.com, ...) in the environment's network settings and re-check the cited rows.

AI disclosure: produced with an AI coding assistant (Claude). AI output is not evidence. Each figure points to a source row that a person should verify.
