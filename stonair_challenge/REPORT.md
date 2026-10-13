# Findings: equipment discovery and scoped resource estimate (P002, P007)

Stonair candidate challenge, Rev A. Prepared 2026-10-13. Scope: 100 modules. Two operations only; **nothing here is module energy.**

## 1. Data integrity

- `bom.csv` matches the xlsx BOM sheet. BOM quantity per parent equals the JSON instance count and the STEP assembly occurrence count for all 10 parts (53 instances in total).
- All 53 JSON positions equal the STEP placement translations, and there are no rotations.
- Mixed units are normalised: P002 is 0.2 cm = 2 mm, and P006–P008 are 300 µm = 0.3 mm. µm arrives as U+00B5. Both readings agree with the smallest CAD extent.
- Blank cells stay `None`. No fabricated-part mass is supplied, so mass = CAD net volume × sourced density.
- A STEP parser re-derives the geometry. P002 is 172 × 124 × 2 mm with 4 × Ø3.4 holes. P007 is 12 × 106 × 0.3 mm with 1 × Ø5.5 hole. Prism-minus-holes reproduces both CAD net volumes to 1e-12, so there are no hidden features.
- The P002 holes coincide with the four M3 screws above the cover.

Results: 74 checks pass, 0 fail, 17 flags and 9 unresolved (`outputs/reconciliation_checks.csv`).

**Flags raised rather than silently resolved:**
1. STEP names differ from BOM names for P001, P006 and P007. P009's "×" is double-encoded in the STEP file; I repaired and flagged it.
2. Local origin conventions differ per part: corner for the plates; offsets for the cells, carriers and fasteners.
3. P008 is not a prism with round holes.
4. The collector plates sit 0.8 mm (top) and 2.2 mm (bottom) away from the cell ends, consistent with "joining interfaces absent".
5. Copper temper is "unselected", which matters for blanking force: shear strength is 150–230 MPa.

## 2. Routes and selected operations

Detail is in `docs/process_routes.md`.

- **OP-A:** fiber-laser cutting of P002 from 2 mm 6061-T6 sheet. The part is a flat profile with holes, 100 parts fit on one sheet, and no part tooling is needed.
- **OP-B:** pierce Ø5.5 + cut-off of P007 from 12 mm-slit 0.3 mm C11000 strip on a gap-frame press.

Alternatives are kept separate: turret punch, press blanking or waterjet for P002; compound blank, laser, etching or EDM for P007. At 200 parts, laser cutting of P007 is probably cheaper because it needs no die. I studied stamping as the scale-representative route, and I flag that as a judgement call.

## 3. Machine discovery

There were 35 logged searches plus 1 failed retrieval test (`data/evidence/search_log.csv`), 50 registered sources, 51 machine specifications (`outputs/machine_evidence.csv`) and 15 process or material values. Each specification carries machine identity, configuration, original and normalised value, source, location, excerpt, applicability and uncertainty. 17 leads were rejected with reasons, for example a CO₂ machine listed as fiber, marketplace listings, oversized presses and a Wikipedia engine article.

| OP-A (laser) | Status / rank | Best energy evidence |
|---|---|---|
| TRUMPF TruLaser 1030 fiber, 3 kW | eligible, **1** | OEM *average power input in production* 12 kW (3 kW); 10 kW (current-gen 4 kW) |
| Mazak OPTIPLEX 3015 FIBER III, 2 kW | eligible, 2 | 58 kVA total electrical requirement (connection rating only) |
| Mitsubishi 3015GX-F F40 | eligible, 3 | none (optical rating only) |
| Amada VENTIS-3015 AJ 4 kW | eligible, 4 | "-30 % electricity" claim; unusable |
| Salvagnini L5, 3 kW | eligible_unverified, 5 | OEM *average absorbed* 12 kW, *maximum absorbed* 18 kW; work area not captured |
| Bystronic BySmart Fiber 3015; Amada ENSIS-3015 AJ | secondary evidence only, 6–7 | resale listings only |
| Prima Power Laser Genius+ | excluded | OEM page gives no electrical or thickness data |

| OP-B (press) | Status / rank | Best energy evidence |
|---|---|---|
| AIDA NC1-350(1)E | eligible, **1** | main motor 5.5 kW (OEM) |
| Komatsu H1F35-1 (servo) | eligible, 2 | 6.7 hp (directory) + OEM "30 % less" claim |
| Bruderer BSTA 200-60 | unverified, 3 | 15 kW motor, 59 kVA (OEM); coil-fed, minimum 100 spm, wrong scale |
| Chin Fong OCP-25E | unverified, 4 | motor field "VFD 3*4", not interpreted as kW |
| SEYI SN1-25 | secondary only, 5 | 3.7 kW (dealer) |

Gates are applied first: thickness and work envelope for the lasers; force ≥ 22.7 kN worst case and bolster ≥ 300 × 250 mm for the presses. The ranking then scores 2 × energy-evidence quality + source tier + scale fit.

The power evidence uses distinct types: rated laser output, motor rating, installed or connected load (kVA, never converted to kW), maximum operating, average operating, and relative claim. No source anywhere gave **energy per part or per stroke**.

## 4. Estimates (central, with all-favourable / all-unfavourable envelope)

| | OP-A P002 laser | OP-B P007 press (AIDA, 5.5 kW motor) |
|---|---|---|
| Good parts / started | 100 / 103 (2 % reject assumed) | 200 / 205 + 10 try-out |
| Input material | 1 sheet 1219 × 2438 × 2 mm = **16.05 kg** | 23.3 m of 12 mm strip = **0.745 kg** |
| Retained in good parts | 11.50 kg (114.98 g each) | 0.666 kg (3.329 g each) |
| Rejected parts | 0.345 kg | 0.017 kg + 0.033 kg try-out |
| Removed / waste | holes 0.020 kg; skeleton 0.837 kg; **sheet remainder 3.35 kg (reusable remnant)** | slugs 0.014 kg; strip ends 0.016 kg |
| Electricity | **3.25 kWh** (1.25–12.0) = 0.033 kWh per part | **0.81 kWh** (0.16–2.70) = 0.004 kWh per part |

**OP-A formula.** E = N_started × t_part × P_proc + N_sheets × t_handling × P_standby.

- t_part = 592 mm / v + 42.7 mm / (v·f_hole) + 5 × (t_pierce + t_move), which gives 8.3 s.
- P_proc = 12 kW. TRUMPF (3 kW) and Salvagnini (3 kW) agree on this average. Both are measured over mixed production; Salvagnini's basis is mild steel 0.8–6 mm, not aluminium.
- The cutting speed of 9 m/min is a research value for 2 mm AA6061 at 3 kW. Alternatives are 4 m/min from a guide and 16.5 m/min from a laser-source chart.

**OP-B formula.** E = P_motor × f_idle × (t_run + t_setup) + N × W / η, with W = k × F × t and F = L_shear × t × τ × factor.

- F is 2.0 kN, so W ≈ 0.3 J/stroke. Deformation energy (≈ 2.6e-5 kWh) is negligible. **Idle draw is ≈ 100 % of the total.**
- Multiplying the motor rating by run time would give 1.31 kWh, which shows why a brochure rating does not equal part energy.
- By press: SEYI 0.54, Komatsu 0.73 (likely overstated for a servo), AIDA 0.81, Bruderer 2.20 kWh.

**What drives the uncertainty** (`outputs/sensitivity.csv`):

- **OP-A:** cutting speed (+63 / −23 %), processing power (−15 / +44 %), pierce and move times (up to +26 %). Sheet size moves material by up to +35 %: 1000 × 2000 needs 2 sheets.
- **OP-B:** idle fraction (−50 / +67 %), setup time (−34 / +51 %), feed rate (+43 / −24 %). Route moves strip input by +33 % for a compound blank. Copper shear strength changes energy by < 0.01 %, though it does matter for die design.

## 5. Reliability and unresolved gaps

1. **No primary document was opened.** WebFetch failed DNS for every host, and container egress returned 403. Every value is a search-engine excerpt, marked `primary_document_opened = no`. Several conflicts were found and kept, not averaged: TRUMPF 12 vs 13 kW; Salvagnini 10 kW at 45 vs 42 kW; SEYI 70–180 vs 35–65 spm; AIDA bolster dimensions. The first step for a reviewer is to open SRC-TRU-02, SRC-SAL-01, SRC-AID-01 and SRC-BRU-01 and confirm the cited rows.
2. **Unsourced and dominant:** laser standby power, press idle fraction (only a qualitative "large share" was found), pierce and move times, and feed rate. These are labelled assumptions in `data/assumptions.csv`.
3. **OEM averages are not aluminium- or part-specific.** No energy per part or per stroke was published by any OEM.
4. Salvagnini's work envelope, Bystronic and Amada ENSIS OEM data sheets, and the Chin Fong motor kW were not captured.
5. Not modelled: assist gas (N₂) and compressed air, extraction beyond the OEM bundle, deburring, slitting of the copper strip, die manufacture, and transport.

Out of time box: the laser route for P007, and a measured-power request to OEMs.

*AI disclosure:* this work was produced with an AI coding assistant (Claude), which ran the searches and wrote the code and text. Search excerpts are machine-summarised and are **not** verified quotations. Every value is attributed so it can be checked.
