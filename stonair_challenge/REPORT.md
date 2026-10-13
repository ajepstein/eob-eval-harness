# Findings: equipment discovery and scoped resource estimate (P002, P007)

Stonair candidate challenge, Rev A. Prepared 2026-10-13. Scope: 100 modules. One operation for P002, two alternative routes for P007, and a die cost estimate for the press route; **nothing here is module energy.**

## 1. Data integrity

- `bom.csv` matches the xlsx. Quantity per parent equals the JSON instance count and the STEP occurrence count for all 10 parts (53 instances). All 53 JSON positions equal the STEP translations, with no rotations.
- Mixed units are normalised (P002 0.2 cm = 2 mm; P006–P008 300 µm = 0.3 mm) and agree with the smallest CAD extent. Blank cells stay `None`; mass = CAD net volume × sourced density.
- A STEP parser re-derives the geometry: P002 is 172 × 124 × 2 mm with 4 × Ø3.4 holes under the four M3 screws; P007 is 12 × 106 × 0.3 mm with 1 × Ø5.5 hole. Prism-minus-holes reproduces both CAD volumes to 1e-12, so there are no hidden features.

Results: 74 checks pass, 0 fail, 17 flags and 9 unresolved (`outputs/reconciliation_checks.csv`).

**Flagged, not silently resolved:** STEP names differ from the BOM for P001, P006 and P007, and P009's "×" is double-encoded in the STEP file (repaired). Origin conventions differ per part. P008 is not a prism with round holes. The collector plates sit 0.8 mm (top) and 2.2 mm (bottom) off the cell ends, consistent with "joining interfaces absent". Copper temper is "unselected", so shear strength spans 150–230 MPa.

## 2. Routes and selected operations

Detail is in `docs/process_routes.md`.

- **OP-A:** fiber-laser cutting of P002 from 2 mm 6061-T6 sheet. The part is a flat profile with holes, 100 parts fit on one sheet, and no part tooling is needed.
- **OP-B:** pierce Ø5.5 + cut-off of P007 from 12 mm-slit 0.3 mm C11000 strip on a gap-frame press.
- **OP-C (added after the first submission):** fiber-laser cutting of P007 from 0.3 mm C11000 sheet. This is the no-die alternative to OP-B, estimated on the same demand so the two routes can be compared.

Other alternatives are kept separate but not estimated: turret punch, press blanking or waterjet for P002; compound blank, etching or EDM for P007.

## 3. Machine discovery

There were 59 logged searches plus 1 failed retrieval test (`data/evidence/search_log.csv`), 70 registered sources, 70 machine specifications (`outputs/machine_evidence.csv`) and 33 process, material or cost values. Each specification carries machine identity, configuration, original and normalised value, source, location, excerpt, applicability and uncertainty. 30 leads were rejected with reasons, for example a CO₂ machine listed as fiber, marketplace listings, job-shop service pages, oversized presses and a Wikipedia engine article.

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

| OP-C (laser, copper) | Status / rank | Copper evidence / best energy evidence |
|---|---|---|
| TRUMPF TruLaser 1030 fiber, 4 kW | eligible, **1** | Cu 8 mm; OEM average power in production 10 kW |
| Han's Laser MPS0606D, 1 kW (600 × 600 mm precision cutter) | eligible, 2 | OEM lists copper, no thickness; optical rating only |
| IPG LaserCube (1250 × 1250 mm) | eligible, 3 | OEM names copper; only a breaker rating for a 4 kW unit (not applied) |
| Amada VENTIS-3015 AJ, 4 kW | eligible, 4 | Cu 8 mm (OEM press release); no power data |
| Mitsubishi 3015GX-F F40 | eligible, 5 | Cu 6/8 mm (reseller); no power data |
| Salvagnini L5, 3 kW | unverified, 6 | Cu 5–8 mm; OEM average 12 kW, maximum 18 kW; work area not captured |

Mazak was dropped for OP-C (no copper data). The small-format machines suit 12 × 106 mm parts better than a flatbed but publish no power data, so OP-C uses the flatbed power figures.

Gates are applied first: material thickness (or an OEM material listing) and a work envelope that takes the smallest sheet scenario for the lasers; force ≥ 22.7 kN worst case and bolster ≥ 300 × 250 mm for the presses. The ranking then scores 2 × energy-evidence quality + source tier + scale fit.

The power evidence uses distinct types: rated laser output, motor rating, installed or connected load (kVA, never converted to kW), maximum operating, average operating, and relative claim. No source anywhere gave **energy per part or per stroke**.

## 4. Estimates (central, with all-favourable / all-unfavourable envelope)

| | OP-A P002 laser | OP-B P007 press (AIDA, 5.5 kW motor) | OP-C P007 laser |
|---|---|---|---|
| Good parts / started | 100 / 103 (2 % reject assumed) | 200 / 205 + 10 try-out | 200 / 205 |
| Input material | 1 sheet 1219 × 2438 × 2 mm = **16.05 kg** | 23.3 m of 12 mm strip = **0.745 kg** | 1 sheet 600 × 1500 × 0.3 mm = **2.40 kg** |
| Retained in good parts | 11.50 kg (114.98 g each) | 0.666 kg (3.329 g each) | 0.666 kg |
| Rejected parts | 0.345 kg | 0.017 kg + 0.033 kg try-out | 0.017 kg |
| Removed / waste | holes 0.020 kg; skeleton 0.837 kg; **sheet remainder 3.35 kg (reusable remnant)** | slugs 0.014 kg; strip ends 0.016 kg | hole 0.014 kg; skeleton 0.198 kg; **remainder 1.51 kg (reusable)** |
| Electricity | **3.25 kWh** (1.25–12.0) = 0.033 kWh per part | **0.81 kWh** (0.16–2.70) = 0.004 kWh per part | **2.82 kWh** (1.02–11.1) = 0.014 kWh per part |

**OP-A formula.** E = N_started × t_part × P_proc + N_sheets × t_handling × P_standby.

- t_part = 592 mm / v + 42.7 mm / (v·f_hole) + 5 × (t_pierce + t_move), which gives 8.3 s.
- P_proc = 12 kW. TRUMPF (3 kW) and Salvagnini (3 kW) agree on this average. Both are measured over mixed production; Salvagnini's basis is mild steel 0.8–6 mm, not aluminium.
- The cutting speed of 9 m/min is a research value for 2 mm AA6061 at 3 kW. Alternatives are 4 m/min from a guide and 16.5 m/min from a laser-source chart.

**OP-B formula.** E = P_motor × f_idle × (t_run + t_setup) + N × W / η, with W = k × F × t and F = L_shear × t × τ × factor.

- F is 2.0 kN, so W ≈ 0.3 J/stroke and deformation energy is negligible: **idle draw is ≈ 100 % of the total.** Motor rating × run time would give 1.31 kWh instead, so a brochure rating is not part energy.
- By press: SEYI 0.54, Komatsu 0.73 (likely overstated for a servo), AIDA 0.81, Bruderer 2.20 kWh.

**OP-C** uses the OP-A formula with copper inputs: 236 mm perimeter + Ø5.5 hole, 0.5 s pierce (copper pierces less readily), P_proc 10–12–18 kW, giving 3.5 s per part. No source gives a 0.3 mm copper speed, so the speeds are thicker-sheet proxies: 3.4 m/min (1.2 mm, 3 kW), 10 m/min central (1 mm, 1 kW, N₂) and 21 m/min (1 mm, 3 kW).

**P007 routes compared** (`outputs/p007_route_comparison.csv`): the laser uses about 3.5× the press's electricity (2.82 vs 0.81 kWh) and buys about 3.2× the copper (2.40 vs 0.75 kg, 1.51 kg of it reusable offcut), but needs no die.

**Die cost for the press route** (`outputs/die_cost.csv`, nominal USD, not a quote):

| Die | Method | Low / central / high | Per good part, one 200-part order |
|---|---|---|---|
| Pierce + cut-off (simple die) | published ranges, 6 sources | $1,000 / **$4,000** / $10,000 | **$20** |
| Pierce + cut-off (simple die) | die set + parts + toolmaker hours | $2,970 / $5,800 / $11,780 | $29 |
| Compound blank (alternative) | published ranges, 3 sources | $2,000 / $7,000 / $25,000 | $35 |

- **Published ranges:** low = smallest published low, high = largest published high, central = median of the range midpoints.
- **Die set + toolmaker hours:** a $436–1,038 catalog die set, $300–1,200 of components (assumed) and 30–80 toolmaker hours. Hours are costed at a sourced $29.82/h wage × a 2.5–4× overhead multiplier (assumed, since no shop rate was found).
- **The two methods overlap.** Every die price comes from a die maker's or supplier's own marketing page, mixing US and Chinese pricing.

**What it means for P007:** the press route's 2 kWh electricity saving is negligible beside a die of roughly $4,000. At 200 parts the die adds about **$20 per good part**; spread over 10 repeat orders it falls to about $2. The laser route is therefore the likely choice for this order unless a laser job costs more than about $20 per part above stamping. Laser job pricing was not researched, so that comparison stays open. The only copper example found (a laser vendor's 2 mm busbar case, with a $2,500 die) implies a break-even near 25,000 parts. It covers labour and consumables only, so it is context, not evidence.

**What drives the uncertainty** (`outputs/sensitivity.csv`):

- **OP-A:** cutting speed (+63 / −23 %), processing power (−15 / +44 %), pierce and move times (up to +26 %). Sheet size moves material by up to +35 %: 1000 × 2000 needs 2 sheets.
- **OP-B:** idle fraction (−50 / +67 %), setup time (−34 / +51 %), feed rate (+43 / −24 %). Route moves strip input by +33 % for a compound blank. Copper shear strength changes energy by < 0.01 %, though it does matter for die design.
- **OP-C:** cutting speed (+81 / −22 %), processing power (−14 / +43 %), pierce time (−15 / +24 %). Sheet size moves purchased copper from −20 % (two 600 × 600 blanks) to +122 % (one 1000 × 2000 sheet); the extra is reusable offcut, not scrap.

## 5. Reliability and unresolved gaps

1. **No primary document was opened.** WebFetch failed DNS for every host, and container egress returned 403. Every value is a search-engine excerpt, marked `primary_document_opened = no`. Several conflicts were found and kept, not averaged: TRUMPF 12 vs 13 kW; Salvagnini 10 kW at 45 vs 42 kW; SEYI 70–180 vs 35–65 spm; AIDA bolster dimensions. The first step for a reviewer is to open SRC-TRU-02, SRC-SAL-01, SRC-AID-01 and SRC-BRU-01 and confirm the cited rows.
2. **Unsourced and dominant:** laser standby power, press idle fraction (only a qualitative "large share" was found), pierce and move times, and feed rate. These are labelled assumptions in `data/assumptions.csv`.
3. **OEM averages are not aluminium-, copper- or part-specific.** No energy per part or per stroke was published by any OEM. For OP-C, no 0.3 mm copper speed and no power figure for the small-format machines was found.
4. Salvagnini's work envelope, Bystronic and Amada ENSIS OEM data sheets, and the Chin Fong motor kW were not captured.
5. Not modelled: assist gas (N₂) and compressed air, extraction beyond the OEM bundle, deburring, slitting of the copper strip, die maintenance and wear, thin-copper distortion and burr, laser job pricing, and transport.

Still outside the time box: a measured-power request to OEMs, a real die quote, and laser job-shop prices to close the P007 cost comparison.

*AI disclosure:* this work was produced with an AI coding assistant (Claude), which ran the searches and wrote the code and text. Search excerpts are machine-summarised and are **not** verified quotations. Every value is attributed so it can be checked.
