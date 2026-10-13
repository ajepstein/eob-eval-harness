# Findings: equipment discovery and scoped resource estimate (P002, P007)

Stonair candidate challenge, Rev A, 2026-10-13. Scope: 100 modules. One operation for P002 and two alternative routes for P007. **Nothing here is module energy.** Details live in `docs/process_routes.md`, `data/evidence/` and `outputs/`.

## 1. Data integrity

`bom.csv` matches the xlsx. Quantities per parent match the JSON and STEP instance counts (53 instances), and every JSON position equals its STEP placement. Mixed units are normalised (0.2 cm = 2 mm; 300 µm = 0.3 mm) and agree with the CAD. Blank cells stay `None`.

A STEP parser confirms the two parts:
- **P002:** 172 × 124 × 2 mm with 4 × Ø3.4 holes under the M3 screws.
- **P007:** 12 × 106 × 0.3 mm with 1 × Ø5.5 hole.

Both rebuild the CAD volumes exactly, so there are no hidden features. Results: 74 checks pass, 0 fail, 17 flagged and 9 unresolved (`outputs/reconciliation_checks.csv`).

**Flagged rather than silently resolved:**
- Part names differ between the BOM and STEP, and P009's "×" is garbled in the STEP file.
- Origin conventions differ between parts.
- The collector plates sit 0.8–2.2 mm off the cell ends, so there is no joining interface.
- Copper temper is unselected, so shear strength spans 150–230 MPa.

## 2. Operations

- **OP-A:** fiber-laser cutting of P002 from 2 mm 6061-T6 sheet.
- **OP-B:** pierce Ø5.5 + cut-off of P007 from 12 mm-slit 0.3 mm C11000 strip on a gap-frame press.
- **OP-C:** fiber-laser cutting of P007 from 0.3 mm sheet. This is the no-die alternative to OP-B.

Other routes are listed but not estimated: turret punch or waterjet for P002; compound blank, etching or EDM for P007.

## 3. Machine discovery

59 searches, 70 sources, 70 machine specifications and 33 process, material or cost values; 30 leads rejected with reasons. Machines must first pass the compatibility checks: material thickness and work envelope for lasers; force ≥ 22.7 kN and bolster area for presses. Eligible machines are then ranked by 2 × energy-evidence quality + source tier + scale fit. The full table is `outputs/candidate_ranking.csv`.

| Op | Rank 1 (best energy evidence) | Other eligible | Not eligible / weaker |
|---|---|---|---|
| OP-A | TRUMPF TruLaser 1030 fiber, 3 kW (OEM average power in production 12 kW) | Mazak (58 kVA, connection rating only), Mitsubishi, Amada VENTIS | Salvagnini L5 (average 12 kW, maximum 18 kW; work area missing); Bystronic, Amada ENSIS (resale only); Prima Power excluded |
| OP-B | AIDA NC1-350 (5.5 kW motor, OEM) | Komatsu H1F35 servo | Bruderer (wrong scale), Chin Fong (motor kW unreadable), SEYI (dealer data only) |
| OP-C | TRUMPF 1030 fiber, 4 kW (Cu 8 mm; average 10 kW) | Han's Laser MPS0606D, IPG LaserCube, Amada VENTIS, Mitsubishi | Salvagnini (work area missing); Mazak dropped (no copper data) |

The power evidence keeps its types separate: laser output, motor rating, connection rating (kVA, never converted to kW), maximum or average operating power, and relative claims. **No source gave energy per part or per stroke.**

## 4. Estimates (central; range = all-favourable to all-unfavourable inputs)

| | OP-A P002 laser | OP-B P007 press | OP-C P007 laser |
|---|---|---|---|
| Good / started parts | 100 / 103 | 200 / 205 + 10 try-out | 200 / 205 |
| Material bought | 16.05 kg (1 sheet) | 0.745 kg strip | 2.40 kg (1 sheet) |
| Kept in good parts | 11.50 kg | 0.666 kg | 0.666 kg |
| Reusable offcut | 3.35 kg | — | 1.51 kg |
| Scrap (rejects, holes, skeleton, strip ends) | 1.20 kg | 0.080 kg | 0.228 kg |
| Electricity | **3.25 kWh** (1.25–12.0) | **0.81 kWh** (0.16–2.70) | **2.82 kWh** (1.02–11.1) |

**Laser (OP-A, OP-C).** Energy = parts × cut time × processing power + sheets × handling time × standby power.
- Processing power is 12 kW, where the TRUMPF and Salvagnini averages agree; both are measured over mixed production, not aluminium or copper.
- P002 speed is 9 m/min, from research on 2 mm AA6061 at 3 kW.
- No speed for 0.3 mm copper exists. P007 uses thicker-copper proxies: 3.4, 10 and 21 m/min.

**Press (OP-B).** Energy = motor rating × idle fraction × running time + strokes × shear work / efficiency.
- The shear force is only 2 kN (about 0.3 J per stroke), so **idle running is about 100 % of the energy**. Motor rating × run time would wrongly give 1.31 kWh.

**Die cost for OP-B** (`outputs/die_cost.csv`, nominal USD, not a quote):
- **From published price ranges** (6 die makers and suppliers): $1,000 / **$4,000** / $10,000.
- **From die set + parts + toolmaker hours:** $2,970 / $5,800 / $11,780. Uses a sourced $29.82/h wage; the 2.5–4× overhead and the 30–80 hours are assumptions.
- A compound blanking die would cost $2,000 / $7,000 / $25,000.

**P007 route choice.** The press uses about 3.5× less electricity and 3.2× less copper than the laser. But its 2 kWh saving is negligible beside the die, which adds about **$20 per good part** on this 200-part order (about $2 over 10 repeat orders). So the laser is the likely choice unless laser jobs cost over $20 per part more than stamping. Laser job prices were not researched.

**Largest uncertainties** (`outputs/sensitivity.csv`):
- **OP-A:** cutting speed (+63 / −23 %) and processing power (−15 / +44 %).
- **OP-B:** idle fraction (−50 / +67 %), setup time (−34 / +51 %) and feed rate (+43 / −24 %).
- **OP-C:** cutting speed (+81 / −22 %).
- **Material:** sheet size for the lasers (P002 +35 %; P007 −20 % to +122 %); route for the press (+33 % for a compound blank).

## 5. Reliability and gaps

1. **No primary document was opened**: page and PDF retrieval was blocked in this environment. Every value is a search-engine excerpt, flagged as such, and conflicts are kept rather than averaged (TRUMPF 12 vs 13 kW; SEYI 70–180 vs 35–65 spm; AIDA bolster dimensions). Verify first: SRC-TRU-02, SRC-SAL-01, SRC-AID-01 and SRC-BRU-01.
2. **Unsourced but dominant inputs:** laser standby power, press idle fraction, pierce and move times, feed rate, and the die overhead and hours. All are labelled in `data/assumptions.csv`.
3. **Not material- or part-specific:** OEM power averages; no 0.3 mm copper speed; no power data for the small-format cutters; die prices come from marketing pages, mixing US and Chinese pricing.
4. **Not modelled:** assist gas and compressed air, deburring, copper slitting, die wear, burr and distortion on thin copper, laser job pricing, and transport.

*AI disclosure:* produced with an AI coding assistant (Claude), which ran the searches and wrote the code and text. Excerpts are machine-summarised, not verified quotations; every value is attributed so it can be checked.
