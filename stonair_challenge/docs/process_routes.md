# Process routes for P002 and P007

Geometry facts below come from the STEP files (`src/stonair/step_geom.py`, checks C06–C07). Route choices are engineering judgement, and each one names its basis. Alternatives are kept separate. Only the **selected operation** for each part is researched and estimated.

## P002 Top cover: 6061-T6, 2 mm, 1 per module (100 needed)

**What the data shows.** The part is a flat 172 × 124 × 2 mm plate with four Ø3.4 mm through-holes at (6, 6), (6, 118), (166, 6) and (166, 118). A prism with those holes reproduces the CAD net volume exactly, so there are no other features: no flanges, bends, pockets or corner radii. The holes line up with the four M3 screws above the cover, and Ø3.4 is a clearance hole for the Ø3.0 shank. The cover's function is closure and mechanical protection, and it is clamped by screws into the tray bosses. Tolerance, finish and flatness are not supplied.

| Route | Operations | Why plausible | Why not selected |
|---|---|---|---|
| **R-A1 (selected)** | Buy 2 mm 6061-T6 sheet → **fiber-laser cut contour + 4 holes** → deburr (optional) | Flat profile with holes only. 100 parts fit on one standard sheet. No part-specific tooling. 2 mm Al is far inside every candidate's capability. | — |
| R-A2 | Shear blanks → CNC turret punch 4 holes (or drill) | Standard tooling (round Ø3.4 punch). No thermal edge. | Two machines and setups. Shearing accuracy is lower than laser. Stock tooling availability not checked. |
| R-A3 | Blanking + piercing die on a press | Lowest cost per part at volume. | A dedicated die for 100 parts is not justified. |
| R-A4 | CNC router / mill from sheet | Good edge finish. | Slower. Also a fixturing burden for thin plate. |
| R-A5 | Abrasive waterjet | No heat-affected zone, which matters if the T6 edge temper is critical. | Slower and wetter. No requirement for a cold-cut edge was stated. |

**Selected operation (OP-A):** 2-D fiber-laser cutting of 2 mm 6061-T6, with N₂ assist assumed. The open questions are edge HAZ on T6 and dross. Both depend on the unsupplied finish and tolerance requirements.

## P007 End collector plate: C11000 copper, 0.3 mm, 2 per module (200 needed)

**What the data shows.** The part is a flat 12 × 106 × 0.3 mm strip with one Ø5.5 mm hole at (6, 6). Ø5.5 is a medium clearance size for M5, which suggests a terminal stud or bolt, though the CAD does not model one. Prism plus hole reproduces the CAD volume exactly. The part's function is current collection from cell column 1 or 6 to a terminal. In the assembly the plate sits 0.8 mm above the cell tops (check C10), and START_HERE confirms that no joining interface is defined. Weld or joining features may be added later, which would change the route. Copper is highly reflective for 1 µm lasers, and 0.3 mm stock deforms easily.

| Route | Operations | Why plausible | Why not selected |
|---|---|---|---|
| **R-B1 (selected)** | Buy C11000 strip **slit to 12 mm** → **pierce Ø5.5 + cut-off at 106 mm** in a simple 2-station die on a gap-frame press | Rectangular part with constant width. The cut-off layout has no side scrap. Blanking is the usual route for collector and busbar strip, and the same family of die would also make P006. | — |
| R-B1' (scenario of R-B1) | Compound blank 12 × 106 with Ø5.5 from wider strip | Better edge control on all four sides. | About 33 % more strip input (see sensitivity). |
| **R-B2 (studied as OP-C)** | Fiber-laser cut from 0.3 mm Cu sheet | No die. Suits 200 prototype parts. TRUMPF, Amada, Mitsubishi and Salvagnini list Cu capability; small-format precision cutters (Han's Laser MPS0606D, IPG LaserCube) name copper. | Added after the first submission as a second estimated route for P007 (see OP-C below). Risks the max-thickness figures do not address: copper reflectivity at pierce, and thin-sheet distortion and burr. |
| R-B3 | Photochemical etching | Burr-free and stress-free. Good for thin copper. | Not researched; energy and chemistry data would also be needed. |
| R-B4 | Wire EDM stack | Possible for prototypes. | Slow, and over-precise for this part. |

**Selected operation (OP-B):** piercing and cut-off (blanking family) of 0.3 mm C11000 on a gap-frame mechanical or servo press. Die clearance for 0.3 mm copper is very small (roughly 15–30 µm per side, an assumption), so the die, not the press, is the capability constraint. That is why the press gates check force, bolster area and stroke range only.

**Second studied operation (OP-C):** fiber-laser cutting of the 12 × 106 mm outline and the Ø5.5 hole from 0.3 mm C11000 sheet, N₂ assist assumed. It is estimated on the same 200-part demand as OP-B so the two routes compare directly (`outputs/p007_route_comparison.csv`). The press route uses less electricity and less copper. The laser route avoids the die. A simple pierce + cut-off die is estimated at about $4,000 (range $1,000–$11,800; `outputs/die_cost.csv`), or about $20 per good part for one 200-part order. A laser job for the same 200 parts is estimated at $88–750, central $209 (`outputs/laser_job_cost.csv`). The die therefore pays back only after about 5,000 parts, and that is a lower bound. Laser is the choice at prototype volume. None of the online laser services checked lists 0.3 mm copper (thinnest 0.40 mm), so P007 needs a conventional job shop, or a design change to a stocked gauge.
