"""Load the candidate pack, normalise units and reconcile BOM, CSV, JSON and STEP.

Nothing here invents a value. Blank BOM cells stay ``None``; every check that
cannot be decided is reported as ``unresolved`` rather than passed.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import units
from .step_geom import assembly_occurrences, part_geometry, read_entities

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MODULES_REQUIRED = 100  # START_HERE.txt and the xlsx Assignment sheet: "100 identical mechanical modules"
ASSEMBLY_ID = "STM-BM-6S4P-A"

# Field dictionary for the normalised BOM. 'source_field' is the column in
# bom.csv / cad_parts.csv; 'ambiguity' records what the input does not settle.
DATA_DICTIONARY = [
    # (field, source_field, meaning, unit, quantity_basis, ambiguity)
    ("part_id", "Part ID", "Stable part key shared by BOM, CAD files and placement JSON", "-", "per part definition", ""),
    ("parent_id", "Parent ID", "Assembly the quantity is counted against", "-", "-", "Root STM-BM-6S4P-A is a grouping, not a raw material (START_HERE)."),
    ("part_name_bom", "Part name", "Design-intent name in the BOM", "-", "-", "Names differ from STEP product names for P006, P007 (see reconciliation). A name does not prove a process route."),
    ("material_stated", "Material stated", "Material text as supplied", "-", "-", "Temper/grade unselected for Cu, PA66-GF30, laminate and steel; 6061-T6 has no specification standard (e.g. ASTM B209 vs EN 485)."),
    ("qty_per_parent", "Qty / parent", "Count of this part per one parent assembly", "each", "per 1 x STM-BM-6S4P-A", "Per parent, not per batch (xlsx Assignment sheet)."),
    ("qty_unit", "Qty unit", "Unit of qty_per_parent", "each", "-", ""),
    ("qty_total_demand", "(derived)", "qty_per_parent x 100 modules", "each", "per 100-module order", "Excludes spares, rejects and setup scrap (estimated separately)."),
    ("thickness_stated_value", "Thickness stated", "Stated sheet thickness, original value", "see thickness_stated_unit", "per part", "Mixed units in the BOM (cm for P002, µm for P006-P008). Blank = not supplied."),
    ("thickness_stated_unit", "Thickness unit", "Unit of thickness_stated_value as written", "-", "-", "'µm' arrives as U+00B5 micro sign."),
    ("thickness_mm", "(derived)", "Stated thickness converted to mm", "mm", "per part", "Nominal only; no tolerance supplied."),
    ("purchased_unit_mass_g", "Purchased unit mass + Mass unit", "Mass of a purchased item", "g", "per each", "Only P005; 69 g is an OEM MAXIMUM, not a measured mass (START_HERE)."),
    ("geometry_file", "Geometry file", "Relative STEP path in the pack", "-", "-", "Pack paths CAD/parts/Pxxx.step; files here are under data/raw/CAD."),
    ("requirement_note", "Requirement note", "Free-text gaps stated by Stonair", "-", "-", "Tolerance and surface finish not supplied for any part."),
    ("net_volume_mm3", "Net volume (mm³)", "CAD solid volume of ONE instance", "mm3", "per 1 part", "Net (finished) volume; not stock, not mass. Float noise in last digits."),
    ("bbox_x_mm / bbox_y_mm / bbox_z_mm", "Bounding X/Y/Z (mm)", "Axis-aligned extents of one part in its own STEP frame", "mm", "per 1 part", "Extents, not features; orientation is the part's local frame. 0.30000000000000004 is float noise for 0.3."),
    ("position_mm", "cad_instances.json position_mm", "Translation of the part's local origin in the assembly frame", "mm", "per instance", "Origin convention differs by part (corner for plates, centre for cells and fasteners). No rotation in the JSON; STEP confirms none."),
    ("instance_id", "cad_instances.json instance_id", "Unique occurrence of a part in the assembly", "-", "per instance", ""),
]


@dataclass
class BomRow:
    part_id: str
    parent_id: str
    part_name_bom: str
    material_stated: str
    qty_per_parent: float | None
    qty_unit: str
    thickness_stated_value: float | None
    thickness_stated_unit: str | None
    thickness_mm: float | None
    purchased_unit_mass_g: float | None
    geometry_file: str
    requirement_note: str
    net_volume_mm3: float | None = None
    bbox_x_mm: float | None = None
    bbox_y_mm: float | None = None
    bbox_z_mm: float | None = None

    @property
    def qty_total_demand(self) -> float | None:
        return None if self.qty_per_parent is None else self.qty_per_parent * MODULES_REQUIRED


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_bom(raw: Path = RAW) -> dict[str, BomRow]:
    rows = {}
    for r in _read_csv(raw / "bom.csv"):
        t_val = units.parse_blank(r["Thickness stated"])
        t_unit = r["Thickness unit"].strip() or None
        if (t_val is None) != (t_unit is None):
            raise ValueError(f"{r['Part ID']}: thickness value/unit present without the other")
        m_val = units.parse_blank(r["Purchased unit mass"])
        m_unit = r["Mass unit"].strip() or None
        if (m_val is None) != (m_unit is None):
            raise ValueError(f"{r['Part ID']}: mass value/unit present without the other")
        qty = units.parse_blank(r["Qty / parent"])
        if r["Qty unit"].strip() != "each":
            raise ValueError(f"{r['Part ID']}: unexpected qty unit {r['Qty unit']!r}")
        rows[r["Part ID"]] = BomRow(
            part_id=r["Part ID"], parent_id=r["Parent ID"], part_name_bom=r["Part name"],
            material_stated=r["Material stated"], qty_per_parent=qty, qty_unit=r["Qty unit"],
            thickness_stated_value=t_val, thickness_stated_unit=t_unit,
            thickness_mm=units.convert(t_val, t_unit, "mm") if t_val is not None else None,
            purchased_unit_mass_g=units.convert(m_val, m_unit, "g") if m_val is not None else None,
            geometry_file=r["Geometry file"], requirement_note=r["Requirement note"],
        )
    for r in _read_csv(raw / "cad_parts.csv"):
        b = rows[r["Part ID"]]
        b.net_volume_mm3 = units.parse_blank(r["Net volume (mm³)"])
        b.bbox_x_mm = units.parse_blank(r["Bounding X (mm)"])
        b.bbox_y_mm = units.parse_blank(r["Bounding Y (mm)"])
        b.bbox_z_mm = units.parse_blank(r["Bounding Z (mm)"])
    return rows


def load_instances(raw: Path = RAW) -> dict:
    with open(raw / "cad_instances.json", encoding="utf-8") as f:
        return json.load(f)


def load_xlsx_bom(raw: Path = RAW) -> list[tuple]:
    """Rows of the xlsx BOM sheet (header row onward), for the csv/xlsx consistency check."""
    import openpyxl
    ws = openpyxl.load_workbook(raw / "Stonair_Candidate_BOM.xlsx", read_only=True)["BOM"]
    rows = [r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]
    start = next(i for i, r in enumerate(rows) if r[0] == "Part ID")
    return rows[start + 1:]


@dataclass
class Check:
    check_id: str
    part_id: str
    description: str
    expected: str
    observed: str
    status: str  # pass | flag | fail | unresolved


def _close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=tol, abs_tol=1e-9)


def reconcile(raw: Path = RAW) -> list[Check]:
    bom = load_bom(raw)
    inst = load_instances(raw)
    checks: list[Check] = []

    # 1. csv vs xlsx BOM content
    try:
        xl = {r[0]: r for r in load_xlsx_bom(raw)}
        for pid, b in bom.items():
            x = xl.get(pid)
            same = x is not None and (x[4] == b.qty_per_parent) and (
                (x[6] is None and b.thickness_stated_value is None) or _close(float(x[6]), b.thickness_stated_value)
            ) and (x[7] or None) == b.thickness_stated_unit
            checks.append(Check("C01", pid, "bom.csv vs xlsx BOM sheet (qty, thickness, unit)",
                                "identical", "identical" if same else f"xlsx={x}", "pass" if same else "fail"))
    except ImportError:
        checks.append(Check("C01", "*", "bom.csv vs xlsx", "identical", "openpyxl not installed", "unresolved"))

    # 2. parent ids and assembly root
    for pid, b in bom.items():
        ok = b.parent_id == ASSEMBLY_ID == inst["assembly_id"]
        checks.append(Check("C02", pid, "parent id matches assembly root", ASSEMBLY_ID, b.parent_id, "pass" if ok else "fail"))

    # 3. instance count in JSON and STEP vs qty per parent
    json_counts: dict[str, int] = {}
    for i in inst["instances"]:
        json_counts[i["part_id"]] = json_counts.get(i["part_id"], 0) + 1
        if i["parent_id"] != ASSEMBLY_ID:
            checks.append(Check("C03b", i["part_id"], "instance parent", ASSEMBLY_ID, i["parent_id"], "fail"))
    occ = assembly_occurrences(raw / "CAD" / "Battery_Module_Assembly.step")
    step_counts: dict[str, int] = {}
    for o in occ:
        p = o.product_name.split("|")[0].strip()
        step_counts[p] = step_counts.get(p, 0) + 1
    for pid, b in bom.items():
        exp = int(b.qty_per_parent)
        obs = (json_counts.get(pid, 0), step_counts.get(pid, 0))
        checks.append(Check("C03", pid, "qty/parent vs JSON instances vs STEP occurrences",
                            str(exp), f"json={obs[0]}, step={obs[1]}", "pass" if obs == (exp, exp) else "fail"))
    total = len(inst["instances"])
    checks.append(Check("C03c", "*", "total instance count", "53 (START_HERE) and sum of BOM qty",
                        f"json={total}, step={len(occ)}, bom_sum={int(sum(b.qty_per_parent for b in bom.values()))}",
                        "pass" if total == len(occ) == 53 == sum(b.qty_per_parent for b in bom.values()) else "fail"))

    # 4. JSON placement vs STEP placement (translation) and rotation
    by_id = {o.instance_id: o for o in occ}
    mism = [i["instance_id"] for i in inst["instances"]
            if i["instance_id"] not in by_id
            or any(abs(a - b) > 1e-6 for a, b in zip(i["position_mm"], by_id[i["instance_id"]].translation))]
    checks.append(Check("C04", "*", "JSON position_mm equals STEP placement translation for all instances",
                        "0 mismatches", f"{len(mism)} mismatches {mism[:5]}", "pass" if not mism else "fail"))
    rot = [o.instance_id for o in occ if o.rotated]
    checks.append(Check("C04b", "*", "No rotated placements (JSON has none)", "0 rotated", f"{len(rot)} rotated",
                        "pass" if not rot else "flag"))
    checks.append(Check("C04c", "*", "JSON position_unit", "mm", inst.get("position_unit"),
                        "pass" if inst.get("position_unit") == "mm" else "fail"))

    # 5. STEP product names vs BOM names (design-intent text only)
    names = {o.product_name.split("|")[0].strip(): o.product_name.split("|", 1)[1].strip() for o in occ}
    for pid, b in bom.items():
        sn = names.get(pid, "")
        same = sn == b.part_name_bom
        rep = next((o.name_encoding_repairs for o in occ if o.product_name.startswith(pid)), 0)
        note = f" (STEP text double-encoded; repaired x{rep})" if rep else ""
        checks.append(Check("C05", pid, "BOM part name vs STEP product name", b.part_name_bom, sn + note,
                            "pass" if same and not rep else "flag"))

    # 6. per-part STEP geometry vs cad_parts.csv, units and thickness
    for pid, b in bom.items():
        g = part_geometry(raw / "CAD" / "parts" / f"{pid}.step")
        checks.append(Check("C06a", pid, "STEP length unit", "mm", str(g.unit), "pass" if g.unit == "mm" else "fail"))
        ext = g.extents
        ok = all(_close(a, c, 1e-9) for a, c in zip(ext, (b.bbox_x_mm, b.bbox_y_mm, b.bbox_z_mm)))
        checks.append(Check("C06b", pid, "STEP vertex bbox vs cad_parts.csv extents",
                            f"{b.bbox_x_mm} x {b.bbox_y_mm} x {b.bbox_z_mm}", f"{ext[0]} x {ext[1]} x {ext[2]}",
                            "pass" if ok else "fail"))
        if b.thickness_mm is not None:
            tmin = min(ext)
            checks.append(Check("C06c", pid, "stated thickness (converted to mm) vs smallest CAD extent",
                                f"{b.thickness_mm:g} mm ({b.thickness_stated_value:g} {b.thickness_stated_unit})",
                                f"{tmin:g} mm", "pass" if _close(tmin, b.thickness_mm, 1e-9) else "fail"))
            pv = g.prism_volume()
            ok = _close(pv, b.net_volume_mm3, 1e-9)
            checks.append(Check("C06d", pid, "flat prism + detected round holes reproduces CAD net volume",
                                f"{b.net_volume_mm3:.4f} mm3",
                                f"{pv:.4f} mm3; holes={[round(h.diameter, 3) for h in g.holes]}",
                                "pass" if ok else "flag"))

    # 7. fastener / hole relationship for P002 (cover holes vs screw positions)
    g2 = part_geometry(raw / "CAD" / "parts" / "P002.step")
    cover = next(i for i in inst["instances"] if i["part_id"] == "P002")
    hole_xy = {(round(cover["position_mm"][0] + h.x, 6), round(cover["position_mm"][1] + h.y, 6)) for h in g2.holes}
    screws_top = {(round(i["position_mm"][0], 6), round(i["position_mm"][1], 6)) for i in inst["instances"]
                  if i["part_id"] == "P009" and i["position_mm"][2] > cover["position_mm"][2]}
    checks.append(Check("C07", "P002", "cover hole centres coincide with screws above the cover",
                        str(sorted(screws_top)), str(sorted(hole_xy)), "pass" if hole_xy == screws_top else "fail"))
    shank = min(round(2 * float(b.rsplit(",", 1)[1].rstrip(")")), 6)
                for b in read_entities(raw / "CAD" / "parts" / "P009.step").values()
                if b.startswith("CIRCLE") and float(b.rsplit(",", 1)[1].rstrip(")")) >= 1.4)
    checks.append(Check("C07b", "P002", "cover hole Ø vs simplified M3 screw shank Ø (fit not specified)",
                        "hole > shank (clearance)", f"hole Ø{g2.holes[0].diameter:g} mm vs shank Ø{shank:g} mm",
                        "pass" if g2.holes[0].diameter > shank else "fail"))

    # 9. Local origin convention per part (STEP bbox_min) and world z-extent
    geo = {pid: part_geometry(raw / "CAD" / "parts" / f"{pid}.step") for pid in bom}
    for pid, g in geo.items():
        conv = "corner" if all(abs(v) < 1e-9 for v in g.bbox_min) else f"offset {tuple(round(v, 3) for v in g.bbox_min)}"
        checks.append(Check("C09", pid, "local origin convention (STEP bbox min)", "documented per part", conv,
                            "pass" if conv == "corner" else "flag"))

    def zspan(i):
        g = geo[i["part_id"]]
        z0 = i["position_mm"][2]
        return z0 + g.bbox_min[2], z0 + g.bbox_max[2]
    cells = [zspan(i) for i in inst["instances"] if i["part_id"] == "P005"]
    cell_lo, cell_hi = min(c[0] for c in cells), max(c[1] for c in cells)
    for i in inst["instances"]:
        if i["part_id"] in ("P006", "P007"):
            lo, hi = zspan(i)
            gap = cell_lo - hi if hi <= cell_lo else lo - cell_hi
            checks.append(Check("C10", i["instance_id"], "collector-to-cell-end gap (z) - electrical joining interface",
                                "contact or defined joint", f"gap {gap:.2f} mm (cells z {cell_lo:g}-{cell_hi:g})",
                                "flag" if gap > 1e-6 else "pass"))

    # 8. Blank-means-missing: densities, masses
    for pid, b in bom.items():
        if b.purchased_unit_mass_g is None and "Purchased" not in b.material_stated:
            checks.append(Check("C08", pid, "fabricated-part mass supplied?", "not supplied (blank)",
                                "None - must be derived from net volume x sourced density", "unresolved"))
    return checks


def normalised_bom_rows(raw: Path = RAW) -> list[dict]:
    out = []
    for b in load_bom(raw).values():
        out.append({
            "part_id": b.part_id, "parent_id": b.parent_id, "part_name_bom": b.part_name_bom,
            "material_stated": b.material_stated, "qty_per_parent": b.qty_per_parent, "qty_unit": b.qty_unit,
            "qty_total_demand_100_modules": b.qty_total_demand,
            "thickness_stated_value": b.thickness_stated_value, "thickness_stated_unit": b.thickness_stated_unit,
            "thickness_mm": b.thickness_mm, "purchased_unit_mass_g": b.purchased_unit_mass_g,
            "purchased_unit_mass_basis": "OEM maximum (START_HERE)" if b.purchased_unit_mass_g is not None else None,
            "net_volume_mm3": b.net_volume_mm3, "bbox_x_mm": b.bbox_x_mm, "bbox_y_mm": b.bbox_y_mm,
            "bbox_z_mm": b.bbox_z_mm, "geometry_file": b.geometry_file,
            "in_scope": b.part_id in ("P002", "P007"),
            "make_or_buy": "buy" if b.part_id in ("P005", "P009", "P010") else "make (route not supplied)",
            "requirement_note": b.requirement_note,
        })
    return out
