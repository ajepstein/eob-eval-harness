"""Minimal STEP (ISO 10303-21) reader for the supplied files.

It is not a general CAD kernel. It reads just enough to check what the CSVs
claim about the two scoped parts:

* the length unit declared in the file,
* the bounding box of the B-rep vertices (VERTEX_POINT only, so axis
  placement points that sit outside the solid are ignored),
* circular through-holes (CYLINDRICAL_SURFACE radius and axis location),
* assembly occurrences: NEXT_ASSEMBLY_USAGE_OCCURRENCE names and the
  translation of the placement each one uses.

Rotations in assembly placements are read only to flag them, not applied.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_ENTITY = re.compile(r"#(\d+)\s*=\s*(.*)", re.S)
_REF = re.compile(r"#(\d+)")
_NUM = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\.?"


def read_entities(path: str | Path) -> dict[int, str]:
    text = Path(path).read_text(encoding="latin-1")
    data = text.split("DATA;", 1)[1].rsplit("ENDSEC;", 1)[0]
    out: dict[int, str] = {}
    # Entities end with ';'. Strings in these files contain no ';'.
    for chunk in data.split(";"):
        chunk = " ".join(chunk.split())
        m = _ENTITY.match(chunk)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def _floats(s: str) -> list[float]:
    return [float(x.rstrip(".")) for x in re.findall(_NUM, s)]


def _point(ents: dict[int, str], ref: int) -> tuple[float, ...]:
    body = ents[ref]
    assert body.startswith("CARTESIAN_POINT"), body
    inner = body[body.index("(", body.index("(") + 1) + 1: body.rindex(")")]
    return tuple(_floats(inner))


def length_unit(ents: dict[int, str]) -> str | None:
    for body in ents.values():
        if "LENGTH_UNIT" in body and "SI_UNIT" in body:
            if ".MILLI.,.METRE." in body.replace(" ", ""):
                return "mm"
            if "$,.METRE." in body.replace(" ", ""):
                return "m"
    for body in ents.values():
        if "CONVERSION_BASED_UNIT" in body and "INCH" in body.upper():
            return "in"
    return None


@dataclass
class Hole:
    x: float
    y: float
    radius: float

    @property
    def diameter(self) -> float:
        return 2 * self.radius


@dataclass
class PartGeometry:  # noqa: D101
    unit: str | None
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    holes: list[Hole] = field(default_factory=list)
    n_faces: int = 0
    other_curved_surfaces: int = 0

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(round(b - a, 9) for a, b in zip(self.bbox_min, self.bbox_max))

    def prism_volume(self) -> float:
        """Volume if the part is a flat prism of its bounding box with the
        detected circular through-holes removed (only valid for flat parts)."""
        x, y, t = self.extents
        return x * y * t - sum(math.pi * h.radius ** 2 * t for h in self.holes)

    def outer_perimeter(self) -> float:
        x, y, _ = self.extents
        return 2 * (x + y)

    def hole_perimeter(self) -> float:
        return sum(math.pi * h.diameter for h in self.holes)


def part_geometry(path: str | Path) -> PartGeometry:
    ents = read_entities(path)
    pts = []
    for body in ents.values():
        if body.startswith("VERTEX_POINT"):
            pts.append(_point(ents, int(_REF.findall(body)[0])))
    # Round edges have no vertex at their extremes: add the extreme points of
    # every circle whose axis is parallel to a coordinate axis.
    for body in ents.values():
        if body.startswith("CIRCLE"):
            ax_ref = int(_REF.findall(body)[0])
            r = _floats(body.rsplit(",", 1)[1])[0]
            if not ents[ax_ref].startswith("AXIS2_PLACEMENT_3D"):
                continue  # 2-D circles live in parameter space (pcurves)
            refs = [int(x) for x in _REF.findall(ents[ax_ref])]
            c = _point(ents, refs[0])
            d = tuple(_floats(ents[refs[1]].split("(", 2)[2]))
            k = next((i for i in range(3) if abs(abs(d[i]) - 1) < 1e-12), None)
            if k is None:
                continue
            for i in range(3):
                if i != k:
                    for sgn in (-1, 1):
                        q = list(c)
                        q[i] += sgn * r
                        pts.append(tuple(q))
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    holes: dict[tuple[float, float, float], Hole] = {}
    other = 0
    for body in ents.values():
        if body.startswith("CYLINDRICAL_SURFACE"):
            ax_ref = int(_REF.findall(body)[0])
            radius = _floats(body.rsplit(",", 1)[1])[0]
            ax = ents[ax_ref]
            loc = _point(ents, int(_REF.findall(ax)[0]))
            key = (round(loc[0], 6), round(loc[1], 6), round(radius, 6))
            holes[key] = Hole(loc[0], loc[1], radius)
        elif body.startswith(("CONICAL_SURFACE", "SPHERICAL_SURFACE",
                              "TOROIDAL_SURFACE", "B_SPLINE_SURFACE")):
            other += 1
    n_faces = sum(1 for b in ents.values() if b.startswith("ADVANCED_FACE"))
    return PartGeometry(length_unit(ents), lo, hi, sorted(holes.values(), key=lambda h: (h.x, h.y)),
                        n_faces, other)


def fix_mojibake(text: str) -> tuple[str, int]:
    """Undo repeated UTF-8-read-as-Latin-1 encoding. Returns (text, rounds repaired)."""
    rounds = 0
    while True:
        try:
            nxt = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text, rounds
        if nxt == text:
            return text, rounds
        text, rounds = nxt, rounds + 1


@dataclass
class Occurrence:
    instance_id: str
    product_name: str
    translation: tuple[float, float, float]
    rotated: bool
    name_encoding_repairs: int = 0


def assembly_occurrences(path: str | Path) -> list[Occurrence]:
    ents = read_entities(path)
    products = {}
    for eid, body in ents.items():
        if body.startswith("PRODUCT_DEFINITION("):
            form = int(_REF.findall(body)[0])
            prod = int(_REF.findall(ents[form])[0])
            products[eid] = fix_mojibake(re.match(r"PRODUCT\('([^']*)'", ents[prod]).group(1))
    occ = []
    for body in ents.values():
        if not body.startswith("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            continue
        rel_ref, pds_ref = map(int, _REF.findall(body)[:2])
        idt_ref = int(re.search(r"WITH_TRANSFORMATION\(#(\d+)\)", ents[rel_ref]).group(1))
        to_ax = int(_REF.findall(ents[idt_ref])[1])
        loc_ref, z_ref, x_ref = map(int, _REF.findall(ents[to_ax])[:3])
        loc = _point(ents, loc_ref)
        z_dir = tuple(_floats(ents[z_ref].split("(", 2)[2]))
        x_dir = tuple(_floats(ents[x_ref].split("(", 2)[2]))
        rotated = not (z_dir == (0.0, 0.0, 1.0) and x_dir == (1.0, 0.0, 0.0))
        nauo_ref = int(_REF.findall(ents[pds_ref])[0])
        nauo = ents[nauo_ref]
        name = re.search(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','([^']*)'", nauo).group(1)
        child = int(_REF.findall(nauo)[1])
        pname, repairs = products[child]
        occ.append(Occurrence(name, pname, tuple(loc), rotated, repairs))
    return occ
