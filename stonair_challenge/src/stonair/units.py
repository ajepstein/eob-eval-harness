"""Explicit unit conversion.

Every quantity is converted through a table of factors to one SI-ish base per
dimension. Unknown units raise instead of passing through, and blank inputs
stay ``None`` (not supplied) rather than becoming zero.
"""
from __future__ import annotations

# unit -> (dimension, factor to base unit)
_UNITS: dict[str, tuple[str, float]] = {
    # length, base mm
    "mm": ("length", 1.0),
    "cm": ("length", 10.0),
    "m": ("length", 1000.0),
    "µm": ("length", 1e-3),
    "um": ("length", 1e-3),
    "μm": ("length", 1e-3),  # Greek mu (U+03BC) as well as micro sign (U+00B5)
    "in": ("length", 25.4),
    # mass, base g
    "g": ("mass", 1.0),
    "kg": ("mass", 1000.0),
    # power, base kW
    "W": ("power", 1e-3),
    "kW": ("power", 1.0),
    "hp": ("power", 0.745699872),
    # apparent power, base kVA. Kept separate from kW on purpose: a kVA supply
    # rating is not real power drawn, and converting needs a power factor.
    "kVA": ("apparent_power", 1.0),
    # force, base kN
    "kN": ("force", 1.0),
    "N": ("force", 1e-3),
    "tf": ("force", 9.80665),       # metric tonne-force
    "US ton": ("force", 8.896443),  # short ton-force
    # energy, base kWh
    "J": ("energy", 1 / 3.6e6),
    "kJ": ("energy", 1 / 3.6e3),
    "kWh": ("energy", 1.0),
    # speed, base mm/s
    "m/min": ("speed", 1000.0 / 60.0),
    "mm/s": ("speed", 1.0),
    # rate, base 1/min
    "spm": ("rate", 1.0),
    # time, base s
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    # stress, base MPa
    "MPa": ("stress", 1.0),
    "ksi": ("stress", 6.894757),
    # density, base g/cm3
    "g/cm3": ("density", 1.0),
    "g/cc": ("density", 1.0),
    "lb/in3": ("density", 27.679905),
    # counts and ratios
    "each": ("count", 1.0),
    "%": ("ratio", 0.01),
}

BASE_UNIT = {
    "length": "mm", "mass": "g", "power": "kW", "apparent_power": "kVA",
    "force": "kN", "energy": "kWh", "speed": "mm/s", "rate": "spm",
    "time": "s", "stress": "MPa", "density": "g/cm3", "count": "each",
    "ratio": "fraction",
}


class UnitError(ValueError):
    pass


def dimension(unit: str) -> str:
    try:
        return _UNITS[unit.strip()][0]
    except KeyError:
        raise UnitError(f"unknown unit {unit!r}") from None


def to_base(value: float | None, unit: str) -> float | None:
    """Convert to the base unit of the unit's dimension. ``None`` stays ``None``."""
    if value is None:
        return None
    dim, factor = _UNITS.get(unit.strip(), (None, None))
    if dim is None:
        raise UnitError(f"unknown unit {unit!r}")
    return float(value) * factor


def convert(value: float | None, from_unit: str, to_unit: str) -> float | None:
    if value is None:
        return None
    d1, d2 = dimension(from_unit), dimension(to_unit)
    if d1 != d2:
        raise UnitError(f"cannot convert {from_unit!r} ({d1}) to {to_unit!r} ({d2})")
    return to_base(value, from_unit) / _UNITS[to_unit.strip()][1]


def parse_blank(raw: str | None) -> float | None:
    """BOM cells: blank means not supplied, never zero."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    return float(s)
