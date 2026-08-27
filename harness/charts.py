"""Hand-written inline SVG charts.

Three chart types are needed — scatter with error bars, heatmap, and
horizontal bars with intervals — and writing the SVG directly is less code
than the matplotlib integration would be, with no font resolution, no
rasterisation, and nothing to load at view time. The output goes straight
into the HTML and works from `file://` with no network.

Two accessibility rules hold throughout:

**Colour never carries meaning alone.** Every colour distinction is paired
with a shape, a label, or an ordering, so the charts survive greyscale
printing and colour-blind readers.

**The palette is Okabe-Ito**, which is designed to stay distinguishable
under the common forms of colour vision deficiency.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

# Okabe-Ito. Black is reserved for text and axes.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
)
GRID = "#d6d8dc"
AXIS = "#3a3f46"
MUTED = "#6b7280"


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float, places: int = 3) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{places}f}"


@dataclass(frozen=True)
class ScatterPoint:
    label: str
    x: float
    y: float
    x_low: float | None = None
    x_high: float | None = None
    y_low: float | None = None
    y_high: float | None = None
    highlighted: bool = False


def scatter_with_error_bars(
    points: list[ScatterPoint],
    x_label: str = "cost per task (USD)",
    y_label: str = "mean F1",
    width: int = 720,
    height: int = 420,
    title: str = "Cost against quality",
) -> str:
    """Scatter with error bars on both axes.

    Pareto-optimal points are drawn as filled circles and dominated ones as
    hollow squares, so the distinction survives without colour.
    """
    if not points:
        return '<p class="empty">No runs to plot.</p>'

    pad_l, pad_r, pad_t, pad_b = 78, 28, 34, 62
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    xs = [p.x for p in points] + [p.x_low or p.x for p in points] + [p.x_high or p.x for p in points]
    ys = [p.y for p in points] + [p.y_low or p.y for p in points] + [p.y_high or p.y for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Pad the ranges so points never sit on the axis line.
    x_span = (x_max - x_min) or max(x_max, 1e-6)
    y_span = (y_max - y_min) or 0.1
    x_min, x_max = x_min - 0.15 * x_span, x_max + 0.15 * x_span
    y_min, y_max = max(0.0, y_min - 0.15 * y_span), min(1.0, y_max + 0.15 * y_span)

    def sx(value: float) -> float:
        return pad_l + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return pad_t + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-labelledby="scatter-title" class="chart">',
        f'<title id="scatter-title">{_esc(title)}</title>',
    ]

    # Gridlines and ticks.
    for i in range(5):
        y_value = y_min + (y_max - y_min) * i / 4
        y = sy(y_value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{_fmt(y_value, 2)}</text>'
        )
    for i in range(4):
        x_value = x_min + (x_max - x_min) * i / 3
        x = sx(x_value)
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 20}" text-anchor="middle" '
            f'class="tick">{x_value:.5f}</text>'
        )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="{AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        f'stroke="{AXIS}" stroke-width="1.5"/>'
    )

    for index, point in enumerate(points):
        colour = PALETTE[index % len(PALETTE)]
        x, y = sx(point.x), sy(point.y)

        if point.y_low is not None and point.y_high is not None:
            parts.append(
                f'<line x1="{x:.1f}" y1="{sy(point.y_low):.1f}" x2="{x:.1f}" '
                f'y2="{sy(point.y_high):.1f}" stroke="{colour}" stroke-width="1.5" '
                f'opacity="0.75"/>'
            )
        if point.x_low is not None and point.x_high is not None:
            parts.append(
                f'<line x1="{sx(point.x_low):.1f}" y1="{y:.1f}" '
                f'x2="{sx(point.x_high):.1f}" y2="{y:.1f}" stroke="{colour}" '
                f'stroke-width="1.5" opacity="0.75"/>'
            )

        if point.highlighted:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colour}" '
                f'stroke="#fff" stroke-width="1.5" data-point="{_esc(point.label)}"/>'
            )
        else:
            parts.append(
                f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" '
                f'fill="none" stroke="{colour}" stroke-width="2.5" '
                f'data-point="{_esc(point.label)}"/>'
            )
        parts.append(
            f'<text x="{x + 12:.1f}" y="{y - 9:.1f}" class="point-label">'
            f'{_esc(point.label)}</text>'
        )

    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" '
        f'text-anchor="middle" class="axis-label">{_esc(x_label)}</text>'
    )
    parts.append(
        f'<text x="18" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {pad_t + plot_h / 2:.1f})" '
        f'class="axis-label">{_esc(y_label)}</text>'
    )
    parts.append("</svg>")
    parts.append(
        '<p class="legend">Filled circle = on the cost/quality frontier. '
        'Hollow square = dominated (another run is no dearer and no worse). '
        'Bars are 95% intervals.</p>'
    )
    return "\n".join(parts)


def heatmap(
    row_labels: list[str],
    column_labels: list[str],
    values: list[list[float | None]],
    width: int = 720,
    cell_height: int = 44,
    title: str = "Score by category",
) -> str:
    """Models against categories.

    The numeric value is printed in every cell, so the colour ramp is a
    reading aid rather than the data itself.
    """
    if not row_labels or not column_labels:
        return '<p class="empty">Nothing to chart.</p>'

    label_w = 150
    cell_w = max(72, (width - label_w) // max(1, len(column_labels)))
    height = cell_height * (len(row_labels) + 1) + 12
    total_w = label_w + cell_w * len(column_labels)

    present = [v for row in values for v in row if v is not None]
    low, high = (min(present), max(present)) if present else (0.0, 1.0)
    if math.isclose(low, high):
        low, high = low - 0.01, high + 0.01

    parts = [
        f'<svg viewBox="0 0 {total_w} {height}" width="100%" role="img" '
        f'aria-labelledby="heat-title" class="chart">',
        f'<title id="heat-title">{_esc(title)}</title>',
    ]

    for c, column in enumerate(column_labels):
        parts.append(
            f'<text x="{label_w + c * cell_w + cell_w / 2:.1f}" y="24" '
            f'text-anchor="middle" class="tick">{_esc(column)}</text>'
        )

    for r, row_label in enumerate(row_labels):
        y = cell_height * (r + 1) - 8
        parts.append(
            f'<text x="{label_w - 12}" y="{y + cell_height / 2 - 2:.1f}" '
            f'text-anchor="end" class="tick">{_esc(row_label)}</text>'
        )
        for c in range(len(column_labels)):
            value = values[r][c] if c < len(values[r]) else None
            x = label_w + c * cell_w
            if value is None:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 3}" '
                    f'height="{cell_height - 6}" fill="#f3f4f6"/>'
                )
                parts.append(
                    f'<text x="{x + cell_w / 2 - 1.5:.1f}" '
                    f'y="{y + cell_height / 2 + 2:.1f}" text-anchor="middle" '
                    f'class="cell-null">n/a</text>'
                )
                continue
            t = (value - low) / (high - low)
            # Single-hue ramp: lightness carries the magnitude, and the
            # printed number carries the value.
            shade = int(238 - 150 * t)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 3}" '
                f'height="{cell_height - 6}" fill="rgb({shade},{min(255, shade + 26)},'
                f'{min(255, shade + 42)})"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2 - 1.5:.1f}" '
                f'y="{y + cell_height / 2 + 2:.1f}" text-anchor="middle" '
                f'class="{"cell-dark" if t > 0.55 else "cell-light"}">'
                f'{_fmt(value)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def barh_with_intervals(
    labels: list[str],
    points: list[float],
    lows: list[float],
    highs: list[float],
    width: int = 720,
    bar_height: int = 34,
    title: str = "Differences with 95% intervals",
    zero_line: bool = True,
) -> str:
    """Horizontal bars with intervals, for paired differences.

    A zero line is drawn because whether an interval crosses it is the whole
    question; bars whose interval spans zero are hatched so that reading is
    available without colour.
    """
    if not labels:
        return '<p class="empty">No comparisons to chart.</p>'

    label_w, pad_r, pad_t, pad_b = 210, 24, 18, 44
    plot_w = width - label_w - pad_r
    height = pad_t + bar_height * len(labels) + pad_b

    span = max([abs(v) for v in lows + highs + points] or [0.05]) * 1.2 or 0.05

    def sx(value: float) -> float:
        return label_w + plot_w / 2 + (value / span) * (plot_w / 2)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-labelledby="bar-title" class="chart">',
        f'<title id="bar-title">{_esc(title)}</title>',
        '<defs><pattern id="hatch" width="6" height="6" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="#ffffff"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{MUTED}" stroke-width="2.5"/>'
        "</pattern></defs>",
    ]

    for i, label in enumerate(labels):
        y = pad_t + i * bar_height
        mid = y + bar_height / 2 - 2
        spans_zero = lows[i] <= 0.0 <= highs[i]
        colour = MUTED if spans_zero else PALETTE[i % len(PALETTE)]

        parts.append(
            f'<text x="{label_w - 12}" y="{mid + 4:.1f}" text-anchor="end" '
            f'class="tick">{_esc(label)}</text>'
        )
        parts.append(
            f'<line x1="{sx(lows[i]):.1f}" y1="{mid:.1f}" x2="{sx(highs[i]):.1f}" '
            f'y2="{mid:.1f}" stroke="{colour}" stroke-width="3"/>'
        )
        for end in (lows[i], highs[i]):
            parts.append(
                f'<line x1="{sx(end):.1f}" y1="{mid - 6:.1f}" x2="{sx(end):.1f}" '
                f'y2="{mid + 6:.1f}" stroke="{colour}" stroke-width="2"/>'
            )
        fill = "url(#hatch)" if spans_zero else colour
        parts.append(
            f'<circle cx="{sx(points[i]):.1f}" cy="{mid:.1f}" r="6" fill="{fill}" '
            f'stroke="{colour}" stroke-width="2"/>'
        )

    if zero_line:
        parts.append(
            f'<line x1="{sx(0.0):.1f}" y1="{pad_t - 6}" x2="{sx(0.0):.1f}" '
            f'y2="{pad_t + bar_height * len(labels):.1f}" stroke="{AXIS}" '
            f'stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{sx(0.0):.1f}" y="{height - 22}" text-anchor="middle" '
            f'class="tick">0</text>'
        )
    parts.append("</svg>")
    parts.append(
        '<p class="legend">Hatched marker and grey bar = the interval spans '
        'zero, so the difference is not distinguishable from noise.</p>'
    )
    return "\n".join(parts)
