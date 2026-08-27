"""Four ways the judge — or the human labels — could be measuring the wrong thing.

Kappa says how often the judge agrees. It says nothing about *why*. A judge
can reach respectable agreement while actually scoring position, verbosity,
or family resemblance, and a labeller can drift over a long session. Each
test below isolates one of those.

All four return structured results with an explicit `fired` flag and a
threshold, so the report can state what was tested and what was found
rather than leaving the reader to interpret a bare correlation.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field

from harness.adapters.base import Adapter, AdapterError
from harness.extract import extract_json
from harness.prompts import load_prompt

# A judge whose verdict flips this often when the two values are swapped is
# partly scoring position rather than content.
POSITION_FLIP_THRESHOLD = 0.05
# Point-biserial correlation past this is a material length effect.
LENGTH_CORRELATION_THRESHOLD = 0.30


@dataclass(frozen=True)
class BiasResult:
    name: str
    fired: bool
    statistic: float
    threshold: float
    n: int
    detail: dict = field(default_factory=dict)
    interpretation: str = ""


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


# --- 1. position bias --------------------------------------------------------

_NEUTRAL_TEMPLATE = """You are comparing two values for one field of a health insurance
Explanation of Benefits, drawn from the document below.

Source document:
---
{document}
---

Field under review: {field}
Value A: {value_a}
Value B: {value_b}

Decide whether Value A and Value B name the same thing. Neither value is
authoritative; reason from the document.

Return only: {{"verdict": "equivalent" | "different", "reason": "<one sentence>"}}
"""


async def _neutral_verdict(
    adapter: Adapter, document: str, field: str, value_a: str, value_b: str
) -> str | None:
    prompt = (
        _NEUTRAL_TEMPLATE.replace("{document}", document)
        .replace("{field}", field)
        .replace("{value_a}", value_a)
        .replace("{value_b}", value_b)
    )
    try:
        response = await adapter.complete(prompt, max_tokens=512)
    except AdapterError:
        return None
    parsed, _ = extract_json(response.text)
    verdict = (parsed or {}).get("verdict")
    return verdict if verdict in ("equivalent", "different") else None


async def position_bias(
    adapter: Adapter, items: list[dict], documents: dict[str, str]
) -> BiasResult:
    """Does the verdict change when the two values swap places?

    The judge sees expected then predicted, always in that order. Here each
    item is asked twice with neutral labels — once in each order — and the
    flip rate measures how much of the verdict came from position rather
    than content.
    """
    flips = 0
    compared = 0
    examples = []

    for item in items:
        document = documents.get(item["task_id"], "")
        forward, reverse = await asyncio.gather(
            _neutral_verdict(adapter, document, item["field"],
                             item["expected"], item["predicted"]),
            _neutral_verdict(adapter, document, item["field"],
                             item["predicted"], item["expected"]),
        )
        if forward is None or reverse is None:
            continue
        compared += 1
        if forward != reverse:
            flips += 1
            if len(examples) < 5:
                examples.append(
                    {**item, "forward": forward, "reverse": reverse}
                )

    rate = flips / compared if compared else float("nan")
    fired = compared > 0 and rate > POSITION_FLIP_THRESHOLD
    return BiasResult(
        name="position",
        fired=fired,
        statistic=rate,
        threshold=POSITION_FLIP_THRESHOLD,
        n=compared,
        detail={"flips": flips, "examples": examples},
        interpretation=(
            f"{rate:.0%} of verdicts changed when the values were swapped. "
            + (
                "That is above the 5% threshold: the judge is partly scoring "
                "position rather than content, and the rubric should be "
                "revised to force symmetric reasoning."
                if fired
                else "That is within tolerance for order effects."
            )
        ) if compared else "No comparable items.",
    )


# --- 2. length bias ----------------------------------------------------------


def length_bias(items: list[dict]) -> BiasResult:
    """Are longer predictions systematically judged equivalent?

    If so the rubric is rewarding verbosity rather than correctness.
    Point-biserial correlation between verdict and predicted length.
    """
    scored = [
        (1.0 if i["verdict"] == "equivalent" else 0.0, float(len(i["predicted"] or "")))
        for i in items
        if i.get("verdict") in ("equivalent", "different")
    ]
    if len(scored) < 3:
        return BiasResult(
            name="length", fired=False, statistic=float("nan"),
            threshold=LENGTH_CORRELATION_THRESHOLD, n=len(scored),
            interpretation="Too few items to estimate a length effect.",
        )

    verdicts = [v for v, _ in scored]
    lengths = [l for _, l in scored]
    r = _pearson(verdicts, lengths)
    fired = r == r and abs(r) > LENGTH_CORRELATION_THRESHOLD

    eq = [l for v, l in scored if v == 1.0]
    diff = [l for v, l in scored if v == 0.0]
    return BiasResult(
        name="length",
        fired=fired,
        statistic=r,
        threshold=LENGTH_CORRELATION_THRESHOLD,
        n=len(scored),
        detail={
            "mean_length_equivalent": statistics.fmean(eq) if eq else None,
            "mean_length_different": statistics.fmean(diff) if diff else None,
        },
        interpretation=(
            f"Point-biserial r = {r:.2f} between 'equivalent' and predicted "
            f"length. "
            + (
                "Longer predictions are being judged equivalent more often; "
                "the rubric is rewarding verbosity."
                if fired
                else "No material length effect."
            )
        ),
    )


# --- 3. self-preference ------------------------------------------------------


def self_preference(
    verdicts_by_family: dict[str, list[dict]]
) -> BiasResult:
    """Does a judge favour outputs from its own model family?

    A judge that does cannot be used to compare those models against each
    other, and the remedy is a judge from a third family.

    `verdicts_by_family` maps judge family -> rows carrying `source_family`
    and `verdict`.
    """
    rates: dict[str, dict[str, float]] = {}
    for judge_family, rows in verdicts_by_family.items():
        own = [r for r in rows if r["source_family"] == judge_family]
        other = [r for r in rows if r["source_family"] != judge_family]
        if not own or not other:
            continue
        rates[judge_family] = {
            "own": sum(r["verdict"] == "equivalent" for r in own) / len(own),
            "other": sum(r["verdict"] == "equivalent" for r in other) / len(other),
            "n_own": len(own),
            "n_other": len(other),
        }

    if not rates:
        return BiasResult(
            name="self_preference", fired=False, statistic=float("nan"),
            threshold=0.10, n=0,
            interpretation="Needs judges from two families scoring both families.",
        )

    effects = {f: r["own"] - r["other"] for f, r in rates.items()}
    largest = max(effects.values(), key=abs)
    fired = abs(largest) > 0.10
    return BiasResult(
        name="self_preference",
        fired=fired,
        statistic=largest,
        threshold=0.10,
        n=sum(r["n_own"] + r["n_other"] for r in rates.values()),
        detail={"rates": rates, "effects": effects},
        interpretation=(
            f"Largest own-family advantage {largest:+.0%}. "
            + (
                "That is material: this judge should not be used to compare "
                "those model families. Use a judge from a third family."
                if fired
                else "No material self-preference."
            )
        ),
    )


# --- 4. human order and fatigue effects --------------------------------------


def human_drift(labels: list[dict]) -> BiasResult:
    """Did the labeller's own verdicts drift with position or time?

    This checks the human data, not the judge. If it drifts, the human
    ceiling is measured on inconsistent labels and that has to be said when
    reporting it.
    """
    rows = [
        r for r in labels if r.get("verdict") in ("equivalent", "different")
    ]
    if len(rows) < 5:
        return BiasResult(
            name="human_drift", fired=False, statistic=float("nan"),
            threshold=LENGTH_CORRELATION_THRESHOLD, n=len(rows),
            interpretation="Too few labels to assess drift.",
        )

    ordered = list(enumerate(rows))
    verdicts = [1.0 if r["verdict"] == "equivalent" else 0.0 for _, r in ordered]
    positions = [float(i) for i, _ in ordered]
    seconds = [float(r.get("seconds", 0.0)) for _, r in ordered]

    r_position = _pearson(positions, verdicts)
    r_time = _pearson(positions, seconds)
    fired = r_position == r_position and abs(r_position) > LENGTH_CORRELATION_THRESHOLD

    return BiasResult(
        name="human_drift",
        fired=fired,
        statistic=r_position,
        threshold=LENGTH_CORRELATION_THRESHOLD,
        n=len(rows),
        detail={
            "verdict_vs_position_r": r_position,
            "seconds_vs_position_r": r_time,
            "median_seconds": statistics.median(seconds) if seconds else None,
        },
        interpretation=(
            f"Verdict against queue position r = {r_position:.2f}; time per "
            f"label against position r = {r_time:.2f}. "
            + (
                "Verdicts drifted over the session, so the human ceiling rests "
                "on inconsistent labels and should be reported with that caveat."
                if fired
                else "No material drift over the session."
            )
        ),
    )
