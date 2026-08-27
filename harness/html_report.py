"""A single self-contained HTML report.

One file, inlined CSS and SVG, no build step and nothing fetched at view
time. It opens from `file://`, commits to the repo, and attaches to an
email — which is the whole point: a report that needs a server is a report
nobody reads.

**The section order is the argument.** Calibration comes second, before any
score, so the reader learns how far the judge can be trusted *before* being
shown numbers that depend on it. Putting it at the end would let a skimmer
take the scores at face value, which is precisely the failure this project
is about.

This module renders what already exists. It defines no metric of its own.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from harness import __version__
from harness.charts import (
    ScatterPoint,
    barh_with_intervals,
    heatmap,
    scatter_with_error_bars,
)
from harness.prompts import judge_prompt_hash, load_prompt
from harness.stats import (
    FrontierPoint,
    bootstrap_ci,
    describe_difference,
    holm_correction,
    mcnemar,
    minimum_detectable_effect,
    paired_bootstrap_diff,
    pareto_frontier,
)
from harness.store import (
    DEFAULT_DB_PATH,
    _session,
    find_calibration,
    list_judge_calls,
    load_run,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 0


def _per_task(record, scorer: str = "fields") -> dict[str, float]:
    return {
        r.task_id: next((s.value for s in r.scores if s.scorer == scorer), None)
        for r in record.results
        if any(s.scorer == scorer for s in r.scores)
    }


def _field_outcomes(record) -> dict[str, Counter]:
    """Per-field outcome counts across the run."""
    out: dict[str, Counter] = defaultdict(Counter)
    for result in record.results:
        detail = next(
            (s.detail for s in result.scores if s.scorer == "fields"), None
        )
        for field, outcome in (detail or {}).get("fields", {}).items():
            out[field][outcome] += 1
    return out


def _documents() -> dict[str, dict]:
    docs = {}
    for path in sorted(Path("tasks").glob("**/*.yaml")):
        raw = yaml.safe_load(path.read_text())
        docs[raw.get("id")] = raw
    return docs


def _task_distribution(documents: dict[str, dict]) -> list[tuple[str, int]]:
    counts = Counter(d.get("category", "?") for d in documents.values())
    return sorted(counts.items())


def _headline(records) -> list[dict]:
    rows = []
    for record in records:
        scores = [v for v in _per_task(record).values() if v is not None]
        schema = [v for v in _per_task(record, "schema").values() if v is not None]
        interval = bootstrap_ci(
            scores, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED
        )
        meta = record.meta
        rows.append({
            "run_id": meta.run_id,
            "short_id": meta.run_id[:8],
            "adapter": meta.adapter,
            "model_id": meta.model_id,
            "n": meta.task_count,
            "schema_pass_rate": statistics.fmean(schema) if schema else float("nan"),
            "f1": interval.point,
            "f1_low": interval.low,
            "f1_high": interval.high,
            "cost": meta.total_cost_usd,
            "cost_per_task": (
                meta.total_cost_usd / meta.task_count if meta.task_count else 0.0
            ),
            "failures": meta.failures,
            "wall": meta.wall_seconds,
            "git_commit": (meta.git_commit or "unknown")[:8],
            "git_dirty": meta.git_dirty,
            "prompt": f"{meta.prompt_name}@{meta.prompt_hash}",
        })
    return rows


def _categories(records) -> tuple[list[str], list[str], list[list[float | None]]]:
    names = sorted({r.category for rec in records for r in rec.results if r.category})
    rows, matrix = [], []
    for record in records:
        buckets: dict[str, list[float]] = defaultdict(list)
        for result in record.results:
            value = next(
                (s.value for s in result.scores if s.scorer == "fields"), None
            )
            if value is not None and result.category:
                buckets[result.category].append(value)
        rows.append(f"{record.meta.adapter} ({record.meta.run_id[:8]})")
        matrix.append([
            statistics.fmean(buckets[c]) if buckets.get(c) else None for c in names
        ])
    return rows, names, matrix


def _fields_table(records) -> list[dict]:
    """Per-field breakdown, with hallucination called out separately.

    Hallucination rate is measured against the cases where the document
    genuinely had no value — the only cases where inventing one is possible.
    Dividing by all fields would dilute it into invisibility.
    """
    all_fields = sorted({f for rec in records for f in _field_outcomes(rec)})
    rows = []
    for field in all_fields:
        row = {"field": field, "per_run": []}
        for record in records:
            counts = _field_outcomes(record).get(field, Counter())
            nullable_cases = counts["tn"] + counts["fp_hallucinated"]
            row["per_run"].append({
                "adapter": record.meta.adapter,
                "correct": counts["tp"],
                "wrong": counts["fp_fn_wrong"],
                "missed": counts["fn_missed"],
                "hallucinated": counts["fp_hallucinated"],
                "nullable_cases": nullable_cases,
                "hallucination_rate": (
                    counts["fp_hallucinated"] / nullable_cases
                    if nullable_cases else None
                ),
            })
        rows.append(row)
    return rows


def _pairwise(records) -> tuple[list[dict], str]:
    """Every model pair, with Holm-corrected p-values across the family."""
    comparisons = []
    for a, b in combinations(records, 2):
        scores_a, scores_b = _per_task(a), _per_task(b)
        shared = sorted(
            t for t in set(scores_a) & set(scores_b)
            if scores_a[t] is not None and scores_b[t] is not None
        )
        if not shared:
            continue
        av = [scores_a[t] for t in shared]
        bv = [scores_b[t] for t in shared]
        diff = paired_bootstrap_diff(
            av, bv, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED
        )
        test = mcnemar([v == 1.0 for v in av], [v == 1.0 for v in bv])
        label_a = f"{a.meta.adapter} ({a.meta.run_id[:8]})"
        label_b = f"{b.meta.adapter} ({b.meta.run_id[:8]})"
        comparisons.append({
            "label": f"{label_a} vs {label_b}",
            "a": label_a, "b": label_b, "n": len(shared),
            "diff": diff.point, "low": diff.low, "high": diff.high,
            "spans_zero": diff.spans_zero,
            "verdict": describe_difference(diff, label_a, label_b),
            "mcnemar_b": test.b, "mcnemar_c": test.c,
            "p_raw": test.p_exact,
        })

    adjusted = holm_correction([c["p_raw"] for c in comparisons])
    for comparison, value in zip(comparisons, adjusted):
        comparison["p_holm"] = value

    chart = barh_with_intervals(
        [c["label"] for c in comparisons],
        [c["diff"] for c in comparisons],
        [c["low"] for c in comparisons],
        [c["high"] for c in comparisons],
    ) if comparisons else ""
    return comparisons, chart


def _failures(records, documents, db_path) -> list[dict]:
    """Every task any model got less than perfect, with full context."""
    judge_by_key: dict[tuple[str, str, str], dict] = {}
    for record in records:
        for call in list_judge_calls(record.meta.run_id, db_path):
            judge_by_key[(record.meta.run_id, call["task_id"], call["field"])] = call

    imperfect = set()
    for record in records:
        for task_id, value in _per_task(record).items():
            if value is not None and value < 1.0:
                imperfect.add(task_id)
        for result in record.results:
            if result.error:
                imperfect.add(result.task_id)

    gallery = []
    for task_id in sorted(imperfect):
        source = documents.get(task_id, {})
        entry = {
            "task_id": task_id,
            "category": source.get("category", "?"),
            "document": source.get("input", ""),
            "expected": source.get("expected", {}),
            "runs": [],
        }
        for record in records:
            result = next(
                (r for r in record.results if r.task_id == task_id), None
            )
            if result is None:
                continue
            detail = next(
                (s.detail for s in result.scores if s.scorer == "fields"), {}
            )
            bad = {
                f: o for f, o in (detail or {}).get("fields", {}).items()
                if o not in ("tp", "tn")
            }
            entry["runs"].append({
                "adapter": record.meta.adapter,
                "f1": next(
                    (s.value for s in result.scores if s.scorer == "fields"), None
                ),
                "error": result.error,
                "response_text": result.response_text or "",
                "bad_fields": bad,
                "judge": {
                    f: judge_by_key.get((record.meta.run_id, task_id, f))
                    for f in bad
                    if judge_by_key.get((record.meta.run_id, task_id, f))
                },
            })
        gallery.append(entry)
    return gallery


def render_report(
    run_ids: list[str],
    out_path: Path,
    calibration_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    """Render a self-contained HTML report for the given runs."""
    records = [load_run(run_id, db_path) for run_id in run_ids]
    documents = _documents()

    rubric = judge_prompt_hash()
    calibration = find_calibration(rubric, db_path=db_path)

    frontier_points = []
    for record, row in zip(records, _headline(records)):
        frontier_points.append(
            FrontierPoint(
                label=f"{record.meta.adapter} ({record.meta.run_id[:8]})",
                cost_per_task=row["cost_per_task"],
                quality=row["f1"],
            )
        )
    frontier = pareto_frontier(frontier_points)
    scatter = scatter_with_error_bars([
        ScatterPoint(
            label=point.label, x=point.cost_per_task, y=point.quality,
            y_low=row["f1_low"], y_high=row["f1_high"],
            highlighted=point.pareto_optimal,
        )
        for point, row in zip(frontier, _headline(records))
    ])

    rows, category_names, matrix = _categories(records)
    comparisons, comparison_chart = _pairwise(records)

    n_tasks = max((r.meta.task_count for r in records), default=0)
    baseline = statistics.fmean(
        [r.meta.mean_f1 for r in records]
    ) if records else 0.0

    template = Environment(undefined=StrictUndefined, autoescape=True).from_string(
        TEMPLATE_PATH.read_text()
    )
    rendered = template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        harness_version=__version__,
        headline=_headline(records),
        calibration=calibration,
        rubric_hash=rubric,
        rubric_text=load_prompt("judge_v1"),
        scatter_svg=scatter,
        frontier=frontier,
        heatmap_svg=heatmap(rows, category_names, matrix),
        category_names=category_names,
        category_rows=rows,
        category_matrix=matrix,
        fields=_fields_table(records),
        comparisons=comparisons,
        comparison_chart=comparison_chart,
        failures=_failures(records, documents, db_path),
        task_distribution=_task_distribution(documents),
        suite_size=len(documents),
        mde=minimum_detectable_effect(n_tasks, baseline) if n_tasks else float("nan"),
        n_tasks=n_tasks,
        baseline=baseline,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
        normalization_rules=Path("SCHEMA.md").read_text()
        if Path("SCHEMA.md").exists() else "",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path
