"""Threshold and regression gates, so the numbers become load-bearing.

A prompt change that degrades quality should fail CI rather than reach
production. That only works if the gate is trusted, and a gate is trusted
only if it stays quiet when nothing is wrong.

Hence `require_significant`. With ~78 tasks the run-to-run wobble on mean F1
is a couple of points; a gate that fires on any point decrease fires
constantly, gets muted within a week, and then protects nothing. Regression
gates can therefore demand that the paired interval exclude zero before
failing — fail on regressions the statistics can actually distinguish from
noise, and stay silent on the rest.

Thresholds live in version-controlled YAML rather than code so that changing
what counts as acceptable is a reviewable diff.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness.stats import Interval, paired_bootstrap_diff
from harness.store import DEFAULT_DB_PATH, load_run

# Every metric a gate may reference. An unknown name is a config error, not
# a silently-skipped gate.
KNOWN_METRICS = (
    "schema_pass_rate",
    "mean_f1",
    "cost_per_task",
    "p95_latency_ms",
    "hallucination_rate",
)
_BOUNDS = ("min", "max", "max_regression_vs_baseline")

# Metrics where a *higher* number is worse. Regression direction depends on
# this, and getting it backwards would make a cost blowout look like an
# improvement.
_LOWER_IS_BETTER = {"cost_per_task", "p95_latency_ms", "hallucination_rate"}


class GateConfigError(ValueError):
    """The gate configuration is malformed."""


@dataclass(frozen=True)
class Gate:
    metric: str
    bound: str
    threshold: float
    require_significant: bool = False


@dataclass(frozen=True)
class GateResult:
    metric: str
    bound: str
    passed: bool
    observed: float | None = None
    threshold: float | None = None
    baseline: float | None = None
    difference: Interval | None = None
    significant: bool | None = None
    skipped_reason: str | None = None
    driving_tasks: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.skipped_reason:
            return "skipped"
        return "pass" if self.passed else "FAIL"


@dataclass(frozen=True)
class GateReport:
    run_id: str
    baseline_run_id: str | None
    results: list[GateResult]

    @property
    def passed(self) -> bool:
        # A skipped gate does not fail the build; it is reported as skipped
        # so the reason is visible rather than silently treated as a pass.
        return all(r.passed or r.skipped_reason for r in self.results)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


# --- config ------------------------------------------------------------------


def load_config(path: str | Path) -> tuple[str | None, list[Gate]]:
    """Parse and validate the gate config, failing loudly on anything odd."""
    path = Path(path)
    if not path.exists():
        raise GateConfigError(f"{path}: gate config not found")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise GateConfigError(f"{path}: invalid YAML ({exc})") from exc

    if not isinstance(raw, dict):
        raise GateConfigError(f"{path}: expected a mapping at the top level")

    unknown_top = set(raw) - {"baseline_run_id", "gates"}
    if unknown_top:
        raise GateConfigError(f"{path}: unknown top-level key(s) {sorted(unknown_top)}")

    entries = raw.get("gates")
    if not isinstance(entries, list) or not entries:
        raise GateConfigError(f"{path}: 'gates' must be a non-empty list")

    gates: list[Gate] = []
    for index, entry in enumerate(entries):
        where = f"{path}: gates[{index}]"
        if not isinstance(entry, dict):
            raise GateConfigError(f"{where}: expected a mapping")
        metric = entry.get("metric")
        if metric not in KNOWN_METRICS:
            raise GateConfigError(
                f"{where}: unknown metric {metric!r}. "
                f"Valid metrics: {', '.join(KNOWN_METRICS)}"
            )
        present = [b for b in _BOUNDS if b in entry]
        if len(present) != 1:
            raise GateConfigError(
                f"{where}: expected exactly one of {', '.join(_BOUNDS)}, got "
                f"{present or 'none'}"
            )
        bound = present[0]
        threshold = entry[bound]
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise GateConfigError(f"{where}: {bound} must be a number, got {threshold!r}")
        require_significant = entry.get("require_significant", False)
        if not isinstance(require_significant, bool):
            raise GateConfigError(f"{where}: require_significant must be true or false")
        if require_significant and bound != "max_regression_vs_baseline":
            raise GateConfigError(
                f"{where}: require_significant only applies to "
                f"max_regression_vs_baseline gates"
            )
        unknown = set(entry) - {"metric", "require_significant", *_BOUNDS}
        if unknown:
            raise GateConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
        gates.append(Gate(metric, bound, float(threshold), require_significant))

    return raw.get("baseline_run_id"), gates


# --- metrics -----------------------------------------------------------------


def _per_task_f1(record) -> dict[str, float]:
    return {
        r.task_id: v
        for r in record.results
        if (v := next((s.value for s in r.scores if s.scorer == "fields"), None))
        is not None
    }


def metrics_for(record) -> dict[str, float]:
    """Every gate-able metric for a stored run. Defines nothing new."""
    meta = record.meta
    f1 = list(_per_task_f1(record).values())
    schema = [
        v
        for r in record.results
        if (v := next((s.value for s in r.scores if s.scorer == "schema"), None))
        is not None
    ]
    latencies = sorted(
        r.latency_ms for r in record.results if r.error is None and r.latency_ms
    )

    hallucinated = nullable = 0
    for result in record.results:
        detail = next((s.detail for s in result.scores if s.scorer == "fields"), None)
        for outcome in (detail or {}).get("fields", {}).values():
            if outcome == "fp_hallucinated":
                hallucinated += 1
                nullable += 1
            elif outcome == "tn":
                nullable += 1

    return {
        "schema_pass_rate": statistics.fmean(schema) if schema else float("nan"),
        "mean_f1": statistics.fmean(f1) if f1 else float("nan"),
        "cost_per_task": (
            meta.total_cost_usd / meta.task_count if meta.task_count else 0.0
        ),
        "p95_latency_ms": (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies
            else 0.0
        ),
        # Measured against the cases where the document genuinely had no
        # value — the only cases where one can be invented.
        "hallucination_rate": hallucinated / nullable if nullable else 0.0,
    }


# --- evaluation --------------------------------------------------------------


def _regression(
    gate: Gate, record, baseline_record, seed: int = 0
) -> GateResult:
    current = _per_task_f1(record)
    previous = _per_task_f1(baseline_record)
    shared = sorted(set(current) & set(previous))
    if not shared:
        return GateResult(
            gate.metric, gate.bound, passed=True,
            skipped_reason="no tasks in common with the baseline run",
        )

    before = [previous[t] for t in shared]
    after = [current[t] for t in shared]
    diff = paired_bootstrap_diff(before, after, seed=seed)

    # `diff` is after - before. For higher-is-better metrics a regression is
    # a negative difference; for cost and latency it is a positive one.
    regression = diff.point if gate.metric in _LOWER_IS_BETTER else -diff.point
    significant = not diff.spans_zero

    exceeds = regression > gate.threshold
    passed = not exceeds or (gate.require_significant and not significant)

    driving = sorted(
        (
            {
                "task_id": t,
                "baseline": previous[t],
                "current": current[t],
                "delta": current[t] - previous[t],
            }
            for t in shared
            if current[t] < previous[t]
        ),
        key=lambda d: d["delta"],
    )

    return GateResult(
        metric=gate.metric, bound=gate.bound, passed=passed,
        observed=regression, threshold=gate.threshold,
        baseline=statistics.fmean(before), difference=diff,
        significant=significant, driving_tasks=driving,
    )


def evaluate_gates(
    run_id: str,
    config_path: str | Path = "eval_gates.yaml",
    db_path: str | Path = DEFAULT_DB_PATH,
    seed: int = 0,
) -> GateReport:
    """Check one run against the configured gates."""
    baseline_run_id, gates = load_config(config_path)
    record = load_run(run_id, db_path)
    values = metrics_for(record)

    baseline_record = None
    baseline_error = None
    if baseline_run_id:
        try:
            baseline_record = load_run(baseline_run_id, db_path)
        except LookupError as exc:
            baseline_error = str(exc)

    results: list[GateResult] = []
    for gate in gates:
        if gate.bound == "max_regression_vs_baseline":
            if baseline_record is None:
                results.append(
                    GateResult(
                        gate.metric, gate.bound, passed=True,
                        threshold=gate.threshold,
                        skipped_reason=(
                            baseline_error
                            or "no baseline_run_id set — promote one with "
                            "scripts/set_baseline.py"
                        ),
                    )
                )
                continue
            results.append(_regression(gate, record, baseline_record, seed=seed))
            continue

        observed = values[gate.metric]
        if gate.bound == "min":
            passed = observed >= gate.threshold
        else:
            passed = observed <= gate.threshold
        results.append(
            GateResult(
                gate.metric, gate.bound, passed=passed,
                observed=observed, threshold=gate.threshold,
            )
        )

    return GateReport(record.meta.run_id, baseline_run_id, results)
