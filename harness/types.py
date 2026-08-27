"""Core data types for the evaluation harness.

Frozen dataclasses only. This module must not import anything else from
the `harness` package, so it stays circular-import-proof for every other
module that needs these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    difficulty: str
    edge_case: bool
    input: str
    expected: dict[str, Any]
    # Set once a human has read the document alongside its answer key and
    # confirmed they agree. Unverified tasks still load — they just warn —
    # because a wrong answer key silently biases every score computed
    # against it, and that is worth being noisy about.
    verified: bool = False


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    finish_reason: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Score:
    scorer: str
    value: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    response: ModelResponse | None
    scores: list[Score]
    error: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class RunMeta:
    """One row of the `runs` table, plus aggregates recomputed from scores.

    The aggregates are derived rather than stored so that `--rescore` moves
    them automatically — a stored copy would silently go stale the moment a
    scorer was fixed.
    """

    run_id: str
    created_at: str
    git_commit: str | None
    git_dirty: bool
    adapter: str
    model_id: str
    prompt_name: str
    prompt_hash: str
    task_count: int
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    wall_seconds: float
    harness_version: str
    schema_pass_rate: float = 0.0
    mean_f1: float = 0.0
    failures: int = 0


@dataclass(frozen=True)
class StoredResult:
    task_id: str
    category: str
    difficulty: str
    edge_case: bool
    response_text: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    cached: bool
    error: str | None
    scores: list[Score]


@dataclass(frozen=True)
class RunRecord:
    meta: RunMeta
    results: list[StoredResult]


@dataclass(frozen=True)
class TaskDelta:
    """One task's score change between two runs."""

    task_id: str
    category: str
    scorer: str
    value_a: float | None
    value_b: float | None
    delta: float


@dataclass(frozen=True)
class RunDiff:
    run_id_a: str
    run_id_b: str
    scorer: str
    regressed: list[TaskDelta]
    improved: list[TaskDelta]
    unchanged: int
    only_in_a: list[str]
    only_in_b: list[str]
    mean_a: float
    mean_b: float
    mean_delta: float


@dataclass(frozen=True)
class RunSummary:
    results: list[TaskResult]
    adapter_name: str
    model_id: str
    prompt_name: str
    prompt_hash: str
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    wall_clock_seconds: float
    succeeded: int
    failed: int
    cached: int
    latency_p50_ms: float
    latency_p95_ms: float
    # Accuracy aggregates. Computed over results that produced a response;
    # tasks that errored are counted in `failed` rather than being folded in
    # as zeros, so infrastructure failure stays distinguishable from model
    # failure. Read these alongside `failed`, never on their own.
    schema_pass_rate: float = 0.0
    mean_f1: float = 0.0
    schema_pass_rate_by_category: dict[str, float] = field(default_factory=dict)
    mean_f1_by_category: dict[str, float] = field(default_factory=dict)
    # Judge spend is tracked apart from extraction spend: it is a cost of
    # running the harness, not of the model under test, and conflating them
    # would misreport the cost-per-task of every adapter.
    judge_cost_usd: float = 0.0
    judge_calls: int = 0
    mean_judge_f1: float = 0.0
    judge_prompt_hash: str | None = None
