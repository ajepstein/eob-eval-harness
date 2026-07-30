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
    response: ModelResponse
    scores: list[Score]
