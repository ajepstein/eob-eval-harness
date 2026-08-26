"""Scorer protocol and the safe runner around it.

Scorers are **sync and pure** — no network, no state, no mutation of the
task or response. Week 2's LLM judge needs to make API calls and will get
its own ``AsyncScorer`` protocol then; making everything async now would
complicate the runner for a requirement that doesn't exist yet.
"""

from __future__ import annotations

from typing import Protocol

from harness.types import ModelResponse, Score, Task


class Scorer(Protocol):
    name: str

    def score(self, task: Task, response: ModelResponse) -> Score: ...


def score_result(
    task: Task, response: ModelResponse, scorers: list[Scorer]
) -> list[Score]:
    """Run every scorer, converting a scorer crash into a failed Score.

    A buggy scorer must not abort a run that costs real money. The
    exception is recorded in ``Score.detail`` so the failure stays visible
    rather than being silently scored as zero.
    """
    scores: list[Score] = []
    for scorer in scorers:
        try:
            scores.append(scorer.score(task, response))
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            scores.append(
                Score(
                    scorer=getattr(scorer, "name", type(scorer).__name__),
                    value=0.0,
                    passed=False,
                    detail={"scorer_error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return scores
