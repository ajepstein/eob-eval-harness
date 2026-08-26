"""SQLite persistence for eval runs.

Stdlib ``sqlite3``, no ORM. Three tables:

``runs``
    One row per invocation: what ran, against which model and prompt, at
    which commit, and what it cost.

``results``
    One row per task within a run, including **the model's raw response
    text**. Storing that is what makes ``rescore_run`` possible — a scorer
    bug can be fixed and every historical run corrected without re-paying
    for a single generation.

``scores``
    One row per (result, scorer). Normalized into its own table rather than
    widening ``results`` so that Week 2's judge scorer is an insert, not a
    schema migration.

Aggregates (schema pass rate, mean F1) are **not** stored on the run row.
They are recomputed from ``scores`` on read, so a rescore moves them
automatically instead of leaving a stale copy behind.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from harness import __version__
from harness.scorers.base import Scorer, score_result
from harness.types import (
    ModelResponse,
    RunDiff,
    RunMeta,
    RunRecord,
    RunSummary,
    Score,
    StoredResult,
    Task,
    TaskDelta,
)

DEFAULT_DB_PATH = "eval_runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT    NOT NULL,
    git_commit        TEXT,
    git_dirty         INTEGER NOT NULL,
    adapter           TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    prompt_name       TEXT    NOT NULL,
    prompt_hash       TEXT    NOT NULL,
    task_count        INTEGER NOT NULL,
    total_cost_usd    REAL    NOT NULL,
    total_tokens_in   INTEGER NOT NULL,
    total_tokens_out  INTEGER NOT NULL,
    wall_seconds      REAL    NOT NULL,
    harness_version   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id        TEXT    NOT NULL,
    category       TEXT    NOT NULL,
    difficulty     TEXT    NOT NULL,
    edge_case      INTEGER NOT NULL,
    response_text  TEXT,
    tokens_in      INTEGER NOT NULL,
    tokens_out     INTEGER NOT NULL,
    latency_ms     REAL    NOT NULL,
    cost_usd       REAL    NOT NULL,
    cached         INTEGER NOT NULL,
    error          TEXT,
    UNIQUE(run_id, task_id)
);

CREATE TABLE IF NOT EXISTS scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id    INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    scorer       TEXT    NOT NULL,
    value        REAL    NOT NULL,
    passed       INTEGER NOT NULL,
    detail_json  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run   ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_result ON scores(result_id);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at DESC);
"""


class RunNotFound(LookupError):
    """No run matches the given id or prefix."""


class AmbiguousRunId(LookupError):
    """A run id prefix matches more than one run."""


# --- connection -------------------------------------------------------------


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Transactional connection that is actually closed afterwards.

    ``with sqlite3.connect(...)`` commits but does *not* close the
    connection — a long-lived process calling save_run in a loop would leak
    a file handle every time. The inner ``with conn`` keeps the
    commit-on-success / rollback-on-exception behaviour.
    """
    conn = _connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the schema. Idempotent — safe to call on every startup."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _session(path) as conn:
        conn.executescript(_SCHEMA)


# --- git provenance ---------------------------------------------------------


def _git_info() -> tuple[str | None, bool]:
    """Current commit and whether tracked files are modified.

    "Dirty" deliberately ignores untracked files (``--untracked-files=no``).
    A run is reproducible if checking out the recorded commit reproduces the
    code, and untracked files are outside the commit either way. Counting
    them would make the flag permanently true in any working directory with
    stray scratch files, and a flag that is never false warns nobody.
    """
    def _git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return commit, bool(status)


# --- writing ----------------------------------------------------------------


def save_run(
    summary: RunSummary,
    tasks: list[Task],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Persist a run and its results. Returns the new run id."""
    init_db(db_path)
    run_id = str(uuid.uuid4())
    commit, dirty = _git_info()
    by_id = {t.id: t for t in tasks}

    with _session(db_path) as conn:
        conn.execute(
            """INSERT INTO runs (id, created_at, git_commit, git_dirty, adapter,
                                 model_id, prompt_name, prompt_hash, task_count,
                                 total_cost_usd, total_tokens_in, total_tokens_out,
                                 wall_seconds, harness_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                commit,
                int(dirty),
                summary.adapter_name,
                summary.model_id,
                summary.prompt_name,
                summary.prompt_hash,
                len(summary.results),
                summary.total_cost_usd,
                summary.total_tokens_in,
                summary.total_tokens_out,
                summary.wall_clock_seconds,
                __version__,
            ),
        )

        for result in summary.results:
            task = by_id.get(result.task_id)
            response = result.response
            cursor = conn.execute(
                """INSERT INTO results (run_id, task_id, category, difficulty,
                                        edge_case, response_text, tokens_in,
                                        tokens_out, latency_ms, cost_usd, cached, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    result.task_id,
                    task.category if task else "",
                    task.difficulty if task else "",
                    int(task.edge_case) if task else 0,
                    response.text if response else None,
                    response.tokens_in if response else 0,
                    response.tokens_out if response else 0,
                    response.latency_ms if response else 0.0,
                    response.cost_usd if response else 0.0,
                    int(result.cached),
                    result.error,
                ),
            )
            _insert_scores(conn, cursor.lastrowid, result.scores)

    return run_id


def _insert_scores(conn: sqlite3.Connection, result_id: int, scores: list[Score]) -> None:
    conn.executemany(
        "INSERT INTO scores (result_id, scorer, value, passed, detail_json) VALUES (?,?,?,?,?)",
        [
            (result_id, s.scorer, s.value, int(s.passed), json.dumps(s.detail, default=str))
            for s in scores
        ],
    )


# --- reading ----------------------------------------------------------------


def _resolve_run_id(conn: sqlite3.Connection, run_id: str) -> str:
    """Accept a full run id or any unambiguous prefix, git-style."""
    rows = conn.execute(
        "SELECT id FROM runs WHERE id = ? OR id LIKE ? || '%'", (run_id, run_id)
    ).fetchall()
    exact = [r["id"] for r in rows if r["id"] == run_id]
    if exact:
        return exact[0]
    if not rows:
        raise RunNotFound(f"No run matching {run_id!r}")
    if len(rows) > 1:
        raise AmbiguousRunId(
            f"{run_id!r} matches {len(rows)} runs: {', '.join(r['id'][:8] for r in rows)}"
        )
    return rows[0]["id"]


def _aggregates(conn: sqlite3.Connection, run_id: str) -> tuple[float, float, int]:
    """(schema pass rate, mean F1, failure count) recomputed from scores."""
    row = conn.execute(
        """SELECT
             AVG(CASE WHEN s.scorer = 'schema' THEN s.value END) AS schema_rate,
             AVG(CASE WHEN s.scorer = 'fields' THEN s.value END) AS mean_f1
           FROM results r LEFT JOIN scores s ON s.result_id = r.id
           WHERE r.run_id = ?""",
        (run_id,),
    ).fetchone()
    failures = conn.execute(
        "SELECT COUNT(*) AS n FROM results WHERE run_id = ? AND error IS NOT NULL",
        (run_id,),
    ).fetchone()["n"]
    return (row["schema_rate"] or 0.0), (row["mean_f1"] or 0.0), failures


def _meta_from_row(conn: sqlite3.Connection, row: sqlite3.Row) -> RunMeta:
    schema_rate, mean_f1, failures = _aggregates(conn, row["id"])
    return RunMeta(
        run_id=row["id"],
        created_at=row["created_at"],
        git_commit=row["git_commit"],
        git_dirty=bool(row["git_dirty"]),
        adapter=row["adapter"],
        model_id=row["model_id"],
        prompt_name=row["prompt_name"],
        prompt_hash=row["prompt_hash"],
        task_count=row["task_count"],
        total_cost_usd=row["total_cost_usd"],
        total_tokens_in=row["total_tokens_in"],
        total_tokens_out=row["total_tokens_out"],
        wall_seconds=row["wall_seconds"],
        harness_version=row["harness_version"],
        schema_pass_rate=schema_rate,
        mean_f1=mean_f1,
        failures=failures,
    )


def load_run(run_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> RunRecord:
    """Load a run and all its results. Accepts a full id or unique prefix."""
    with _session(db_path) as conn:
        full_id = _resolve_run_id(conn, run_id)
        run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (full_id,)).fetchone()
        meta = _meta_from_row(conn, run_row)

        results: list[StoredResult] = []
        for row in conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY task_id", (full_id,)
        ).fetchall():
            score_rows = conn.execute(
                "SELECT * FROM scores WHERE result_id = ? ORDER BY id", (row["id"],)
            ).fetchall()
            results.append(
                StoredResult(
                    task_id=row["task_id"],
                    category=row["category"],
                    difficulty=row["difficulty"],
                    edge_case=bool(row["edge_case"]),
                    response_text=row["response_text"],
                    tokens_in=row["tokens_in"],
                    tokens_out=row["tokens_out"],
                    latency_ms=row["latency_ms"],
                    cost_usd=row["cost_usd"],
                    cached=bool(row["cached"]),
                    error=row["error"],
                    scores=[
                        Score(
                            scorer=s["scorer"],
                            value=s["value"],
                            passed=bool(s["passed"]),
                            detail=json.loads(s["detail_json"]),
                        )
                        for s in score_rows
                    ],
                )
            )
    return RunRecord(meta=meta, results=results)


def list_runs(
    limit: int = 20,
    adapter: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[RunMeta]:
    """Most recent runs first, optionally filtered to one adapter."""
    if not Path(db_path).exists():
        return []
    query = "SELECT * FROM runs"
    params: list[object] = []
    if adapter is not None:
        query += " WHERE adapter = ?"
        params.append(adapter)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    params.append(limit)

    with _session(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [_meta_from_row(conn, row) for row in rows]


# --- comparison -------------------------------------------------------------


def compare_runs(
    run_id_a: str,
    run_id_b: str,
    scorer: str = "fields",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> RunDiff:
    """Per-task score changes from run A to run B.

    This is the shape Week 3C's CI gate consumes: which tasks regressed,
    which improved, and the aggregate delta. Positive `delta` means B scored
    higher than A.
    """
    record_a = load_run(run_id_a, db_path)
    record_b = load_run(run_id_b, db_path)

    def scores_by_task(record: RunRecord) -> dict[str, tuple[float | None, str]]:
        out: dict[str, tuple[float | None, str]] = {}
        for result in record.results:
            value = next((s.value for s in result.scores if s.scorer == scorer), None)
            out[result.task_id] = (value, result.category)
        return out

    a, b = scores_by_task(record_a), scores_by_task(record_b)
    shared = sorted(set(a) & set(b))

    regressed: list[TaskDelta] = []
    improved: list[TaskDelta] = []
    unchanged = 0

    for task_id in shared:
        value_a, category = a[task_id]
        value_b, _ = b[task_id]
        # A task that lost its score entirely (e.g. the run errored) is a
        # regression, not an absence — treat a missing side as 0.0 so it
        # cannot silently drop out of the comparison.
        delta = (value_b if value_b is not None else 0.0) - (
            value_a if value_a is not None else 0.0
        )
        entry = TaskDelta(
            task_id=task_id,
            category=category,
            scorer=scorer,
            value_a=value_a,
            value_b=value_b,
            delta=delta,
        )
        if delta < 0:
            regressed.append(entry)
        elif delta > 0:
            improved.append(entry)
        else:
            unchanged += 1

    regressed.sort(key=lambda d: d.delta)
    improved.sort(key=lambda d: -d.delta)

    def mean(values: list[float | None]) -> float:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else 0.0

    mean_a = mean([a[t][0] for t in shared])
    mean_b = mean([b[t][0] for t in shared])

    return RunDiff(
        run_id_a=record_a.meta.run_id,
        run_id_b=record_b.meta.run_id,
        scorer=scorer,
        regressed=regressed,
        improved=improved,
        unchanged=unchanged,
        only_in_a=sorted(set(a) - set(b)),
        only_in_b=sorted(set(b) - set(a)),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=mean_b - mean_a,
    )


# --- rescoring --------------------------------------------------------------


def rescore_run(
    run_id: str,
    tasks: list[Task],
    scorers: list[Scorer],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> RunRecord:
    """Re-run scorers over a run's stored response text, in place.

    The payoff for storing raw text: a scorer fix corrects historical runs
    at zero API cost. Results whose response text is missing (the task
    errored) keep their existing scores.
    """
    by_id = {t.id: t for t in tasks}

    with _session(db_path) as conn:
        full_id = _resolve_run_id(conn, run_id)
        rows = conn.execute(
            "SELECT * FROM results WHERE run_id = ?", (full_id,)
        ).fetchall()

        for row in rows:
            task = by_id.get(row["task_id"])
            if task is None or row["response_text"] is None:
                continue

            response = ModelResponse(
                text=row["response_text"],
                model_id="",
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                latency_ms=row["latency_ms"],
                cost_usd=row["cost_usd"],
                finish_reason="",
                raw={},
            )
            conn.execute("DELETE FROM scores WHERE result_id = ?", (row["id"],))
            _insert_scores(conn, row["id"], score_result(task, response, scorers))

    return load_run(full_id, db_path)
