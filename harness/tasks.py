"""Loading and validation of the EOB task suite."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from harness.normalize import SCHEMA_FIELDS
from harness.types import Task

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NPI_RE = re.compile(r"^\d{10}$")


class TaskLoadError(ValueError):
    """Raised when a task YAML file fails validation, naming the file."""


class _ExpectedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str | None
    date_of_service: str | None
    provider_npi: str | None
    payer_name: str | None
    member_id: str | None
    cpt_codes: list[str] | None
    billed_amount: float | None
    patient_responsibility: float | None

    @field_validator("date_of_service")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE_RE.match(v):
            raise ValueError(
                f"date_of_service must match YYYY-MM-DD, got {v!r}"
            )
        return v

    @field_validator("provider_npi")
    @classmethod
    def _check_npi(cls, v: str | None) -> str | None:
        if v is not None and not _NPI_RE.match(v):
            raise ValueError(
                f"provider_npi must be exactly 10 digits, got {v!r}"
            )
        return v


class _TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    difficulty: str
    edge_case: bool
    input: str
    expected: _ExpectedModel
    verified: bool = False


def _load_one(file_path: Path) -> Task:
    try:
        raw = yaml.safe_load(file_path.read_text())
    except yaml.YAMLError as exc:
        raise TaskLoadError(f"{file_path}: invalid YAML ({exc})") from exc

    if not isinstance(raw, dict):
        raise TaskLoadError(f"{file_path}: file does not contain a YAML mapping")

    for key in ("id", "category", "difficulty", "edge_case", "input", "expected"):
        if key not in raw:
            raise TaskLoadError(f"{file_path}: missing required key {key!r}")
        if raw[key] == "" or raw[key] is None:
            raise TaskLoadError(f"{file_path}: required key {key!r} is empty")

    expected_raw = raw.get("expected")
    if isinstance(expected_raw, dict):
        unknown = set(expected_raw) - set(SCHEMA_FIELDS)
        if unknown:
            raise TaskLoadError(
                f"{file_path}: expected has unknown key(s) {sorted(unknown)} "
                f"not present in the schema"
            )
        missing = set(SCHEMA_FIELDS) - set(expected_raw)
        if missing:
            raise TaskLoadError(
                f"{file_path}: expected is missing schema field(s) {sorted(missing)}"
            )

    try:
        model = _TaskModel.model_validate(raw)
    except ValidationError as exc:
        raise TaskLoadError(f"{file_path}: {exc}") from exc

    return Task(
        id=model.id,
        category=model.category,
        difficulty=model.difficulty,
        edge_case=model.edge_case,
        input=model.input,
        expected=model.expected.model_dump(),
        verified=model.verified,
    )


def load_tasks(
    path: str | Path,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[Task]:
    """Load and validate all task YAML files under `path`.

    Files are discovered recursively (`**/*.yaml`). Results are sorted by
    task id. Raises `TaskLoadError` on any validation failure, naming the
    offending file.
    """
    root = Path(path)
    files = sorted(root.glob("**/*.yaml"))

    tasks: list[Task] = []
    seen_ids: dict[str, Path] = {}
    for file_path in files:
        task = _load_one(file_path)
        if task.id in seen_ids:
            raise TaskLoadError(
                f"{file_path}: duplicate task id {task.id!r} "
                f"(already defined in {seen_ids[task.id]})"
            )
        seen_ids[task.id] = file_path
        tasks.append(task)

    tasks.sort(key=lambda t: t.id)

    if categories is not None:
        tasks = [t for t in tasks if t.category in categories]

    if limit is not None:
        tasks = tasks[:limit]

    unverified = [t.id for t in tasks if not t.verified]
    if unverified:
        shown = ", ".join(unverified[:5])
        more = f" (+{len(unverified) - 5} more)" if len(unverified) > 5 else ""
        warnings.warn(
            f"{len(unverified)} of {len(tasks)} tasks are unverified: {shown}{more}. "
            f"Run scripts/verify_tasks.py — an unreviewed answer key biases "
            f"every score computed against it.",
            UserWarning,
            stacklevel=2,
        )

    return tasks
