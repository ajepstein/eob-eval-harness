from pathlib import Path

import pytest

from harness.tasks import TaskLoadError, load_tasks
from harness.types import Task

FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_file_loads_with_correct_field_values():
    tasks = load_tasks(FIXTURES / "valid")

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, Task)
    assert task.id == "eob-fixture-valid"
    assert task.category == "clean"
    assert task.difficulty == "easy"
    assert task.edge_case is False
    assert "SAMPLE PAYER" in task.input
    assert task.expected == {
        "patient_name": "Test Patient",
        "date_of_service": "2026-05-01",
        "provider_npi": "1112223334",
        "payer_name": "Sample Payer",
        "member_id": "TP-000111",
        "cpt_codes": ["99213"],
        "billed_amount": 100.00,
        "patient_responsibility": 20.00,
    }


def test_duplicate_ids_raise():
    with pytest.raises(TaskLoadError, match="duplicate task id"):
        load_tasks(FIXTURES / "duplicate_ids")


def test_unknown_expected_key_raises():
    with pytest.raises(TaskLoadError, match="unknown key"):
        load_tasks(FIXTURES / "unknown_expected_key")


def test_missing_schema_field_raises():
    with pytest.raises(TaskLoadError, match="missing schema field"):
        load_tasks(FIXTURES / "missing_schema_field")


def test_malformed_date_raises():
    with pytest.raises(TaskLoadError, match="date_of_service"):
        load_tasks(FIXTURES / "malformed_date")


def test_malformed_npi_raises():
    with pytest.raises(TaskLoadError, match="provider_npi"):
        load_tasks(FIXTURES / "bad_npi")


def test_categories_filters_correctly():
    tasks = load_tasks(FIXTURES / "filtering", categories=["clean"])

    assert {t.id for t in tasks} == {"eob-f001", "eob-f002"}


def test_categories_filters_multiple():
    tasks = load_tasks(FIXTURES / "filtering", categories=["hard", "missing_field"])

    assert {t.id for t in tasks} == {"eob-f003", "eob-f004"}


def test_limit_truncates():
    tasks = load_tasks(FIXTURES / "filtering", limit=2)

    assert len(tasks) == 2
    assert [t.id for t in tasks] == ["eob-f001", "eob-f002"]


def test_limit_applies_after_category_filter():
    tasks = load_tasks(FIXTURES / "filtering", categories=["clean"], limit=1)

    assert [t.id for t in tasks] == ["eob-f001"]


def test_full_task_suite_loads_ten_tasks():
    tasks = load_tasks(Path(__file__).parent.parent / "tasks")

    assert len(tasks) == 10
    assert [t.id for t in tasks] == [f"eob-{i:03d}" for i in range(1, 11)]


def test_full_task_suite_category_distribution():
    tasks_dir = Path(__file__).parent.parent / "tasks"

    assert len(load_tasks(tasks_dir, categories=["clean"])) == 3
    assert len(load_tasks(tasks_dir, categories=["missing_field"])) == 3
    assert len(load_tasks(tasks_dir, categories=["format_variance"])) == 2
    assert len(load_tasks(tasks_dir, categories=["hard"])) == 2
