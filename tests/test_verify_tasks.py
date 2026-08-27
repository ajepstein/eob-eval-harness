import sys
import warnings
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from verify_tasks import check_task, set_verified  # noqa: E402

from harness.tasks import load_tasks  # noqa: E402

GOOD = {
    "id": "eob-x",
    "category": "clean",
    "difficulty": "easy",
    "edge_case": False,
    "input": (
        "SAMPLE PAYER\nMember: Jane Doe\nMember ID: SP-1234\n"
        "Date of Service: 03/14/2026\nProvider NPI: 1234567890\n"
        "CPT 99213 $100.00\nCPT 85025 $40.00\n"
        "Total Billed: $140.00\nPatient Responsibility: $28.00\n"
    ),
    "expected": {
        "patient_name": "Jane Doe",
        "date_of_service": "2026-03-14",
        "provider_npi": "1234567890",
        "payer_name": "Sample Payer",
        "member_id": "SP-1234",
        "cpt_codes": ["99213", "85025"],
        "billed_amount": 140.00,
        "patient_responsibility": 28.00,
    },
}


def _task(**overrides):
    task = {**GOOD, "expected": dict(GOOD["expected"])}
    task["expected"].update(overrides)
    return task


def test_consistent_task_passes():
    assert check_task(GOOD) == []


def test_name_absent_from_document_is_caught():
    problems = check_task(_task(patient_name="Somebody Else"))

    assert any("patient_name" in p for p in problems)


def test_npi_absent_from_document_is_caught():
    problems = check_task(_task(provider_npi="9999999999"))

    assert any("provider_npi" in p for p in problems)


def test_ten_digit_number_when_npi_expected_null_is_flagged():
    # The hallucination-bait case: the answer key says "no NPI" but the
    # document contains something that looks exactly like one.
    problems = check_task(_task(provider_npi=None))

    assert any("null" in p and "10-digit" in p for p in problems)


def test_wrong_amount_is_caught():
    problems = check_task(_task(billed_amount=999.00))

    assert any("billed_amount" in p for p in problems)


def test_unparseable_date_is_caught():
    problems = check_task(_task(date_of_service="2026-01-01"))

    assert any("date_of_service" in p for p in problems)


def test_cpt_codes_out_of_document_order_are_caught():
    problems = check_task(_task(cpt_codes=["85025", "99213"]))

    assert any("order" in p for p in problems)


def test_missing_cpt_code_is_caught():
    problems = check_task(_task(cpt_codes=["99213", "00000"]))

    assert any("00000" in p for p in problems)


def test_amount_matches_with_thousands_separator():
    task = _task(billed_amount=1250.00)
    task["input"] = task["input"].replace("Total Billed: $140.00", "Total Billed: $1,250.00")

    assert not any("billed_amount" in p for p in check_task(task))


# --- verified flag round trip ------------------------------------------------


def _write_task(tmp_path: Path, verified_line: str = "") -> Path:
    path = tmp_path / "t.yaml"
    path.write_text(
        "id: eob-x\ncategory: clean\ndifficulty: easy\nedge_case: false\n"
        f"{verified_line}"
        "input: |2\n  LINE ONE\n     INDENTED LINE\n  LINE THREE\n"
        "expected:\n  patient_name: \"Jane Doe\"\n"
    )
    return path


def test_set_verified_inserts_flag(tmp_path: Path):
    path = _write_task(tmp_path)

    set_verified(path, True)

    assert yaml.safe_load(path.read_text())["verified"] is True


def test_set_verified_updates_existing_flag(tmp_path: Path):
    path = _write_task(tmp_path, verified_line="verified: true\n")

    set_verified(path, False)

    assert yaml.safe_load(path.read_text())["verified"] is False
    # Exactly one flag line, not two.
    assert path.read_text().count("verified:") == 1


def test_set_verified_preserves_document_indentation(tmp_path: Path):
    # yaml.dump would reflow the block scalar and silently reformat every
    # document in the suite.
    path = _write_task(tmp_path)
    before = yaml.safe_load(path.read_text())["input"]

    set_verified(path, True)

    assert yaml.safe_load(path.read_text())["input"] == before
    assert "   INDENTED LINE" in path.read_text()


# --- loader warning ----------------------------------------------------------


def test_load_tasks_warns_about_unverified_tasks(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(
        "id: eob-a\ncategory: clean\ndifficulty: easy\nedge_case: false\n"
        "input: |2\n  doc\n"
        "expected:\n  patient_name: \"A\"\n  date_of_service: \"2026-01-01\"\n"
        "  provider_npi: null\n  payer_name: \"P\"\n  member_id: null\n"
        "  cpt_codes: []\n  billed_amount: 1.00\n  patient_responsibility: 1.00\n"
    )

    with pytest.warns(UserWarning, match="unverified"):
        load_tasks(tmp_path)


def test_load_tasks_does_not_warn_when_all_verified(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(
        "id: eob-a\ncategory: clean\ndifficulty: easy\nedge_case: false\n"
        "verified: true\n"
        "input: |2\n  doc\n"
        "expected:\n  patient_name: \"A\"\n  date_of_service: \"2026-01-01\"\n"
        "  provider_npi: null\n  payer_name: \"P\"\n  member_id: null\n"
        "  cpt_codes: []\n  billed_amount: 1.00\n  patient_responsibility: 1.00\n"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tasks = load_tasks(tmp_path)

    assert tasks[0].verified is True


# --- the real suite ----------------------------------------------------------


def test_every_shipped_task_passes_automated_checks():
    # Guards the whole 40-task suite against answer-key drift.
    failures = {}
    for path in sorted(Path("tasks").glob("**/*.yaml")):
        problems = check_task(yaml.safe_load(path.read_text()))
        if problems:
            failures[path.name] = problems

    assert failures == {}
