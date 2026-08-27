"""The report must be self-contained, honest about calibration, and safe."""

import re
from pathlib import Path

import pytest

from harness.charts import (
    ScatterPoint,
    barh_with_intervals,
    heatmap,
    scatter_with_error_bars,
)
from harness.html_report import render_report
from harness.store import init_db, save_calibration, save_run
from harness.calibration import AgreementReport
from harness.types import ModelResponse, RunSummary, Score, Task, TaskResult


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "r.db"
    init_db(path)
    return path


def _task(i: int, category: str = "clean") -> Task:
    return Task(id=f"eob-{i:03d}", category=category, difficulty="easy",
                edge_case=False, input=f"DOCUMENT {i}", expected={})


def _result(i: int, f1: float, text: str = '{"a": 1}') -> TaskResult:
    return TaskResult(
        task_id=f"eob-{i:03d}",
        response=ModelResponse(text=text, model_id="m", tokens_in=10, tokens_out=5,
                               latency_ms=100.0, cost_usd=0.001,
                               finish_reason="end_turn", raw={}),
        scores=[
            Score(scorer="schema", value=1.0, passed=True, detail={"reason": "ok"}),
            Score(scorer="fields", value=f1, passed=f1 == 1.0,
                  detail={"f1": f1, "fields": {
                      "patient_name": "tp" if f1 == 1.0 else "fp_fn_wrong",
                      "provider_npi": "tn",
                  }}),
        ],
    )


def _save(db: Path, adapter: str, f1s: list[float], cost: float = 0.01) -> str:
    tasks = [_task(i) for i in range(len(f1s))]
    summary = RunSummary(
        results=[_result(i, f) for i, f in enumerate(f1s)],
        adapter_name=adapter, model_id=f"{adapter}-1",
        prompt_name="extract_v1", prompt_hash="ph123",
        total_cost_usd=cost, total_tokens_in=100, total_tokens_out=50,
        wall_clock_seconds=1.0, succeeded=len(f1s), failed=0, cached=0,
        latency_p50_ms=100.0, latency_p95_ms=120.0,
    )
    return save_run(summary, tasks, db_path=db)


def _render(db: Path, runs: list[str], tmp_path: Path) -> str:
    out = render_report(runs, tmp_path / "out.html", db_path=db)
    return out.read_text()


# --- rendering ---------------------------------------------------------------


def test_report_renders_from_fixture_runs(db: Path, tmp_path: Path):
    a = _save(db, "anthropic", [1.0, 0.8, 0.9])
    b = _save(db, "openai", [1.0, 0.9, 0.9])

    html = _render(db, [a, b], tmp_path)

    assert "<!doctype html>" in html.lower()
    assert "Headline comparison" in html


def test_single_run_renders_without_comparison_sections(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0, 0.8])

    html = _render(db, [run], tmp_path)

    assert "nothing to compare against" in html
    assert "Methods" in html  # the rest of the report still renders


def test_all_nine_sections_are_present(db: Path, tmp_path: Path):
    a = _save(db, "anthropic", [1.0, 0.5])
    b = _save(db, "openai", [1.0, 0.9])

    html = _render(db, [a, b], tmp_path)

    for section in ("Headline comparison", "Cost against quality", "By category",
                    "By field", "Paired comparisons", "Failures", "Methods"):
        assert section in html, f"missing section: {section}"


# --- self-containment --------------------------------------------------------


def test_report_is_self_contained(db: Path, tmp_path: Path):
    # It must open from file:// with no network. Any external src/href that
    # rendering depends on breaks that.
    run = _save(db, "anthropic", [1.0, 0.9])

    html = _render(db, [run], tmp_path)

    external = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    assert external == []


def test_report_loads_no_external_scripts_or_stylesheets(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0])

    html = _render(db, [run], tmp_path)

    assert "<script src=" not in html
    assert "<link " not in html


# --- calibration banner ------------------------------------------------------


def test_missing_calibration_produces_a_prominent_warning(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0, 0.9])

    html = _render(db, [run], tmp_path)

    assert "No calibration for this judge rubric" in html
    assert "unknown reliability" in html
    # The warning must appear before any score, not buried at the end.
    assert html.index("unknown reliability") < html.index("Headline comparison")


def test_a_too_small_calibration_still_shows_the_warning(db: Path, tmp_path: Path):
    # A kappa computed on a handful of items is not a measurement, and must
    # not turn the banner green.
    from harness.prompts import judge_prompt_hash

    run = _save(db, "anthropic", [1.0, 0.9])
    save_calibration(
        AgreementReport(
            n=2, raw_agreement=0.5, kappa=0.0, kappa_ci=(0.0, 0.0),
            band="close to useless", confusion={}, per_category={},
            excluded={}, human_ceiling_kappa=None, human_ceiling_n=0,
        ),
        "ls-small", "m", judge_prompt_hash(), db_path=db,
    )

    html = _render(db, [run], tmp_path)

    assert "No calibration for this judge rubric" in html
    assert "only 2 labelled item" in html


def test_present_calibration_is_shown_with_its_ceiling(db: Path, tmp_path: Path):
    from harness.prompts import judge_prompt_hash

    run = _save(db, "anthropic", [1.0, 0.9])
    save_calibration(
        AgreementReport(
            n=60, raw_agreement=0.9, kappa=0.71, kappa_ci=(0.55, 0.84),
            band="usable with caveats", confusion={}, per_category={},
            excluded={}, human_ceiling_kappa=0.82, human_ceiling_n=15,
        ),
        "ls1", "claude-sonnet-5", judge_prompt_hash(), db_path=db,
    )

    html = _render(db, [run], tmp_path)

    assert "0.710" in html
    assert "usable with caveats" in html
    assert "0.820" in html  # human ceiling
    assert "never been measured" not in html


# --- charts ------------------------------------------------------------------


def test_scatter_has_one_marker_per_run(db: Path, tmp_path: Path):
    a = _save(db, "anthropic", [1.0, 0.9], cost=0.02)
    b = _save(db, "openai", [1.0, 0.8], cost=0.01)

    html = _render(db, [a, b], tmp_path)

    assert html.count('data-point=') == 2


def test_charts_are_inline_svg_not_images(db: Path, tmp_path: Path):
    a = _save(db, "anthropic", [1.0, 0.9])
    b = _save(db, "openai", [1.0, 0.8])

    html = _render(db, [a, b], tmp_path)

    assert html.count("<svg") >= 3      # scatter, heatmap, comparison bars
    assert "&lt;svg" not in html        # not escaped into visible text
    assert "<img" not in html


def test_heatmap_marks_missing_cells_rather_than_guessing():
    svg = heatmap(["a"], ["clean", "hard"], [[0.9, None]])

    assert "n/a" in svg


def test_bar_chart_hatches_intervals_that_span_zero():
    # Colour alone must not carry the "indistinguishable" reading.
    svg = barh_with_intervals(["a vs b"], [0.0], [-0.02], [0.02])

    assert "url(#hatch)" in svg


def test_charts_escape_labels():
    svg = scatter_with_error_bars([ScatterPoint("<script>x</script>", 0.1, 0.9)])

    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


# --- escaping ----------------------------------------------------------------


def test_model_output_with_angle_brackets_is_escaped(db: Path, tmp_path: Path):
    # Model output is untrusted text and is rendered verbatim in the failure
    # gallery, so it is the escaping path that actually matters.
    tasks = [_task(0)]
    summary = RunSummary(
        results=[_result(0, 0.5, text="<script>alert('xss')</script> & <b>x</b>")],
        adapter_name="anthropic", model_id="m",
        prompt_name="p", prompt_hash="h", total_cost_usd=0.01,
        total_tokens_in=1, total_tokens_out=1, wall_clock_seconds=1.0,
        succeeded=1, failed=0, cached=0, latency_p50_ms=1.0, latency_p95_ms=1.0,
    )
    run = save_run(summary, tasks, db_path=db)

    html = _render(db, [run], tmp_path)

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_document_is_rendered_when_the_task_still_exists_on_disk(tmp_path: Path):
    # Documents are read from tasks/ rather than the store, so a task that
    # has since been deleted or renamed renders empty rather than wrong.
    from harness.html_report import _documents

    docs = _documents()
    assert docs, "expected the real task suite to load"
    assert "eob-001" in docs
    assert docs["eob-001"]["input"]


# --- failure gallery ---------------------------------------------------------


def test_failure_gallery_lists_every_imperfect_task(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0, 0.5, 0.75, 1.0])

    html = _render(db, [run], tmp_path)

    assert "eob-001" in html and "eob-002" in html
    assert html.count("<details") == 2  # only the two imperfect tasks


def test_failure_gallery_is_collapsible(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [0.5])

    html = _render(db, [run], tmp_path)

    assert "<details" in html and "<summary>" in html


def test_perfect_run_reports_no_failures(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0, 1.0])

    html = _render(db, [run], tmp_path)

    assert "No run scored below 1.0" in html


# --- methods -----------------------------------------------------------------


def test_methods_contains_the_judge_rubric_verbatim(db: Path, tmp_path: Path):
    from harness.prompts import load_prompt

    run = _save(db, "anthropic", [1.0])
    html = _render(db, [run], tmp_path)

    # A distinctive sentence from the rubric itself, not a reference to it.
    snippet = load_prompt("judge_v1").split("\n")[0][:40]
    assert snippet.strip() in html


def test_methods_states_bootstrap_parameters(db: Path, tmp_path: Path):
    run = _save(db, "anthropic", [1.0])

    html = _render(db, [run], tmp_path)

    assert "10000" in html or "10,000" in html
    assert "resampling <em>tasks</em>" in html


def test_report_size_stays_reasonable(db: Path, tmp_path: Path):
    a = _save(db, "anthropic", [0.9] * 100)
    b = _save(db, "openai", [0.9] * 100)
    c = _save(db, "together", [0.9] * 100)

    out = render_report([a, b, c], tmp_path / "big.html", db_path=db)

    assert out.stat().st_size < 2_000_000
