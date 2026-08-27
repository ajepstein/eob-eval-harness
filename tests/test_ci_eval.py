"""The CI runner: replay-only by default, loud on cache miss."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ci_eval import CacheMiss, ReplayOnlyAdapter, _markdown  # noqa: E402

from harness.gates import GateReport, GateResult  # noqa: E402


def test_replay_adapter_refuses_to_call_out():
    # The guarantee that CI cannot spend money: a cache miss raises rather
    # than quietly becoming a paid request.
    adapter = ReplayOnlyAdapter("anthropic", "claude-sonnet-5")

    with pytest.raises(CacheMiss, match="no committed response"):
        asyncio.run(adapter.complete("anything"))


def test_replay_adapter_records_what_was_attempted():
    adapter = ReplayOnlyAdapter("openai", "gpt-5.6-terra")
    with pytest.raises(CacheMiss):
        asyncio.run(adapter.complete("a prompt"))

    assert adapter.attempts == ["a prompt"]


def test_replay_adapter_satisfies_the_adapter_protocol():
    adapter = ReplayOnlyAdapter("anthropic", "m")

    assert adapter.name == "anthropic"
    assert adapter.model_id == "m"
    assert hasattr(adapter, "complete")


def test_markdown_table_renders_each_gate():
    report = GateReport(
        run_id="abcdef1234", baseline_run_id=None,
        results=[
            GateResult("mean_f1", "min", passed=True, observed=0.93, threshold=0.8),
            GateResult("cost_per_task", "max", passed=False, observed=0.9, threshold=0.02),
            GateResult("mean_f1", "max_regression_vs_baseline", passed=True,
                       threshold=0.03, skipped_reason="no baseline"),
        ],
    )

    md = _markdown([report])

    assert "| mean_f1 | min | 0.9300 | 0.8000 | pass |" in md
    assert "**FAIL**" in md
    assert "skipped" in md


# --- end to end, no keys -----------------------------------------------------


def _ci_eval(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    return subprocess.run(
        [sys.executable, "scripts/ci_eval.py", "--db", str(tmp_path / "ci.db"), *args],
        capture_output=True, text=True, timeout=180, env=env,
    )


def test_cached_mode_passes_with_no_api_keys(tmp_path: Path):
    done = _ci_eval([], tmp_path)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "all gates passed" in done.stdout


def test_cached_mode_reports_no_network(tmp_path: Path):
    done = _ci_eval([], tmp_path)

    assert "no network, no secrets" in done.stdout


def test_cache_miss_fails_loudly_rather_than_calling_out(tmp_path: Path):
    # Point at an empty cache: every task misses, and the run must fail with
    # an explanation rather than attempt a request.
    empty = tmp_path / "empty-cache"
    empty.mkdir()

    done = _ci_eval(["--cache", str(empty), "--adapter", "anthropic"], tmp_path)

    assert done.returncode == 1
    assert "no committed response" in done.stdout


def test_malformed_config_fails_loudly(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("gates:\n  - metric: nonsense\n    min: 1\n")

    done = _ci_eval(["--config", str(bad)], tmp_path)

    assert done.returncode == 1
    assert "unknown metric" in done.stdout


def test_markdown_output_is_written(tmp_path: Path):
    out = tmp_path / "gates.md"

    _ci_eval(["--markdown", str(out)], tmp_path)

    assert out.exists()
    assert "## Eval gates" in out.read_text()


# --- workflows ---------------------------------------------------------------


def test_both_workflows_are_valid_yaml():
    import yaml

    for name in ("eval.yml", "eval-nightly.yml"):
        doc = yaml.safe_load(Path(f".github/workflows/{name}").read_text())
        assert doc["jobs"], f"{name} defines no jobs"


def test_pr_workflow_uses_no_api_secrets():
    # The PR gate must run without secrets, or forks cannot run it and it
    # becomes a job that costs money on every push.
    text = Path(".github/workflows/eval.yml").read_text()

    assert "ANTHROPIC_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "--live" not in text


def test_nightly_workflow_is_live_and_guarded():
    text = Path(".github/workflows/eval-nightly.yml").read_text()

    assert "--live" in text
    assert "secrets.ANTHROPIC_API_KEY" in text
    assert "concurrency:" in text
    assert "cancel-in-progress: false" in text  # never cancel mid-spend
    assert "MAX_API_CALLS" in text


def test_nightly_opens_an_issue_rather_than_blocking():
    text = Path(".github/workflows/eval-nightly.yml").read_text()

    assert "issues.create" in text
