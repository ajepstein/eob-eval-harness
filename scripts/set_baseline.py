"""Promote a run to baseline in eval_gates.yaml.

Refuses runs that are not reproducible. A baseline is the thing every future
regression is measured against, so promoting a bad one silently corrupts
every comparison that follows — better to refuse and say why.

Three refusals:
  * the run had task failures — an incomplete run is not a baseline
  * it was recorded against a dirty working tree — the code that produced it
    cannot be checked out again
  * no judge calibration exists for the current rubric — judge-adjusted
    numbers of unknown reliability should not become the reference point
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from harness.calibration import calibration_is_usable
from harness.prompts import judge_prompt_hash
from harness.store import DEFAULT_DB_PATH, find_calibration, load_run


def refusals(record, db_path: str, require_calibration: bool = True) -> list[str]:
    reasons = []
    if record.meta.failures:
        reasons.append(
            f"{record.meta.failures} task(s) failed in this run — an incomplete "
            f"run cannot be a baseline"
        )
    if record.meta.git_dirty:
        reasons.append(
            "recorded against a dirty working tree — the code that produced it "
            "cannot be checked out again, so the comparison would not be reproducible"
        )
    if require_calibration:
        rubric = judge_prompt_hash()
        usable, why_not = calibration_is_usable(
            find_calibration(rubric, db_path=db_path)
        )
        if not usable:
            reasons.append(
                f"rubric {rubric} is not calibrated: {why_not} — a reference "
                f"point of unknown reliability corrupts every future comparison"
            )
    return reasons


def write_baseline(config_path: Path, run_id: str, who: str) -> None:
    """Set baseline_run_id, recording who promoted it and when."""
    text = config_path.read_text()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = f"# baseline promoted by {who} on {stamp}\n"

    text = re.sub(r"^# baseline promoted by .*\n", "", text, flags=re.MULTILINE)
    if re.search(r"^baseline_run_id\s*:.*$", text, flags=re.MULTILINE):
        text = re.sub(
            r"^baseline_run_id\s*:.*$",
            f"{note.rstrip()}\nbaseline_run_id: {run_id}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = f"{note}baseline_run_id: {run_id}\n" + text
    config_path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run to promote")
    parser.add_argument("--config", default="eval_gates.yaml")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Permit a baseline with no judge calibration (still refuses on "
             "failures or a dirty tree)",
    )
    args = parser.parse_args()

    console = Console()
    try:
        record = load_run(args.run_id, args.db)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    reasons = refusals(
        record, args.db, require_calibration=not args.allow_uncalibrated
    )
    if reasons:
        console.print(
            f"[red]Refusing to promote {record.meta.run_id[:8]} as baseline:[/red]"
        )
        for reason in reasons:
            console.print(f"  [red]-[/red] {reason}")
        return 1

    write_baseline(Path(args.config), record.meta.run_id, getpass.getuser())
    console.print(
        f"Promoted [bold]{record.meta.run_id[:8]}[/bold] "
        f"({record.meta.adapter}, mean F1 {record.meta.mean_f1:.4f}) to baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
