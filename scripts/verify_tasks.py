"""Review task answer keys against their documents.

The failure mode this exists to prevent: the same process that writes a
synthetic document also writes its answer key, so a mistake in a hard case
becomes a permanent, silent scoring bias that no unit test can catch.

Two layers:

1. **Automated consistency checks** (``--check-only``, and shown inline
   during review). These catch the mechanical errors — an expected value
   that appears nowhere in the document, CPT codes listed out of order, an
   amount that doesn't match. They are a filter, not a substitute: they
   cannot tell you the intended answer is the *defensible* one on an
   ambiguous hard case.
2. **Human review.** You read the document and the key side by side and
   accept or reject. Only this sets ``verified: true``.

Usage:
    python scripts/verify_tasks.py                  # review unverified tasks
    python scripts/verify_tasks.py --all            # review every task
    python scripts/verify_tasks.py --check-only     # automated checks, no prompts
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness.normalize import norm_currency, norm_string

TASK_DIR = Path("tasks")


# --- automated consistency checks -------------------------------------------


def _flat(text: str) -> str:
    """Casefolded text with whitespace and separators stripped, for lookup."""
    return re.sub(r"[\s,\-]", "", text).casefold()


def _date_renderings(iso: str) -> list[str]:
    try:
        d = datetime.strptime(iso, "%Y-%m-%d").date()
    except ValueError:
        return []
    return [
        d.isoformat(),
        d.strftime("%m/%d/%Y"),
        d.strftime("%-m/%-d/%Y"),
        d.strftime("%m-%d-%Y"),
        d.strftime("%d/%m/%Y"),
        d.strftime("%B %-d, %Y"),
        d.strftime("%B %-d %Y"),
        d.strftime("%-d %B %Y"),
        d.strftime("%b %-d, %Y"),
        d.strftime("%Y/%m/%d"),
    ]


def check_task(task: dict) -> list[str]:
    """Return a list of problems found. Empty means the checks passed."""
    problems: list[str] = []
    doc = task.get("input", "") or ""
    flat_doc = _flat(doc)
    expected = task.get("expected", {}) or {}

    # Present string values must actually occur in the document.
    for field in ("patient_name", "payer_name", "member_id"):
        value = expected.get(field)
        if value and _flat(str(value)) not in flat_doc:
            problems.append(f"{field}={value!r} does not appear in the document")

    npi = expected.get("provider_npi")
    if npi:
        if _flat(str(npi)) not in flat_doc:
            problems.append(f"provider_npi={npi!r} does not appear in the document")
    else:
        # Hallucination bait: a 10-digit number in a document whose answer
        # key says "no NPI" is exactly the case a human should eyeball.
        stray = re.findall(r"\b\d{10}\b", doc)
        if stray:
            problems.append(
                f"expected provider_npi is null but the document contains "
                f"10-digit number(s): {', '.join(sorted(set(stray)))}"
            )

    date = expected.get("date_of_service")
    if date:
        if not any(_flat(r) in flat_doc for r in _date_renderings(str(date))):
            problems.append(
                f"date_of_service={date!r} appears in no recognized format"
            )

    codes = expected.get("cpt_codes")
    if isinstance(codes, list):
        positions = []
        for code in codes:
            idx = flat_doc.find(_flat(str(code)))
            if idx == -1:
                problems.append(f"cpt code {code!r} does not appear in the document")
            else:
                positions.append((idx, code))
        ordered = [c for _, c in sorted(positions)]
        listed = [c for _, c in positions]
        if ordered != listed:
            problems.append(
                f"cpt_codes are not in document order — document order is {ordered}"
            )

    for field in ("billed_amount", "patient_responsibility"):
        amount = expected.get(field)
        if amount is None:
            continue
        value = norm_currency(amount)
        renderings = [f"{value:.2f}", f"{value:,.2f}", str(value)]
        if not any(_flat(r) in flat_doc for r in renderings):
            problems.append(f"{field}={amount} does not appear in the document")

    return problems


# --- YAML round trip --------------------------------------------------------


def set_verified(path: Path, value: bool) -> None:
    """Write the verified flag without disturbing the document block scalar.

    yaml.dump would reflow `input: |` and mangle every document, so the
    flag is edited as text.
    """
    lines = path.read_text().splitlines(keepends=True)
    flag = f"verified: {'true' if value else 'false'}\n"

    for i, line in enumerate(lines):
        if re.match(r"^verified\s*:", line):
            lines[i] = flag
            path.write_text("".join(lines))
            return

    for i, line in enumerate(lines):
        if re.match(r"^edge_case\s*:", line):
            lines.insert(i + 1, flag)
            path.write_text("".join(lines))
            return

    path.write_text("".join(lines) + flag)


# --- rendering --------------------------------------------------------------


def render_task(path: Path, task: dict, problems: list[str], console: Console) -> None:
    console.clear()
    header = f"{task.get('id')}  [{task.get('category')} / {task.get('difficulty')}]"
    if task.get("edge_case"):
        header += "  edge_case"
    console.print(Panel(task.get("input", ""), title=header, subtitle=str(path)))

    table = Table(title="Expected", show_header=True)
    table.add_column("field", no_wrap=True)
    table.add_column("value")
    for field, value in (task.get("expected") or {}).items():
        shown = "[dim]null[/dim]" if value is None else repr(value)
        table.add_row(field, shown)
    console.print(table)

    if problems:
        console.print("[red]Automated checks found problems:[/red]")
        for problem in problems:
            console.print(f"  [red]✗[/red] {problem}")
    else:
        console.print("[green]✓ automated checks pass[/green] [dim](mechanical only — "
                      "read the document yourself)[/dim]")


# --- main -------------------------------------------------------------------


def load_raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(TASK_DIR), help="Task directory")
    parser.add_argument("--all", action="store_true", help="Review already-verified tasks too")
    parser.add_argument(
        "--check-only", action="store_true", help="Run automated checks and exit"
    )
    args = parser.parse_args()

    console = Console()
    paths = sorted(Path(args.tasks).glob("**/*.yaml"))
    if not paths:
        console.print(f"[red]No task YAML files under {args.tasks}[/red]")
        return 1

    if args.check_only:
        failing = 0
        for path in paths:
            task = load_raw(path)
            problems = check_task(task)
            if problems:
                failing += 1
                console.print(f"[red]{task.get('id')}[/red] ({path.name})")
                for problem in problems:
                    console.print(f"    {problem}")
        total = len(paths)
        if failing:
            console.print(f"\n[red]{failing} of {total} tasks have problems[/red]")
            return 1
        console.print(f"[green]All {total} tasks pass automated checks[/green]")
        return 0

    queue = []
    for path in paths:
        task = load_raw(path)
        if args.all or not task.get("verified"):
            queue.append((path, task))

    if not queue:
        console.print("[green]Every task is already verified.[/green]")
        return 0

    accepted = rejected = skipped = 0
    for index, (path, task) in enumerate(queue, start=1):
        problems = check_task(task)
        render_task(path, task, problems, console)
        console.print(
            f"\n[bold]{index}/{len(queue)}[/bold]  "
            "[green]y[/green]=verified  [red]n[/red]=reject  "
            "[yellow]s[/yellow]=skip  q=save and quit"
        )
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nStopping.")
            break

        if answer == "y":
            set_verified(path, True)
            accepted += 1
        elif answer == "n":
            set_verified(path, False)
            rejected += 1
            console.print(f"[red]Marked unverified — fix {path} before relying on it.[/red]")
        elif answer == "q":
            break
        else:
            skipped += 1

    console.print(
        f"\n{accepted} verified, {rejected} rejected, {skipped} skipped, "
        f"{len(queue) - accepted - rejected - skipped} not reached"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
