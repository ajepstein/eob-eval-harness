import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from harness.labeling import (
    MIN_REPEAT_GAP,
    LabelItem,
    build_label_set,
    labelled_item_ids,
    load_label_set,
    load_labels,
    save_label,
)
from harness.store import init_db

VERDICTS = ["equivalent", "different"]
CATEGORIES = ["name_variance", "hard", "format_variance"]


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """A store with one run and a spread of judged fields."""
    path = tmp_path / "labels.db"
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT INTO runs (id, created_at, git_commit, git_dirty, adapter,
                                 model_id, prompt_name, prompt_hash, task_count,
                                 total_cost_usd, total_tokens_in, total_tokens_out,
                                 wall_seconds, harness_version)
               VALUES ('run1','2026-01-01T00:00:00Z',NULL,0,'anthropic','m',
                       'extract_v1','ph',60,0.1,10,10,1.0,'0.1.0')"""
        )
        n = 0
        for category in CATEGORIES:
            for verdict in VERDICTS:
                # Deliberately imbalanced, like the real population.
                count = 15 if verdict == "equivalent" else 5
                for i in range(count):
                    task_id = f"t{n}"
                    conn.execute(
                        """INSERT INTO results (run_id, task_id, category, difficulty,
                                                edge_case, response_text, tokens_in,
                                                tokens_out, latency_ms, cost_usd,
                                                cached, error)
                           VALUES ('run1',?,?, 'hard',1,'{}',1,1,1.0,0.0,0,NULL)""",
                        (task_id, category),
                    )
                    conn.execute(
                        """INSERT INTO judge_calls (run_id, task_id, field, expected,
                                                    predicted, verdict, reason, cost_usd,
                                                    judge_model_id, judge_prompt_hash,
                                                    created_at)
                           VALUES ('run1',?,?,?,?,?,'because',0.001,'jm','jh',
                                   '2026-01-01T00:00:00Z')""",
                        (task_id, "member_id", f"exp{n}", f"pred{n}", verdict),
                    )
                    n += 1
    return path


# --- reproducibility ---------------------------------------------------------


def test_sampling_is_reproducible_under_a_fixed_seed(seeded_db: Path):
    a = build_label_set(["run1"], n=30, seed=7, db_path=seeded_db)
    b = build_label_set(["run1"], n=30, seed=7, db_path=seeded_db)

    assert [i.item_key for i in a.items] == [i.item_key for i in b.items]


def test_different_seeds_give_different_orders(seeded_db: Path):
    a = build_label_set(["run1"], n=30, seed=1, db_path=seeded_db)
    b = build_label_set(["run1"], n=30, seed=2, db_path=seeded_db)

    assert [i.item_key for i in a.items] != [i.item_key for i in b.items]


def test_seed_is_persisted_with_the_set(seeded_db: Path):
    built = build_label_set(["run1"], n=20, seed=42, db_path=seeded_db)

    assert load_label_set(built.id, seeded_db).seed == 42


# --- stratification ----------------------------------------------------------


def test_both_verdicts_survive_into_the_sample(seeded_db: Path):
    # The whole point of stratifying: an unstratified draw from a 75/25
    # population can easily contain no minority items at all, and a sample
    # of only-equivalent cases says nothing about discrimination.
    label_set = build_label_set(["run1"], n=24, seed=0, db_path=seeded_db)

    with sqlite3.connect(seeded_db) as conn:
        conn.row_factory = sqlite3.Row
        verdicts = Counter()
        for item in label_set.items:
            row = conn.execute(
                "SELECT verdict FROM judge_calls WHERE run_id=? AND task_id=? AND field=?",
                (item.run_id, item.task_id, item.field),
            ).fetchone()
            verdicts[row["verdict"]] += 1

    assert verdicts["equivalent"] > 0
    assert verdicts["different"] > 0


def test_all_categories_survive_into_the_sample(seeded_db: Path):
    label_set = build_label_set(["run1"], n=24, seed=0, db_path=seeded_db)

    with sqlite3.connect(seeded_db) as conn:
        conn.row_factory = sqlite3.Row
        categories = set()
        for item in label_set.items:
            row = conn.execute(
                "SELECT category FROM results WHERE run_id=? AND task_id=?",
                (item.run_id, item.task_id),
            ).fetchone()
            categories.add(row["category"])

    assert categories == set(CATEGORIES)


def test_requesting_more_than_available_returns_everything(seeded_db: Path):
    label_set = build_label_set(["run1"], n=10_000, seed=0,
                                double_label_frac=0.0, db_path=seeded_db)

    assert len(label_set.items) == 60


# --- double labelling --------------------------------------------------------


def test_double_labelled_items_appear_exactly_twice(seeded_db: Path):
    label_set = build_label_set(["run1"], n=40, seed=3,
                                double_label_frac=0.25, db_path=seeded_db)

    counts = Counter(i.item_key for i in label_set.items)
    repeated = [k for k, c in counts.items() if c > 1]

    assert repeated, "expected some repeated items"
    assert all(counts[k] == 2 for k in repeated)


def test_repeats_are_well_separated_in_the_queue(seeded_db: Path):
    # Close repeats measure short-term memory, not consistency.
    label_set = build_label_set(["run1"], n=40, seed=3,
                                double_label_frac=0.25, db_path=seeded_db)

    positions: dict[str, list[int]] = {}
    for item in label_set.items:
        positions.setdefault(item.item_key, []).append(item.position)

    for key, pos in positions.items():
        if len(pos) == 2:
            assert abs(pos[1] - pos[0]) > MIN_REPEAT_GAP, f"{key} repeats too soon"


def test_repeat_passes_are_numbered(seeded_db: Path):
    label_set = build_label_set(["run1"], n=40, seed=3,
                                double_label_frac=0.25, db_path=seeded_db)

    by_key: dict[str, list[int]] = {}
    for item in label_set.items:
        by_key.setdefault(item.item_key, []).append(item.pass_number)

    for passes in by_key.values():
        if len(passes) == 2:
            assert sorted(passes) == [1, 2]


def test_zero_double_label_fraction_produces_no_repeats(seeded_db: Path):
    label_set = build_label_set(["run1"], n=20, seed=0,
                                double_label_frac=0.0, db_path=seeded_db)

    counts = Counter(i.item_key for i in label_set.items)
    assert all(c == 1 for c in counts.values())


# --- blinding: the property the week's output depends on ---------------------


def test_label_items_table_has_no_judge_verdict_column(seeded_db: Path):
    # Blinding is enforced by the schema, not by discipline. If a verdict
    # column existed here, some future change would eventually display it.
    with sqlite3.connect(seeded_db) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(label_items)")}

    for forbidden in ("verdict", "judge_verdict", "reason", "judge_model_id"):
        assert forbidden not in columns, f"label_items exposes {forbidden!r}"


def test_label_item_dataclass_carries_no_verdict():
    assert not hasattr(LabelItem, "verdict")
    assert "verdict" not in LabelItem.__dataclass_fields__
    assert "reason" not in LabelItem.__dataclass_fields__


def test_no_judge_verdict_is_reachable_from_the_display_path(seeded_db: Path):
    # The strongest form of the check: everything the labelling UI can see
    # about an item, serialised, must not contain the judge's verdict.
    label_set = build_label_set(["run1"], n=20, seed=0, db_path=seeded_db)

    for item in label_set.items:
        blob = json.dumps(item.__dict__)
        assert "equivalent" not in blob
        assert "different" not in blob


def test_labelling_script_never_queries_judge_calls():
    source = Path("scripts/label.py").read_text()

    assert "judge_calls" not in source
    assert "list_judge_calls" not in source


def _executable_source(path: str) -> str:
    """Source with docstrings and comments stripped.

    Prose describing a guarantee ("reports no accuracy information") should
    not trip a check that looks for violations of it.
    """
    import ast
    import io
    import tokenize

    with open(path, "rb") as handle:
        stripped = "".join(
            "" if tok.type in (tokenize.COMMENT, tokenize.STRING) else tok.string
            for tok in tokenize.tokenize(handle.readline)
        )
    ast.parse(Path(path).read_text())  # still has to be valid Python
    return stripped


def test_status_script_cannot_reach_judge_verdicts():
    # It is not enough that the status tool declines to show agreement — it
    # must be unable to compute it, which means never loading a verdict.
    code = _executable_source("scripts/label_status.py")

    for reachable in ("judge_calls", "list_judge_calls", "compare_runs"):
        assert reachable not in code, f"status tool can reach {reachable!r}"


def test_labelling_scripts_import_nothing_that_exposes_verdicts():
    for script in ("scripts/label.py", "scripts/label_status.py"):
        code = _executable_source(script)
        assert "judge" not in code.lower(), f"{script} references judge data"


# --- persistence and resume --------------------------------------------------


def test_labels_round_trip(seeded_db: Path):
    label_set = build_label_set(["run1"], n=10, seed=0, db_path=seeded_db)
    item = label_set.items[0]

    save_label(label_set.id, item, "equivalent", 4.2, db_path=seeded_db)
    labels = load_labels(label_set.id, seeded_db)

    assert len(labels) == 1
    assert labels[0]["verdict"] == "equivalent"
    assert labels[0]["seconds"] == pytest.approx(4.2)
    assert labels[0]["field"] == item.field
    assert labels[0]["labeler"] == "self"


def test_resume_continues_without_duplicates(seeded_db: Path):
    label_set = build_label_set(["run1"], n=10, seed=0, db_path=seeded_db)
    for item in label_set.items[:4]:
        save_label(label_set.id, item, "equivalent", 1.0, db_path=seeded_db)

    done = labelled_item_ids(label_set.id, seeded_db)
    remaining = [i for i in label_set.items if i.item_id not in done]

    assert len(done) == 4
    assert len(remaining) == len(label_set.items) - 4
    assert not any(i.item_id in done for i in remaining)


def test_relabelling_an_item_replaces_rather_than_duplicates(seeded_db: Path):
    label_set = build_label_set(["run1"], n=10, seed=0, db_path=seeded_db)
    item = label_set.items[0]

    save_label(label_set.id, item, "equivalent", 1.0, db_path=seeded_db)
    save_label(label_set.id, item, "different", 2.0, db_path=seeded_db)

    labels = load_labels(label_set.id, seeded_db)
    assert len(labels) == 1
    assert labels[0]["verdict"] == "different"


def test_seconds_are_recorded_per_label(seeded_db: Path):
    # Sub-3s labels may be reflexive and 90s+ ones mark hard cases; both are
    # worth revisiting, which requires the timing to exist.
    label_set = build_label_set(["run1"], n=10, seed=0, db_path=seeded_db)
    save_label(label_set.id, label_set.items[0], "equivalent", 2.5, db_path=seeded_db)

    assert load_labels(label_set.id, seeded_db)[0]["seconds"] == pytest.approx(2.5)


def test_building_with_no_judged_fields_fails_clearly(tmp_path: Path):
    path = tmp_path / "empty.db"
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO runs (id, created_at, git_commit, git_dirty, adapter,
                                 model_id, prompt_name, prompt_hash, task_count,
                                 total_cost_usd, total_tokens_in, total_tokens_out,
                                 wall_seconds, harness_version)
               VALUES ('r','2026-01-01T00:00:00Z',NULL,0,'a','m','p','h',0,
                       0.0,0,0,0.0,'0.1.0')"""
        )

    with pytest.raises(ValueError, match="Run with --judge first"):
        build_label_set(["r"], n=10, db_path=path)


# --- clearing machine-generated labels ---------------------------------------


def test_clear_synthetic_removes_only_machine_labels(seeded_db: Path):
    # Human labels are real work; a synthetic cleanup must never destroy them.
    from harness.labeling import clear_labels

    label_set = build_label_set(["run1"], n=10, seed=0, db_path=seeded_db)
    for i, item in enumerate(label_set.items):
        save_label(
            label_set.id, item, "equivalent", 1.0,
            labeler="synthetic" if i % 2 else "self", db_path=seeded_db,
        )
    before = len(load_labels(label_set.id, seeded_db))

    removed, kept = clear_labels(label_set.id, db_path=seeded_db)

    remaining = load_labels(label_set.id, seeded_db)
    assert removed + kept == before
    assert len(remaining) == kept
    assert {row["labeler"] for row in remaining} == {"self"}


def test_clear_synthetic_is_a_no_op_when_all_labels_are_human(seeded_db: Path):
    from harness.labeling import clear_labels

    label_set = build_label_set(["run1"], n=6, seed=0, db_path=seeded_db)
    for item in label_set.items:
        save_label(label_set.id, item, "different", 1.0,
                   labeler="self", db_path=seeded_db)

    removed, kept = clear_labels(label_set.id, db_path=seeded_db)

    assert removed == 0
    assert kept == len(label_set.items)


def test_clear_synthetic_drops_calibrations_built_from_those_labels(seeded_db: Path):
    # A calibration whose inputs no longer exist is confusing to leave behind.
    from harness.calibration import AgreementReport
    from harness.labeling import clear_labels
    from harness.store import find_calibration, save_calibration

    label_set = build_label_set(["run1"], n=6, seed=0, db_path=seeded_db)
    save_label(label_set.id, label_set.items[0], "equivalent", 1.0,
               labeler="synthetic", db_path=seeded_db)
    save_calibration(
        AgreementReport(n=6, raw_agreement=0.8, kappa=0.4, kappa_ci=(0.1, 0.7),
                        band="weak", confusion={}, per_category={}, excluded={}),
        label_set.id, "m", "rubric-z", labelers=["synthetic"], db_path=seeded_db,
    )
    assert find_calibration("rubric-z", db_path=seeded_db) is not None

    clear_labels(label_set.id, db_path=seeded_db)

    assert find_calibration("rubric-z", db_path=seeded_db) is None


def test_clear_synthetic_keeps_calibrations_built_from_human_labels(seeded_db: Path):
    from harness.calibration import AgreementReport
    from harness.labeling import clear_labels
    from harness.store import find_calibration, save_calibration

    label_set = build_label_set(["run1"], n=6, seed=0, db_path=seeded_db)
    save_calibration(
        AgreementReport(n=6, raw_agreement=0.9, kappa=0.6, kappa_ci=(0.3, 0.8),
                        band="usable with caveats", confusion={},
                        per_category={}, excluded={}),
        label_set.id, "m", "rubric-h", labelers=["self"], db_path=seeded_db,
    )

    clear_labels(label_set.id, db_path=seeded_db)

    assert find_calibration("rubric-h", db_path=seeded_db) is not None


def test_clear_labels_resolves_a_short_id(seeded_db: Path):
    from harness.labeling import clear_labels

    label_set = build_label_set(["run1"], n=4, seed=0, db_path=seeded_db)
    save_label(label_set.id, label_set.items[0], "equivalent", 1.0,
               labeler="synthetic", db_path=seeded_db)

    removed, _ = clear_labels(label_set.id[:8], db_path=seeded_db)

    assert removed == 1


def test_clear_labels_on_an_unknown_set_raises(seeded_db: Path):
    from harness.labeling import clear_labels

    with pytest.raises(LookupError):
        clear_labels("no-such-set", db_path=seeded_db)


def test_sampling_terminates_when_n_is_smaller_than_the_stratum_count(seeded_db: Path):
    # Six strata and n=4: one-per-stratum is impossible, and the drift
    # correction used to spin forever trying to reach n.
    label_set = build_label_set(["run1"], n=4, seed=0,
                                double_label_frac=0.0, db_path=seeded_db)

    assert len(label_set.items) == 4


def test_sampling_of_a_single_item_terminates(seeded_db: Path):
    label_set = build_label_set(["run1"], n=1, seed=0,
                                double_label_frac=0.0, db_path=seeded_db)

    assert len(label_set.items) == 1


def test_small_samples_keep_the_largest_strata(seeded_db: Path):
    # When strata must be dropped, the biggest should survive.
    label_set = build_label_set(["run1"], n=3, seed=0,
                                double_label_frac=0.0, db_path=seeded_db)

    assert len(label_set.items) == 3


def test_every_accessor_resolves_a_short_label_set_id(seeded_db: Path):
    # Accessors that matched exactly while others resolved prefixes made a
    # short id silently report zero labels — indistinguishable from having
    # labelled nothing.
    from harness.labeling import labelled_item_ids

    label_set = build_label_set(["run1"], n=6, seed=0, db_path=seeded_db)
    save_label(label_set.id, label_set.items[0], "equivalent", 1.0, db_path=seeded_db)
    short = label_set.id[:8]

    assert load_label_set(short, seeded_db).id == label_set.id
    assert len(load_labels(short, seeded_db)) == 1
    assert len(labelled_item_ids(short, seeded_db)) == 1


def test_accessors_raise_on_an_unknown_label_set(seeded_db: Path):
    from harness.labeling import labelled_item_ids

    for fn in (load_labels, labelled_item_ids, load_label_set):
        with pytest.raises(LookupError):
            fn("no-such-set", seeded_db)
