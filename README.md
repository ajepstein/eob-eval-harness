# EOB extraction evaluation harness

A provider-agnostic harness for evaluating LLM document extraction, built on
health-insurance Explanation of Benefits documents. It reports accuracy with
confidence intervals, prices quality against cost, validates its own LLM
judge against human labels, and fails CI when a change degrades quality by
more than noise.

The interesting output is not a leaderboard. It is that two frontier models
are **statistically indistinguishable** on this task and separated only by
price — a claim most benchmarks cannot support because they never compute
the interval.

---

## Headline result

78 synthetic EOB documents, both models scored on identical tasks.

| adapter | model | n | schema | mean F1 [95% CI] | $/task |
|---|---|---:|---:|---:|---:|
| anthropic | `claude-sonnet-5` | 78 | 1.00 | **0.929** [0.902, 0.955] | $0.000408 |
| openai | `gpt-5.6-terra` | 78 | 1.00 | **0.929** [0.901, 0.955] | $0.000260 |

**Paired difference: +0.0000, 95% CI [-0.0128, +0.0128]. McNemar exact
p = 1.000.**

> The difference between these two models is not distinguishable from zero
> at n=78. They differ only in cost, where `gpt-5.6-terra` is 1.6× cheaper
> for the same measured quality.

**Judge reliability: measured, and the judge failed.** Against 42 blind
human labels the judge scored **kappa −0.043** — no better than chance. It
and the labeller never once agreed that two values were *different*: all six
of the labeller's "different" calls were judged equivalent. The judge is
systematically too permissive on exactly the distinction that matters, which
is whether a member id carrying an extra component is the same identifier.

That is a result, not a gap. The harness refuses to report judge-adjusted
scores, the report leads with a red banner, and `set_baseline.py` will not
promote a baseline — because a judge measured below the 0.40 floor has been
tested and found wanting, not merely left unmeasured.

---

## The swap demonstration

The thesis is that the model is a swappable component. Adding a third
provider (Together, open-weights) touched **three files, 120 lines, all
inside the adapter layer**:

```
 harness/adapters/__init__.py |  6 +++
 harness/adapters/together.py | 96 +++++++++++++++++++++++++++++++++++++
 harness/config.py            | 18 +++++++
 3 files changed, 120 insertions(+)
```

Nothing outside `harness/adapters/` imports a provider SDK. That is enforced
by test rather than asserted:

```bash
pytest tests/test_provider_boundary.py -q     # 6 passed
```

The three providers report tokens under three different names — Anthropic
`input_tokens`/`output_tokens`, OpenAI the same on a differently-shaped
object, Together `prompt_tokens`/`completion_tokens` — and put the text and
stop reason in three different places. Absorbing that is the adapter's whole
job; the runner, scorers, store, and report cannot tell which one answered.

---

## What the harness found

**The two frontier models are indistinguishable, and cost is the only
separator.** Identical mean F1 to three decimals, with a paired interval
tight around zero. The cost/quality frontier marks `claude-sonnet-5` as
dominated: no better, and 1.6× the price.

**Neither model ever hallucinated.** Across every nullable field in 78
tasks, neither invented a value where the document had none — a rate of
**0.0000**. Twelve `missing_field` tasks exist specifically to provoke that
failure, each baiting it with a plausible nearby identifier (a tax id, a
group number, a licence number). Both models declined every time.

**Difficulty, not sample size, is the binding constraint.** The paired
design already resolves differences of **0.013**, nine times finer than the
unpaired power calculation suggests. Growing the suite would tighten that
slightly and change no conclusion; the models are genuinely equivalent here.

**The semantic-equivalence tail is real but small.** Raw F1 0.929 rises to
0.995 once a judge adjudicates near-misses — a 6.6-point gap consisting
almost entirely of defensible surface-form disagreements: `WHITFIELD,
MARGUERITE A.` versus `Marguerite A. Whitfield`, `99214` versus `99214-25`,
`WB 5520 1147` versus `WB55201147`. **That 0.995 is not currently
reportable**, because the judge producing it is uncalibrated.

**The suite resisted being made harder.** Two separate attempts to engineer
tasks yielding genuine "different" verdicts produced zero near-misses: both
models correctly separated contract from certificate numbers, facility from
rendering NPIs, account totals from claim totals, admission from service
dates. The attempt was abandoned rather than escalated into designing
documents specifically to trick a model, which would be gaming the eval.

---

## How judge reliability is meant to be established

The method is built and tested; it has not been run, because it depends on
human labels that do not yet exist.

- **Near-miss only.** The judge sees a field only where the deterministic
  scorer already found a mismatch and both values are non-null. It cannot
  disturb the ~96% of fields that already match.
- **Blind by construction.** The judge's verdict is read while *sampling*
  the label set and then discarded — it is never written to `label_items`,
  so the labelling UI cannot display it even by accident. Tests assert the
  column does not exist, the dataclass has no such field, no serialised item
  contains a verdict string, and neither labelling script can reach
  `judge_calls`.
- **Kappa, not raw agreement**, with a bootstrap interval, and read against
  an intra-rater ceiling measured from double-labelled items. A judge at
  0.72 against a human ceiling of 0.78 is a very different result from 0.72
  against 0.95.
- **Four bias tests**: position (re-asking with values swapped and neutrally
  labelled), length (point-biserial against predicted length),
  self-preference (own-family versus other-family), and drift in the human
  labels themselves.
- **The kappa paradox is detected**, not glossed. Above 85% single-class
  marginals kappa is unstable even at high agreement, and the report says
  so. The current label population is 93/7, so this will fire.

---

## Limitations

Written before anyone asks.

- **The judge does not work on this task**, at kappa −0.043 over 42 labels.
  The label set is also 86% one class, which makes kappa unstable, and only
  five double-label pairs survive, so the human ceiling is undefined and the
  result cannot be attributed between "bad judge" and "ambiguous task". The
  honest reading is that the rubric needs revising to `judge_v2` and
  re-calibrating, not that the number should be presented more kindly.
- **78 synthetic tasks, one document domain.** Fictional patients, payers
  and providers. Nothing here says how these models behave on real scanned
  EOBs, and OCR noise is entirely absent.
- **Single labeller**, so the ceiling would be intra-rater, not inter-rater.
  The `human_labels` schema carries a `labeler` column so a second person
  could be added without migration.
- **Temperature 0 only**, and run-to-run variance has not been quantified.
- **The third provider was never run live.** The Together adapter is written
  against the real SDK schema and unit-tested, but no live call has been
  made — no key. It appears in no result above.
- **Both models sit at ceiling on most categories**, so the suite
  discriminates mainly on the name-variance tail.
- **Costs are as of 2026-08-27** and are recorded with the source URL and
  date-checked comment in `harness/config.py`.
- **CI has not been observed running.** Both workflows are registered and
  the gate passes locally in 0.46s, but no pull request has exercised them.

---

## Reproduce it

Three commands, no API keys, no network — the committed cache carries a
response for every task.

```bash
git clone https://github.com/ajepstein/eob-eval-harness && cd eob-eval-harness
pip install -e .[dev]
python scripts/ci_eval.py            # replays fixtures/cache, evaluates the gates
```

That runs the full 78-task suite against both adapters and checks every
quality gate in under a second. To go further:

```bash
pytest -q                                        # 390 tests, offline
python scripts/run_eval.py --report --open       # self-contained HTML report
python scripts/run_eval.py --mde                 # what this suite can resolve
python scripts/run_eval.py --frontier            # cost against quality
```

Live runs need keys in `.env` (see `.env.example`) and cost roughly $0.25
for the full suite across two providers.

---

## How it is put together

```
harness/
  adapters/      provider SDKs live here and nowhere else
  tasks.py       validating loader for the YAML task suite
  runner.py      concurrent, retrying, failure-isolating; imports adapters.base only
  normalize.py   the single source of truth for what "equal" means
  extract.py     JSON recovery: direct -> fenced -> balanced braces
  scorers/       schema (structural) and fields (micro-averaged P/R/F1)
  judge.py       LLM judge, near-misses only
  store.py       SQLite: runs, results, scores, judge calls, labels, calibrations
  stats.py       bootstrap CIs, paired tests, McNemar, MDE, Holm
  calibration.py kappa, human ceiling, and the usability threshold
  gates.py       absolute and regression gates
  html_report.py one self-contained file, inlined SVG
```

`NOTES.md` carries the design decisions, the places the abstraction
strained, and the things that turned out wrong — including two tests that
passed while testing nothing, and a two-item calibration that briefly
satisfied every check in the system.
