# Build notes

The decisions behind this harness, including the ones that turned out wrong
and the places the abstraction strained. Written as the work happened rather
than reconstructed afterwards.

---

## Why not an existing eval framework

The point of the project is the evaluation *method* — what counts as
correct, how far a judge can be trusted, which differences are real. A
framework supplies exactly the parts that needed to be argued for, and
hides them behind configuration. Writing the scoring rules meant having to
decide what a wrong value costs relative to a missing one, and that decision
turned out to matter more than any code in the repo.

## Bootstrap over tasks, not fields

Each task contributes eight field decisions, and they are not independent: a
model that misreads a document header gets the payer and the member id wrong
in the same breath. Resampling fields would treat ~600 correlated
observations as 600 independent ones and produce intervals far too narrow —
the specific error that makes a null result look like a finding. Resampling
tasks keeps the correlation intact.

The cost is wider intervals. That is the correct price.

## Why the judge only sees near-misses

A judge applied to every field would introduce disagreement on the ~96% of
fields that already match exactly, at real expense and latency, to fix a
problem those fields do not have. Applied only where the deterministic
scorer already flagged a mismatch, it can only move fields that were going
to score zero anyway. The blast radius is bounded by construction.

The rubric also resolves uncertainty toward `different` on purpose. An
uncertain judge then degrades to the deterministic scorer, which is a
well-understood floor. The opposite default would silently inflate the very
score being measured.

## A wrong value counts twice

Under the counting rules a wrong value is both a false positive and a false
negative; a missing value is only a false negative. So a model that invents
a plausible NPI scores worse than one that leaves the field null.

That asymmetry is deliberate and it is the most consequential decision in
the scoring layer. On a claim form, a confident wrong identifier is worse
than a blank one — a blank gets queried, a plausible wrong value gets paid.

## `require_significant` on the regression gate

A gate that fires on any point decrease fires on noise. It then gets muted,
and afterwards protects nothing. Regression gates can demand that the paired
interval exclude zero before failing, so the gate fires on regressions the
statistics can actually distinguish from chance.

This is the difference between a gate that survives contact with a team and
one that gets a `# TODO: re-enable` above it within a month.

---

## Where the adapter abstraction strained

Three providers, three token vocabularies:

| provider  | tokens in         | tokens out          | text at                        | stop reason at |
|-----------|-------------------|---------------------|--------------------------------|----------------|
| Anthropic | `usage.input_tokens`  | `usage.output_tokens`   | `content[].text`               | `stop_reason` |
| OpenAI    | `usage.input_tokens`  | `usage.output_tokens`   | `output_text`                  | `status` + `incomplete_details.reason` |
| Together  | `usage.prompt_tokens` | `usage.completion_tokens` | `choices[0].message.content` | `choices[0].finish_reason` |

The `Protocol` held without modification. Adding the third provider touched
three files, all inside `harness/adapters/` plus `config.py`, and added 120
lines. Nothing leaked upward.

Two things were genuinely awkward:

**Sampling parameters are not portable.** Claude Sonnet 5 and GPT-5.6 both
reject a non-default `temperature`; the open-weights route accepts one. The
`Adapter` signature keeps `temperature: float = 0.0` because that is the
harness's own default, and each adapter decides whether forwarding it is
valid. So the *interface* is uniform while the *semantics* are not, which is
a real wart — a caller passing `temperature=0.3` gets it honoured on one
provider and rejected on another.

**Stop reasons needed a shared vocabulary, and the first provider set it.**
Anthropic's `end_turn` / `max_tokens` became the canonical names simply
because that adapter was written first. `cache.py` refuses to store a
truncated response by checking for `max_tokens`, so every later provider had
to map into that namespace. It works, but the vocabulary is an accident of
ordering rather than a designed one.

**Together being OpenAI-compatible weakens the demonstration.** Both Together
and Groq expose OpenAI-shaped endpoints, so the swap could have been done by
pointing the existing OpenAI client at a different `base_url` — which would
have proven very little. Using the native SDK gave a genuinely third response
schema to normalise, but it is worth being honest that "we added a provider
that deliberately mimics another provider" is a softer test than the diffstat
suggests.

---

## Things that were wrong, and how they surfaced

**The judge rubric anchored the judge.** Its first line told the judge that
"an automated comparison found that the extracted value differs" — which is
the deterministic verdict, the very thing the judge must not see. It would
have biased every verdict toward `different` and the resulting kappa would
have measured suggestibility. Caught by a blinding test.

**Two labelled items counted as a calibration.** While testing the labelling
tool, two items were labelled and a calibration was saved from them: kappa
0.0, band "close to useless", n=2. That record then satisfied *every*
calibration check in the system — it unblocked the judge gate, would have
permitted a baseline promotion, and turned the report's red warning banner
green. A number that looks measured is worse than an admitted absence,
because it silently unblocks everything downstream. There is now a
`MIN_CALIBRATION_N` floor applied at all three consumers.

**Two tests passed while testing nothing.** One asserted that the labelling
UI contains no agreement information, and matched the word "accuracy" inside
the docstring *describing* that guarantee. Another asserted that document
text is HTML-escaped, using a task id absent from `tasks/` — so the
malicious text never reached the page and "no `<script>` present" was
trivially true. Both now check tokenised source or a path that is actually
rendered.

**The MDE was reported for the wrong design.** The standard minimum
detectable effect formula is unpaired, and every comparison here is paired.
The unpaired formula said 0.114 at n=78; the paired comparison actually
resolves 0.0128 — a factor of nine. Shipping only the former would have made
the suite look far blunter than it is and argued for an expansion that buys
almost nothing.

**The build plan's MDE direction is backwards.** It asks that MDE decrease
as baseline accuracy approaches 0.5. Binomial variance `p(1-p)` peaks at
0.5, so that is where an effect is hardest to resolve. The correct behaviour
is implemented and the test asserts it, with the deviation documented rather
than a knowingly wrong formula written to satisfy the spec.

**The provider-boundary grep broke on an English word.** The acceptance
check `grep -ril "anthropic\|openai\|together" harness/` started reporting
false positives once a provider was named "together", which appears in prose
like "fields fail together". Replaced with AST-based tests that check
*imports* — stricter, and immune to what the comments happen to say.

---

## What was measured, and what could not be

**Two frontier models are statistically indistinguishable on this task.**
Mean F1 0.938 and 0.936; paired difference −0.002 with a 95% interval of
[−0.014, 0.010]; McNemar exact p = 1.000. They are separated by cost alone.
(These numbers moved slightly when `member_id` normalization was defined —
the conclusion did not.)

**Hallucination never occurred.** Across every nullable field in 78 tasks,
neither model invented a value where the document had none — a rate of
0.0000. Since that is the failure mode the entire `missing_field` category
exists to provoke, it is a real finding rather than an absence of testing.

**The suite could not be made to discriminate.** Two separate attempts to
engineer tasks that produce genuine "different" verdicts both yielded zero
near-misses: both models correctly distinguished contract from certificate
numbers, facility from rendering NPIs, account totals from claim totals,
admission from service dates. The seven "different" verdicts that exist
arose incidentally, inside dense-ambiguity documents where a model juggling
several ambiguous fields at once slipped on one. Engineering documents
specifically to trick a model would be gaming the eval rather than measuring
it, so the attempt was abandoned and recorded.

**The judge was calibrated once, and the calibration said it does not
work.** 48 of 101 items were labelled; 42 of them scored a calibration at
kappa −0.043 — band "close to useless", below the `MIN_USABLE_KAPPA` floor
of 0.40. The downstream machinery — the calibration banner, the baseline
promotion, the regression gate — refused to operate on it, which is what it
is for.

The refusal is only half the story, and the attribution turned out to be
recoverable. All 34 of the judge's `cpt_codes` calls returned `equivalent`,
with reasoning defensible on its own terms — "the same base CPT code with a
modifier appended, denoting the same procedure". The human labels on that
same question split 11 `equivalent` to 5 `different`, and gave opposite
verdicts on structurally identical pairs: eob-051 and eob-062 both compare
bare codes against `-25` and `-TC`, and were labelled differently.

"Inconsistent" is too blunt, though, and the repeat labels say something
sharper. Four `cpt_codes` items were sampled twice for the intra-rater
ceiling, and every repeat agreed with itself: eob-051 `equivalent` twice,
eob-054 `equivalent` three times, eob-056 `equivalent` twice, eob-070
`different` twice. The labeller was stable when re-shown the same item and
unstable only across structurally identical *different* items — a rule
applied consistently within each item and reinvented between them. That is
the signature of a missing definition, not a careless labeller, and it is
better evidence for the diagnosis than the raw 11/5 split.

So the judge was perfectly consistent and the human was not, on a question
`SCHEMA.md` had never answered. Kappa −0.043 is substantially a measurement
of that missing definition rather than of judge reliability — which is the
more uncomfortable finding, because the labels were the trusted side of the
comparison.

`SCHEMA.md` now defines the field as codes-only. That makes the judge's rule
the wrong one on all 34 calls and the five `different` labels the right ones.
It does not rescue the label set: those items were labelled before the rule
existed. The calibration was discarded rather than superseded, and the 16
`cpt_codes` labels cleared — leaving 32 labels on questions the schema gap
never touched, and 69 items to label against a defined rule.

One caveat belongs with that next calibration before it is read. 36 of the
69 items are `cpt_codes` comparisons whose answer now follows mechanically
from a rule `judge_v2` states in its own rubric. The judge will get them
right, and kappa will improve for a reason that is not judgment. The signal
worth reading lives in the `patient_name` and `member_id` items.

---

## What I would do differently

**Store the document with the run.** The failure gallery reads task
documents from disk, so a task edited after a run renders the current
document beside historical scores. The same reasoning that made storing raw
response text obviously right applies here and was not followed through.

**Define `provider_npi` precisely in the schema.** The field does not say
*which* provider — billing, rendering, or referring. That ambiguity produced
the only extraction failure in 80 calls, and the judge's own reasoning
confirmed it is a schema question rather than something a judge can settle.
One line in `SCHEMA.md` would have prevented it.

**Design the near-miss population before building the judge.** Week 2A
shipped a judge with literally nothing to judge — the 40-task suite
contained zero near-misses. The judge was correct code with no domain to
operate on, and the suite had to be rebuilt around it afterwards. Sizing the
population first would have reordered a week of work.

**Pick a provider with a genuinely different API shape.** See above: an
OpenAI-compatible third provider is a weaker test of the abstraction than
the diffstat makes it look.
