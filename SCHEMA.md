# EOB Extraction Schema

Target schema for extracting structured data from health insurance
Explanation of Benefits (EOB) documents.

| Field                   | Type              | Nullable          |
|--------------------------|-------------------|-------------------|
| patient_name             | string            | no                |
| date_of_service          | string, ISO 8601  | no                |
| provider_npi              | string, 10 digits | yes               |
| payer_name                | string            | no                |
| member_id                  | string            | yes               |
| cpt_codes                  | list[string]      | no (may be empty) |
| billed_amount              | decimal, 2dp      | no                |
| patient_responsibility     | decimal, 2dp      | no                |

## Field definitions

Where a document contains more than one plausible value for a field, these
decide which one the answer key takes. They are conventions, not derivable
facts — a model cannot infer them from the document, so they belong here.

- **`provider_npi` is the rendering provider on the claim being
  adjudicated** — the clinician who performed the service being explained.
  Documents commonly also carry a facility, billing, or referring NPI; none
  of those is the answer. Where a statement reproduces a previously
  processed claim for reference, the current claim's rendering provider is
  the answer.

  The field is `null` when the document states no NPI for the rendering
  provider, even if other NPIs are present. A facility NPI is not a
  fallback.

- **`member_id` is the subscriber identifier, without a dependent code.**
  A trailing `-A`, `-01` or similar identifies which person on the contract
  the claim is for; it is not part of the member id. `EG-441002-A` extracts
  as `EG-441002`. Like the NPI rule above, this is a convention the document
  cannot settle on its own.

  Whitespace inside the identifier is presentational grouping, the way a
  card number is printed in fours: `PM 4471 2039` and `PM44712039` are the
  same id, and `norm_member_id` removes internal whitespace so they compare
  equal. Hyphens are kept, precisely so the dependent-code distinction above
  survives normalization.

- **`cpt_codes` holds procedure codes only.** A CPT modifier — the
  two-character suffix in `99214-25`, `71046-TC`, `93000-26` — is a separate
  data element qualifying a code, not part of the code. Documents in this
  suite render the two joined by a hyphen; that is a presentation choice, not
  a single value. Extract `99214`, never `99214-25`.

  This follows the claim standards the documents are modelled on, which carry
  code and modifiers in distinct positions (CMS-1500 box 24D; X12 837P
  SV101-2 against SV101-3 onward). It also keeps the field aggregatable: a
  consumer counting `99214` encounters should not need to know every modifier
  that can attach to one.

  The cost is real and worth stating plainly. `-26` and `-TC` are
  payment-determining — `71046-TC` reimburses differently from `71046` — so a
  consumer needing that distinction cannot recover it from this schema. The
  remedy is a separate `cpt_modifiers` field, not a wider reading of this
  one. That field is not in scope here and its absence is a known limit.

## Normalization rules

These rules govern how extracted values are represented, and the Day 4
scorer will depend on them holding exactly:

- A field absent from the source document is `null`, never an empty string
  and never a missing key. Every key is always present in output.
- Currency is a bare number: `1250.00`, not `"$1,250.00"`.
- Dates are always `YYYY-MM-DD` regardless of source format.
- Strings compare case-insensitively with collapsed internal whitespace.
- `cpt_codes` preserve document order.
