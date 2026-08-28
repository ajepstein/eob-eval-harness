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

## Normalization rules

These rules govern how extracted values are represented, and the Day 4
scorer will depend on them holding exactly:

- A field absent from the source document is `null`, never an empty string
  and never a missing key. Every key is always present in output.
- Currency is a bare number: `1250.00`, not `"$1,250.00"`.
- Dates are always `YYYY-MM-DD` regardless of source format.
- Strings compare case-insensitively with collapsed internal whitespace.
- `cpt_codes` preserve document order.
