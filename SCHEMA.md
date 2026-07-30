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

## Normalization rules

These rules govern how extracted values are represented, and the Day 4
scorer will depend on them holding exactly:

- A field absent from the source document is `null`, never an empty string
  and never a missing key. Every key is always present in output.
- Currency is a bare number: `1250.00`, not `"$1,250.00"`.
- Dates are always `YYYY-MM-DD` regardless of source format.
- Strings compare case-insensitively with collapsed internal whitespace.
- `cpt_codes` preserve document order.
