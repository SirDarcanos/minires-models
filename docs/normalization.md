# Local normalization reference

Use Python 3.11 or later. The CLI and notebook callers use `evaluate_records`;
no database connection or notebook execution is needed.

## Run a private audit

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-evaluation.txt
.venv/bin/python -m minires_evaluation \
  --records local-export.jsonl \
  --reconcile data/3d_print_miniatures_base.csv \
  --reconcile data/3d_print_miniatures_data.csv \
  --volume-unit mm3 \
  --private-dir private/audit-001
```

Substitute the retained export's local filename. The directory must be new.
This command supplies volume units only: density and pre-supported-scope
confirmation remain unknown. Supply `--density-g-per-ml` and
`--scope-confirmed` only when you can attest to the labeling conditions and
scope. Historical notebook assumptions are not per-record provenance.

The same call from a notebook:

```python
from minires_evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records

result = evaluate_records(
    "local-export.jsonl",
    EvaluationConfig(None, "mm3", None, seed=0),
    PhysicalBaseline(),
    reconcile_with=["data/3d_print_miniatures_base.csv"],
    output_dir="private/audit-002",
)
summary = result.to_dict(public=True)
```

`output_dir` and `reconcile_with` are optional. The return value contains the
primary dataset's metrics and `canonical_rows`, including rejected rows.
Each reconciliation contains all comparison rows and their outcomes, but does
not pool them into primary metrics. `normalized_records` and `predictions`
contain only included primary rows, in input order; the audit supplies original
zero-based row indices.

## Input contract: `minires-normalization-v1`

Supported inputs are a sequence of mappings, a JSON array, JSONL, or CSV.
JSONL supports the retained export's numeric wrappers: `$numberDouble`,
`$numberInt`, and `$numberLong`. CSV numeric strings follow the same conversion.
`$numberDecimal` is not a supported numeric representation. Measurements use
IEEE-754 binary64, matching the retained double export; there is no float32
conversion, integer truncation of measurements, or decimal rounding. Unit
conversion can introduce binary64 rounding (compare numerically, e.g. relative
tolerance `1e-12`). This is not an arbitrary-precision decimal contract.

| Prediction measurement | Legacy alias | Meaning / unit |
| --- | --- | --- |
| `volume_mm3` | `volume` | Mesh-enclosed volume, mm³ |
| `surface_area_mm2` | `surface_area` | Mesh surface area, mm² |
| `bounding_box_x_mm` | `bbox_x` | Bounding-box extent along X, mm |
| `bounding_box_y_mm` | `bbox_y` | Bounding-box extent along Y, mm |
| `bounding_box_z_mm` | `bbox_z` | Bounding-box extent along Z, mm |
| `bounding_box_volume_mm3` | `bbox_area` | Bounding-box **volume**, mm³; not surface area |
| `euler_number` | `euler_number` | Euler characteristic, dimensionless integer-valued binary64 |

Legacy bounding-box and surface aliases have fixed millimetre-based units.
`volume_unit` applies only to legacy `volume`; `volume_mm3` is self-describing.
The baseline uses only `volume_mm3`. Other measurements are retained for future
evaluation, not fitted or engineered here. A future training caller must select
the schema's `prediction_features` allowlist, never every Parquet column.

`sliced_resin_mass_g` (alias `weight`) is the target in grams, stored separately.
`base_mm` is optional private numeric metadata. Unusable values become null with
a warning. Legacy `kb`, `mass`, and `scale` are retained as optional numeric
metadata named `legacy_*_unit_unknown`: this workflow does not assert their
physical units. Existing engineered CSV columns are ignored; their prior
rounding, filtering, or density assumptions are not repeated.

### Row accounting and missing values

Every parsed record gets exactly one outcome:

- **included**: required measurements, target, density, and scope support the
  physical baseline. Valid large examples remain included.
- **excluded**: malformed/non-object record, invalid supplied measurement,
  non-finite measurement or label, or invalid label. Reasons distinguish
  non-finite values from other invalid values. Positive geometric measurements
  are required when supplied; Euler characteristic may be negative or zero but
  must be integral. The target may be zero, but not negative.
- **needs_review**: unavailable label, unknown/invalid density, unknown volume
  units, unsupported scope (`scope_confirmed=false`), missing scope confirmation,
  invalid tolerance, or a non-finite computed prediction.

Rows can have multiple reasons. Invalid required data takes outcome precedence;
`data_quality.reasons` counts only the first reason, so its total equals the
number of excluded plus needs-review rows. Full reason lists are in the private
audit. Missing optional geometry fields stay null; invalid supplied geometry
measurements are excluded rather than silently trusted. Optional metadata
warnings do not exclude otherwise usable rows. Malformed JSONL lines (including
blank lines) remain excluded rows. An unreadable file or invalid JSON-array
container produces a bounded error instead of a partial dataset.

No mesh is opened: `geometry_valid` and `support_presence` stay null. An explicit
scope attestation is separate from a geometry check. Per-record density and
scope override configuration, including explicit nulls; absence inherits the
caller value. CSV scope accepts case-insensitive `true`/`false`, not truthiness.
The audit records whether density/scope came from a row or configuration.

`slicing_conditions` accepts a mapping or a JSON-encoded mapping. Known numeric
`layer_height_mm`, `exposure_seconds`, and `bottom_exposure_seconds` are retained
when finite and positive. Other profile details are represented only by a
private fingerprint; no free-text profile, identity, or location is copied.
Missing conditions stay unknown. Missing slicing conditions limit interpretation
but do not block the volume-density calculation when density and scope are
explicitly attested. No dataset-wide statistical tail filtering or fitting is
performed. Future fitted transforms belong exclusively to training partitions.

## Reconciliation evidence

Each comparison is reconciled separately against the primary input using
hashed record ID, location, and source-plus-record-name evidence, in that order
of reporting preference. Unique agreeing keys permit a candidate record match;
repeated keys, competing matches, and contradictory available IDs are marked
ambiguous. Every row appears in a match, an unmatched index list, or an ambiguous
index list. Matching is independent of baseline eligibility.

Matches report exact canonical measurement, target, and metadata differences.
Equal measurement-tuple counts are diagnostic only. Equal counts, feature values,
record names, or locations do not prove identical geometry, datasets, or duplicate
isolation. Unmatched rows do not prove absence. `dataset_equivalence_proven` is
always false. No rounding is introduced to make datasets appear equivalent.

## Private artifacts and reproducibility

- `features.parquet`: typed, nullable measurements plus `dataset_index` and
  `row_index` join keys. Join keys are explicitly **not** prediction features.
- `evaluation_metadata.parquet`: target, outcomes/reasons, provenance, optional
  measurements, and private linkage tokens.
- `report.json`: primary baseline metrics/blockers, full row audit, and comparisons.
- `manifest.json`: artifact SHA-256 checksums, row counts, feature units,
  transformation version, input/configuration/code fingerprints, seed, Python,
  platform, and PyArrow version.

Files are created locally with mode `0600` in a new `0700` run directory beneath
`private/`. Runs never overwrite inputs, old reports, notebooks, or weights.
A failed write may leave a partial directory; use a new directory after diagnosis.
Git ignores `private/`, Parquet, the retained export, and its sidecar metadata.

Identity, filenames, free text, and artist mappings are not persisted. Hashed
linkage tokens are still sensitive and remain private; they are not suitable
for publication. Source groups are hashed from available source identity.
Miniature families stay unresolved unless explicitly supplied: record names are
not automatically treated as family labels. No public summary contains per-row
features, targets, linkage, paths, or fingerprints.

Input fingerprints cover the original file bytes (or all in-memory rows,
including rejected rows). Configuration and source-code fingerprints support
reproduction alongside fixed package versions. Same bytes/configuration/code
produce deterministic row accounting and reports on the same platform. There
is no split manifest: split status is `not_applicable`, and no result is evidence
of unseen-source performance. No training, candidate tuning, weights, uncertainty
claims, or uploads are introduced.

## Verify synthetic behavior

```bash
.venv/bin/pip install mypy==1.19.1
.venv/bin/mypy minires_evaluation --ignore-missing-imports
.venv/bin/python -m unittest discover -s tests
```

Tests use synthetic data and temporary directories. Run real-data processing
separately, recording the local command and before/after original-file checksums
under `private/`; do not attach that audit to a public issue.
