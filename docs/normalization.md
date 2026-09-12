# Local normalization reference

Use Python 3.11 or later. The CLI and notebook callers use `evaluate_records`;
no database connection or notebook execution is needed.

## Prepare the historical labeled records

Use the retained export as the primary record set because it contains the private
path evidence needed for grouping. Reconcile the existing labeled CSV without
assuming that equal measurements or record names prove row identity:

```bash
.venv/bin/python -m minires.preparation.prepare \
  --records local-export.jsonl \
  --reconcile data/3d_print_miniatures_base.csv \
  --private-dir private/prepared-v1 \
  --seed 0
```

The command writes no record data to stdout. It creates a new private directory
and fails rather than overwriting one. The prepared JSONL removes source names,
record names, and paths. Stable anonymous source groups come from private source
evidence. A miniature family is assigned only when the record name appears
with exact casing once as a directory in its private path; the evidenced pack
path and family directory form the key, and descendants remain in one family.
If one source/name pair resolves to distinct family directories, all of those
rows require review rather than being linked by name. Repeated byte-identical
path strings provide duplicate evidence. Missing or ambiguous evidence becomes
a `needs_review` outcome. Equal geometry values and names without path evidence
never create a group.

The prepared records attach the pinned slicing provenance: EBMiniManager revision
`1a841195813136ee3b380ab1d192727f385f7a55`, profile
`prediction/config-anycubic-mono.ini` with SHA-256
`06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e`,
density `1.1 g/ml`, layer height `0.05 mm`, disabled slicer-added supports,
`WeightG` extraction through `UVtoolsCmd`, and the maintainer's pre-supported
scope attestation. Targets and supplied features retain binary64 precision. If
the legacy `surface_volume_ratio` is absent, preparation derives it directly as
`surface_area / volume` without decimal rounding; it is not grouping evidence.

Artifacts include `prepared-records.jsonl`, private grouping and source-mapping
evidence, a reconciliation report, provenance, reusable frozen folds, a coverage
report, and SHA-256 checksums. Coverage says whether the grouped records can
support the frozen folds; it does not claim model quality or dataset equivalence.
The input and every reconciliation input are hashed before and after preparation
and must remain byte-for-byte unchanged.

## Assemble the current four-source dataset

Combine the retained historical export with the completed new-STL batch package:

```bash
.venv/bin/python -m minires.preparation.assemble \
  --historical-records private/historical-export.jsonl \
  --exclude-historical-source "$PRIVATE_HISTORICAL_SOURCE_TO_OMIT" \
  --new-batch-result private/issue-30/batch-result.json \
  --seed 23 \
  --private-dir private/current-dataset
```

The command reuses historical measurements as authoritative inputs and performs
only conservative schema validation; it never opens historical meshes or invokes
geometry, PrusaSlicer, or UVtools adapters. The configured historical source must
account for exactly the known 34 rows, and the remaining historical records must
resolve to three anonymous source groups. Accepted batch rows form the fourth
anonymous source group. Source evidence, paths, filenames, input checksums, and
miniature-family guesses are removed from model rows.

Canonical millimetre-based fields and the legacy inference aliases are emitted
together. Surface-to-volume ratios are recalculated from the unrounded underlying
surface area and volume, while authoritative sliced resin mass targets are not
changed. Invalid historical measurements and new batch rejections are retained in
`rejected.jsonl` but cannot enter a model partition. Exact duplicates stay together
when the historical export supplies duplicate evidence, an exact historical path
is repeated, or the new batch supplies checksum-backed duplicate evidence.

The complete private package contains `train.jsonl`, `validation.jsonl`,
`test.jsonl`, `rejected.jsonl`, `provenance.json`, a reconciled `manifest.json`,
and `checksums.json`. It is staged and replaced atomically. The manifest reconciles
historical input, the explicit 34-row exclusion, historical validation outcomes,
new inventory outcomes, eligible rows, and all three partitions. Prediction
features exclude source, identity, duplicate, partition, and target metadata.
The command prints only `completed` on success.

Results from this package support claims about performance on held-out STL rows
drawn from the retained sources. They are not evidence of unseen-source or
miniature-family-independent performance.

## Generate source-balanced partitions

Allocate harmonized labeled records independently within each retained anonymous
source group:

```bash
.venv/bin/python -m minires.preparation.partition \
  --records private/harmonized-records.jsonl \
  --exclude-source "$PRIVATE_SOURCE_TO_OMIT" \
  --seed 23 \
  --private-dir private/current-partitions
```

`--exclude-source` is compared with `anonymous_source_group` and is never written
to the output. Each eligible record requires a stable `_id` or
`record_identity`. An explicit `duplicate_group` keeps exact duplicates together;
no miniature-family field is read, required, or inferred. Membership depends on
source stratum, stable record identity, duplicate evidence, and seed—not targets,
geometry measurements, or input order.

The command writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and
`manifest.json`. Allocation targets 70%, 15%, and 15% of each source; indivisible
duplicate components and very small sources can cause rounding differences. The
manifest records row accounting, per-source allocation under private hashed
aliases, canonical prediction-feature units, the input fingerprint, and SHA-256
checksums for all three record artifacts. Its prediction-feature allowlist omits
source groups, split metadata, targets, identities, duplicate evidence, and join
keys.

The output directory represents the one current artifact set. An intentional
rerun stages a complete replacement beside it and installs the set only after all
artifacts and the manifest have been written. Input or write failure leaves the
previous complete set in place. The directory must be beneath `private/`; the
command writes no record data or source identity to stdout.

## Run a private audit

```bash
python3 -m venv .venv
.venv/bin/pip install -e . -r requirements/evaluation.txt
.venv/bin/python -m minires \
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
from minires import EvaluationConfig, PhysicalBaseline, evaluate_records

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
zero-based row indices. When grouped evaluation is blocked, normalized records
remain available but predictions are empty.

## Input contract: `minires-normalization-v3`

Version 3 accepts bounded preparation outcomes so ambiguous source/family
evidence remains unscored instead of being silently included. Version 2 added
explicit anonymous source aliases and duplicate/geometry evidence to private
metadata; prediction measurements and units remain unchanged.

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
rounding, filtering, or density assumptions are not repeated. The checksum-pinned
legacy model has a separate, non-fitting compatibility contract and does not
change this normalization version; see [Pinned legacy reference](legacy-reference.md).

### Row accounting and missing values

Every parsed record gets exactly one outcome:

- **included**: required measurements, target, density, and scope support the
  physical baseline. Valid large examples remain included.
- **excluded**: malformed/non-object record, invalid supplied measurement,
  non-finite measurement or label, or invalid label. Reasons distinguish
  non-finite values from other invalid values. Positive geometric measurements
  are required when supplied; Euler characteristic may be negative or zero but
  must be integral. The target may be zero, but not negative.
- **needs_review**: unavailable label, unresolved or ambiguous preparation
  evidence, unknown/invalid density, unknown volume units, unsupported scope
  (`scope_confirmed=false`), missing scope confirmation, invalid tolerance, or a
  non-finite computed prediction.

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
Git ignores `private/`, the entire local `data/` directory, Parquet, the retained export, and its sidecar metadata.

Identity, filenames, free text, and artist mappings are not persisted. Hashed
linkage tokens are still sensitive and remain private; they are not suitable
for publication. Source groups are hashed from explicit `anonymous_source_group`
or available source identity; supplied aliases and identity linkage stay private.
Miniature families stay unresolved unless explicitly supplied: record names are
not automatically treated as family labels. No public summary contains per-row
features, targets, linkage, paths, or fingerprints.

Input fingerprints cover the original file bytes (or all in-memory rows,
including rejected rows). Configuration and source-code fingerprints support
reproduction alongside fixed package versions. Same bytes/configuration/code
produce deterministic row accounting and reports on the same platform. There
is an optional frozen split manifest for [source-holdout evaluation](physical-baseline.md#frozen-source-holdouts).
Without it, split status is `not_applicable` and the result is not evidence of
unseen-source performance. No training, candidate tuning, weights, uncertainty
claims, or uploads are introduced.

## Verify synthetic behavior

```bash
.venv/bin/pip install mypy==1.19.1
.venv/bin/mypy src/minires --ignore-missing-imports
.venv/bin/python -m unittest discover -s tests
```

Tests use synthetic data and temporary directories. Run real-data processing
separately, recording the local command and before/after original-file checksums
under `private/`; do not attach that audit to a public issue.
