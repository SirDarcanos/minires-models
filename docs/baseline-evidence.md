# Reproducible baseline evidence

Issue-level evidence is assembled by `minires.evaluation.evidence`. The command runs the physical, pinned legacy-reference, and clean fixed-configuration baselines twice through `evaluate_records`. Split allocation depends only on the shared data and split controls, never on model configuration. The package keeps one manifest file per baseline because input contracts can classify rows differently; when eligible rows agree, the allocation is identical across baseline types. Both repetitions reuse that baseline's exact create-only manifest.

The runner does not tune candidates, choose replacement weights, decide release thresholds, upload artifacts, or publish its summary draft. Missing density, scope evidence, source/family evidence, dependencies, weights, or model provenance is preserved as a concrete blocker rather than replaced with an assumption.

## Prepare the private inputs

Use the normalized local export and only evidence you can attest:

- `resin_density_g_per_ml` must describe the slicing labels; a typical material value is not evidence.
- `scope_confirmed=True` requires confirmation that records are pre-supported miniatures under the intended slicing conditions.
- `anonymous_source_group` and `miniature_family` are private evaluation evidence, not prediction features.
- The legacy provenance defaults to unknown. Do not claim source-held-out provenance without separate evidence.

The entire local `data/` directory is ignored and must remain uncommitted. Released legacy weights belong in an ignored private cache. Do not modify the retained export, local comparison datasets, or released weights.

## Run the package

Use Python 3.11–3.13 with all pinned requirements for scored legacy and learned runs:

```bash
python3.13 -m venv .venv-evidence
.venv-evidence/bin/pip install \
  -e . \
  -r requirements/evaluation.txt \
  -r requirements/legacy.txt \
  -r requirements/learned.txt
.venv-evidence/bin/python -m minires.evaluation.evidence \
  --records private/evaluation-records.json \
  --reconcile data/3d_print_miniatures_base.csv \
  --reconcile data/3d_print_miniatures_data.csv \
  --legacy-artifacts private/legacy-artifacts \
  --split-manifest private/baseline-evidence/split.json \
  --output-root private/baseline-evidence/run-001 \
  --verification-record private/baseline-evidence/verification.json \
  --volume-unit mm3 \
  --density-g-per-ml ATTESTED_DENSITY \
  --scope-confirmed
```

Replace `ATTESTED_DENSITY` with the evidenced value. Omit the density and scope flags when they are unknown; the resulting evidence package will remain blocked honestly. The optional verification record is a private JSON array of objects with string `command` and `outcome` fields, created only after those synthetic checks have completed. Output paths are create-only, so every attempt needs a new output root. The supplied split filename becomes one manifest per baseline, such as `split-physical.json`; it should be outside the output root when manifests must be reused by a later package.

For aggregate inspection, use the output-free [`notebooks/baseline_analysis.ipynb`](../notebooks/baseline_analysis.ipynb). Execute a copy under `private/` and never commit notebook output. The notebook calls `evaluate_records` directly and displays only `to_dict(public=True)` data.

## Evidence contents

Each baseline repetition writes the standard private Parquet files, full report, fitted-fold artifacts when available, and checksummed manifest. `evidence.json` adds:

- row accounting, status, blockers, and provenance classification for each baseline;
- interpreter, platform, and installed dependency versions;
- wall time, process CPU time, and the process peak-RSS high-water mark;
- exact split, input/config/code identity, blocker, and row-accounting comparisons;
- exact physical and legacy prediction comparisons;
- learned prediction comparison at the declared absolute and relative tolerances, or an explicit `not_observed_no_predictions` result when scoring is blocked;
- before/after SHA-256 and byte sizes for input datasets and weight files;
- checksums for every evidence artifact written before the index;
- completed synthetic verification commands and outcomes supplied to the runner; and
- an explicit no-tuning, no-threshold, no-weight-selection conclusion.

`public-summary-draft.json` is separately generated from the public serializer. `public-summary-review.json` records automated key and path-marker screening for identifying metadata and source mappings, local paths and fingerprints, row/notebook output, and raw errors. The screening does not certify arbitrary free-text values: manually review the concrete draft, bind the review to its SHA-256, and keep that review record private. Publication requires a separate approval; both screening and review records must state that no publication occurred.

## Interpret blockers and limitations

Report invalid inputs, unvalidated categories, and missing scope confirmation separately. A geometry check cannot establish validated scope. Source-balanced metrics over a small number of anonymous source groups provide limited unseen-source evidence and do not establish population-wide performance. No prediction interval or uncertainty claim is valid without separate held-out coverage evidence.

Conclude with the observed baseline evidence, unresolved limitations, and measured resource use needed for a later threshold and compute-budget decision. If required evidence is blocked, do not claim phase completion.

## Verification

Automated tests use synthetic fixtures only:

```bash
.venv/bin/mypy src/minires --ignore-missing-imports
.venv/bin/python -m unittest discover -s tests
```

Before committing, verify originals and downloads remain untouched:

```bash
git diff -- data/
git status --short
git diff --cached --name-only
```
