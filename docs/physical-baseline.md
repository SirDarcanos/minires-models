# Physical baseline evaluation

The physical baseline estimates **sliced resin mass** from mesh volume and an
explicit resin density. It is a local, reproducible reference for MiniRes
experiments, not a pricing calculation or a measurement of shop consumption.

It accepts volume in cubic millimetres (`mm3`), cubic centimetres (`cm3`), or
millilitres (`ml`), and normalizes it to `mm3`. The calculation is:

`volume_mm3 / 1000 × resin_density_g_per_ml`

The command requires an explicit density and explicit confirmation that records
are within the intended pre-supported-miniature scope. It does not infer either
condition. Missing or unsupported configuration produces a bounded `blocked`
or `needs_review` result rather than a default estimate.

## Run locally

Create a local JSON array. `volume` is in `mm3`; `sliced_resin_mass_g` is
the reference sliced resin mass in grams. `weight` remains a compatibility
alias for existing data, but new inputs should use the unit-bearing canonical
name. Canonical unit-bearing fields take precedence over legacy aliases.
See the [normalization reference](normalization.md) for local file adapters,
private Parquet, reconciliation, and the full feature contract.

```json
[
  {"volume": 1000.0, "sliced_resin_mass_g": 1.1}
]
```

Install the project, then run the public-safe summary:

```bash
python3 -m pip install -e . -r requirements/evaluation.txt
python3 -m minires \
  --records local-records.json \
  --density-g-per-ml 1.1 \
  --volume-unit mm3 \
  --scope-confirmed \
  --public \
  --output baseline-summary.json
```

Without `--public` or `--output`, the CLI creates a new run directory under
`private/`, containing typed Parquet, predictions, a row audit, reconciliation,
and fingerprints. Install `requirements/evaluation.txt` for Parquet support.
Use `--private-dir private/my-run` to choose a new directory. Existing run
directories and output files are never overwritten. A private JSON-only
`--output` must also be beneath a `private/` directory.

The public form contains aggregate metrics, data-quality reason counts, and
allowlisted run configuration only. It omits fingerprints, linkage tokens,
per-record data, and reconciliation. Review even this summary before publication;
the workflow performs no uploads.

## Frozen source holdouts

Add `--split-manifest private/splits.json` to the command above, or pass
`split_manifest="private/splits.json"` to `evaluate_records`. This creates a
private, create-only manifest **before scoring**. Reuse the same path for repeated
runs; changed input bytes, configuration, allocation version, or assignments
raise `split_manifest_mismatch`. Choose a new path for a different experiment.
Keep the split path outside a new `--private-dir` run directory, since the latter
must not already exist.

Supply `anonymous_source_group` and `miniature_family` in the private input.
Source identity can also supply the source group through the legacy adapter.
When both identity and aliases are present, their mapping must be one-to-one.
These fields become private hashed metadata, never prediction features.
Family labels are dataset-global: use the same label for related parts/variants;
record names and equal geometry measurements do not establish a family.

Optional `duplicate_group` and `geometry_fingerprint` fields assert externally
verified duplicate/mesh evidence. Only supply these when supported by an audit,
not by a hash of the tabular features. Shared family, duplicate, geometry, record
ID, or location evidence joins rows transitively into indivisible components.
ID/location matches are conservative co-isolation evidence, not proof of equal
geometry. Repeated rows remain counted; the workflow does not discard or average
duplicate observations. Unscorable rows also participate as evidence bridges.
Unavailable duplicate evidence and unreported variants remain limitations.

Every source with at least one included row is eligible, with no minimum sample
threshold or tail filter. Each eligible source is held out exactly once. A fold
requires at least two components outside its held-out source: one reserved for
inner validation, and at least one for training. At least two sources are needed.
The inner component is selected by a seeded hash ordering, independent of target
values and errors. This partition is common to future baselines **within that
outer fold**, not shared across outer folds. A globally shared validation set
would conflict with rotating source holdouts.

Unknown source/family membership, contradictory source evidence, or any
infeasible fold blocks the entire rotation with explicit reasons and zero
predictions. No source is silently dropped and no isolation rule is relaxed.
The private manifest records included and unscorable row indices, components,
folds, input fingerprint, transformation/allocation versions, and configuration.
Assertions check row, component/family/duplicate, and holdout-source isolation.

The stateless physical formula scores each held-out observation once; it uses
neither the training nor validation labels. No fitting or selection occurs.
Top-level `metrics` pool these observations with equal sample weight.
`grouped_evaluation.source_balanced` gives each source weight `1 / source_count`
and each observation within that source weight `1 / source_sample_count`.
Balanced RMSE is the square root of the equally weighted source mean squared
errors, not the mean of source RMSEs. Balanced mean underestimation conditions
these same weights on negative errors. Zero underestimations give a null
conditional mean. Counts and denominators accompany both summaries; balanced
error and volume bins are weighted fractions, while pooled bins are counts.

Detailed per-source metrics and train/validation/test sizes appear only in the
private report. Public output includes aggregate counts, size/error diagnostics,
and limitations, but omits manifests, fingerprints, paths, and grouping data.
A blocked split leaves normalization accounting intact: accepted rows can exist
with zero scored rows. Source balance describes the observed sources, not a
claim of population representativeness or an uncertainty interval.

## Output semantics

- Without `--split-manifest`, `split_status` is `not_applicable`; the report is
  an ungrouped diagnostic. With a manifest it is `frozen_source_holdout`, or
  `blocked` when grouping or partition sizes prevent isolation.
- `signed_error_g` means predicted sliced resin mass minus reference sliced
  resin mass. A negative value is underestimation.
- `within_tolerance_fraction` includes errors exactly equal to the configured
  two-gram tolerance; `within_tolerance_percent` is the same value multiplied
  by 100.
- Error and volume diagnostic bins are declared in the evaluation module before
  processing data. Empty bins are retained with count zero.
- The physical baseline does not create uncertainty intervals, operational
  allowances, prices, or replacement model weights.
