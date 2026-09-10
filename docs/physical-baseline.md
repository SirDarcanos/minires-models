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

Run the public-safe summary:

```bash
python3 -m minires_evaluation \
  --records local-records.json \
  --density-g-per-ml 1.1 \
  --volume-unit mm3 \
  --scope-confirmed \
  --public \
  --output baseline-summary.json
```

Without `--public` or `--output`, the CLI creates a new run directory under
`private/`, containing typed Parquet, predictions, a row audit, reconciliation,
and fingerprints. Install `requirements-evaluation.txt` for Parquet support.
Use `--private-dir private/my-run` to choose a new directory. Existing run
directories and output files are never overwritten. A private JSON-only
`--output` must also be beneath a `private/` directory.

The public form contains aggregate metrics, data-quality reason counts, and
allowlisted run configuration only. It omits fingerprints, linkage tokens,
per-record data, and reconciliation. Review even this summary before publication;
the workflow performs no uploads.

## Output semantics

- `split_status` is `not_applicable`: this ticket does not yet establish
  unseen-source performance.
- `signed_error_g` means predicted sliced resin mass minus reference sliced
  resin mass. A negative value is underestimation.
- `within_tolerance_fraction` includes errors exactly equal to the configured
  two-gram tolerance; `within_tolerance_percent` is the same value multiplied
  by 100.
- Error and volume diagnostic bins are declared in the evaluation module before
  processing data. Empty bins are retained with count zero.
- The physical baseline does not create uncertainty intervals, operational
  allowances, prices, or replacement model weights.
