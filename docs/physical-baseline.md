# Physical baseline evaluation

The physical baseline estimates **sliced resin mass** from mesh volume and an
explicit resin density. It is a local, reproducible reference for MiniRes
experiments, not a pricing calculation or a measurement of shop consumption.

It supports volume in cubic millimetres (`mm3`) only. The calculation is:

`volume_mm3 / 1000 × resin_density_g_per_ml`

The command requires an explicit density and explicit confirmation that records
are within the intended pre-supported-miniature scope. It does not infer either
condition. Missing or unsupported configuration produces a bounded `blocked`
or `needs_review` result rather than a default estimate.

## Run locally

Create a local JSON array. `volume` is in `mm3`; `weight` is the reference
sliced resin mass in grams. Extra fields are ignored by the evaluation module.

```json
[
  {"volume": 1000.0, "weight": 1.1}
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

Without `--public`, the result also contains normalized records, predictions,
and a private input fingerprint. Keep that output local. The public form is
allowlisted: it contains aggregate metrics, data-quality reason counts, and
non-identifying run configuration only.

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
