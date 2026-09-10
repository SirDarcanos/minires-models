# Pinned legacy reference

The released MiniRes artifacts may be evaluated through the same
`evaluate_records` seam as the physical baseline. Their results are **legacy
reference only** unless separate evidence proves that the artifact was not
trained on the evaluated anonymous source group. The bundled datasets and the
historical notebooks are not clean holdouts.

This workflow does not modify the notebooks, source tables, or released
weights. It does not fit a model, search hyperparameters, publish replacement
weights, infer uncertainty, or reuse a historical export threshold as a release
gate.

## Immutable artifact set

The compatibility adapter accepts only revision
`ef3fe89d643739fa79da2930455eb02e26cc9e2e` of the public model repository.
Each download URL includes that full revision. Files are accepted only after
size and SHA-256 verification:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `minires.keras` | 9,992,026 | `369cbb70097ab12ca08cad89c4ad356e115f85feacebe1ebddc5222d92b4fe85` |
| `minires_xgb.json` | 16,117,079 | `928bb7f87c704de47cc54d387a78e5eb74584c9ad117b24855a3abbf018f80ed` |
| `minires_meta.json` | 159 | `12d17f5538081f8f47de7c752b3117e96a0da60edab37cfd073209b9c45d90f2` |

A missing file, checksum mismatch, malformed metadata, unavailable dependency,
or load failure produces named blockers. No fallback model or substitute
prediction is used. Artifact cache paths remain local and are not reported.

## Verified contracts and mismatches

### Released inference contract

The deleted release wrapper and pinned metadata agree on this exact order:

1. `kb` (legacy unit unknown)
2. `volume` (mm³)
3. `surface_area` (mm²)
4. `bbox_area` (bounding-box **volume**, mm³)
5. `euler_number` (dimensionless)
6. `scale` (legacy unit unknown)
7. `surface_volume_ratio` (mm⁻¹)

The wrapper selected these caller-supplied columns in this order and applied no
rounding, conversion, ratio recomputation, or fitted preprocessing. Legacy mode
therefore requires these exact columns and explicit `mm3` volume units. It does
not silently derive an input from the canonical normalization contract.

The neural network casts the matrix to float32 and then uses the learned Keras
`Normalization(axis=-1)` layer embedded in `minires.keras`. Those learned
normalization weights are part of the checksummed artifact and are not adapted
again. XGBoost receives the same float32 legacy matrix directly, with no neural
normalization. The ensemble is:

```text
0.2 * neural_network_prediction + 0.8 * xgboost_prediction
```

All outputs are interpreted as grams of sliced resin mass.

### Training notebook contract

The training notebook does more preprocessing than the released inference
wrapper: it casts `kb` to integer, rounds `volume` upward to 0.1 mm³, computes
`surface_volume_ratio` from that volume, and rounds selected measurements and
ratios to one decimal. The retained engineered CSV does not reproduce every one
of those operations exactly, while the release wrapper passes its values
through. This is a recorded training/inference mismatch; the compatibility
adapter follows released inference behavior rather than assuming notebook
parity.

The training notebook also removes a fitted whole-dataset upper volume tail
before a row-level split. The separate test notebook uses a different upper
quantile. Neither behavior is applied here. They prevent a clean holdout claim
and should not be copied into a future baseline.

### Diagnostics corrections

Historical notebook functions label mean squared error as “RMSE.” The shared
evaluator reports the square root as `rmse_g`. It reports tolerance performance
both as a fraction in `[0, 1]` and a percentage in `[0, 100]`. Historical save
helpers compare NN percentage to `90`, but XGBoost and ensemble percentages to
`0.9`; those thresholds have inconsistent units and are not used here.

The shared evaluator reports corrected diagnostics separately for the neural
network, XGBoost, and ensemble, including MAE, RMSE, signed error,
underestimation, tolerance fraction/percentage, and fixed diagnostic bins.

## Fixed configuration record

The pinned Keras artifact reports Keras 3.12.0 and contains:

- embedded seven-feature normalization;
- dense 448 + SELU;
- dense 601 + Mish + dropout 0.3;
- dense 544 + Mish + dropout 0.3;
- dense 416 + SELU + dropout 0.3;
- one linear output.

The notebook records seed 34, batch size 256, MSE loss, MAE monitoring, up to 100
final-fit epochs, and early stopping. The pinned XGBoost JSON reports XGBoost
3.1.2, squared-error regression, seven named features, and best iteration 896.
The notebook configuration is 900 estimators, depth 9, learning rate 0.01,
subsample 0.7, column sample 0.9, histogram trees, MAE evaluation, seed 34, and
50-round early stopping. These values are a reproducibility record for a later
clean refit, not an invitation to tune against legacy evaluation data.

## Run a private parity diagnostic

Use Python 3.11–3.13 for the pinned optional dependencies:

```bash
python3.13 -m venv .venv-legacy
.venv-legacy/bin/pip install -r requirements-legacy.txt
.venv-legacy/bin/python -m minires_evaluation \
  --records data/3d_print_miniatures_data.csv \
  --volume-unit mm3 \
  --scope-confirmed \
  --legacy-artifacts private/legacy-artifacts \
  --download-legacy-artifacts \
  --legacy-provenance overlap \
  --public \
  --output private/legacy-reference-summary.json
```

`overlap` is the appropriate conservative label for a table used during model
development. Use the Python API's
`LegacyProvenance.source_held_out(excluded_source_groups, evidence)` only when
private evidence establishes exclusion of those evaluated anonymous source
groups from fitting and selection. Evaluation verifies that every input source
group is covered. The private report records an evidence fingerprint and source
count; the public report does not disclose them. Without matching evidence,
evaluation is blocked and labeled unverified rather than held out.

For a representative private inference-parity check, compare predictions from
`load_legacy_reference` with the historical wrapper on the same rows and exact
ordered legacy columns. Keep row-level predictions and any source mapping under
`private/`; publish only the allowlisted summary. If artifacts or dependencies
are unavailable, retain the blocker output instead of inventing predictions.
