# Clean fixed-configuration learned baselines

This workflow refits the documented neural-network and XGBoost baselines inside
each frozen unseen-source fold. It reports predictions for the neural network,
XGBoost, and their ensemble through `evaluate_records`. It does not tune a
replacement model, retrain on validation examples, promote a release candidate,
or infer prediction intervals.

## Evaluation contract

Create or reuse a private split manifest as described in
[the physical baseline guide](physical-baseline.md). For every outer fold:

1. the neural-network normalizer is adapted on `train` only;
2. both models fit on `train` only;
3. both early-stopping callbacks observe the common family-isolated
   `validation` partition only;
4. the bounded ensemble candidates are scored on `validation` only; and
5. `test` is passed only to the fitted predictors after fitting and selection.

No final refit combines train and validation. Thus each outer source remains
unseen by preprocessing, fitting, early stopping, and ensemble selection. The
private fold report records partition indices, fitted-state and test-data
fingerprints, fit metadata, component predictions, and corrected diagnostics.
Public output contains only pooled and source-balanced aggregates.

## Fixed configuration

The configuration is copied from the verified legacy record, rather than
searched again. Input order is `kb`, `volume`, `surface_area`, `bbox_area`,
`euler_number`, `scale`, and `surface_volume_ratio`.

The neural network uses float32 inputs, a train-fold-only normalization layer,
448 SELU, 601 Mish/dropout 0.3, 544 Mish/dropout 0.3, 416 SELU/dropout 0.3,
and a linear output. Every dense hidden layer uses L2
`3.436477039390904e-06`. AdamW uses learning rate
`0.0006350310563507329`; MSE is the loss. Training is capped at 100 epochs with
batch size 256. Validation MAE early stopping uses minimum delta 0.005,
patience 8, and restored best weights; learning-rate reduction uses factor 0.5,
patience 3, and minimum `1e-8`.

XGBoost uses 900 estimators, depth 9, learning rate 0.01, subsample 0.7,
column sample 0.9, histogram trees, squared-error regression, MAE evaluation,
seed 34, and 50-round early stopping. `n_jobs=1` replaces the notebook's
unbounded `-1` to make resource use explicit. The fixed ensemble candidate is
`0.2 * NN + 0.8 * XGBoost`; the validation-only selection machinery remains
bounded to that single verified candidate.

Unlike the historical notebook, clean preprocessing does not cast `kb` to an
integer, round measurements, recompute a supplied ratio, or fit and remove a
whole-dataset volume tail. Unlike released inference, the neural normalizer is
newly fitted on each permitted train partition rather than loaded from released
weights. XGBoost receives the same unnormalized float32 matrix.

## Reproducibility and private artifacts

The run records the dataset fingerprint, transformation and allocation
versions, split configuration/fingerprint, seed, complete model configuration,
Python/platform and dependency versions, elapsed runtime, fold counts, and
absolute/relative numerical tolerances of `1e-6`. TensorFlow dataset operations
run synchronously and deterministically to avoid platform-specific asynchronous
input-pipeline stalls. The run is capped at 32
folds, 100 neural epochs, 900 trees, 21 ensemble candidates, and one XGBoost
worker. A limit, missing dependency, unsupported Python, or runtime failure
returns a bounded blocker instead of falling back or starting a search.

Use Python 3.11–3.13:

```bash
python3.13 -m venv .venv-learned
.venv-learned/bin/pip install -r requirements-evaluation.txt -r requirements-learned.txt
.venv-learned/bin/python -m minires_evaluation \
  --records private/evaluation-records.json \
  --volume-unit mm3 --scope-confirmed \
  --split-manifest private/splits.json \
  --learned-baselines --private-dir private/learned-run
```

Fitted `.keras` and XGBoost files, row-level predictions, source reports, and
partition audits stay beneath the requested `private/` directory. The
create-only manifest checksums all fitted and report artifacts. Inputs,
released weights, and notebooks are never modified. If the supported optional
dependencies are unavailable, the result is
`learned_baseline_dependencies_required`; preserve that blocker rather than
claiming a smoke result.
