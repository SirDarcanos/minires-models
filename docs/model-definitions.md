# Model definitions

`src/minires/modeling/definitions.py` is the single maintainer entry point for
MiniRes architecture knowledge. “Model kind” there means neural network, XGBoost,
or ensemble. It is deliberately distinct from a **miniature family**, which is
private grouping evidence used only to prevent evaluation leakage.

## What to inspect

- `ModelSpecification` is the validated, immutable representation. It exposes the
  model kind, ordered prediction features, preprocessing, architecture or estimator
  parameters, training parameters, output unit, readable description, and
  content-derived stable identity.
- `fixed_model_specification()` defines the historical fixed neural network,
  XGBoost estimator, and 0.2/0.8 ensemble through that same interface.
- `candidate_model_specification()` validates one neural-network or XGBoost search
  candidate. The bounded search domains remain next to plan generation in
  `src/minires/modeling/tuning.py`; generated neural-network and XGBoost candidates
  expose their specification through `Candidate.specification`. Validation-selected
  ensemble rules become concrete fold specifications when their members and weight
  are known, and a resolved `LockedCandidate` exposes its final specification.
- `TailCorrectionModelSpecification` describes the frozen two-base continuous
  anchor and bounded correction. The tail lock's public `.specification` exposes
  this immutable composite, including both fixed-count member specifications,
  correction scale, numerical contract, description and content-derived identity.
  Loading and lock verification reconstruct the same composite.
- `ModelRuntime` owns fitting, ensemble composition, serialization, and loading.
  Callers provide explicit `TrainingData`, optional `ValidationData`, a seed, and a
  backend. `TensorflowXGBoostBackend` is the production backend and is the only
  place that constructs TensorFlow layers or an XGBoost estimator.
  Tail locks use `ModelRuntime.load_verified_component`: every backend receives
  the same strictly validated preprocessing state. Neural state has exactly
  `mean` and `variance` arrays matching the feature width, with finite float32-
  representable values and nonnegative variances. XGBoost state is exactly
  `{"xgboost": "unnormalized_float32"}`. The production backend additionally
  checks the serialized NN's normalization and input width, and the serialized
  XGBoost feature count, before returning a predictor. Custom backends must emit
  that same state schema; there is no permissive test-only schema. Existing
  generic loading behavior remains unchanged.
- The locked-model contract is created and verified in
  `src/minires/modeling/tuning.py`. `LockedCandidate.load_predictor()` is the
  assessment seam: assessment asks for a verified predictor and does not inspect
  layer, estimator, or ensemble internals.

## Numerical compatibility

The definitions preserve the established seven-feature order and float32 input.
The neural network fits normalization on training rows, ends in one linear output,
and retains the declared activations, dropout, L2 regularization, optimizer, loss,
early stopping, and optional learning-rate reduction. XGBoost remains unnormalized
and uses the declared tree parameters plus `hist`, MAE evaluation, and the supplied
evaluation seed. A component explicitly declares either grams or the internal
bounding-box occupancy factor as its output unit. Ensemble members must have the
same unit, and ensemble prediction remains a convex weighted mean in that unit.
Target-decomposition locks reconstruct grams outside the component ensemble.

Candidate IDs retain their existing tuning-version payload and format, with a
literal compatibility example covered by the focused tests. A model specification
also has its own `model-<kind>-…` identity for architecture inspection; this
additional identity does not replace candidate IDs, search-plan IDs, or locked
artifact checksums. Newly generated search-plan and fitted-state fingerprints do
change because those identities intentionally bind the refactored source and the
new explicit representation.
