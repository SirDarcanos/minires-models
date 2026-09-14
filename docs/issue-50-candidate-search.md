# Large-batch extended candidate search

Issue #50 continues governed candidate development after the regenerated-data and
tail-aware expanded rounds produced no eligible candidate.

## Falsifiable hypothesis

The preceding round's best aggregate result remained an ensemble, but its serious-
error rates exceeded every fixed gate. A materially distinct round will test this
hypothesis: larger neural-network batches and longer training, higher XGBoost tree
ceilings, and twice the deterministic component coverage can reduce serious
absolute errors enough for a candidate to satisfy all fixed validation gates.

The hypothesis is falsified for this round if no candidate satisfies every gate.
The round will then stop without new seeds, altered gates, omitted rows, or an
expanded plan.

## Predeclared plan

The `large_batch_extended` plan is fixed before fitting:

- primary seed 41 and second seed 42;
- 24 neural-network candidates using batch sizes 512 or 1,024, maximum training
  lengths of 150 or 200 epochs, and early-stopping patience of 12 or 16 epochs;
- 24 XGBoost candidates using ceilings of 1,500, 1,800, or 2,400 trees;
- the existing unweighted and sliced-resin-mass-band-weighted training choices;
- 12 validation-selected deterministic ensembles;
- second-seed repetition of at most 20 eligible initial candidates;
- at most 80 candidate runs and 14,400 elapsed seconds; and
- the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

The larger batch sizes and tree ceilings ensure that component configurations are
materially distinct from the completed tail-aware expanded plan. The seeded mixed-
radix traversal makes every component configuration within this round unique.
Unused finalist capacity cannot be reassigned.

The fixed eligibility gates remain at most 1% validation above-5-g errors pooled,
1% under equal source weighting, and 2% for every anonymous source group with at
least 200 accepted records. Eligible candidates retain the fixed ranking order:
source-balanced mean absolute error, pooled mean absolute error, within-2-g
fraction, and stable candidate identity.

## Development boundary

Before execution, the committed training and validation artifact SHA-256 values
must match `data/manifest.json`. Fitting and preprocessing use training rows only.
Validation is used only for early stopping, unweighted ranking, eligibility,
ensemble decisions, and locking. Anonymous source groups remain evaluation-only
metadata. The held-out test artifact is not passed, loaded, inspected,
fingerprinted, or otherwise exposed to development.

Execution uses the next fresh create-only directory and preserves every outcome:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-006 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind large_batch_extended
```

## Aggregate result

The one permitted execution completed all 60 initial candidates: 24 neural
networks, 24 XGBoost models, and 12 ensembles. No initial candidate satisfied
every serious-error gate, so none advanced to second-seed repetition and the 20
reserved repetition slots remained unused.

The best result was an ensemble with these source-neutral validation metrics:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6021 g |
| Source-balanced mean absolute error | 0.6592 g |
| Pooled within-2-g fraction | 95.90% |
| Source-balanced within-2-g fraction | 94.65% |
| Pooled above-5-g fraction | 1.22% |
| Source-balanced above-5-g fraction | 1.68% |
| Maximum qualifying-source above-5-g fraction | 5.19% |

The round used 338.10 seconds elapsed time, 1,169.91 process CPU seconds, and a
process high-water resident-set measurement of 1,460,715,520 platform units. Its
create-only manifest was checksum-verified after completion.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. The
hypothesis was falsified for this plan: larger batches, longer training, higher
tree ceilings, and denser deterministic coverage improved the aggregate ranking
metrics but did not satisfy the fixed serious-error gates.

No candidate was repeated or refitted, no checksum-verified lock was created, and
the held-out assessment was not started. This round stops without automatic
expansion. Issue #50 remains open for a future separately justified and
predeclared hypothesis. Private row-level results, source reports, paths,
fingerprints, mappings, and model artifacts remain unpublished.

# Geometry-regime candidate search (predeclared)

## Falsifiable hypothesis

The completed rounds varied model capacity while retaining the seven legacy-compatible
inputs. Aggregate private diagnostics show that serious errors remain concentrated in
a geometrically distinct input regime. This round tests one isolated change: a richer,
deterministic, source-neutral geometry representation with raw, logarithmic, ratio,
and orientation-invariant measurements can reduce serious absolute errors enough for
at least one candidate to satisfy every unchanged validation gate.

The hypothesis is falsified if no candidate satisfies every gate. The round will then
stop without stacking, new model families, added seeds, changed weighting, altered
gates, omitted rows, or an expanded plan.

## Feature contract

This round replaces, rather than augments, the legacy-compatible candidate inputs.
Its ordered 16-feature representation is:

1. mesh volume and surface area;
2. shortest, middle, and longest bounding-box dimensions;
3. bounding-box volume and Euler number;
4. `log1p` transforms of volume, surface area, bounding-box volume, and the three
   ordered bounding-box dimensions;
5. log mesh-volume-to-bounding-box-volume ratio;
6. log surface-area-to-mesh-volume ratio; and
7. log longest-to-shortest bounding-box dimension ratio.

All values are derived deterministically from the seven canonical geometry
measurements already normalized for each record. Bounding-box dimensions are sorted
to remove axis orientation. Missing or non-finite measurements block the round before
fitting; size, area, and volume measurements must also be positive, while Euler number
may be negative but must remain integral. Values that cannot be represented as finite
float32 inputs also block. Anonymous source groups, miniature families, identities,
linkage evidence, file-size proxies, legacy scale, partitions, and join keys never
enter the feature matrix. The fixed clean control retains its legacy feature contract.

## Fixed plan and budget

The `geometry_regime` plan is fixed before fitting:

- primary seed 41 and second seed 42, enforced by plan validation;
- 12 neural-network candidates and 12 XGBoost candidates using the unchanged
  large-batch parameter domains and existing target-weighting choices;
- six deterministic validation-selected ensembles;
- second-seed repetition of at most ten eligible initial candidates;
- at most 40 candidate runs and 7,200 elapsed seconds; and
- the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

Every candidate identity and model specification binds the new ordered feature
contract and its `minires-geometry-regime-features-v1` transformation version. Lock
verification rejects a missing or changed geometry transformation version before
loading or assessment. The eligibility gates remain at most 1% above-5-g errors pooled, 1% under
equal source weighting, and 2% for every anonymous source group with at least 200
accepted validation records. Ranking remains source-balanced mean absolute error,
pooled mean absolute error, within-2-g fraction, and stable candidate identity.

Immediately before execution, the committed training and validation SHA-256 values
must match `data/manifest.json`. Fitting and preprocessing may use training records
only; validation remains limited to early stopping, unweighted ranking, eligibility,
ensemble decisions, and locking. The held-out test artifact is not an argument to
this development seam and must not be loaded, inspected, or fingerprinted.

The one permitted execution will use the next fresh create-only directory:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-007 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind geometry_regime
```

## Aggregate result

The one permitted execution completed all 30 initial candidates: 12 neural
networks, 12 XGBoost models, and six ensembles. No initial candidate satisfied
every serious-error gate, so none advanced to second-seed repetition and the ten
reserved repetition slots remained unused.

The best result was an ensemble with these source-neutral validation metrics:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6662 g |
| Source-balanced mean absolute error | 0.7499 g |
| Pooled within-2-g fraction | 95.09% |
| Source-balanced within-2-g fraction | 93.81% |
| Pooled above-5-g fraction | 1.76% |
| Source-balanced above-5-g fraction | 2.54% |
| Maximum qualifying-source above-5-g fraction | 7.27% |

The round used 155.00 seconds elapsed time, 455.09 process CPU seconds, and a
process high-water resident-set measurement of 1,180,499,968 platform units. Its
create-only manifest was checksum-verified after completion.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. The
hypothesis was falsified for this plan: the richer source-neutral geometry
representation performed worse than the preceding large-batch result on aggregate
MAE and every serious-error measure.

No candidate was repeated or refitted, no checksum-verified lock was created, and
the held-out assessment was not started. This round stops without stacking or
automatic expansion. Issue #50 remains open for a future separately justified and
predeclared hypothesis. Private row-level results, source reports, paths,
fingerprints, mappings, and model artifacts remain unpublished.

# Cross-fitted geometry-gate candidate search (predeclared)

## Falsifiable hypothesis

Existing neural-network and XGBoost candidates have complementary errors that a
constant convex weight cannot exploit. This round tests one isolated change: a
constrained gate fitted from training-only out-of-fold predictions and conditioned
only on deterministic source-neutral geometry can choose between a fixed pair of
existing model families well enough for at least one candidate to satisfy every
unchanged validation gate.

The hypothesis is falsified if no candidate satisfies every gate. The round will
then stop without new component families, targets, feature contracts, seeds,
penalties, folds, gates, omitted rows, or expanded compute.

## Cross-fitting and gate contract

The component search returns to the existing baseline domains: six deterministic
neural-network candidates and six deterministic XGBoost candidates. The existing
validation ranking pairs the three equal-rank members used by three gate slots;
there is no constant-convex ensemble slot in this plan.

For each pair and seed, every training row receives neural-network and XGBoost
out-of-fold predictions from exactly one of five folds. Fold membership is a
stable ordering of the SHA-256 digest of the declared seed and private record
identity, assigned round-robin. Anonymous source groups and miniature families do
not affect fold assignment and never enter a prediction matrix. Each base fit sees
only the other four training folds; its held fold remains inside the training
artifact and supplies early stopping for that cross-fit fit.

The gate minimizes a deterministic ridge-regularized squared-error objective over
the training targets and out-of-fold component predictions. Its linear covariates
are the same ordered 16 source-neutral geometry values defined by
`minires-geometry-regime-features-v1`. The three predeclared ridge penalties are
0.01, 0.1, and 1.0, one per equal-rank pair. The resulting neural-network weight is
clipped to the inclusive interval from zero through one. The gate contract is
versioned as `minires-cross-fitted-geometry-gate-v1`.

After the gate is fixed, both base models fit all training rows. Validation remains
limited to permitted base-model early stopping, unweighted scoring, fixed
eligibility gates, ranking, ensemble decisions, and locking; it never fits gate
coefficients. A selected lock derives fixed component training counts from both
seeds, repeats gate cross-fitting on training only, refits both bases on all
training rows without validation, and checksum-binds both component artifacts,
their preprocessing, and `gate-state.json`.

## Fixed plan and budget

The `cross_fitted_geometry_gate` plan is fixed before fitting:

- primary seed 41 and second seed 42, enforced by plan validation;
- six baseline-domain neural-network and six baseline-domain XGBoost candidates;
- three equal-rank cross-fitted geometry gates with penalties 0.01, 0.1, and 1.0;
- five deterministic training-only cross-fit folds per gate;
- second-seed repetition of at most five eligible initial candidates;
- at most 20 candidate runs and 7,200 elapsed seconds; and
- the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

Unused finalist capacity cannot be reassigned. The eligibility gates remain at
most 1% above-5-g errors pooled, 1% under equal source weighting, and 2% for every
anonymous source group with at least 200 accepted validation records. Ranking
remains source-balanced mean absolute error, pooled mean absolute error,
within-2-g fraction, and stable candidate identity.

Immediately before execution, the committed training and validation SHA-256 values
must match `data/manifest.json`. The held-out test artifact is not an argument to
this development seam and must not be loaded, inspected, or fingerprinted.

The one permitted execution will use the next fresh create-only directory:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-008 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind cross_fitted_geometry_gate
```

## Aggregate result

The one permitted execution completed all 15 initial candidates: six neural
networks, six XGBoost models, and three cross-fitted geometry gates. No initial
candidate satisfied every serious-error gate, so none advanced to second-seed
repetition and the five reserved repetition slots remained unused.

The best result was a cross-fitted geometry gate with these source-neutral
validation metrics:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6891 g |
| Source-balanced mean absolute error | 0.7467 g |
| Pooled within-2-g fraction | 95.14% |
| Source-balanced within-2-g fraction | 94.28% |
| Pooled above-5-g fraction | 1.44% |
| Source-balanced above-5-g fraction | 1.86% |
| Maximum qualifying-source above-5-g fraction | 4.84% |

The round used 238.41 seconds elapsed time, 561.80 process CPU seconds, and a
process high-water resident-set measurement of 1,166,245,888 platform units. Its
create-only manifest was checksum-verified after completion.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. The
hypothesis was falsified for this plan: training-only cross-fitted,
geometry-conditioned gating did not satisfy the fixed serious-error gates.

No candidate was repeated or refitted, no checksum-verified lock was created, and
the held-out assessment was not started. This round stops without automatic
expansion. Issue #50 remains open for a future separately justified and
predeclared hypothesis. Private row-level results, source reports, paths,
fingerprints, mappings, candidate configurations, and model artifacts remain
unpublished.
