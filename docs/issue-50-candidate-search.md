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

# Legacy geometry-augmentation search (predeclared)

## Falsifiable hypothesis

A small augmentation of the retained seven-feature legacy candidate matrix with
explicit nonlinear compactness, shape, and volume-curvature terms can reduce
serious validation errors enough for at least one otherwise unchanged candidate
to satisfy every fixed eligibility gate.

The hypothesis is falsified if no candidate satisfies every gate. The round will
then stop without new features, model families, architectures, weighting, seeds,
gates, omitted rows, or expanded compute.

## Ordered feature contract

The candidate matrix retains these seven legacy inputs in their existing order:
`kb`, `volume`, `surface_area`, `bbox_area`, `euler_number`, `scale`, and
`surface_volume_ratio`. It then appends exactly four values derived from canonical
geometry:

1. mesh volume divided by bounding-box volume;
2. surface area divided by bounding-box volume;
3. squared `log1p` mesh volume; and
4. longest divided by shortest bounding-box dimension.

The curvature choice is frozen as squared `log1p` volume rather than raw volume
squared to limit numerical range while exposing an explicit nonlinear size term.
Bounding-box dimensions are sorted before the aspect ratio is calculated. The
ordered 11-feature contract is versioned as
`minires-legacy-geometry-augmentation-v1` and is bound into every candidate,
model specification, plan identity, and any resulting lock.

Retaining `kb` and `scale` is scientifically acceptable only for this isolated
comparison because it holds the existing candidate contract constant while
adding geometry terms. Their historical semantics remain uncertain and they may
act as source or style proxies. This limitation is predeclared; it is not evidence
that those two legacy inputs are source-neutral. Anonymous source groups,
miniature families, identities, linkage evidence, partitions, paths, artist
identity, and held-out information remain excluded from prediction matrices.

Missing, non-positive, non-finite, or non-float32 canonical volume, area,
bounding-box volume, or dimensions block before fitting. No imputation or row
removal is permitted. The fixed clean control continues to use only its legacy
seven-feature contract.

## Fixed plan and budget

The `legacy_geometry_augmentation` plan is fixed before fitting:

- primary seed 41 and second seed 42, enforced by plan validation;
- six neural-network and six XGBoost candidates from the existing bounded
  baseline component domains;
- three deterministic validation-selected equal-rank constant convex ensembles;
- second-seed repetition of at most five eligible initial candidates;
- at most 20 candidate runs and 7,200 elapsed seconds; and
- the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

This reuses the smallest existing bounded component plan because the feature
contract is the sole intervention. The NN/XGBoost families, component parameter
domains, target, unweighted training, ensemble rule, eligibility gates, ranking,
and lock rules are unchanged. Unused finalist capacity cannot be reassigned.

Immediately before execution, `data/train.jsonl` and `data/validation.jsonl` must
match the SHA-256 values in `data/manifest.json`. The held-out test artifact must
not be read, inspected, or fingerprinted. The one permitted execution will use
the next fresh create-only directory:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-009 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind legacy_geometry_augmentation
```

The fixed serious-error limits remain 1% pooled, 1% source-balanced, and 2% for
every anonymous source group with at least 200 accepted validation records.
Ranking remains source-balanced mean absolute error, pooled mean absolute error,
within-2-g fraction, and stable candidate identity. If no initial candidate is
eligible, there is no second seed, refit, lock, held-out assessment, or automatic
expansion.

## Aggregate result

The one permitted execution completed all 15 initial candidates: six neural
networks, six XGBoost models, and three ensembles. No initial candidate satisfied
every serious-error gate, so none advanced to second-seed repetition and the five
reserved repetition slots remained unused.

The best result was an ensemble with these source-neutral validation metrics:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6269 g |
| Source-balanced mean absolute error | 0.6962 g |
| Pooled within-2-g fraction | 95.27% |
| Source-balanced within-2-g fraction | 94.24% |
| Pooled above-5-g fraction | 1.35% |
| Source-balanced above-5-g fraction | 1.76% |
| Maximum qualifying-source above-5-g fraction | 4.84% |

The round used 93.92 seconds elapsed time, 226.61 process CPU seconds, and a
process high-water resident-set measurement of 1,001,029,632 platform units. Its
create-only manifest was checksum-verified after completion.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. The
hypothesis was falsified for this plan: augmenting the legacy matrix with the four
predeclared geometry terms did not satisfy the fixed serious-error gates.

No candidate was repeated or refitted, no checksum-verified lock was created, and
the held-out assessment was not started. This round stops without automatic
expansion. Issue #50 remains open for a future separately justified and
predeclared hypothesis. Private row-level results, source reports, paths,
fingerprints, mappings, candidate configurations, and model artifacts remain
unpublished.

# Tail-aligned selection search (predeclared)

## Falsifiable hypothesis

The completed searches select neural-network epochs, XGBoost tree counts, component
pairs, and convex ensemble weights primarily by validation mean absolute error,
although eligibility is determined by three serious-error gates. Preserved
source-neutral diagnostics also show substantial component diversity on the serious
misses. This round tests one isolated intervention: aligning every validation-based
selection decision with the fixed serious-error gates before applying the unchanged
ranking metrics can reduce serious errors enough for at least one candidate to
satisfy every gate.

The hypothesis is falsified if no candidate satisfies every gate. The round then
stops without new losses, features, model families, seeds, gates, omitted rows,
reslicing, or expanded compute.

## Selection contract

The versioned `serious_error_gates_then_ranking_v1` rule scores predictions in this
order:

1. eligible checkpoints before ineligible checkpoints;
2. smallest maximum normalized excess over the pooled, source-balanced, and
   qualifying-source serious-error limits;
3. smallest sum of those normalized excesses;
4. source-balanced mean absolute error;
5. pooled mean absolute error;
6. pooled within-2-g fraction; and
7. stable checkpoint, weight, or candidate identity.

Normalized excess is zero at or below a fixed limit and otherwise the observed rate
divided by its limit, minus one. Every bounded epoch and tree checkpoint is scored;
there is no patience-based tail stopping. The selected count is fixed for any
training-only refit. Equal-rank component pairing and every predeclared convex-weight
grid use the same selection contract. Final candidate eligibility and ranking are
unchanged.

Anonymous source groups remain evaluation metadata. A validation scorer may group
completed prediction vectors to calculate the fixed gates, but source metadata is
not passed in prediction matrices, model inputs, training weights, preprocessing, or
model construction.

## Fixed plan and budget

The `tail_aligned_selection` plan is fixed before fitting:

- the unchanged seven legacy candidate inputs and transformation contract;
- primary seed 41 and second seed 42;
- six Huber neural-network candidates and six XGBoost candidates from the existing
  bounded baseline domains;
- three deterministic equal-rank convex ensembles;
- at most five eligible second-seed repetitions;
- at most 20 candidate runs and 7,200 elapsed seconds; and
- the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

The explicit validation-selection parameter makes every component contract distinct
from completed configurations. Huber is fixed because it produced the strongest
neural tail result in the preserved large-batch evidence; this is not another broad
hyperparameter sweep.

Immediately before execution, only `data/train.jsonl` and `data/validation.jsonl`
will be checksum-verified against `data/manifest.json`. Fitting and preprocessing
use training rows only. Validation is limited to checkpoint selection, component and
ensemble decisions, unweighted scoring, eligibility, ranking, and locking. The
held-out test artifact is not an argument and will not be read or fingerprinted.

The one permitted execution will use the next fresh create-only directory:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-010 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind tail_aligned_selection
```

## Aggregate result

The one permitted execution completed all 15 initial candidates: six neural
networks, six XGBoost models, and three ensembles. No initial candidate satisfied
every serious-error gate, so none advanced to second-seed repetition and the five
reserved repetition slots remained unused.

The best result under the unchanged development ranking was an ensemble:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.5867 g |
| Source-balanced mean absolute error | 0.6326 g |
| Pooled within-2-g fraction | 95.72% |
| Source-balanced within-2-g fraction | 94.68% |
| Pooled above-5-g fraction | 0.90% |
| Source-balanced above-5-g fraction | 1.16% |
| Maximum qualifying-source above-5-g fraction | 2.77% |

A different ensemble came closest to eligibility: it passed the pooled gate at
0.77% and the source-balanced gate at 0.92%, while missing the qualifying-source
gate at 2.42%. The fixed limits remain 1%, 1%, and 2%, respectively.

The round used 1,635.38 seconds elapsed time, 1,995.04 process CPU seconds, and a
process high-water resident-set measurement of 1,001,291,776 platform units. Its
create-only manifest was checksum-verified after completion.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. Tail-
aligned validation selection appeared to materially improve the serious-error
results and produced an ensemble that passed both aggregate gates, but it did not
satisfy the unchanged qualifying-source gate.

Independent review then found that non-finite checkpoint predictions were not
explicitly rejected before calculating the selection key. The selected final
predictions were finite, but the private evidence does not retain every intermediate
checkpoint vector and therefore cannot prove that the intended ordering was applied
to every checkpoint. This completed run remains preserved and is not interpreted as
a valid test of the predeclared selection hypothesis.

No candidate was repeated or refitted, no checksum-verified lock was created, and
the held-out assessment was not started. A corrected execution requires a new
create-only directory and predeclaration. Private row-level results, source reports,
paths, fingerprints, mappings, candidate configurations, and model artifacts remain
unpublished.

# Corrected tail-aligned selection search (predeclared)

The first execution exposed one bounded implementation defect during independent
review: non-finite checkpoint predictions could compare as eligible because IEEE
NaN comparisons do not increment threshold counts. The completed run remains
preserved and is not reinterpreted.

The correction rejects any checkpoint prediction vector with the wrong length or a
non-finite value before metric calculation. An invalid checkpoint ranks behind every
finite checkpoint; if all checkpoints are invalid, final prediction validation
blocks the candidate. Final source metrics also reject non-finite targets or
predictions. Regression coverage binds this behavior.

This corrected run tests the already documented hypothesis and changes nothing else:
seven legacy inputs, Huber neural candidates, the same XGBoost domain, the versioned
gate-first selection rule, seeds 41/42, six plus six components, three ensembles, at
most five eligible repetitions, at most 20 runs, and 7,200 seconds. Gates, final
ranking, checksum and environment verification, source-metadata isolation, and the
held-out-test exclusion remain unchanged.

The one permitted corrected execution will use the next fresh create-only directory:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-011 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind tail_aligned_selection
```

## Corrected aggregate result

The corrected execution completed all 15 initial candidates: six neural networks,
six XGBoost models, and three ensembles. No initial candidate satisfied every gate,
so none advanced to second-seed repetition and the five reserved slots remained
unused.

The best result under the unchanged development ranking was an ensemble:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.5867 g |
| Source-balanced mean absolute error | 0.6326 g |
| Pooled within-2-g fraction | 95.72% |
| Source-balanced within-2-g fraction | 94.68% |
| Pooled above-5-g fraction | 0.90% |
| Source-balanced above-5-g fraction | 1.16% |
| Maximum qualifying-source above-5-g fraction | 2.77% |

A different ensemble was closest to eligibility. It passed the pooled gate at
0.77% and the source-balanced gate at 0.92%, but its maximum qualifying-source
rate was 2.42% against the unchanged 2% limit.

The run used 1,639.57 seconds elapsed time, 1,992.96 process CPU seconds, and a
process high-water resident-set measurement of 1,018,101,760 platform units. Its
create-only manifest and all four recorded artifact checksums were independently
verified after completion.

## Corrected decision

The corrected round ended as `completed_no_candidate` with
`no_eligible_candidate`. The selection hypothesis improved tail performance enough
for one ensemble to pass both aggregate gates, but not the qualifying-source gate.
It is therefore falsified for this fixed plan.

No candidate was repeated or refitted, no lock was created, and held-out assessment
was not started. The round stops without automatic expansion. Issue #50 remains
open for a separately justified, predeclared hypothesis. Private row-level results,
source reports, paths, fingerprints, mappings, and model artifacts remain
unpublished.

## Predeclaration: nonlinear training-OOF stacking (not executed)

The next hypothesis is that a fixed shallow nonlinear combiner can exploit
complementary errors among four frozen, structurally diverse existing bases enough
to satisfy the unchanged serious-error gates. It uses no reslicing and no new raw
prediction feature. Exact base and combiner contracts, five-fold identity-hash
assignment, seeds 41/42, the eight prediction-only meta inputs, 54-fit and
7,200-second limits, unchanged gates/ranking, training-only refit, serialization,
and strict stop behavior are fully declared in
[`candidate-tuning.md`](candidate-tuning.md#predeclared-nonlinear-out-of-fold-stacking-plan-not-yet-executed)
and enforced by `minires.modeling.tuning`.

Anonymous source groups remain evaluation-only. Validation labels score, gate, and
rank; they never fit a base or combiner. Each training row must receive exactly one
finite OOF prediction from each base. Every candidate is evaluated under both
seeds, with unfavorable evidence retained. No candidate is locked unless both seed
results and their equal combination pass every fixed gate. Any incomplete or
invalid run stops in a fresh create-only directory without replacement, tuning,
budget recycling, or automatic expansion. This predeclaration does not start the
private run or access held-out assessment data.

## Nonlinear training-OOF stacking result

The one permitted execution completed all three fixed stack candidates under both
seeds. No candidate was eligible under either seed or their equal combination.
The best candidate under the unchanged ranking was also closest to the fixed gates:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 1.5295 g |
| Source-balanced mean absolute error | 1.4626 g |
| Pooled within-2-g fraction | 76.78% |
| Source-balanced within-2-g fraction | 79.85% |
| Pooled above-5-g fraction | 3.65% |
| Source-balanced above-5-g fraction | 3.77% |
| Maximum qualifying-source above-5-g fraction | 6.12% |

The round used all 54 predeclared fits in 345.68 elapsed seconds and 1,217.66
process CPU seconds, with a process high-water resident-set measurement of
1,167,556,608 platform units. Its four-file create-only manifest was independently
checksum-verified. No candidate was refitted beyond the predeclared training-only
seed fits, no lock was created, and held-out assessment was not started.

Post-run review found that the generic `second_seed_comparison.shortfall` report
counted seed-41 eligibility rather than the three mandatory stack repetitions. The
preserved allocation, run count, candidate history, and results correctly record all
three seed-42 repetitions, so this reporting-only defect did not affect fitting,
predictions, metrics, eligibility, or ranking. The reporter and its regression test
were corrected after the run; the create-only run remains unchanged.

The nonlinear stacking hypothesis is falsified for this fixed plan. This round
stops without automatic expansion, and Issue #50 remains open for a separately
justified and predeclared hypothesis. Private row-level results, source reports,
paths, fingerprints, mappings, and model artifacts remain unpublished.

## Predeclaration: guarded training-OOF residual stacking (not executed)

The failed direct-target stack strongly compressed its prediction range and
replaced already-strong continuous base predictions with shallow tree plateaus.
The next materially distinct hypothesis is that one deterministic ridge residual
learner, constrained to make only a small additive correction to a fixed four-base
mean anchor, can improve serious validation errors without being able to collapse
the anchor.

The complete contract is declared in
[`candidate-tuning.md`](candidate-tuning.md#predeclared-guarded-residual-stacking-plan)
and enforced by `minires.modeling.tuning`. It freezes the same four base contracts,
five identity-only training folds, and seeds 41/42. The anchor is the arithmetic
mean of the four base predictions. The residual learner receives only that anchor,
the four base-minus-anchor differences, and base spread. Training-OOF statistics
standardize and clip those inputs; ridge penalty 1.0 fits a target residual clipped
to ±2 g. The three fixed variants are exact identity, half-strength correction
bounded to ±1 g, and full correction bounded to ±2 g.

All three candidates run under both seeds. The finite ceiling is six candidate
evaluations, 50 fits, and 7,200 seconds. Existing serious-error gates and ranking
are unchanged. Aggregate private evidence records OOF-versus-full-fit prediction
shift without row, source, family, or identity values. No source metadata,
miniature family, identity, linkage, partition, path, or held-out information enters
model fitting or fold assignment beyond deterministic identity-only assignment.

The round must stop without substitution or expansion if incomplete or
unsuccessful. It may execute once in fresh create-only `run-013` only after this
implementation/predeclaration commit and a source-neutral Issue #50 comment. This
predeclaration did not fit a model, access held-out evidence, or start Issue #33.

## Guarded training-OOF residual-stacking result

The round ended as `completed_no_candidate`. The one permitted execution completed
all three fixed candidates under both seeds. No candidate was eligible under either
seed or their equal combination. The best development result was the full ±2 g
correction under seed 41:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.5930 g |
| Source-balanced mean absolute error | 0.6587 g |
| Pooled within-2-g fraction | 95.68% |
| Source-balanced within-2-g fraction | 94.74% |
| Pooled above-5-g fraction | 1.22% |
| Source-balanced above-5-g fraction | 1.55% |
| Maximum qualifying-source above-5-g fraction | 4.15% |

The run used all six candidate evaluations and all 50 fits: 40 OOF base fits,
eight full-training base fits, and two analytical residual fits. It used 349.54
elapsed seconds, 1,212.51 process CPU seconds, and a process high-water resident-
set measurement of 1,224,359,936 platform units. The complete five-artifact
create-only manifest was independently checksum-verified.

The bounded correction avoided the prediction collapse of the preceding direct-
target stack and retained aggregate accuracy near the anchor, but it did not pass
any fixed serious-error gate. The hypothesis is falsified for this plan. No lock
was created, held-out evidence was not read or fingerprinted, and Issue #33 was not
started. The round stops without changed bounds, replacement candidates, added
seeds, omitted rows, or automatic expansion. Issue #50 remains open for a
separately justified and predeclared hypothesis.

## Predeclaration: tail-focused closest-anchor correction

The run-013 four-base mean was weaker than run-011's closest 80% NN/20% XGBoost
ensemble. Its clipped-target ridge correction remained small and did not satisfy
any gate. Preserved diagnostics do not support a simple calibration-bias
hypothesis: the closest anchor's 17 pooled serious errors were split between eight
underestimates and nine overestimates, with ten errors above 8 g and only one
between 5 and 6 g. Eighteen further errors were between 4 and 5 g. A new correction
must preserve severe-error information rather than flatten all residual targets
to the same small bound.

The falsifiable hypothesis is that a bounded additive learner with an unclipped,
serious-error excess loss can improve the **exact closest anchor**, first on
independently held-out training rows, then satisfy every unchanged validation gate.
This is not another model-family or configuration sweep, a distribution/window
estimator, or a replacement for the continuous anchor.

The complete fixed contract is declared in
[`candidate-tuning.md`](candidate-tuning.md#predeclared-tail-focused-closest-anchor-correction)
and enforced by the existing candidate-development and lock-loading interfaces.
The frozen pair is the Mish 256/128/64 NN with Adam/Huber, batch 256 and fixed 87
epochs, and the depth-9 XGBoost with learning rate 0.05 and fixed 1,091 trees. The
original component ceilings and selection settings remain identifiable but every
new fit uses the selected counts with no early stopping. The exact anchor operation
is `0.8 * neural + (1.0 - 0.8) * xgboost` in float64, using the original legacy
feature/preprocessing contracts.

Only two variants are declared: exact identity and one full correction bounded to
±2 g. The correction uses the anchor, signed disagreement, and absolute
disagreement; training-OOF standardization and clipping bound the design inputs.
A four-coefficient L1 constraint bounds departure everywhere. Exactly 2,000
projected-gradient steps fit the fixed convex objective: ordinary squared error
weighted 0.1, squared excess beyond 4.5 g weighted 4, squared anchor departure
weighted 1, and squared coefficients weighted 0.01. These are fixed engineering
choices, not empirically chosen optima. Labels and residuals are not clipped.

Before any production fit or validation scoring, **both seeds 41 and 42** must
complete an honest 20% training holdout evaluation. Each honest learner uses only
the other 80%, including nested five-fold OOF base predictions and all learned
preprocessing. Identity-hash folds do not use source/family metadata or targets.
For each seed, correction must strictly improve the declared prediction loss
without increasing MAE or the above-5-g count. Failure under either seed stops the
whole round as `training_evidence_rejected`, preserving aggregate evidence and
performing no validation scoring or lock. This evidence remains conditional on
the historical pair/count choice made with reused validation; it is not wholly
untouched pipeline evidence.

If both seeds qualify, each seed independently repeats OOF correction training and
base fitting on all training rows. Identity and corrected candidates are then
scored under both seeds regardless of unfavorable validation results. The finite
ceiling is 26 honest-stage fits plus 26 conditional production-stage fits, four
validation candidate evaluations, and 7,200 seconds. Existing serious-error gates
(1% pooled, 1% source-balanced, 2% per qualifying source) and ranking are unchanged.
No budget is recycled. Source groups remain evaluation-only; no new source/family,
identity, linkage, partition, or path prediction input is added.

A successful lock uses the exact eligible seed-42 training-only state already
scored; equal seed weighting combines metrics, not locked predictions. There is
no undisclosed refit. Checksums and semantic verification bind the precise model,
optimizer, preprocessing, splits, complete artifacts, honest qualification,
OOF/full-fit shift, dependencies, gates, and no-test-access contracts. Incomplete
or invalid evidence cannot qualify. No eligible candidate, weak honest evidence,
or any failure stops without changed constants, folds, seeds, omitted rows,
replacement candidates, or expansion.

This implementation and predeclaration do not authorize execution. A separately
authorized execution, after the committed predeclaration and source-neutral Issue
#50 comment, would use the fresh create-only `run-014` directory. This change does
not create that directory, fit private models, access or fingerprint held-out
evidence, publish results, or start Issue #33. Existing runs and training/validation
artifacts remain unchanged.

## Tail-focused closest-anchor correction result

The separately authorized execution ended as `training_evidence_rejected` with
`honest_training_correction_gate_failed`. Both seeds completed the honest stage;
seed 41 failed every qualification condition, while seed 42 passed all three.
Each seed evaluated 2,074 independently held-out training rows:

| Training-holdout metric | Seed 41 anchor | Seed 41 corrected | Seed 42 anchor | Seed 42 corrected |
| --- | ---: | ---: | ---: | ---: |
| Pooled MAE | 0.6234 g | 0.7056 g | 0.7856 g | 0.6997 g |
| Above-5-g count | 34 | 37 | 39 | 35 |
| Declared prediction loss | 9.6453 | 9.6811 | 17.8303 | 16.3845 |

These are training-holdout comparisons, not validation eligibility or ranking
metrics. The declared loss excludes the optimization's correction and coefficient
penalties. The seeds use different deterministic holdouts; the paired comparisons
are within each seed. Evidence remains conditional on the historically selected
anchor and training counts.

The run completed 26 of at most 52 fits: 20 inner-OOF base fits, four partition-wide
base fits, and two numerical correction fits. It used 72.40 elapsed seconds,
202.12 process CPU seconds, and a process high-water resident-set measurement of
909,754,368 platform units. The complete six-artifact create-only manifest and
recorded qualification decisions were independently verified.

Because both seeds had to qualify, the round stopped before production fitting
or validation scoring. All four validation candidate slots were skipped and unused
capacity was not recycled. The unchanged validation gates were not evaluated;
no candidate was selected, no lock was created, and held-out evidence was not
read or fingerprinted. Issue #33 was not started.

The correction did not show the required consistent training-only improvement.
This fixed hypothesis is rejected at its prerequisite gate and stops without
changed losses, bounds, folds, seeds, replacement candidates, or expansion. Issue
#50 remains open for a separately justified and predeclared hypothesis. Existing
runs remain preserved; private row-level evidence and source identities remain
unpublished.

## Predeclaration: crossed training split/model-seed stability diagnostic

The two tail-focused honest-stage seeds changed both model randomness and the
training holdout, so their difference cannot identify which factor caused the
opposite qualification outcomes. Before proposing another correction or anchor,
this diagnostic holds the exact run-011 closest anchor and the existing
`minires-tail-focused-correction-v1` algorithm fixed and crosses two deterministic
training splits with two model seeds.

Outer split seeds 101 and 202 assign training identities by the existing five-fold
SHA-256 round-robin algorithm, with fold zero as the 20% training holdout. Their
nested correction-OOF assignments use fixed seeds 1101 and 1202, respectively.
Model seeds 41 and 42 reach only model initialization and sampling. Every one of
the four split/model cells fits the frozen neural network for 87 epochs and XGBoost
for 1,091 trees without early stopping, learns the unchanged correction from the
remaining 80% training rows' five-fold OOF predictions, and then reports anchor
and corrected metrics on that cell's outer holdout.

The finite allocation is 40 inner-OOF base fits, eight outer-partition base fits,
and four deterministic correction fits: 52 fits within 7,200 seconds. Aggregate
private evidence will contain cell metrics, within-split model-seed contrasts on
identical held rows, and within-model split contrasts. The latter are descriptive
comparisons of different held rows, not paired or causal estimates. No metric
qualifies, ranks, selects, locks, or promotes a candidate.

The diagnostic seam accepts training records only. It cannot receive validation or
held-out-test records, and its create-only `run-015` package contains only a fixed
plan, aggregate evidence, and a checksummed manifest—no raw identities, source
groups, rows, predictions, or source mappings. It verifies only the committed
training checksum and pinned environment before fitting. This predeclaration does
not execute the command, read validation or held-out evidence, or start Issue #33.
A separately authorized one-time execution requires this committed change and a
source-neutral Issue #50 comment:

```bash
.venv-candidates/bin/python -m minires.modeling.training_stability \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-015 \
  --volume-unit mm3 \
  --scope-confirmed
```

## Crossed training split/model-seed stability result

The separately authorized `run-015` execution completed all four cells and all 52
predeclared fits. It used 150.97 elapsed seconds and 411.65 process CPU seconds.
The private create-only manifest's two artifact checksums were recomputed and
matched. No validation labels or held-out evidence were accessed.

| Outer split / model seed | Anchor MAE | Corrected MAE | Anchor above-5-g count | Corrected above-5-g count | Anchor loss | Corrected loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 / 41 | 0.8953 g | 0.9646 g | 34 | 37 | 511.7415 | 509.1025 |
| 101 / 42 | 0.8838 g | 0.9672 g | 34 | 40 | 474.2077 | 472.0289 |
| 202 / 41 | 0.8988 g | 0.8503 g | 35 | 37 | 444.7951 | 440.9340 |
| 202 / 42 | 0.9501 g | 0.8396 g | 42 | 37 | 405.7071 | 405.5510 |

The correction lowered its declared loss in every cell, but it was not stable on
ordinary or serious-error measures. It worsened MAE and above-5-g count for both
model seeds under split 101; under split 202 it improved MAE for both seeds and
reduced the above-5-g count only for seed 42. The observed training-stage result
varied across model seeds and split-associated held-out samples. Corrected-MAE
differences across splits were about 0.11–0.13 g, compared with about 0.01 g across
model seeds within a split. Because split contrasts cover different held rows, this
is descriptive evidence, not a causal attribution to row assignment.

This diagnostic neither evaluates the unchanged validation gates nor selects a
candidate, lock, or next intervention. It stops without a retry, correction change,
budget expansion, validation access, held-out assessment, or Issue #33 work. The
completed private run remains preserved.

## Predeclaration: paired correction-transition diagnostic

Status: **IMPLEMENTED NOT EXECUTED**.

The preserved `run-015` reports lower unpenalized prediction loss in all four
crossed cells, worse above-5-g counts in three, and worse MAE under both split-101
model seeds. It contains no row prediction vectors from which repair/harm evidence
can be reconstructed. This new diagnostic collects prospective paired accounting;
it neither changes that completed diagnostic nor treats the observed trade-off as
a software bug, candidate experiment, or causal conclusion.

The complete predeclared accounting and interpretation contract is in
[`candidate-tuning.md`](candidate-tuning.md#predeclared-correction-transition-diagnostic).
The exact closest anchor and tail-focused correction remain fixed: 87 NN epochs,
1,091 XGBoost trees, outer split seeds 101/202, respective inner seeds 1101/1202,
and model seeds 41/42. Four cells use 40 inner-OOF base fits, eight outer-partition
base fits, and four correction fits, **at most 52 fits and 7,200 seconds**.

Each cell records aggregate 2×2 above-5-g transitions, MAE and separate ordinary/
excess-squared unpenalized-loss contributions by transition and fixed anchor bins
`[0,4]`, `(4,5]`, `(5,7]`, `>7`, residual signs, correction magnitudes/directions,
and existing OOF/full-fit shifts. Exactly ±5 is nonserious; exactly ±7 is in the
potentially repairable bin. The theoretical label-aware ±2 oracle is not an
estimator. Fixed descriptive rules ask whether near-threshold harms offset repairs
(nonzero harms at least repairs and a strict majority in `(4,5]`), whether severe
tails dominate anchor excess-squared loss (strictly more than half contributed by
`>7`), and verify the bound cannot repair `>7`. Contributions divide by total cell
count, and conservation/finite checks fail closed. These are predeclared accounting
conventions, not learned gates or intervention-selection rules.

The training-only CLI verifies only the committed training checksum and pinned
Python/dependency environment. Its private create-only plan, aggregate evidence,
and checksummed manifest record contracts, provenance, exact fit accounting, and
completed/failed/uncompleted outcomes without row vectors, IDs, paths, or
source/family values in evidence. Validation/test inputs are unavailable. There
is no selection, promotion, lock, early stopping, row exclusion, changed seed,
correction change, tuned gate, retry, or capacity recycling.

The prospective command is fixed to a new create-only directory; it has not run
and the directory was not created or inspected during implementation:

```bash
.venv-candidates/bin/python -m minires.modeling.correction_transition \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-016 \
  --volume-unit mm3 \
  --scope-confirmed
```

One execution requires separate authorization after committed predeclaration and
a source-neutral Issue #50 predeclaration. It stops after one completed or blocked
attempt without automatic expansion. Evidence remains conditional on the
historically validation-selected anchor/counts; within-cell pairing does not make
contrasts between different held training splits causal. No conclusion or targeted
correction change is justified before executed evidence supports it and a separate
decision authorizes it. No private fitting, dataset reads, validation/test reads or
fingerprints, publication, candidate locking, or Issue #33 work occurred during
this implementation. All prior runs remain preserved.

## Correction-transition diagnostic result

The separately authorized diagnostic, predeclared at commit `04af4e0` and on
Issue #50 before fitting, completed all four fixed cells and 52 fits. It used
147.80 elapsed seconds and 406.20 process CPU seconds. Both artifacts in the
create-only manifest were independently checksum-verified; transition counts and
MAE/loss contribution conservation were independently recomputed from the
preserved aggregate evidence. No validation or held-out-test evidence was accessed.

Each cell contains 2,074 held-out training rows. A repair moves an anchor error
strictly above 5 g to at most 5 g; a harm makes the reverse transition.

| Outer split / model seed | Repairs | Harms | Above-5-g count, anchor → corrected | MAE change | Prediction-loss change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 / 41 | 4 | 7 | 34 → 37 | +0.0693 g | -2.6390 |
| 101 / 42 | 3 | 9 | 34 → 40 | +0.0834 g | -2.1788 |
| 202 / 41 | 2 | 4 | 35 → 37 | -0.0485 g | -3.8612 |
| 202 / 42 | 5 | 0 | 42 → 37 | -0.1105 g | -0.1560 |

The first three cells met the predeclared near-threshold-harm flag: respectively
7/7, 8/9, and 3/4 introduced serious errors began with anchor errors in (4,5] g.
The favorable cell repaired five errors without introducing any. This accounts
for the serious-count contrast; it does not identify a causal property of either
split or establish that model seed 42 is generally preferable.

Errors beyond 7 g contributed 99.9815%–99.9936% of anchor weighted excess-squared
loss across the four cells. That bin's prediction-loss reductions were 2.7158,
2.2843, 3.8871, and 0.1059, respectively. In the first three cells those reductions
exceeded the total loss reduction, offsetting a net loss increase elsewhere.
The loss can therefore improve while serious-error count worsens. The ±2 g
label-aware oracle leaves at least 24, 22, 26, and 24 serious errors respectively;
these are theoretical training-holdout limits, not deployable predictions or
validation eligibility results. No assertion of label defects or missing inputs
follows from this accounting alone.

The evidence supports considering a separately predeclared correction objective
that limits the influence of errors beyond the correction's repair capacity and
explicitly evaluates newly introduced serious errors on training holdouts. It does
not yet establish that such a change will help, choose its constants, or authorize
implementation/fitting. The frozen anchor and correction remain unchanged.

This is conditional training-only diagnostic evidence, not untouched pipeline or
validation evidence: historical anchor/count selection used reused validation.
No candidate was selected and no lock was created. The attempt stops without
retries, new seeds, changed bounds/losses, budget recycling, validation access, or
Issue #33 work. Issue #50 stays open and all prior runs remain preserved.
