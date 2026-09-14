# Candidate tuning on the regenerated dataset

Issue #49 completed one governed candidate-development round against the committed
regenerated training and validation artifacts.

## Execution summary

- Both development artifact checksums matched the committed manifest before the
  round started.
- The run used the next available create-only sequential private directory; older
  local run directories were preserved and were not reinterpreted as evidence for
  the regenerated dataset.
- Training and validation identities were disjoint, and anonymous source metadata
  remained evaluation-only rather than entering prediction features.
- Development received only the training and validation artifacts. The held-out
  test artifact was not passed to, loaded by, or fingerprinted for candidate
  development.
- The predeclared plan used primary seed 41, second seed 42, six neural-network
  trials, six XGBoost trials, three ensembles, at most five second-seed finalists,
  at most 20 candidate runs, and a 7,200-second limit.
- The pinned Python 3.13 candidate environment supplied NumPy 2.2.6, Keras 3.15.0,
  TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2.

## Aggregate result

The round completed all 15 initial candidates: six neural networks, six XGBoost
models, and three ensembles. No initial candidate satisfied every fixed serious-
error gate, so no candidate was eligible for second-seed repetition and the five
reserved repetition slots remained unused.

The best development result was an ensemble with these source-neutral validation
metrics:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6520 g |
| Source-balanced mean absolute error | 0.7117 g |
| Pooled within-2-g fraction | 94.55% |
| Source-balanced within-2-g fraction | 93.56% |
| Pooled above-5-g fraction | 1.35% |
| Source-balanced above-5-g fraction | 1.72% |
| Maximum qualifying-source above-5-g fraction | 4.15% |

The fixed gates require at most 1% above 5 g both pooled and under equal source
weighting, and at most 2% for each source with at least 200 accepted validation
records. The best result exceeded all three limits. Its ranking metrics therefore
do not make it eligible.

The run used 80.19 seconds elapsed time and 222.50 process CPU seconds. The process
high-water resident-set measurement was 976,322,560 platform units.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. No model
was refitted, no checksum-verified candidate lock was created, and final assessment
was not started.

Candidate tuning stops here. Any later round requires a documented hypothesis and
a newly predeclared finite plan, informed by the serious-error investigation in
issue #46. The observed result does not justify changing the fixed eligibility
gates, expanding this round, or accessing held-out test evidence.
