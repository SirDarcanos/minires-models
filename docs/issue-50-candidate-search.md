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
