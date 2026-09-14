# Tail-aware expanded candidate tuning

Issue #46 completed a private serious-error audit and one newly predeclared,
bounded candidate-development round against the regenerated training and
validation artifacts.

## Audit result

The audit reconciled all 30 validation rows whose best prior-round prediction
missed sliced resin mass by more than 5 g. It found no inconsistency in scope
confirmation, slicing conditions, target aliases, required geometry, or feature
units. Every prediction feature was finite, and none of the serious-error rows
fell outside the corresponding training-feature range.

Aggregate error increased materially with target mass:

| Sliced resin mass band | Rows | Above-5-g fraction | Mean absolute error | Mean signed error |
| --- | ---: | ---: | ---: | ---: |
| Below 10 g | 1,457 | 0.07% | 0.209 g | +0.043 g |
| 10–25 g | 384 | 1.56% | 0.698 g | +0.282 g |
| 25–50 g | 221 | 4.98% | 1.150 g | +0.151 g |
| 50 g and above | 160 | 7.50% | 3.890 g | −1.876 g |

This supported a bounded hypothesis: target-only mass-band weighting plus broader
configuration coverage might reduce serious absolute errors, particularly for
heavier miniatures. It did not support excluding rows, using anonymous source
metadata as a feature, or changing the fixed gates.

## Predeclared plan

The follow-up plan was documented before fitting. It used primary seed 41 and
second seed 42 with:

- 12 neural-network candidates;
- 12 XGBoost candidates;
- six validation-selected ensembles;
- at most ten eligible second-seed repetitions;
- at most 40 candidate runs and 7,200 seconds; and
- the pinned Python 3.13 candidate environment.

Both component families sampled unweighted training and mean-normalized target
weights of 1× below 10 g, 2× from 10–25 g, 3× from 25–50 g, and 4× at 50 g or
above. Weights came only from training targets. Validation scoring and every
eligibility gate remained unweighted.

## Aggregate result

All 30 initial candidates completed. None passed every serious-error gate, so no
candidate advanced to second-seed repetition or locking.

The best expanded-round result was an ensemble:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6760 g |
| Source-balanced mean absolute error | 0.7659 g |
| Pooled within-2-g fraction | 94.24% |
| Source-balanced within-2-g fraction | 92.82% |
| Pooled above-5-g fraction | 1.49% |
| Source-balanced above-5-g fraction | 2.11% |
| Maximum qualifying-source above-5-g fraction | 6.23% |

The fixed serious-error limits remain 1% pooled, 1% source-balanced, and 2% for
each qualifying source. The result therefore remained ineligible. It also did not
improve on the previous round's best source-neutral ranking metrics.

Target weighting improved the best sampled neural-network tail result relative to
the unweighted neural-network candidates in this round, but neither weighting nor
the expanded configuration coverage produced an eligible candidate. The round
used 158.89 seconds elapsed time, 349.07 process CPU seconds, and a process
high-water resident-set measurement of 1,065,041,920 platform units.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. No model
was refitted, no checksum-verified candidate lock was created, and held-out final
assessment was not started.

The experiment stops here without automatic search expansion. Further work needs
a separate, predeclared hypothesis rather than additional seed or configuration
shopping. Private row-level audit evidence, source reports, paths, fingerprints,
and model artifacts remain unpublished.
