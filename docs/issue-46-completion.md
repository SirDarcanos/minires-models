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

Aggregate error increased materially with sliced resin mass:

| Sliced resin mass band | Rows | Above-5-g fraction | Mean absolute error | Mean signed error |
| --- | ---: | ---: | ---: | ---: |
| Below 10 g | 1,457 | 0.07% | 0.209 g | +0.043 g |
| 10–25 g | 384 | 1.56% | 0.698 g | +0.282 g |
| 25–50 g | 221 | 4.98% | 1.150 g | +0.151 g |
| 50 g and above | 160 | 7.50% | 3.890 g | −1.876 g |

This supported a bounded hypothesis: sliced-resin-mass-band weighting plus broader
configuration coverage might reduce serious absolute errors at larger sliced resin
masses. It did not support excluding rows, using anonymous source
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

Both component families sampled unweighted training and mean-normalized sliced
resin mass weights of 1× below 10 g, 2× from 10–25 g, 3× from 25–50 g, and 4× at
50 g or above. Weights came only from training sliced resin mass labels. Validation scoring and every
eligibility gate remained unweighted.

## Aggregate result

The first execution of the expanded plan exposed a generator defect during
independent review: its 12 XGBoost slots represented only four unique complete
configurations. That completed failed run remains preserved and is not interpreted
as the intended broader search. The generator was corrected to use a deterministic
mixed-radix traversal, and the corrected plan was documented before a new
create-only execution.

All 30 corrected initial candidates completed, including 12 distinct configurations
from each component family. None passed every serious-error gate, so no candidate
advanced to second-seed repetition or locking.

The best corrected expanded-round result was an ensemble combining an unweighted
neural network with a sliced-resin-mass-weighted XGBoost model:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.6470 g |
| Source-balanced mean absolute error | 0.7178 g |
| Pooled within-2-g fraction | 94.96% |
| Source-balanced within-2-g fraction | 93.89% |
| Pooled above-5-g fraction | 1.53% |
| Source-balanced above-5-g fraction | 2.13% |
| Maximum qualifying-source above-5-g fraction | 5.19% |

The fixed serious-error limits remain 1% pooled, 1% source-balanced, and 2% for
each qualifying source. The result therefore remained ineligible. Its pooled mean
absolute error improved slightly over the prior governed round, but its
source-balanced mean absolute error and serious-error rates did not.

Neither sliced resin mass weighting nor the expanded distinct configuration
coverage produced an eligible candidate. The corrected round used 191.17 seconds
elapsed time, 367.22 process CPU seconds, and a process high-water resident-set
measurement of 1,086,980,096 platform units.

## Decision

The round ended as `completed_no_candidate` with `no_eligible_candidate`. No model
was refitted, no checksum-verified candidate lock was created, and held-out final
assessment was not started.

The experiment stops here without automatic search expansion. Further work needs
a separate, predeclared hypothesis rather than additional seed or configuration
shopping. Private row-level audit evidence, source reports, paths, fingerprints,
and model artifacts remain unpublished.
