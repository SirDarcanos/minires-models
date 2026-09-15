# Candidate tuning and locked assessment

Use this private workflow to search for a replacement model without changing the
clean fixed-configuration baseline. Candidate tuning and final assessment are
separate phases. Neither phase publishes evidence.

## Preconditions

Use Python 3.11–3.13 and install the pinned evaluation and learned-model
requirements:

```bash
python3.13 -m venv .venv-candidates
.venv-candidates/bin/pip install \
  -e . \
  -r requirements/evaluation.txt \
  -r requirements/legacy.txt \
  -r requirements/learned.txt
```

Model development consumes the explicit `train.jsonl` and `validation.jsonl`
artifacts produced by the expanded-dataset workflow. Every row needs a unique
private `_id` and an anonymous source group; missing source-group evidence, an
invalid identity, or evidenced duplicate/linkage that crosses the two artifacts
blocks development. Anonymous source groups remain evaluation-only metadata.
They, partition fields, miniature families, duplicate/linkage values, and join
keys never enter the model feature matrix. This mixed-source path does not require
miniature-family evidence or leave-one-source-out folds.

## Run the complete command-level workflow

The end-to-end command is the preferred lifecycle tracer. It creates the search
plan, runs the bounded development search, repeats finalists, locks an eligible
candidate, and then assesses that immutable candidate against explicitly supplied
final evidence. It never publishes anything.

First create a private slicing-configuration record. The volume unit must agree
with the command and every accepted final row must carry exactly the declared
supported slicing conditions:

```json
{
  "volume_unit": "mm3",
  "slicing_conditions": {
    "layer_height_mm": 0.05
  }
}
```

Then choose a new output directory beneath `private/`:

```bash
.venv-candidates/bin/python -m minires.evaluation.workflow \
  --training-records private/issue-31/train.jsonl \
  --validation-records private/issue-31/validation.jsonl \
  --final-records private/issue-31/test.jsonl \
  --legacy-artifacts private/legacy-artifacts \
  --slicing-configuration private/slicing-configuration.json \
  --output-root private/end-to-end/run-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --second-seed 42 \
  --bootstrap-seed 1729 \
  --neural-network-trials 6 \
  --xgboost-trials 6 \
  --ensemble-trials 3 \
  --second-seed-candidates 5 \
  --maximum-candidate-runs 20 \
  --maximum-elapsed-seconds 7200
```

The fixed budget is six neural-network trials, six XGBoost trials, three
validation-selected ensembles, and a second-seed repetition of the best five
eligible initial candidates. The fixed learned model remains a separate control.
Changing the 6/6/3 plus best-five allocation, exceeding 20 candidate runs, or
requesting more than 7,200 seconds is rejected rather than treated as permission
to expand the experiment.

Training rows alone are used for fitting, learned preprocessing, and the locked
refit. Validation rows are available only to early stopping, candidate and
ensemble ranking, threshold decisions, training-duration selection, and candidate
locking. The development seam accepts no test argument or path. The encompassing
workflow passes only the two development artifacts into that seam and does not
read final evidence until the lock has been created and checksum-verified. The
final stage only performs paired assessment; it cannot change candidate weights,
training duration, ranking, or the compute budget.

The create-only package contains:

- `tuning/search-plan.json`, every candidate outcome, both seed results, rankings,
  selected epoch/tree counts, partition evidence, and the locked fitted artifacts;
- `assessment/assessment.json` with the paired final result, blockers, confidence
  analysis, gate outcomes, and private row-level evidence;
- `evidence-index.json` with code/configuration and normalized-input identities,
  development split identities, dependency and platform versions, elapsed and CPU
  time, peak memory, before/after input checks, phase status, promotion decision,
  and checksums;
- `public-summary-draft.json`, containing only aggregate statuses, metrics,
  resource summaries, gates, and limitations; and
- `public-summary-review.json` plus `manifest.json`, recording automated screening,
  the required manual content review, create-only checksums, and that publication
  did not occur.

Interrupted, time-limited, failed, or blocked runs retain completed phase evidence
and skipped-run accounting. They cannot claim an incomplete search, candidate lock,
or assessment as complete. Missing or mismatched slicing evidence, unavailable
pinned artifacts, insufficient source coverage, or failed promotion gates remain
bounded blockers. Inputs, released weights, and supplied final records are never
written by this workflow.

To reassess an existing lock without tuning, candidate selection, or additional
compute, use the same command in assessment-only mode:

```bash
.venv-candidates/bin/python -m minires.evaluation.workflow \
  --assessment-only \
  --locked-candidate private/end-to-end/run-001/tuning/locked-candidate \
  --final-records private/final-test-records.json \
  --legacy-artifacts private/legacy-artifacts \
  --slicing-configuration private/slicing-configuration.json \
  --output-root private/end-to-end/reassessment-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --bootstrap-seed 1729
```

Issue #11 can supply at most one of the three required untouched final source
groups. Actual promotion remains blocked until at least two additional qualifying
untouched groups are supplied, with at least 200 accepted records in every group.
This blocker does not invalidate completed tuning evidence and must not be resolved
by weakening the final-evidence requirement.

Automated screening is only a mechanical safety gate. A person must review and
explicitly publish an approved aggregate separately. Even a promoted result is
limited unseen-source evidence for internal advisory use with human review of every
estimate. It does not establish population-wide performance, a prediction interval,
actual shop consumption, pricing accuracy, or an operational allowance.

## Evaluate one declared candidate

Use `evaluate_declared_candidate` as the high-level tracer before orchestration. Both
supported model families use the same rotating anonymous-source holdout interface.
The declared contract is complete and serializable, and its `candidate_id` is derived
from the configuration content rather than supplied by the caller.

```python
from minires import EvaluationConfig
from minires.modeling.tuning import (
    DeclaredCandidate,
    TensorflowXGBoostCandidateRuntime,
    evaluate_declared_candidate,
)

candidate = DeclaredCandidate(
    family="xgboost",
    parameters={
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.9,
        "min_child_weight": 3.0,
        "gamma": 0.05,
        "reg_alpha": 0.001,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "n_jobs": 1,
        "early_stopping_rounds": 50,
    },
)
result = evaluate_declared_candidate(
    records,
    EvaluationConfig(None, "mm3", True, seed=17),
    candidate=candidate,
    runtime=TensorflowXGBoostCandidateRuntime(),
    seed=41,
    output_root="private/declared-candidates/xgb-001",
)
```

The output directory is create-only. Its private report includes row-level
predictions, partition audits, fitted-state and held-out-data fingerprints, fit
metadata, aggregate diagnostics, dependency versions, and process resource use.
Each fold's fitted artifact and every report file is SHA-256 checksummed in
`manifest.json`. Invalid contracts, partition failures, missing runtimes,
non-finite predictions, model failures, and artifact failures return bounded blocker
codes without exposing raw third-party errors. Neural-network normalization and both
families' early stopping receive only fold-training and fold-validation records,
respectively.

## Generate the search plan

Generate and review the complete plan before starting expensive fitting. The
plan-generation interface normalizes the development input only to derive stable
input and source-allocation identities; it does not fit, score, read prior candidate
results, or use targets to choose candidate parameters.

```python
from minires import EvaluationConfig
from minires.modeling.tuning import SearchLimits, create_search_plan

plan = create_search_plan(
    records,
    EvaluationConfig(None, "mm3", True, seed=17),
    limits=SearchLimits(seed=41),
    dependency_versions={"tensorflow": "pinned-version", "xgboost": "pinned-version"},
    output_root="private/candidate-plans/plan-001",
)
```

The output directory is private and create-only. `search-plan.json` has canonical,
repeatable JSON content and `manifest.json` records its SHA-256 checksum. The plan
identity binds the raw and normalized development input, source-allocation contract,
code and evaluation configuration, complete parameter domains, exact dependencies,
seed, and generator version. A changed bound identity produces a changed plan ID.
The plan also records all 12 ordered component candidates, three equal-rank
ensemble-construction rules, eligibility and ranking rules, the best-five second-seed
repetition, the 20-run and two-hour limits, and the immutable fixed learned control.
Private source names are represented only by one-way identities and are never written
to the plan.

Only supported subsets of the declared domains are accepted. Missing or empty
domains, unsupported choices, non-finite values, duplicate/impossible trial
allocations, unsafe worker settings, and invalid resource limits fail with
`invalid_search_plan` before any runtime can fit.

## Run the bounded candidate search

Choose a new output directory beneath `private/` for every attempt:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records private/issue-31/train.jsonl \
  --validation-records private/issue-31/validation.jsonl \
  --output-root private/candidate-tuning/run-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41
```

The workflow creates `search-plan.json` before fitting. The fixed-seed plan
contains six neural-network trials and six XGBoost trials and predeclares three
deterministic equal-rank ensemble rules. Every component fits once on the explicit
training artifact and is scored on the explicit validation artifact. Ensemble
pairing, convex-weight selection, candidate ranking, and the best-five second-seed
repetition use those validation predictions only. One complete explicit-partition
candidate evaluation is one run, including an ensemble whose components must be
refitted. The fixed learned configuration is evaluated as a control and does not
consume a candidate run. Historical source-holdout evaluation remains available
through `evaluate_declared_candidate`; it is not reinterpreted as evidence under
this mixed-source contract.

The baseline workflow never starts a candidate after 20 new runs or after 7,200
elapsed seconds. If fewer than five initial candidates are eligible, it repeats
every eligible candidate and records the finalist shortfall; unused capacity does
not expand the search. A partial repetition stage cannot select or lock a candidate.

### Run the predeclared tail-aware expanded plan

Use this plan only for the follow-up round justified by the private serious-error
audit. It retains the same data-separation, eligibility, ranking, and locking
rules while broadening deterministic configuration coverage:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records <training-artifact> \
  --validation-records <validation-artifact> \
  --output-root <new-private-run-directory> \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind tail_aware_expanded
```

The expanded plan contains 12 neural-network trials, 12 XGBoost trials, six
validation-selected ensembles, and second-seed repetition of at most ten eligible
initial candidates, with a hard maximum of 40 candidate runs and 7,200 seconds.
Both component families sample either unweighted training or sliced-resin-mass-
band weights of 1× below 10 g, 2× from 10–25 g, 3× from 25–50 g, and 4× at 50 g
or above. Weights are normalized to mean one. They are derived solely from
training sliced resin mass labels; validation metrics and eligibility gates remain
unweighted. A seeded mixed-radix traversal guarantees that each component-family
slot has a distinct complete
configuration. The plan records its tail-error hypothesis and plan kind before
fitting starts.
### Run the predeclared large-batch extended plan

Use this plan only for the issue #50 round predeclared after the tail-aware plan
failed its fixed eligibility gates:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records <training-artifact> \
  --validation-records <validation-artifact> \
  --output-root <new-private-run-directory> \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind large_batch_extended
```

The plan contains 24 neural-network trials, 24 XGBoost trials, 12
validation-selected ensembles, and second-seed repetition of at most 20 eligible
initial candidates. Its hard limits are 80 candidate runs and 14,400 seconds.
Neural candidates use the larger batch sizes 512 and 1,024, train for at most 150
or 200 epochs, and use early-stopping patience of 12 or 16 epochs. XGBoost
candidates use ceilings of 1,500, 1,800, or 2,400 trees. These exclusive new
ranges prevent repetition of complete component configurations from the preceding
expanded round. The target-weighting choices and every fixed development,
eligibility, ranking, and locking rule remain unchanged. See
[the issue #50 round ledger](issue-50-candidate-search.md) for the falsifiable
hypothesis, dependency versions, and execution status.

### Run the predeclared geometry-regime plan

Use this plan only for the issue #50 round predeclared after the large-batch plan
failed its fixed eligibility gates:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records <training-artifact> \
  --validation-records <validation-artifact> \
  --output-root <new-private-run-directory> \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind geometry_regime
```

The plan contains 12 neural-network trials, 12 XGBoost trials, six deterministic
validation-selected ensembles, and second-seed repetition of at most ten eligible
initial candidates. Its hard limits are 40 candidate runs and 7,200 seconds. It
retains the preceding large-batch parameter domains and target-weighting choices,
so the isolated intervention is the candidate feature representation.

The representation replaces the seven legacy-compatible candidate inputs with 16
source-neutral geometry features: raw mesh volume and surface area, sorted
bounding-box dimensions, bounding-box volume, Euler number, logarithmic size
features, mesh-volume and surface-area ratios, and a bounding-box aspect ratio.
Feature construction is deterministic, versioned as
`minires-geometry-regime-features-v1`, and uses only normalized canonical geometry.
The plan rejects any seed pair other than the predeclared 41/42. The fixed control
continues to use the legacy input order. Anonymous source groups,
miniature families, partition and linkage evidence, identities, file-size proxies,
legacy scale, and join keys remain outside every candidate feature matrix. Missing
or invalid required geometry, including inputs that are not finite float32 values,
blocks before fitting rather than triggering imputation or row removal. Size, area,
and volume inputs must be positive; Euler number may be negative but must be finite
and integral.

The ordered feature contract is bound into the search plan, every candidate model
specification and identity, and any resulting lock. All training/validation
separation, eligibility, ranking, locking, and held-out-test exclusions remain
unchanged. See [the issue #50 round ledger](issue-50-candidate-search.md) for the
complete predeclared hypothesis and stop rule.

### Run the predeclared cross-fitted geometry-gate plan

Use this plan only for the issue #50 round predeclared after the geometry-regime
plan failed its fixed eligibility gates:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records <training-artifact> \
  --validation-records <validation-artifact> \
  --output-root <new-private-run-directory> \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41 \
  --plan-kind cross_fitted_geometry_gate
```

The plan restores the baseline six neural-network and six XGBoost component
spaces and replaces the three constant convex ensembles with three constrained
geometry-conditioned gates. Equal-rank component pairs are selected by the
existing validation ranking. Each gate is fitted from five-fold out-of-fold
predictions made for every training row; validation labels never fit the gate.
Fold assignment is deterministic from the declared seed and stable private record
identity, without source or miniature-family metadata. The three fixed ridge
penalties are 0.01, 0.1, and 1.0. Gate weights are linear functions of the
versioned 16-feature source-neutral geometry representation and are clipped to
zero through one.

After gate fitting, its two base models are fitted on all training rows and the
unchanged validation seam supplies only permitted early stopping, scoring,
eligibility, ranking, ensemble decisions, and locking. A locked gate repeats
cross-fitting on training only with fixed training counts, refits both bases on
all training rows without validation, and checksum-binds both model artifacts and
the gate state. The plan uses seeds 41/42, 15 initial candidates, second-seed
repetition of at most five eligible candidates, at most 20 candidate runs, and
7,200 seconds. Every unchanged eligibility, ranking, create-only, and held-out-test
rule remains in force. See [the issue #50 round ledger](issue-50-candidate-search.md)
for the complete falsifiable hypothesis and stop rule.

### Run the predeclared legacy geometry-augmentation plan

Use this plan only for the issue #50 round predeclared after the cross-fitted
geometry gate failed its fixed eligibility gates:

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

The plan keeps the seven ordered legacy candidate inputs and appends four
source-neutral geometry terms in this exact order: mesh volume divided by
bounding-box volume, surface area divided by bounding-box volume, squared
`log1p` mesh volume, and longest divided by shortest bounding-box dimension.
The complete 11-feature contract is versioned as
`minires-legacy-geometry-augmentation-v1`. The fixed control remains on the
seven-feature legacy contract.

Legacy `kb` and `scale` are retained deliberately so augmentation is the only
intervention. Their historical meaning is not fully established and they may act
as source or style proxies; this is an explicit scientific limitation rather
than a claim that they are source-neutral. No source group, miniature family,
identity, linkage, partition, path, or held-out information is added.

The plan reuses the bounded baseline component domains: six neural-network and
six XGBoost candidates, three validation-selected constant convex ensembles,
and second-seed repetition of at most five eligible initial candidates. Seeds
41/42, at most 20 candidate runs, and 7,200 seconds are fixed. The target,
weighting, architecture families, eligibility gates, ranking, and development
boundary are unchanged. Missing, non-positive, non-finite, or non-float32
canonical geometry blocks before fitting. The ordered feature contract and
transformation version are bound into plan, candidate, model-specification, and
lock identities and are reconstructed from canonical geometry during assessment.

Immediately before execution, the command checksum-verifies only `train.jsonl`
and `validation.jsonl` against their adjacent `manifest.json`. It also verifies
the pinned Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0, TensorFlow
2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2. The test artifact is not read or
fingerprinted. See [the issue #50 round ledger](issue-50-candidate-search.md) for
the complete falsifiable hypothesis and stop rule.

### Run the predeclared tail-aligned selection plan

Use this plan only for the issue #50 round predeclared after geometry augmentation
failed the fixed eligibility gates:

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

The plan retains the seven legacy candidate inputs and adds no prediction feature,
source metadata, or resliced evidence. It contains six Huber neural-network
candidates, six XGBoost candidates, three equal-rank convex ensembles, and at most
five eligible second-seed repetitions, with seeds 41/42, at most 20 runs, and
7,200 seconds.

For each component, every bounded validation checkpoint is scored by the versioned
`serious_error_gates_then_ranking_v1` rule. It prefers eligible checkpoints, then
minimizes the maximum and sum of normalized excess above the unchanged pooled,
source-balanced, and qualifying-source serious-error limits. Existing ranking
metrics and the checkpoint number break ties. Anonymous source groups are held
inside an opaque validation prediction scorer and never enter TensorFlow or
XGBoost prediction matrices, training weights, or model construction. The selected
epoch/tree count is retained for training-only refit.

Component pairing and each predeclared ensemble-weight grid use the same gate-first
selection rule. Final eligibility and ranking remain unchanged. The command verifies
the committed development checksums and pinned environment before fitting, receives
no test argument, writes once to the next create-only directory, and stops without
expansion if no candidate qualifies. See
[the issue #50 round ledger](issue-50-candidate-search.md) for the complete
hypothesis and decision record.

Runtime failures, invalid plans, missing grouping evidence, deadline exhaustion,
and the absence of eligible candidates remain bounded outcomes in
`tuning-result.json`. The private `candidate_history` records the control, every
completed or failed launch, and every skipped slot with its bounded stop reason.
A stopped search preserves all completed results and cannot lock a candidate.

Eligibility groups explicit validation predictions by their anonymous source-group
metadata and applies the observed above-5-g error limits before ranking: no more
than 1% pooled, 1% under equal source weighting, and 2% for each source with at
least 200 accepted records. Smaller sources remain part of pooled and
source-balanced metrics, but do not receive the separate per-source gate. The
private tuning report retains per-source rows and metrics; public summaries omit
source reports and source identities.
Ineligible results remain in the private initial results and history but are
excluded from `initial_promotable_ranking`. Eligible candidates rank by
source-balanced mean absolute error, pooled mean absolute error, within-2-g
fraction, and stable candidate ID. The five highest-ranked eligible candidates,
or every eligible candidate when fewer than five qualify, receive the second
seed. First- and second-seed metrics receive equal weight, including unfavorable
results. A finalist must satisfy every fixed eligibility gate under each seed
individually and under the combined metrics before it can enter
`promotable_ranking`. The report records both seed eligibility outcomes, both
rankings, the finalist shortfall, and each ensemble's selected fold weights
explicitly.

A complete repetition stage selects the best combined eligible result by the
same deterministic rule. Epoch and tree counts are fixed from the validation
outcomes across both seeds. The chosen component or resolved ensemble is refitted
on training rows only, without validation data or final-test early stopping. A
resolved ensemble keeps its component contracts and fixes its convex weight from
the same two-seed validation evidence. If no initial candidate qualifies, the run
records the best initial development result, performs no repetitions or refit, and ends as
`completed_no_candidate` without expanding the budget.

### Predeclared nonlinear out-of-fold stacking plan

Issue #50's next bounded round is `nonlinear_oof_stacking`. It retains the
ordered seven-feature legacy input contract and performs no reslicing and adds no
raw prediction features. The four frozen base contracts are, in order:

1. neural network: layers 512/256/128/64, ReLU, dropout 0, AdamW, Huber,
   learning rate 0.001, L2 0.000001, batch 32, fixed 61 epochs;
2. neural network: layers 256/128/64, Mish, dropout 0.1, Adam, Huber,
   learning rate 0.003, L2 0.00001, batch 256, fixed 87 epochs;
3. XGBoost: 584 trees, depth 9, learning rate 0.01, subsample 0.9,
   column sample 1.0, minimum child weight 1, gamma 0.05, alpha 0.00001,
   lambda 10; and
4. XGBoost: 1,091 trees, depth 9, learning rate 0.05, subsample 0.9,
   column sample 0.9, minimum child weight 10, gamma 0.2, alpha 0.1,
   lambda 10.

Both tree bases use `reg:squarederror`, `hist`, and one worker. The complete
contracts, including the otherwise unused early-stopping settings inherited from
the existing candidates, are content-identified in the generated plan. During
this round every base fit uses its fixed count with no early stopping.

For each seed 41 and 42, training identities are sorted by SHA-256 of
`seed:record_identity` and assigned round-robin to five folds. Identity is used
only for assignment. Each base fits five times on four folds and predicts the
held fold, so every training row must receive exactly one finite OOF prediction
per base. Source group, target, miniature family, linkage, partition, path, and
identity are unavailable to model matrices. The combiner matrix is exactly the
four ordered base predictions followed by their float64 mean, minimum, maximum,
and spread (`maximum - minimum`), cast to float32. Wrong-length, non-finite, or
out-of-float32 predictions block the round.

Exactly three shallow XGBoost combiners are declared: (1) 64 trees/depth 1/rate
0.03/subsample 0.80/column sample 1.00/child weight 20/gamma 0/alpha 0/lambda 10;
(2) 96 trees/depth 2/rate 0.03/subsample 0.80/column sample 0.80/child weight
20/gamma 0.05/alpha 0.1/lambda 10; and (3) 64 trees/depth 3/rate
0.02/subsample 0.75/column sample 0.80/child weight 30/gamma 0.1/alpha 1/lambda
20. All use `reg:squarederror`, `hist`, one worker, their evaluation seed, fixed
tree counts, and no early stopping. They fit only training targets against OOF
predictions. Four bases are then fit once on all training rows and produce
validation inputs; validation labels only score eligibility and ranking.

The finite allocation is three candidates under each of two seeds: six candidate
evaluations, 40 OOF base fits, eight full-training base fits, and six combiner
fits (54 model fits total), with a 7,200-second ceiling checked before every fit.
No budget is recycled. Existing serious-error eligibility gates and ranking are
unchanged. All three candidates run under both seeds, unfavorable evidence is
retained, and eligibility is required under seed 41, seed 42, and their equal
combination. Incomplete execution, any invalid vector, or no qualifying candidate
stops without substitution, extra seeds, changed folds, tuning, or expansion.

The seed-42 training-only fitted state is the predeclared refit used for a selected
lock; validation never fits it. The create-only lock checksum-binds all four base
contracts/artifacts and preprocessing states, the combiner contract/artifact and
state, fixed counts, base/meta ordering, fold algorithm/version/count, seeds,
training and validation identity fingerprints, dependencies, code and feature
contracts, gates/ranking, and no-test-access attestation. Loading verifies every
checksum and reconstructs the exact ordering before prediction.

The predeclaration change did not run the command. The one authorized private
round subsequently used the fresh `run-012` directory; its source-neutral outcome
is recorded in [the issue #50 round ledger](issue-50-candidate-search.md).

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-012 \
  --volume-unit mm3 --scope-confirmed --seed 41 \
  --plan-kind nonlinear_oof_stacking
```

## Predeclared guarded residual-stacking plan

Issue #50's next bounded round is `guarded_residual_stacking`. The preceding
nonlinear stack replaced already-strong base predictions with piecewise-constant
direct target estimates and compressed the prediction range. This round tests a
materially distinct hypothesis: a small, bounded additive residual correction,
fitted only from training out-of-fold predictions, can improve a fixed base anchor
without being able to replace or collapse it.

The four base contracts, their order, fixed epoch/tree counts, five-fold
identity-hash assignment, and seeds 41/42 are exactly those declared for
`nonlinear_oof_stacking`. Identity affects fold assignment only. Anonymous source
groups, miniature families, targets, linkage evidence, paths, partitions, and
identities do not affect fold assignment and never enter a prediction matrix.
Every training row must receive exactly one finite OOF prediction from every base.

For each row, the fixed anchor is the float64 arithmetic mean of the four ordered
base predictions, cast to finite float32. The residual feature vector is exactly:

1. the anchor;
2. each of the four base predictions minus the anchor, in base order; and
3. the maximum base prediction minus the minimum base prediction.

Training-OOF means and population standard deviations standardize these six
features. A standard deviation no greater than `1e-12` becomes `1.0`; standardized
values are clipped to `[-4, 4]`. One deterministic NumPy least-squares ridge model
per seed fits the training residual target `target - OOF anchor`, clipped to
`[-2 g, 2 g]`, with penalty 1.0 and an unpenalized intercept. Validation labels do
not fit the anchor, preprocessing, coefficients, bounds, or scales.

Exactly three candidates reuse each seed's one fitted residual state:

| Slot | Scale | Maximum additive correction | Identity behavior |
| --- | ---: | ---: | --- |
| 1 | 0.0 | 0 g | returns the anchor directly |
| 2 | 0.5 | 1 g | anchor plus half the clipped residual |
| 3 | 1.0 | 2 g | anchor plus the clipped residual |

Wrong-length, non-finite, or out-of-float32 base predictions, features, state,
corrections, or final predictions block the round. There is no fallback, omitted
row, substitute candidate, added seed, changed fold, recycled budget, or automatic
expansion.

The finite allocation is three candidate evaluations under each seed, 40 OOF base
fits, eight full-training base fits, and two analytical residual fits: six
candidate evaluations and 50 fits total, with a 7,200-second ceiling checked before
every fit. All three candidates run under both seeds, including unfavorable ones.
Eligibility remains at most 1% above-5-g errors pooled, 1% source-balanced, and 2%
for every qualifying anonymous source group. Ranking remains source-balanced MAE,
pooled MAE, within-2-g fraction, and stable candidate identity. A lock requires
eligibility under seed 41, seed 42, and their equal combination.

Private aggregate `oof-full-fit-shift.json` evidence compares each base and the
anchor under training OOF and full-training fitting, records validation anchor and
bounded-correction distributions without validation labels, and contains no row,
source, family, or identity values. Its fingerprint is bound into any lock. The
create-only lock also binds the analytical residual state, exact anchor and feature
contracts, bound and scale, base order and artifacts, fixed counts, recomputable
fold assignments and fingerprints, seeds, dependencies, development identities,
unchanged gates/ranking, and the no-test-access attestation. The top-level run
manifest explicitly records create-only output and checksums the complete lock
tree.

The implementation and predeclaration do not authorize fitting by themselves. The
one permitted execution requires the committed change and a source-neutral Issue
#50 predeclaration comment, then uses fresh directory `run-013`:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-013 \
  --volume-unit mm3 --scope-confirmed --seed 41 \
  --plan-kind guarded_residual_stacking
```

The command verifies only the committed training and validation checksums and the
pinned Python 3.13 environment: NumPy 2.2.6, Keras 3.15.0, TensorFlow 2.20.0,
scikit-learn 1.7.2, and XGBoost 3.1.2. It has no held-out assessment argument and
must not read or fingerprint held-out evidence.

### Guarded residual-stacking result

The round ended as `completed_no_candidate`. The one permitted execution completed
all three candidates under both seeds and all 50 predeclared fits. No candidate was
eligible under either seed or their equal combination. The best development result
was the full ±2 g correction under seed 41:

| Metric | Result |
| --- | ---: |
| Pooled mean absolute error | 0.5930 g |
| Source-balanced mean absolute error | 0.6587 g |
| Pooled within-2-g fraction | 95.68% |
| Source-balanced within-2-g fraction | 94.74% |
| Pooled above-5-g fraction | 1.22% |
| Source-balanced above-5-g fraction | 1.55% |
| Maximum qualifying-source above-5-g fraction | 4.15% |

The round used 349.54 elapsed seconds, 1,212.51 process CPU seconds, and a process
high-water resident-set measurement of 1,224,359,936 platform units. The
create-only manifest's complete five-artifact inventory was independently
checksum-verified. No lock was created, and held-out assessment was not started.

The bounded correction preserved aggregate accuracy near the strong anchor but did
not satisfy any unchanged serious-error gate. This hypothesis is falsified for the
fixed plan and stops without changed bounds, replacement candidates, added seeds,
or automatic expansion.

## Predeclared tail-focused closest-anchor correction

The `tail_focused_correction` plan tests a bounded additive correction to the exact
closest run-011 ensemble, not the weaker four-base mean used in run-013. It uses
one identity control and one corrected candidate. The loss and feature constants
below are predeclared engineering choices, not validation-tuned, training-OOF-
selected, or empirically optimal settings. This implementation does not authorize
execution or claim improvement or eligibility.

### Frozen anchor

The original ordered inputs remain `kb`, `volume`, `surface_area`, `bbox_area`,
`euler_number`, `scale`, and `surface_volume_ratio`, using
`minires-normalization-v3` and finite float32 matrices. The two original run-011
contracts are frozen separately from their selected training counts:

| Setting | Neural network | XGBoost |
| --- | --- | --- |
| Architecture | 256/128/64, Mish, dropout 0.1 | depth 9 |
| Optimization | Adam, Huber, learning rate 0.003, L2 0.00001 | squared-error objective, learning rate 0.05 |
| Sampling | batch 256, training shuffle | subsample 0.9, column sample 0.9 |
| Regularization | as above | minimum child weight 10, gamma 0.2, alpha 0.1, lambda 10 |
| Original ceiling | 100 epochs | 1,200 trees |
| Original early-stopping setting | patience 8 | 50 rounds |
| Fixed training count in **every** new fit | **87 epochs** | **1,091 trees** |
| Preprocessing | normalization fitted on that fit's training rows | no normalization |

The original gate-first `serious_error_gates_then_ranking_v1` selection setting is
retained in the historical contracts but never invoked in this round. No inner,
outer, full-training, or validation partition supplies early stopping or checkpoint
selection. The NN retains a linear one-unit output; XGBoost retains `hist`, one
worker, and its declared evaluation seed. Plan identities bind both the original
contracts and the effective fixed-count model specifications and runtime settings.

For each prediction, preserve the original float64 operation order exactly:
`anchor = 0.8 * neural + (1.0 - 0.8) * xgboost`. Do not cast the anchor to float32,
replace it with a mean, or quantize it. Retained `kb` and `scale` have uncertain
historical semantics and may be source/style proxies; this round does not establish
that they are source-neutral. No new raw geometry, source, family, identity,
linkage, partition, path, or join-key prediction inputs are added.

### Fixed numerical correction

The ordered correction inputs are the anchor, `neural - xgboost`, and the absolute
value of that disagreement. Training-OOF means and population standard deviations
standardize these three inputs. A standard deviation at or below `1e-12` becomes
1.0. Standardized values are clipped to `[-1, 1]`; prepend a constant intercept to
form the four-column design matrix `D`.

Let `c = D * coefficients`, `e = anchor + c - target`, and `n` be the number of
correction-training rows. Minimize this fixed convex objective:

```text
mean(0.1 * e² + 4 * max(abs(e) - 4.5 g, 0)² + c²)
    + 0.01 * sum(coefficients²)
subject to sum(abs(coefficients)) <= 2 g
```

The target residual and error are **not clipped**: severe errors retain their
excess-loss gradients. The margin is deliberately below the unchanged 5 g serious-
error threshold. The ordinary squared-error term protects ordinary accuracy and
the departure term penalizes changing the anchor. The intercept is included in
both coefficient penalties; this is not an intercept-only calibration model.

Use exactly 2,000 deterministic projected-gradient steps from zero coefficients,
with constant step `1 / (2 * 5.1 * sum(D²) / n + 0.02)`. Each update is projected
onto the L1 ball by Euclidean soft thresholding; a roundoff normalization enforces
the exact coefficient bound. No convergence-based extension, optimizer search,
feature selection, changed margin, or bound sweep is permitted. The L1 constraint
and bounded inputs guarantee a correction of at most ±2 g globally. A final
roundoff clip to ±2 g is applied at prediction. The identity variant returns the
anchor directly; the corrected variant adds the full bounded correction. Neither
can replace the continuous anchor. In particular, this bound cannot by itself
bring an anchor error greater than 7 g inside 5 g.

Wrong-length, non-finite, or out-of-float32 base predictions, inputs, state, or
outputs block rather than dropping rows, imputing, or substituting predictions.
The numerical contract is versioned `minires-tail-focused-correction-v1`.

### Honest training evidence comes before validation scoring

For each seed 41 and 42, sort training identities by SHA-256 of
`seed:record_identity`, breaking ties by identity, and assign ranks round-robin to
five folds. Rank modulo five equal to zero is the outer 20% holdout. Only identity
and seed determine assignment; source groups, miniature families, labels, and
features do not.

Within the remaining 80%, independently apply the same five-fold assignment. Each
base fits on four inner folds and predicts the fifth using its frozen count and
training-only preprocessing. Fit the correction on these inner OOF predictions
and training labels only. Fit both bases on the 80% partition and predict the
outer holdout. No outer row, target, or preprocessing statistic influences **any**
of these base or correction fits. These are independent correction evaluation
predictions, not scores on the correction's own OOF meta-training rows.

Complete this honest stage under **both** seeds before any production-stage fit
or validation prediction. For each seed, the corrected outer-holdout predictions
must have strictly lower `mean(0.1 * e² + 4 * max(abs(e) - 4.5, 0)²)`, no higher
pooled MAE, and no higher count of errors above 5 g than the identity control.
All values must be finite. If either seed fails, the whole round ends as
`training_evidence_rejected` with `honest_training_correction_gate_failed`, retains
both seeds' aggregate evidence, and performs **no production fit, validation
scoring, or lock**. Unused budget is not recycled, even to score the identity.

This honest evidence is conditional on the historical choice of pair and counts,
which used reused validation evidence. It is not an untouched evaluation of the
entire historically selected pipeline. Outer labels only apply this fixed gate;
they do not select features, coefficients, loss constants, or training counts.

### Conditional production stage, budget, and lock

Only if both honest seeds qualify, independently repeat five-fold OOF correction
training on **all** training rows, then fit both bases on all training rows. Score
identity and corrected predictions on validation under **both** seeds, retaining
unfavorable results. Validation labels fit nothing: they only apply the unchanged
eligibility gates and ranking. The gates remain at most 1% pooled above-5-g errors,
1% source-balanced, and 2% for each anonymous source group with at least 200
accepted records. Groups remain evaluation-only. Ranking remains source-balanced
MAE, pooled MAE, within-2-g fraction, and stable candidate identity. Eligibility
is required for each seed separately and their equal-weight metric combination.

Each seed/stage uses ten OOF base fits, two partition-wide base fits, and one
numerical correction fit. The ceiling is 26 honest-stage fits plus 26 conditional
production-stage fits: **52 fits**, at most **four validation candidate evaluations**,
and **7,200 seconds**. The deadline is checked before and after fits, before
prediction/scoring, and before completion or locking, including unsuccessful
searches. An expired final correction fit cannot score further validation variants.
The historical source-holdout `tune_candidates` interface rejects this plan; use
`develop_candidates` with explicit training and validation artifacts or the CLI.

Aggregate private `honest-training-evidence.json` retains paired outer metrics,
prediction distributions, and honest-stage OOF/full-fit shift. The production
`oof-full-fit-shift.json` records per-base, anchor, and corrected-prediction shifts,
anchor distributions, and validation correction distributions without validation
labels. These files contain no row, source, family, or identity values. Private
lock split evidence separately records recomputable identity-only assignments.

A selected lock serializes the **exact eligible seed-42 production state already
scored**, not an average of the two seeds' predictions and not an extra refit.
Equal weighting combines development metrics only. This follows the existing
prefitted-stack lock semantics. No validation rows enter this training-only state.
The create-only lock binds the full plan, exact pair/counts/runtime contracts,
numerical optimizer/state, feature order/version, honest gate and both seeds'
evidence, recomputable outer/inner/full-training assignments, OOF/full-fit shift,
dependencies, code/development identities, unchanged gates/ranking, complete base
artifact inventory and preprocessing, and no-test-access attestation. Loading
verifies checksums and reconstructs fixed contracts and predictions through the
existing model backend seam; modified contracts or incomplete evidence block.
The public lock specification is a typed `TailCorrectionModelSpecification` with
reproducible identity, not a separate dictionary-only representation.

Every tail lock, including one made through a custom backend, requires neural
preprocessing with exactly `mean` and `variance` arrays of length seven. Values
must be finite and float32-representable, and variances must be nonnegative.
XGBoost preprocessing must be exactly `{"xgboost": "unnormalized_float32"}`.
The schema is bound as `normalization-vectors-and-unnormalized-xgboost-v1`.
Production loading also compares neural normalization and input width with the
serialized model and checks the serialized XGBoost feature count. Empty mappings,
missing or invalid statistics, and inconsistent serialized preprocessing block.

Any invalid vector/state, failed fit, incomplete stage, expired budget, failed
honest gate, or absence of an eligible validation candidate ends this round
without replacement candidates, additional seeds, changed folds, changed bounds,
omitted rows, or automatic expansion. Completed evidence and skipped slots remain
preserved. No held-out assessment is part of this plan.

Execution requires separate authorization after the implementation/predeclaration
commit and source-neutral Issue #50 predeclaration. The intended next fresh private
directory is `run-014`; this change does not create it:

```bash
.venv-candidates/bin/python -m minires.modeling.tuning \
  --training-records data/train.jsonl \
  --validation-records data/validation.jsonl \
  --output-root private/candidate-tuning/run-014 \
  --volume-unit mm3 --scope-confirmed --seed 41 \
  --plan-kind tail_focused_correction
```

The command verifies the frozen committed training/validation checksums and pinned
Python 3.13 environment with NumPy 2.2.6, Keras 3.15.0, TensorFlow 2.20.0,
scikit-learn 1.7.2, and XGBoost 3.1.2. It has no held-out argument and must not read
or fingerprint held-out evidence. Existing runs and dataset artifacts remain
immutable.

### Completed outcome

The separately authorized execution ended as `training_evidence_rejected` after
26 honest-stage fits. Seed 41 failed the fixed training-only qualification; seed
42 passed. The round therefore performed no production fits or validation scoring
and created no lock. The complete six-artifact manifest was independently
checksum-verified. See the [aggregate outcome in the Issue #50 ledger](issue-50-candidate-search.md#tail-focused-closest-anchor-correction-result)
for paired training-holdout metrics and resource use. This round is closed to
further execution or expansion; the command above is its historical invocation.

## Predeclared crossed training-stability diagnostic

This separately bounded diagnostic does not search, select, lock, or assess a
candidate. It holds the frozen closest 80% neural / 20% XGBoost anchor and the
existing tail-focused correction algorithm constant while distinguishing two
sources of training-stage variation: model randomness and which training rows are
held out.

The plan fixes model seeds 41 and 42, outer split seeds 101 and 202, and inner
split seeds 1101 and 1202. Each outer split assigns training identities by the
existing five-fold SHA-256 round-robin rule and holds fold zero out. For every
outer split/model-seed cell, the correction is trained from five-fold OOF base
predictions on the remaining 80%, then evaluated only on its outer holdout. The
base fits use the fixed 87 epochs and 1,091 trees with no early stopping. Model
seeds affect only model initialization/sampling; split seeds affect only the
outer and nested fold assignments.

The four cells consume exactly 40 inner-OOF base fits, eight outer-partition base
fits, and four deterministic numerical-correction fits: 52 fits within 7,200
seconds. The private aggregate evidence records each cell's anchor and corrected
training-holdout metrics, plus model-seed contrasts on the same outer holdout and
split contrasts for each fixed model seed. Different-split contrasts are not
paired row comparisons and are descriptive, not causal evidence.

The public `run_training_stability_diagnostic` seam receives only training
records; it has no validation or held-out-test argument. Its create-only package
contains the fixed plan, aggregate evidence, and checksummed manifest, with no
row, identity, source-group, family, or prediction values. It has no
qualification, ranking, promotion, or lock path. The command verifies only the
committed `data/train.jsonl` checksum and the pinned environment before any fit:

```bash
.venv-candidates/bin/python -m minires.modeling.training_stability \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-015 \
  --volume-unit mm3 \
  --scope-confirmed
```

Implementation and predeclaration do not authorize execution. It may run once
only after this change is committed and a source-neutral Issue #50 predeclaration
comment is posted. It must stop after one completed or blocked attempt; no metric
may select an intervention, expand its budget, or expose validation or held-out
records.

### Completed outcome

The separately authorized execution completed all four training-only cells and all
52 predeclared fits in 150.97 elapsed seconds and 411.65 process CPU seconds. Its
create-only two-artifact manifest was checksum-verified. Validation labels and
held-out evidence were not accessed.

| Outer split / model seed | Anchor MAE | Corrected MAE | Anchor above-5-g count | Corrected above-5-g count | Anchor loss | Corrected loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 / 41 | 0.8953 g | 0.9646 g | 34 | 37 | 511.7415 | 509.1025 |
| 101 / 42 | 0.8838 g | 0.9672 g | 34 | 40 | 474.2077 | 472.0289 |
| 202 / 41 | 0.8988 g | 0.8503 g | 35 | 37 | 444.7951 | 440.9340 |
| 202 / 42 | 0.9501 g | 0.8396 g | 42 | 37 | 405.7071 | 405.5510 |

The correction lowered the declared loss in all four cells, but its MAE and
above-5-g effects reversed by split: it worsened both MAE and above-5-g count for
both split-101 model seeds, while it improved MAE for both split-202 seeds and
reduced the above-5-g count only for model seed 42. The crossed evidence therefore
shows instability under both factors without isolating a cause. The split-to-split
corrected-MAE differences (about 0.11–0.13 g) were larger than the within-split
model-seed differences (about 0.01 g), but the split comparisons use different
held rows and remain descriptive rather than causal.

This is training-holdout evidence only. It does not evaluate validation eligibility
or ranking, select an intervention, create a lock, or authorize a correction
change, retry, expansion, or held-out assessment. The completed `run-015` directory
is preserved.

## Predeclared correction-transition diagnostic

Predeclaration status at commit `04af4e0`: **IMPLEMENTED NOT EXECUTED**.
The separately authorized execution has since completed; see the
[correction-transition result](issue-50-candidate-search.md#correction-transition-diagnostic-result).
This is new observational instrumentation,
not a candidate experiment or a retry/reinterpretation of `run-015`. That preserved
run contains aggregate metrics, not paired row predictions, so it cannot recover
repair/harm transitions retrospectively. Its loss improved in all four cells while
serious-error count worsened in three; both split-101 MAEs worsened. These are
observations to account for, not evidence of a software performance bug or a cause.

`minires.modeling.correction_transition` freezes exactly the preceding diagnostic's
closest anchor, original base contracts, effective runtime/preprocessing contracts,
87 epochs, 1,091 trees, and unchanged `minires-tail-focused-correction-v1` algorithm.
It uses the same four cells: outer splits 101/202, respective inner splits
1101/1202, and model seeds 41/42. Five-fold identity-only assignment holds outer
fold zero out, fits the correction using only the remaining rows' inner-OOF
predictions, and evaluates it only on the outer holdout. There is no early stopping.
The allocation remains 40 inner-OOF base fits, eight outer-partition base fits,
and four numerical corrections: **52 fits maximum, 7,200 seconds maximum**. Every
cell is retained regardless of favorable or unfavorable metrics. No capacity is
recycled, and no row, seed, fold, count, bound, loss, feature, or correction changes.

### Fixed accounting and interpretation before execution

The versioned `minires-correction-transition-diagnostic-v1` plan declares these
rules before any fit. Residual means **prediction minus target**: negative is an
underestimate, positive an overestimate, and zero exact agreement.

- A serious error is strictly `abs(residual) > 5 g`; exactly +5 or -5 is not
  serious. The paired 2×2 table is stable nonserious (neither serious), harm
  (anchor nonserious, corrected serious), repair (anchor serious, corrected
  nonserious), and persistent serious (both serious).
- Fixed anchor absolute-error bins are `[0,4]`, `(4,5]`, `(5,7]`, and `(7,infinity)`.
  Exactly ±4 belongs to the first, ±5 to the second, and ±7 to the third.
  Every transition and every bin retains its count, anchor/corrected residual-sign
  counts, above-5-g counts, and MAE and prediction-loss contributions. Bin tables
  also retain the four transition counts, so threshold harm is directly accounted
  for rather than inferred from marginal metrics.
- The unpenalized prediction loss is `0.1*e² + 4*max(abs(e)-4.5,0)²`; report the
  ordinary and weighted excess-squared components separately. Correction-departure
  and coefficient penalties are excluded. Contributions divide subgroup sums by
  the **entire cell's row count**, not subgroup counts; corrected-minus-anchor
  contributions add to the cell-wide change. Empty groups have zero contribution.
  Counts conserve exactly; floating conservation uses relative and absolute
  tolerance `1e-12`. Serious-count change must equal harms minus repairs.
- Correction summaries contain signed/absolute sums, maximum absolute magnitude,
  negative/zero/positive counts, and toward/away/neutral direction counts. Toward
  means opposite signs for anchor residual and correction; away means matching
  nonzero signs; neutral means either is zero. Separately count absolute-error
  improvement, worsening, and equality, since moving toward the target can overshoot.
  Departure is computed from the actual paired predictions; floating addition can
  differ from the algorithm's ±2 bound by a few ulps, checked with four ulps of
  the anchor/corrected values. The unchanged numerical state still enforces ±2.
- The **label-aware theoretical oracle**, never a deployable estimator, has minimum
  possible absolute error `max(anchor_absolute_error-2,0)`. Report its minimum MAE
  and unpenalized-loss contributions, theoretically repairable serious count
  `(5,7]`, and unavoidable serious count `>7`. Exactly ±7 can reach ±5; errors
  strictly beyond ±7 cannot become nonserious under an ideal ±2 correction.
- Three per-cell descriptive flags have fixed decision rules: (1) near-threshold
  harm offsets repairs only when harms are nonzero, harms are at least repairs,
  and **more than half** of harms began in `(4,5]`; (2) severe tails dominate the
  anchor excess-squared loss only when the `>7` contribution is **strictly more
  than half** its total (zero total does not qualify); (3) the `>7` oracle serious
  count must equal the `>7` count, a theorem even when the bin is empty. These
  definitions are accounting conventions, not learned/tuned gates or causal tests.

The existing per-base, anchor, and corrected OOF/full-fit shift and anchor
prediction distributions are retained with their complete finite aggregate schema.
Wrong shape, nonfinite values, arithmetic overflow, invalid correction departure,
incomplete evidence or fit accounting, deadlines, and fit limits block completion.
Completed cells and failed/uncompleted cell accounting are preserved with exact
started fits, fits in completed cells, unused capacity, and elapsed/CPU resources.
The deadline is checked around fits, predictions, summaries, and completion;
it cannot forcibly interrupt a running backend fit. Artifact-write failures cannot
claim a complete checksummed package.

The training-only public seam and CLI have **no validation/test input**, candidate
selection, qualification, ranking, promotion, or lock path. The CLI checksum-verifies
only committed `data/train.jsonl` and checks Python 3.13, NumPy 2.2.6, Keras 3.15.0,
TensorFlow 2.20.0, scikit-learn 1.7.2, and XGBoost 3.1.2 before fitting. The fixed
plan binds these required versions, observed runtime/environment, training-only
input identities, code identity, exact model contracts and accounting rules. The
create-only manifest checksums plan and evidence. Evidence contains only
source-neutral aggregates—no row vectors, IDs, paths, source/family values, or
source mappings. Plan input/code checksums are private provenance, not published
results. The in-memory runtime seam remains usable for synthetic tests; deployment
checksum/environment enforcement is at the CLI boundary, as for `training_stability`.

A separately authorized one-time execution, after committed predeclaration and a
source-neutral Issue #50 predeclaration, would use the next fresh create-only
`run-016` directory; this implementation does not create or inspect it:

```bash
.venv-candidates/bin/python -m minires.modeling.correction_transition \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-016 \
  --volume-unit mm3 \
  --scope-confirmed
```

Evidence remains conditional on the historically validation-selected anchor and
counts, not untouched pipeline evidence. Pairing is within a cell; different
training splits contain different held rows and comparisons are descriptive, not
causal. No conclusion or targeted correction change is justified until executed
evidence supports it and a separate decision authorizes it. Stop after one completed
or blocked attempt: no automatic expansion, learned/tuned gate, correction change,
validation access, candidate promotion, row exclusion, or capacity recycling.
Existing `run-015` and all prior contracts and outcomes remain immutable.

## Predeclared bounded-influence training prerequisite

Predeclaration status at commit `88f81ec`: **IMPLEMENTED NOT EXECUTED**.
The separately authorized execution has since completed and failed the all-four
training prerequisite; see the [recorded result](issue-50-candidate-search.md#bounded-influence-training-prerequisite-result).
This is a new training-only prerequisite experiment, not a production candidate
round or authorization for additional fitting.
All existing runs, numerical contracts, and loaders remain unchanged.

### Rationale and materially distinct objective

The completed run-016 paired diagnostic found repairs/harms of 4/7, 3/9, 2/4,
and 5/0 in cells 101/41, 101/42, 202/41, and 202/42 respectively. Its old loss
fell in every cell even though strict above-5-g count worsened in three. Errors
beyond 7 g contributed more than 99.98% of anchor excess-squared loss; a ±2 g
correction cannot make such errors nonserious. This supports testing bounded
influence, not choosing a favorable seed, identifying defective labels, or claiming
that a bounded-influence correction will help.

The new `minires-bounded-influence-correction-v1` contract freezes the same exact
float64 operation `0.8 * neural + (1.0 - 0.8) * xgboost`. The run-011 closest
Mish 256/128/64 NN (Adam/Huber, batch 256) and depth-9 XGBoost (learning rate
0.05) retain their complete original base contracts, legacy seven-feature order,
and validated component preprocessing. All new base fits use exactly **87 epochs
and 1,091 trees**, without early stopping. Retained legacy `kb` and `scale` still
have uncertain semantics and may be source/style proxies; no new raw or source
feature is added. Source/family/identity/linkage/partition/path metadata never
enters prediction or fitting matrices.

The correction retains only three ordered prediction inputs: anchor, signed
NN-minus-XGBoost disagreement, and absolute disagreement. Means and population
standard deviations come only from training OOF predictions. A standard deviation
at or below `1e-12` becomes 1; standardized inputs clip to ±1, and an intercept
forms the four-column design `D`. With `c = D*b` and `e = anchor+c-target`, fit:

```text
H_d(z) = z²                         if |z| <= d
         2*d*|z| - d²              otherwise

mean[0.1*H_5(e) + 4*H_2.5(max(|e|-4.5,0)) + c²] + 0.01*sum(b²)
subject to sum(|b|) <= 2 g

prediction-loss derivative = 0.2*clip(e,-5,5)
                           + 8*sign(e)*min(max(|e|-4.5,0),2.5)
```

**Both** ordinary and excess derivatives are bounded; their respective magnitudes
are at most 1 and 20, so the total is at most 21. No unbounded ordinary squared
residual remains. The ordinary Huber transition is tied to the 5 g serious
boundary; the 4.5 g margin anticipates that boundary by 0.5 g, and its 2.5 g Huber
width saturates at absolute error 7 g, the limit of possible repair by ±2 g.
These and the unchanged weights/penalties are engineering constants, not optimized
or empirically supported optima. Targets, residuals, and rows are not clipped,
dropped, imputed, or weighted. Huberization changes influence, not observations.

Use exactly 2,000 zero-initialized Euclidean L1-projected-gradient steps with
`step = 1/(2*5.1*sum(D²)/n + 0.02)`. Prediction curvature is at most `0.2+8`;
the departure penalty adds 2. Thus the objective gradient's Lipschitz constant is
at most `10.2*||D||_F²/n+0.02` including ridge, across the continuously
differentiable piecewise boundaries. The unchanged conservative step is valid.
The intercept participates in both penalties and the L1 bound. Bounded design
entries and `||b||_1 <= 2` guarantee global ±2 g departure, with a final roundoff
clip. Identity returns the exact anchor directly. This changes neither anchor
capacity nor the theorem that errors strictly beyond 7 g cannot be repaired.

The new typed state has its own version and exact contract validation; old loaders
reject it and the new numeric seam rejects old state. Invalid shapes, nonfinite
values, float32 range violations, overflowing preprocessing, or contract/state
tampering fail closed even for identity. There is no registration as a production
candidate or new lock loader. A dedicated small stage reuses fixed base primitives
but fits exactly one **new** correction per cell; it never fits/discards an old
correction or monkeypatches the old stage.

### Frozen four-cell qualification and finite stop rule

Use the same four cells as run-015/016: outer split seeds **101/202**, respective
inner split seeds **1101/1202**, and model seeds **41/42**. Stable identity-SHA-256
round-robin assigns five folds; outer fold zero is the 20% training holdout. Each
cell fits bases and preprocessing only on the remaining 80%; five nested OOF
folds give exactly one excluded-fold prediction per row/base for correction fitting.
Model seeds affect initialization/sampling only, never partition assignment.
Outer targets score the prerequisite only; they fit no base, preprocessing, or
correction. No full-training production stage exists.

For **each** cell, compare corrected to the exact identity anchor on identical
outer-held training rows. All three conditions must hold:

1. strictly lower **new unpenalized prediction loss**
   `mean[0.1*H_5(e)+4*H_2.5(max(|e|-4.5,0))]`;
2. pooled MAE no greater; and
3. strict `abs(e)>5 g` count no greater (exactly ±5 is nonserious).

Every one of all four cells must pass. Complete all four after any metric failure;
report every cell's conditions and the full failed-cell list, with no seed selection
or averaging. All-four success means `training_prerequisite_supported` **only**:
not validation eligible, no production fitting, validation scoring, ranking,
selection, lock, promotion, or automatic continuation. Metric failure means
`training_evidence_rejected`. Invalid data/state/evidence, runtime failure,
incomplete accounting, or deadline means `blocked`, retaining completed cells,
failed/uncompleted cell accounting, and started fits without replacement.

The exact ceiling is **52 fits and 7,200 seconds**: 40 inner-OOF base fits, eight
outer-partition base fits, and four correction fits. No budget recycling, repeated
attempt, alternate rows/folds/seeds, changed constants, or automatic expansion is
permitted. Deadline checks surround fits, predictions, summaries, and completion;
an in-progress backend fit cannot be forcibly interrupted. Artifact-write failure
cannot claim a complete checksummed package.

### Evidence, provenance, and prospective command

`paired_transitions_with_old_diagnostic_loss` preserves the complete run-016
2×2 repair/harm table, `[0,4]`, `(4,5]`, `(5,7]`, `>7` bins, signs, correction
magnitudes/directions, conservation checks, descriptive flags and fixed theoretical
label-aware ±2 oracle summaries. **Every loss in that subtree, including the
oracle and severe-tail flag, is the OLD squared diagnostic loss, never the new
qualification loss.** The oracle is not an estimator or qualification criterion.
`new_qualification_loss` separately decomposes the new ordinary/excess Huber loss
by total, transition, and anchor bin. Contributions divide by the whole cell count;
counts conserve exactly and floating contributions within relative/absolute
`1e-12`. Existing complete finite OOF/full-fit shift summaries are retained.

The create-only private package contains `prerequisite-plan.json`,
`bounded-influence-evidence.json`, and `manifest.json`. Before fitting, the plan
freezes the entire numeric/base/runtime/preprocessing contract, ordered features,
normalization, all seeds/folds, qualification, old/new accounting, budget, expected
training checksum, observed training-only input identities, code identity, required
and observed dependencies/platform, and historical-selection caveat. The manifest
SHA-256 binds plan and evidence. Evidence includes aggregate outcomes/resources
and explicitly false validation-eligibility/production-continuation attestations,
not source names, row vectors, IDs, mappings, or artifacts for production loading.

The training-only CLI accepts no validation/test, seed, budget, rank, lock, or
promotion input. It verifies only the existing pinned training checksum and pinned
Python **3.13**, NumPy **2.2.6**, Keras **3.15.0**, TensorFlow **2.20.0**,
scikit-learn **1.7.2**, and XGBoost **3.1.2** before fitting. The in-memory runtime
seam permits synthetic tests; checksum/environment enforcement remains at the CLI.
A separately authorized one-time execution, after committed predeclaration and a
source-neutral Issue #50 predeclaration, would use fresh create-only **run-017**:

```bash
.venv-candidates/bin/python -m minires.modeling.bounded_influence_training \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-017 \
  --volume-unit mm3 \
  --scope-confirmed
```

This implementation does not create/inspect that directory, fit private models,
read/fingerprint datasets or validation/test evidence, publish, or start Issue #33.
Training-holdout evidence remains conditional on the historically reused-validation
selection of the anchor and counts, not untouched pipeline evidence. Pairing is
within cell; different split comparisons are not paired or causal. There is no
empirical success claim. Stop after the single prospective attempt even if every
cell passes; any subsequent experiment requires a separate decision/predeclaration.

## Locked candidate

A successful search creates `locked-candidate/` with:

- the complete component or ensemble configuration and output unit;
- the two seeds, equal-weight combined evidence, deterministic ranking and
  eligibility rules, and fixed training counts;
- feature, preprocessing, dependency-environment, input, code, search-plan, and
  explicit training/validation artifact identities;
- an explicit usage record that fitting and preprocessing used training rows only,
  selection decisions used validation rows only, and no test argument or path was
  available to development;
- the fitted model artifact or artifacts;
- the fitted preprocessing state; and
- SHA-256 checksums in a create-only lock manifest.

The tuning and refit seam receives only normalized development features and
labels. Evaluation separately retains one-way anonymous source-group identifiers
for validation grouping and later source-overlap checks; those identifiers never enter
the feature matrix or public output. The seam has no final-test argument or
final-test path, and the contract records that final-test access did not occur.
Final-test records, source metadata, fingerprints, labels, and predictions enter
only through the later assessment command after lock verification.

Explicit-partition runs created before this correction collapsed validation into
one synthetic report and are not source-balanced evidence. Do not reinterpret or
publish them as such. Run development again into a fresh create-only output
directory; the recorded code fingerprint distinguishes the corrected run. Locks
also carry the `source-grouped-validation-v1` evidence contract. Lock verification
rejects affected earlier explicit-partition locks that lack this attestation.

Use `load_locked_candidate` from `minires.modeling.tuning` when assessment runs
in a later process. Loading and assessment fail closed if an artifact,
configuration, preprocessing record, dependency contract, or checksum changes.
The assessment boundary verifies the lock and reloads the predictor from those
verified artifacts; it never executes a caller-supplied in-memory predictor.
Final-test records are not loaded until lock verification succeeds. Assessment
also compares their one-way anonymous-source identities with the locked development
source inventory. Any overlap blocks scoring as
`final_source_used_in_candidate_development`.

## Assess untouched final evidence

Final records need explicit validated-scope confirmation and evidenced slicing
conditions. At least three untouched anonymous source groups must each contain
at least 200 accepted records. Missing coverage produces a blocked assessment;
it does not reduce the required source or row count. Unreadable or malformed final
evidence returns the bounded `final_evidence_unavailable_or_malformed` outcome and
still writes the create-only assessment evidence package.

```bash
.venv-candidates/bin/python -m minires.evaluation.assessment \
  --records private/final-test-records.json \
  --locked-candidate private/candidate-tuning/run-001/locked-candidate \
  --legacy-artifacts private/legacy-artifacts \
  --output-root private/candidate-assessment/run-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --bootstrap-seed 1729
```

The candidate and checksum-pinned legacy reference are scored on identical
accepted rows. Assessment uses 10,000 deterministic bootstrap replicates,
resampling complete miniature families within each observed source. Promotion
requires all of these inclusive gates:

- pooled and source-balanced candidate-to-legacy MAE upper one-sided 95% bounds
  are no greater than 2%;
- pooled and source-balanced candidate-minus-legacy within-2-g lower one-sided
  95% bounds are no less than -1 percentage point; and
- observed above-5-g fractions are no greater than 1% pooled, 1%
  source-balanced, and 2% for every source.

Tail confidence intervals are reported but do not replace the observed absolute
gates. An inconclusive or non-finite confidence result does not pass.

## Evidence and interpretation

Both phases write private row-level evidence, partition audits, fit metadata,
resource use, environment versions, blockers, and checksummed manifests. A
separate aggregate `public-summary-draft.json` is screened for private keys,
paths, fingerprints, source reports, row predictions, and raw errors.
`public-summary-review.json` records automated screening only; manual approval
is still required and no publication is performed.

A promoted candidate is for internal advisory use with human review of every
estimate. The result is limited to the observed anonymous source groups. It is
not evidence of population-wide performance, a prediction interval, actual
shop consumption, pricing accuracy, or an operational allowance. The legacy
reference remains a numerical comparator whose historical training provenance
is unknown; candidate source isolation does not reclassify it as clean holdout
evidence.

## Predeclared bounded-influence feature-signal diagnostic

Predeclaration status at commit `1308516`: **IMPLEMENTED NOT EXECUTED**.
The separately authorized execution has since completed; see the
[recorded result](issue-50-candidate-search.md#bounded-influence-feature-signal-diagnostic-result).
The predeclaration below does not authorize a further attempt.
This descriptive investigation asks whether the **existing correction inputs**
show associations distinguishing severe errors helped versus hurt by the frozen
bounded-influence correction. It does not fit a gate or qualify a candidate.
**The historical >7 g cohorts contain only 22–26 rows per cell. Three-dimensional
binning will be sparse; more bins do not create more samples.**

### Rationale and frozen fitting contract

Preserved run-017 improved MAE in all four cells without increasing serious-error
count, but only three cells passed its new-loss prerequisite. Split 101/model 42
had new-loss change +0.002716911. Its changes by anchor absolute-error bin were
-0.002645285 for `[0,4]`, -0.000545987 for `(4,5]`, -0.002984902 for `(5,7]`,
and +0.008893085 for `>7`; rounded contributions sum to the rounded total.
Of its 22 errors beyond 7 g, 13 worsened and nine improved. Split 101/model 41
also worsened that severe bin, offset elsewhere; both split-202 cells improved it.
Those preserved aggregates cannot correlate outcomes with correction features.
They justify prospective descriptive instrumentation, not an objective change,
claim of missing signal, or reinterpretation of the failed prerequisite.

`minires-bounded-influence-feature-signal-v1` reuses
`bounded_influence_training._fit_stage` without modifying any prior contract.
The exact float64 anchor remains `0.8*NN + (1.0-0.8)*XGBoost`; original base
contracts, seven legacy inputs, preprocessing, and **87 epochs / 1,091 trees**
are frozen, without early stopping. No new geometry, source/family/identity,
linkage, partition, path, or join-key prediction input is added. Retained `kb` and
`scale` have uncertain historical semantics and may remain source/style proxies.

The unchanged `minires-bounded-influence-correction-v1` fits

```text
H_d(z) = z² if |z| <= d, otherwise 2*d*|z| - d²
c = D*b; e = anchor + c - target
minimize mean[0.1*H_5(e) + 4*H_2.5(max(|e|-4.5,0)) + c²]
         + 0.01*sum(b²), subject to sum(|b|) <= 2 g
```

Its four-column design adds an intercept to the three inputs below. Exactly
2,000 zero-initialized Euclidean L1-projected-gradient steps use the existing
`1/(2*5.1*sum(D²)/n_train + 0.02)` step. The intercept participates in both
penalties and the L1 bound; prediction retains the final ±2 g roundoff clip.
No loss, constant, bound, feature, coefficient-fitting algorithm, or target changes.

The fixed four cells cross outer split seeds **101/202** (respective inner split
seeds **1101/1202**) with model seeds **41/42**. Five-fold identity-only SHA-256
round-robin assigns outer fold zero as the 20% training holdout. Each cell fits
its bases only on the other 80%, forms exactly one excluded-inner-fold OOF
prediction per training row/base, fits one frozen correction on those OOF inputs
and training targets, and fits both bases on that 80% partition for outer-held
prediction. All preprocessing is learned only on the relevant training rows;
correction means/scales come only from **training OOF** predictions, never full-fit
or outer-held statistics. Outer targets measure outcomes, not fitting or bins.

Exactly 40 inner-OOF base fits, eight outer-partition base fits, and four existing
numerical correction fits are allocated: **52 fits maximum / 7,200 seconds of
timed computation**, with no additional learner, discarded correction, or production stage.
All four cells are measured regardless of favorable/unfavorable metrics.

### Fixed feature bins, cohorts, and accounting

Bin the exact correction input representation, in this fixed order:

1. `anchor_g`;
2. `neural_minus_xgboost_g` (signed disagreement);
3. `absolute_disagreement_g`.

For each input use the frozen state's training-OOF mean and population standard
deviation (the existing floor rule replaces scales <=`1e-12` with 1), then
`z_j = clip((input_j - mean_j)/scale_j, -1, 1)`, exactly as at prediction.
No intercept is binned. Every feature has exactly these three fixed bins:

| Label | Interval | Boundary assignment |
| --- | --- | --- |
| `low` | `[-1,-1/3]` | -1 and exactly -1/3 included |
| `middle` | `(-1/3,1/3]` | exactly +1/3 included |
| `high` | `(1/3,1]` | +1 included |

Edges use float64 `-1.0/3.0` and `1.0/3.0`; equality goes to the lower bin.
Emit all **nine marginal groups** (three views of three bins) and all **27 joint
groups** in the feature order above, including every empty bin. Joint keys use
`anchor_bin|signed_disagreement_bin|absolute_disagreement_bin`, in fixed
low/middle/high Cartesian order, never performance order. No adaptive quantiles,
direction selection, best-bin ranking, or threshold/feature optimization.

Each view is emitted for `all` outer-held rows, `anchor_gt5` with strict anchor
absolute error >5 g, and `anchor_gt7` with strict anchor absolute error >7 g.
**These cohorts are nested; never sum across cohorts. Each marginal repeats the
same cohort, so never sum across marginal views either.** Targets define only
these evaluation cohorts and outcomes, not preprocessing, input bins, or models.

Every group records:

- count; `support_sufficient = count >= 20`, a fixed engineering **descriptive**
  threshold, not a statistical test; `support_status` is `empty`, `insufficient`,
  or `sufficient`. Retain every count/contribution below 20. Empty or insufficient
  support is **not evidence of no signal**, and sufficient support proves nothing;
- anchor and corrected residual-sign counts: prediction minus target, negative
  means underestimate, positive overestimate, zero exact agreement;
- absolute-error improved/worsened/unchanged counts using strict less/greater/equal;
- actual correction `corrected-anchor`: signed and absolute sums, maximum absolute
  magnitude, negative/zero/positive counts, and toward/away/neutral counts. Toward
  means opposite nonzero signs of anchor error and correction, away matching
  nonzero signs, neutral either zero. **Toward is not improved: overshoot can
  worsen absolute error or leave it equal**;
- the paired serious-error 2×2 counts: stable nonserious, harm (nonserious→serious),
  repair (serious→nonserious), persistent serious; strict `abs(error)>5 g`, exactly
  ±5 nonserious; and anchor/corrected above-5-g counts and their difference;
- anchor/corrected MAE, ordinary `0.1*H_5(e)`, weighted excess
  `4*H_2.5(max(|e|-4.5,0))`, and total new prediction-loss contributions, plus
  corrected-minus-anchor deltas. **All contributions divide subgroup sums by the
  entire held cell's n, not subgroup size or cohort size.** Both fitting penalties
  are excluded. Empty groups have zero sums/contributions and maximum magnitude.

Within each cohort, each marginal and the joint partition separately conserve all
counts exactly and all contributions/sums to relative/absolute `1e-12`; maxima
reconcile by maximum, not sum. Serious-count delta equals harms minus repairs.
Cohort totals independently reconcile to the existing run-017 numeric accounting
(total, `(5,7]` plus `>7`, and `>7` respectively). The complete old transition
helper output is retained under **`paired_transitions_with_old_diagnostic_loss`**:
all its losses, label-aware theoretical oracle and old flags use the OLD squared
loss, not bounded-influence loss or qualification. New baseline loss is separately
labelled **`new_descriptive_loss`** and never used to qualify. This preserves
full-cell/anchor-bin checks for later aggregate comparison with run-017 without
reading its private evidence or assuming numerical rerun equality now. Existing
complete finite per-base, anchor, and corrected OOF/full-fit shift summaries remain.

Wrong shape, nonfinite/out-of-range values, overflowing standardization/arithmetic,
malformed state, departure beyond ±2 g (allowing the existing four-ulp paired
addition roundoff), nonconserving evidence, invalid backend preprocessing, or
incomplete fits block completion. No row is dropped, imputed, or substituted.

### Artifacts, command, and stop/interpretation limits

The create-only package contains `diagnostic-plan.json` (before any fit),
`feature-signal-evidence.json`, and `manifest.json` checksumming both. The private
plan binds full frozen numeric/base/runtime/preprocessing contracts, ordered
features/bins/cohorts, normalization, seeds/folds, support/accounting, exact budget,
expected training checksum, observed training-only input/code identities, and
required/observed dependencies/platform. Evidence retains completed cells and
failed/uncompleted accounting, fits started/completed/unused, elapsed and CPU time,
and explicitly false qualification/selection/locking/continuation attestations.
No raw rows, predictions, per-row feature values, identities, source groups,
source/family values, or mappings are emitted. Existing private plan checksum
provenance is not published evidence; no model/state artifacts are saved for use.

The training-only CLI has no validation/test, seed, budget, bins/support, feature,
loss, qualification, selection, or lock knob. It verifies only the committed
training checksum and pinned Python **3.13**, NumPy **2.2.6**, Keras **3.15.0**,
TensorFlow **2.20.0**, scikit-learn **1.7.2**, and XGBoost **3.1.2** before fitting.
The in-memory runtime seam remains available for synthetic tests. A separately
authorized one-time execution after committed predeclaration and a source-neutral
Issue #50 predeclaration would use fresh create-only **run-018**:

```bash
.venv-candidates/bin/python -m minires.modeling.bounded_influence_feature_signal \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-018 \
  --volume-unit mm3 \
  --scope-confirmed
```

Stop after one completed or blocked attempt, **regardless of outcome**. The
7,200-second deadline and reported elapsed/CPU time cover the runner from its
post-directory-creation clock start through the final post-summary clock sample.
They include loading/normalization and the initial plan write, but exclude CLI
preflight and final artifact packaging (finite-tree validation, evidence JSON
serialization/writes, checksum reads, and manifest writing). They are not a
whole-command wall-clock limit. Deadline checks surround fits, predictions in the
reused stage, summaries, and the computation-completion decision; a running
backend fit cannot be forcibly interrupted. Invalid/runtime/deadline
failure preserves completed cells and partial fit counts without replacement.
Artifact-write failure cannot return a complete checksummed package. No selection,
qualification, production fit, validation/test access or fingerprints, ranking,
locking, learned gate, retry, budget recycling, or automatic continuation exists.

Fixed-bin descriptive associations can inform a later separately authorized
investigation but **cannot prove out-of-sample gating, establish predictive or
causal conclusions, or rule out nonlinear signal**. Sparse joint bins and empty
bins do not imply no signal. Signed and absolute disagreement are dependent, not
three independent information sources. Comparisons are paired within cell;
overlapping training holdouts are **not independent**, and different split
contrasts are not paired or causal. Evidence is conditional on historically
reused-validation-selected anchor/counts, not untouched pipeline evidence. No
many-bin multiplicity is treated as increased sample size. Any learned gate or
further experiment requires a new decision/predeclaration and authorization.
This implementation performs synthetic tests only: no private fitting, dataset
contents or validation/test reads/fingerprints, run-directory creation, publication,
or Issue #33 work. Prior runs and contracts remain immutable.

## Predeclared training-only data and feature audit

Status: **IMPLEMENTED NOT EXECUTED**. This is a bounded audit of the frozen base
predictor, not a correction, candidate round, seed/loss/feature/ensemble search, or
authorization to fit. The audit asks three falsifiable questions before proposing
another model intervention:

1. Do explicit canonical/legacy unit, geometry, density, scope, or slicing-condition
   declarations contradict one another?
2. Do records with exact or deterministically similar available canonical geometry
   have materially different sliced resin mass targets?
3. If the first two routes do not explain the evidence, does the frozen closest
   base anchor have stable, supported strict-above-5-g enrichment in a predeclared
   geometry regime across both training splits and both model seeds?

### What existing evidence can answer

No fitting is needed to reconcile every accepted training row's canonical and
legacy aliases, explicit volume unit, bounding-box product, mesh-within-box
constraint, surface-to-volume ratio, density, layer height, slicer-added-support
setting, and scope declaration. Missing or
unknown evidence is counted separately and is not called a contradiction. The
committed provenance states that historical measurements were reused as
authoritative without mesh probing or slicing, while only the new batch has the
pinned measurement contract. Assembly harmonized row fields and removed row origin,
so this audit cannot retrospectively verify which rows actually used the new slicing
contract. It also does not treat sliced-resin-mass-to-mesh-volume ratio as a defect
test: legitimate slicing geometry can make that ratio differ from nominal resin
density.

The same no-fitting stage compares sliced resin mass within exact seven-field
canonical-geometry groups. It also applies one deterministic source-neutral nearest-
geometry contract. Positive geometry fields are log transformed; Euler number uses
signed `log1p(abs(euler))`. Columns use median centering and the greater of
`1.4826*MAD`, `IQR/1.349`, and `1e-12` (a remaining zero scale becomes one). Each
row is related to the lowest-row-order nearest nonidentical geometry by RMS robust-
standardized distance. A relationship is similar only at distance at most `0.10`.
All relationship counts and target-difference bins are reported; no pair, row,
identity, feature value, target, source, or path is emitted. Material heterogeneity
means target difference strictly above 5 g. An exact-group signal requires at least
20 repeated-group rows and at least 10% of those rows in materially heterogeneous
groups. A nearest-geometry signal requires at least 100 similar relationships and a
materially heterogeneous fraction of at least 10%. These fixed thresholds are
descriptive engineering rules, not optimized statistical cutoffs.

Preserved run evidence is aggregate-only and cannot attribute severe base errors to
row geometry. That question therefore requires new computation. The minimum fixed
allocation is eight fits: the original closest neural network and XGBoost component
fit once in each of four cells crossing outer split seeds 101/202 with model seeds
41/42. Existing identity-SHA-256 five-fold assignment holds fold zero out. Each base
fits the other 80% at exactly 87 neural epochs or 1,091 XGBoost trees, without early
stopping, then predicts only the outer-held training rows. The anchor preserves the
exact float64 operation `0.8*NN + (1.0-0.8)*XGBoost`. There is no inner OOF fitting,
correction, production fit, validation score, or saved model artifact.

### Geometry regimes and decision rules

Each cell reports all relative quintiles for log mesh volume, surface compactness
`surface_area / volume^(2/3)`, mesh-volume-to-bounding-box-volume fill, and sorted-
dimension longest/shortest aspect ratio. Linear quintile edges come only from that
cell's outer fitting partition; outer-held targets and predictions do not choose
edges. Euler number has fixed negative/zero/positive bins. Thus all 23 declared
view/bin combinations remain visible, including empty and unfavorable bins.

A bin is supported at 100 held rows. It is enriched only if its strict-above-5-g
rate is at least twice the cell rate and at least 0.01 higher in absolute rate. A
base-predictor regime is stable only when the same named relative view/bin is
supported and enriched in all four cells. These multiple overlapping views are a
descriptive screen, not independent tests; no p-value, favorable-bin selection,
threshold change, or causal claim follows. Missing required slicing attributes use
the same per-cell support/enrichment rule and must pass in all four cells before
they are called error-enriched.

The final route is mutually prioritized and fixed before execution:

1. `investigate_data_contracts` for any explicit alias, geometry, slicing, density,
   or scope contradiction/invalid explicit value;
2. otherwise `obtain_better_inputs` if exact/similar-geometry target heterogeneity
   passes its fixed support rule, or missing required slicing attributes are
   supported and error-enriched in all four cells;
3. otherwise `improve_base_predictor` only if at least one predeclared geometry
   regime is supported and enriched in all four cells; or
4. `inconclusive_collect_evidence`, which explicitly says no model run is warranted.

A route is diagnostic guidance, not proof of a defect, permission to repair a row,
or authorization for another fit. Every row remains in place. Historical anchor
and count choices used reused validation evidence, so results are conditional on
that selection and are not untouched pipeline evidence. Outer holdouts overlap
across split definitions, split contrasts are not paired or causal, and this audit
cannot establish slicing conditions absent from the artifacts.

### Finite command, artifacts, and stop

The one prospective attempt has a hard maximum of **8 fits and 7,200 seconds**.
Every cell runs regardless of findings; unused capacity is not recycled. Wrong
shape, nonfinite/overflowing input or prediction, invalid preprocessing, incomplete
accounting, runtime failure, or deadline blocks without replacement. The create-
only private package contains `audit-plan.json`, aggregate
`training-data-feature-audit.json`, and `manifest.json` checksumming both. It writes
no row vectors, IDs, paths, sources/source groups, artist identity, per-row geometry,
targets, predictions, nearest-neighbor pairs, source mappings, or model artifacts.
Private plan fingerprints bind training/code provenance but are not publishable
results.

The CLI accepts only training records, output root, volume unit, and scope
confirmation. It checksum-verifies only committed `data/train.jsonl` and the pinned
candidate environment before fitting. Validation and held-out-test paths are not
arguments and must not be read or fingerprinted. After this predeclaration is
committed and a source-neutral Issue #50 comment is posted, a separately authorized
single execution would use fresh create-only `run-019`:

```bash
.venv-candidates/bin/python -m minires.modeling.training_data_feature_audit \
  --training-records data/train.jsonl \
  --output-root private/candidate-tuning/run-019 \
  --volume-unit mm3 \
  --scope-confirmed
```

Stop after one completed or blocked attempt regardless of outcome. There is no row
repair/removal, retry, threshold tuning, candidate search, correction, seed/loss/
feature/ensemble search, gate relaxation, validation access, lock, promotion,
automatic continuation, or Issue #33 work. Implementation used synthetic tests
only; `run-019` was not created or inspected and no private model fitting occurred.

### Completed outcome

The separately authorized `run-019` audit completed all four cells and all eight
fixed base fits without blockers. It used 28.06 elapsed seconds and 79.60 process
CPU seconds. Both create-only artifact checksums were independently recomputed and
matched; all 20 per-cell view partitions conserved row counts, serious-error counts,
and MAE contributions. Validation and held-out-test evidence were not accessed,
source groups were not used, no rows were changed, and no candidate or lock was
created.

The no-fitting contract stage reconciled 10,368 training rows. Every canonical/
legacy alias pair matched, as did every bounding-box dimension product,
surface-to-volume ratio, and explicit unit, density, scope, and slicing declaration.
It nevertheless found 1,346 rows whose recorded mesh volume was greater than their
recorded axis-aligned bounding-box volume. Under the predeclared rule this is an
explicit geometry-contract contradiction and therefore triggers the first-priority
`investigate_data_contracts` route. This is a mechanical routing result, **not** a
finding that 1,346 labels or rows are defective. Overlapping shells, topology, or
historical measurement semantics could make the assumed reconciliation rule
inapplicable; that question must be resolved before considering repair. Harmonized
per-row declarations also cannot prove the actual historical slicing conditions.

Target heterogeneity did not trigger its fixed rule. There were 793 repeated exact-
geometry groups containing 1,870 rows and none had sliced resin mass spread above
5 g. Of 6,606 fixed similar-geometry relationships, 566 had target differences above
5 g (8.57%), below the predeclared 10% trigger. These results do not prove targets
are interchangeable or complete; they only fail this bounded descriptive screen.

Every cell evaluated 2,074 outer-held training rows. Anchor MAE was 0.8953, 0.8838,
0.8988, and 0.9501 g, with respectively 34, 34, 35, and 42 strict-above-5-g errors.
The highest relative mesh-volume quintile and highest bounding-box-fill quintile
were both supported and enriched under the fixed rule in all four cells. The
mesh-volume quintile held 23–38 serious errors among 420–423 rows (5.48%–8.98%,
3.34–4.48 times each cell rate); the fill quintile held 15–19 among 378–386 rows
(3.89%–5.03%, 2.37–2.98 times each cell rate). These stable descriptive regimes are
retained but do not override the higher-priority contract route, prove causality,
or authorize predictor work.

The audit therefore stops at `investigate_data_contracts`. It does not authorize
row repair/removal, reslicing, further fitting, validation access, candidate search,
selection, locking, promotion, or Issue #33. Any follow-up must first specify how to
check the geometry measurement contract without identifying sources or changing the
canonical artifacts, and requires a separate decision and authorization.
