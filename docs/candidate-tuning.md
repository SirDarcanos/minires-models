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
