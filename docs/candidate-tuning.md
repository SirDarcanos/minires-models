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
  -r requirements-evaluation.txt \
  -r requirements-legacy.txt \
  -r requirements-learned.txt
```

Development records must use the canonical private source and miniature-family
evidence described in [the normalization reference](normalization.md). Source,
family, duplicate, location, and record identity evidence is used only for
partitioning; it never enters a model feature matrix.

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
.venv-candidates/bin/python -m minires_evaluation.workflow \
  --development-records private/development-records.json \
  --final-records private/final-test-records.json \
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

Development evidence is used for fitting, preprocessing, early stopping,
ensemble selection, threshold selection, ranking, training-duration selection,
and candidate locking. Final evidence is not read until the lock has been created
and checksum-verified. The final stage only performs paired assessment; it cannot
change candidate weights, training duration, ranking, or the compute budget.

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
bounded blockers. Inputs, released weights, retained notebooks, and supplied final
records are never written by this workflow.

To reassess an existing lock without tuning, candidate selection, or additional
compute, use the same command in assessment-only mode:

```bash
.venv-candidates/bin/python -m minires_evaluation.workflow \
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
from minires_evaluation import EvaluationConfig
from minires_evaluation.tuning import (
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
from minires_evaluation import EvaluationConfig
from minires_evaluation.tuning import SearchLimits, create_search_plan

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
.venv-candidates/bin/python -m minires_evaluation.tuning \
  --records private/development-records.json \
  --output-root private/candidate-tuning/run-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41
```

The workflow creates `search-plan.json` before fitting. The fixed-seed plan
contains six dense neural-network trials and six XGBoost trials and predeclares
three deterministic equal-rank ensemble rules. Within each outer source holdout,
it applies those rules to component rankings from that fold's permitted validation
predictions only. It then repeats the five highest-ranked eligible candidates with
the second seed.
One complete
cross-validated candidate evaluation is one run, including an ensemble whose
components must be refitted. The fixed learned configuration is evaluated as a
control and does not consume a candidate run.

The workflow never starts a candidate after 20 new runs or after 7,200 elapsed
seconds. If fewer than five initial candidates are eligible, it repeats every
eligible candidate and records the finalist shortfall; unused capacity does not
expand the search. A partial repetition stage cannot select or lock a candidate.
Runtime failures, invalid plans, missing grouping evidence, deadline exhaustion,
and the absence of eligible candidates remain bounded outcomes in
`tuning-result.json`. The private `candidate_history` records the control, every
completed or failed launch, and every skipped slot with its bounded stop reason.
A stopped search preserves all completed results and cannot lock a candidate.

Eligibility applies the observed above-5-g error limits before ranking: no more
than 1% pooled, 1% under equal source weighting, and 2% for each source with at
least 200 accepted records. Smaller sources remain part of pooled and
source-balanced metrics, but do not receive the separate per-source gate.
Ineligible results remain in the private initial results and history but are
excluded from `initial_promotable_ranking`. Eligible candidates rank by
source-balanced mean absolute error, pooled mean absolute error, within-2-g
fraction, and stable candidate ID. The five highest-ranked eligible candidates,
or every eligible candidate when fewer than five qualify, receive the second
seed. First- and second-seed metrics receive equal weight, including unfavorable
results; eligibility and `promotable_ranking` are then recomputed from those
combined metrics. The report records both rankings, the finalist shortfall, and
each ensemble's selected fold weights explicitly.

A complete repetition stage selects the best combined eligible result by the
same deterministic rule. Epoch and tree counts are fixed from the median values
selected across every permitted development fold and both seeds. The chosen
component or resolved ensemble is refitted on all included development records,
without validation data or final-test early stopping. A resolved ensemble keeps
its component contracts and fixes its convex weight from the same two-seed fold
evidence. If no initial candidate qualifies, the run records the best initial
development result, performs no repetitions or refit, and ends as
`completed_no_candidate` without expanding the budget.

## Locked candidate

A successful search creates `locked-candidate/` with:

- the complete component or ensemble configuration and output unit;
- the two seeds, equal-weight combined evidence, deterministic ranking and
  eligibility rules, and fixed training counts;
- feature, preprocessing, dependency-environment, input, code, search-plan, and
  development-split identities;
- the one-way identities of every development source and an explicit record that
  fitting, preprocessing, early stopping, ensemble selection, threshold selection,
  and candidate locking used development records only;
- the fitted model artifact or artifacts;
- the fitted preprocessing state; and
- SHA-256 checksums in a create-only lock manifest.

The tuning and refit seam receives only normalized development features and
labels. It has no final-test argument or final-test path, and the contract records
that final-test access did not occur. Final-test records, source metadata,
fingerprints, labels, and predictions enter only through the later assessment
command after lock verification.

Use `load_locked_candidate` from `minires_evaluation.tuning` when assessment runs
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
.venv-candidates/bin/python -m minires_evaluation.assessment \
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
