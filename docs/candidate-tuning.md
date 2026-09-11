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
contains six dense neural-network trials and six XGBoost trials. Within each
outer source holdout, it builds three ensembles from the strongest equal-rank
component pairings on that fold's permitted validation predictions only. It
then repeats the five highest-ranked eligible candidates with the second seed.
One complete
cross-validated candidate evaluation is one run, including an ensemble whose
components must be refitted. The fixed learned configuration is evaluated as a
control and does not consume a candidate run.

The workflow never starts a candidate after 20 new runs or after 7,200 elapsed
seconds. A partial search cannot select or lock a candidate. Runtime failures,
invalid plans, missing grouping evidence, deadline exhaustion, and the absence
of eligible candidates remain bounded blockers in `tuning-result.json`.

Eligibility applies the observed above-5-g error limits before ranking. Eligible
candidates rank by source-balanced mean absolute error, pooled mean absolute
error, within-2-g fraction, and stable candidate ID. First- and second-seed
metrics receive equal weight. The selected configuration is refitted on all
eligible development records with median cross-validation-derived epoch and
tree counts and without final-test early stopping.

## Locked candidate

A successful search creates `locked-candidate/` with:

- the complete candidate, ranking, eligibility, feature, preprocessing,
  dependency, input, code, and development-split contract;
- the fitted model artifact or artifacts;
- the fitted preprocessing state; and
- SHA-256 checksums in a create-only lock manifest.

Use `load_locked_candidate` from `minires_evaluation.tuning` when assessment runs
in a later process. Loading and assessment fail closed if an artifact,
configuration, preprocessing record, dependency contract, or checksum changes.
The assessment boundary verifies the lock and reloads the predictor from those
verified artifacts; it never executes a caller-supplied in-memory predictor.
Final-test records are not loaded until lock verification succeeds.

## Assess untouched final evidence

Final records need explicit validated-scope confirmation and evidenced slicing
conditions. At least three untouched anonymous source groups must each contain
at least 200 accepted records. Missing coverage produces a blocked assessment;
it does not reduce the required source or row count.

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
shop consumption, pricing accuracy, or an operational allowance.
