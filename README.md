# MiniRes model development

MiniRes is a private-data preparation, training, and evaluation workspace for an estimator of **sliced resin mass** for pre-supported miniatures.

This repository is developing and validating the next MiniRes model. It does not treat the previously released ensemble as the current model. The released artifacts remain available only as a checksum-pinned comparison baseline while replacement candidates are trained and assessed on held-out evidence.

## What this repository provides

- A one-STL preparation workflow that inventories geometry, slices with a bundled checksum-pinned profile, and extracts UVtools `WeightG` without exposing the input identity.
- Deterministic private dataset preparation and source-balanced train, validation, and held-out test partitions.
- Explicit neural-network, XGBoost, and ensemble model specifications.
- Bounded candidate search, candidate locking, and final assessment workflows.
- Reproducible physical, clean-refit, and legacy comparison baselines.

MiniRes estimates sliced resin mass under validated slicing conditions. It does not estimate prices or actual shop resin consumption.

## Repository status

The current code supports data preparation and governed model development. The latest tail-aware expanded candidate round produced no eligible lock, so final assessment was not started and no replacement production model has been declared. Further tuning requires a separate predeclared hypothesis and finite plan rather than automatic search expansion.

Start with:

- [Repository structure](docs/repository-structure.md) for package interfaces and maintainer setup.
- [Model definitions](docs/model-definitions.md) for active architecture and runtime contracts.
- [Candidate tuning and locked assessment](docs/candidate-tuning.md) for the replacement-model workflow.
- [Source-balanced partitions](docs/normalization.md#generate-source-balanced-partitions) for current data allocation.
- [STL preparation](docs/stl-preparation.md) for one-file smoke checks and resumable batches.

## Prepare STL files

Install the project and geometry dependency, then make `prusa-slicer` and `UVtoolsCmd` available on `PATH`:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements/preparation.txt
python3 -m minires.preparation.diagnose_toolchain
mkdir -p private/smoke
python3 -m minires.preparation.prepare_one \
  --stl /private/path/to/pre-supported-input.stl \
  --private-output private/smoke/result.json \
  --scope-confirmed
```

The synthetic diagnostic verifies installed-tool interoperability without claiming validated-scope eligibility. The preparation workflow then verifies the bundled profile checksum and slicing settings before processing an operator-confirmed private pre-supported miniature. Both write generated output only inside temporary workspaces and print no filename, path, checksum, measurements, or label.

After that smoke check succeeds, prepare a directory with durable private checkpoints. Concurrency defaults to one worker:

```bash
mkdir -p private/stl-batch
python3 -m minires.preparation.prepare_batch \
  --input-directory /private/path/to/pre-supported-stls \
  --private-output private/stl-batch/result.json \
  --private-checkpoints private/stl-batch/checkpoints \
  --scope-confirmed
```

See [STL preparation](docs/stl-preparation.md) for checkpoint invalidation, resumption, safe concurrency, and private output semantics.

After the complete new-source batch is available, assemble it with the retained
historical measurements and create the current source-balanced package:

```bash
python3 -m minires.preparation.assemble \
  --historical-records private/historical-export.jsonl \
  --exclude-historical-source "$PRIVATE_HISTORICAL_SOURCE_TO_OMIT" \
  --new-batch-result private/issue-30/batch/result.json \
  --seed 23 \
  --private-dir private/current-dataset
```

Historical rows are validated without re-probing or re-slicing them. See
[Normalization and private data preparation](docs/normalization.md#assemble-the-current-four-source-dataset)
for the package contract and the held-out-row claim it supports.

## Model development

Install the project and evaluation dependencies:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements/evaluation.txt
```

Optional model runtimes have separate pinned requirement files:

- `requirements/learned.txt` for clean refits and candidate training;
- `requirements/legacy.txt` for comparison with the released reference only.

Run bounded development with the explicit prepared partitions. The development
command deliberately has no test-artifact option:

```bash
python3 -m minires.modeling.tuning \
  --training-records private/current-dataset/train.jsonl \
  --validation-records private/current-dataset/validation.jsonl \
  --output-root private/candidate-tuning/run-001 \
  --volume-unit mm3 \
  --scope-confirmed \
  --seed 41
```

Fitting and learned preprocessing use training rows only. Validation is grouped
by anonymous source group for pooled, equal-source-weighted, and qualifying-source
eligibility checks, then used for early stopping and candidate/ensemble selection.
Source groups remain outside the prediction features and public output. See the
candidate-tuning reference before separately assessing a lock against the test
artifact.

The main references are:

- [Normalization and private data preparation](docs/normalization.md)
- [Physical baseline](docs/physical-baseline.md)
- [Clean learned baselines](docs/learned-baselines.md)
- [Model definitions](docs/model-definitions.md)
- [Candidate tuning](docs/candidate-tuning.md)
- [Regenerated-data tuning result](docs/issue-49-completion.md)
- [Tail-aware expanded tuning result](docs/issue-46-completion.md)
- [Baseline evidence](docs/baseline-evidence.md)

`notebooks/baseline_analysis.ipynb` remains because it is a current, output-free view over the public aggregate evaluation interface. It is not a historical training notebook and is not required by the runtime.

## Privacy

Source identities, raw paths, STL files, checksums, row-level measurements, labels, predictions, and mappings stay in ignored private storage. Published output must not identify specific artists. Anonymous source groups are evaluation metadata, not model inputs.

## Tests

Install the project in editable mode before running tests so they resolve the installed `src/minires` package rather than source from the repository root:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests
```

Some tests require the optional dependencies listed above. See [Repository structure](docs/repository-structure.md) for the package layout and maintainer workflow.

[MIT License](LICENSE)
