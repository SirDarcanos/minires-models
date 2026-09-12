# Repository structure

This reference is for maintainers changing MiniRes preparation, modeling, or evaluation behavior.

```text
src/
  minires/
    preparation/   Private record preparation, partitioning, and STL slicing
    modeling/      Model definitions, fitting, tuning, and legacy comparison
    evaluation/    Baseline evaluation, assessment, evidence, and workflows
tests/             Tests of installed package interfaces and command adapters
docs/              Maintainer and workflow documentation
notebooks/         Output-free analysis callers; executed copies stay private
requirements/      Pinned dependencies grouped by workflow
```

## Package interfaces

`src/` is the source root and `minires` is the installed import package. Use `minires` in imports and module commands; never import `src.minires`.

The package exposes three responsibility-based interfaces:

- `minires.preparation` turns private source data into compatible records and deterministic partitions. Its command adapters also probe and slice one pre-supported STL while keeping generated output private.
- `minires.modeling` defines and fits replacement candidates. `minires.modeling.legacy` is isolated here because released model artifacts are supported only as a checksum-pinned comparison.
- `minires.evaluation` evaluates estimates and produces governed assessment evidence.

Shared record normalization and private-file handling remain package-level modules because all three interfaces use them. Command modules are adapters over these interfaces; they do not contain duplicate domain logic.

## Set up a development environment

Create a virtual environment, install the project in editable mode, and add only the dependencies required by the workflow you are changing:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements/evaluation.txt
```

Use `requirements/preparation.txt` for STL geometry probing, `requirements/learned.txt` for replacement-model fitting, and `requirements/legacy.txt` only for comparison with released artifacts.

## Run checks

Run tests with the interpreter where the project is installed:

```bash
.venv/bin/python -m unittest discover -s tests
```

The source layout prevents the repository root from satisfying `import minires`. This makes missing installation metadata and missing package resources fail instead of being hidden by the working directory. The bundled slicing profile is declared as package data and loaded through `importlib.resources`.

Run static checks against the installed source tree:

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/mypy src/minires --ignore-missing-imports
```

Some tests and static checks require the optional pinned runtimes above.

## Notebooks and private data

`notebooks/baseline_analysis.ipynb` is an output-free caller of the aggregate evaluation interface. Execute a copy under `private/`; do not commit its outputs. Source identities, raw paths, row measurements, labels, predictions, and artifact mappings also stay in ignored private storage.
