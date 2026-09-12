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

The current code supports data preparation and governed model development. A replacement production model has not been declared by this repository yet. Candidate selection and final assessment must complete before a model is presented as the current MiniRes estimator.

Start with:

- [Model definitions](docs/model-definitions.md) for active architecture and runtime contracts.
- [Candidate tuning and locked assessment](docs/candidate-tuning.md) for the replacement-model workflow.
- [Source-balanced partitions](docs/normalization.md#generate-source-balanced-partitions) for current data allocation.
- [One-STL preparation](docs/stl-preparation.md) for generating a new compatible record.

## Prepare one STL

Install the geometry dependency and make `prusa-slicer` and `UVtoolsCmd` available on `PATH`:

```bash
python3 -m pip install -r requirements-preparation.txt
mkdir -p private/smoke
python3 -m minires_evaluation.prepare_one \
  --stl /private/path/to/pre-supported-input.stl \
  --private-output private/smoke/result.json \
  --scope-confirmed
```

The workflow verifies the bundled profile checksum and slicing settings before processing. It writes generated sliced output only inside a private temporary workspace and prints no filename, path, checksum, measurements, or label.

## Model development

Install the evaluation dependencies:

```bash
python3 -m pip install -r requirements-evaluation.txt
```

Optional model runtimes have separate pinned requirement files:

- `requirements-learned.txt` for clean refits and candidate training;
- `requirements-legacy.txt` for comparison with the released reference only.

The main references are:

- [Normalization and private data preparation](docs/normalization.md)
- [Physical baseline](docs/physical-baseline.md)
- [Clean learned baselines](docs/learned-baselines.md)
- [Model definitions](docs/model-definitions.md)
- [Candidate tuning](docs/candidate-tuning.md)
- [Baseline evidence](docs/baseline-evidence.md)

`baseline_analysis.ipynb` remains because it is a current, output-free view over the public aggregate evaluation interface. It is not a historical training notebook and is not required by the runtime.

## Privacy

Source identities, raw paths, STL files, checksums, row-level measurements, labels, predictions, and mappings stay in ignored private storage. Published output must not identify specific artists. Anonymous source groups are evaluation metadata, not model inputs.

## Tests

```bash
python3 -m unittest discover -s tests
```

Some tests require the optional dependencies listed above.

[MIT License](LICENSE)
