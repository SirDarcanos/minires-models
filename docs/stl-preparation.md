# Prepare one pre-supported STL

The one-STL workflow creates either one identity-free MiniRes record or one named rejection. It inventories the source, probes canonical geometry, slices a private copy, and reads the sliced resin mass from UVtools `WeightG`. It never uses a failure as a zero-gram target.

## Pinned contract

The workflow fails closed unless all of these checks pass before geometry processing or slicing:

- the EBMiniManager checkout is exactly revision `1a841195813136ee3b380ab1d192727f385f7a55`;
- `prediction/config-anycubic-mono.ini` has SHA-256 `06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e`;
- the profile specifies density 1.1 g/ml, layer height 0.05 mm, and `supports_enable = 0`;
- PrusaSlicer, UVtools, and trimesh are available.

The private result records the Python version and the versions reported by each available tool. Geometry values and `WeightG` are serialized without workflow-level rounding. Input and generated sliced-output byte counts and SHA-256 checksums are recorded before cleanup.

## Install and run a smoke input

Install the geometry dependency and make `prusa-slicer` and `UVtoolsCmd` available on `PATH`:

```bash
python3 -m pip install -r requirements-preparation.txt
git -C /path/to/EBMiniManager checkout 1a841195813136ee3b380ab1d192727f385f7a55
mkdir -p private/smoke
python3 -m minires_evaluation.prepare_one \
  --stl /private/path/to/pre-supported-input.stl \
  --ebminimanager-dir /path/to/EBMiniManager \
  --private-output private/smoke/result.json
```

Use one representative input as the smoke invocation before arranging any larger private batch. Every invocation completes the full environment preflight before it copies or processes the STL.

The command prints only `prepared` or a bounded rejection code. The input filename, source identity, raw path, checksum, measurements, and sliced resin mass remain in the private result and never appear in command output.

## Process and cleanup boundaries

Every external application receives an argument vector directly; no shell command is built. Each call has a timeout and must exit successfully. PrusaSlicer receives an explicit output path in a mode-0700 temporary workspace. Geometry and slicing operate on a private copy named `input.stl`, not the source. The entire workspace, including sliced output, is removed after success, rejection, or timeout, and the source checksum is checked again before returning.

Named processing rejections include:

- `corrupt_geometry` and `invalid_measurements`;
- `slicing_failed`, `slicing_timeout`, and `sliced_output_absent`;
- `uvtools_failed`, `uvtools_timeout`, `weight_g_missing`, and `invalid_weight_g`;
- `cleanup_failed` and `source_checksum_changed`.

Preflight contract and dependency failures have separate codes such as `profile_checksum_mismatch`, `profile_contract_mismatch`, `missing_prusaslicer`, `missing_uvtools`, and `missing_geometry_dependency`.
