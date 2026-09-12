# Prepare pre-supported STL files

The one-STL workflow creates either one identity-free MiniRes record or one named rejection. The resumable batch workflow extends the same measurement path to a directory. Both inventory source files, probe canonical geometry, slice private copies, and read sliced resin mass from UVtools `WeightG`. Neither uses a failure as a zero-gram target.

## Pinned contract

The workflow fails closed unless all of these checks pass before geometry processing or slicing:

- the bundled `src/minires/preparation/profiles/config-anycubic-mono.ini` has SHA-256 `06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e`;
- the profile specifies density 1.1 g/ml, layer height 0.05 mm, and `supports_enable = 0`;
- PrusaSlicer, UVtools, and trimesh are available.

The private result records the Python version and the versions reported by each available tool. Geometry values and `WeightG` are serialized without workflow-level rounding. Input and generated sliced-output byte counts and SHA-256 checksums are recorded before cleanup.

## Install and run a smoke input

Install the project and geometry dependency, then make `prusa-slicer` and `UVtoolsCmd` available on `PATH`:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements/preparation.txt
mkdir -p private/smoke
python3 -m minires.preparation.prepare_one \
  --stl /private/path/to/pre-supported-input.stl \
  --private-output private/smoke/result.json \
  --scope-confirmed
```

Use one representative input as the smoke invocation before arranging any larger private batch. Every invocation completes the full environment preflight before it copies or processes the STL.

`--scope-confirmed` is the operator's explicit attestation that the input is a pre-supported miniature in the validated scope; geometry checks do not establish scope. Without it, the command records `scope_confirmation_required`.

The command prints only `prepared` or a bounded rejection code. The input filename, source identity, raw path, checksum, measurements, and sliced resin mass remain in the private result and never appear in command output.

## Run a resumable batch

Run the one-file smoke invocation above before increasing the batch size or concurrency. Then choose a private result path and a private checkpoint directory:

```bash
mkdir -p private/stl-batch
python3 -m minires.preparation.prepare_batch \
  --input-directory /private/path/to/pre-supported-stls \
  --private-output private/stl-batch/result.json \
  --private-checkpoints private/stl-batch/checkpoints \
  --scope-confirmed
```

The command performs the profile and external-tool preflight before per-file work. It then recursively inventories directory entries without following symlinks. Regular `.stl` files receive byte counts and SHA-256 evidence before processing. Empty, unreadable, changing, symlinked, non-regular, and unsupported-extension entries receive bounded private outcomes. A mixed batch with accountable per-file rejections still completes successfully. A run-level preflight, interruption, persistence, or reconciliation failure exits nonzero.

The default is one worker because individual meshes can be very large. After a representative smoke run establishes safe memory and disk use, `--workers N` permits explicit concurrency up to four workers. Start with one and increase conservatively.

### Checkpoints and resumption

A completed accepted or rejected outcome is committed atomically after each unique input. Its key combines the input SHA-256 with a complete measurement-contract fingerprint. The fingerprint covers the pinned slicing contract, record and checkpoint schema versions, the geometry algorithm version, and the reported Python, trimesh, PrusaSlicer, and UVtools versions.

Rerun the same command to resume. A matching checkpoint is reused without geometry probing, slicing, or UVtools extraction. Changed bytes or any fingerprint component use a different key and trigger fresh processing. Exact checksum duplicates share one processed or reused outcome and carry common duplicate evidence, while each inventory entry still receives one accepted row or rejection. The final result is rebuilt and atomically replaced rather than appended, so resumption cannot duplicate accepted rows.

An interruption leaves every already committed checkpoint intact and does not publish a partial final result. Correct the interruption and run the same command again. Do not edit checkpoint files manually; malformed or incomplete files are ignored and recomputed.

### Private batch output

`result.json` is private row-level evidence. It contains the inventory, accepted canonical and legacy-compatible records, bounded rejections, exact-duplicate groups, count reconciliation, elapsed resource accounting, contract provenance, and tool versions. Checkpoint files also contain row-level measurements and labels. Keep both locations under ignored private storage and do not publish them.

The command prints only `completed`, `interrupted`, or a bounded run-level error. It never prints paths, filenames, checksums, measurements, labels, duplicate mappings, per-entry failures, or tool diagnostics. A completed result always satisfies:

```text
inventory_count = accepted_count + rejected_count
```

## Process and cleanup boundaries

Every external application receives an argument vector directly; no shell command is built. Each call has a timeout and must exit successfully. PrusaSlicer receives an explicit output path in a mode-0700 temporary workspace. Geometry and slicing operate on a private copy named `input.stl`, not the source. The entire workspace, including sliced output, is removed after success, rejection, or timeout, and the source checksum is checked again before returning.

Named processing rejections include:

- `corrupt_geometry` and `invalid_measurements`;
- `slicing_failed`, `slicing_timeout`, and `sliced_output_absent`;
- `uvtools_failed`, `uvtools_timeout`, `weight_g_missing`, and `invalid_weight_g`;
- `cleanup_failed` and `source_checksum_changed`.

Preflight contract and dependency failures have separate codes such as `profile_checksum_mismatch`, `profile_contract_mismatch`, `missing_prusaslicer`, `missing_uvtools`, and `missing_geometry_dependency`.
