# Issue 53: print-axis surface-signature feasibility

This investigation asks whether the existing pre-supported development STL files can all provide one deterministic inference-time geometry representation under bounded resources. It does not fit a model, inspect sliced resin mass, select features from target performance, or access held-out-test geometry.

## Representation decision

The investigation uses `minires-print-axis-surface-signature-v1`, not voxel occupancy. The earlier voxel route was not reliable on the available toolchain: interior filling required an undeclared dependency and surface voxelization exceeded exploratory time limits. Existing sampled files were also non-watertight, so an interior-fill interpretation would require a separate contract.

The selected representation has 32 equal bins along the STL's existing Z axis, normalized between its minimum and maximum vertex Z coordinates. Each triangle is assigned to the bin containing its centroid. Two channels are accumulated and independently normalized to sum to one:

1. triangle surface area;
2. absolute triangle area projected onto the XY plane.

The extractor loads with trimesh 4.10.1 and `process=False`. It does not rotate, scale, repair, fill, merge, or reslice geometry. Uniform scaling and translation do not change the normalized result. Absolute area terms make it insensitive to winding reversal. Open and overlapping shells are accepted as additive triangle surfaces. The representation is not occupied volume and does not claim to approximate cross-sectional area.

A mesh blocks extraction if it is empty; contains invalid indices or non-finite geometry; has no positive Z extent, surface area, or absolute projected area; or produces a non-finite channel. Every declared development STL must produce the same 64 finite values. Partial-row omission and mixed feature semantics are forbidden.

## Private inventory contract

The run accepts one ignored private JSON manifest with this shape:

```json
{
  "version": "minires-development-geometry-inventory-v1",
  "scope": "pre_supported_training_and_validation_only",
  "presupported_scope_confirmed": true,
  "held_out_test_geometry_included": false,
  "root": "/private/path/to/geometry",
  "expected_stl_count": 1,
  "development_rows": [
    {
      "partition": "training",
      "row_index": 0,
      "relative_stl_path": "private-relative-name.stl"
    }
  ],
  "reconciliation": {
    "existing_presupported_stl_count": 1931,
    "training_row_count": 1,
    "validation_row_count": 0,
    "held_out_test_stl_count": 100,
    "noncanonical_presupported_stl_count": 1830,
    "missing_development_geometry_count": 0,
    "duplicate_development_geometry_count": 0,
    "development_rows_one_to_one": true
  }
}
```

The manifest is private because its root, row mapping, and relative paths may disclose identity. Its scope fields and reconciliation are maintainer attestations, not facts inferred from mesh geometry. `development_rows` must contain between 1 and 2,000 mappings to unique, existing, non-symlinked `.stl` paths beneath the root. Raw-string and resolved-path aliases are rejected.

Training and validation row indexes must each be unique and contiguous from zero through their declared count, making missing or duplicate development-row mappings invalid. The two row counts must sum to `expected_stl_count`. Training, validation, held-out-test, and noncanonical counts must also reconcile exactly to the frozen existing pre-supported count of 1,931. Missing or duplicate development geometry must be zero, and the one-to-one attestation must be true. The held-out count is aggregate reconciliation only; its paths are neither listed nor accessed.

The corrected exploratory count of 1,931 pre-supported files is not by itself an eligible inventory. Before execution, private reconciliation must prove exhaustive one-to-one coverage of the training and validation rows while excluding held-out-test geometry. If that boundary cannot be established without accessing held-out-test geometry, the run remains blocked.

## Dataset isolation

The dataset used for Issue 50 remains immutable. Geometry-feature work must not add columns to, replace, repartition, or write files into the existing canonical training, validation, or held-out-test artifact set.

Run 001 reads the separately inventoried existing raw pre-supported STL collection and writes only aggregate feasibility evidence under `private/geometry-feature-feasibility/run-001`. It does not create model-ready rows.

If additional raw STL files are restored or downloaded later, they must be stored under a different ignored private root with their own immutable acquisition inventory. Do not append them to or copy them into the existing raw-STL root. A later reconciliation manifest may reference both inventories privately, but it must preserve their separate corpus identities and prove path and row disjointness.

Any eventual model-ready geometry features must be written as a new versioned private dataset under a geometry-feature-specific root. It may bind to immutable canonical partition rows by private partition and row index, but it must not modify the canonical records. Training and validation feature artifacts remain separate from held-out-test feature artifacts; the latter must not be generated or accessed during development. Combining tabular and geometry features is a later modeling input operation, not a dataset rewrite.

## Synthetic contract checks

Before private geometry, source-neutral tests freeze repeat determinism; winding, translation, and positive uniform-scale invariance; exact centroid behavior at normalized bin boundaries; chunk invariance; open and overlapping-shell semantics; empty, degenerate, non-finite, zero-Z-extent, and zero-projected-area rejection; real subprocess extraction; observed resident-memory blocking; and whole-run deadline accounting.

## Frozen resource and stop contract

The create-only feasibility run uses:

- one worker at a time;
- 100,000 faces per numerical accumulation chunk;
- 120 seconds per STL;
- 8 GiB maximum observed worker resident memory, sampled every 0.25 seconds and checked against the worker's terminal peak;
- 14,400 seconds maximum elapsed time;
- zero retries, substitutions, omissions, or budget recycling.

The command is:

```bash
python3 -m minires.preparation.surface_signature_feasibility \
  --inventory-manifest private/geometry-feature-feasibility/development-inventory.json \
  --authorization-record private/geometry-feature-feasibility/execution-authorization.json \
  --output-root private/geometry-feature-feasibility/run-001
```

Only `private/geometry-feature-feasibility/run-001` is accepted by the command. It is create-only. Inventory reconciliation does not authorize execution. After this predeclaration is committed and reviewed, obtain separate maintainer authorization specifically for Issue 53 surface-signature feasibility run 001. Record it in an ignored private JSON file using version `minires-private-geometry-authorization-v1`, issue `53`, scope `surface_signature_feasibility_run_001`, and `authorized: true`. The command fails closed without that separate record. Do not execute it until both the development-only inventory and authorization record exist.

## Evidence and decision rule

The output contains the frozen plan, aggregate evidence, and artifact checksums. It never persists or prints input paths, identities, fingerprints, per-file feature values, or per-file failures.

The feasibility decision is `completed` only when every declared STL succeeds on its single attempt. Any dependency mismatch, invalid inventory, timeout, resource-limit breach, input change, invalid geometry, worker failure, or incomplete accounting blocks the investigation. A completed result establishes only deterministic mechanical availability over the declared development inventory. It does not establish predictive usefulness, occupied-geometry semantics, validated-scope performance, or permission to inspect held-out evidence.

Downloading the remaining corpus may be considered only after a completed run over the privately reconciled existing development inventory. Any subsequent modeling round needs a separate predeclaration and must apply one frozen feature contract to every training and validation row.
