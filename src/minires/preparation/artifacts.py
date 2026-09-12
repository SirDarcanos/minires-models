"""Private, create-only artifacts. PyArrow is needed only when persisting a run."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..ingestion import FEATURE_ALIASES, TRANSFORMATION_VERSION, InputError
from ..private_io import create_private_file, write_private_json

if TYPE_CHECKING:
    from ..evaluation.baseline import EvaluationResult


FEATURE_UNITS = {name: "mm3" if name.endswith("mm3") else "mm2" if name.endswith("mm2")
                 else "mm" if name.endswith("mm") else "dimensionless"
                 for name in FEATURE_ALIASES}


def write_private(result: EvaluationResult, output_dir: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise InputError("parquet_dependency_required") from None
    output = Path(output_dir)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        if output.exists():
            if not output.is_dir() or any(item.name != "fitted-folds" for item in output.iterdir()):
                raise InputError("private_output_directory_unavailable")
        else:
            output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None
    datasets = [[asdict(row) for row in result.canonical_rows]] + [
        report["comparison_rows"] for report in result.reconciliations]
    feature_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for dataset_index, rows in enumerate(datasets):
        for row in rows:
            keys = {"dataset_index": dataset_index, "row_index": row["row_index"]}
            feature_rows.append({**keys, **row["features"]})
            metadata_rows.append({**keys, **row["metadata"],
                                  "sliced_resin_mass_g": row["sliced_resin_mass_g"],
                                  "outcome": row["outcome"], "reasons": row["reasons"],
                                  "warnings": row["warnings"]})
    join_fields = [pa.field("dataset_index", pa.int64()), pa.field("row_index", pa.int64())]
    feature_schema = pa.schema(join_fields + [
        pa.field(name, pa.float64(), metadata={"unit": FEATURE_UNITS[name],
                 "legacy_alias": alias, "missing": "null; see evaluation_metadata reasons"})
        for name, alias in FEATURE_ALIASES.items()
    ], metadata={"transformation_version": TRANSFORMATION_VERSION,
                 "prediction_features": json.dumps(list(FEATURE_ALIASES)),
                 "join_keys_are_not_features": "dataset_index,row_index"})
    metadata_schema = pa.schema(join_fields + [
        pa.field("sliced_resin_mass_g", pa.float64(), metadata={"unit": "g"}),
        pa.field("outcome", pa.string()), pa.field("reasons", pa.list_(pa.string())),
        pa.field("warnings", pa.list_(pa.string())),
        *[pa.field(key, pa.string()) for key in (
            "anonymous_source_group", "miniature_family", "record_identity", "location_evidence",
            "source_name_evidence", "source_identity_evidence", "duplicate_group", "geometry_fingerprint", "volume_unit", "scope_confirmation_origin", "density_origin",
            "slicing_conditions_fingerprint")],
        *[pa.field("legacy_" + key + "_unit_unknown", pa.float64()) for key in ("kb", "mass", "scale")],
        pa.field("base_mm", pa.float64()), pa.field("resin_density_g_per_ml", pa.float64()),
        pa.field("scope_confirmed", pa.bool_()), pa.field("geometry_valid", pa.bool_()),
        pa.field("support_presence", pa.bool_()),
        pa.field("slicing_conditions", pa.struct([
            pa.field(key, pa.float64()) for key in
            ("layer_height_mm", "exposure_seconds", "bottom_exposure_seconds")]))
    ], metadata={"transformation_version": TRANSFORMATION_VERSION,
                 "classification": "private_evaluation_only"})
    try:
        with create_private_file(output / "features.parquet") as stream:
            pq.write_table(pa.Table.from_pylist(feature_rows, schema=feature_schema), stream)
        with create_private_file(output / "evaluation_metadata.parquet") as stream:
            pq.write_table(pa.Table.from_pylist(metadata_rows, schema=metadata_schema), stream)
        write_private_json(output / "report.json", result.to_dict())
        manifest = {
            "transformation_version": TRANSFORMATION_VERSION,
            "run_metadata": asdict(result.run_metadata),
            "pyarrow_version": pa.__version__,
            **({"learned_baseline": {
                "version": result.model_contract["version"],
                "configuration": result.model_contract["run_configuration"],
                "dependency_versions": result.model_contract.get("run", {}).get("dependency_versions", {}),
                "numerical_reproducibility": result.model_contract["numerical_reproducibility"],
            }} if result.model_contract.get("classification") == "clean_fixed_configuration_baseline" else {}),
            "dataset_row_counts": [len(rows) for rows in datasets],
            "feature_units": FEATURE_UNITS,
            "artifacts": {
                str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
            "limitations": ["no_mesh_validation", "support_presence_unknown",
                            *([] if result.model_contract.get("classification") == "clean_fixed_configuration_baseline"
                              else ["no_fitted_transforms"]),
                            *([] if result.split_status == 'frozen_source_holdout' else ['no_held_out_evidence']),
                            "binary64_precision_no_rounding",
                            "hashed_linkage_is_private_not_proof_of_geometry_identity"],
        }
        write_private_json(output / "manifest.json", manifest)
    except (OSError, ValueError, TypeError):
        # A partial run has no successful manifest; preserve it for local diagnosis.
        raise InputError("private_artifact_write_failed") from None
