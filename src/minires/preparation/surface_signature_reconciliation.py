"""Reconcile existing pre-supported STL paths to development partition rows.

This seam reads training and validation identities but accepts no held-out record
artifact. Row-level mappings remain private; durable evidence is aggregate-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..ingestion import InputError, fingerprint
from ..private_io import PrivateArgumentParser, is_private_path, write_private_json
from .assembly import ASSEMBLY_VERSION
from .surface_signature_batch_linkage import (
    VERSION as BATCH_LINKAGE_VERSION,
    _select_top_level,
)
from .surface_signature_feasibility import (
    EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT,
    INVENTORY_VERSION,
    RECONCILIATION_VERSION,
)

VERSION = "minires-surface-signature-reconciliation-v1"
CORPUS_ID = "existing-presupported-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT_ROOT = (
    PROJECT_ROOT / "private" / "geometry-feature-feasibility" / "reconciliation-001"
)


@dataclass(frozen=True)
class SurfaceSignatureReconciliationResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]


def _plan() -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": "existing_presupported_development_geometry_reconciliation",
        "corpus": {
            "id": CORPUS_ID,
            "expected_stl_count": EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT,
            "must_be_separate_from_canonical_dataset": True,
            "future_restored_corpus_must_use_another_root": True,
        },
        "partition_inputs": ["training_records", "validation_records"],
        "held_out_test_records": "unavailable_to_this_interface",
        "row_linkage": "assembly_record_identity_from_checksum_bound_identity_only_batch_linkage",
        "geometry_access": "mapped_development_files_stat_only_without_mesh_loading",
        "held_out_geometry": "path_metadata_used_only_for_exclusion_without_filesystem_access",
        "target_access": "record_payloads_unavailable_to_this_interface",
        "output": "private_one_to_one_development_row_mapping_and_aggregate_evidence",
        "stop_rule": "one_create_only_attempt_without_repair_omission_or_inference",
    }


def _read_json(path: Path) -> Any:
    if not is_private_path(path):
        raise ValueError("private input required")
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid private input") from None


def _partition_identities(path: Path) -> list[str]:
    if not is_private_path(path):
        raise ValueError("private input required")
    identities: list[str] = []
    try:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                identity = row.get("_id") if isinstance(row, Mapping) else None
                if not isinstance(identity, str) or not identity:
                    raise ValueError("invalid partition")
                identities.append(identity)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid partition") from None
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate partition identity")
    return identities


def _overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _validated_dataset(
    training: Path,
    validation: Path,
    manifest_path: Path,
    provenance_path: Path,
) -> tuple[list[str], list[str], Mapping[str, Any], str]:
    if (
        training.parent.resolve() != validation.parent.resolve()
        or manifest_path.parent.resolve() != training.parent.resolve()
        or provenance_path.parent.resolve() != training.parent.resolve()
    ):
        raise ValueError("partition roots differ")
    manifest = _read_json(manifest_path)
    provenance = _read_json(provenance_path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("assembly_version") != ASSEMBLY_VERSION
        or not isinstance(provenance, Mapping)
        or provenance.get("assembly_version") != ASSEMBLY_VERSION
    ):
        raise ValueError("invalid dataset manifest")
    artifacts = manifest.get("artifacts")
    accounting = manifest.get("row_accounting")
    if not isinstance(artifacts, Mapping) or not isinstance(accounting, Mapping):
        raise ValueError("invalid dataset manifest")
    for path in (training, validation, provenance_path):
        if artifacts.get(path.name) != sha256(path.read_bytes()).hexdigest():
            raise ValueError("partition checksum mismatch")
    batch_fingerprint = provenance.get("new_batch_fingerprint")
    if not isinstance(batch_fingerprint, str) or len(batch_fingerprint) != 64:
        raise ValueError("batch provenance unavailable")
    training_ids = _partition_identities(training)
    validation_ids = _partition_identities(validation)
    if (
        set(training_ids) & set(validation_ids)
        or accounting.get("train") != len(training_ids)
        or accounting.get("validation") != len(validation_ids)
        or not isinstance(accounting.get("test"), int)
        or accounting.get("eligible")
        != len(training_ids) + len(validation_ids) + accounting["test"]
    ):
        raise ValueError("invalid partition accounting")
    return training_ids, validation_ids, accounting, batch_fingerprint


def _validated_linkage(
    path: Path,
    batch_result_path: Path,
    expected_batch_fingerprint: str,
) -> tuple[dict[str, Mapping[str, Any]], int, int]:
    linkage = _read_json(path)
    if not is_private_path(batch_result_path):
        raise ValueError("private batch result required")
    try:
        batch_bytes = batch_result_path.read_bytes()
        selected = _select_top_level(batch_bytes)
    except (OSError, ValueError):
        raise ValueError("invalid source batch") from None
    if not isinstance(linkage, Mapping):
        raise ValueError("invalid linkage")
    inventory = linkage.get("inventory")
    accounting = linkage.get("accounting")
    accepted_count = accounting.get("accepted_count") if isinstance(accounting, Mapping) else None
    rejected_count = accounting.get("rejected_count") if isinstance(accounting, Mapping) else None
    if (
        linkage.get("version") != BATCH_LINKAGE_VERSION
        or sha256(batch_bytes).hexdigest() != expected_batch_fingerprint
        or linkage.get("source_batch_sha256") != expected_batch_fingerprint
        or linkage.get("schema_version") != selected.get("schema_version")
        or linkage.get("outcome") != selected.get("outcome")
        or linkage.get("accounting") != selected.get("accounting")
        or linkage.get("inventory") != selected.get("inventory")
        or linkage.get("schema_version") != 1
        or linkage.get("outcome") != "completed"
        or linkage.get("record_payloads_decoded") is not False
        or linkage.get("labels_decoded_or_inspected") is not False
        or not isinstance(inventory, list)
        or not isinstance(accounting, Mapping)
        or len(inventory) != EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
        or accounting.get("inventory_count") != len(inventory)
        or not isinstance(accepted_count, int)
        or isinstance(accepted_count, bool)
        or not isinstance(rejected_count, int)
        or isinstance(rejected_count, bool)
        or accepted_count < 0
        or rejected_count < 0
        or accepted_count + rejected_count != len(inventory)
    ):
        raise ValueError("invalid linkage")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in inventory:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid linkage inventory")
        entry_id = entry.get("entry_id")
        relative = entry.get("relative_path")
        byte_count = entry.get("byte_count")
        digest = entry.get("sha256")
        if (
            not isinstance(entry_id, str)
            or not isinstance(relative, str)
            or not relative
            or entry_id != sha256(relative.encode("utf-8")).hexdigest()
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or entry_id in entries
        ):
            raise ValueError("invalid linkage inventory")
        entries[entry_id] = entry
    return entries, accepted_count, rejected_count


def _validate_development_geometry(
    root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    development_entry_ids: set[str],
) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir() or not development_entry_ids:
        raise ValueError("geometry root unavailable")
    mapped: dict[str, str] = {}
    resolved_paths: set[Path] = set()
    for entry_id in sorted(development_entry_ids):
        entry = entries[entry_id]
        candidate = Path(str(entry["relative_path"]))
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".stl":
            raise ValueError("invalid geometry path")
        unresolved = root / candidate
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError("invalid geometry path") from None
        if (
            Path(os.path.abspath(unresolved)) != resolved
            or resolved in resolved_paths
            or not resolved.is_file()
            or resolved.stat().st_size != entry["byte_count"]
        ):
            raise ValueError("geometry inventory mismatch")
        mapped[entry_id] = candidate.as_posix()
        resolved_paths.add(resolved)
    return mapped


def _artifact_manifest(output: Path, artifacts: Sequence[Path]) -> None:
    write_private_json(output / "manifest.json", {
        "version": VERSION,
        "create_only": True,
        "artifacts": {
            artifact.name: sha256(artifact.read_bytes()).hexdigest()
            for artifact in artifacts
        },
        "held_out_test_records_accessed": False,
        "mesh_geometry_loaded": False,
        "paths_or_identities_published": False,
    })


def reconcile_surface_signature_inventory(
    *,
    training_records: str | Path,
    validation_records: str | Path,
    dataset_manifest: str | Path,
    dataset_provenance: str | Path,
    batch_result: str | Path,
    batch_linkage: str | Path,
    presupported_root: str | Path,
    output_root: str | Path,
    scope_confirmed: bool,
) -> SurfaceSignatureReconciliationResult:
    """Create a development-only STL mapping without reading held-out records."""
    output = Path(output_root)
    if not is_private_path(output):
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None
    plan_path = output / "reconciliation-plan.json"
    write_private_json(plan_path, _plan())

    blockers: tuple[str, ...] = ()
    development_rows: list[dict[str, Any]] = []
    held_out_count = 0
    rejected_count = 0
    training_count = 0
    validation_count = 0
    development_partitions_accessed = False
    linkage_validated = False
    development_geometry_validated = False
    training = Path(training_records)
    validation = Path(validation_records)
    geometry_root = Path(presupported_root)
    if scope_confirmed is not True:
        blockers = ("presupported_scope_confirmation_required",)
    elif _overlap(geometry_root, training.parent) or _overlap(geometry_root, validation.parent):
        blockers = ("geometry_corpus_not_separate",)
    else:
        try:
            (
                training_ids,
                validation_ids,
                dataset_accounting,
                batch_fingerprint,
            ) = _validated_dataset(
                training,
                validation,
                Path(dataset_manifest),
                Path(dataset_provenance),
            )
            development_partitions_accessed = True
            entries, accepted_count, rejected_count = _validated_linkage(
                Path(batch_linkage), Path(batch_result), batch_fingerprint
            )
            linkage_validated = True
            if (
                dataset_accounting.get("new_inventory")
                != EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
                or dataset_accounting.get("new_accepted") != accepted_count
                or dataset_accounting.get("new_rejected") != rejected_count
            ):
                raise ValueError("dataset and linkage disagree")
        except (OSError, TypeError, ValueError):
            blockers = ("development_partition_reconciliation_failed",)
        matched_rows: list[tuple[str, int, str]] = []
        matched_entries: set[str] = set()
        if not blockers:
            canonical_to_entry = {
                fingerprint([ASSEMBLY_VERSION, "record", entry_id]): entry_id
                for entry_id in entries
            }
            for partition, identities in (
                ("training", training_ids), ("validation", validation_ids)
            ):
                for row_index, identity in enumerate(identities):
                    entry_id = canonical_to_entry.get(identity)
                    if entry_id is None:
                        continue
                    if entry_id in matched_entries:
                        blockers = ("development_partition_reconciliation_failed",)
                        break
                    matched_entries.add(entry_id)
                    matched_rows.append((partition, row_index, entry_id))
                if blockers:
                    break
            training_count = sum(row[0] == "training" for row in matched_rows)
            validation_count = sum(row[0] == "validation" for row in matched_rows)
            held_out_count = accepted_count - len(matched_entries)
            if (
                not matched_rows
                or held_out_count < 0
                or len(matched_rows) + held_out_count + rejected_count
                != EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
            ):
                blockers = ("development_partition_reconciliation_failed",)
        if not blockers:
            try:
                relative_by_entry = _validate_development_geometry(
                    geometry_root, entries, matched_entries
                )
                development_geometry_validated = True
            except (OSError, TypeError, ValueError):
                blockers = ("existing_geometry_inventory_mismatch",)
        if not blockers:
            development_rows = [
                {
                    "partition": partition,
                    "partition_row_index": row_index,
                    "relative_stl_path": relative_by_entry[entry_id],
                }
                for partition, row_index, entry_id in matched_rows
            ]

    status = "completed" if not blockers else "blocked"
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "existing_presupported_stl_count": (
            EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
            if linkage_validated else 0
        ),
        "training_stl_count": training_count if not blockers else 0,
        "validation_stl_count": validation_count if not blockers else 0,
        "held_out_test_stl_count": held_out_count if not blockers else 0,
        "noncanonical_presupported_stl_count": rejected_count if not blockers else 0,
        "training_records_accessed": development_partitions_accessed,
        "validation_records_accessed": development_partitions_accessed,
        "held_out_test_records_accessed": False,
        "held_out_test_path_metadata_used_for_exclusion": linkage_validated,
        "held_out_test_geometry_accessed": False,
        "development_geometry_stat_checked": development_geometry_validated,
        "mesh_geometry_loaded": False,
        "batch_bytes_hashed": linkage_validated,
        "batch_record_payloads_decoded": False,
        "record_payloads_available": False,
        "labels_inspected_or_used": False,
        "canonical_dataset_modified": False,
        "geometry_corpus_modified": False,
        "paths_or_identities_published": False,
    }
    artifacts = [plan_path]
    if status == "completed":
        inventory_path = output / "development-inventory.json"
        write_private_json(inventory_path, {
            "version": INVENTORY_VERSION,
            "scope": "pre_supported_training_and_validation_only",
            "presupported_scope_confirmed": True,
            "held_out_test_geometry_included": False,
            "corpus": {
                "id": CORPUS_ID,
                "separate_from_canonical_dataset": True,
            },
            "root": str(geometry_root.resolve()),
            "expected_stl_count": len(development_rows),
            "development_rows": development_rows,
            "reconciliation": {
                "version": RECONCILIATION_VERSION,
                "existing_presupported_stl_count": EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT,
                "training_row_count": training_count,
                "validation_row_count": validation_count,
                "held_out_test_stl_count": held_out_count,
                "noncanonical_presupported_stl_count": rejected_count,
                "missing_development_geometry_count": 0,
                "duplicate_development_geometry_count": 0,
                "development_rows_one_to_one": True,
            },
        })
        artifacts.append(inventory_path)
    evidence_path = output / "reconciliation-evidence.json"
    write_private_json(evidence_path, evidence)
    artifacts.append(evidence_path)
    _artifact_manifest(output, artifacts)
    return SurfaceSignatureReconciliationResult(status, blockers, evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Reconcile existing pre-supported geometry to development rows."
    )
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--validation-records", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-provenance", required=True, type=Path)
    parser.add_argument("--batch-result", required=True, type=Path)
    parser.add_argument("--batch-linkage", required=True, type=Path)
    parser.add_argument("--presupported-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("surface_signature_reconciliation_output_root_mismatch")
        result = reconcile_surface_signature_inventory(
            training_records=args.training_records,
            validation_records=args.validation_records,
            dataset_manifest=args.dataset_manifest,
            dataset_provenance=args.dataset_provenance,
            batch_result=args.batch_result,
            batch_linkage=args.batch_linkage,
            presupported_root=args.presupported_root,
            output_root=args.output_root,
            scope_confirmed=args.scope_confirmed,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SystemExit("surface_signature_reconciliation_failed") from None
    print(json.dumps({
        "status": result.status,
        "blockers": result.blockers,
        "training_stl_count": result.evidence["training_stl_count"],
        "validation_stl_count": result.evidence["validation_stl_count"],
        "held_out_test_stl_count": result.evidence["held_out_test_stl_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
