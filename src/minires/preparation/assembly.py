"""Assemble reused historical measurements and a completed new-STL batch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from ..ingestion import FEATURE_ALIASES, InputError, fingerprint, load_records, number
from ..private_io import create_private_file, is_private_path, write_private_json
from .artifacts import FEATURE_UNITS
from .partitioning import PARTITIONS, PARTITION_ALLOCATION_VERSION, PROPORTIONS, _allocate_source, _install_directory
from .slicing_contract import DENSITY_G_PER_ML, LAYER_HEIGHT_MM, SLICER_ADDED_SUPPORTS

ASSEMBLY_VERSION = "expanded-four-source-dataset-v1"
EXCLUDED_HISTORICAL_SOURCE_ROWS = 34

_CANONICAL_TO_LEGACY = {
    "file_size_kib": "kb",
    "volume_mm3": "volume",
    "surface_area_mm2": "surface_area",
    "bounding_box_x_mm": "bbox_x",
    "bounding_box_y_mm": "bbox_y",
    "bounding_box_z_mm": "bbox_z",
    "bounding_box_volume_mm3": "bbox_area",
    "mesh_mass_at_unit_density": "mass",
    "euler_characteristic": "euler_number",
    "mesh_scale_mm": "scale",
}
_POSITIVE_FIELDS = set(_CANONICAL_TO_LEGACY) - {"euler_characteristic"}
_NON_FEATURE_FIELDS = {
    "_id", "record_identity", "anonymous_source_group", "artist", "source",
    "partition", "duplicate_group", "miniature_family", "sliced_resin_mass_g",
    "weight", "outcome", "preparation_reasons",
}


@dataclass(frozen=True)
class AssemblyResult:
    historical_input_count: int
    excluded_source_count: int
    new_inventory_count: int
    eligible_count: int
    rejected_count: int
    partition_counts: dict[str, int]
    output_dir: Path


def _bounded_number(raw: Mapping[str, Any], canonical: str, legacy: str) -> int | float | None:
    raw_value = raw.get(canonical, raw.get(legacy))
    value = number(raw_value)
    if value is None or not math.isfinite(value):
        return None
    if canonical in _POSITIVE_FIELDS and value <= 0:
        return None
    if canonical == "euler_characteristic" and not value.is_integer():
        return None
    return raw_value if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) else value


def _stable_identity(raw: Mapping[str, Any], *fields: str) -> str | None:
    for field in fields:
        value = raw.get(field)
        if isinstance(value, Mapping) and set(value) == {"$oid"}:
            value = value["$oid"]
        if value is None or isinstance(value, (dict, list, bool)):
            continue
        text = str(value)
        if text:
            return text
    return None


def _harmonize(
    raw: Any,
    *,
    identity: str | None,
    source_alias: str,
    duplicate_group: Any = None,
    allow_zero_target: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "invalid_record"
    if identity is None:
        return None, "unresolved_record_identity"

    values: dict[str, int | float] = {}
    for canonical, legacy in _CANONICAL_TO_LEGACY.items():
        value = _bounded_number(raw, canonical, legacy)
        if value is None:
            return None, "invalid_measurement"
        values[canonical] = value
    target_raw = raw.get("sliced_resin_mass_g", raw.get("weight"))
    target = number(target_raw)
    if (
        target is None
        or not math.isfinite(target)
        or target < 0
        or (target == 0 and not allow_zero_target)
    ):
        return None, "invalid_target"

    ratio = float(values["surface_area_mm2"]) / float(values["volume_mm3"])
    record: dict[str, Any] = {
        "_id": fingerprint([ASSEMBLY_VERSION, "record", str(identity)]),
        "anonymous_source_group": source_alias,
        **values,
        **{legacy: values[canonical] for canonical, legacy in _CANONICAL_TO_LEGACY.items()},
        "surface_to_volume_ratio_per_mm": ratio,
        "surface_volume_ratio": ratio,
        "sliced_resin_mass_g": target_raw if isinstance(target_raw, (int, float)) and not isinstance(target_raw, bool) else target,
        "weight": target_raw if isinstance(target_raw, (int, float)) and not isinstance(target_raw, bool) else target,
        "volume_unit": "mm3",
        "resin_density_g_per_ml": DENSITY_G_PER_ML,
        "scope_confirmed": True,
        "slicing_conditions": {
            "layer_height_mm": LAYER_HEIGHT_MM,
            "slicer_added_supports": SLICER_ADDED_SUPPORTS,
        },
    }
    if isinstance(duplicate_group, str) and duplicate_group:
        record["duplicate_group"] = fingerprint(
            [ASSEMBLY_VERSION, "exact-duplicate", duplicate_group]
        )
    return record, None


def _historical_source(raw: Mapping[str, Any]) -> str | None:
    for field in ("anonymous_source_group", "artist", "source"):
        value = raw.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _read_batch(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        content = Path(path).read_bytes()
        payload = json.loads(content.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise InputError("new_batch_unreadable") from None
    if not isinstance(payload, dict):
        raise InputError("invalid_new_batch")
    return payload, sha256(content).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with create_private_file(path) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))


def assemble_expanded_dataset(
    historical_records: str | Path,
    *,
    excluded_historical_source: str,
    new_batch_result: str | Path,
    output_dir: str | Path,
    seed: int,
) -> AssemblyResult:
    """Atomically replace the current four-source private dataset package."""
    if not isinstance(excluded_historical_source, str) or not excluded_historical_source:
        raise InputError("excluded_source_required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputError("integer_seed_required")
    output = Path(output_dir)
    if not is_private_path(output):
        raise InputError("private_output_directory_required")

    historical, historical_fingerprint = load_records(historical_records)
    batch, batch_fingerprint = _read_batch(new_batch_result)
    accounting = batch.get("accounting")
    accepted_wrappers = batch.get("accepted_records")
    rejected_entries = batch.get("rejected_entries")
    inventory = batch.get("inventory")
    if (
        batch.get("schema_version") != 1
        or batch.get("outcome") != "completed"
        or not isinstance(accounting, Mapping)
        or not isinstance(inventory, list)
        or not isinstance(accepted_wrappers, list)
        or not isinstance(rejected_entries, list)
        or accounting.get("inventory_count") != len(inventory)
        or accounting.get("inventory_count") != len(accepted_wrappers) + len(rejected_entries)
        or accounting.get("accepted_count") != len(accepted_wrappers)
        or accounting.get("rejected_count") != len(rejected_entries)
    ):
        raise InputError("invalid_new_batch_accounting")
    inventory_ids = [
        _stable_identity(entry, "entry_id") if isinstance(entry, Mapping) else None
        for entry in inventory
    ]
    outcome_ids = [
        _stable_identity(entry, "entry_id") if isinstance(entry, Mapping) else None
        for entry in [*accepted_wrappers, *rejected_entries]
    ]
    if (
        any(identity is None for identity in inventory_ids + outcome_ids)
        or len(set(inventory_ids)) != len(inventory_ids)
        or len(set(outcome_ids)) != len(outcome_ids)
        or set(inventory_ids) != set(outcome_ids)
    ):
        raise InputError("invalid_new_inventory_reconciliation")

    eligible_by_source: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    rejected: list[dict[str, Any]] = []
    historical_source_excluded = 0
    retained_historical_sources: set[str] = set()
    identities: set[str] = set()
    retained_paths = Counter(
        raw.get("file") for raw in historical
        if isinstance(raw, Mapping)
        and _historical_source(raw) != excluded_historical_source
        and isinstance(raw.get("file"), str)
        and raw.get("file")
    )

    for index, raw in enumerate(historical):
        source = _historical_source(raw) if isinstance(raw, Mapping) else None
        if source == excluded_historical_source:
            historical_source_excluded += 1
            continue
        source_alias = (
            fingerprint([ASSEMBLY_VERSION, "historical-source", source])
            if source is not None else ""
        )
        identity = _stable_identity(raw, "record_identity", "_id") if isinstance(raw, Mapping) else None
        explicit_duplicate = raw.get("duplicate_group") if isinstance(raw, Mapping) else None
        historical_path = raw.get("file") if isinstance(raw, Mapping) else None
        duplicate_evidence = explicit_duplicate or (
            f"repeated-private-path:{historical_path}"
            if isinstance(historical_path, str) and retained_paths[historical_path] > 1
            else None
        )
        record, reason = _harmonize(
            raw, identity=identity, source_alias=source_alias,
            duplicate_group=duplicate_evidence, allow_zero_target=True,
        )
        if source is None:
            reason = "unresolved_source_group"
            record = None
        if record is None:
            rejected.append({
                "record_identity": fingerprint([ASSEMBLY_VERSION, "historical-rejection", index, identity]),
                "origin": "historical",
                "reason": reason,
            })
            continue
        if record["_id"] in identities:
            raise InputError("duplicate_record_identity")
        identities.add(record["_id"])
        retained_historical_sources.add(source_alias)
        eligible_by_source.setdefault(source_alias, []).append((record["_id"], record))

    if historical_source_excluded != EXCLUDED_HISTORICAL_SOURCE_ROWS:
        raise InputError("excluded_historical_source_count_mismatch")
    if len(retained_historical_sources) != 3:
        raise InputError("retained_historical_source_count_mismatch")

    new_source_alias = fingerprint([ASSEMBLY_VERSION, "new-source"])
    for index, wrapper in enumerate(accepted_wrappers):
        if not isinstance(wrapper, Mapping):
            raise InputError("invalid_new_accepted_record")
        entry_id = _stable_identity(wrapper, "entry_id")
        record, reason = _harmonize(
            wrapper.get("record"), identity=entry_id, source_alias=new_source_alias,
            duplicate_group=wrapper.get("duplicate_group"),
        )
        if record is None:
            rejected.append({
                "record_identity": fingerprint([ASSEMBLY_VERSION, "new-rejection", index, entry_id]),
                "origin": "new",
                "reason": reason,
            })
            continue
        if record["_id"] in identities:
            raise InputError("duplicate_record_identity")
        identities.add(record["_id"])
        eligible_by_source.setdefault(new_source_alias, []).append((record["_id"], record))

    for index, entry in enumerate(rejected_entries):
        entry_id = entry.get("entry_id") if isinstance(entry, Mapping) else None
        reason = entry.get("reason") if isinstance(entry, Mapping) else None
        rejected.append({
            "record_identity": fingerprint([ASSEMBLY_VERSION, "new-batch-rejection", index, entry_id]),
            "origin": "new",
            "reason": reason if isinstance(reason, str) and reason else "invalid_new_rejection",
        })

    if len(eligible_by_source) != 4 or new_source_alias not in eligible_by_source:
        raise InputError("retained_source_count_mismatch")
    duplicate_sources: dict[str, set[str]] = {}
    for source, rows in eligible_by_source.items():
        for _, component_record in rows:
            duplicate = component_record.get("duplicate_group")
            if isinstance(duplicate, str):
                duplicate_sources.setdefault(duplicate, set()).add(source)
    if any(len(sources) > 1 for sources in duplicate_sources.values()):
        raise InputError("duplicate_group_crosses_sources")

    partitions: dict[str, list[Mapping[str, Any]]] = {name: [] for name in PARTITIONS}
    source_partition_counts: dict[str, dict[str, int]] = {}
    for source in sorted(eligible_by_source):
        allocated = _allocate_source(eligible_by_source[source], seed, source)
        source_partition_counts[fingerprint([ASSEMBLY_VERSION, "manifest-source", source])] = {
            name: len(allocated[name]) for name in PARTITIONS
        }
        for name in PARTITIONS:
            partitions[name].extend(allocated[name])

    partition_counts = {name: len(partitions[name]) for name in PARTITIONS}
    eligible_count = sum(partition_counts.values())
    historical_unusable = sum(row["origin"] == "historical" for row in rejected)
    new_rejected = sum(row["origin"] == "new" for row in rejected)
    new_accepted = len(accepted_wrappers) - (new_rejected - len(rejected_entries))
    expected_eligible = len(historical) - historical_source_excluded - historical_unusable + new_accepted
    if eligible_count != expected_eligible:
        raise InputError("dataset_reconciliation_failed")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    os.chmod(staged, 0o700)
    try:
        for name in PARTITIONS:
            _write_jsonl(staged / f"{name}.jsonl", sorted(partitions[name], key=lambda row: str(row["_id"])))
        _write_jsonl(staged / "rejected.jsonl", sorted(rejected, key=lambda row: str(row["record_identity"])))
        provenance = {
            "assembly_version": ASSEMBLY_VERSION,
            "historical_measurements": "reused_as_authoritative_without_mesh_probing_or_slicing",
            "historical_input_fingerprint": historical_fingerprint,
            "new_batch_fingerprint": batch_fingerprint,
            "new_measurement_contract": batch.get("provenance", {}).get("contract") if isinstance(batch.get("provenance"), Mapping) else None,
            "new_tool_versions": batch.get("provenance", {}).get("tool_versions") if isinstance(batch.get("provenance"), Mapping) else None,
        }
        write_private_json(staged / "provenance.json", provenance)
        artifact_digests = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.iterdir()) if path.is_file()
        }
        manifest = {
            "assembly_version": ASSEMBLY_VERSION,
            "allocation_version": PARTITION_ALLOCATION_VERSION,
            "allocation_policy": {
                "within_each_anonymous_source_group": PROPORTIONS,
                "membership_inputs": ["stable_private_record_identity", "seed", "explicit_duplicate_group"],
                "targets_and_geometry_affect_membership": False,
                "miniature_family_required": False,
            },
            "seed": seed,
            "retained_anonymous_source_count": 4,
            "row_accounting": {
                "historical_input": len(historical),
                "historical_source_excluded": historical_source_excluded,
                "historical_unusable": historical_unusable,
                "new_inventory": accounting["inventory_count"],
                "new_accepted": new_accepted,
                "new_rejected": new_rejected,
                "eligible": eligible_count,
                "rejected": len(rejected),
                **partition_counts,
            },
            "source_partition_counts": source_partition_counts,
            "feature_schema": {
                "prediction_features": list(FEATURE_ALIASES),
                "feature_units": FEATURE_UNITS,
                "legacy_compatible_features": list(_CANONICAL_TO_LEGACY.values()) + ["surface_volume_ratio"],
                "excluded_metadata": sorted(_NON_FEATURE_FIELDS),
            },
            "claim_scope": [
                "performance_on_held_out_stl_rows_drawn_from_retained_sources",
                "not_family_independent_evidence",
            ],
            "artifacts": artifact_digests,
        }
        write_private_json(staged / "manifest.json", manifest)
        checksummed = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.iterdir()) if path.is_file()
        }
        write_private_json(staged / "checksums.json", {
            "assembly_version": ASSEMBLY_VERSION,
            "artifacts": checksummed,
        })
        _install_directory(staged, output)
    except InputError:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    except (OSError, TypeError, ValueError):
        if staged.exists():
            shutil.rmtree(staged)
        raise InputError("private_assembly_write_failed") from None

    return AssemblyResult(
        historical_input_count=len(historical),
        excluded_source_count=historical_source_excluded,
        new_inventory_count=accounting["inventory_count"],
        eligible_count=eligible_count,
        rejected_count=len(rejected),
        partition_counts=partition_counts,
        output_dir=output,
    )
