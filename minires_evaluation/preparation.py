"""Prepare identity-redacted, evidence-grouped records for private evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ingestion import Dataset, InputError, fingerprint, load_records, number, normalize
from .private_io import create_private_file, write_private_json
from .reconciliation import reconcile
from .slicing_contract import (
    DENSITY_G_PER_ML,
    EBMINIMANAGER_REVISION,
    LAYER_HEIGHT_MM,
    PROFILE_RELATIVE_PATH,
    PROFILE_SHA256,
    SLICER_ADDED_SUPPORTS,
)

PREPARATION_VERSION = "private-evaluation-preparation-v1"
GROUPING_VERSION = "private-path-grouping-v1"
PROFILE_PATH = PROFILE_RELATIVE_PATH.as_posix()

_SAFE_NUMERIC_FIELDS = (
    "kb", "volume", "surface_area", "bbox_x", "bbox_y", "bbox_z",
    "bbox_area", "mass", "euler_number", "scale", "surface_volume_ratio",
)


@dataclass(frozen=True)
class PreparationResult:
    input_count: int
    included_count: int
    needs_review_count: int
    excluded_count: int
    source_group_count: int
    miniature_family_count: int
    output_dir: Path


def _private_output(path: Path) -> None:
    if "private" not in path.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None


def _path_parts(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [part for part in value.replace("\\", "/").split("/") if part not in ("", ".")]


def _source_alias(raw: Mapping[str, Any]) -> str | None:
    source = raw.get("artist")
    if not isinstance(source, str) or not source:
        return None
    return fingerprint([GROUPING_VERSION, "source", source])


def _family_alias(raw: Mapping[str, Any], source: str | None) -> tuple[str | None, str]:
    name = raw.get("mini")
    parts = _path_parts(raw.get("file"))
    if source is None or not isinstance(name, str) or not name:
        return None, "unresolved_miniature_family"
    matches = [index for index, part in enumerate(parts[:-1]) if part == name]
    if len(matches) != 1:
        return None, "ambiguous_family_path" if len(matches) > 1 else "unresolved_miniature_family"
    family_directory = parts[:matches[0] + 1]
    return fingerprint([
        GROUPING_VERSION, "family-directory", source, family_directory,
    ]), "unique_exact_directory_in_private_path"


def _safe_numeric(value: Any) -> Any:
    parsed = number(value)
    if parsed is None:
        return None if value in (None, "") else "invalid_numeric_value"
    if not math.isfinite(parsed):
        return str(parsed)
    return value


def _safe_record(
    raw: Any,
    repeated_paths: set[str],
    ambiguous_families: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {"preparation_reasons": ["invalid_record"]}, {
            "source_group": None, "miniature_family": None,
            "duplicate_group": None, "evidence": "invalid_record",
            "preparation_reasons": ["invalid_record"],
        }
    source = _source_alias(raw)
    family, family_evidence = _family_alias(raw, source)
    name = raw.get("mini")
    if source is not None and isinstance(name, str) and (source, name) in ambiguous_families:
        family, family_evidence = None, "ambiguous_family_path"
    reasons = []
    if source is None:
        reasons.append("unresolved_source_group")
    if family is None:
        reasons.append(family_evidence)
    exact_path = raw.get("file") if isinstance(raw.get("file"), str) else None
    duplicate = (
        fingerprint([GROUPING_VERSION, "exact-private-path", exact_path])
        if exact_path and exact_path in repeated_paths else None
    )
    record: dict[str, Any] = {
        field: _safe_numeric(raw.get(field)) for field in _SAFE_NUMERIC_FIELDS
    }
    if number(record["surface_volume_ratio"]) is None:
        surface, volume = number(record["surface_area"]), number(record["volume"])
        if (
            surface is not None and volume is not None
            and math.isfinite(surface) and math.isfinite(volume) and volume > 0
        ):
            record["surface_volume_ratio"] = surface / volume
    record.update({
        "sliced_resin_mass_g": _safe_numeric(raw.get("sliced_resin_mass_g", raw.get("weight"))),
        "volume_unit": "mm3",
        "resin_density_g_per_ml": DENSITY_G_PER_ML,
        "scope_confirmed": True,
        "slicing_conditions": {"layer_height_mm": LAYER_HEIGHT_MM},
        "anonymous_source_group": source,
        "miniature_family": family,
        "duplicate_group": duplicate,
        "_id": fingerprint([GROUPING_VERSION, "record", raw.get("_id")]) if raw.get("_id") is not None else None,
        "preparation_reasons": reasons,
    })
    evidence = {
        "source_group": source,
        "miniature_family": family,
        "duplicate_group": duplicate,
        "evidence": family_evidence,
        "preparation_reasons": reasons,
    }
    return record, evidence


def _provenance() -> dict[str, Any]:
    return {
        "preparation_version": PREPARATION_VERSION,
        "ebminimanager_repository": "SirDarcanos/EBMiniManager",
        "ebminimanager_revision": EBMINIMANAGER_REVISION,
        "profile_path": PROFILE_PATH,
        "profile_sha256": PROFILE_SHA256,
        "resin_density_g_per_ml": DENSITY_G_PER_ML,
        "layer_height_mm": LAYER_HEIGHT_MM,
        "slicer_added_supports": SLICER_ADDED_SUPPORTS,
        "pre_supported_scope_attested": True,
        "label_source": "UVtoolsCmd print-properties WeightG",
        "attestation": "maintainer_confirmed_all_historical_records_were_pre_supported",
    }


def prepare_private_dataset(
    records: Dataset,
    *,
    comparison_records: Sequence[Dataset] = (),
    output_dir: str | Path,
    seed: int = 0,
) -> PreparationResult:
    """Create deterministic private evaluation records without persisting identities."""
    loaded, input_fingerprint = load_records(records)
    input_path = Path(records) if isinstance(records, (str, Path)) else None
    original_bytes = input_path.read_bytes() if input_path is not None else None
    comparison_inputs = [load_records(dataset) for dataset in comparison_records]
    comparison_bytes = [
        Path(dataset).read_bytes() if isinstance(dataset, (str, Path)) else None
        for dataset in comparison_records
    ]
    path_counts: Counter[str] = Counter()
    for raw in loaded:
        path = raw.get("file") if isinstance(raw, Mapping) else None
        if isinstance(path, str) and path:
            path_counts[path] += 1
    repeated_paths = {path for path, count in path_counts.items() if count > 1}
    family_candidates: dict[tuple[str, str], set[str]] = {}
    for raw in loaded:
        if not isinstance(raw, Mapping):
            continue
        source = _source_alias(raw)
        name = raw.get("mini")
        family, _ = _family_alias(raw, source)
        if source is not None and isinstance(name, str) and family is not None:
            family_candidates.setdefault((source, name), set()).add(family)
    ambiguous_families = {
        key for key, families in family_candidates.items() if len(families) > 1
    }
    prepared_and_evidence = [
        _safe_record(raw, repeated_paths, ambiguous_families) for raw in loaded
    ]
    prepared = [item[0] for item in prepared_and_evidence]
    evidence = [dict(row_index=index, **item[1]) for index, item in enumerate(prepared_and_evidence)]
    source_mapping = sorted({
        (
            fingerprint([GROUPING_VERSION, "private-source-evidence", raw.get("artist")]),
            source,
        )
        for raw in loaded if isinstance(raw, Mapping)
        if (source := _source_alias(raw)) is not None
    })

    from .evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records
    config = EvaluationConfig(DENSITY_G_PER_ML, "mm3", True, seed=seed)
    canonical = normalize(prepared, config)
    original_canonical = normalize(loaded, config)
    reconciliations = [
        reconcile(original_canonical, normalize(comparison, config), comparison_fingerprint)
        for comparison, comparison_fingerprint in comparison_inputs
    ]
    output = Path(output_dir)
    _private_output(output)
    try:
        prepared_path = output / "prepared-records.jsonl"
        with create_private_file(prepared_path) as stream:
            for row in prepared:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode())
        write_private_json(output / "grouping-evidence.json", {
            "grouping_version": GROUPING_VERSION,
            "identity_values_persisted": False,
            "rules": {
                "source": "stable_alias_from_private_source_evidence",
                "family": "unique_exact_family_directory_and_pack_path",
                "duplicate": "repeated_exact_private_path_only",
                "equal_features_or_names_alone": "never_grouped",
            },
            "source_mapping": [
                {"private_source_evidence": private_source, "anonymous_source_group": alias}
                for private_source, alias in source_mapping
            ],
            "rows": evidence,
        })
        write_private_json(output / "provenance.json", _provenance())
        write_private_json(output / "reconciliation-report.json", {
            "primary_count": len(loaded),
            "comparison_counts": [len(comparison) for comparison, _ in comparison_inputs],
            "comparisons": reconciliations,
            "dataset_equivalence_claimed": False,
        })
        evaluated = evaluate_records(
            prepared_path,
            config,
            PhysicalBaseline(),
            split_manifest=output / "frozen-splits.json",
        )
        grouped = evaluated.grouped_evaluation or {}
        outcome_reasons = Counter(reason for row in canonical for reason in row.reasons)
        write_private_json(output / "coverage.json", {
            "sufficient_for_frozen_evaluation_folds": evaluated.split_status == "frozen_source_holdout",
            "split_status": evaluated.split_status,
            "blockers": list(evaluated.blockers),
            "eligible_source_count": grouped.get("eligible_source_count", 0),
            "miniature_family_count": len({row.metadata["miniature_family"] for row in canonical if row.metadata["miniature_family"]}),
            "duplicate_group_count": len({row.metadata["duplicate_group"] for row in canonical if row.metadata["duplicate_group"]}),
            "included_count": sum(row.outcome == "included" for row in canonical),
            "needs_review_count": sum(row.outcome == "needs_review" for row in canonical),
            "excluded_count": sum(row.outcome == "excluded" for row in canonical),
            "outcome_reasons": dict(sorted(outcome_reasons.items())),
        })
        input_after = input_path.read_bytes() if input_path is not None else None
        if original_bytes is not None and input_after != original_bytes:
            raise InputError("original_input_changed")
        comparison_after = [
            Path(dataset).read_bytes() if isinstance(dataset, (str, Path)) else None
            for dataset in comparison_records
        ]
        if any(
            before is not None and after != before
            for before, after in zip(comparison_bytes, comparison_after)
        ):
            raise InputError("original_input_changed")
        artifacts = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir()) if path.is_file()
        }
        write_private_json(output / "checksums.json", {
            "preparation_version": PREPARATION_VERSION,
            "input": {
                "fingerprint": input_fingerprint,
                "byte_sha256_before": sha256(original_bytes).hexdigest() if original_bytes is not None else None,
                "byte_sha256_after": sha256(input_after).hexdigest() if input_after is not None else None,
                "unchanged": True,
            },
            "comparisons": [
                {
                    "fingerprint": fingerprint_value,
                    "byte_sha256_before": sha256(before).hexdigest() if before is not None else None,
                    "byte_sha256_after": sha256(after).hexdigest() if after is not None else None,
                    "unchanged": True,
                }
                for (_, fingerprint_value), before, after in zip(
                    comparison_inputs, comparison_bytes, comparison_after
                )
            ],
            "artifacts": artifacts,
        })
    except InputError:
        raise
    except (OSError, TypeError, ValueError):
        raise InputError("private_preparation_write_failed") from None
    return PreparationResult(
        input_count=len(canonical),
        included_count=sum(row.outcome == "included" for row in canonical),
        needs_review_count=sum(row.outcome == "needs_review" for row in canonical),
        excluded_count=sum(row.outcome == "excluded" for row in canonical),
        source_group_count=len({row.metadata["anonymous_source_group"] for row in canonical if row.metadata["anonymous_source_group"]}),
        miniature_family_count=len({row.metadata["miniature_family"] for row in canonical if row.metadata["miniature_family"]}),
        output_dir=output,
    )
