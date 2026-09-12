"""Deterministic source-balanced allocation of private harmonized records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .artifacts import FEATURE_UNITS
from ..ingestion import Dataset, FEATURE_ALIASES, InputError, load_records, fingerprint
from ..private_io import create_private_file, write_private_json

PARTITION_ALLOCATION_VERSION = "source-balanced-70-15-15-v1"
PARTITIONS = ("train", "validation", "test")
PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
_NON_FEATURE_FIELDS = {
    "_id", "record_identity", "anonymous_source_group", "artist", "source",
    "partition", "duplicate_group", "geometry_fingerprint", "miniature_family",
    "sliced_resin_mass_g", "weight", "outcome", "preparation_reasons",
}


@dataclass(frozen=True)
class PartitionResult:
    input_count: int
    eligible_count: int
    excluded_count: int
    partition_counts: dict[str, int]
    output_dir: Path


def _stable_identity(record: Mapping[str, Any]) -> str | None:
    for value in (record.get("record_identity"), record.get("_id")):
        if value is None or isinstance(value, (dict, list, bool)):
            continue
        text = str(value)
        if text:
            return text
    return None


def _source(record: Mapping[str, Any]) -> str | None:
    value = record.get("anonymous_source_group")
    return value if isinstance(value, str) and value else None


def _duplicate(record: Mapping[str, Any]) -> str | None:
    value = record.get("duplicate_group")
    return value if isinstance(value, str) and value else None


def _target_counts(total: int) -> dict[str, int]:
    exact = {name: total * PROPORTIONS[name] for name in PARTITIONS}
    counts = {name: int(exact[name]) for name in PARTITIONS}
    remaining = total - sum(counts.values())
    order = sorted(PARTITIONS, key=lambda name: (-(exact[name] - counts[name]), PARTITIONS.index(name)))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _allocate_source(
    records: Sequence[tuple[str, Mapping[str, Any]]], seed: int, source: str
) -> dict[str, list[Mapping[str, Any]]]:
    components: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for identity, record in records:
        duplicate = _duplicate(record)
        component_key = "duplicate:" + duplicate if duplicate else "record:" + identity
        components.setdefault(component_key, []).append((identity, record))
    ordered = sorted(
        components.items(),
        key=lambda item: (
            -len(item[1]),
            fingerprint([
                PARTITION_ALLOCATION_VERSION,
                seed,
                source,
                sorted(identity for identity, _ in item[1]),
            ]),
        ),
    )
    targets = _target_counts(len(records))
    allocated: dict[str, list[Mapping[str, Any]]] = {name: [] for name in PARTITIONS}
    for _, component_rows in ordered:
        size = len(component_rows)
        destination = min(
            PARTITIONS,
            key=lambda name: (
                abs((len(allocated[name]) + size) - targets[name]) - abs(len(allocated[name]) - targets[name]),
                -max(targets[name] - len(allocated[name]), 0),
                PARTITIONS.index(name),
            ),
        )
        allocated[destination].extend(record for _, record in component_rows)
    return allocated


def _write_dataset(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with create_private_file(path) as stream:
        for row in sorted(rows, key=lambda value: _stable_identity(value) or ""):
            serialized = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
            stream.write(serialized.encode("utf-8"))


def _install_directory(staged: Path, destination: Path) -> None:
    """Atomically switch a stable path to a complete staged directory."""
    if destination.exists() and not destination.is_symlink():
        raise InputError("private_output_directory_unavailable")
    old_target: Path | None = None
    if destination.is_symlink():
        try:
            target = Path(os.readlink(destination))
        except OSError:
            raise InputError("private_output_directory_unavailable") from None
        old_target = target if target.is_absolute() else destination.parent / target
        expected_prefix = "." + destination.name + "-"
        if (
            old_target.parent.resolve() != destination.parent.resolve()
            or not old_target.name.startswith(expected_prefix)
            or not old_target.is_dir()
        ):
            raise InputError("private_output_directory_unavailable")

    temporary_link = staged.with_name(staged.name + ".link")
    try:
        os.symlink(staged.name, temporary_link, target_is_directory=True)
        os.replace(temporary_link, destination)
    except OSError:
        temporary_link.unlink(missing_ok=True)
        raise InputError("private_partition_replacement_failed") from None

    # The new complete set is already current. Stale-set cleanup is best effort
    # and must not turn a committed replacement into a reported failure.
    if old_target is not None:
        shutil.rmtree(old_target, ignore_errors=True)


def partition_private_dataset(
    records: Dataset,
    *,
    excluded_source: str,
    output_dir: str | Path,
    seed: int,
) -> PartitionResult:
    """Replace the current private 70/15/15 artifacts as one complete set."""
    if not isinstance(excluded_source, str) or not excluded_source:
        raise InputError("excluded_source_required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputError("integer_seed_required")
    loaded, input_fingerprint = load_records(records)
    output = Path(output_dir)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")

    eligible_by_source: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    exclusion_reasons: Counter[str] = Counter()
    identities: set[str] = set()
    duplicate_sources: dict[str, set[str]] = {}
    for raw in loaded:
        if not isinstance(raw, Mapping):
            exclusion_reasons["invalid_record"] += 1
            continue
        source = _source(raw)
        if source == excluded_source:
            exclusion_reasons["configured_source"] += 1
            continue
        identity = _stable_identity(raw)
        reasons = raw.get("preparation_reasons")
        outcome = raw.get("outcome")
        if source is None:
            exclusion_reasons["unresolved_source_group"] += 1
        elif identity is None:
            exclusion_reasons["unresolved_record_identity"] += 1
        elif reasons or outcome in {"excluded", "needs_review", "rejected"}:
            exclusion_reasons["ineligible_record"] += 1
        elif identity in identities:
            raise InputError("duplicate_record_identity")
        else:
            identities.add(identity)
            eligible_by_source.setdefault(source, []).append((identity, raw))
            duplicate = _duplicate(raw)
            if duplicate:
                duplicate_sources.setdefault(duplicate, set()).add(source)
    if any(len(sources) > 1 for sources in duplicate_sources.values()):
        raise InputError("duplicate_group_crosses_sources")
    if not eligible_by_source:
        raise InputError("no_eligible_records")

    partitions: dict[str, list[Mapping[str, Any]]] = {name: [] for name in PARTITIONS}
    source_counts: dict[str, dict[str, int]] = {}
    for source in sorted(eligible_by_source):
        allocated = _allocate_source(eligible_by_source[source], seed, source)
        source_key = fingerprint(["anonymous-source-allocation", source])
        source_counts[source_key] = {name: len(allocated[name]) for name in PARTITIONS}
        for name in PARTITIONS:
            partitions[name].extend(allocated[name])

    eligible_count = sum(len(rows) for rows in eligible_by_source.values())
    partition_counts = {name: len(partitions[name]) for name in PARTITIONS}
    if sum(partition_counts.values()) != eligible_count:
        raise InputError("partition_accounting_failed")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    os.chmod(staged, 0o700)
    try:
        for name in PARTITIONS:
            _write_dataset(staged / f"{name}.jsonl", partitions[name])
        artifacts = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.iterdir()) if path.is_file()
        }
        excluded_count = len(loaded) - eligible_count
        manifest = {
            "allocation_version": PARTITION_ALLOCATION_VERSION,
            "allocation_policy": {
                "within_each_anonymous_source_group": PROPORTIONS,
                "membership_inputs": ["stable_private_record_identity", "seed", "explicit_duplicate_group"],
                "targets_and_geometry_affect_membership": False,
                "miniature_family_required": False,
            },
            "input_identity": {"fingerprint": input_fingerprint},
            "seed": seed,
            "row_accounting": {
                "input": len(loaded),
                "excluded": excluded_count,
                "eligible": eligible_count,
                **partition_counts,
            },
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "source_partition_counts": source_counts,
            "feature_schema": {
                "prediction_features": list(FEATURE_ALIASES),
                "feature_units": FEATURE_UNITS,
                "excluded_metadata": sorted(_NON_FEATURE_FIELDS),
            },
            "artifacts": artifacts,
            "limitations": [
                "exact_duplicates_grouped_only_when_explicit_evidence_is_supplied",
                "no_miniature_family_or_near_duplicate_grouping",
                "held_out_rows_are_not_unseen_source_evidence",
            ],
        }
        write_private_json(staged / "manifest.json", manifest)
        _install_directory(staged, output)
    except InputError:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    except (OSError, TypeError, ValueError):
        if staged.exists():
            shutil.rmtree(staged)
        raise InputError("private_partition_write_failed") from None

    return PartitionResult(
        input_count=len(loaded),
        eligible_count=eligible_count,
        excluded_count=len(loaded) - eligible_count,
        partition_counts=partition_counts,
        output_dir=output,
    )
