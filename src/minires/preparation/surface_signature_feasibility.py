"""Bounded aggregate-only feasibility run for print-axis surface signatures."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from ..ingestion import InputError
from ..private_io import PrivateArgumentParser, is_private_path, write_private_json
from . import surface_signature

VERSION = "minires-surface-signature-feasibility-v1"
INVENTORY_VERSION = "minires-development-geometry-inventory-v1"
AUTHORIZATION_VERSION = "minires-private-geometry-authorization-v1"
RECONCILIATION_VERSION = "minires-development-geometry-reconciliation-v1"
RECONCILIATION_PACKAGE_VERSION = "minires-surface-signature-reconciliation-v1"
EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT = 1_931
EXPECTED_TRIMESH_VERSION = "4.10.1"
MAXIMUM_STL_COUNT = 2_000
PER_FILE_TIMEOUT_SECONDS = 120.0
MAXIMUM_ELAPSED_SECONDS = 14_400.0
MAXIMUM_WORKER_RESIDENT_BYTES = 8 * 1024 ** 3
RESOURCE_SAMPLE_SECONDS = 0.25
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT_ROOT = (
    PROJECT_ROOT / "private" / "geometry-feature-feasibility" / "run-001"
)
_ALLOWED_FAILURES = {
    "surface_signature_dependency_unavailable",
    "surface_signature_invalid_geometry",
    "surface_signature_timeout",
    "surface_signature_worker_failed",
    "surface_signature_input_changed",
    "surface_signature_resource_limit",
    "surface_signature_resource_monitor_unavailable",
    "surface_signature_total_deadline",
}


@dataclass(frozen=True)
class SurfaceSignatureFeasibilityResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]


def _trimesh_version() -> str:
    try:
        return version("trimesh")
    except PackageNotFoundError:
        return "unavailable"


def _plan(observed_version: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": "pre_supported_development_geometry_surface_signature_feasibility",
        "hypothesis": (
            "a fixed winding-insensitive additive triangle-surface signature can be "
            "extracted from every declared development STL within bounded resources"
        ),
        "representation": {
            "version": surface_signature.VERSION,
            "bin_count": surface_signature.BIN_COUNT,
            "print_axis": "input_z_axis_without_rotation",
            "bin_assignment": "triangle_centroid_normalized_between_vertex_z_extents",
            "channels": [
                "normalized_triangle_surface_area",
                "normalized_absolute_xy_projected_triangle_area",
            ],
            "semantics": surface_signature.SEMANTICS,
            "occupied_volume_claim": False,
            "cross_sectional_area_claim": False,
            "overlapping_shells": "additive",
            "winding": "absolute_area_terms_are_winding_insensitive",
            "non_watertight_meshes": "supported_without_repair_or_interior_inference",
            "mesh_loading": "trimesh_load_process_false",
            "face_chunk_size": surface_signature.FACE_CHUNK_SIZE,
        },
        "required_dependency": {"trimesh": EXPECTED_TRIMESH_VERSION},
        "observed_dependency": {"trimesh": observed_version},
        "inventory_contract": {
            "version": INVENTORY_VERSION,
            "scope": "pre_supported_training_and_validation_only",
            "held_out_test_geometry_included": False,
            "existing_presupported_stl_count": EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT,
            "maximum_stl_count": MAXIMUM_STL_COUNT,
            "one_to_one_development_reconciliation_required": True,
            "all_declared_rows_must_succeed": True,
        },
        "resource_limits": {
            "per_file_timeout_seconds": PER_FILE_TIMEOUT_SECONDS,
            "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS,
            "maximum_worker_resident_bytes": MAXIMUM_WORKER_RESIDENT_BYTES,
            "resource_sample_seconds": RESOURCE_SAMPLE_SECONDS,
            "retry_count": 0,
        },
        "evidence_policy": "aggregate_counts_only_without_paths_identities_values_or_fingerprints",
        "model_fits": 0,
        "labels": "unavailable_to_this_interface",
        "source_groups": "unavailable_to_this_interface",
        "held_out_test_geometry": "forbidden_by_inventory_contract",
        "execution_authorization": {
            "version": AUTHORIZATION_VERSION,
            "issue": 53,
            "scope": "surface_signature_feasibility_run_001",
            "separate_private_record_required": True,
        },
        "stop_rule": "one_attempt_without_retry_omission_modeling_or_automatic_continuation",
    }


def _load_authorization(path: Path) -> None:
    if not is_private_path(path):
        raise ValueError("invalid authorization")
    try:
        raw = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid authorization") from None
    if (
        not isinstance(raw, Mapping)
        or raw.get("version") != AUTHORIZATION_VERSION
        or raw.get("issue") != 53
        or raw.get("scope") != "surface_signature_feasibility_run_001"
        or raw.get("authorized") is not True
    ):
        raise ValueError("invalid authorization")


def _load_inventory(path: Path) -> tuple[Path, ...]:
    if not is_private_path(path):
        raise ValueError("invalid inventory")
    try:
        content = path.read_bytes()
        package = json.loads(path.with_name("manifest.json").read_text())
        if (
            not isinstance(package, Mapping)
            or package.get("version") != RECONCILIATION_PACKAGE_VERSION
            or not isinstance(package.get("artifacts"), Mapping)
            or package["artifacts"].get(path.name) != sha256(content).hexdigest()
        ):
            raise ValueError("invalid inventory package")
        raw = json.loads(content)
        root = Path(raw["root"]).resolve()
        development_rows = raw["development_rows"]
        expected = raw["expected_stl_count"]
        corpus = raw["corpus"]
        reconciliation = raw["reconciliation"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid inventory") from None
    if (
        not isinstance(raw, Mapping)
        or raw.get("version") != INVENTORY_VERSION
        or raw.get("scope") != "pre_supported_training_and_validation_only"
        or raw.get("presupported_scope_confirmed") is not True
        or raw.get("held_out_test_geometry_included") is not False
        or not isinstance(expected, int) or isinstance(expected, bool)
        or expected <= 0 or expected > MAXIMUM_STL_COUNT
        or not isinstance(development_rows, list) or len(development_rows) != expected
        or not root.is_dir()
        or not isinstance(corpus, Mapping)
        or corpus.get("id") != "existing-presupported-v1"
        or corpus.get("separate_from_canonical_dataset") is not True
        or not isinstance(reconciliation, Mapping)
    ):
        raise ValueError("invalid inventory")
    count_fields = (
        "existing_presupported_stl_count",
        "training_row_count",
        "validation_row_count",
        "held_out_test_stl_count",
        "noncanonical_presupported_stl_count",
        "missing_development_geometry_count",
        "duplicate_development_geometry_count",
    )
    counts = tuple(reconciliation.get(field) for field in count_fields)
    if (
        any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
        or reconciliation.get("version") != RECONCILIATION_VERSION
        or reconciliation.get("existing_presupported_stl_count")
        != EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
        or reconciliation.get("training_row_count", 0)
        + reconciliation.get("validation_row_count", 0) != expected
        or reconciliation.get("training_row_count", 0)
        + reconciliation.get("validation_row_count", 0)
        + reconciliation.get("held_out_test_stl_count", 0)
        + reconciliation.get("noncanonical_presupported_stl_count", 0)
        != EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT
        or reconciliation.get("missing_development_geometry_count") != 0
        or reconciliation.get("duplicate_development_geometry_count") != 0
        or reconciliation.get("development_rows_one_to_one") is not True
    ):
        raise ValueError("invalid inventory")
    paths: list[Path] = []
    seen_items: set[str] = set()
    seen_paths: set[Path] = set()
    seen_rows: set[tuple[str, int]] = set()
    for row in development_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid inventory")
        partition = row.get("partition")
        row_index = row.get("partition_row_index")
        item = row.get("relative_stl_path")
        if (
            not isinstance(partition, str)
            or partition not in {"training", "validation"}
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
            or not isinstance(item, str)
            or not item
            or item in seen_items
        ):
            raise ValueError("invalid inventory")
        row_key = (partition, row_index)
        if row_key in seen_rows:
            raise ValueError("invalid inventory")
        candidate = Path(item)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".stl":
            raise ValueError("invalid inventory")
        unresolved = root / candidate
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError("invalid inventory") from None
        if (
            not resolved.is_file()
            or Path(os.path.abspath(unresolved)) != resolved
            or resolved in seen_paths
        ):
            raise ValueError("invalid inventory")
        seen_rows.add(row_key)
        seen_items.add(item)
        seen_paths.add(resolved)
        paths.append(resolved)
    return tuple(paths)


def _worker(path: str, queue: Any) -> None:
    try:
        signature = surface_signature.extract(path)
        valid = (
            len(signature.surface_area_by_normalized_z) == surface_signature.BIN_COUNT
            and len(signature.absolute_xy_projected_area_by_normalized_z)
            == surface_signature.BIN_COUNT
            and signature.semantics == surface_signature.SEMANTICS
        )
        failure = None if valid else "surface_signature_invalid_geometry"
    except ImportError:
        failure = "surface_signature_dependency_unavailable"
    except MemoryError:
        failure = "surface_signature_resource_limit"
    except Exception:
        failure = "surface_signature_invalid_geometry"
    queue.put((failure, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _bounded_extract(
    path: Path, timeout_seconds: float = PER_FILE_TIMEOUT_SECONDS
) -> str | None:
    try:
        before = (path.stat().st_size, path.stat().st_mtime_ns)
    except OSError:
        return "surface_signature_input_changed"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(str(path), queue))
    process.start()
    deadline = time.monotonic() + timeout_seconds
    while process.is_alive():
        if time.monotonic() >= deadline:
            process.terminate()
            process.join()
            return "surface_signature_timeout"
        try:
            sampled = subprocess.run(
                ("ps", "-o", "rss=", "-p", str(process.pid)),
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            resident_bytes = int(sampled.stdout.strip()) * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            if not process.is_alive():
                break
            process.terminate()
            process.join()
            return "surface_signature_resource_monitor_unavailable"
        if resident_bytes > MAXIMUM_WORKER_RESIDENT_BYTES:
            process.terminate()
            process.join()
            return "surface_signature_resource_limit"
        process.join(RESOURCE_SAMPLE_SECONDS)
    try:
        result, peak_rss = queue.get(timeout=1.0)
        after = (path.stat().st_size, path.stat().st_mtime_ns)
    except Exception:
        return "surface_signature_worker_failed"
    peak_bytes = int(peak_rss) if sys.platform == "darwin" else int(peak_rss) * 1024
    if peak_bytes > MAXIMUM_WORKER_RESIDENT_BYTES:
        return "surface_signature_resource_limit"
    if before != after:
        return "surface_signature_input_changed"
    if result is None:
        return None
    return result if result in _ALLOWED_FAILURES else "surface_signature_worker_failed"


def run_surface_signature_feasibility(
    *,
    inventory_manifest: str | Path,
    authorization_record: str | Path,
    output_root: str | Path,
) -> SurfaceSignatureFeasibilityResult:
    """Process one attested development-only inventory and emit aggregate evidence."""
    output = Path(output_root)
    if not is_private_path(output):
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    observed_version = _trimesh_version()
    plan = _plan(observed_version)
    write_private_json(output / "investigation-plan.json", plan)
    blockers: tuple[str, ...] = ()
    paths: tuple[Path, ...] = ()
    if observed_version == "unavailable":
        blockers = ("surface_signature_dependency_unavailable",)
    elif observed_version != EXPECTED_TRIMESH_VERSION:
        blockers = ("surface_signature_dependency_mismatch",)
    else:
        try:
            _load_authorization(Path(authorization_record))
        except ValueError:
            blockers = ("private_geometry_execution_not_authorized",)
        if not blockers:
            try:
                paths = _load_inventory(Path(inventory_manifest))
            except ValueError:
                blockers = ("invalid_development_geometry_inventory",)

    started = time.monotonic()
    failures: Counter[str] = Counter()
    completed = 0
    unattempted = 0
    if not blockers:
        for index, path in enumerate(paths):
            remaining = MAXIMUM_ELAPSED_SECONDS - (time.monotonic() - started)
            if remaining <= 0.0:
                failures["surface_signature_total_deadline"] += 1
                unattempted = len(paths) - index
                break
            failure = _bounded_extract(
                path, timeout_seconds=min(PER_FILE_TIMEOUT_SECONDS, remaining)
            )
            if failure is None:
                completed += 1
            else:
                if failure == "surface_signature_timeout" and remaining < PER_FILE_TIMEOUT_SECONDS:
                    failure = "surface_signature_total_deadline"
                failures[failure] += 1
        if failures or completed != len(paths):
            blockers = ("surface_signature_extraction_incomplete",)

    elapsed = max(0.0, time.monotonic() - started)
    failed = max(0, len(paths) - completed - unattempted)
    status = "completed" if not blockers else "blocked"
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "feature_contract": {
            "version": surface_signature.VERSION,
            "bin_count": surface_signature.BIN_COUNT,
            "channels": [
                "surface_area_by_normalized_z",
                "absolute_xy_projected_area_by_normalized_z",
            ],
            "semantics": surface_signature.SEMANTICS,
        },
        "input_stl_count": len(paths),
        "attempted_stl_count": completed + failed,
        "completed_stl_count": completed,
        "failed_stl_count": failed,
        "failure_reasons": dict(sorted(failures.items())),
        "unattempted_stl_count": unattempted,
        "elapsed_seconds": elapsed,
        "process_peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_fits": 0,
        "labels_accessed": False,
        "source_groups_used": False,
        "held_out_test_geometry_accessed": False,
        "rows_omitted": 0,
        "candidate_selected": False,
        "lock_created": False,
        "publication_performed": False,
    }
    write_private_json(output / "surface-signature-evidence.json", evidence)
    artifacts = (output / "investigation-plan.json", output / "surface-signature-evidence.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION,
        "create_only": True,
        "artifacts": {
            artifact.name: sha256(artifact.read_bytes()).hexdigest()
            for artifact in artifacts
        },
        "publication_performed": False,
        "paths_or_identities_persisted": False,
        "feature_values_persisted": False,
        "held_out_test_geometry_accessed": False,
    })
    return SurfaceSignatureFeasibilityResult(status, blockers, evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Private bounded pre-supported surface-signature feasibility run."
    )
    parser.add_argument("--inventory-manifest", required=True, type=Path)
    parser.add_argument("--authorization-record", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("surface_signature_feasibility_output_root_mismatch")
        result = run_surface_signature_feasibility(
            inventory_manifest=args.inventory_manifest,
            authorization_record=args.authorization_record,
            output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SystemExit("surface_signature_feasibility_failed") from None
    print(json.dumps({
        "status": result.status,
        "blockers": result.blockers,
        "input_stl_count": result.evidence["input_stl_count"],
        "completed_stl_count": result.evidence["completed_stl_count"],
        "failed_stl_count": result.evidence["failed_stl_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
