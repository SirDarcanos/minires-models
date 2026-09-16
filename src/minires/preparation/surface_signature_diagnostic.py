"""Bounded aggregate diagnosis of surface-signature extraction failures."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from ..ingestion import InputError
from ..private_io import PrivateArgumentParser, is_private_path, write_private_json
from . import surface_signature
from . import surface_signature_feasibility as feasibility

VERSION = "minires-surface-signature-diagnostic-v1"
AUTHORIZATION_VERSION = feasibility.AUTHORIZATION_VERSION
AUTHORIZATION_SCOPE = "surface_signature_failure_diagnostic_001"
EXPECTED_DEVELOPMENT_STL_COUNT = 1_636
SAMPLE_SIZE = 16
SAMPLE_RULE = "sixteen_evenly_spaced_fixed_inventory_ordinals_including_endpoints"
PER_FILE_TIMEOUT_SECONDS = 30.0
MAXIMUM_ELAPSED_SECONDS = 600.0
MAXIMUM_WORKER_RESIDENT_BYTES = feasibility.MAXIMUM_WORKER_RESIDENT_BYTES
RESOURCE_SAMPLE_SECONDS = feasibility.RESOURCE_SAMPLE_SECONDS
EXPECTED_TRIMESH_VERSION = feasibility.EXPECTED_TRIMESH_VERSION
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT_ROOT = (
    PROJECT_ROOT / "private" / "geometry-feature-feasibility" / "diagnostic-001"
)

_ALLOWED_FAILURES = frozenset({
    "surface_signature_diagnostic_load_failed",
    "surface_signature_diagnostic_mesh_type_invalid",
    "surface_signature_diagnostic_array_invalid",
    "surface_signature_diagnostic_extents_invalid",
    "surface_signature_diagnostic_surface_area_invalid",
    "surface_signature_diagnostic_projected_area_invalid",
    "surface_signature_diagnostic_binning_invalid",
    "surface_signature_diagnostic_extractor_mismatch",
    "surface_signature_diagnostic_timeout",
    "surface_signature_diagnostic_input_changed",
    "surface_signature_diagnostic_resource_limit",
    "surface_signature_diagnostic_resource_monitor_unavailable",
    "surface_signature_diagnostic_worker_transport_failed",
    "surface_signature_diagnostic_total_deadline",
})


@dataclass(frozen=True)
class SurfaceSignatureDiagnosticResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]


def sample_ordinals(inventory_count: int) -> tuple[int, ...]:
    """Return the frozen content-neutral sample for the reconciled run-001 inventory."""
    if inventory_count != EXPECTED_DEVELOPMENT_STL_COUNT:
        raise ValueError("diagnostic_inventory_mismatch")
    return tuple(
        index * (inventory_count - 1) // (SAMPLE_SIZE - 1)
        for index in range(SAMPLE_SIZE)
    )


def classify_surface_signature_path(path: str | Path) -> str | None:
    """Classify one source-neutral STL without returning geometry or exception details."""
    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(Path(path), process=False)
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_load_failed"
    if not isinstance(mesh, trimesh.Trimesh):
        return "surface_signature_diagnostic_mesh_type_invalid"
    try:
        if mesh.is_empty:
            return "surface_signature_diagnostic_array_invalid"
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        arrays_valid = (
            vertices.ndim == 2
            and vertices.shape[1:] == (3,)
            and len(vertices) > 0
            and np.isfinite(vertices).all()
            and faces.ndim == 2
            and faces.shape[1:] == (3,)
            and len(faces) > 0
            and not (faces < 0).any()
            and not (faces >= len(vertices)).any()
        )
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_array_invalid"
    if not arrays_valid:
        return "surface_signature_diagnostic_array_invalid"

    try:
        z_min = float(vertices[:, 2].min())
        z_max = float(vertices[:, 2].max())
        z_span = z_max - z_min
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_extents_invalid"
    if not math.isfinite(z_span) or z_span <= 0.0:
        return "surface_signature_diagnostic_extents_invalid"

    surface_total = 0.0
    projected_total = 0.0
    try:
        for start in range(0, len(faces), surface_signature.FACE_CHUNK_SIZE):
            triangles = vertices[faces[start:start + surface_signature.FACE_CHUNK_SIZE]]
            crosses = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            areas = np.linalg.norm(crosses, axis=1) * 0.5
            if not np.isfinite(crosses).all() or not np.isfinite(areas).all():
                return "surface_signature_diagnostic_surface_area_invalid"
            surface_total += float(areas.sum(dtype=np.float64))
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_surface_area_invalid"
    if not math.isfinite(surface_total) or surface_total <= 0.0:
        return "surface_signature_diagnostic_surface_area_invalid"

    try:
        for start in range(0, len(faces), surface_signature.FACE_CHUNK_SIZE):
            triangles = vertices[faces[start:start + surface_signature.FACE_CHUNK_SIZE]]
            crosses = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            projected = np.abs(crosses[:, 2]) * 0.5
            if not np.isfinite(projected).all():
                return "surface_signature_diagnostic_projected_area_invalid"
            projected_total += float(projected.sum(dtype=np.float64))
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_projected_area_invalid"
    if not math.isfinite(projected_total) or projected_total <= 0.0:
        return "surface_signature_diagnostic_projected_area_invalid"

    try:
        for start in range(0, len(faces), surface_signature.FACE_CHUNK_SIZE):
            triangles = vertices[faces[start:start + surface_signature.FACE_CHUNK_SIZE]]
            normalized_z = (triangles[:, :, 2].mean(axis=1) - z_min) / z_span
            bins = np.minimum(
                (normalized_z * surface_signature.BIN_COUNT).astype(np.int64),
                surface_signature.BIN_COUNT - 1,
            )
            if (
                not np.isfinite(normalized_z).all()
                or (normalized_z < 0.0).any()
                or (normalized_z > 1.0).any()
                or (bins < 0).any()
                or (bins >= surface_signature.BIN_COUNT).any()
            ):
                return "surface_signature_diagnostic_binning_invalid"
        signature = surface_signature.from_mesh(mesh)
    except MemoryError:
        raise
    except Exception:
        return "surface_signature_diagnostic_extractor_mismatch"
    if (
        len(signature.surface_area_by_normalized_z) != surface_signature.BIN_COUNT
        or len(signature.absolute_xy_projected_area_by_normalized_z)
        != surface_signature.BIN_COUNT
        or not all(math.isfinite(value) for value in (
            *signature.surface_area_by_normalized_z,
            *signature.absolute_xy_projected_area_by_normalized_z,
        ))
    ):
        return "surface_signature_diagnostic_binning_invalid"
    return None


def _load_authorization(path: Path) -> str:
    if not is_private_path(path):
        raise ValueError("invalid authorization")
    try:
        raw = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid authorization") from None
    inventory_sha256 = raw.get("inventory_manifest_sha256") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or raw.get("version") != AUTHORIZATION_VERSION
        or raw.get("issue") != 53
        or raw.get("scope") != AUTHORIZATION_SCOPE
        or not isinstance(inventory_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", inventory_sha256) is None
        or raw.get("authorized") is not True
    ):
        raise ValueError("invalid authorization")
    return inventory_sha256


def _worker(path: str, queue: Any) -> None:
    try:
        failure = classify_surface_signature_path(path)
    except MemoryError:
        failure = "surface_signature_diagnostic_resource_limit"
    except Exception:
        failure = "surface_signature_diagnostic_worker_transport_failed"
    queue.put((failure, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _bounded_classify(
    path: Path, timeout_seconds: float = PER_FILE_TIMEOUT_SECONDS
) -> str | None:
    try:
        before = (path.stat().st_size, path.stat().st_mtime_ns)
    except OSError:
        return "surface_signature_diagnostic_input_changed"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(str(path), queue))
    process.start()
    deadline = time.monotonic() + timeout_seconds
    while process.is_alive():
        if time.monotonic() >= deadline:
            process.terminate()
            process.join()
            return "surface_signature_diagnostic_timeout"
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
            return "surface_signature_diagnostic_resource_monitor_unavailable"
        if resident_bytes > MAXIMUM_WORKER_RESIDENT_BYTES:
            process.terminate()
            process.join()
            return "surface_signature_diagnostic_resource_limit"
        process.join(RESOURCE_SAMPLE_SECONDS)
    try:
        result, peak_rss = queue.get(timeout=1.0)
        after = (path.stat().st_size, path.stat().st_mtime_ns)
    except Exception:
        return "surface_signature_diagnostic_worker_transport_failed"
    peak_bytes = int(peak_rss) if sys.platform == "darwin" else int(peak_rss) * 1024
    if peak_bytes > MAXIMUM_WORKER_RESIDENT_BYTES:
        return "surface_signature_diagnostic_resource_limit"
    if before != after:
        return "surface_signature_diagnostic_input_changed"
    if result is None:
        return None
    return result if result in _ALLOWED_FAILURES else "surface_signature_diagnostic_worker_transport_failed"


def _plan(observed_version: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": "surface_signature_failure_stage_diagnostic",
        "hypothesis": (
            "the uniform run-001 failure may arise from one shared loader, mesh-type, "
            "array, geometry-contract, binning, resource, or worker-transport stage"
        ),
        "observed_dependency": {"trimesh": observed_version},
        "required_dependency": {"trimesh": EXPECTED_TRIMESH_VERSION},
        "input_contract": {
            "inventory_version": feasibility.INVENTORY_VERSION,
            "expected_development_stl_count": EXPECTED_DEVELOPMENT_STL_COUNT,
            "held_out_test_geometry": "forbidden",
        },
        "sample": {
            "rule": SAMPLE_RULE,
            "size": SAMPLE_SIZE,
            "ordinals": list(sample_ordinals(EXPECTED_DEVELOPMENT_STL_COUNT)),
            "content_or_failure_dependent": False,
            "replacement_or_budget_recycling": False,
        },
        "reason_allowlist": sorted(_ALLOWED_FAILURES),
        "resource_limits": {
            "per_file_timeout_seconds": PER_FILE_TIMEOUT_SECONDS,
            "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS,
            "maximum_worker_resident_bytes": MAXIMUM_WORKER_RESIDENT_BYTES,
            "resource_sample_seconds": RESOURCE_SAMPLE_SECONDS,
            "retry_count": 0,
        },
        "evidence_policy": (
            "aggregate_counts_only_without_paths_identities_feature_values_"
            "stack_traces_or_exception_text"
        ),
        "labels": "unavailable_to_this_interface",
        "source_groups": "unavailable_to_this_interface",
        "model_fits": 0,
        "stop_rule": (
            "stop_after_one_fixed_sample_pass_or_at_the_total_deadline_without_"
            "retry_replacement_repair_or_semantic_change"
        ),
        "execution_authorization": {
            "version": AUTHORIZATION_VERSION,
            "issue": 53,
            "scope": AUTHORIZATION_SCOPE,
            "inventory_manifest_sha256_required": True,
            "separate_private_record_required": True,
        },
    }


def run_surface_signature_diagnostic(
    *,
    inventory_manifest: str | Path,
    authorization_record: str | Path,
    output_root: str | Path,
) -> SurfaceSignatureDiagnosticResult:
    """Run the separately authorized fixed diagnostic sample once."""
    output = Path(output_root)
    if not is_private_path(output):
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    observed_version = feasibility._trimesh_version()
    write_private_json(output / "diagnostic-plan.json", _plan(observed_version))
    blockers: tuple[str, ...] = ()
    paths: tuple[Path, ...] = ()
    if observed_version != EXPECTED_TRIMESH_VERSION:
        blockers = ("surface_signature_diagnostic_dependency_mismatch",)
    else:
        try:
            authorized_inventory_sha256 = _load_authorization(Path(authorization_record))
        except ValueError:
            blockers = ("private_geometry_diagnostic_not_authorized",)
        if not blockers:
            try:
                paths = feasibility._load_inventory(
                    Path(inventory_manifest),
                    expected_sha256=authorized_inventory_sha256,
                )
                ordinals = sample_ordinals(len(paths))
            except (OSError, ValueError) as error:
                if str(error) == "diagnostic_inventory_mismatch":
                    blocker = "diagnostic_inventory_mismatch"
                elif str(error) == "inventory authorization mismatch":
                    blocker = "diagnostic_inventory_authorization_mismatch"
                else:
                    blocker = "invalid_development_geometry_inventory"
                blockers = (blocker,)

    started = time.monotonic()
    reasons: Counter[str] = Counter()
    completed = 0
    attempted = 0
    unattempted = 0
    if not blockers:
        for position, ordinal in enumerate(ordinals):
            remaining = MAXIMUM_ELAPSED_SECONDS - (time.monotonic() - started)
            if remaining <= 0.0:
                reasons["surface_signature_diagnostic_total_deadline"] += 1
                unattempted = len(ordinals) - position
                blockers = ("surface_signature_diagnostic_incomplete",)
                break
            failure = _bounded_classify(
                paths[ordinal], timeout_seconds=min(PER_FILE_TIMEOUT_SECONDS, remaining)
            )
            attempted += 1
            deadline_reached = time.monotonic() - started >= MAXIMUM_ELAPSED_SECONDS
            if deadline_reached:
                failure = "surface_signature_diagnostic_total_deadline"
                blockers = ("surface_signature_diagnostic_incomplete",)
                unattempted = len(ordinals) - position - 1
            elif (
                failure == "surface_signature_diagnostic_timeout"
                and remaining < PER_FILE_TIMEOUT_SECONDS
            ):
                failure = "surface_signature_diagnostic_total_deadline"
                blockers = ("surface_signature_diagnostic_incomplete",)
                unattempted = len(ordinals) - position - 1

            if failure is None:
                completed += 1
            else:
                reasons[failure] += 1
                if failure in {
                    "surface_signature_diagnostic_resource_monitor_unavailable",
                    "surface_signature_diagnostic_worker_transport_failed",
                }:
                    blockers = ("surface_signature_diagnostic_infrastructure_failed",)
                    unattempted = len(ordinals) - position - 1
            if blockers:
                break

    status = "completed" if not blockers else "blocked"
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "declared_inventory_stl_count": len(paths),
        "sample_size": SAMPLE_SIZE,
        "attempted_stl_count": attempted,
        "completed_stl_count": completed,
        "failed_stl_count": attempted - completed,
        "unattempted_stl_count": unattempted,
        "failure_reasons": dict(sorted(reasons.items())),
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "model_fits": 0,
        "labels_accessed": False,
        "source_groups_used": False,
        "held_out_test_geometry_accessed": False,
        "paths_or_identities_persisted": False,
        "feature_values_persisted": False,
        "exception_details_persisted": False,
        "retries": 0,
        "rows_repaired_rotated_scaled_resliced_replaced_or_omitted": 0,
    }
    write_private_json(output / "diagnostic-evidence.json", evidence)
    artifacts = (output / "diagnostic-plan.json", output / "diagnostic-evidence.json")
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
        "exception_details_persisted": False,
        "held_out_test_geometry_accessed": False,
    })
    return SurfaceSignatureDiagnosticResult(status, blockers, evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Private bounded surface-signature failure-stage diagnostic."
    )
    parser.add_argument("--inventory-manifest", required=True, type=Path)
    parser.add_argument("--authorization-record", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("surface_signature_diagnostic_output_root_mismatch")
        result = run_surface_signature_diagnostic(
            inventory_manifest=args.inventory_manifest,
            authorization_record=args.authorization_record,
            output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SystemExit("surface_signature_diagnostic_failed") from None
    print(json.dumps({
        "status": result.status,
        "blockers": result.blockers,
        "attempted_stl_count": result.evidence["attempted_stl_count"],
        "completed_stl_count": result.evidence["completed_stl_count"],
        "failed_stl_count": result.evidence["failed_stl_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
