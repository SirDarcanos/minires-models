"""Bounded synthetic investigation of the versioned STL geometry measurements.

The public seam accepts no dataset. It exercises the production measurement
operations on four source-neutral synthetic meshes and emits aggregate evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from ..ingestion import InputError
from ..private_io import PrivateArgumentParser, write_private_json
from .stl_probe import _trimesh_version, probe

VERSION = "minires-geometry-contract-investigation-v1"
EXPECTED_TRIMESH_VERSION = "4.10.1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT_ROOT = PROJECT_ROOT / "private" / "candidate-tuning" / "run-020"


@dataclass(frozen=True)
class GeometryContractInvestigationResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]


def _plan(observed_version: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": "source_neutral_synthetic_geometry_measurement_contract_investigation",
        "question": (
            "does the production mesh volume represent occupied union volume, or a signed "
            "additive surface integral for which volume may exceed the axis-aligned bounding box"
        ),
        "production_operations": {
            "mesh_volume": "trimesh.Trimesh.volume_via_mass_properties_surface_integral",
            "bounding_box_volume": "product_of_trimesh_axis_aligned_bounding_box_extents",
        },
        "required_dependency": {"trimesh": EXPECTED_TRIMESH_VERSION},
        "observed_dependency": {"trimesh": observed_version},
        "synthetic_cases": [
            "closed_box",
            "partially_overlapping_shells",
            "reversed_winding_box",
            "open_box",
        ],
        "falsifiable_expectations": {
            "closed_box": "positive_volume_equals_axis_aligned_bounding_box_volume",
            "partially_overlapping_shells": (
                "watertight_and_winding_consistent_but_additive_volume_2_exceeds_global_bounding_box_1.5"
            ),
            "reversed_winding_box": "volume_sign_reverses_while_bounding_box_is_unchanged",
            "open_box": "non_watertight_mesh_still_returns_a_finite_surface_integral_result",
        },
        "decision_rules": {
            "confirmed": (
                "revise_the_audit_invariant; volume_above_bounding_box_is_not_by_itself_a_contract_"
                "contradiction; retain_historical_semantics_as_unresolved; require_better_geometry_"
                "or_provenance_to_classify_topology_or_row_defects"
            ),
            "not_confirmed": "block_without_reinterpreting_the_completed_audit",
        },
        "input_data": "none_synthetic_geometry_only",
        "model_fits": 0,
        "row_policy": "no_canonical_row_read_repair_removal_replacement_or_reslicing",
        "validation_input": "unavailable_to_this_interface",
        "held_out_test_input": "unavailable_to_this_interface",
        "source_group_use": "none",
        "stop_rule": "one_attempt_only_without_retry_candidate_run_or_automatic_continuation",
    }


def _case(mesh: object) -> dict[str, Any]:
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("invalid synthetic mesh")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.stl"
        mesh.export(path)
        measured = probe(path)
        volume = float(measured["volume"])
        bounding_box_volume = float(measured["bbox_area"])
        if (
            not math.isfinite(volume)
            or not math.isfinite(bounding_box_volume)
            or bounding_box_volume <= 0
        ):
            raise ValueError("invalid synthetic measurement")
        loaded = trimesh.load(path)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError("invalid loaded synthetic mesh")
    return {
        "volume_mm3": volume,
        "bounding_box_volume_mm3": bounding_box_volume,
        "volume_to_bounding_box_ratio": volume / bounding_box_volume,
        "watertight": bool(loaded.is_watertight),
        "winding_consistent": bool(loaded.is_winding_consistent),
    }


def _synthetic_cases() -> dict[str, dict[str, Any]]:
    import trimesh

    closed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    shifted = closed.copy()
    shifted.apply_translation((0.5, 0.0, 0.0))
    overlap = trimesh.util.concatenate([closed.copy(), shifted])
    reversed_box = closed.copy()
    reversed_box.invert()
    open_box = trimesh.Trimesh(
        vertices=closed.vertices.copy(),
        faces=closed.faces[:-1].copy(),
        process=False,
    )
    return {
        "closed_box": _case(closed),
        "partially_overlapping_shells": _case(overlap),
        "reversed_winding_box": _case(reversed_box),
        "open_box": _case(open_box),
    }


def _semantics_confirmed(cases: Mapping[str, Mapping[str, Any]]) -> bool:
    closed = cases["closed_box"]
    overlap = cases["partially_overlapping_shells"]
    reversed_box = cases["reversed_winding_box"]
    open_box = cases["open_box"]
    return bool(
        math.isclose(closed["volume_mm3"], 1.0)
        and math.isclose(closed["bounding_box_volume_mm3"], 1.0)
        and closed["watertight"]
        and closed["winding_consistent"]
        and math.isclose(overlap["volume_mm3"], 2.0)
        and math.isclose(overlap["bounding_box_volume_mm3"], 1.5)
        and overlap["watertight"]
        and overlap["winding_consistent"]
        and math.isclose(reversed_box["volume_mm3"], -1.0)
        and math.isclose(reversed_box["bounding_box_volume_mm3"], 1.0)
        and not open_box["watertight"]
        and math.isfinite(open_box["volume_mm3"])
    )


def run_geometry_contract_investigation(
    *, output_root: str | Path
) -> GeometryContractInvestigationResult:
    """Run the fixed four-case synthetic investigation and stop."""
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    cases: dict[str, dict[str, Any]] = {}
    blockers: tuple[str, ...] = ()
    try:
        observed_version = _trimesh_version()
    except RuntimeError:
        observed_version = "unavailable"
        blockers = ("geometry_contract_dependency_unavailable",)
    plan = _plan(observed_version)
    write_private_json(output / "investigation-plan.json", plan)
    if blockers:
        status = "blocked"
    elif observed_version != EXPECTED_TRIMESH_VERSION:
        status = "blocked"
        blockers = ("geometry_contract_dependency_mismatch",)
    else:
        try:
            cases = _synthetic_cases()
            confirmed = _semantics_confirmed(cases)
        except (
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            confirmed = False
        if confirmed:
            status = "completed"
        else:
            status = "blocked"
            blockers = ("geometry_contract_semantics_not_confirmed",)

    decision = (
        {
            "audit_invariant": "invalid",
            "new_stl_volume_semantics": "signed_additive_surface_integral_not_occupied_union_volume",
            "volume_above_bounding_box_interpretation": "not_a_contract_contradiction_by_itself",
            "historical_measurement_semantics": "unresolved_from_committed_provenance",
            "unsupported_or_malformed_geometry": "requires_topology_or_source_geometry_evidence",
            "canonical_data_contract_defect": "not_established",
            "next_action": "revise_audit_invariant_and_documentation_then_obtain_better_inputs_if_row_classification_is_needed",
        }
        if status == "completed"
        else {
            "audit_invariant": "undetermined",
            "next_action": "obtain_compatible_geometry_contract_evidence",
        }
    )
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "synthetic_cases": cases,
        "decision": decision,
        "model_fits": 0,
        "training_records_accessed": False,
        "validation_records_accessed": False,
        "held_out_test_accessed": False,
        "source_groups_used": False,
        "rows_removed_or_repaired": 0,
        "candidate_selected": False,
        "lock_created": False,
    }
    write_private_json(output / "geometry-contract-evidence.json", evidence)
    files = (
        output / "investigation-plan.json",
        output / "geometry-contract-evidence.json",
    )
    write_private_json(
        output / "manifest.json",
        {
            "version": VERSION,
            "create_only": True,
            "artifacts": {
                path.name: sha256(path.read_bytes()).hexdigest() for path in files
            },
            "dataset_accessed": False,
            "publication_performed": False,
            "candidate_selected": False,
            "lock_created": False,
        },
    )
    return GeometryContractInvestigationResult(status, blockers, evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Private bounded synthetic geometry-contract investigation."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("geometry_contract_investigation_output_root_mismatch")
        result = run_geometry_contract_investigation(output_root=args.output_root)
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        raise SystemExit("geometry_contract_investigation_failed") from None
    print(json.dumps({"status": result.status, "blockers": result.blockers}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
