"""Synthetic public-seam tests for the geometry measurement contract investigation."""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires.preparation import geometry_contract_investigation as investigation


class GeometryContractInvestigationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "private" / "synthetic-run-020"

    def test_confirms_surface_integral_semantics_without_reading_dataset_rows(self):
        result = investigation.run_geometry_contract_investigation(
            output_root=self.output,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.blockers, ())
        self.assertEqual(result.evidence["decision"]["audit_invariant"], "invalid")
        self.assertEqual(
            result.evidence["decision"]["new_stl_volume_semantics"],
            "signed_additive_surface_integral_not_occupied_union_volume",
        )
        self.assertEqual(
            result.evidence["decision"]["historical_measurement_semantics"],
            "unresolved_from_committed_provenance",
        )
        cases = result.evidence["synthetic_cases"]
        self.assertEqual(cases["closed_box"]["volume_mm3"], 1.0)
        self.assertEqual(cases["closed_box"]["bounding_box_volume_mm3"], 1.0)
        self.assertEqual(
            cases["partially_overlapping_shells"]["bounding_box_volume_mm3"],
            1.5,
        )
        self.assertEqual(cases["partially_overlapping_shells"]["volume_mm3"], 2.0)
        self.assertTrue(cases["partially_overlapping_shells"]["watertight"])
        self.assertTrue(cases["partially_overlapping_shells"]["winding_consistent"])
        self.assertEqual(cases["reversed_winding_box"]["volume_mm3"], -1.0)
        self.assertFalse(cases["open_box"]["watertight"])
        self.assertEqual(result.evidence["model_fits"], 0)
        self.assertFalse(result.evidence["training_records_accessed"])
        self.assertFalse(result.evidence["validation_records_accessed"])
        self.assertFalse(result.evidence["held_out_test_accessed"])
        self.assertEqual(
            set(path.name for path in self.output.iterdir()),
            {"investigation-plan.json", "geometry-contract-evidence.json", "manifest.json"},
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_dependency_mismatch_blocks_without_a_favorable_decision(self):
        with patch.object(investigation, "_trimesh_version", return_value="unexpected"):
            result = investigation.run_geometry_contract_investigation(
                output_root=self.output,
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("geometry_contract_dependency_mismatch",))
        self.assertEqual(result.evidence["decision"]["audit_invariant"], "undetermined")
        self.assertEqual(result.evidence["synthetic_cases"], {})

    def test_non_finite_semantic_result_writes_a_complete_blocked_package(self):
        with patch.object(
            investigation,
            "probe",
            return_value={
                "volume": float("nan"),
                "bbox_area": 1.0,
            },
        ):
            result = investigation.run_geometry_contract_investigation(
                output_root=self.output,
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("geometry_contract_semantics_not_confirmed",))
        self.assertEqual(result.evidence["synthetic_cases"], {})
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(
            set(manifest["artifacts"]),
            {"investigation-plan.json", "geometry-contract-evidence.json"},
        )

    def test_missing_dependency_still_writes_bounded_create_only_evidence(self):
        with patch.object(
            investigation,
            "_trimesh_version",
            side_effect=RuntimeError("geometry_dependency_unavailable"),
        ):
            result = investigation.run_geometry_contract_investigation(
                output_root=self.output,
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("geometry_contract_dependency_unavailable",))
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_create_only_output_cannot_overwrite_preserved_evidence(self):
        investigation.run_geometry_contract_investigation(output_root=self.output)
        snapshot = {path.name: path.read_bytes() for path in self.output.iterdir()}

        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            investigation.run_geometry_contract_investigation(output_root=self.output)

        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_cli_and_public_seam_accept_no_dataset_or_model_arguments(self):
        parameters = inspect.signature(
            investigation.run_geometry_contract_investigation
        ).parameters
        self.assertEqual(set(parameters), {"output_root"})
        parser = investigation.build_parser()
        self.assertEqual(
            {action.dest for action in parser._actions},
            {"help", "output_root"},
        )
        for flag in (
            "--training-records",
            "--validation-records",
            "--test-records",
            "--source-group",
            "--model",
            "--threshold",
        ):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--output-root", "unused", flag, "unused"])


if __name__ == "__main__":
    unittest.main()
