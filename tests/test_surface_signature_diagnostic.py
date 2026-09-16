from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from minires.ingestion import InputError
from minires.preparation import surface_signature_diagnostic as diagnostic


class SurfaceSignatureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "private"
        self.private.mkdir()
        self.inventory = self.private / "development-inventory.json"
        self.inventory.write_text("synthetic inventory placeholder")
        self.authorization = self.private / "diagnostic-authorization.json"
        self.authorization.write_text(json.dumps({
            "version": diagnostic.AUTHORIZATION_VERSION,
            "issue": 53,
            "scope": diagnostic.AUTHORIZATION_SCOPE,
            "inventory_manifest_sha256": sha256(self.inventory.read_bytes()).hexdigest(),
            "authorized": True,
        }))
        self.output = self.private / "diagnostic-001"

    def test_source_neutral_stl_fixtures_classify_bounded_stages(self):
        valid = self.private / "valid.stl"
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(valid)
        flat = self.private / "flat.stl"
        trimesh.Trimesh(
            vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                 (0.0, 1.0, 0.0))),
            faces=np.asarray(((0, 1, 2),)), process=False,
        ).export(flat)
        vertical = self.private / "vertical.stl"
        trimesh.Trimesh(
            vertices=np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                 (0.0, 0.0, 1.0))),
            faces=np.asarray(((0, 1, 2),)), process=False,
        ).export(vertical)
        malformed = self.private / "malformed.stl"
        malformed.write_bytes(b"not an stl")
        missing = self.private / "missing.stl"

        self.assertIsNone(diagnostic.classify_surface_signature_path(valid))
        self.assertEqual(
            diagnostic.classify_surface_signature_path(flat),
            "surface_signature_diagnostic_extents_invalid",
        )
        self.assertEqual(
            diagnostic.classify_surface_signature_path(vertical),
            "surface_signature_diagnostic_projected_area_invalid",
        )
        self.assertEqual(
            diagnostic.classify_surface_signature_path(malformed),
            "surface_signature_diagnostic_mesh_type_invalid",
        )
        self.assertEqual(
            diagnostic.classify_surface_signature_path(missing),
            "surface_signature_diagnostic_load_failed",
        )

        empty = trimesh.Trimesh(vertices=[], faces=[], process=False)
        collinear = trimesh.Trimesh(
            vertices=np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                 (0.0, 0.0, 2.0))),
            faces=np.asarray(((0, 1, 2),)), process=False,
        )
        with patch.object(trimesh, "load", return_value=empty):
            self.assertEqual(
                diagnostic.classify_surface_signature_path(valid),
                "surface_signature_diagnostic_array_invalid",
            )
        with patch.object(trimesh, "load", return_value=collinear):
            self.assertEqual(
                diagnostic.classify_surface_signature_path(valid),
                "surface_signature_diagnostic_surface_area_invalid",
            )
        with patch.object(
            diagnostic.surface_signature, "from_mesh", side_effect=ValueError
        ):
            self.assertEqual(
                diagnostic.classify_surface_signature_path(valid),
                "surface_signature_diagnostic_extractor_mismatch",
            )

    def test_bounded_worker_classifies_a_source_neutral_stl(self):
        valid = self.private / "bounded-valid.stl"
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(valid)

        self.assertIsNone(diagnostic._bounded_classify(valid))

    def test_fixed_content_neutral_sample_rule_is_frozen(self):
        self.assertEqual(diagnostic.EXPECTED_DEVELOPMENT_STL_COUNT, 1_636)
        self.assertEqual(diagnostic.SAMPLE_SIZE, 16)
        self.assertEqual(
            diagnostic.sample_ordinals(1_636),
            (0, 109, 218, 327, 436, 545, 654, 763,
             872, 981, 1090, 1199, 1308, 1417, 1526, 1635),
        )
        with self.assertRaises(ValueError):
            diagnostic.sample_ordinals(1_635)

    def test_one_aggregate_only_pass_uses_each_sample_ordinal_once(self):
        paths = tuple(self.private / f"synthetic-{index}.stl" for index in range(1_636))
        outcomes = iter((None,) * 15 + ("surface_signature_diagnostic_mesh_type_invalid",))
        with (
            patch.object(
                diagnostic,
                "_load_authorization",
                return_value=sha256(self.inventory.read_bytes()).hexdigest(),
            ),
            patch.object(diagnostic.feasibility, "_load_inventory", return_value=paths),
            patch.object(
                diagnostic, "_bounded_classify", side_effect=lambda *_args, **_kwargs: next(outcomes)
            ) as classify,
        ):
            result = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )

        expected_paths = [paths[index] for index in diagnostic.sample_ordinals(1_636)]
        self.assertEqual([call.args[0] for call in classify.call_args_list], expected_paths)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence["attempted_stl_count"], 16)
        self.assertEqual(result.evidence["completed_stl_count"], 15)
        self.assertEqual(
            result.evidence["failure_reasons"],
            {"surface_signature_diagnostic_mesh_type_invalid": 1},
        )
        serialized = json.dumps(result.evidence)
        self.assertNotIn("synthetic-", serialized)
        self.assertNotIn(str(self.private), serialized)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"diagnostic-plan.json", "diagnostic-evidence.json", "manifest.json"},
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_authorization_and_fixed_inventory_count_block_before_geometry_access(self):
        with (
            patch.object(diagnostic, "_load_authorization", side_effect=ValueError),
            patch.object(diagnostic.feasibility, "_load_inventory") as inventory,
            patch.object(diagnostic, "_bounded_classify") as classify,
        ):
            unauthorized = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )
        self.assertEqual(unauthorized.blockers, ("private_geometry_diagnostic_not_authorized",))
        inventory.assert_not_called()
        classify.assert_not_called()

        second_output = self.private / "diagnostic-002"
        with (
            patch.object(
                diagnostic,
                "_load_authorization",
                return_value=sha256(self.inventory.read_bytes()).hexdigest(),
            ),
            patch.object(
                diagnostic.feasibility,
                "_load_inventory",
                return_value=(self.private / "one.stl",),
            ),
            patch.object(diagnostic, "_bounded_classify") as classify,
        ):
            wrong_count = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=second_output,
            )
        self.assertEqual(wrong_count.blockers, ("diagnostic_inventory_mismatch",))
        classify.assert_not_called()

    def test_authorization_is_bound_to_exact_inventory_bytes(self):
        self.inventory.write_text("changed after authorization")
        with patch.object(diagnostic, "_bounded_classify") as classify:
            result = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )

        self.assertEqual(result.blockers, ("diagnostic_inventory_authorization_mismatch",))
        classify.assert_not_called()

    def test_worker_transport_failure_blocks_and_stops_private_access(self):
        paths = tuple(self.private / f"synthetic-{index}.stl" for index in range(1_636))
        with (
            patch.object(
                diagnostic,
                "_load_authorization",
                return_value=sha256(self.inventory.read_bytes()).hexdigest(),
            ),
            patch.object(diagnostic.feasibility, "_load_inventory", return_value=paths),
            patch.object(
                diagnostic,
                "_bounded_classify",
                return_value="surface_signature_diagnostic_worker_transport_failed",
            ) as classify,
        ):
            result = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )

        self.assertEqual(classify.call_count, 1)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("surface_signature_diagnostic_infrastructure_failed",))
        self.assertEqual(result.evidence["attempted_stl_count"], 1)
        self.assertEqual(result.evidence["unattempted_stl_count"], 15)

    def test_total_deadline_stops_without_retry_or_sample_replacement(self):
        paths = tuple(self.private / f"synthetic-{index}.stl" for index in range(1_636))
        with (
            patch.object(
                diagnostic,
                "_load_authorization",
                return_value=sha256(self.inventory.read_bytes()).hexdigest(),
            ),
            patch.object(diagnostic.feasibility, "_load_inventory", return_value=paths),
            patch.object(diagnostic, "MAXIMUM_ELAPSED_SECONDS", 0.0),
            patch.object(diagnostic, "_bounded_classify") as classify,
        ):
            result = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )

        classify.assert_not_called()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.evidence["attempted_stl_count"], 0)
        self.assertEqual(result.evidence["unattempted_stl_count"], 16)
        self.assertEqual(
            result.evidence["failure_reasons"],
            {"surface_signature_diagnostic_total_deadline": 1},
        )

    def test_deadline_expiry_during_final_attempt_blocks_the_diagnostic(self):
        paths = tuple(self.private / f"synthetic-{index}.stl" for index in range(1_636))
        attempts = 0

        def classify(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return None

        def clock():
            return 601.0 if attempts == 16 else 0.0

        with (
            patch.object(
                diagnostic,
                "_load_authorization",
                return_value=sha256(self.inventory.read_bytes()).hexdigest(),
            ),
            patch.object(diagnostic.feasibility, "_load_inventory", return_value=paths),
            patch.object(diagnostic, "_bounded_classify", side_effect=classify),
            patch.object(diagnostic.time, "monotonic", side_effect=clock),
        ):
            result = diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=self.output,
            )

        self.assertEqual(attempts, 16)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.evidence["completed_stl_count"], 15)
        self.assertEqual(result.evidence["failed_stl_count"], 1)
        self.assertEqual(result.evidence["unattempted_stl_count"], 0)
        self.assertEqual(
            result.evidence["failure_reasons"],
            {"surface_signature_diagnostic_total_deadline": 1},
        )

    def test_public_seam_and_cli_expose_no_labels_or_held_out_inputs(self):
        self.assertEqual(
            set(inspect.signature(diagnostic.run_surface_signature_diagnostic).parameters),
            {"inventory_manifest", "authorization_record", "output_root"},
        )
        parser = diagnostic.build_parser()
        self.assertEqual(
            {action.dest for action in parser._actions},
            {"help", "inventory_manifest", "authorization_record", "output_root"},
        )
        with self.assertRaisesRegex(InputError, "private_output_directory_required"):
            diagnostic.run_surface_signature_diagnostic(
                inventory_manifest=self.inventory,
                authorization_record=self.authorization,
                output_root=Path.cwd() / "public-output",
            )


if __name__ == "__main__":
    unittest.main()
