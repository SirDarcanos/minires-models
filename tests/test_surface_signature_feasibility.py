from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import trimesh

from minires.ingestion import InputError
from minires.preparation import surface_signature_feasibility as feasibility


class SurfaceSignatureFeasibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "private"
        self.geometry = self.private / "development-geometry"
        self.geometry.mkdir(parents=True)
        self.paths = (self.geometry / "first.stl", self.geometry / "second.stl")
        for path in self.paths:
            path.write_bytes(b"synthetic-placeholder")
        self.inventory = self.private / "development-geometry-inventory.json"
        self.inventory.write_text(json.dumps({
            "version": feasibility.INVENTORY_VERSION,
            "scope": "pre_supported_training_and_validation_only",
            "presupported_scope_confirmed": True,
            "held_out_test_geometry_included": False,
            "corpus": {
                "id": "existing-presupported-v1",
                "separate_from_canonical_dataset": True,
            },
            "root": str(self.geometry),
            "expected_stl_count": 2,
            "development_rows": [
                {"partition": "training", "partition_row_index": 4,
                 "relative_stl_path": self.paths[0].name},
                {"partition": "validation", "partition_row_index": 7,
                 "relative_stl_path": self.paths[1].name},
            ],
            "reconciliation": {
                "version": feasibility.RECONCILIATION_VERSION,
                "existing_presupported_stl_count": 1931,
                "training_row_count": 1,
                "validation_row_count": 1,
                "held_out_test_stl_count": 100,
                "noncanonical_presupported_stl_count": 1829,
                "missing_development_geometry_count": 0,
                "duplicate_development_geometry_count": 0,
                "development_rows_one_to_one": True,
            },
        }))
        self.refresh_inventory_manifest()
        self.authorization = self.private / "execution-authorization.json"
        self.authorization.write_text(json.dumps({
            "version": feasibility.AUTHORIZATION_VERSION,
            "issue": 53,
            "scope": "surface_signature_feasibility_run_001",
            "authorized": True,
        }))
        self.output = self.private / "surface-signature-run"

    def refresh_inventory_manifest(self):
        self.inventory.with_name("manifest.json").write_text(json.dumps({
            "version": feasibility.RECONCILIATION_PACKAGE_VERSION,
            "artifacts": {
                self.inventory.name: sha256(self.inventory.read_bytes()).hexdigest(),
            },
        }))

    def run_feasibility(self):
        return feasibility.run_surface_signature_feasibility(
            inventory_manifest=self.inventory,
            authorization_record=self.authorization,
            output_root=self.output,
        )

    def test_bounded_worker_extracts_a_synthetic_stl_through_the_public_seam(self):
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(self.paths[0])
        inventory = json.loads(self.inventory.read_text())
        inventory["expected_stl_count"] = 1
        inventory["development_rows"] = [inventory["development_rows"][0]]
        reconciliation = inventory["reconciliation"]
        reconciliation["validation_row_count"] = 0
        reconciliation["noncanonical_presupported_stl_count"] = 1830
        self.inventory.write_text(json.dumps(inventory))
        self.refresh_inventory_manifest()

        result = self.run_feasibility()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence["completed_stl_count"], 1)
        self.assertEqual(result.evidence["failure_reasons"], {})

    def test_complete_inventory_produces_aggregate_only_create_only_evidence(self):
        with patch.object(feasibility, "_bounded_extract", return_value=None) as extract:
            result = self.run_feasibility()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.blockers, ())
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(result.evidence["input_stl_count"], 2)
        self.assertEqual(result.evidence["completed_stl_count"], 2)
        self.assertEqual(result.evidence["failed_stl_count"], 0)
        self.assertEqual(result.evidence["feature_contract"]["bin_count"], 32)
        self.assertEqual(result.evidence["model_fits"], 0)
        self.assertFalse(result.evidence["labels_accessed"])
        self.assertFalse(result.evidence["held_out_test_geometry_accessed"])
        serialized = json.dumps(result.evidence)
        self.assertNotIn("first.stl", serialized)
        self.assertNotIn("second.stl", serialized)
        self.assertNotIn(str(self.private), serialized)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"investigation-plan.json", "surface-signature-evidence.json", "manifest.json"},
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_one_extraction_failure_blocks_without_omission_or_retry(self):
        outcomes = iter((None, "surface_signature_resource_limit"))
        with patch.object(
            feasibility, "_bounded_extract", side_effect=lambda _, **__: next(outcomes)
        ) as extract:
            result = self.run_feasibility()

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("surface_signature_extraction_incomplete",))
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(result.evidence["completed_stl_count"], 1)
        self.assertEqual(result.evidence["failed_stl_count"], 1)
        self.assertEqual(
            result.evidence["failure_reasons"], {"surface_signature_resource_limit": 1}
        )
        self.assertEqual(result.evidence["rows_omitted"], 0)
        self.assertFalse(result.evidence["candidate_selected"])

    def test_resident_memory_ceiling_terminates_worker(self):
        class FakeProcess:
            pid = 123

            def __init__(self):
                self.alive = True
                self.terminated = False

            def start(self):
                pass

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive = False
                self.terminated = True

            def join(self, timeout=None):
                pass

        process = FakeProcess()
        context = SimpleNamespace(
            Queue=lambda: SimpleNamespace(),
            Process=lambda **kwargs: process,
        )
        excessive_kib = feasibility.MAXIMUM_WORKER_RESIDENT_BYTES // 1024 + 1
        with (
            patch.object(feasibility.multiprocessing, "get_context", return_value=context),
            patch.object(
                feasibility.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=str(excessive_kib)),
            ),
        ):
            failure = feasibility._bounded_extract(self.paths[0])

        self.assertEqual(failure, "surface_signature_resource_limit")
        self.assertTrue(process.terminated)

    def test_total_deadline_blocks_without_starting_or_omitting_rows(self):
        with (
            patch.object(feasibility, "MAXIMUM_ELAPSED_SECONDS", 0.0),
            patch.object(feasibility, "_bounded_extract") as extract,
        ):
            result = self.run_feasibility()

        extract.assert_not_called()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.evidence["attempted_stl_count"], 0)
        self.assertEqual(result.evidence["unattempted_stl_count"], 2)
        self.assertEqual(
            result.evidence["failure_reasons"], {"surface_signature_total_deadline": 1}
        )

    def test_inventory_loader_parses_the_exact_authorized_bytes(self):
        authorized_sha256 = sha256(self.inventory.read_bytes()).hexdigest()
        self.inventory.write_text(self.inventory.read_text() + "\n")
        self.refresh_inventory_manifest()

        with self.assertRaisesRegex(ValueError, "inventory authorization mismatch"):
            feasibility._load_inventory(
                self.inventory, expected_sha256=authorized_sha256
            )

    def test_nonseparate_corpus_contract_blocks_before_geometry_access(self):
        inventory = json.loads(self.inventory.read_text())
        inventory["corpus"]["separate_from_canonical_dataset"] = False
        self.inventory.write_text(json.dumps(inventory))

        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.blockers, ("invalid_development_geometry_inventory",))
        extract.assert_not_called()

    def test_incomplete_reconciliation_blocks_before_geometry_access(self):
        incomplete = json.loads(self.inventory.read_text())
        incomplete["reconciliation"]["validation_row_count"] = 2
        self.inventory.write_text(json.dumps(incomplete))

        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.blockers, ("invalid_development_geometry_inventory",))
        extract.assert_not_called()

    def test_resolved_path_aliases_block_one_to_one_inventory(self):
        aliased = json.loads(self.inventory.read_text())
        aliased["development_rows"][1]["relative_stl_path"] = "./first.stl"
        self.inventory.write_text(json.dumps(aliased))

        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.blockers, ("invalid_development_geometry_inventory",))
        extract.assert_not_called()

    def test_duplicate_or_missing_partition_row_blocks_reconciliation(self):
        duplicate = json.loads(self.inventory.read_text())
        duplicate["development_rows"][1]["partition"] = "training"
        duplicate["development_rows"][1]["partition_row_index"] = 4
        self.inventory.write_text(json.dumps(duplicate))

        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.blockers, ("invalid_development_geometry_inventory",))
        extract.assert_not_called()

    def test_test_contaminated_inventory_blocks_before_geometry_access(self):
        contaminated = json.loads(self.inventory.read_text())
        contaminated["held_out_test_geometry_included"] = True
        self.inventory.write_text(json.dumps(contaminated))

        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("invalid_development_geometry_inventory",))
        extract.assert_not_called()
        self.assertEqual(result.evidence["input_stl_count"], 0)

    def test_missing_separate_authorization_blocks_before_geometry_access(self):
        self.authorization.unlink()
        with patch.object(feasibility, "_bounded_extract") as extract:
            result = self.run_feasibility()

        self.assertEqual(result.blockers, ("private_geometry_execution_not_authorized",))
        extract.assert_not_called()

    def test_create_only_and_private_interfaces_reject_reuse_or_public_paths(self):
        with patch.object(feasibility, "_bounded_extract", return_value=None):
            self.run_feasibility()
            with self.assertRaisesRegex(InputError, "private_output_directory_unavailable"):
                self.run_feasibility()
            with self.assertRaisesRegex(InputError, "private_output_directory_required"):
                feasibility.run_surface_signature_feasibility(
                    inventory_manifest=self.inventory,
                    authorization_record=self.authorization,
                    output_root=Path.cwd() / "public-output",
                )

    def test_cli_exposes_only_private_inventory_authorization_and_output_seams(self):
        self.assertEqual(
            set(inspect.signature(feasibility.run_surface_signature_feasibility).parameters),
            {"inventory_manifest", "authorization_record", "output_root"},
        )
        parser = feasibility.build_parser()
        self.assertEqual(
            {action.dest for action in parser._actions},
            {"help", "inventory_manifest", "authorization_record", "output_root"},
        )
        for flag in ("--training-records", "--validation-records", "--test-records",
                     "--labels", "--source-group", "--model"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args([
                    "--inventory-manifest", str(self.inventory),
                    "--authorization-record", str(self.authorization),
                    "--output-root", str(self.output),
                    flag, "unused",
                ])


if __name__ == "__main__":
    unittest.main()
