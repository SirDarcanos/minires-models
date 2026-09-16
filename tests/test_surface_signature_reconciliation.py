from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires.ingestion import fingerprint
from minires.preparation import surface_signature_reconciliation as reconciliation
from minires.preparation.assembly import ASSEMBLY_VERSION


class SurfaceSignatureReconciliationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "private"
        self.canonical = self.private / "canonical-dataset"
        self.geometry = self.private / "existing-presupported-corpus"
        self.inputs = self.private / "preparation-evidence"
        self.output = self.private / "geometry-feature-reconciliation"
        self.canonical.mkdir(parents=True)
        self.geometry.mkdir(parents=True)
        self.inputs.mkdir(parents=True)

        relative_paths = ("first.stl", "nested/second.stl", "rejected.stl")
        for index, relative in enumerate(relative_paths, 1):
            path = self.geometry / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"mesh-{index}".encode())
        self.entries = [
            {
                "entry_id": sha256(relative.encode()).hexdigest(),
                "relative_path": relative,
                "byte_count": (self.geometry / relative).stat().st_size,
                "sha256": sha256((self.geometry / relative).read_bytes()).hexdigest(),
                "status": "candidate" if index < 2 else "rejected",
                "reason": None if index < 2 else "bounded_rejection",
            }
            for index, relative in enumerate(relative_paths)
        ]
        accepted = [
            {
                "entry_id": entry["entry_id"],
                "input_sha256": entry["sha256"],
                "record": {},
                "duplicate_group": None,
            }
            for entry in self.entries[:2]
        ]
        rejected = [{"entry_id": self.entries[2]["entry_id"], "reason": "bounded_rejection"}]
        self.batch = self.inputs / "batch-result.json"
        self.batch.write_text(json.dumps({
            "schema_version": 1,
            "outcome": "completed",
            "inventory": self.entries,
            "accepted_records": accepted,
            "rejected_entries": rejected,
            "accounting": {"inventory_count": 3, "accepted_count": 2, "rejected_count": 1},
        }))
        self.batch_linkage = self.inputs / "batch-linkage.json"
        self.batch_linkage.write_text(json.dumps({
            "version": reconciliation.BATCH_LINKAGE_VERSION,
            "source_batch_sha256": sha256(self.batch.read_bytes()).hexdigest(),
            "schema_version": 1,
            "outcome": "completed",
            "accounting": {"inventory_count": 3, "accepted_count": 2, "rejected_count": 1},
            "inventory": self.entries,
            "record_payloads_decoded": False,
            "labels_decoded_or_inspected": False,
        }))

        first_id = fingerprint([ASSEMBLY_VERSION, "record", self.entries[0]["entry_id"]])
        second_id = fingerprint([ASSEMBLY_VERSION, "record", self.entries[1]["entry_id"]])
        self.training = self.canonical / "train.jsonl"
        self.validation = self.canonical / "validation.jsonl"
        self.training.write_text(
            json.dumps({"_id": "historical-row", "sliced_resin_mass_g": 2.0}) + "\n"
            + json.dumps({"_id": first_id, "sliced_resin_mass_g": 3.0}) + "\n"
        )
        self.validation.write_text(
            json.dumps({"_id": second_id, "sliced_resin_mass_g": 4.0}) + "\n"
        )
        self.dataset_provenance = self.canonical / "provenance.json"
        self.dataset_provenance.write_text(json.dumps({
            "assembly_version": ASSEMBLY_VERSION,
            "new_batch_fingerprint": sha256(self.batch.read_bytes()).hexdigest(),
        }))
        self.dataset_manifest = self.canonical / "manifest.json"
        self.dataset_manifest.write_text(json.dumps({
            "assembly_version": ASSEMBLY_VERSION,
            "row_accounting": {
                "new_inventory": 3,
                "new_accepted": 2,
                "new_rejected": 1,
                "eligible": 4,
                "train": 2,
                "validation": 1,
                "test": 1,
            },
            "artifacts": {
                "train.jsonl": sha256(self.training.read_bytes()).hexdigest(),
                "validation.jsonl": sha256(self.validation.read_bytes()).hexdigest(),
                "provenance.json": sha256(self.dataset_provenance.read_bytes()).hexdigest(),
            },
        }))

    def run_reconciliation(self):
        with patch.object(reconciliation, "EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT", 3):
            return reconciliation.reconcile_surface_signature_inventory(
                training_records=self.training,
                validation_records=self.validation,
                dataset_manifest=self.dataset_manifest,
                dataset_provenance=self.dataset_provenance,
                batch_result=self.batch,
                batch_linkage=self.batch_linkage,
                presupported_root=self.geometry,
                output_root=self.output,
                scope_confirmed=True,
            )

    def test_reconciles_development_rows_without_a_test_record_input(self):
        result = self.run_reconciliation()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.blockers, ())
        self.assertEqual(result.evidence["training_stl_count"], 1)
        self.assertEqual(result.evidence["validation_stl_count"], 1)
        self.assertEqual(result.evidence["held_out_test_stl_count"], 0)
        inventory = json.loads((self.output / "development-inventory.json").read_text())
        self.assertEqual(
            inventory["corpus"],
            {"id": "existing-presupported-v1", "separate_from_canonical_dataset": True},
        )
        self.assertEqual(
            [(row["partition"], row["partition_row_index"])
             for row in inventory["development_rows"]],
            [("training", 1), ("validation", 0)],
        )
        self.assertEqual(inventory["reconciliation"]["noncanonical_presupported_stl_count"], 1)
        serialized_evidence = json.dumps(result.evidence)
        for private_value in ("first.stl", "second.stl", str(self.private)):
            self.assertNotIn(private_value, serialized_evidence)

    def test_unmatched_accepted_geometry_is_counted_as_held_out_without_reading_test(self):
        self.validation.write_text(json.dumps({"_id": "historical-validation"}) + "\n")
        (self.geometry / "nested/second.stl").unlink()
        manifest = json.loads(self.dataset_manifest.read_text())
        manifest["artifacts"]["validation.jsonl"] = sha256(self.validation.read_bytes()).hexdigest()
        self.dataset_manifest.write_text(json.dumps(manifest))

        result = self.run_reconciliation()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence["validation_stl_count"], 0)
        self.assertEqual(result.evidence["held_out_test_stl_count"], 1)
        inventory = json.loads((self.output / "development-inventory.json").read_text())
        self.assertEqual(len(inventory["development_rows"]), 1)

    def test_mutated_linkage_content_blocks_even_when_source_hash_is_preserved(self):
        linkage = json.loads(self.batch_linkage.read_text())
        linkage["inventory"][0]["byte_count"] += 1
        self.batch_linkage.write_text(json.dumps(linkage))

        result = self.run_reconciliation()

        self.assertEqual(result.blockers, ("development_partition_reconciliation_failed",))
        self.assertFalse((self.output / "development-inventory.json").exists())

    def test_linkage_must_match_checksum_bound_canonical_provenance(self):
        linkage = json.loads(self.batch_linkage.read_text())
        linkage["source_batch_sha256"] = "0" * 64
        self.batch_linkage.write_text(json.dumps(linkage))

        result = self.run_reconciliation()

        self.assertEqual(result.blockers, ("development_partition_reconciliation_failed",))
        self.assertFalse((self.output / "development-inventory.json").exists())

    def test_changed_geometry_blocks_without_writing_development_inventory(self):
        (self.geometry / "first.stl").write_bytes(b"changed-size")

        result = self.run_reconciliation()

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("existing_geometry_inventory_mismatch",))
        self.assertFalse((self.output / "development-inventory.json").exists())

    def test_geometry_root_must_be_separate_from_canonical_dataset(self):
        result = None
        with patch.object(reconciliation, "EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT", 3):
            result = reconciliation.reconcile_surface_signature_inventory(
                training_records=self.training,
                validation_records=self.validation,
                dataset_manifest=self.dataset_manifest,
                dataset_provenance=self.dataset_provenance,
                batch_result=self.batch,
                batch_linkage=self.batch_linkage,
                presupported_root=self.canonical,
                output_root=self.output,
                scope_confirmed=True,
            )

        self.assertEqual(result.blockers, ("geometry_corpus_not_separate",))

    def test_scope_confirmation_and_create_only_output_fail_closed(self):
        with patch.object(reconciliation, "EXPECTED_EXISTING_PRESUPPORTED_STL_COUNT", 3):
            blocked = reconciliation.reconcile_surface_signature_inventory(
                training_records=self.training,
                validation_records=self.validation,
                dataset_manifest=self.dataset_manifest,
                dataset_provenance=self.dataset_provenance,
                batch_result=self.batch,
                batch_linkage=self.batch_linkage,
                presupported_root=self.geometry,
                output_root=self.output,
                scope_confirmed=False,
            )
        self.assertEqual(blocked.blockers, ("presupported_scope_confirmation_required",))
        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            self.run_reconciliation()

    def test_cli_has_no_test_records_or_modeling_inputs(self):
        self.assertEqual(
            set(inspect.signature(reconciliation.reconcile_surface_signature_inventory).parameters),
            {"training_records", "validation_records", "dataset_manifest",
             "dataset_provenance", "batch_result", "batch_linkage",
             "presupported_root", "output_root", "scope_confirmed"},
        )
        parser = reconciliation.build_parser()
        self.assertEqual(
            {action.dest for action in parser._actions},
            {"help", "training_records", "validation_records", "dataset_manifest",
             "dataset_provenance", "batch_result", "batch_linkage",
             "presupported_root", "output_root", "scope_confirmed"},
        )
        for flag in ("--test-records", "--labels", "--model", "--source-group"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args([
                    "--training-records", "train", "--validation-records", "validation",
                    "--dataset-manifest", "manifest",
                    "--dataset-provenance", "provenance",
                    "--batch-result", "batch", "--batch-linkage", "linkage",
                    "--presupported-root", "geometry", "--output-root", "output",
                    "--scope-confirmed", flag, "unused",
                ])


if __name__ == "__main__":
    unittest.main()
