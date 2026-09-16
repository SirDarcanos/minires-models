from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from minires.preparation import surface_signature_batch_linkage as linkage


class SurfaceSignatureBatchLinkageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "private"
        self.private.mkdir()
        self.batch = self.private / "batch-result.json"
        self.output = self.private / "batch-linkage.json"
        self.payload = {
            "schema_version": 1,
            "outcome": "completed",
            "accounting": {"inventory_count": 2, "accepted_count": 1, "rejected_count": 1},
            "accepted_records": [{
                "entry_id": "accepted-id",
                "record": {
                    "sliced_resin_mass_g": 12.345,
                    "private_label_canary": "must-not-be-decoded-or-copied",
                },
            }],
            "inventory": [
                {"entry_id": "accepted-id", "relative_path": "first.stl",
                 "byte_count": 10, "sha256": "a" * 64, "status": "candidate"},
                {"entry_id": "rejected-id", "relative_path": "second.stl",
                 "byte_count": 20, "sha256": "b" * 64, "status": "rejected"},
            ],
            "rejected_entries": [{"entry_id": "rejected-id", "reason": "bounded"}],
        }
        self.batch.write_text(json.dumps(self.payload, indent=2, sort_keys=True))

    def test_exports_only_checksum_bound_inventory_and_aggregate_accounting(self):
        result = linkage.export_batch_identity_linkage(
            batch_result=self.batch,
            private_output=self.output,
        )

        self.assertEqual(result.status, "completed")
        exported = json.loads(self.output.read_text())
        self.assertEqual(exported["version"], linkage.VERSION)
        self.assertEqual(exported["source_batch_sha256"], sha256(self.batch.read_bytes()).hexdigest())
        self.assertEqual(exported["inventory"], self.payload["inventory"])
        self.assertEqual(exported["accounting"], self.payload["accounting"])
        rendered = self.output.read_text()
        self.assertNotIn("accepted_records", rendered)
        self.assertNotIn("sliced_resin_mass_g", rendered)
        self.assertNotIn("private_label_canary", rendered)
        self.assertNotIn("rejected_entries", rendered)

    def test_top_level_selector_ignores_nested_key_canaries(self):
        self.payload["accepted_records"][0]["record"]["inventory"] = ["nested-canary"]
        self.batch.write_text(json.dumps(self.payload, indent=2, sort_keys=True))

        linkage.export_batch_identity_linkage(
            batch_result=self.batch,
            private_output=self.output,
        )

        self.assertNotIn("nested-canary", self.output.read_text())

    def test_output_is_private_and_create_only(self):
        linkage.export_batch_identity_linkage(
            batch_result=self.batch,
            private_output=self.output,
        )
        with self.assertRaisesRegex(Exception, "private_output_unavailable"):
            linkage.export_batch_identity_linkage(
                batch_result=self.batch,
                private_output=self.output,
            )
        with self.assertRaisesRegex(Exception, "private_output_required"):
            linkage.export_batch_identity_linkage(
                batch_result=self.batch,
                private_output=Path.cwd() / "public-linkage.json",
            )

    def test_cli_accepts_no_partition_test_or_model_inputs(self):
        self.assertEqual(
            set(inspect.signature(linkage.export_batch_identity_linkage).parameters),
            {"batch_result", "private_output"},
        )
        parser = linkage.build_parser()
        self.assertEqual(
            {action.dest for action in parser._actions},
            {"help", "batch_result", "private_output"},
        )
        for flag in ("--training-records", "--validation-records", "--test-records",
                     "--model", "--labels"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args([
                    "--batch-result", str(self.batch),
                    "--private-output", str(self.output),
                    flag, "unused",
                ])


if __name__ == "__main__":
    unittest.main()
