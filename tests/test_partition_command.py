from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SourceBalancedPartitionCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "private" / "current-partitions"

    def write_records(self, records):
        path = self.root / "harmonized.json"
        path.write_text(json.dumps(records))
        return path

    def run_command(self, records, *, seed=23, excluded_source="omit-canary", check=True):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "minires_evaluation.partition",
                "--records",
                str(records),
                "--private-dir",
                str(self.output),
                "--seed",
                str(seed),
                "--exclude-source",
                excluded_source,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    @staticmethod
    def records():
        rows = []
        for source in ("source-a", "source-b", "omit-canary"):
            for index in range(20):
                rows.append({
                    "_id": f"{source}-record-{index}",
                    "anonymous_source_group": source,
                    "volume_mm3": 1000.0 + index,
                    "surface_area_mm2": 100.0 + index,
                    "bounding_box_x_mm": 10.0,
                    "bounding_box_y_mm": 11.0,
                    "bounding_box_z_mm": 12.0,
                    "bounding_box_volume_mm3": 1320.0,
                    "euler_number": 2,
                    "sliced_resin_mass_g": 1.1 + index,
                })
        return rows

    def test_command_writes_balanced_disjoint_private_partitions(self):
        records = self.write_records(self.records())

        completed = self.run_command(records)

        self.assertEqual(completed.stdout, "")
        partitions = {
            name: [json.loads(line) for line in (self.output / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation", "test")
        }
        identities = {
            name: {row["_id"] for row in rows}
            for name, rows in partitions.items()
        }
        self.assertFalse(identities["train"] & identities["validation"])
        self.assertFalse(identities["train"] & identities["test"])
        self.assertFalse(identities["validation"] & identities["test"])
        self.assertEqual(set.union(*identities.values()), {
            f"{source}-record-{index}"
            for source in ("source-a", "source-b")
            for index in range(20)
        })
        for rows in partitions.values():
            self.assertNotIn("omit-canary", json.dumps(rows))
        for source in ("source-a", "source-b"):
            self.assertEqual(sum(row["anonymous_source_group"] == source for row in partitions["train"]), 14)
            self.assertEqual(sum(row["anonymous_source_group"] == source for row in partitions["validation"]), 3)
            self.assertEqual(sum(row["anonymous_source_group"] == source for row in partitions["test"]), 3)

        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["row_accounting"], {
            "input": 60,
            "excluded": 20,
            "eligible": 40,
            "train": 28,
            "validation": 6,
            "test": 6,
        })
        self.assertNotIn("omit-canary", json.dumps(manifest))
        self.assertEqual(set(manifest["artifacts"]), {"train.jsonl", "validation.jsonl", "test.jsonl"})
        self.assertEqual(manifest["artifacts"], {
            name: sha256((self.output / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
        })
        forbidden = {"anonymous_source_group", "partition", "_id", "record_identity", "duplicate_group"}
        self.assertFalse(forbidden & set(manifest["feature_schema"]["prediction_features"]))

    def memberships(self):
        return {
            json.loads(line)["_id"]: name
            for name in ("train", "validation", "test")
            for line in (self.output / f"{name}.jsonl").read_text().splitlines()
        }

    def test_membership_is_identity_based_and_explicit_duplicates_stay_together(self):
        rows = self.records()
        rows[0]["duplicate_group"] = "exact-duplicate-evidence"
        rows[1]["duplicate_group"] = "exact-duplicate-evidence"
        records = self.write_records(rows)
        self.run_command(records)
        first = self.memberships()

        rows[0]["sliced_resin_mass_g"] = 9999.25
        rows[0]["volume_mm3"] = 0.125
        rows[1]["surface_area_mm2"] = 8000.5
        records.write_text(json.dumps(list(reversed(rows))))
        self.run_command(records)
        second = self.memberships()

        self.assertEqual(first, second)
        self.assertEqual(second["source-a-record-0"], second["source-a-record-1"])
        self.assertFalse(any("miniature_family" in row for row in rows))

    def test_failed_intentional_rerun_leaves_previous_complete_set_intact(self):
        records = self.write_records(self.records())
        self.run_command(records)
        before = {
            path.name: path.read_bytes()
            for path in self.output.iterdir()
            if path.is_file()
        }
        records.write_text("not valid json")

        failed = self.run_command(records, check=False)

        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "")
        self.assertNotIn(str(records), failed.stderr)
        self.assertEqual(before, {
            path.name: path.read_bytes()
            for path in self.output.iterdir()
            if path.is_file()
        })


if __name__ == "__main__":
    unittest.main()
