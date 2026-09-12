from hashlib import sha256
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest

from minires.preparation.assemble import main as assemble_main


class ExpandedDatasetAssemblyCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.private.mkdir()
        self.historical = self.private / "historical.json"
        self.new_batch = self.private / "new-batch.json"
        self.output = self.private / "current-dataset"

    @staticmethod
    def measurement_record(identity, source, index):
        volume = 1000.123456789 + index
        surface = 400.987654321 + index
        return {
            "_id": identity,
            "artist": source,
            "file": f"/private/{source}/filename-{index}.stl",
            "kb": 10.125 + index,
            "volume": volume,
            "surface_area": surface,
            "bbox_x": 10.1,
            "bbox_y": 11.2,
            "bbox_z": 12.3,
            "bbox_area": 1385.376,
            "mass": volume,
            "euler_number": 2,
            "scale": 19.75,
            "surface_volume_ratio": 999.0,
            "weight": 1.23456789012345 + index,
        }

    def fixtures(self):
        historical = []
        for source in ("private-source-a", "private-source-b", "private-source-c"):
            historical.extend(
                self.measurement_record(f"{source}-{index}", source, index)
                for index in range(20)
            )
        historical[0]["_id"] = {"$oid": "historical-object-id"}
        historical[1]["_id"] = None
        historical[1]["record_identity"] = "historical-fallback-id"
        historical[1]["file"] = historical[0]["file"]
        historical[40]["weight"] = 0.0
        historical.extend(
            self.measurement_record(f"omit-{index}", "private-source-to-omit", index)
            for index in range(34)
        )
        unusable = self.measurement_record("bad-history", "private-source-a", 99)
        unusable["volume"] = None
        historical.append(unusable)

        accepted = []
        for index in range(20):
            record = self.measurement_record(f"ignored-{index}", "new-private-source", index)
            record.update({
                "volume_mm3": record["volume"],
                "surface_area_mm2": record["surface_area"],
                "bounding_box_x_mm": record["bbox_x"],
                "bounding_box_y_mm": record["bbox_y"],
                "bounding_box_z_mm": record["bbox_z"],
                "bounding_box_volume_mm3": record["bbox_area"],
                "mesh_mass_at_unit_density": record["mass"],
                "euler_characteristic": record["euler_number"],
                "mesh_scale_mm": record["scale"],
                "sliced_resin_mass_g": record["weight"],
            })
            accepted.append({
                "entry_id": f"new-entry-{index}",
                "input_sha256": f"digest-{index}",
                "duplicate_group": "duplicate-pair" if index in (0, 1) else None,
                "record": record,
            })
        batch = {
            "schema_version": 1,
            "outcome": "completed",
            "inventory": [
                *[{"entry_id": f"new-entry-{index}", "status": "candidate"} for index in range(20)],
                {"entry_id": "new-rejected", "status": "rejected"},
            ],
            "accepted_records": accepted,
            "rejected_entries": [{"entry_id": "new-rejected", "reason": "corrupt_geometry"}],
            "accounting": {"inventory_count": 21, "accepted_count": 20, "rejected_count": 1},
            "provenance": {"contract": {"profile_sha256": "profile-digest"}, "tool_versions": {"prusaslicer": "test"}},
            "resources": {"elapsed_seconds": 1.25},
        }
        return historical, batch

    def run_command(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = assemble_main([
                "--historical-records", str(self.historical),
                "--exclude-historical-source", "private-source-to-omit",
                "--new-batch-result", str(self.new_batch),
                "--seed", "23",
                "--private-dir", str(self.output),
            ])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_assembles_four_sources_into_one_reconciled_atomic_package(self):
        historical, batch = self.fixtures()
        self.historical.write_text(json.dumps(historical))
        self.new_batch.write_text(json.dumps(batch))

        status, stdout, stderr = self.run_command()

        self.assertEqual((status, stdout, stderr), (0, "completed\n", ""))
        partitions = {
            name: [json.loads(line) for line in (self.output / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation", "test")
        }
        all_rows = [row for rows in partitions.values() for row in rows]
        self.assertEqual({name: len(rows) for name, rows in partitions.items()}, {
            "train": 56, "validation": 12, "test": 12,
        })
        self.assertEqual(len({row["anonymous_source_group"] for row in all_rows}), 4)
        self.assertEqual(sum(row["sliced_resin_mass_g"] == 0.0 for row in all_rows), 1)
        self.assertFalse(any("miniature_family" in row for row in all_rows))
        self.assertFalse(any("artist" in row or "file" in row for row in all_rows))
        self.assertFalse(any("input_inventory" in row or "input_sha256" in row for row in all_rows))
        precise = [
            row for row in all_rows
            if row["sliced_resin_mass_g"] == 1.23456789012345
        ]
        self.assertEqual(len(precise), 3)
        for row in precise:
            self.assertEqual(row["surface_to_volume_ratio_per_mm"],
                             400.987654321 / 1000.123456789)
            self.assertEqual(row["surface_volume_ratio"],
                             400.987654321 / 1000.123456789)
        duplicate_partitions = {}
        for name, rows in partitions.items():
            for row in rows:
                if "duplicate_group" in row:
                    duplicate_partitions.setdefault(row["duplicate_group"], set()).add(name)
        self.assertEqual(len(duplicate_partitions), 2)
        self.assertTrue(all(len(names) == 1 for names in duplicate_partitions.values()))

        rejected = [json.loads(line) for line in (self.output / "rejected.jsonl").read_text().splitlines()]
        self.assertEqual(len(rejected), 2)
        self.assertEqual({row["origin"] for row in rejected}, {"historical", "new"})
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["row_accounting"], {
            "historical_input": 95,
            "historical_source_excluded": 34,
            "historical_unusable": 1,
            "new_inventory": 21,
            "new_accepted": 20,
            "new_rejected": 1,
            "eligible": 80,
            "rejected": 2,
            "train": 56,
            "validation": 12,
            "test": 12,
        })
        self.assertEqual(manifest["retained_anonymous_source_count"], 4)
        forbidden = {"anonymous_source_group", "artist", "source", "partition", "_id", "record_identity", "duplicate_group"}
        self.assertFalse(forbidden & set(manifest["feature_schema"]["prediction_features"]))
        self.assertIn("performance_on_held_out_stl_rows_drawn_from_retained_sources", manifest["claim_scope"])
        self.assertNotIn("unseen_source_performance", manifest["claim_scope"])
        checksums = json.loads((self.output / "checksums.json").read_text())
        for name, digest in checksums["artifacts"].items():
            self.assertEqual(digest, sha256((self.output / name).read_bytes()).hexdigest())

        private_canaries = (
            "private-source-a", "private-source-b", "private-source-c",
            "private-source-to-omit", "/private/", "filename-",
        )
        self.assertFalse(any(canary in stdout + stderr for canary in private_canaries))
        artifact_text = "".join(path.read_text() for path in self.output.iterdir() if path.is_file())
        self.assertFalse(any(canary in artifact_text for canary in private_canaries))

    def test_mismatched_new_inventory_fails_closed(self):
        historical, batch = self.fixtures()
        batch["inventory"].pop()
        self.historical.write_text(json.dumps(historical))
        self.new_batch.write_text(json.dumps(batch))

        status, stdout, stderr = self.run_command()

        self.assertEqual((status, stdout, stderr), (2, "", "dataset_assembly_failed\n"))
        self.assertFalse(self.output.exists())

    def test_cross_source_duplicate_evidence_fails_closed(self):
        historical, batch = self.fixtures()
        historical[2]["duplicate_group"] = "cross-source-duplicate"
        historical[22]["duplicate_group"] = "cross-source-duplicate"
        self.historical.write_text(json.dumps(historical))
        self.new_batch.write_text(json.dumps(batch))

        status, stdout, stderr = self.run_command()

        self.assertEqual((status, stdout, stderr), (2, "", "dataset_assembly_failed\n"))
        self.assertFalse(self.output.exists())

    def test_invalid_rerun_leaves_previous_complete_package_intact(self):
        historical, batch = self.fixtures()
        self.historical.write_text(json.dumps(historical))
        self.new_batch.write_text(json.dumps(batch))
        self.assertEqual(self.run_command()[0], 0)
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.new_batch.write_text("not-json")

        status, stdout, stderr = self.run_command()

        self.assertEqual((status, stdout), (2, ""))
        self.assertEqual(stderr, "dataset_assembly_failed\n")
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})


if __name__ == "__main__":
    unittest.main()
