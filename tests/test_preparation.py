import json
from pathlib import Path
import tempfile
import unittest

from minires_evaluation.prepare import main as prepare_main
from minires_evaluation import (
    EvaluationConfig,
    PhysicalBaseline,
    evaluate_records,
    prepare_private_dataset,
)


class PrivateDatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def records(self):
        rows = []
        for source in ("source-canary-a", "source-canary-b"):
            for family_number in (1, 2):
                family = f"family-canary-{family_number}"
                rows.append({
                    "_id": f"id-{source}-{family}",
                    "artist": source,
                    "mini": family,
                    "file": f"/private-root/{source}/{family}/parts/model.stl",
                    "kb": 10.25,
                    "volume": 1234.56789012345 + family_number,
                    "surface_area": 400.125,
                    "bbox_x": 1.25,
                    "bbox_y": 2.5,
                    "bbox_z": 3.75,
                    "bbox_area": 11.71875,
                    "mass": 1.3579,
                    "euler_number": 2,
                    "scale": 1,
                    "surface_volume_ratio": 0.25,
                    "weight": 1.23456789012345 + family_number,
                })
        return rows

    def test_prepares_private_evaluator_input_without_identity_or_precision_loss(self):
        records = self.records()
        records[0].pop("surface_volume_ratio")
        original = json.loads(json.dumps(records))
        output = self.root / "private" / "prepared-a"

        result = prepare_private_dataset(records, output_dir=output)

        self.assertEqual(records, original)
        self.assertEqual(result.input_count, 4)
        self.assertEqual(result.included_count, 4)
        prepared = [json.loads(line) for line in (output / "prepared-records.jsonl").read_text().splitlines()]
        self.assertEqual(prepared[0]["sliced_resin_mass_g"], 2.23456789012345)
        self.assertEqual(prepared[0]["volume"], 1235.56789012345)
        self.assertEqual(prepared[0]["surface_volume_ratio"], 400.125 / 1235.56789012345)
        self.assertTrue(all(row["scope_confirmed"] for row in prepared))
        self.assertTrue(all(row["resin_density_g_per_ml"] == 1.1 for row in prepared))
        self.assertTrue(all(row["slicing_conditions"]["layer_height_mm"] == 0.05 for row in prepared))
        serialized = "".join(path.read_text(errors="ignore") for path in output.rglob("*") if path.is_file())
        for canary in ("source-canary", "family-canary", "private-root"):
            self.assertNotIn(canary, serialized)

        evaluated = evaluate_records(
            output / "prepared-records.jsonl",
            EvaluationConfig(1.1, "mm3", True, seed=17),
            PhysicalBaseline(),
            split_manifest=output / "verification-splits.json",
        )
        self.assertEqual(evaluated.split_status, "frozen_source_holdout")
        self.assertEqual(evaluated.data_quality.accepted_count, 4)

        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["ebminimanager_revision"], "1a841195813136ee3b380ab1d192727f385f7a55")
        self.assertEqual(provenance["profile_sha256"], "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e")
        self.assertFalse(provenance["slicer_added_supports"])
        self.assertTrue(provenance["pre_supported_scope_attested"])
        coverage = json.loads((output / "coverage.json").read_text())
        self.assertTrue(coverage["sufficient_for_frozen_evaluation_folds"])
        self.assertEqual(coverage["eligible_source_count"], 2)
        self.assertEqual(coverage["miniature_family_count"], 4)
        grouping = json.loads((output / "grouping-evidence.json").read_text())
        self.assertEqual(len(grouping["source_mapping"]), 2)
        self.assertTrue(all(set(item) == {"private_source_evidence", "anonymous_source_group"}
                            for item in grouping["source_mapping"]))

    def test_ambiguous_grouping_and_reconciliation_are_bounded_outcomes(self):
        records = self.records()
        records.append({
            **records[0],
            "_id": "ambiguous-id",
            "file": "/private-root/source/family-canary-1/family-canary-1/model.stl",
        })
        comparison = [
            dict(records[1]),
            {
                **records[1],
                "_id": None,
                "artist": "different-source-canary",
                "mini": "different-family-canary",
            },
        ]
        output = self.root / "private" / "prepared-b"

        result = prepare_private_dataset(
            records,
            comparison_records=[comparison],
            output_dir=output,
        )

        self.assertEqual(result.input_count, 5)
        self.assertEqual(result.included_count, 4)
        self.assertEqual(result.needs_review_count, 1)
        prepared = [json.loads(line) for line in (output / "prepared-records.jsonl").read_text().splitlines()]
        self.assertEqual(prepared[-1]["preparation_reasons"], ["ambiguous_family_path"])
        coverage = json.loads((output / "coverage.json").read_text())
        self.assertEqual(coverage["outcome_reasons"], {"ambiguous_family_path": 1})
        reconciliation = json.loads((output / "reconciliation-report.json").read_text())["comparisons"][0]
        self.assertFalse(reconciliation["dataset_equivalence_proven"])
        self.assertEqual(reconciliation["primary_count"], 5)
        self.assertEqual(reconciliation["comparison_count"], 2)
        primary_accounted = (
            reconciliation["matched_count"]
            + len(reconciliation["unmatched_primary_indices"])
            + len(reconciliation["ambiguous_primary_indices"])
        )
        comparison_accounted = (
            reconciliation["matched_count"]
            + len(reconciliation["unmatched_comparison_indices"])
            + len(reconciliation["ambiguous_comparison_indices"])
        )
        self.assertEqual(primary_accounted, 5)
        self.assertEqual(comparison_accounted, 2)

    def test_path_evidence_is_exact_and_distinct_pack_locations_fail_closed(self):
        records = self.records()
        records.extend([
            {
                **records[0],
                "_id": "case-mismatch",
                "mini": "CaseSensitiveFamily",
                "file": "/root/source/casesensitivefamily/model.stl",
            },
            {
                **records[0],
                "_id": "pack-a",
                "mini": "RepeatedFamily",
                "file": "/root/source/pack-a/RepeatedFamily/model.stl",
            },
            {
                **records[0],
                "_id": "pack-b",
                "mini": "RepeatedFamily",
                "file": "/root/source/pack-b/RepeatedFamily/model.stl",
            },
            {
                **records[0],
                "_id": "path-form-a",
                "mini": "PathFormA",
                "file": "/root/source/PathFormA/model.stl",
            },
            {
                **records[0],
                "_id": "path-form-b",
                "mini": "PathFormB",
                "file": "root/./source/PathFormB/model.stl",
            },
        ])
        output = self.root / "private" / "path-evidence"

        result = prepare_private_dataset(records, output_dir=output)

        self.assertEqual(result.needs_review_count, 3)
        prepared = [json.loads(line) for line in (output / "prepared-records.jsonl").read_text().splitlines()]
        self.assertEqual(prepared[4]["preparation_reasons"], ["unresolved_miniature_family"])
        self.assertEqual(prepared[5]["preparation_reasons"], ["ambiguous_family_path"])
        self.assertEqual(prepared[6]["preparation_reasons"], ["ambiguous_family_path"])
        coverage = json.loads((output / "coverage.json").read_text())
        self.assertEqual(coverage["duplicate_group_count"], 0)

    def test_invalid_numeric_evidence_remains_a_row_outcome(self):
        records = self.records()
        records.extend([
            {
                **records[0],
                "_id": "invalid-number",
                "mini": "InvalidNumberFamily",
                "file": "/root/source/InvalidNumberFamily/model.stl",
                "volume": "not-a-number-canary",
                "weight": "invalid-target-canary",
            },
            {
                **records[0],
                "_id": "non-finite",
                "mini": "NonFiniteFamily",
                "file": "/root/source/NonFiniteFamily/model.stl",
                "volume": float("inf"),
            },
        ])
        output = self.root / "private" / "invalid-evidence"

        result = prepare_private_dataset(records, output_dir=output)

        self.assertEqual(result.input_count, 6)
        self.assertEqual(result.excluded_count, 2)
        coverage = json.loads((output / "coverage.json").read_text())
        self.assertEqual(coverage["outcome_reasons"]["invalid_volume"], 1)
        self.assertEqual(coverage["outcome_reasons"]["invalid_target_sliced_resin_mass"], 1)
        self.assertEqual(coverage["outcome_reasons"]["non_finite_volume_mm3"], 1)
        serialized = (output / "prepared-records.jsonl").read_text()
        self.assertNotIn("not-a-number-canary", serialized)
        self.assertNotIn("invalid-target-canary", serialized)

    def test_file_preparation_is_byte_stable_and_cli_never_changes_inputs(self):
        records = self.root / "records.jsonl"
        records.write_text("".join(json.dumps(row) + "\n" for row in self.records()))
        comparison = self.root / "comparison.json"
        comparison.write_text(json.dumps(self.records()))
        originals = {path: path.read_bytes() for path in (records, comparison)}
        outputs = [self.root / "private" / name for name in ("repeat-a", "repeat-b")]

        for output in outputs:
            self.assertEqual(prepare_main([
                "--records", str(records),
                "--reconcile", str(comparison),
                "--private-dir", str(output),
                "--seed", "23",
            ]), 0)

        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        first = {path.name: path.read_bytes() for path in outputs[0].iterdir() if path.is_file()}
        second = {path.name: path.read_bytes() for path in outputs[1].iterdir() if path.is_file()}
        self.assertEqual(first, second)
        checksums = json.loads((outputs[0] / "checksums.json").read_text())
        self.assertTrue(checksums["input"]["unchanged"])
        self.assertTrue(checksums["comparisons"][0]["unchanged"])


if __name__ == "__main__":
    unittest.main()
