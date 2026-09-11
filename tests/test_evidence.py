import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires_evaluation import (
    EvaluationConfig,
    LegacyProvenance,
    PhysicalBaseline,
    evaluate_records,
    load_legacy_reference,
)
from minires_evaluation.evidence import (
    assess_repeatability,
    build_public_summary_draft,
    produce_evidence_package,
    review_public_summary,
)


class BaselineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "volume": 1000,
                "weight": 1.25,
                "anonymous_source_group": source,
                "miniature_family": family,
            }
            for source, family in (("a", "a1"), ("a", "a2"), ("b", "b1"), ("c", "c1"))
        ]
        self.config = EvaluationConfig(1.0, "mm3", True, seed=7)

    def test_analysis_notebook_is_an_output_free_shared_interface_caller(self):
        notebook = json.loads(
            (Path(__file__).parents[1] / "baseline_analysis.ipynb").read_text()
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertIn("evaluate_records(", code)
        self.assertIn("to_dict(public=True)", code)
        self.assertNotIn("def preprocess", code)
        self.assertNotIn("def metric", code)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_repeatability_requires_the_same_split_and_exact_physical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "private" / "split.json"
            first = evaluate_records(
                self.records, self.config, PhysicalBaseline(), split_manifest=manifest
            )
            second = evaluate_records(
                self.records, self.config, PhysicalBaseline(), split_manifest=manifest
            )

        assessment = assess_repeatability(first, second, exact_predictions=True)

        self.assertEqual(assessment["status"], "repeatable")
        self.assertTrue(assessment["split_exact"])
        self.assertTrue(assessment["predictions_within_tolerance"])
        self.assertEqual(assessment["absolute_tolerance"], 0.0)
        self.assertEqual(assessment["relative_tolerance"], 0.0)

    def test_legacy_public_contract_omits_release_location_and_artifact_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = load_legacy_reference(
                Path(directory), provenance=LegacyProvenance.unknown()
            )
            result = evaluate_records(self.records, self.config, legacy)
            contract = result.to_dict(public=True)["model_contract"]

        self.assertNotIn("repository", contract)
        self.assertNotIn("artifacts", contract)
        self.assertNotIn("revision", contract)

    def test_blocked_runs_do_not_claim_numerical_repeatability_without_predictions(self):
        blocked = evaluate_records(
            self.records,
            EvaluationConfig(None, "mm3", None),
            PhysicalBaseline(),
        )

        assessment = assess_repeatability(blocked, blocked, exact_predictions=True)

        self.assertEqual(
            assessment["status"], "structurally_repeatable_no_numerical_result"
        )
        self.assertEqual(
            assessment["prediction_comparison_status"], "not_observed_no_predictions"
        )

    def test_public_draft_is_allowlisted_and_records_limitations_without_private_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_records(
                self.records,
                self.config,
                PhysicalBaseline(),
                split_manifest=Path(directory) / "private" / "split.json",
            )
        draft = build_public_summary_draft({"physical": result})
        serialized = json.dumps(draft)

        self.assertEqual(draft["publication_status"], "draft_not_approved")
        self.assertIn("unseen_source_claim_strength", draft["limitations"])
        self.assertIn("uncertainty", draft["limitations"])
        self.assertNotIn("canonical_rows", serialized)
        self.assertNotIn("predictions", serialized)
        self.assertNotIn("input_fingerprint", serialized)
        self.assertNotIn("miniature_family", serialized)

        review = review_public_summary(draft)
        self.assertEqual(review["status"], "automated_screening_passed")
        self.assertEqual(review["findings"], [])
        self.assertFalse(review["publication_performed"])

    def test_private_package_runs_all_baselines_twice_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.json"
            records.write_text(json.dumps([
                {
                    **row,
                    "kb": 1,
                    "surface_area": 100,
                    "bbox_area": 2000,
                    "euler_number": 2,
                    "scale": 1,
                    "surface_volume_ratio": 0.1,
                }
                for row in self.records
            ]))
            original = records.read_bytes()
            comparison = root / "comparison.json"
            comparison.write_bytes(original)
            legacy = root / "legacy"
            legacy.mkdir()
            output = root / "private" / "evidence"
            # This tests package orchestration and bounded dependency handling,
            # not the optional third-party training runtime exercised by smoke tests.
            with patch(
                "minires_evaluation.learned.TensorflowXGBoostRuntime",
                side_effect=ImportError,
            ), patch(
                "minires_evaluation.evidence.enable_synchronous_dataset_execution",
                create=True,
            ) as enable_synchronous:
                evidence = produce_evidence_package(
                    records=records,
                    reconciliations=(comparison,),
                    output_root=output,
                    split_manifest=root / "private" / "split.json",
                    legacy_artifacts=legacy,
                    config=self.config,
                    verification=[{"command": "python -m unittest", "outcome": "passed"}],
                )

            enable_synchronous.assert_called_once_with()
            self.assertEqual(
                set(evidence["runs"]),
                {"physical", "legacy_reference", "clean_fixed_configuration"},
            )
            self.assertTrue(evidence["input_integrity"]["unchanged"])
            self.assertEqual(evidence["verification"][0]["outcome"], "passed")
            self.assertIn("python", evidence["environment"])
            self.assertIn("pyarrow", evidence["environment"]["dependency_versions"])
            self.assertEqual(records.read_bytes(), original)
            self.assertTrue(evidence["runs"]["physical"]["repeatability"]["split_exact"])
            for repetition in (1, 2):
                report = json.loads(
                    (output / f"physical-run-{repetition}" / "report.json").read_text()
                )
                self.assertEqual(len(report["reconciliations"]), 1)
            self.assertTrue((output / "public-summary-draft.json").exists())
            self.assertEqual(
                json.loads((output / "public-summary-review.json").read_text())["status"],
                "automated_screening_passed",
            )
            self.assertTrue((output / "evidence.json").exists())

    def test_public_review_rejects_paths_mappings_rows_and_raw_errors(self):
        unsafe = {
            "local_path": "/Users/private/input.json",
            "source_mapping": {"alias": "origin"},
            "predictions": [1.0],
            "error": "third-party detail",
        }

        review = review_public_summary(unsafe)

        self.assertEqual(review["status"], "rejected")
        self.assertEqual(
            set(review["findings"]),
            {"identifying_or_private_key", "local_path", "raw_error_detail", "row_level_output"},
        )


if __name__ == "__main__":
    unittest.main()
