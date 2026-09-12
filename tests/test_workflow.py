from hashlib import sha256
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires import EvaluationConfig, LegacyProvenance, LegacyReference
from minires.modeling.tuning import CandidateFoldFit, LockedFit, SearchLimits
from minires.evaluation.workflow import main as workflow_main, run_end_to_end_workflow


class SyntheticWorkflowRuntime:
    dependency_versions = {"runtime": "synthetic-1"}

    def __init__(self):
        self.fit_count = 0
        self.refit_count = 0

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.fit_count += 1
        return CandidateFoldFit(
            predictor=lambda rows: [row[1] / 1000.0 for row in rows],
            metadata={"selected_epochs": 4, "selected_trees": 20},
            fitted_state={"candidate": candidate.candidate_id, "seed": seed},
        )

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.refit_count += 1
        return LockedFit(
            predictor=lambda rows: [row[1] / 1000.0 for row in rows],
            preprocessing_state={"fit_rows": len(features)},
            artifacts={"model.bin": candidate.candidate_id.encode()},
            metadata={"seed": seed},
        )

    def load_locked(self, candidate, directory, contract):
        return lambda rows: [row[1] / 1000.0 for row in rows]

    def serialize_fold(self, fitted):
        return {"model.bin": str(fitted.fitted_state).encode()}


class EndToEndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.development = self.root / "development.json"
        self.final = self.root / "final.json"
        self.legacy_artifacts = self.root / "legacy"
        self.legacy_artifacts.mkdir()
        self.slicing = self.root / "slicing.json"
        self.slicing.write_text(json.dumps({
            "volume_unit": "mm3",
            "slicing_conditions": {"layer_height_mm": 0.05},
        }))
        self.development.write_text(json.dumps([
            self.row(source, family, index + 1)
            for index, (source, family) in enumerate((
                ("development-a", "a1"), ("development-a", "a2"),
                ("development-b", "b1"), ("development-b", "b2"),
                ("development-c", "c1"), ("development-c", "c2"),
            ))
        ]))
        predictor = lambda rows: [row[1] / 1000.0 + 0.5 for row in rows]
        self.legacy = LegacyReference.from_predictors(
            neural_network=predictor,
            xgboost=predictor,
            neural_network_weight=0.2,
            provenance=LegacyProvenance.unknown(),
        )

    @staticmethod
    def row(source, family, value):
        return {
            "kb": value,
            "volume": value * 1000,
            "surface_area": value * 100,
            "bbox_area": value * 1100,
            "euler_number": value,
            "scale": value,
            "surface_volume_ratio": 0.1,
            "weight": value,
            "anonymous_source_group": source,
            "miniature_family": family,
            "slicing_conditions": {"layer_height_mm": 0.05},
        }

    def final_rows(self, sources=("final-a", "final-b", "final-c")):
        return [
            self.row(source, f"{source}-family", source_index * 200 + index + 1)
            for source_index, source in enumerate(sources)
            for index in range(200)
        ]

    def test_command_runs_the_successful_lifecycle_and_creates_screened_evidence(self):
        self.final.write_text(json.dumps(self.final_rows()))
        originals = {
            path: path.read_bytes()
            for path in (self.development, self.final, self.slicing)
        }
        runtime = SyntheticWorkflowRuntime()
        output = self.root / "private" / "workflow-success"
        stdout = io.StringIO()

        with patch(
            "minires.evaluation.workflow.TensorflowXGBoostCandidateRuntime",
            return_value=runtime,
        ), patch(
            "minires.evaluation.workflow.load_legacy_reference",
            return_value=self.legacy,
        ), contextlib.redirect_stdout(stdout):
            code = workflow_main([
                "--development-records", str(self.development),
                "--final-records", str(self.final),
                "--legacy-artifacts", str(self.legacy_artifacts),
                "--slicing-configuration", str(self.slicing),
                "--output-root", str(output),
                "--volume-unit", "mm3",
                "--scope-confirmed",
                "--seed", "41",
                "--second-seed", "42",
                "--bootstrap-seed", "1729",
                "--maximum-candidate-runs", "20",
                "--maximum-elapsed-seconds", "7200",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "completed")
        evidence = json.loads((output / "evidence-index.json").read_text())
        self.assertEqual(evidence["phases"]["tuning"]["completed_candidate_runs"], 20)
        self.assertTrue(evidence["phases"]["tuning"]["candidate_locking_completed"])
        self.assertEqual(
            evidence["phases"]["assessment"]["promotion_decision"],
            "promoted_for_internal_advisory_use",
        )
        self.assertTrue(evidence["input_integrity"]["unchanged"])
        self.assertEqual(
            json.loads((output / "public-summary-review.json").read_text())["status"],
            "automated_screening_passed",
        )
        public_text = (output / "public-summary-draft.json").read_text()
        for marker in ("development-a", "final-a", str(self.root), "fingerprint", "sha256"):
            self.assertNotIn(marker, public_text)
        manifest = json.loads((output / "manifest.json").read_text())
        for name, checksum in manifest["artifacts"].items():
            self.assertEqual(sha256((output / name).read_bytes()).hexdigest(), checksum)
        self.assertFalse(manifest["publication_performed"])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_insufficient_final_sources_preserves_valid_tuning_and_blocks_promotion(self):
        self.final.write_text(json.dumps(self.final_rows(("only-final-source",))))
        runtime = SyntheticWorkflowRuntime()
        output = self.root / "private" / "workflow-blocked"

        evidence = run_end_to_end_workflow(
            development_records=self.development,
            final_records=self.final,
            legacy_artifacts=self.legacy_artifacts,
            slicing_configuration=self.slicing,
            output_root=output,
            evaluation_config=EvaluationConfig(None, "mm3", True, seed=41),
            bootstrap_seed=1729,
            runtime=runtime,
            limits=SearchLimits(seed=41, second_seed=42),
            legacy_reference=self.legacy,
        )

        self.assertEqual(evidence["status"], "completed_with_blockers")
        self.assertEqual(evidence["phases"]["tuning"]["status"], "completed")
        self.assertTrue(evidence["phases"]["tuning"]["candidate_locking_completed"])
        self.assertEqual(evidence["phases"]["assessment"]["status"], "blocked")
        self.assertIn("insufficient_final_source_groups", evidence["blockers"])
        self.assertEqual(
            evidence["phases"]["assessment"]["promotion_decision"], "blocked"
        )
        self.assertTrue((output / "tuning" / "locked-candidate" / "lock-manifest.json").exists())
        assessment = json.loads((output / "assessment" / "assessment.json").read_text())
        self.assertEqual(assessment["predictions"], [])

    def test_interrupted_search_keeps_truthful_partial_accounting_without_reading_final_input(self):
        self.final.write_text(json.dumps(self.final_rows()))
        output = self.root / "private" / "workflow-interrupted"

        with patch(
            "minires.evaluation.workflow.tune_candidates", side_effect=KeyboardInterrupt
        ):
            evidence = run_end_to_end_workflow(
                development_records=self.development,
                final_records=self.final,
                legacy_artifacts=self.legacy_artifacts,
                slicing_configuration=self.slicing,
                output_root=output,
                evaluation_config=EvaluationConfig(None, "mm3", True, seed=41),
                bootstrap_seed=1729,
                runtime=SyntheticWorkflowRuntime(),
                limits=SearchLimits(seed=41, second_seed=42),
            )

        self.assertEqual(evidence["status"], "blocked")
        self.assertIn("workflow_interrupted", evidence["blockers"])
        self.assertEqual(evidence["phases"]["tuning"]["status"], "interrupted")
        self.assertIsNone(evidence["phases"]["tuning"]["completed_candidate_runs"])
        self.assertEqual(
            evidence["input_integrity"]["before"]["final_records"]["status"],
            "not_accessed_before_lock",
        )
        self.assertTrue((output / "manifest.json").exists())

    def test_assessment_only_does_not_fit_or_select_candidates(self):
        self.final.write_text(json.dumps(self.final_rows()))
        runtime = SyntheticWorkflowRuntime()
        tuning_output = self.root / "private" / "workflow-for-lock"
        run_end_to_end_workflow(
            development_records=self.development,
            final_records=self.final,
            legacy_artifacts=self.legacy_artifacts,
            slicing_configuration=self.slicing,
            output_root=tuning_output,
            evaluation_config=EvaluationConfig(None, "mm3", True, seed=41),
            bootstrap_seed=1729,
            runtime=runtime,
            limits=SearchLimits(seed=41, second_seed=42),
            legacy_reference=self.legacy,
        )
        fit_count = runtime.fit_count
        refit_count = runtime.refit_count

        evidence = run_end_to_end_workflow(
            locked_candidate=tuning_output / "tuning" / "locked-candidate",
            final_records=self.final,
            legacy_artifacts=self.legacy_artifacts,
            slicing_configuration=self.slicing,
            output_root=self.root / "private" / "assessment-only",
            evaluation_config=EvaluationConfig(None, "mm3", True, seed=41),
            bootstrap_seed=1729,
            runtime=runtime,
            legacy_reference=self.legacy,
        )

        self.assertEqual(evidence["mode"], "assessment_only")
        self.assertEqual(evidence["phases"]["tuning"]["status"], "not_run")
        self.assertEqual(runtime.fit_count, fit_count)
        self.assertEqual(runtime.refit_count, refit_count)


if __name__ == "__main__":
    unittest.main()
