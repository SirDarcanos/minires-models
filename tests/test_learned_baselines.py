import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from minires import (
    EvaluationConfig, LearnedBaseline, LearnedBaselineConfig, evaluate_records,
)
from minires.modeling.learned import FittedFold, enable_synchronous_dataset_execution
from minires.__main__ import main


class RecordingRuntime:
    dependency_versions = {"runtime": "recording-1"}

    def __init__(self):
        self.calls = []

    def fit_fold(self, train_features, train_targets, validation_features, validation_targets,
                 config, artifact_dir):
        self.calls.append({
            "train_features": tuple(train_features),
            "train_targets": tuple(train_targets),
            "validation_features": tuple(validation_features),
            "validation_targets": tuple(validation_targets),
            "artifact_dir": artifact_dir,
        })
        train_mean = sum(train_targets) / len(train_targets)
        return FittedFold(
            neural_network=lambda rows: [train_mean for _ in rows],
            xgboost=lambda rows: [row[1] / 1000 for row in rows],
            metadata={"normalization_mean": [sum(column) / len(train_features)
                                               for column in zip(*train_features)],
                      "neural_network_epochs": 2, "xgboost_best_iteration": 3},
        )


class LearnedBaselineInterfaceTests(unittest.TestCase):
    def test_synchronous_dataset_setup_fails_closed_after_tensorflow_initialization(self):
        tensorflow = types.ModuleType("tensorflow")
        tensorflow.data = types.SimpleNamespace(
            experimental=types.SimpleNamespace(
                enable_debug_mode=lambda: (_ for _ in ()).throw(ValueError("already initialized"))
            )
        )

        with patch.dict(sys.modules, {"tensorflow": tensorflow}):
            enabled = enable_synchronous_dataset_execution()

        self.assertFalse(enabled)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / "private" / "splits.json"
        self.config = EvaluationConfig(None, "mm3", True, seed=17)
        self.rows = [self.row(source, family, index + 1, target)
                     for index, (source, family, target) in enumerate([
                         ("a", "a1", 2.0), ("a", "a2", 4.0),
                         ("b", "b1", 1.0), ("c", "c1", 3.0),
                     ])]

    @staticmethod
    def row(source, family, value, target):
        return {
            "kb": value * 10, "volume": value * 1000,
            "surface_area": value * 100, "bbox_area": value * 1100,
            "euler_number": value, "scale": value * 2,
            "surface_volume_ratio": 0.1, "weight": target,
            "anonymous_source_group": source, "miniature_family": family,
        }

    def evaluate(self, runtime, rows=None, output_dir=None):
        return evaluate_records(
            self.rows if rows is None else rows,
            self.config,
            LearnedBaseline(runtime=runtime),
            split_manifest=self.manifest,
            output_dir=output_dir,
        )

    def test_fits_each_fold_only_on_train_and_uses_common_validation_for_selection(self):
        runtime = RecordingRuntime()
        result = self.evaluate(runtime)

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(runtime.calls), 3)
        private = result.to_dict()["grouped_evaluation"]
        self.assertEqual(result.metrics.sample_count, 4)
        self.assertEqual(set(result.model_diagnostics), {"neural_network", "xgboost", "ensemble"})
        self.assertEqual(set(result.to_dict(public=True)["grouped_evaluation"]["component_source_balanced"]),
                         {"neural_network", "xgboost", "ensemble"})
        for fold in private["source_reports"]:
            audit = fold["fit_audit"]
            self.assertFalse(set(audit["train_rows"]) & set(audit["validation_rows"]))
            self.assertFalse(set(audit["train_rows"]) & set(audit["test_rows"]))
            self.assertEqual(audit["ensemble_selection_rows"], audit["validation_rows"])
            self.assertEqual(set(fold["predictions"]), {"neural_network", "xgboost", "ensemble"})
        self.assertEqual(result.model_contract["ensemble"]["candidate_neural_network_weights"], [0.2])
        self.assertEqual(result.model_contract["preprocessing"]["normalization_fit_partition"], "fold_train_only")
        self.assertNotIn("no_fitting_or_model_selection", private["limitations"])

    def test_outer_holdout_changes_cannot_change_that_folds_fitted_state(self):
        first_runtime = RecordingRuntime()
        first = self.evaluate(first_runtime)
        first_report = next(report for report in first.to_dict()["grouped_evaluation"]["source_reports"]
                            if 0 in report["fit_audit"]["test_rows"])

        changed = [dict(row) for row in self.rows]
        changed[0]["weight"] = 999.0
        changed[0]["volume"] = 999000.0
        self.manifest = self.manifest.with_name("changed.json")
        second_runtime = RecordingRuntime()
        second = self.evaluate(second_runtime, changed)
        second_report = next(report for report in second.to_dict()["grouped_evaluation"]["source_reports"]
                             if 0 in report["fit_audit"]["test_rows"])

        self.assertEqual(first_report["fit_audit"]["fitted_state_fingerprint"],
                         second_report["fit_audit"]["fitted_state_fingerprint"])
        self.assertNotEqual(first_report["fit_audit"]["test_data_fingerprint"],
                            second_report["fit_audit"]["test_data_fingerprint"])

    def test_missing_supported_dependencies_returns_bounded_blocker(self):
        with patch("minires.modeling.learned.TensorflowXGBoostRuntime", side_effect=ImportError):
            result = self.evaluate(None)

        self.assertEqual(result.status, "blocked")
        self.assertIn("learned_baseline_dependencies_required", result.blockers)
        self.assertEqual(result.predictions, ())

    def test_rejects_changes_to_the_verified_fixed_configuration(self):
        result = evaluate_records(
            self.rows, self.config,
            LearnedBaseline(config=LearnedBaselineConfig(xgboost_n_jobs=-1),
                            runtime=RecordingRuntime()),
            split_manifest=self.manifest,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("invalid_learned_baseline_configuration",))

    def test_cli_reports_unavailable_optional_runtime_honestly(self):
        records = Path(self.temp.name) / "records.json"
        records.write_text(json.dumps(self.rows))
        output = io.StringIO()
        with patch("minires.modeling.learned.TensorflowXGBoostRuntime", side_effect=ImportError), \
             contextlib.redirect_stdout(output):
            code = main(["--records", str(records), "--volume-unit", "mm3",
                         "--scope-confirmed", "--split-manifest", str(self.manifest),
                         "--learned-baselines", "--public"])
        report = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["blockers"], ["learned_baseline_dependencies_required"])

    def test_private_artifacts_include_reproducibility_record_and_public_output_is_aggregate_only(self):
        runtime = RecordingRuntime()
        output = Path(self.temp.name) / "private" / "learned-run"
        result = self.evaluate(runtime, output_dir=output)
        manifest = json.loads((output / "manifest.json").read_text())
        private = result.to_dict()
        public = result.to_dict(public=True)

        self.assertIn("split_fingerprint", private["model_contract"]["run"])
        self.assertNotIn("split_fingerprint", public["model_contract"]["run"])
        self.assertEqual(manifest["learned_baseline"]["dependency_versions"], {"runtime": "recording-1"})
        self.assertIn("runtime_seconds", private["model_contract"]["run"])
        self.assertNotIn("source_reports", public["grouped_evaluation"])
        self.assertNotIn("fit_audit", str(public))
        self.assertTrue(all("private" in str(call["artifact_dir"]) for call in runtime.calls))


if __name__ == "__main__":
    unittest.main()
