"""Synthetic-only four-cell prerequisite lifecycle, isolation and failure evidence."""
import copy
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires import EvaluationConfig
from minires.ingestion import normalize
from minires.modeling import bounded_influence_training as experiment
from minires.modeling import bounded_influence_correction as numeric
from minires.modeling import tail_correction as old
from minires.modeling import correction_transition as transition
from minires.modeling import training_stability as stability
from minires.modeling.tuning import LockedFit
from test_training_stability import RecordingRuntime, row


class BoundedInfluenceTrainingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name) / "private" / "synthetic-run-017"
        self.records = [row(f"synthetic-{i}", i + 1) for i in range(50)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def run_experiment(self, runtime=None, **kwargs):
        return experiment.run_bounded_influence_training(
            self.records, self.config, runtime=runtime or RecordingRuntime(),
            output_root=self.output, clock=kwargs.pop("clock", lambda: 0.), **kwargs)

    def test_unpredeclared_normalization_is_rejected_before_output_or_fitting(self):
        for config in (EvaluationConfig(None, "cm3", True, seed=17),
                       EvaluationConfig(None, "mm3", None, seed=17),
                       EvaluationConfig(None, "mm3", True, seed=41)):
            self.config = config
            runtime = RecordingRuntime()
            with self.assertRaisesRegex(Exception, "invalid_bounded_influence_configuration"):
                self.run_experiment(runtime)
            self.assertEqual(runtime.fits, [])
            self.assertFalse(self.output.exists())

    def test_real_four_cell_success_only_supports_training_prerequisite_and_create_only(self):
        runtime = RecordingRuntime()
        with patch.object(numeric, "fit_state", wraps=numeric.fit_state) as fitted, \
             patch.object(old, "fit_state", side_effect=AssertionError("old correction must not fit")):
            result = self.run_experiment(runtime)
        self.assertEqual(result.status, "training_prerequisite_supported")
        self.assertEqual(fitted.call_count, 4)
        self.assertEqual(len(runtime.fits), 48)
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(result.resource_use["unused_fit_capacity"], 0)
        self.assertFalse(result.evidence["validation_eligible"])
        self.assertFalse(result.evidence["production_continuation"])
        self.assertEqual(result.evidence["metric_failed_cells"], [])
        self.assertEqual(result.evidence["uncompleted_cells"], [])
        for cell in result.evidence["cells"].values():
            self.assertTrue(cell["qualification"]["qualified"])
            self.assertTrue(all(cell["qualification"]["conditions"].values()))
            self.assertEqual(cell["fits"], 13)
            self.assertEqual(cell["held_out_record_count"], 10)
            self.assertEqual(cell["fitting_record_count"], 40)
            self.assertIn("paired_transitions_with_old_diagnostic_loss", cell)
            self.assertIn("new_qualification_loss", cell)
        plan = json.loads((self.output / "prerequisite-plan.json").read_text())
        self.assertEqual(plan["correction_contract"], numeric.CONTRACT)
        self.assertEqual(plan["qualification_contract"], experiment.QUALIFICATION_CONTRACT)
        self.assertEqual(plan["fit_allocation"], {"inner_oof_bases": 40, "outer_partition_bases": 8, "corrections": 4})
        self.assertEqual(plan["fixed_training_counts"], {"neural_network_epochs": 87, "xgboost_trees": 1091})
        self.assertEqual(plan["outer_split_seeds"], [101, 202])
        self.assertEqual(plan["inner_split_seeds"], {"101": 1101, "202": 1202})
        self.assertEqual(plan["model_seeds"], [41, 42])
        self.assertEqual(plan["required_environment"]["python"], "3.13")
        self.assertEqual(plan["required_environment"]["numpy"], "2.2.6")
        self.assertIn("base_preprocessing_state_contract", plan["frozen_base_contract"])
        self.assertNotIn("honest_contract", plan["frozen_base_contract"])
        self.assertNotIn("numeric_contract", plan["frozen_base_contract"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, checksum in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), checksum)
        self.assertEqual(set(p.name for p in self.output.iterdir()),
                         {"prerequisite-plan.json", "bounded-influence-evidence.json", "manifest.json"})
        for path in self.output.iterdir():
            self.assertNotIn("synthetic-group", path.read_text())
            self.assertNotIn("synthetic-0", path.read_text())
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            self.run_experiment()
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_one_favorable_seed_or_cell_cannot_qualify_and_metric_failure_never_stops_cells(self):
        for favorable in ({1, 3}, {3}, set()):
            with self.subTest(favorable=favorable), tempfile.TemporaryDirectory() as temp:
                self.output = Path(temp) / "private" / "synthetic"
                class ScheduledBias(RecordingRuntime):
                    def refit(inner, *args):
                        index = len(inner.fits) // 12
                        fitted = super().refit(*args)
                        offset = 1.0 if index in favorable else 0.0
                        return LockedFit(lambda rows: [r[1] / 1000 + offset for r in rows],
                                         fitted.preprocessing_state, fitted.artifacts, fitted.metadata)
                runtime = ScheduledBias()
                result = self.run_experiment(runtime)
                self.assertEqual(result.status, "training_evidence_rejected")
                self.assertEqual(result.resource_use["fits_started"], 52)
                self.assertEqual(len(runtime.fits), 48)
                self.assertEqual(len(result.evidence["cells"]), 4)
                self.assertEqual(len(result.evidence["metric_failed_cells"]), 4 - len(favorable))
                self.assertEqual(result.evidence["uncompleted_cells"], [])
                self.assertEqual([c["qualification"]["qualified"] for c in result.evidence["cells"].values()],
                                 [i in favorable for i in range(4)])

    def test_each_metric_is_required_and_new_not_old_loss_qualifies(self):
        # Old loss improves thanks to the extreme row, yet new loss and serious count worsen.
        targets, anchor, corrected = [0.] * 11, [1000.] + [4.9] * 10, [999.9] + [5.1] * 10
        paired = transition.summarize_correction_transitions(targets, anchor, corrected)
        new = experiment._new_loss_accounting(targets, anchor, corrected)
        self.assertLess(paired["total"]["delta"]["prediction_loss_contribution"], 0)
        self.assertGreater(new["total"]["delta"]["prediction_loss_contribution"], 0)
        self.assertFalse(experiment._qualification(paired, new)["qualified"])
        paired = transition.summarize_correction_transitions([0], [6], [4])
        new = experiment._new_loss_accounting([0], [6], [4])
        self.assertTrue(experiment._qualification(paired, new)["qualified"])
        for key in ("mae_contribution_g", "above_5g_count"):
            invalid = copy.deepcopy(paired)
            invalid["total"]["delta"][key] = 1
            self.assertFalse(experiment._qualification(invalid, new)["qualified"])
        new["total"]["corrected"] = copy.deepcopy(new["total"]["anchor"])
        self.assertFalse(experiment._qualification(paired, new)["qualified"])

    def test_new_loss_decomposition_conserves_every_transition_and_bin(self):
        result = experiment._new_loss_accounting(
            [0] * 8, [4, -4, 5, -5, 7, -7, 30, -30], [5, -5, 6, -6, 5, -5, 28, -28])
        for key in ("transitions", "anchor_error_bins"):
            self.assertEqual(sum(group["count"] for group in result[key].values()), 8)
            for model in ("anchor", "corrected", "delta"):
                for component, total in result["total"][model].items():
                    self.assertAlmostEqual(sum(group[model][component] for group in result[key].values()), total)
        self.assertEqual([group["count"] for group in result["anchor_error_bins"].values()], [2] * 4)
        for args in (([], [], []), ([0], [0, 1], [0]), ([math.nan], [0], [0]), ([0], [1e308], [1e308])):
            with self.assertRaises((ValueError, OverflowError)):
                experiment._new_loss_accounting(*args)

    def test_honest_inner_and_outer_exclusion_fixed_counts_oof_only_preprocessing(self):
        training = normalize(self.records, self.config, contract="legacy")
        testcase = self
        class IsolationRuntime(RecordingRuntime):
            def refit(inner, candidate, seed, features, targets, fixed_training_counts):
                key = "neural_network_epochs" if candidate.family == "neural_network" else "xgboost_trees"
                testcase.assertEqual(fixed_training_counts, {key: experiment.tail.FIXED_COUNTS[key]})
                testcase.assertTrue(all(len(row) == 7 for row in features))
                fitted = super().refit(candidate, seed, features, targets, fixed_training_counts)
                seen = {r[1] for r in features}
                def predict(rows):
                    # Inner held folds cannot appear in their own fitting partition.
                    if len(features) == 32:
                        testcase.assertFalse(seen.intersection(r[1] for r in rows))
                    return fitted.predictor(rows)
                return LockedFit(predict, fitted.preprocessing_state, fitted.artifacts, fitted.metadata)
        runtime = IsolationRuntime()
        with patch.object(numeric, "fit_state", wraps=numeric.fit_state) as fit:
            result = self.run_experiment(runtime)
        self.assertEqual(result.status, "training_prerequisite_supported")
        for split_index, split in enumerate((101, 202)):
            assignments = experiment.t._cross_fit_assignments(training, split, 5)
            held_values = {float(r["volume"]) for r, fold in zip(self.records, assignments) if fold == 0}
            fitted_targets = [float(r["weight"]) for r, fold in zip(self.records, assignments) if fold != 0]
            for _, _, features in runtime.fits[split_index * 24:(split_index + 1) * 24]:
                self.assertFalse(held_values.intersection(r[1] for r in features))
            for model_index in (0, 1):
                columns, targets = fit.call_args_list[split_index * 2 + model_index].args
                self.assertEqual(list(targets), fitted_targets)
                self.assertEqual(len(columns), 2)
                self.assertTrue(all(len(c) == 40 for c in columns))
                state = numeric.fit_state(columns, targets)
                _, design = numeric.features(columns)
                self.assertEqual(state["feature_means"], design.mean(axis=0).tolist())
        # Changing only evaluation metadata cannot change fits, folds or qualification.
        self.output = self.output.parent / "renamed-sources"
        for r in self.records:
            r["anonymous_source_group"] = "different-synthetic-group"
        rerun = self.run_experiment()
        self.assertEqual(result.evidence["cells"], rerun.evidence["cells"])

    def test_runtime_failure_preserves_completed_cell_and_started_fit_without_retry(self):
        class FailLater(RecordingRuntime):
            def refit(inner, *args):
                if len(inner.fits) == 12:
                    raise ArithmeticError("sensitive backend error")
                return super().refit(*args)
        result = self.run_experiment(FailLater())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("bounded_influence_runtime_failed",))
        self.assertEqual(list(result.evidence["cells"]), ["split-101-model-41"])
        self.assertEqual(result.evidence["runtime_failed_cell"], "split-101-model-42")
        self.assertEqual(result.resource_use["fits_started"], 14)
        self.assertEqual(result.resource_use["fits_in_completed_cells"], 13)
        self.assertEqual(len(result.evidence["uncompleted_cells"]), 3)
        self.assertNotIn("sensitive", json.dumps(result.evidence))
        self.assertTrue((self.output / "manifest.json").exists())

    def test_deadlines_budget_invalid_clock_and_invalid_data_fail_closed(self):
        runtime = RecordingRuntime()
        result = self.run_experiment(runtime, clock=lambda: 7200. if runtime.fits else 0.)
        self.assertEqual(result.blockers, ("bounded_influence_deadline_reached",))
        self.assertEqual(result.resource_use["fits_started"], 1)
        def exceed(runtime, fitted, held, seed, before_fit, check_deadline, **kwargs):
            for _ in range(53):
                before_fit()
            self.fail("budget failed to stop")
        self.output = self.output.parent / "exceeded"
        with patch.object(experiment, "_fit_stage", side_effect=exceed):
            result = self.run_experiment()
        self.assertEqual(result.blockers, ("bounded_influence_fit_limit_reached",))
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.output = self.output.parent / "invalid-clock"
        self.assertEqual(self.run_experiment(clock=lambda: math.nan).blockers, ("bounded_influence_invalid_clock",))
        self.output = self.output.parent / "invalid-data"
        self.records[0]["_id"] = ""
        result = self.run_experiment()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 0)
        self.assertTrue((self.output / "manifest.json").exists())

    def test_final_summary_deadline_cannot_support_prerequisite(self):
        now, calls = 0., 0
        real = experiment._new_loss_accounting
        def expire(*args):
            nonlocal now, calls
            calls += 1
            value = real(*args)
            if calls == 4:
                now = 7200.
            return value
        with patch.object(experiment, "_new_loss_accounting", side_effect=expire):
            result = self.run_experiment(clock=lambda: now)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(len(result.evidence["cells"]), 3)

    def test_cli_has_no_validation_test_seed_budget_or_continuation_path(self):
        params = inspect.signature(experiment.run_bounded_influence_training).parameters
        self.assertNotIn("validation_records", params)
        self.assertNotIn("test_records", params)
        parser = experiment.build_parser()
        for flag in ("--validation-records", "--test-records", "--seed", "--maximum-model-fits", "--promote", "--lock"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--training-records", "unused", "--output-root", "unused", flag, "unused"])
        with patch.object(stability, "_verify_predeclared_training_artifact") as verify:
            with self.assertRaisesRegex(SystemExit, "bounded_influence_output_root_mismatch"):
                experiment.main(["--training-records", "unused", "--output-root", str(self.output)])
            verify.assert_not_called()
        with patch.object(experiment, "PREDECLARED_OUTPUT_ROOT", self.output), \
             patch.object(stability, "_verify_predeclared_training_artifact") as verify, \
             patch.object(experiment.t, "TensorflowXGBoostCandidateRuntime", return_value=RecordingRuntime()), \
             patch.object(experiment.t, "_verify_predeclared_environment", side_effect=ValueError("blocked")) as env, \
             patch.object(experiment, "run_bounded_influence_training") as run:
            with self.assertRaisesRegex(SystemExit, "bounded_influence_training_failed"):
                experiment.main(["--training-records", "unused", "--output-root", str(self.output)])
            verify.assert_called_once_with(Path("unused"))
            env.assert_called_once_with(RecordingRuntime.dependency_versions)
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_pinned_training_checksum_reads_only_synthetic_training_and_manifest(self):
        expected = stability.PROJECT_ROOT / "data" / "train.jsonl"
        payload = b"synthetic only"
        checksum = sha256(payload).hexdigest()
        with patch.object(experiment.t, "GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS", {"train.jsonl": checksum}), \
             patch.object(Path, "read_text", autospec=True, return_value=json.dumps({"artifacts": {"train.jsonl": checksum}})) as manifest, \
             patch.object(Path, "read_bytes", autospec=True, return_value=payload) as artifact:
            stability._verify_predeclared_training_artifact(expected)
            manifest.assert_called_once_with(expected.parent / "manifest.json")
            artifact.assert_called_once_with(expected)
            artifact.return_value = b"changed synthetic training"
            with self.assertRaisesRegex(Exception, "artifact_checksum_mismatch"):
                stability._verify_predeclared_training_artifact(expected)
        with patch.object(experiment, "PREDECLARED_OUTPUT_ROOT", self.output), \
             patch.object(stability, "_verify_predeclared_training_artifact", side_effect=ValueError("checksum mismatch")), \
             patch.object(experiment.t, "TensorflowXGBoostCandidateRuntime") as runtime:
            with self.assertRaises(SystemExit):
                experiment.main(["--training-records", "unused", "--output-root", str(self.output)])
            runtime.assert_not_called()

    def test_required_environment_is_exact_not_merely_recorded(self):
        deps = dict(experiment.t.CROSS_FITTED_GATE_DEPENDENCY_VERSIONS)
        with patch.object(experiment.t.platform, "python_version_tuple", return_value=("3", "13", "0")), \
             patch.object(experiment.t, "package_version", return_value="1.7.2"):
            experiment.t._verify_predeclared_environment(deps)
            for key in deps:
                changed = {**deps, key: "unapproved"}
                with self.assertRaisesRegex(Exception, "dependency_mismatch"):
                    experiment.t._verify_predeclared_environment(changed)
        with patch.object(experiment.t.platform, "python_version_tuple", return_value=("3", "14", "0")), \
             patch.object(experiment.t, "package_version", return_value="1.7.2"):
            with self.assertRaisesRegex(Exception, "dependency_mismatch"):
                experiment.t._verify_predeclared_environment(deps)
        with patch.object(experiment.t.platform, "python_version_tuple", return_value=("3", "13", "0")), \
             patch.object(experiment.t, "package_version", return_value="unapproved"):
            with self.assertRaisesRegex(Exception, "dependency_mismatch"):
                experiment.t._verify_predeclared_environment(deps)

    def test_invalid_backend_predictions_preprocessing_or_new_state_preserve_blocked_evidence(self):
        class InvalidRuntime(RecordingRuntime):
            def refit(inner, *args):
                fitted = super().refit(*args)
                return LockedFit(lambda rows: [math.nan] * len(rows), fitted.preprocessing_state,
                                 fitted.artifacts, fitted.metadata)
        result = self.run_experiment(InvalidRuntime())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 1)
        self.output = self.output.parent / "bad-preprocessing"
        class InvalidPreprocessing(RecordingRuntime):
            def refit(inner, *args):
                fitted = super().refit(*args)
                return LockedFit(fitted.predictor, {}, fitted.artifacts, fitted.metadata)
        self.assertEqual(self.run_experiment(InvalidPreprocessing()).status, "blocked")
        self.output = self.output.parent / "bad-state"
        with patch.object(numeric, "fit_state", return_value={}):
            result = self.run_experiment()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 13)
        self.assertFalse(result.evidence["validation_eligible"])
        self.assertTrue((self.output / "manifest.json").exists())

    def test_old_contracts_not_mutated_or_registered_under_new_version(self):
        previous = copy.deepcopy(old.CONTRACT)
        before = transition._plan("synthetic", "synthetic", {})
        plan = experiment._plan("synthetic", "synthetic", {})
        plan["correction_contract"]["iterations"] = 0
        plan["old_diagnostic_accounting_contract"]["serious"] = "changed"
        self.assertEqual(previous, old.CONTRACT)
        self.assertEqual(before, transition._plan("synthetic", "synthetic", {}))
        self.assertEqual(numeric.CONTRACT["iterations"], 2000)
        self.assertNotEqual(numeric.VERSION, old.VERSION)


if __name__ == "__main__":
    unittest.main()
