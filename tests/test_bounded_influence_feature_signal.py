"""Synthetic-only actual numeric summaries and frozen four-cell descriptive lifecycle."""
import copy
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from minires import EvaluationConfig
from minires.ingestion import normalize
from minires.modeling import bounded_influence_feature_signal as diagnostic
from minires.modeling import bounded_influence_correction as numeric
from minires.modeling import bounded_influence_training as prerequisite
from minires.modeling import correction_transition as transition
from minires.modeling import tail_correction as old
from minires.modeling import training_stability as stability
from minires.modeling.tuning import LockedFit
from test_training_stability import RecordingRuntime, row


def state(coefficients=(1., 0., 0., 0.)):
    return {"contract": copy.deepcopy(numeric.CONTRACT), "feature_means": [0.] * 3,
            "feature_scales": [1.] * 3, "coefficients": list(coefficients)}


class FeatureSignalSummaryTests(unittest.TestCase):
    def test_exact_signed_thirds_and_clip_limits_in_actual_preprocessing(self):
        s = state()
        s["feature_scales"] = [3.] * 3
        values = [-6., -3., -1., 0., 1., 3., 6.]
        with patch.object(np, "searchsorted", wraps=np.searchsorted) as assign:
            bins = diagnostic._feature_bin_indices(s, [values, values])
        self.assertEqual(assign.call_args.args[1][:, 0].tolist(), [-1., -1., -1 / 3, 0., 1 / 3, 1., 1.])
        self.assertEqual(bins[:, 0].tolist(), [0, 0, 0, 1, 1, 2, 2])
        bins = diagnostic._feature_bin_indices(s, [values, [0.] * 7])
        self.assertEqual(bins[:, 1].tolist(), [0, 0, 0, 1, 1, 2, 2])
        s["feature_means"][2] = 3.
        bins = diagnostic._feature_bin_indices(s, [[0., 1., 2., 3., 4., 5., 6.], [0.] * 7])
        self.assertEqual(bins[:, 2].tolist(), [0, 0, 0, 1, 1, 2, 2])
        # Adjacent float64 values distinguish exact equality from just-above edges.
        edges = [-1 / 3, 1 / 3]
        values = [v for e in edges for v in (math.nextafter(e, -math.inf), e, math.nextafter(e, math.inf))]
        bins = diagnostic._feature_bin_indices(state(), [values, [0.] * 6])
        self.assertEqual(bins[:, 1].tolist(), [0, 0, 1, 1, 1, 2])
        # Test bins through the complete summary, not merely the index helper.
        s = state((0., 0., 0., 0.))
        s["feature_scales"] = [3.] * 3
        result = diagnostic.summarize_feature_signal(s, [[-6., -3., -1., 0., 1., 3., 6.]] * 2, [0.] * 7)
        self.assertEqual([g["count"] for g in result["cohorts"]["all"]["marginal"]["anchor_g"].values()], [3, 2, 2])

    def test_all_bins_nested_cohorts_conserve_numeric_pipeline_and_baseline(self):
        columns = [[-20., -7., -5., -4., 0., 4., 5., 7., 20.],
                   [-18., -8., -4., -5., 0., 3., 6., 8., 18.]]
        anchor = numeric.features(columns)[0]
        errors = np.asarray([-8., -7., -5., -4., 0., 4., 5., 7., 8.])
        targets = anchor - errors
        result = diagnostic.summarize_feature_signal(state(), columns, targets)
        self.assertEqual(result["count"], 9)
        self.assertEqual(result["contribution_denominator"], 9)
        cohorts = result["cohorts"]
        self.assertEqual([c["total"]["count"] for c in cohorts.values()], [9, 4, 2])
        # Cohorts overlap: summing their counts would double-count severe rows.
        self.assertGreater(sum(c["total"]["count"] for c in cohorts.values()), 9)
        for cohort in cohorts.values():
            self.assertEqual(len(cohort["joint"]), 27)
            self.assertEqual(tuple(cohort["marginal"]), numeric.FEATURES)
            for partition in (*cohort["marginal"].values(), cohort["joint"]):
                self.assertIn(len(partition), (3, 27))
                diagnostic._check_partition(cohort["total"], list(partition.values()))
                for group in partition.values():
                    self.assertEqual(sum(group["transition_counts"].values()), group["count"])
                    self.assertEqual(sum(group["anchor"]["residual_sign_counts"].values()), group["count"])
                    self.assertEqual(sum(group["corrected"]["residual_sign_counts"].values()), group["count"])
                    for keys in (("toward_count", "away_count", "neutral_count"),
                                 ("absolute_error_improved_count", "absolute_error_worsened_count", "absolute_error_unchanged_count")):
                        self.assertEqual(sum(group["correction"][k] for k in keys), group["count"])
                    if not group["count"]:
                        self.assertEqual(group["support_status"], "empty")
                        self.assertFalse(group["support_sufficient"])
                        self.assertEqual(group["anchor"]["prediction_loss_contribution"], 0.)
        # Direct independent Huber arithmetic and whole-cell denominator, not subgroup mean.
        def huber(z, d):
            return z * z if abs(z) <= d else 2 * d * abs(z) - d * d
        for index, model in enumerate(("anchor", "corrected")):
            e = errors + index
            severe = cohorts["anchor_gt7"]["total"][model]
            ordinary = sum(.1 * huber(v, 5) for v in e[[0, 8]]) / 9
            excess = sum(4 * huber(max(abs(v) - 4.5, 0), 2.5) for v in e[[0, 8]]) / 9
            self.assertAlmostEqual(severe["ordinary_loss_contribution"], ordinary)
            self.assertAlmostEqual(severe["excess_loss_contribution"], excess)
        baseline = prerequisite._new_loss_accounting(targets, anchor, anchor + 1)
        self.assertEqual(result["new_descriptive_loss"], baseline)
        self.assertEqual(cohorts["all"]["total"]["anchor"]["above_5g_count"], 4)
        self.assertEqual(cohorts["all"]["total"]["corrected"]["above_5g_count"], 5)
        self.assertEqual(cohorts["all"]["total"]["transition_counts"],
                         {"stable_nonserious": 4, "harm": 1, "repair": 0, "persistent_serious": 4})
        json.dumps(result, allow_nan=False)

    def test_signed_outcomes_toward_can_overshoot_and_old_loss_is_separate(self):
        result = diagnostic.summarize_feature_signal(state(), [[10.] * 5] * 2, [10.25, 10.5, 12., 10., 9.])
        total = result["cohorts"]["all"]["total"]
        self.assertEqual(total["anchor"]["residual_sign_counts"], {"negative": 3, "zero": 1, "positive": 1})
        self.assertEqual(total["corrected"]["residual_sign_counts"], {"negative": 1, "zero": 0, "positive": 4})
        correction = total["correction"]
        self.assertEqual([correction[k] for k in ("toward_count", "away_count", "neutral_count")], [3, 1, 1])
        self.assertEqual([correction[k] for k in ("absolute_error_improved_count", "absolute_error_worsened_count", "absolute_error_unchanged_count")], [1, 3, 1])
        self.assertEqual(correction["signed_sum_g"], 5.)
        self.assertEqual(correction["absolute_sum_g"], 5.)
        self.assertEqual(correction["maximum_absolute_g"], 1.)
        severe = diagnostic.summarize_feature_signal(state(), [[1000.]] * 2, [0.])
        self.assertGreater(severe["paired_transitions_with_old_diagnostic_loss"]["total"]["anchor"]["prediction_loss_contribution"],
                           severe["new_descriptive_loss"]["total"]["anchor"]["prediction_loss_contribution"])

    def test_strict_signed_serious_and_severe_boundaries_and_repairs(self):
        beyond7 = math.nextafter(7., math.inf)
        for sign in (-1., 1.):
            anchors = [sign * value for value in (5., 7., beyond7)]
            s = state((-2. * sign, 0., 0., 0.))
            result = diagnostic.summarize_feature_signal(s, [anchors, anchors], [0.] * 3)
            cohorts = result["cohorts"]
            self.assertEqual(cohorts["anchor_gt5"]["total"]["count"], 2)
            self.assertEqual(cohorts["anchor_gt7"]["total"]["count"], 1)
            total = cohorts["all"]["total"]
            self.assertEqual(total["anchor"]["above_5g_count"], 2)
            self.assertEqual(total["corrected"]["above_5g_count"], 1)
            self.assertEqual(total["transition_counts"],
                             {"stable_nonserious": 1, "harm": 0, "repair": 1, "persistent_serious": 1})
            self.assertEqual(total["delta"]["above_5g_count"], -1)
            self.assertEqual(total["correction"]["absolute_error_improved_count"], 3)

    def test_targets_cannot_change_bins_support_or_preprocessing(self):
        s = state()
        before = copy.deepcopy(s)
        for n in (19, 20):
            first = diagnostic.summarize_feature_signal(s, [[10.] * n] * 2, [10.] * n)
            second = diagnostic.summarize_feature_signal(s, [[10.] * n] * 2, [20.] * n)
            for key in ("marginal", "joint"):
                a, b = first["cohorts"]["all"][key], second["cohorts"]["all"][key]
                partitions = [(a, b)] if key == "joint" else [(a[f], b[f]) for f in numeric.FEATURES]
                for left, right in partitions:
                    for bin_name in left:
                        self.assertEqual(left[bin_name]["count"], right[bin_name]["count"])
                        self.assertEqual(left[bin_name]["support_status"], right[bin_name]["support_status"])
                        if left[bin_name]["count"]:
                            self.assertEqual(left[bin_name]["support_sufficient"], n >= 20)
            self.assertEqual(first["cohorts"]["anchor_gt7"]["total"]["count"], 0)
            self.assertEqual(second["cohorts"]["anchor_gt7"]["total"]["count"], n)
        self.assertEqual(s, before)

    def test_wrong_shapes_nonfinite_range_overflow_malformed_state_and_bound_fail_closed(self):
        for columns, targets in (([[], []], []), ([[0], [0, 1]], [0]), ([[0], [0]], [0, 1]),
                                 ([[0], [0]], [[0]]), ([[math.nan], [0]], [0]),
                                 ([[0], [0]], [math.inf]), ([[1e308], [1e308]], [0]),
                                 ([[numeric.FLOAT32_MAXIMUM], [-numeric.FLOAT32_MAXIMUM]], [0])):
            with self.subTest(columns=columns), self.assertRaises((ValueError, OverflowError)):
                diagnostic.summarize_feature_signal(state(), columns, targets)
        for mutate in (lambda s: s.update(extra=1), lambda s: s.update(feature_means=[0]),
                       lambda s: s.update(coefficients=[3., 0., 0., 0.]),
                       lambda s: s["contract"].update(iterations=1),
                       lambda s: s.update(feature_scales=[0., 1., 1.]),
                       lambda s: s.update(feature_scales=[5e-324, 1., 1.]),
                       lambda s: s.update(feature_means=[math.nan, 0., 0.])):
            invalid = state()
            mutate(invalid)
            with self.assertRaises((ValueError, OverflowError)):
                diagnostic.summarize_feature_signal(invalid, [[10.], [10.]], [0.])
        # Even a corrupted prediction seam cannot evade the paired departure check.
        with patch.object(numeric, "predict", side_effect=[((0.,), (0.,)), ((3.,), (3.,))]):
            with self.assertRaisesRegex(ValueError, "correction_bound_exceeded"):
                diagnostic.summarize_feature_signal(state(), [[0.], [0.]], [0.])

    def test_conservation_guard_rejects_count_and_contribution_corruption(self):
        result = diagnostic.summarize_feature_signal(state(), [[10.]] * 2, [0.])
        total = result["cohorts"]["all"]["total"]
        groups = list(result["cohorts"]["all"]["joint"].values())
        for section, key in ((None, "count"), ("correction", "absolute_error_improved_count"),
                             ("anchor", "mae_contribution_g"), ("delta", "prediction_loss_contribution")):
            bad = copy.deepcopy(groups)
            target = bad[0] if section is None else bad[0][section]
            target[key] += 1
            with self.assertRaisesRegex(ValueError, "nonconserving"):
                diagnostic._check_partition(total, bad)


class FeatureSignalLifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name) / "private" / "synthetic-run-018"
        self.records = [row(f"synthetic-{i}", i + 1) for i in range(50)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def run_diagnostic(self, runtime=None, clock=lambda: 0.):
        return diagnostic.run_feature_signal_diagnostic(
            self.records, self.config, runtime=runtime or RecordingRuntime(), output_root=self.output, clock=clock)

    def test_real_new_corrections_all_four_52_fits_no_qualification_create_only(self):
        runtime = RecordingRuntime()
        with patch.object(numeric, "fit_state", wraps=numeric.fit_state) as fit, \
             patch.object(old, "fit_state", side_effect=AssertionError("no old fit")), \
             patch.object(prerequisite, "_qualification", side_effect=AssertionError("no qualification")):
            result = self.run_diagnostic(runtime)
        self.assertEqual(result.status, "completed")
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(len(runtime.fits), 48)
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(result.resource_use["unused_fit_capacity"], 0)
        self.assertEqual(list(result.evidence["cells"]), [f"split-{s}-model-{m}" for s in (101, 202) for m in (41, 42)])
        self.assertEqual(result.evidence["uncompleted_cells"], [])
        for key in ("qualification_performed", "selection_performed", "locking_performed", "validation_eligible", "production_continuation"):
            self.assertFalse(result.evidence[key])
        for cell in result.evidence["cells"].values():
            self.assertEqual((cell["fits"], cell["held_out_record_count"], cell["fitting_record_count"]), (13, 10, 40))
            self.assertNotIn("qualification", cell)
            transition._validate_shift(cell["oof_full_fit_shift"], 40, 10)
        plan = json.loads((self.output / "diagnostic-plan.json").read_text())
        self.assertEqual(plan["accounting_contract"], diagnostic.ACCOUNTING_CONTRACT)
        self.assertEqual(plan["correction_contract"], numeric.CONTRACT)
        self.assertNotIn("qualification_contract", plan)
        self.assertEqual(plan["fit_allocation"], {"inner_oof_bases": 40, "outer_partition_bases": 8, "corrections": 4})
        self.assertEqual(plan["fixed_training_counts"], {"neural_network_epochs": 87, "xgboost_trees": 1091})
        self.assertEqual(plan["inner_split_seeds"], {"101": 1101, "202": 1202})
        self.assertEqual(plan["required_environment"]["python"], "3.13")
        self.assertEqual(plan["required_environment"]["numpy"], "2.2.6")
        self.assertEqual(plan["maximum_elapsed_seconds"], 7200)
        self.assertEqual(plan["expected_training_sha256"], diagnostic.t.GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS["train.jsonl"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, checksum in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), checksum)
        self.assertEqual(set(p.name for p in self.output.iterdir()), {"diagnostic-plan.json", "feature-signal-evidence.json", "manifest.json"})
        for path in self.output.iterdir():
            for private_value in ("synthetic-group", "synthetic-0", str(self.output)):
                self.assertNotIn(private_value, path.read_text())
        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"row_ids", "predictions", "targets", "feature_values", "source_groups", "record_identity", "feature_means", "feature_scales", "coefficients"})
                for v in value.values():
                    check_keys(v)
            elif isinstance(value, list):
                for v in value:
                    check_keys(v)
        check_keys(result.evidence)
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            self.run_diagnostic()
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_metric_worsening_still_completes_every_cell(self):
        class OppositeBias(RecordingRuntime):
            def refit(inner, *args):
                fitted = super().refit(*args)
                # OOF biases positive; outer partition fits negative: correction hurts.
                offset = 1. if len(args[2]) == 32 else -1.
                return LockedFit(lambda rows: [r[1] / 1000 + offset for r in rows],
                                 fitted.preprocessing_state, fitted.artifacts, fitted.metadata)
        result = self.run_diagnostic(OppositeBias())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.resource_use["fits_started"], 52)
        for cell in result.evidence["cells"].values():
            self.assertGreater(cell["feature_signal"]["cohorts"]["all"]["total"]["delta"]["mae_contribution_g"], 0)

    def test_inner_outer_exclusion_fixed_counts_train_oof_preprocessing_and_state_used(self):
        canonical = normalize(self.records, self.config, contract="legacy")
        testcase = self
        class IsolationRuntime(RecordingRuntime):
            def refit(inner, candidate, seed, features, targets, fixed_training_counts):
                key = "neural_network_epochs" if candidate.family == "neural_network" else "xgboost_trees"
                testcase.assertEqual(fixed_training_counts, {key: diagnostic.tail.FIXED_COUNTS[key]})
                testcase.assertTrue(all(len(r) == 7 for r in features))
                fitted = super().refit(candidate, seed, features, targets, fixed_training_counts)
                seen = {r[1] for r in features}
                def predict(rows):
                    if len(features) == 32:
                        testcase.assertFalse(seen.intersection(r[1] for r in rows))
                    return fitted.predictor(rows)
                return LockedFit(predict, fitted.preprocessing_state, fitted.artifacts, fitted.metadata)
        runtime = IsolationRuntime()
        with patch.object(numeric, "fit_state", wraps=numeric.fit_state) as fit, \
             patch.object(diagnostic, "summarize_feature_signal", wraps=diagnostic.summarize_feature_signal) as summary:
            result = self.run_diagnostic(runtime)
        self.assertEqual(result.status, "completed")
        for split_index, split in enumerate((101, 202)):
            assignments = diagnostic.t._cross_fit_assignments(canonical, split, 5)
            held_values = {float(r["volume"]) for r, fold in zip(self.records, assignments) if fold == 0}
            fitted_targets = [float(r["weight"]) for r, fold in zip(self.records, assignments) if fold != 0]
            for _, _, features in runtime.fits[split_index * 24:(split_index + 1) * 24]:
                self.assertFalse(held_values.intersection(r[1] for r in features))
            for model_index in (0, 1):
                index = split_index * 2 + model_index
                columns, targets = fit.call_args_list[index].args
                self.assertEqual(list(targets), fitted_targets)
                self.assertTrue(all(len(c) == 40 for c in columns))
                supplied_state, held_columns, held_targets = summary.call_args_list[index].args
                _, matrix = numeric.features(columns)
                self.assertEqual(supplied_state["feature_means"], matrix.mean(axis=0).tolist())
                scales = np.where(matrix.std(axis=0) <= 1e-12, 1., matrix.std(axis=0))
                self.assertEqual(supplied_state["feature_scales"], scales.tolist())
                self.assertEqual(len(held_targets), 10)
                self.assertTrue(all(len(c) == 10 for c in held_columns))
        self.output = self.output.parent / "renamed"
        for record in self.records:
            record["anonymous_source_group"] = "different-synthetic-group"
        self.assertEqual(result.evidence["cells"], self.run_diagnostic().evidence["cells"])

    def test_later_failure_retains_completed_cell_partial_fits_no_retry(self):
        class FailLater(RecordingRuntime):
            def refit(inner, *args):
                if len(inner.fits) == 12:
                    raise ArithmeticError("sensitive backend details")
                return super().refit(*args)
        result = self.run_diagnostic(FailLater())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("feature_signal_runtime_failed",))
        self.assertEqual(list(result.evidence["cells"]), ["split-101-model-41"])
        self.assertEqual(result.evidence["failed_cell"], "split-101-model-42")
        self.assertEqual(result.resource_use["fits_started"], 14)
        self.assertEqual(result.resource_use["fits_in_completed_cells"], 13)
        self.assertEqual(len(result.evidence["uncompleted_cells"]), 3)
        self.assertNotIn("sensitive", json.dumps(result.evidence))
        self.assertTrue((self.output / "manifest.json").exists())

    def test_deadlines_fit_budget_invalid_clock_and_bad_state_block(self):
        runtime = RecordingRuntime()
        result = self.run_diagnostic(runtime, clock=lambda: 7200. if runtime.fits else 0.)
        self.assertEqual(result.blockers, ("feature_signal_deadline_reached",))
        self.assertEqual(result.resource_use["fits_started"], 1)
        self.output = self.output.parent / "exceeded"
        def exceed(runtime, fitted, held, seed, before_fit, check_deadline, **kwargs):
            for _ in range(53):
                before_fit()
            self.fail("budget failed to stop")
        with patch.object(prerequisite, "_fit_stage", side_effect=exceed):
            result = self.run_diagnostic()
        self.assertEqual(result.blockers, ("feature_signal_fit_limit_reached",))
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.output = self.output.parent / "invalid-clock"
        self.assertEqual(self.run_diagnostic(clock=lambda: math.nan).blockers, ("feature_signal_invalid_clock",))
        self.output = self.output.parent / "invalid-state"
        with patch.object(numeric, "fit_state", return_value={}):
            result = self.run_diagnostic()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 13)
        self.output = self.output.parent / "invalid-data"
        self.records[0]["_id"] = ""
        self.assertEqual(self.run_diagnostic().resource_use["fits_started"], 0)

    def test_final_summary_deadline_and_write_failure_cannot_claim_completion(self):
        now, calls = 0., 0
        real = diagnostic.summarize_feature_signal
        def expire(*args):
            nonlocal now, calls
            calls += 1
            value = real(*args)
            if calls == 4:
                now = 7200.
            return value
        with patch.object(diagnostic, "summarize_feature_signal", side_effect=expire):
            result = self.run_diagnostic(clock=lambda: now)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(len(result.evidence["cells"]), 3)
        self.output = self.output.parent / "write-failed"
        real_write = diagnostic.write_private_json
        def fail_manifest(path, value):
            if path.name == "manifest.json":
                raise OSError("synthetic disk failure")
            return real_write(path, value)
        with patch.object(diagnostic, "write_private_json", side_effect=fail_manifest), self.assertRaises(OSError):
            self.run_diagnostic()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_cli_has_training_only_fixed_knobs_and_enforces_checksum_environment(self):
        self.assertEqual(set(inspect.signature(diagnostic.run_feature_signal_diagnostic).parameters),
                         {"training_records", "config", "runtime", "output_root", "clock"})
        parser = diagnostic.build_parser()
        self.assertEqual({a.dest for a in parser._actions}, {"help", "training_records", "output_root", "volume_unit", "scope_confirmed"})
        for flag in ("--validation-records", "--test-records", "--seed", "--maximum-model-fits", "--promote", "--lock", "--bins", "--minimum-support", "--features", "--loss"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--training-records", "unused", "--output-root", "unused", flag, "unused"])
        with patch.object(stability, "_verify_predeclared_training_artifact") as verify:
            with self.assertRaisesRegex(SystemExit, "feature_signal_output_root_mismatch"):
                diagnostic.main(["--training-records", "unused", "--output-root", str(self.output)])
            verify.assert_not_called()
        for failing in ("checksum", "environment"):
            with patch.object(diagnostic, "PREDECLARED_OUTPUT_ROOT", self.output), \
                 patch.object(stability, "_verify_predeclared_training_artifact", side_effect=ValueError("blocked") if failing == "checksum" else None) as verify, \
                 patch.object(diagnostic.t, "TensorflowXGBoostCandidateRuntime", return_value=RecordingRuntime()) as runtime, \
                 patch.object(diagnostic.t, "_verify_predeclared_environment", side_effect=ValueError("blocked")) as env, \
                 patch.object(diagnostic, "run_feature_signal_diagnostic") as run:
                with self.assertRaisesRegex(SystemExit, "feature_signal_diagnostic_failed"):
                    diagnostic.main(["--training-records", "unused", "--output-root", str(self.output)])
                verify.assert_called_once_with(Path("unused"))
                if failing == "checksum":
                    runtime.assert_not_called()
                else:
                    env.assert_called_once_with(RecordingRuntime.dependency_versions)
                run.assert_not_called()
        self.assertFalse(self.output.exists())
        for config in (EvaluationConfig(None, "cm3", True, seed=17), EvaluationConfig(None, "mm3", None, seed=17),
                       EvaluationConfig(None, "mm3", True, seed=41)):
            self.config = config
            with self.assertRaisesRegex(Exception, "invalid_feature_signal_configuration"):
                self.run_diagnostic()
        self.assertFalse(self.output.exists())

    def test_old_contracts_and_plans_unchanged(self):
        before = prerequisite._plan("synthetic", "synthetic", {})
        old_contract = copy.deepcopy(old.CONTRACT)
        new = diagnostic._plan("synthetic", "synthetic", {})
        new["correction_contract"]["iterations"] = 0
        new["old_diagnostic_accounting_contract"]["serious"] = "changed"
        self.assertEqual(before, prerequisite._plan("synthetic", "synthetic", {}))
        self.assertEqual(old.CONTRACT, old_contract)
        self.assertEqual(numeric.CONTRACT["iterations"], 2000)
        self.assertEqual(prerequisite.PREDECLARED_OUTPUT_ROOT.name, "run-017")
        self.assertEqual(diagnostic.PREDECLARED_OUTPUT_ROOT.name, "run-018")


if __name__ == "__main__":
    unittest.main()
