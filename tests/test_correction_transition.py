"""Synthetic paired accounting and training-only correction-transition lifecycle."""
from __future__ import annotations

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
from minires.modeling import correction_transition as diagnostic
from minires.modeling import tail_correction as numeric
from minires.modeling import training_stability as stability
from test_training_stability import RecordingRuntime, row


class CorrectionTransitionSummaryTests(unittest.TestCase):
    def test_loss_down_yet_serious_count_up_is_observation_not_failure(self):
        result = diagnostic.summarize_correction_transitions(
            [0] * 5, [20, 4.9, -4.9, 6, 0], [18, 5.1, -5.1, 4, 0],
        )
        self.assertLess(result["total"]["delta"]["prediction_loss_contribution"], 0)
        self.assertEqual(result["total"]["delta"]["above_5g_count"], 1)
        self.assertEqual({k: v["count"] for k, v in result["transitions"].items()},
                         {"stable_nonserious": 1, "harm": 2, "repair": 1, "persistent_serious": 1})
        self.assertTrue(all(result["descriptive_flags"].values()))
        self.assertEqual(result["transitions"]["harm"]["anchor"]["residual_sign_counts"],
                         {"negative": 1, "zero": 0, "positive": 1})
        self.assertEqual(result["total"]["correction"]["away_count"], 2)
        self.assertEqual(result["total"]["correction"]["toward_count"], 2)
        self.assertEqual(result["total"]["correction"]["neutral_count"], 1)

    def test_exact_signed_boundaries_and_theoretical_oracle(self):
        above7 = math.nextafter(7.0, math.inf)
        anchor = [4, -4, 5, -5, 7, -7, above7, -above7]
        corrected = [4, -4, 5, -5, 5, -5, above7 - 2, -above7 + 2]
        result = diagnostic.summarize_correction_transitions([0] * 8, anchor, corrected)
        self.assertEqual([v["count"] for v in result["anchor_error_bins"].values()], [2, 2, 2, 2])
        self.assertEqual(result["total"]["anchor"]["above_5g_count"], 4)
        self.assertEqual(result["total"]["corrected"]["above_5g_count"], 2)
        self.assertEqual(result["transitions"]["repair"]["count"], 2)
        self.assertEqual(result["total"]["oracle"]["unavoidable_above_5g_count"], 2)
        self.assertEqual(result["total"]["oracle"]["repairable_serious_count"], 2)

    def test_every_partition_conserves_counts_mae_loss_signs_and_directions(self):
        result = diagnostic.summarize_correction_transitions(
            [0] * 9, [0, 4, -5, 5.01, -6, 7, -7.01, 30, -30],
            [0, 5.1, -5, 4, -4, 8, -8, 28, -28],
        )
        for partition in (result["transitions"], result["anchor_error_bins"]):
            self.assertEqual(sum(v["count"] for v in partition.values()), 9)
            for model in ("anchor", "corrected", "delta"):
                for key in result["total"]["delta"]:
                    self.assertAlmostEqual(sum(v[model][key] for v in partition.values()),
                                           result["total"][model][key])
            for group in partition.values():
                for model in ("anchor", "corrected"):
                    self.assertEqual(sum(group[model]["residual_sign_counts"].values()), group["count"])
                self.assertEqual(sum(group["correction"][key] for key in
                                     ("toward_count", "away_count", "neutral_count")), group["count"])
                self.assertEqual(sum(group["transition_counts"].values()), group["count"])
        self.assertEqual(result["total"]["delta"]["above_5g_count"],
                         result["transitions"]["harm"]["count"] - result["transitions"]["repair"]["count"])

    def test_empty_groups_have_finite_zero_contributions(self):
        result = diagnostic.summarize_correction_transitions([1], [1], [1])
        self.assertEqual(result["transitions"]["repair"]["count"], 0)
        self.assertFalse(result["descriptive_flags"]["severe_tails_dominate_excess_loss"])
        json.dumps(result, allow_nan=False)

    def test_wrong_shape_nonfinite_overflow_and_unbounded_correction_fail_closed(self):
        cases = [([], [], []), ([0], [0, 1], [0]), ([[0]], [[0]], [[0]]),
                 ([0], [0], [3]), ([1e308], [-1e308], [-1e308]),
                 ([0], [1e200], [1e200])]
        for position in range(3):
            for value in (math.nan, math.inf, -math.inf):
                vectors = [[0.0], [0.0], [0.0]]
                vectors[position] = [value]
                cases.append(tuple(vectors))
        for args in cases:
            with self.subTest(args=args), self.assertRaises((ValueError, OverflowError)):
                diagnostic.summarize_correction_transitions(*args)


class CorrectionTransitionLifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name) / "private" / "run-016"
        self.records = [row(f"synthetic-{i}", i + 1) for i in range(50)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def run_diagnostic(self, **kwargs):
        return diagnostic.run_correction_transition_diagnostic(
            self.records, self.config, runtime=kwargs.pop("runtime", RecordingRuntime()),
            output_root=self.output, clock=kwargs.pop("clock", lambda: 0.0), **kwargs,
        )

    def test_real_summary_real_correction_and_52_fits_create_only_manifest(self):
        runtime = RecordingRuntime()
        with patch.object(diagnostic, "summarize_correction_transitions",
                          wraps=diagnostic.summarize_correction_transitions) as summary:
            result = self.run_diagnostic(runtime=runtime)
        self.assertEqual(result.status, "completed")
        self.assertEqual(summary.call_count, 4)
        self.assertEqual(len(runtime.fits), 48)
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(result.resource_use["unused_fit_capacity"], 0)
        self.assertEqual(result.evidence["uncompleted_cells"], [])
        self.assertEqual(set(result.evidence["cells"]),
                         {f"split-{s}-model-{m}" for s in (101, 202) for m in (41, 42)})
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(manifest["create_only"])
        for name, checksum in manifest["artifacts"].items():
            content = (self.output / name).read_bytes()
            self.assertEqual(sha256(content).hexdigest(), checksum)
            self.assertNotIn(b"synthetic-group", content)
            self.assertNotIn(b"synthetic-0", content)
            self.assertNotIn(str(self.output).encode(), content)
        evidence = json.loads((self.output / "correction-transition-evidence.json").read_text())
        self.assertEqual(evidence["resource_use"], result.resource_use)
        plan = json.loads((self.output / "diagnostic-plan.json").read_text())
        self.assertEqual(plan["accounting_contract"], diagnostic.ACCOUNTING_CONTRACT)
        self.assertEqual(plan["fit_allocation"], {"inner_oof_bases": 40, "outer_partition_bases": 8, "corrections": 4})
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            self.run_diagnostic()
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_deadline_after_fit_preserves_failed_and_uncompleted_cells(self):
        runtime = RecordingRuntime()
        result = self.run_diagnostic(runtime=runtime, clock=lambda: 7200.0 if runtime.fits else 0.0)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("correction_transition_deadline_reached",))
        self.assertEqual(result.resource_use["fits_started"], 1)
        self.assertEqual(len(runtime.fits), 1)
        self.assertEqual(result.evidence["failed_cell"], "split-101-model-41")
        self.assertEqual(len(result.evidence["uncompleted_cells"]), 4)
        self.assertTrue((self.output / "manifest.json").exists())

    def test_incomplete_and_nonfinite_evidence_cannot_complete(self):
        for replacement in ({}, {"invalid": math.nan}):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temp:
                self.output = Path(temp) / "private" / "run"
                with patch.object(diagnostic.tail, "_fit_stage", return_value=(
                    {"contract": copy.deepcopy(numeric.CONTRACT), "feature_means": [0.] * 3,
                     "feature_scales": [1.] * 3, "coefficients": [0.] * 4}, [], [[0.] * 10] * 2, replacement,
                )):
                    result = self.run_diagnostic()
                self.assertEqual(result.status, "blocked")
                self.assertEqual(result.resource_use["fits_started"], 0)

    def test_budget_guard_blocks_extra_fit_without_recycling(self):
        def exceed(runtime, fitted, held, seed, before_fit, check_deadline, **kwargs):
            for _ in range(53):
                before_fit()
            self.fail("fit budget did not stop")
        with patch.object(diagnostic.tail, "_fit_stage", side_effect=exceed):
            result = self.run_diagnostic()
        self.assertEqual(result.blockers, ("correction_transition_fit_limit_reached",))
        self.assertEqual(result.resource_use["fits_started"], 52)

    def test_shift_rejects_missing_nonfinite_and_private_fields(self):
        with patch.object(diagnostic, "_validate_shift", wraps=diagnostic._validate_shift) as validate:
            self.assertEqual(self.run_diagnostic().status, "completed")
        shift, fitting_count, held_count = validate.call_args.args
        for mutate in (
            lambda value: value.pop("anchor_shift"),
            lambda value: value["anchor_shift"].update({"mean_signed_shift_g": math.nan}),
            lambda value: value["anchor_shift"].update({"row_ids": ["private"]}),
            lambda value: value["anchor_distributions"]["evaluation"].update({"count": 1}),
        ):
            invalid = copy.deepcopy(shift)
            mutate(invalid)
            with self.assertRaises(ValueError):
                diagnostic._validate_shift(invalid, fitting_count, held_count)

    def test_failed_later_fit_preserves_completed_cell_without_raw_error(self):
        class FailSecondCell(RecordingRuntime):
            def refit(self, *args):
                if len(self.fits) == 12:
                    raise RuntimeError("sensitive runtime details")
                return super().refit(*args)
        result = self.run_diagnostic(runtime=FailSecondCell())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(list(result.evidence["cells"]), ["split-101-model-41"])
        self.assertEqual(result.resource_use["fits_started"], 14)
        self.assertEqual(result.resource_use["fits_in_completed_cells"], 13)
        self.assertNotIn("sensitive", json.dumps(result.evidence))

    def test_deadline_during_final_summary_blocks_completion(self):
        now = 0.0
        count = 0
        real_summary = diagnostic.summarize_correction_transitions
        def expire(*args):
            nonlocal now, count
            count += 1
            result = real_summary(*args)
            if count == 4:
                now = 7200.0
            return result
        with patch.object(diagnostic, "summarize_correction_transitions", side_effect=expire):
            result = self.run_diagnostic(clock=lambda: now)
        self.assertEqual(result.blockers, ("correction_transition_deadline_reached",))
        self.assertEqual(result.resource_use["fits_started"], 52)
        self.assertEqual(len(result.evidence["cells"]), 3)

    def test_invalid_records_preserve_zero_fit_blocked_package(self):
        self.records[0]["_id"] = ""
        result = self.run_diagnostic()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 0)
        self.assertTrue((self.output / "manifest.json").exists())

    def test_invalid_clock_preserves_blocked_package(self):
        result = self.run_diagnostic(clock=lambda: math.nan)
        self.assertEqual(result.blockers, ("correction_transition_invalid_clock",))
        self.assertEqual(result.resource_use["fits_started"], 0)
        self.assertTrue((self.output / "manifest.json").exists())

    def test_cli_and_public_seam_have_no_validation_test_seed_or_budget_input(self):
        parameters = inspect.signature(diagnostic.run_correction_transition_diagnostic).parameters
        self.assertNotIn("validation_records", parameters)
        self.assertNotIn("test_records", parameters)
        parser = diagnostic.build_parser()
        for flag in ("--validation-records", "--test-records", "--seed", "--maximum-model-fits"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--training-records", "unused", "--output-root", "unused", flag, "unused"])
        with patch.object(stability, "_verify_predeclared_training_artifact") as verify:
            with self.assertRaisesRegex(SystemExit, "correction_transition_output_root_mismatch"):
                diagnostic.main(["--training-records", "unused", "--output-root", str(self.output)])
            verify.assert_not_called()

    def test_cli_verifies_training_checksum_and_environment_before_runner(self):
        with patch.object(stability, "_verify_predeclared_training_artifact") as training_check, \
             patch.object(diagnostic.t, "TensorflowXGBoostCandidateRuntime", return_value=RecordingRuntime()), \
             patch.object(diagnostic.t, "_verify_predeclared_environment", side_effect=ValueError("blocked")) as env, \
             patch.object(diagnostic, "run_correction_transition_diagnostic") as run, \
             patch.object(diagnostic, "PREDECLARED_OUTPUT_ROOT", self.output):
            with self.assertRaisesRegex(SystemExit, "correction_transition_diagnostic_failed"):
                diagnostic.main(["--training-records", "unused", "--output-root", str(self.output)])
            training_check.assert_called_once_with(Path("unused"))
            env.assert_called_once_with(RecordingRuntime.dependency_versions)
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_all_fits_exclude_outer_holdout_and_keep_fixed_counts(self):
        from minires.ingestion import normalize
        training = normalize(self.records, self.config, contract="legacy")
        class FixedCountsRuntime(RecordingRuntime):
            def refit(inner_self, candidate, seed, features, targets, fixed_training_counts):
                key = "neural_network_epochs" if candidate.family == "neural_network" else "xgboost_trees"
                self.assertEqual(fixed_training_counts, {key: diagnostic.tail.FIXED_COUNTS[key]})
                return super(FixedCountsRuntime, inner_self).refit(
                    candidate, seed, features, targets, fixed_training_counts,
                )
        runtime = FixedCountsRuntime()
        self.assertEqual(self.run_diagnostic(runtime=runtime).status, "completed")
        for split_index, split in enumerate((101, 202)):
            assignment = diagnostic.t._cross_fit_assignments(training, split, 5)
            held_values = {float(record["volume"]) for record, fold in zip(self.records, assignment) if fold == 0}
            for _, _, features in runtime.fits[split_index * 24:(split_index + 1) * 24]:
                self.assertFalse(held_values.intersection(value[1] for value in features))

    def test_training_checksum_verifier_reads_no_other_artifact(self):
        expected = stability.PROJECT_ROOT / "data" / "train.jsonl"
        payload = b"synthetic only"
        checksum = sha256(payload).hexdigest()
        with patch.object(diagnostic.t, "GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS", {"train.jsonl": checksum}), \
             patch.object(Path, "read_text", autospec=True, return_value=json.dumps({"artifacts": {"train.jsonl": checksum}})) as manifest, \
             patch.object(Path, "read_bytes", autospec=True, return_value=payload) as artifact:
            stability._verify_predeclared_training_artifact(expected)
            manifest.assert_called_once_with(expected.parent / "manifest.json")
            artifact.assert_called_once_with(expected)

    def test_prior_plan_and_correction_contract_are_immutable(self):
        before = stability._plan("raw", "normalized")
        contract = copy.deepcopy(numeric.CONTRACT)
        new = diagnostic._plan("raw", "normalized", RecordingRuntime.dependency_versions)
        new["correction_contract"]["iterations"] = 0
        new["accounting_contract"]["serious"] = "changed"
        self.assertEqual(stability._plan("raw", "normalized"), before)
        self.assertEqual(numeric.CONTRACT, contract)
        self.assertEqual(before["version"], "minires-training-stability-diagnostic-v1")
        self.assertEqual(stability.PREDECLARED_OUTPUT_ROOT.name, "run-015")
        self.assertEqual(before["maximum_fits"], 52)
        self.assertEqual(before["maximum_elapsed_seconds"], 7200)
        self.assertEqual(before["inner_split_seeds"], {"101": 1101, "202": 1202})


if __name__ == "__main__":
    unittest.main()
