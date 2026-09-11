from dataclasses import replace
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires_evaluation import EvaluationConfig, LegacyProvenance, LegacyReference
from minires_evaluation.assessment import (
    FinalAssessmentConfig,
    assess_locked_candidate,
    evaluate_promotion_gates,
)
from minires_evaluation.ingestion import fingerprint, load_records
from minires_evaluation.tuning import (
    CandidateFoldFit,
    LockedFit,
    SearchLimits,
    generate_search_plan,
    load_locked_candidate,
    main as tuning_main,
    tune_candidates,
)


class RecordingTuningRuntime:
    dependency_versions = {"runtime": "synthetic-1"}

    def __init__(self):
        self.fit_calls = []
        self.refit_calls = []

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.fit_calls.append((candidate.candidate_id, seed, tuple(train_features),
                               tuple(validation_features)))
        return CandidateFoldFit(
            predictor=lambda rows: [row[1] / 1000.0 for row in rows],
            metadata={
                "selected_epochs": 4 if candidate.family == "neural_network" else None,
                "selected_trees": 20 if candidate.family == "xgboost" else None,
            },
            fitted_state={"candidate": candidate.candidate_id, "seed": seed},
        )

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.refit_calls.append((candidate.candidate_id, tuple(features), fixed_training_counts))
        return LockedFit(
            predictor=lambda rows: [row[1] / 1000.0 for row in rows],
            preprocessing_state={"fit_rows": len(features)},
            artifacts={"model.bin": candidate.candidate_id.encode()},
            metadata={"seed": seed},
        )

    def load_locked(self, candidate, directory, contract):
        return lambda rows: [row[1] / 1000.0 for row in rows]


class SeedShiftRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        shift = 2.0 if seed == 42 else 0.0
        return replace(
            fitted,
            predictor=lambda rows: [row[1] / 1000.0 + shift for row in rows],
        )


class FailingCandidateRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        if candidate.family == "neural_network":
            raise RuntimeError("private runtime detail")
        return super().fit_fold(candidate, seed, train_features, train_targets,
                                validation_features, validation_targets)


class CandidateSearchPlanTests(unittest.TestCase):
    def test_same_identity_and_seed_generate_the_same_bounded_plan(self):
        limits = SearchLimits(seed=41)

        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(first, second)
        self.assertEqual([item.family for item in first.component_trials].count("neural_network"), 6)
        self.assertEqual([item.family for item in first.component_trials].count("xgboost"), 6)
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 20)
        self.assertEqual(first.resource_limits["maximum_elapsed_seconds"], 7200.0)
        self.assertEqual(first.ensemble_rule["trial_count"], 3)
        self.assertIn("layers", first.parameter_domains["neural_network"])
        self.assertIn("n_jobs", first.parameter_domains["xgboost"])

    def test_search_plan_cannot_expand_the_two_hour_budget(self):
        with self.assertRaisesRegex(ValueError, "invalid_search_plan"):
            generate_search_plan(
                SearchLimits(seed=41, maximum_elapsed_seconds=7200.001),
                input_fingerprint="input", code_fingerprint="code",
                dependency_versions={"runtime": "synthetic-1"},
            )


class CandidateTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private" / "tuning"
        self.records = [
            self.row(source, family, index + 1)
            for index, (source, family) in enumerate(
                (("a", "a1"), ("a", "a2"), ("b", "b1"),
                 ("b", "b2"), ("c", "c1"), ("c", "c2"))
            )
        ]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    @staticmethod
    def row(source, family, value):
        return {
            "kb": value, "volume": value * 1000, "surface_area": value * 100,
            "bbox_area": value * 1100, "euler_number": value, "scale": value,
            "surface_volume_ratio": 0.1, "weight": value,
            "anonymous_source_group": source, "miniature_family": family,
        }

    def test_allocates_exact_initial_and_second_seed_runs_then_locks_the_winner(self):
        runtime = RecordingTuningRuntime()

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 20)
        self.assertTrue(all(seed == 34 for _, seed, *_ in runtime.fit_calls[:3]))
        self.assertTrue(all(seed != 34 for _, seed, *_ in runtime.fit_calls[3:]))
        self.assertEqual(result.allocation, {
            "neural_network": 6, "xgboost": 6, "ensemble": 3,
            "second_seed": 5, "control": 1,
        })
        self.assertEqual(len(result.initial_results), 15)
        self.assertEqual(len(result.second_seed_results), 5)
        self.assertIsNotNone(result.locked_candidate)
        self.assertEqual(len(runtime.refit_calls), 1)
        self.assertEqual(len(runtime.refit_calls[0][1]), len(self.records))
        self.assertTrue(all(len(features) == 7 for call in runtime.fit_calls
                            for partition in call[2:4] for features in partition))
        self.assertTrue((self.root / "search-plan.json").exists())
        self.assertTrue((self.root / "locked-candidate" / "lock-manifest.json").exists())
        self.assertTrue((self.root / "manifest.json").exists())
        reloaded = load_locked_candidate(self.root / "locked-candidate", runtime)
        self.assertEqual(reloaded.candidate, result.locked_candidate.candidate)
        public = result.to_dict(public=True)
        self.assertNotIn("source_reports", str(public))
        self.assertNotIn("miniature_family", str(public))
        for report in result.initial_results[0].source_reports:
            self.assertFalse(set(report["train_rows"]) & set(report["test_rows"]))
            self.assertFalse(set(report["validation_rows"]) & set(report["test_rows"]))
        ensembles = [item for item in result.initial_results
                     if item.candidate.family == "ensemble"]
        self.assertEqual(len(ensembles), 3)
        self.assertTrue(all(
            item.candidate.parameters["selection_partition"] == "fold_validation_only"
            for item in ensembles
        ))

    def test_second_seed_aggregation_retains_the_unfavorable_repetition(self):
        result = tune_candidates(
            self.records, self.config, runtime=SeedShiftRuntime(),
            output_root=self.root, limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.combined_results), 5)
        for combined in result.combined_results:
            self.assertEqual(combined["seed_results"], [41, 42])
            self.assertEqual(combined["equal_seed_weight"], 0.5)
            self.assertAlmostEqual(combined["metrics"]["pooled_mae_g"], 1.0)

    def test_runtime_failure_stops_the_round_and_cannot_lock_another_candidate(self):
        result = tune_candidates(
            self.records, self.config, runtime=FailingCandidateRuntime(),
            output_root=self.root, limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.run_count, 1)
        self.assertIn("candidate_runtime_failed", result.blockers)
        self.assertIsNone(result.locked_candidate)
        self.assertNotIn("private runtime detail", json.dumps(result.to_dict()))

    def test_deadline_stops_before_a_later_launch_and_marks_the_search_partial(self):
        runtime = RecordingTuningRuntime()
        moments = iter((0.0, 0.0, 7200.0, 7200.0, 7200.0))

        result = tune_candidates(
            self.records, self.config, runtime=runtime,
            output_root=self.root, limits=SearchLimits(seed=41),
            clock=lambda: next(moments),
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.run_count, 1)
        self.assertIn("candidate_search_deadline_reached", result.blockers)
        self.assertEqual(result.to_dict()["skipped_candidate_runs"], 19)
        self.assertEqual(len(result.to_dict()["skipped_candidates"]), 19)
        self.assertIsNone(result.locked_candidate)

    def test_cli_creates_private_result_and_emits_only_bounded_status(self):
        records = Path(self.temp.name) / "records.json"
        records.write_text(json.dumps(self.records))
        output = io.StringIO()
        with patch(
            "minires_evaluation.tuning.TensorflowXGBoostCandidateRuntime",
            return_value=RecordingTuningRuntime(),
        ), contextlib.redirect_stdout(output):
            code = tuning_main([
                "--records", str(records), "--output-root", str(self.root),
                "--volume-unit", "mm3", "--scope-confirmed", "--seed", "41",
            ])

        status = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["run_count"], 20)
        self.assertNotIn(str(records), output.getvalue())
        self.assertNotIn("source_reports", output.getvalue())
        self.assertTrue((self.root / "tuning-result.json").exists())

    def test_invalid_unbounded_worker_plan_is_rejected_before_runtime_fitting(self):
        runtime = RecordingTuningRuntime()
        loaded, input_identity = load_records(self.records)
        del loaded
        code_identity = fingerprint({
            path.name: path.read_text()
            for path in sorted((Path(__file__).parents[1] / "minires_evaluation").glob("*.py"))
        })
        plan = generate_search_plan(
            SearchLimits(seed=41), input_fingerprint=input_identity,
            code_fingerprint=code_identity, dependency_versions=runtime.dependency_versions,
        )
        target = next(item for item in plan.component_trials if item.family == "xgboost")
        unsafe = replace(target, parameters={**target.parameters, "n_jobs": -1})
        plan = replace(plan, component_trials=tuple(
            unsafe if item == target else item for item in plan.component_trials
        ))

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), plan=plan,
        )

        self.assertEqual(result.blockers, ("invalid_search_plan",))
        self.assertEqual(result.run_count, 0)
        self.assertEqual(runtime.fit_calls, [])


class LockedAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / "private"
        self.runtime = RecordingTuningRuntime()
        development = [
            CandidateTuningTests.row(source, family, index + 1)
            for index, (source, family) in enumerate(
                (("a", "a1"), ("a", "a2"), ("b", "b1"),
                 ("b", "b2"), ("c", "c1"), ("c", "c2"))
            )
        ]
        tuned = tune_candidates(
            development, EvaluationConfig(None, "mm3", True, seed=17),
            runtime=self.runtime, output_root=root / "tuning",
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )
        assert tuned.locked_candidate is not None
        self.locked = tuned.locked_candidate
        self.root = root
        predictor = lambda rows: [row[1] / 1000.0 + 0.5 for row in rows]
        self.legacy = LegacyReference.from_predictors(
            neural_network=predictor, xgboost=predictor,
            neural_network_weight=0.2, provenance=LegacyProvenance.unknown(),
        )

    def final_rows(self, sources=("new-a", "new-b", "new-c")):
        rows = []
        for source_index, source in enumerate(sources):
            for index in range(200):
                value = source_index * 200 + index + 1
                rows.append({
                    **CandidateTuningTests.row(source, f"{source}-family", value),
                    "slicing_conditions": {"layer_height_mm": 0.05},
                })
        return rows

    def test_promotes_only_after_paired_clustered_assessment_on_three_qualifying_sources(self):
        result = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment",
            runtime=self.runtime,
        )

        self.assertEqual(result.status, "promoted")
        self.assertTrue(result.promoted)
        self.assertTrue(all(result.gates.values()))
        self.assertEqual(result.confidence_analysis["replicates"], 10_000)
        self.assertEqual(result.confidence_analysis["seed"], 1729)
        self.assertTrue(result.confidence_analysis["tail_intervals_are_reported_not_gated"])
        public = result.to_dict(public=True)
        self.assertNotIn("predictions", public)
        self.assertNotIn("source_reports", public)
        self.assertNotIn("source_counts", public["row_accounting"])

    def test_insufficient_final_sources_block_before_scoring(self):
        result = assess_locked_candidate(
            self.final_rows(("only-one",)), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment",
            runtime=self.runtime,
        )

        self.assertEqual(result.status, "blocked")
        self.assertIn("insufficient_final_source_groups", result.blockers)
        self.assertEqual(result.predictions, ())

    def test_changed_locked_artifact_blocks_assessment(self):
        (self.locked.directory / "model.bin").write_bytes(b"changed")

        result = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment",
            runtime=self.runtime,
        )

        self.assertEqual(result.status, "blocked")
        self.assertIn("locked_candidate_checksum_mismatch", result.blockers)
        self.assertEqual(result.row_accounting["input_count"], 0)

    def test_assessment_reloads_verified_artifacts_instead_of_using_supplied_callable(self):
        substituted = replace(
            self.locked,
            predictor=lambda rows: [10_000.0 for _ in rows],
        )

        result = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            substituted, self.legacy, output_root=self.root / "assessment",
            runtime=self.runtime,
        )

        self.assertEqual(result.status, "promoted")

    def test_empty_slicing_condition_evidence_blocks_promotion(self):
        rows = self.final_rows()
        for row in rows:
            row["slicing_conditions"] = {}

        result = assess_locked_candidate(
            rows, EvaluationConfig(None, "mm3", True), self.locked, self.legacy,
            output_root=self.root / "assessment", runtime=self.runtime,
        )

        self.assertEqual(result.status, "blocked")
        self.assertIn("final_scope_evidence_incomplete", result.blockers)
        self.assertEqual(result.row_accounting["accepted_count"], 0)

    def test_noninferiority_and_tail_boundaries_are_inclusive(self):
        observed = {
            "pooled_above_5g_fraction": 0.01,
            "source_balanced_above_5g_fraction": 0.01,
            "maximum_source_above_5g_fraction": 0.02,
        }
        confidence = {
            "pooled_mae_relative_regression_upper_95": 0.02,
            "source_balanced_mae_relative_regression_upper_95": 0.02,
            "pooled_within_2g_difference_lower_95": -0.01,
            "source_balanced_within_2g_difference_lower_95": -0.01,
        }

        gates = evaluate_promotion_gates(observed, confidence, FinalAssessmentConfig())

        self.assertTrue(all(gates.values()))
        outside = dict(confidence, pooled_mae_relative_regression_upper_95=0.020001)
        self.assertFalse(evaluate_promotion_gates(observed, outside)["pooled_mae_noninferior"])


if __name__ == "__main__":
    unittest.main()
