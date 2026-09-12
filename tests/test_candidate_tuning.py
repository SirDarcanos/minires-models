from dataclasses import replace
from hashlib import sha256
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
    main as assessment_main,
)
from minires_evaluation.ingestion import fingerprint, load_records
from minires_evaluation.tuning import (
    CandidateFoldFit,
    DeclaredCandidate,
    LockedFit,
    SearchLimits,
    create_search_plan,
    evaluate_declared_candidate,
    generate_search_plan,
    load_locked_candidate,
    main as tuning_main,
    tune_candidates,
    _selected_epoch_count,
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

    def serialize_fold(self, fitted):
        return {"model.bin": str(fitted.fitted_state).encode()}


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


class SeedTailRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        return replace(fitted, predictor=lambda rows: [
            row[1] / 1000.0 + (
                6.0 if seed == 42 and int(row[1] / 1000.0) % 100 < 2 else 0.0
            )
            for row in rows
        ])


class CrossSourceTailRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        shifted_source = 0 if seed == 41 else 1
        return replace(fitted, predictor=lambda rows: [
            row[1] / 1000.0 + (
                6.0
                if (int(row[1] / 1000.0) - 1) // 200 == shifted_source
                and (int(row[1] / 1000.0) - 1) % 200 < 4
                else 0.0
            )
            for row in rows
        ])


class SeedDurationRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        metadata = {
            "selected_epochs": 5 if seed == 41 else 15,
            "selected_trees": 10 if seed == 41 else 100,
        }
        return replace(fitted, metadata=metadata)


class SecondSeedFailingRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        if seed == 42:
            raise RuntimeError("private second seed failure")
        return super().fit_fold(candidate, seed, train_features, train_targets,
                                validation_features, validation_targets)


class OneIneligibleCandidateRuntime(RecordingTuningRuntime):
    def __init__(self):
        super().__init__()
        self.ineligible_candidate_id = None

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        if candidate.family == "neural_network" and self.ineligible_candidate_id is None:
            self.ineligible_candidate_id = candidate.candidate_id
        shift = 6.0 if candidate.candidate_id == self.ineligible_candidate_id else 0.0
        return replace(
            fitted,
            predictor=lambda rows: [row[1] / 1000.0 + shift for row in rows],
        )


class MostlyIneligibleRuntime(RecordingTuningRuntime):
    def __init__(self, eligible_ids):
        super().__init__()
        self.eligible_ids = set(eligible_ids)

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        shift = 0.0 if (
            candidate.family == "control" or candidate.candidate_id in self.eligible_ids
        ) else 6.0
        return replace(
            fitted,
            predictor=lambda rows: [row[1] / 1000.0 + shift for row in rows],
        )


class UnavailableTuningRuntime(RecordingTuningRuntime):
    startup_blockers = ("candidate_tuning_dependencies_required",)


class FailingCandidateRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        if candidate.family == "neural_network":
            raise RuntimeError("private runtime detail")
        return super().fit_fold(candidate, seed, train_features, train_targets,
                                validation_features, validation_targets)


class NonFiniteCandidateRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        return replace(fitted, predictor=lambda rows: [math.nan for _ in rows])


class SelectiveTailRuntime(RecordingTuningRuntime):
    def __init__(self, shifted_values):
        super().__init__()
        self.shifted_values = set(shifted_values)

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        fitted = super().fit_fold(candidate, seed, train_features, train_targets,
                                  validation_features, validation_targets)
        return replace(fitted, predictor=lambda rows: [
            row[1] / 1000.0 + (6.0 if row[1] / 1000.0 in self.shifted_values else 0.0)
            for row in rows
        ])


class PredictionMutatingRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.fit_calls.append((candidate.candidate_id, seed, tuple(train_features),
                               tuple(validation_features)))
        state = {"prediction_inputs": []}
        metadata = {"prediction_call_count": 0}

        def predict(rows):
            state["prediction_inputs"].append(tuple(rows))
            metadata["prediction_call_count"] += 1
            return [row[1] / 1000.0 for row in rows]

        return CandidateFoldFit(predictor=predict, metadata=metadata, fitted_state=state)


class CandidateSearchPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records = [
            CandidateTuningTests.row(source, family, index + 1)
            for index, (source, family) in enumerate(
                (("private-a", "a1"), ("private-a", "a2"),
                 ("private-b", "b1"), ("private-b", "b2"))
            )
        ]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

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
        self.assertEqual(len(first.ensemble_rules), 3)
        self.assertEqual(
            [rule["component_rank"] for rule in first.ensemble_rules], [1, 2, 3]
        )
        self.assertEqual(first.second_seed_rule["candidate_count"], 5)
        self.assertEqual(first.eligibility_gates["pooled_above_5g_fraction_maximum"], 0.01)
        self.assertFalse(first.control["candidate_slot_consumed"])
        self.assertIn("layers", first.parameter_domains["neural_network"])
        self.assertIn("n_jobs", first.parameter_domains["xgboost"])
        self.assertTrue(first.plan_id.startswith("search-plan-"))

    def test_plan_generation_persists_repeatable_private_checksummed_content(self):
        first_root = Path(self.temp.name) / "private" / "plan-a"
        second_root = Path(self.temp.name) / "private" / "plan-b"

        first = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"}, output_root=first_root,
        )
        second = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"}, output_root=second_root,
        )

        first_bytes = (first_root / "search-plan.json").read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, (second_root / "search-plan.json").read_bytes())
        manifest = json.loads((first_root / "manifest.json").read_text())
        self.assertEqual(
            manifest["artifacts"]["search-plan.json"], sha256(first_bytes).hexdigest()
        )
        self.assertTrue(manifest["create_only"])
        self.assertFalse(manifest["publication_performed"])
        self.assertNotIn("private-a", first_bytes.decode())
        self.assertNotIn("private-b", first_bytes.decode())
        with self.assertRaisesRegex(ValueError, "search_plan_output_unavailable"):
            create_search_plan(
                self.records, self.config, limits=SearchLimits(seed=41),
                dependency_versions={"runtime": "synthetic-1"}, output_root=first_root,
            )

    def test_plan_identity_binds_inputs_allocation_code_configuration_domain_and_version(self):
        baseline = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
        )
        changed_records = [dict(row) for row in self.records]
        changed_records[0]["weight"] += 1
        changed_input = create_search_plan(
            changed_records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
        )
        changed_allocation = create_search_plan(
            self.records, replace(self.config, seed=18), limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
        )
        changed_seed = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=42),
            dependency_versions={"runtime": "synthetic-1"},
        )
        changed_domains = {
            family: {name: tuple(values) for name, values in domain.items()}
            for family, domain in baseline.parameter_domains.items()
        }
        changed_domains["neural_network"]["activation"] = ("relu", "selu")
        changed_domain = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
            parameter_domains=changed_domains,
        )

        self.assertEqual(len({
            baseline.plan_id, changed_input.plan_id, changed_allocation.plan_id,
            changed_seed.plan_id, changed_domain.plan_id,
        }), 5)
        self.assertNotEqual(baseline.input_fingerprint, changed_input.input_fingerprint)
        self.assertNotEqual(
            baseline.source_allocation_fingerprint,
            changed_allocation.source_allocation_fingerprint,
        )
        self.assertTrue(baseline.normalized_input_fingerprint)
        self.assertTrue(baseline.code_configuration_fingerprint)
        self.assertEqual(baseline.generator_version, baseline.version)

    def test_domain_key_order_does_not_change_candidates_or_plan_identity(self):
        baseline = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
        )
        reversed_domains = {
            family: dict(reversed(tuple(domain.items())))
            for family, domain in reversed(tuple(baseline.parameter_domains.items()))
        }

        reordered = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions={"runtime": "synthetic-1"},
            parameter_domains=reversed_domains,
        )

        self.assertEqual(reordered, baseline)

    def test_invalid_domains_and_resources_fail_at_the_plan_generation_interface(self):
        invalid = {
            "neural_network": {
                "layers": ((64, 32),), "activation": ("unsupported",),
                "dropout": (math.nan,), "optimizer": ("adam",),
                "loss": ("mean_absolute_error",), "learning_rate": (0.001,),
                "l2": (0.0,), "batch_size": (32,), "maximum_epochs": (100,),
                "early_stopping_patience": (8,),
            },
            "xgboost": {},
        }
        invalid_cases = (
            {"parameter_domains": invalid},
            {"limits": SearchLimits(seed=41, ensemble_neural_network_weights=())},
            {"limits": SearchLimits(seed=41, neural_network_trials=7)},
            {"limits": SearchLimits(seed=41, maximum_elapsed_seconds=7200.001)},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, "invalid_search_plan"
            ):
                create_search_plan(
                    self.records, self.config,
                    limits=overrides.get("limits", SearchLimits(seed=41)),
                    dependency_versions={"runtime": "synthetic-1"},
                    parameter_domains=overrides.get("parameter_domains"),
                )


class DeclaredCandidateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records = [
            CandidateTuningTests.row(source, family, index + 1)
            for index, (source, family) in enumerate(
                (("a", "a1"), ("a", "a2"), ("b", "b1"),
                 ("b", "b2"), ("c", "c1"), ("c", "c2"))
            )
        ]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    @staticmethod
    def candidate(family):
        plan = generate_search_plan(
            SearchLimits(seed=41), input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        generated = next(item for item in plan.component_trials if item.family == family)
        return DeclaredCandidate(family=family, parameters=generated.parameters)

    def test_neural_network_and_xgboost_use_the_same_rotating_holdout_seam(self):
        for family in ("neural_network", "xgboost"):
            with self.subTest(family=family):
                runtime = RecordingTuningRuntime()
                result = evaluate_declared_candidate(
                    self.records, self.config, candidate=self.candidate(family),
                    runtime=runtime, seed=41,
                )

                self.assertEqual(result.status, "completed")
                self.assertEqual(result.contract["family"], family)
                self.assertEqual(len(result.source_reports), 3)
                self.assertEqual(len(runtime.fit_calls), 3)
                self.assertEqual(result.metrics["sample_count"], len(self.records))
                self.assertEqual(result.dependency_versions, runtime.dependency_versions)
                self.assertTrue(all(
                    {"sample_count", "mae_g", "within_2g_fraction", "above_5g_fraction"}
                    <= set(report)
                    for report in result.source_reports
                ))
                self.assertTrue(all(
                    "fitted_state_fingerprint" in metadata
                    for metadata in result.fit_metadata
                ))
                self.assertTrue(all(
                    not set(report["test_rows"]) &
                    (set(report["train_rows"]) | set(report["validation_rows"]))
                    for report in result.source_reports
                ))

    def test_contract_identifier_is_stable_and_invalid_values_block_before_fitting(self):
        candidate = self.candidate("xgboost")
        serialized = json.loads(json.dumps(candidate.to_dict(), sort_keys=True))
        equivalent = DeclaredCandidate(
            family=serialized["family"], parameters=serialized["parameters"]
        )
        self.assertEqual(candidate.candidate_id, equivalent.candidate_id)
        self.assertEqual(
            serialized["runtime_configuration"]["tree_method"], "hist"
        )
        self.assertEqual(
            serialized["runtime_configuration"]["eval_metric"], "mae"
        )
        with self.assertRaises(TypeError):
            candidate.parameters["n_jobs"] = -1

        runtime = RecordingTuningRuntime()
        invalid = DeclaredCandidate(
            family="xgboost", parameters={**candidate.parameters, "n_jobs": -1}
        )
        result = evaluate_declared_candidate(
            self.records, self.config, candidate=invalid, runtime=runtime, seed=41,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("invalid_candidate_configuration",))
        self.assertEqual(runtime.fit_calls, [])

    def test_missing_runtime_and_non_finite_predictions_return_bounded_blockers(self):
        candidate = self.candidate("neural_network")
        with patch(
            "minires_evaluation.tuning.TensorflowXGBoostCandidateRuntime",
            side_effect=ImportError("private dependency detail"),
        ):
            missing = evaluate_declared_candidate(
                self.records, self.config, candidate=candidate, seed=41,
            )
        self.assertEqual(
            missing.blockers, ("candidate_evaluation_dependencies_required",)
        )

        non_finite = evaluate_declared_candidate(
            self.records, self.config, candidate=candidate,
            runtime=NonFiniteCandidateRuntime(), seed=41,
        )
        self.assertEqual(non_finite.blockers, ("candidate_runtime_failed",))
        self.assertNotIn("nan", json.dumps(non_finite.to_dict()))

    def test_per_source_tail_gate_applies_only_at_200_accepted_records(self):
        candidate = self.candidate("xgboost")
        small_sources = [
            CandidateTuningTests.row(f"source-{source}", f"family-{source}-{row}",
                                     source * 2 + row + 1)
            for source in range(101) for row in range(2)
        ]

        small_tail = evaluate_declared_candidate(
            small_sources, self.config, candidate=candidate,
            runtime=SelectiveTailRuntime({1}), seed=41,
        )

        self.assertEqual(small_tail.status, "completed")
        self.assertEqual(small_tail.blockers, ())
        self.assertIsNone(small_tail.metrics["maximum_qualifying_source_above_5g_fraction"])

        qualifying_sources = [
            CandidateTuningTests.row(f"source-{source}", f"family-{source}-{row}",
                                     source * 200 + row + 1)
            for source in range(3) for row in range(200)
        ]
        qualifying_tail = evaluate_declared_candidate(
            qualifying_sources, self.config, candidate=candidate,
            runtime=SelectiveTailRuntime({1, 2, 3, 4, 5}), seed=41,
        )

        self.assertEqual(qualifying_tail.status, "completed")
        self.assertEqual(
            qualifying_tail.blockers, ("development_serious_error_gate_failed",)
        )
        self.assertGreater(
            qualifying_tail.metrics["maximum_qualifying_source_above_5g_fraction"],
            0.02,
        )

    def test_private_report_and_fold_artifacts_are_create_only_and_checksummed(self):
        root = Path(self.temp.name) / "private" / "declared-candidate"
        result = evaluate_declared_candidate(
            self.records, self.config, candidate=self.candidate("neural_network"),
            runtime=RecordingTuningRuntime(), seed=41, output_root=root,
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue((root / "candidate-report.json").exists())
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(set(result.artifact_checksums), {
            "fold-artifacts/fold-000/model.bin",
            "fold-artifacts/fold-001/model.bin",
            "fold-artifacts/fold-002/model.bin",
        })
        for name, checksum in manifest["artifacts"].items():
            self.assertEqual(sha256((root / name).read_bytes()).hexdigest(), checksum)
        blocked = evaluate_declared_candidate(
            self.records, self.config, candidate=self.candidate("neural_network"),
            runtime=RecordingTuningRuntime(), seed=41, output_root=root,
        )
        self.assertEqual(blocked.blockers, ("candidate_artifact_failed",))

    def test_held_out_data_changes_only_test_evidence_for_its_fold(self):
        baseline_root = Path(self.temp.name) / "private" / "baseline"
        baseline = evaluate_declared_candidate(
            self.records, self.config, candidate=self.candidate("neural_network"),
            runtime=PredictionMutatingRuntime(), seed=41, output_root=baseline_root,
        )
        changed = [dict(row) for row in self.records]
        changed[0] = {**changed[0], "kb": 999, "volume": 999000,
                      "surface_area": 99900, "bbox_area": 1098900,
                      "euler_number": 999, "scale": 999, "weight": 999}
        repeated_root = Path(self.temp.name) / "private" / "repeated"
        repeated = evaluate_declared_candidate(
            changed, self.config, candidate=self.candidate("neural_network"),
            runtime=PredictionMutatingRuntime(), seed=41, output_root=repeated_root,
        )

        before_index, before = next(
            (index, report) for index, report in enumerate(baseline.source_reports)
            if 0 in report["test_rows"]
        )
        after_index, after = next(
            (index, report) for index, report in enumerate(repeated.source_reports)
            if 0 in report["test_rows"]
        )
        self.assertEqual(baseline.fit_metadata[before_index], repeated.fit_metadata[after_index])
        self.assertEqual(
            baseline.artifact_checksums[
                f"fold-artifacts/fold-{before_index:03d}/model.bin"
            ],
            repeated.artifact_checksums[
                f"fold-artifacts/fold-{after_index:03d}/model.bin"
            ],
        )
        self.assertNotEqual(before["test_data_fingerprint"], after["test_data_fingerprint"])


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
        expected_finalists = [
            run.candidate.candidate_id
            for run in sorted(result.initial_results, key=lambda run: (
                run.metrics["source_balanced_mae_g"],
                run.metrics["pooled_mae_g"],
                -run.metrics["pooled_within_2g_fraction"],
                run.candidate.candidate_id,
            ))[:5]
        ]
        self.assertEqual(
            [run.candidate.candidate_id for run in result.second_seed_results],
            expected_finalists,
        )
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
        private_report = result.to_dict()
        initial_ranked_ids = [
            item["candidate_id"] for item in private_report["initial_promotable_ranking"]
        ]
        execution_ids = [item.candidate.candidate_id for item in result.initial_results]
        self.assertNotEqual(execution_ids, sorted(execution_ids))
        self.assertEqual(initial_ranked_ids, sorted(execution_ids))
        self.assertEqual(
            [item["rank"] for item in private_report["initial_promotable_ranking"]],
            list(range(1, 16)),
        )
        self.assertEqual(len(private_report["promotable_ranking"]), 5)
        self.assertTrue(all(
            item["rationale"] == list(result.plan.ranking_rule)
            for item in private_report["promotable_ranking"]
        ))
        self.assertEqual(
            len(private_report["candidate_history"]),
            1 + result.run_count,
        )
        self.assertEqual(private_report["candidate_history"][0]["candidate_id"],
                         "clean-fixed-control")
        self.assertEqual(private_report["candidate_history"][0]["allocation"], "control")
        self.assertEqual(len(private_report["selected_ensemble_weights"]), 3)

    def test_ineligible_candidate_remains_in_history_but_not_promotable_ranking(self):
        runtime = OneIneligibleCandidateRuntime()

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        report = result.to_dict()
        assert runtime.ineligible_candidate_id is not None
        ineligible = next(
            item for item in report["initial_results"]
            if item["candidate"]["candidate_id"] == runtime.ineligible_candidate_id
        )
        self.assertFalse(ineligible["eligible"])
        self.assertIn("development_serious_error_gate_failed", ineligible["blockers"])
        self.assertNotIn(
            runtime.ineligible_candidate_id,
            [item["candidate_id"] for item in report["promotable_ranking"]],
        )
        self.assertIn(
            runtime.ineligible_candidate_id,
            [item["candidate_id"] for item in report["candidate_history"]],
        )

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

    def test_combined_eligibility_is_recomputed_instead_of_requiring_each_seed_to_pass(self):
        records = [
            self.row(f"source-{source}", f"family-{source}-{row}", source * 200 + row)
            for source in range(3) for row in range(200)
        ]

        result = tune_candidates(
            records, self.config, runtime=SeedTailRuntime(), output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertTrue(all(not run.eligible for run in result.second_seed_results))
        self.assertTrue(all(item["eligible"] for item in result.combined_results))
        self.assertTrue(all(
            0.0 < item["metrics"]["pooled_above_5g_fraction"] <= 0.01
            for item in result.combined_results
        ))
        self.assertIsNotNone(result.locked_candidate)

    def test_combined_per_source_gate_retains_each_seed_before_taking_the_maximum(self):
        records = [
            self.row(f"source-{source}", f"family-{source}-{row}", source * 200 + row + 1)
            for source in range(3) for row in range(200)
        ]

        result = tune_candidates(
            records, self.config, runtime=CrossSourceTailRuntime(), output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertTrue(all(run.eligible for run in result.second_seed_results))
        self.assertTrue(all(
            item["metrics"]["maximum_qualifying_source_above_5g_fraction"] == 0.01
            for item in result.combined_results
        ))

    def test_repeats_every_eligible_candidate_and_records_a_best_five_shortfall(self):
        plan = create_search_plan(
            self.records, self.config, limits=SearchLimits(seed=41),
            dependency_versions=RecordingTuningRuntime.dependency_versions,
        )
        neural = next(item for item in plan.component_trials if item.family == "neural_network")
        xgboost = next(item for item in plan.component_trials if item.family == "xgboost")
        runtime = MostlyIneligibleRuntime({neural.candidate_id, xgboost.candidate_id})

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0, plan=plan,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 18)
        self.assertEqual(len(result.second_seed_results), 3)
        self.assertEqual(result.to_dict()["second_seed_comparison"], {
            "planned_finalists": 5,
            "eligible_initial_candidates": 3,
            "repeated_candidates": 3,
            "shortfall": 2,
            "complete": True,
        })
        self.assertIsNotNone(result.locked_candidate)

    def test_fixed_training_counts_use_both_seeds_and_lock_the_ensemble_contract(self):
        runtime = SeedDurationRuntime()

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertIsNotNone(result.locked_candidate)
        assert result.locked_candidate is not None
        contract = result.locked_candidate.contract
        self.assertEqual(contract["selection_seeds"], [41, 42])
        self.assertEqual(contract["fixed_training_counts"], {
            "neural_network_epochs": 10,
            "xgboost_trees": 55,
            "ensemble_neural_network_weight": 0.0,
        })
        self.assertEqual(contract["candidate"]["family"], "ensemble")
        self.assertIn("neural_network", contract["candidate"]["parameters"])
        self.assertIn("xgboost", contract["candidate"]["parameters"])
        self.assertEqual(contract["runtime_configuration"]["combination"],
                         "locked_convex_weight")
        self.assertEqual(contract["model_specification"]["model_kind"], "ensemble")
        self.assertEqual(
            contract["model_specification"]["stable_identity"],
            result.locked_candidate.specification.stable_identity,
        )
        self.assertEqual(contract["development_evidence"]["search_plan_id"], result.plan.plan_id)
        self.assertEqual(contract["dependency_environment"]["versions"], runtime.dependency_versions)
        self.assertEqual(contract["refit_partition"], "all_included_development_records")

    def test_neural_refit_count_matches_the_epoch_restored_by_early_stopping(self):
        self.assertEqual(_selected_epoch_count([5.0, 4.5, 4.45, 4.39], 0.1), 4)

    def test_partial_second_seed_stage_cannot_lock_a_candidate(self):
        result = tune_candidates(
            self.records, self.config, runtime=SecondSeedFailingRuntime(),
            output_root=self.root, limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(result.second_seed_results), 1)
        self.assertFalse(result.to_dict()["second_seed_comparison"]["complete"])
        self.assertIsNone(result.locked_candidate)

    def test_no_eligible_candidate_records_the_best_development_result_without_refitting(self):
        runtime = MostlyIneligibleRuntime(set())

        result = tune_candidates(
            self.records, self.config, runtime=runtime, output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        report = result.to_dict()
        self.assertEqual(result.status, "completed_no_candidate")
        self.assertEqual(result.blockers, ("no_eligible_candidate",))
        self.assertEqual(result.run_count, 15)
        self.assertEqual(report["second_seed_comparison"]["shortfall"], 5)
        self.assertIsNotNone(report["best_development_result"])
        self.assertFalse(report["best_development_result"]["eligible"])
        self.assertEqual(runtime.refit_calls, [])
        self.assertIsNone(result.locked_candidate)

    def test_startup_failure_records_its_reason_for_every_skipped_slot(self):
        result = tune_candidates(
            self.records, self.config, runtime=UnavailableTuningRuntime(),
            output_root=self.root, limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        report = result.to_dict()
        self.assertEqual(result.blockers, ("candidate_tuning_dependencies_required",))
        self.assertEqual(result.run_count, 0)
        self.assertEqual(len(report["candidate_history"]), 20)
        self.assertTrue(all(
            item["status"] == "skipped"
            and item["reason"] == "candidate_tuning_dependencies_required"
            for item in report["candidate_history"]
        ))

    def test_runtime_failure_stops_the_round_and_cannot_lock_another_candidate(self):
        result = tune_candidates(
            self.records, self.config, runtime=FailingCandidateRuntime(),
            output_root=self.root, limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.run_count, 1)
        self.assertIn("candidate_runtime_failed", result.blockers)
        self.assertIsNone(result.locked_candidate)
        private_report = result.to_dict()
        self.assertNotIn("private runtime detail", json.dumps(private_report))
        self.assertEqual(result.initial_results[0].status, "failed")
        self.assertEqual(private_report["promotable_ranking"], [])
        self.assertEqual(
            [item["status"] for item in private_report["candidate_history"][:2]],
            ["completed", "failed"],
        )
        self.assertTrue(all(
            item["status"] == "skipped"
            and item["reason"] == "candidate_runtime_failed"
            for item in private_report["candidate_history"][2:]
        ))

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
        self.assertTrue(all(
            item["reason"] == "candidate_search_deadline_reached"
            for item in result.to_dict()["skipped_candidates"]
        ))
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


class ShortLockedPredictionRuntime(RecordingTuningRuntime):
    def load_locked(self, candidate, directory, contract):
        return lambda rows: [row[1] / 1000.0 for row in rows[:-1]]


class CandidatePredictionRuntime(RecordingTuningRuntime):
    def __init__(self, predict):
        super().__init__()
        self.predict = predict

    def load_locked(self, candidate, directory, contract):
        return self.predict


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
        self.assertIn("not_operational_allowance", public["limitations"])
        self.assertEqual(
            json.loads((self.root / "assessment" / "public-summary-review.json").read_text())[
                "status"
            ],
            "automated_screening_passed",
        )
        manifest = json.loads((self.root / "assessment" / "manifest.json").read_text())
        for name, checksum in manifest["artifacts"].items():
            self.assertEqual(
                sha256((self.root / "assessment" / name).read_bytes()).hexdigest(),
                checksum,
            )
        self.assertFalse(manifest["publication_performed"])

    def test_one_failed_gate_composes_an_honest_not_promoted_outcome(self):
        runtime = CandidatePredictionRuntime(
            lambda rows: [row[1] / 1000.0 + 3.0 for row in rows]
        )

        result = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment-not-promoted",
            runtime=runtime,
        )

        self.assertEqual(result.status, "not_promoted")
        self.assertFalse(result.promoted)
        self.assertEqual(result.blockers, ("promotion_gates_not_met",))
        self.assertFalse(result.gates["pooled_mae_noninferior"])
        self.assertTrue(result.gates["pooled_tail"])
        self.assertIn("pooled_tail_confidence_interval_95", result.confidence_analysis)
        self.assertIn("tail_confidence_interval_95", result.source_reports[0])

    def test_insufficient_final_sources_block_before_scoring(self):
        result = assess_locked_candidate(
            self.final_rows(("only-one",)), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment",
            runtime=self.runtime,
        )

        self.assertEqual(result.status, "blocked")
        self.assertIn("insufficient_final_source_groups", result.blockers)
        self.assertEqual(result.predictions, ())

    def test_final_sources_are_confirmed_absent_from_candidate_development(self):
        self.assertEqual(
            len(self.locked.contract["development_source_groups"]), 3
        )
        self.assertEqual(
            set(self.locked.contract["development_data_usage"]),
            {
                "fitting", "preprocessing", "early_stopping", "ensemble_selection",
                "threshold_selection", "candidate_locking",
            },
        )

        overlap = assess_locked_candidate(
            self.final_rows(("a", "new-b", "new-c")),
            EvaluationConfig(None, "mm3", True), self.locked, self.legacy,
            output_root=self.root / "assessment-overlap", runtime=self.runtime,
        )
        self.assertEqual(overlap.status, "blocked")
        self.assertIn("final_source_used_in_candidate_development", overlap.blockers)
        self.assertEqual(overlap.predictions, ())

    def test_undersized_sources_and_distinct_row_outcomes_block_with_accounting(self):
        undersized = self.final_rows()
        del undersized[-1]
        result = assess_locked_candidate(
            undersized, EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment-undersized",
            runtime=self.runtime,
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("insufficient_final_source_records", result.blockers)
        self.assertEqual(result.row_accounting["source_counts"], {
            fingerprint("new-a"): 200,
            fingerprint("new-b"): 200,
            fingerprint("new-c"): 199,
        })

        mixed = self.final_rows()
        invalid = dict(mixed[0], volume=0)
        outside_scope = dict(mixed[1], scope_confirmed=False)
        missing_scope = dict(mixed[2], scope_confirmed=None)
        result = assess_locked_candidate(
            [*mixed, invalid, outside_scope, missing_scope],
            EvaluationConfig(None, "mm3", True), self.locked, self.legacy,
            output_root=self.root / "assessment-accounting", runtime=self.runtime,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.row_accounting["accepted_count"], 600)
        self.assertEqual(result.row_accounting["excluded_count"], 1)
        self.assertEqual(result.row_accounting["needs_review_count"], 2)
        self.assertEqual(result.row_accounting["reasons"]["invalid_volume"], 1)
        self.assertEqual(result.row_accounting["reasons"]["unsupported_scope"], 1)
        self.assertEqual(
            result.row_accounting["reasons"]["scope_confirmation_required"], 1
        )
        self.assertIn("final_row_accounting_incomplete", result.blockers)
        self.assertEqual(result.predictions, ())

    def test_missing_evidence_and_prediction_row_mismatch_return_bounded_blockers(self):
        missing = assess_locked_candidate(
            self.root / "missing-final-records.json",
            EvaluationConfig(None, "mm3", True), self.locked, self.legacy,
            output_root=self.root / "assessment-missing", runtime=self.runtime,
        )
        self.assertEqual(missing.status, "blocked")
        self.assertEqual(missing.blockers, ("final_evidence_unavailable_or_malformed",))
        self.assertTrue((self.root / "assessment-missing" / "assessment.json").exists())

        mismatch = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy, output_root=self.root / "assessment-mismatch",
            runtime=ShortLockedPredictionRuntime(),
        )
        self.assertEqual(mismatch.status, "blocked")
        self.assertEqual(
            mismatch.blockers, ("paired_prediction_row_mismatch_or_failure",)
        )
        self.assertEqual(mismatch.predictions, ())

        malformed_path = self.root / "malformed-final-records.jsonl"
        malformed_path.write_text('{"partial":\n')
        malformed = assess_locked_candidate(
            malformed_path, EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy,
            output_root=self.root / "assessment-malformed", runtime=self.runtime,
        )
        self.assertEqual(malformed.status, "blocked")
        self.assertEqual(
            malformed.blockers, ("final_evidence_unavailable_or_malformed",)
        )

    def test_malformed_locked_contract_returns_persisted_bounded_outcomes(self):
        contract_path = self.locked.directory / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["dependency_environment"] = None
        contract_path.write_text(json.dumps(contract))
        manifest_path = self.locked.directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["candidate-contract.json"] = sha256(
            contract_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

        result = assess_locked_candidate(
            self.final_rows(), EvaluationConfig(None, "mm3", True),
            self.locked, self.legacy,
            output_root=self.root / "assessment-invalid-lock", runtime=self.runtime,
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("locked_candidate_contract_mismatch", result.blockers)

        cli_output = io.StringIO()
        with patch(
            "minires_evaluation.assessment.TensorflowXGBoostCandidateRuntime",
            return_value=self.runtime,
        ), contextlib.redirect_stdout(cli_output):
            code = assessment_main([
                "--records", str(self.root / "unused.json"),
                "--locked-candidate", str(self.locked.directory),
                "--legacy-artifacts", str(self.root / "unused-legacy"),
                "--output-root", str(self.root / "assessment-invalid-lock-cli"),
                "--volume-unit", "mm3", "--scope-confirmed",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(cli_output.getvalue())["status"], "blocked"
        )
        self.assertTrue(
            (self.root / "assessment-invalid-lock-cli" / "assessment.json").exists()
        )

    def test_changed_locked_artifact_blocks_assessment(self):
        (self.locked.directory / "model.bin").write_bytes(b"changed")

        with patch("minires_evaluation.assessment.load_records") as final_loader:
            result = assess_locked_candidate(
                self.final_rows(), EvaluationConfig(None, "mm3", True),
                self.locked, self.legacy, output_root=self.root / "assessment",
                runtime=self.runtime,
            )

        final_loader.assert_not_called()
        self.assertEqual(result.status, "blocked")
        self.assertIn("locked_candidate_checksum_mismatch", result.blockers)
        self.assertEqual(result.row_accounting["input_count"], 0)

    def test_changed_contract_preprocessing_or_dependency_invalidates_the_lock(self):
        cases = {
            "candidate-contract.json": lambda value: {**value, "output_unit": "kg"},
            "preprocessing-state.json": lambda value: {**value, "fit_rows": 999},
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                target = self.locked.directory / name
                original = target.read_bytes()
                value = json.loads(original)
                target.write_text(json.dumps(mutate(value)))
                with self.assertRaisesRegex(Exception, "locked_candidate_checksum_mismatch"):
                    load_locked_candidate(self.locked.directory, self.runtime)
                target.write_bytes(original)

        mismatched = RecordingTuningRuntime()
        mismatched.dependency_versions = {"runtime": "synthetic-2"}
        with self.assertRaisesRegex(Exception, "locked_candidate_dependency_mismatch"):
            load_locked_candidate(self.locked.directory, mismatched)

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

    def test_family_clustered_bootstrap_has_known_deterministic_bounds(self):
        rows = self.final_rows()
        for index, row in enumerate(rows):
            row["miniature_family"] = f"{row['anonymous_source_group']}-{index % 200 // 100}"
        runtime = CandidatePredictionRuntime(lambda matrix: [
            row[1] / 1000.0 + (0.0 if (int(row[1] / 1000.0) - 1) % 200 < 100 else 2.0)
            for row in matrix
        ])
        legacy_predictor = lambda matrix: [row[1] / 1000.0 + 1.0 for row in matrix]
        legacy = LegacyReference.from_predictors(
            neural_network=legacy_predictor, xgboost=legacy_predictor,
            neural_network_weight=0.2, provenance=LegacyProvenance.unknown(),
        )

        first = assess_locked_candidate(
            rows, EvaluationConfig(None, "mm3", True), self.locked, legacy,
            output_root=self.root / "assessment-clustered-a", runtime=runtime,
        )
        second = assess_locked_candidate(
            rows, EvaluationConfig(None, "mm3", True), self.locked, legacy,
            output_root=self.root / "assessment-clustered-b", runtime=runtime,
        )

        self.assertEqual(first.confidence_analysis, second.confidence_analysis)
        self.assertAlmostEqual(
            first.confidence_analysis["pooled_mae_relative_regression_upper_95"],
            2 / 3,
        )
        self.assertAlmostEqual(
            first.confidence_analysis[
                "source_balanced_mae_relative_regression_upper_95"
            ],
            2 / 3,
        )
        self.assertEqual(
            first.confidence_analysis["source_balanced_weighting"],
            "each_observed_source_equal_in_every_replicate",
        )

    def test_source_balanced_bootstrap_weights_unequal_sources_equally(self):
        rows = []
        for source_index, count in enumerate((200, 300, 400)):
            for index in range(count):
                value = source_index * 1000 + index + 1
                rows.append({
                    **CandidateTuningTests.row(
                        f"unequal-{source_index}", f"family-{source_index}", value
                    ),
                    "slicing_conditions": {"layer_height_mm": 0.05},
                })
        runtime = CandidatePredictionRuntime(lambda matrix: [
            row[1] / 1000.0 + int(row[1] / 1_000_000.0) for row in matrix
        ])
        legacy_predictor = lambda matrix: [row[1] / 1000.0 + 1.0 for row in matrix]
        legacy = LegacyReference.from_predictors(
            neural_network=legacy_predictor, xgboost=legacy_predictor,
            neural_network_weight=0.2, provenance=LegacyProvenance.unknown(),
        )

        result = assess_locked_candidate(
            rows, EvaluationConfig(None, "mm3", True), self.locked, legacy,
            output_root=self.root / "assessment-unequal-sources", runtime=runtime,
        )

        self.assertAlmostEqual(result.observed["pooled_candidate_mae_g"], 11 / 9)
        self.assertAlmostEqual(
            result.observed["source_balanced_candidate_mae_g"], 1.0
        )
        self.assertAlmostEqual(
            result.confidence_analysis["pooled_mae_relative_regression_upper_95"],
            2 / 9,
        )
        self.assertAlmostEqual(
            result.confidence_analysis[
                "source_balanced_mae_relative_regression_upper_95"
            ],
            0.0,
        )

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
            "pooled_tail_confidence_interval_95": [0.0, 1.0],
            "source_balanced_tail_confidence_interval_95": [0.0, 1.0],
        }

        gates = evaluate_promotion_gates(observed, confidence, FinalAssessmentConfig())

        self.assertTrue(all(gates.values()))
        cases = {
            "pooled_mae_noninferior": (
                confidence, "pooled_mae_relative_regression_upper_95", 0.020001
            ),
            "source_balanced_mae_noninferior": (
                confidence, "source_balanced_mae_relative_regression_upper_95", 0.020001
            ),
            "pooled_within_2g_noninferior": (
                confidence, "pooled_within_2g_difference_lower_95", -0.010001
            ),
            "source_balanced_within_2g_noninferior": (
                confidence, "source_balanced_within_2g_difference_lower_95", -0.010001
            ),
            "pooled_tail": (observed, "pooled_above_5g_fraction", 0.010001),
            "source_balanced_tail": (
                observed, "source_balanced_above_5g_fraction", 0.010001
            ),
            "every_source_tail": (
                observed, "maximum_source_above_5g_fraction", 0.020001
            ),
        }
        for gate, (target, key, value) in cases.items():
            with self.subTest(gate=gate):
                changed_observed = dict(observed)
                changed_confidence = dict(confidence)
                if target is observed:
                    changed_observed[key] = value
                else:
                    changed_confidence[key] = value
                self.assertFalse(evaluate_promotion_gates(
                    changed_observed, changed_confidence
                )[gate])
        inconclusive = dict(confidence, pooled_mae_relative_regression_upper_95=None)
        self.assertFalse(evaluate_promotion_gates(
            observed, inconclusive
        )["pooled_mae_noninferior"])


if __name__ == "__main__":
    unittest.main()
