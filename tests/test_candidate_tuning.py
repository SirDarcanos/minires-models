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

from minires import EvaluationConfig, LegacyProvenance, LegacyReference
from minires.evaluation.assessment import (
    FinalAssessmentConfig,
    assess_locked_candidate,
    evaluate_promotion_gates,
    main as assessment_main,
)
from minires.ingestion import fingerprint, load_records, normalize
from minires.source_identity import code_fingerprint
from minires.modeling.guarded_residual import (
    FLOAT32_MAXIMUM as GUARDED_RESIDUAL_FLOAT32_MAXIMUM,
    fit_state as fit_guarded_residual_state,
    predict as predict_guarded_residual,
    residual_features as guarded_residual_features,
    valid_numeric_state as valid_guarded_residual_state,
)
from minires.modeling.tuning import (
    CandidateFoldFit,
    DeclaredCandidate,
    LockedFit,
    SearchLimits,
    build_parser as tuning_parser,
    create_search_plan,
    develop_candidates,
    evaluate_declared_candidate,
    generate_search_plan,
    load_locked_candidate,
    main as tuning_main,
    tune_candidates,
    verify_locked_candidate_files,
    _cross_fit_assignments,
    _nonlinear_oof_stack_candidate,
    _select_ensemble_weight,
    _selected_epoch_count,
    _stack_meta_matrix,
    _source_candidate_metrics,
    _tail_selection_key,
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


class StackRecordingRuntime(RecordingTuningRuntime):
    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.refit_calls.append((candidate.candidate_id, tuple(features), fixed_training_counts))
        is_meta = candidate.candidate_id.endswith("-meta")
        return LockedFit(
            predictor=(lambda rows: [row[0] for row in rows]) if is_meta
            else (lambda rows: [row[1] / 1000.0 for row in rows]),
            preprocessing_state={"fit_rows": len(features), "meta": is_meta},
            artifacts={"model.bin": candidate.candidate_id.encode()},
            metadata={"seed": seed, "validation_used": False},
        )


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
        self.assertTrue(first.second_seed_rule["both_seed_results_must_be_eligible"])
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

    def test_tail_aware_plan_expands_the_search_and_records_its_hypothesis(self):
        limits = SearchLimits.for_plan(41, "tail_aware_expanded")

        plan = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(len(plan.component_trials), 24)
        self.assertEqual(len(plan.ensemble_rules), 6)
        self.assertEqual(plan.second_seed_rule["candidate_count"], 10)
        self.assertEqual(plan.resource_limits["maximum_candidate_runs"], 40)
        self.assertEqual(plan.generator["plan_kind"], "tail_aware_expanded")
        self.assertIn("sliced resin mass weighting", plan.generator["hypothesis"])
        for family in ("neural_network", "xgboost"):
            self.assertEqual(
                plan.parameter_domains[family]["target_weighting"],
                ("none", "sliced_resin_mass_band_1_2_3_4"),
            )
        self.assertTrue(any(
            candidate.parameters["target_weighting"]
            == "sliced_resin_mass_band_1_2_3_4"
            for candidate in plan.component_trials
        ))
        for family in ("neural_network", "xgboost"):
            configurations = [
                json.dumps(candidate.parameters, sort_keys=True)
                for candidate in plan.component_trials
                if candidate.family == family
            ]
            self.assertEqual(len(configurations), len(set(configurations)))

    def test_large_batch_extended_plan_adds_distinct_capacity_and_records_its_hypothesis(self):
        limits = SearchLimits.for_plan(41, "large_batch_extended")

        plan = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(len(plan.component_trials), 48)
        self.assertEqual(len(plan.ensemble_rules), 12)
        self.assertEqual(plan.second_seed_rule["candidate_count"], 20)
        self.assertEqual(plan.resource_limits["maximum_candidate_runs"], 80)
        self.assertEqual(plan.resource_limits["maximum_elapsed_seconds"], 14_400.0)
        self.assertEqual(plan.generator["plan_kind"], "large_batch_extended")
        self.assertIn("larger neural-network batches", plan.generator["hypothesis"])
        neural_candidates = [
            candidate for candidate in plan.component_trials
            if candidate.family == "neural_network"
        ]
        xgboost_candidates = [
            candidate for candidate in plan.component_trials
            if candidate.family == "xgboost"
        ]
        self.assertEqual(plan.parameter_domains["neural_network"]["batch_size"], (512, 1024))
        self.assertEqual(
            plan.parameter_domains["xgboost"]["n_estimators"],
            (1500, 1800, 2400),
        )
        self.assertTrue(all(
            candidate.parameters["batch_size"] >= 512 for candidate in neural_candidates
        ))
        self.assertTrue(all(
            candidate.parameters["n_estimators"] >= 1500
            for candidate in xgboost_candidates
        ))
        for family_candidates in (neural_candidates, xgboost_candidates):
            configurations = [
                json.dumps(candidate.parameters, sort_keys=True)
                for candidate in family_candidates
            ]
            self.assertEqual(len(configurations), len(set(configurations)))

    def test_tail_aligned_selection_plan_is_finite_and_changes_only_selection(self):
        limits = SearchLimits.for_plan(41, "tail_aligned_selection")

        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(first, second)
        self.assertEqual((first.seed, first.second_seed), (41, 42))
        self.assertEqual(first.generator["plan_kind"], "tail_aligned_selection")
        self.assertEqual(
            first.generator["validation_selection"],
            "serious_error_gates_then_ranking_v1",
        )
        self.assertEqual(len(first.component_trials), 12)
        self.assertEqual(len(first.ensemble_rules), 3)
        self.assertEqual(first.second_seed_rule["candidate_count"], 5)
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 20)
        self.assertEqual(first.resource_limits["maximum_elapsed_seconds"], 7200.0)
        self.assertTrue(all(
            candidate.ordered_prediction_features
            == ("kb", "volume", "surface_area", "bbox_area", "euler_number",
                "scale", "surface_volume_ratio")
            and candidate.parameters["validation_selection"]
            == "serious_error_gates_then_ranking_v1"
            for candidate in first.component_trials
        ))
        self.assertEqual(
            first.parameter_domains["neural_network"]["loss"], ("huber",),
        )
        self.assertTrue(all(
            rule["selection_rule"] == "serious_error_gates_then_ranking_v1"
            for rule in first.ensemble_rules
        ))
        serialized = json.dumps(first.to_dict(), sort_keys=True)
        self.assertNotIn("anonymous_source_group", serialized)
        self.assertNotIn("miniature_family", serialized)

    def test_tail_selection_prioritizes_gate_violation_before_mae(self):
        worse_gate = {
            "pooled_above_5g_fraction": 0.012,
            "source_balanced_above_5g_fraction": 0.016,
            "maximum_qualifying_source_above_5g_fraction": 0.05,
            "source_balanced_mae_g": 0.5,
            "pooled_mae_g": 0.5,
            "pooled_within_2g_fraction": 0.97,
        }
        better_gate = {
            **worse_gate,
            "pooled_above_5g_fraction": 0.011,
            "source_balanced_above_5g_fraction": 0.014,
            "maximum_qualifying_source_above_5g_fraction": 0.03,
            "source_balanced_mae_g": 0.8,
            "pooled_mae_g": 0.8,
        }
        self.assertLess(
            _tail_selection_key(better_gate), _tail_selection_key(worse_gate)
        )
        non_finite = {**better_gate, "pooled_mae_g": math.nan}
        self.assertLess(
            _tail_selection_key(better_gate), _tail_selection_key(non_finite)
        )
        with self.assertRaisesRegex(ValueError, "invalid_candidate_predictions"):
            _source_candidate_metrics((1.0,), (math.nan,))

        selected = _select_ensemble_weight(
            (0.0, 10.0), (0.0, 10.0), (6.0, 16.0), (0.0, 1.0),
            prediction_ranker=lambda predictions: (
                sum(abs(value - target) > 5.0
                    for value, target in zip(predictions, (0.0, 10.0))),
            ),
        )
        self.assertEqual(selected, 1.0)

    def test_geometry_regime_plan_predeclares_one_richer_source_neutral_representation(self):
        limits = SearchLimits.for_plan(41, "geometry_regime")

        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        expected_features = (
            "volume_mm3", "surface_area_mm2", "bounding_box_short_mm",
            "bounding_box_middle_mm", "bounding_box_long_mm",
            "bounding_box_volume_mm3", "euler_number", "log1p_volume_mm3",
            "log1p_surface_area_mm2", "log1p_bounding_box_volume_mm3",
            "log1p_bounding_box_short_mm", "log1p_bounding_box_middle_mm",
            "log1p_bounding_box_long_mm", "log_volume_to_bounding_box_volume_ratio",
            "log_surface_to_volume_ratio_per_mm", "log_bounding_box_long_to_short_ratio",
        )
        self.assertEqual(first, second)
        self.assertEqual(first.generator["plan_kind"], "geometry_regime")
        self.assertIn("source-neutral geometry representation", first.generator["hypothesis"])
        self.assertEqual(first.prediction_features, expected_features)
        self.assertEqual(
            first.feature_transformation_version,
            "minires-geometry-regime-features-v1",
        )
        self.assertEqual(len(first.component_trials), 24)
        self.assertEqual(len(first.ensemble_rules), 6)
        self.assertEqual(first.second_seed_rule["candidate_count"], 10)
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 40)
        self.assertEqual(first.resource_limits["maximum_elapsed_seconds"], 7200.0)
        self.assertEqual((first.seed, first.second_seed), (41, 42))
        baseline = generate_search_plan(
            SearchLimits(seed=41), input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        self.assertEqual(first.eligibility_gates, baseline.eligibility_gates)
        self.assertEqual(
            baseline.prediction_features,
            ("kb", "volume", "surface_area", "bbox_area", "euler_number", "scale",
             "surface_volume_ratio"),
        )
        self.assertTrue(all(
            candidate.ordered_prediction_features == expected_features
            for candidate in first.component_trials
        ))
        self.assertEqual(
            len({candidate.candidate_id for candidate in first.component_trials}), 24
        )

    def test_legacy_geometry_augmentation_plan_is_finite_and_binds_its_feature_contract(self):
        limits = SearchLimits.for_plan(41, "legacy_geometry_augmentation")

        plan = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        expected_features = (
            "kb", "volume", "surface_area", "bbox_area", "euler_number", "scale",
            "surface_volume_ratio", "mesh_volume_to_bounding_box_volume_ratio",
            "surface_area_to_bounding_box_volume_ratio_per_mm",
            "log1p_volume_squared", "bounding_box_long_to_short_ratio",
        )
        self.assertEqual((plan.seed, plan.second_seed), (41, 42))
        self.assertEqual(plan.generator["plan_kind"], "legacy_geometry_augmentation")
        self.assertEqual(plan.prediction_features, expected_features)
        self.assertEqual(
            plan.feature_transformation_version,
            "minires-legacy-geometry-augmentation-v1",
        )
        self.assertEqual(len(plan.component_trials), 12)
        self.assertEqual(len(plan.ensemble_rules), 3)
        self.assertEqual(plan.second_seed_rule["candidate_count"], 5)
        self.assertEqual(plan.resource_limits["maximum_candidate_runs"], 20)
        self.assertEqual(plan.resource_limits["maximum_elapsed_seconds"], 7200.0)
        self.assertTrue(all(
            candidate.ordered_prediction_features == expected_features
            and candidate.feature_transformation_version
            == "minires-legacy-geometry-augmentation-v1"
            for candidate in plan.component_trials
        ))
        serialized = json.dumps(plan.to_dict(), sort_keys=True)
        self.assertNotIn("anonymous_source_group", serialized)
        self.assertNotIn("miniature_family", serialized)

    def test_cross_fitted_geometry_gate_plan_is_finite_training_only_and_source_neutral(self):
        limits = SearchLimits.for_plan(41, "cross_fitted_geometry_gate")

        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(first, second)
        self.assertEqual((first.seed, first.second_seed), (41, 42))
        self.assertEqual(first.generator["plan_kind"], "cross_fitted_geometry_gate")
        self.assertIn("training-only out-of-fold", first.generator["hypothesis"])
        self.assertEqual(len(first.component_trials), 12)
        self.assertEqual(len(first.ensemble_rules), 3)
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 20)
        self.assertEqual(first.resource_limits["maximum_elapsed_seconds"], 7200.0)
        for rule, penalty in zip(first.ensemble_rules, (0.01, 0.1, 1.0)):
            self.assertEqual(rule["construction"], "cross_fitted_geometry_gate")
            self.assertEqual(rule["cross_fit_folds"], 5)
            self.assertEqual(rule["cross_fit_partition"], "training_records_only")
            self.assertEqual(rule["gate_fit_partition"], "training_oof_predictions_only")
            self.assertEqual(rule["validation_use"], "scoring_eligibility_ranking_and_locking_only")
            self.assertEqual(rule["ridge_penalty"], penalty)
            self.assertEqual(
                rule["gate_features"],
                (
                    "volume_mm3", "surface_area_mm2", "bounding_box_short_mm",
                    "bounding_box_middle_mm", "bounding_box_long_mm",
                    "bounding_box_volume_mm3", "euler_number", "log1p_volume_mm3",
                    "log1p_surface_area_mm2", "log1p_bounding_box_volume_mm3",
                    "log1p_bounding_box_short_mm", "log1p_bounding_box_middle_mm",
                    "log1p_bounding_box_long_mm",
                    "log_volume_to_bounding_box_volume_ratio",
                    "log_surface_to_volume_ratio_per_mm",
                    "log_bounding_box_long_to_short_ratio",
                ),
            )
            self.assertNotIn("anonymous_source_group", json.dumps(rule))
            self.assertNotIn("miniature_family", json.dumps(rule))
        self.assertTrue(first.second_seed_rule["both_seed_results_must_be_eligible"])

    def test_nonlinear_oof_stacking_plan_is_complete_finite_and_source_neutral(self):
        limits = SearchLimits.for_plan(41, "nonlinear_oof_stacking")
        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(first, second)
        self.assertEqual((first.seed, first.second_seed), (41, 42))
        self.assertEqual([item.family for item in first.component_trials],
                         ["neural_network", "neural_network", "xgboost", "xgboost"])
        self.assertEqual([
            item.parameters.get("maximum_epochs", item.parameters.get("n_estimators"))
            for item in first.component_trials
        ], [61, 87, 584, 1091])
        self.assertEqual(first.parameter_domains,
                         {"neural_network": {}, "xgboost": {}})
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 6)
        self.assertEqual(first.resource_limits["maximum_model_fits"], 54)
        self.assertEqual(first.resource_limits["oof_base_fits"], 40)
        self.assertEqual(first.resource_limits["full_training_base_fits"], 8)
        self.assertEqual(first.resource_limits["meta_fits"], 6)
        self.assertEqual(first.second_seed_rule["selection"],
                         "all_predeclared_meta_candidates")
        self.assertEqual(len(first.ensemble_rules), 3)
        self.assertEqual([rule["meta_parameters"]["max_depth"]
                          for rule in first.ensemble_rules], [1, 2, 3])
        for index, rule in enumerate(first.ensemble_rules):
            self.assertEqual(rule["stack_candidate_id"],
                             _nonlinear_oof_stack_candidate(index).candidate_id)
            self.assertEqual(rule["cross_fit_folds"], 5)
            self.assertEqual(rule["stack_fit_partition"],
                             "training_oof_predictions_only")
            self.assertEqual(rule["validation_use"],
                             "scoring_eligibility_ranking_and_locking_only")
            self.assertEqual(tuple(rule["meta_features"]), (
                "base_1_g", "base_2_g", "base_3_g", "base_4_g",
                "mean_g", "min_g", "max_g", "spread_g",
            ))
        serialized = json.dumps(first.to_dict(), sort_keys=True)
        self.assertNotIn("anonymous_source_group", serialized)
        self.assertNotIn("miniature_family", serialized)
        with self.assertRaisesRegex(ValueError, "invalid_search_plan"):
            generate_search_plan(
                SearchLimits.for_plan(99, "nonlinear_oof_stacking"),
                input_fingerprint="input", code_fingerprint="code",
                dependency_versions={"runtime": "synthetic-1"},
            )

    def test_guarded_residual_stacking_plan_is_finite_anchored_and_source_neutral(self):
        limits = SearchLimits.for_plan(41, "guarded_residual_stacking")
        first = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )
        second = generate_search_plan(
            limits, input_fingerprint="input", code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual(first, second)
        self.assertEqual((first.seed, first.second_seed), (41, 42))
        self.assertEqual([item.family for item in first.component_trials],
                         ["neural_network", "neural_network", "xgboost", "xgboost"])
        self.assertEqual(first.parameter_domains,
                         {"neural_network": {}, "xgboost": {}})
        self.assertEqual(first.resource_limits["maximum_candidate_runs"], 6)
        self.assertEqual(first.resource_limits["maximum_model_fits"], 50)
        self.assertEqual(first.resource_limits["oof_base_fits"], 40)
        self.assertEqual(first.resource_limits["full_training_base_fits"], 8)
        self.assertEqual(first.resource_limits["residual_fits"], 2)
        self.assertEqual(first.second_seed_rule["selection"],
                         "all_predeclared_residual_candidates")
        self.assertEqual(len(first.ensemble_rules), 3)
        self.assertEqual([rule["correction_scale"] for rule in first.ensemble_rules],
                         [0.0, 0.5, 1.0])
        self.assertTrue(all(
            rule["construction"] == "guarded_residual_stacking"
            and rule["anchor_formula"] == "float64_arithmetic_mean_of_four_bases"
            and rule["residual_fit_partition"] == "training_oof_predictions_only"
            and rule["cross_fit_folds"] == 5
            for rule in first.ensemble_rules
        ))
        serialized = json.dumps(first.to_dict(), sort_keys=True)
        self.assertNotIn("anonymous_source_group", serialized)
        self.assertNotIn("miniature_family", serialized)
        with self.assertRaisesRegex(ValueError, "invalid_search_plan"):
            generate_search_plan(
                SearchLimits.for_plan(99, "guarded_residual_stacking"),
                input_fingerprint="input", code_fingerprint="code",
                dependency_versions={"runtime": "synthetic-1"},
            )

    def test_stack_fold_assignment_is_deterministic_and_ignores_source_and_target(self):
        raw = [
            {**CandidateTuningTests.row("source-a", f"family-{index}", index + 1),
             "_id": f"record-{index}"}
            for index in range(20)
        ]
        rows = normalize(raw, self.config)
        changed = [dict(item, weight=item["weight"] + 1000,
                        anonymous_source_group="source-b") for item in raw]
        changed_rows = normalize(changed, self.config)

        first = _cross_fit_assignments(rows, 41, 5)
        self.assertEqual(first, _cross_fit_assignments(rows, 41, 5))
        self.assertEqual(first, _cross_fit_assignments(changed_rows, 41, 5))
        self.assertNotEqual(first, _cross_fit_assignments(rows, 42, 5))
        self.assertEqual(sorted(first), [0, 0, 0, 0, 1, 1, 1, 1,
                                         2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4])

    def test_stack_meta_matrix_is_deterministic_and_rejects_invalid_vectors(self):
        columns = ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0))
        self.assertEqual(_stack_meta_matrix(columns, 2), (
            (1.0, 2.0, 3.0, 4.0, 2.5, 1.0, 4.0, 3.0),
            (2.0, 3.0, 4.0, 5.0, 3.5, 2.0, 5.0, 3.0),
        ))
        with self.assertRaisesRegex(ValueError, "invalid_stack_prediction_matrix"):
            _stack_meta_matrix((columns[0], columns[1], columns[2], (4.0,)), 2)
        with self.assertRaisesRegex(ValueError, "invalid_stack_prediction_matrix"):
            _stack_meta_matrix((columns[0], columns[1], columns[2], (4.0, math.nan)), 2)

    def test_guarded_residual_numeric_contract_uses_cast_anchor_and_rejects_out_of_range_state(self):
        anchors, features = guarded_residual_features((
            (16_777_216.0,), (16_777_217.0,),
            (16_777_218.0,), (16_777_219.0,),
        ), 1)
        self.assertEqual(anchors, (16_777_218.0,))
        self.assertEqual(features[0][1:5], (-2.0, -1.0, 0.0, 1.0))

        state = fit_guarded_residual_state(features, (16_777_218.0,), anchors)
        self.assertTrue(valid_guarded_residual_state(state))
        invalid = {**state, "coefficients": [
            GUARDED_RESIDUAL_FLOAT32_MAXIMUM * 2,
            *state["coefficients"][1:],
        ]}
        self.assertFalse(valid_guarded_residual_state(invalid))
        with self.assertRaisesRegex(ValueError, "invalid_guarded_residual_state"):
            predict_guarded_residual(invalid, features, anchors, 1.0)

    def test_geometry_regime_rejects_seeds_outside_its_predeclared_pair(self):
        with self.assertRaisesRegex(ValueError, "invalid_search_plan"):
            generate_search_plan(
                SearchLimits.for_plan(99, "geometry_regime"),
                input_fingerprint="input", code_fingerprint="code",
                dependency_versions={"runtime": "synthetic-1"},
            )

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
            "minires.modeling.tuning.TensorflowXGBoostCandidateRuntime",
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


class ValidationSensitiveRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.fit_calls.append((candidate.candidate_id, seed, tuple(train_features),
                               tuple(validation_features)))
        shift = 5.0 if candidate.family == "xgboost" else 0.0
        return CandidateFoldFit(
            predictor=lambda rows: [row[1] / 1000.0 + shift for row in rows],
            metadata={"selected_epochs": 4, "selected_trees": 20},
            fitted_state={"training_features": tuple(train_features)},
        )


class ComplementaryGeometryRuntime(RecordingTuningRuntime):
    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.fit_calls.append((candidate.candidate_id, seed, tuple(train_features),
                               tuple(validation_features)))
        if candidate.family == "control":
            predictor = lambda rows: [row[1] / 1000.0 for row in rows]
        elif candidate.family == "neural_network":
            predictor = lambda rows: [
                row[1] / 1000.0 + (0.0 if row[1] / 1000.0 <= 10 else 6.0)
                for row in rows
            ]
        else:
            predictor = lambda rows: [
                row[1] / 1000.0 + (6.0 if row[1] / 1000.0 <= 10 else 0.0)
                for row in rows
            ]
        return CandidateFoldFit(
            predictor=predictor,
            metadata={
                "selected_epochs": 4 if candidate.family == "neural_network" else None,
                "selected_trees": 20 if candidate.family == "xgboost" else None,
            },
            fitted_state={"training_features": tuple(train_features)},
        )


class GeometryRecordingRuntime(RecordingTuningRuntime):
    def __init__(self):
        super().__init__()
        self.feature_calls = []
        self.loaded_feature_widths = []

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.feature_calls.append((candidate.family, tuple(train_features),
                                   tuple(validation_features)))
        predictor = (
            (lambda rows: [row[0] / 1000.0 for row in rows])
            if candidate.family != "control"
            else (lambda rows: [row[1] / 1000.0 for row in rows])
        )
        return CandidateFoldFit(
            predictor=predictor,
            metadata={"selected_epochs": 4, "selected_trees": 20},
            fitted_state={"training_features": tuple(train_features)},
        )

    def load_locked(self, candidate, directory, contract):
        def predict(rows):
            self.loaded_feature_widths.extend(len(row) for row in rows)
            return [row[1] / 1000.0 for row in rows]
        return predict


class AugmentedGeometryRecordingRuntime(RecordingTuningRuntime):
    def __init__(self):
        super().__init__()
        self.feature_calls = []
        self.loaded_feature_widths = []

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        self.feature_calls.append((candidate.family, tuple(train_features),
                                   tuple(validation_features)))
        return super().fit_fold(
            candidate, seed, train_features, train_targets,
            validation_features, validation_targets,
        )

    def load_locked(self, candidate, directory, contract):
        def predict(rows):
            self.loaded_feature_widths.extend(len(row) for row in rows)
            return [row[1] / 1000.0 for row in rows]
        return predict


class ExplicitPartitionDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private"
        self.config = EvaluationConfig(None, "mm3", True, seed=17)
        self.training = [self.row(f"train-{index}", index + 1) for index in range(6)]
        self.validation = [self.row(f"validation-{index}", index + 20) for index in range(4)]

    @staticmethod
    def row(identity, value):
        return {
            "_id": identity,
            "kb": value, "volume": value * 1000, "surface_area": value * 100,
            "bbox_area": value * 1100, "euler_number": value, "scale": value,
            "surface_volume_ratio": 0.1, "weight": value,
            "anonymous_source_group": "private-source", "partition": "poison",
            "duplicate_group": f"private-link-{identity}", "join_key": "private-join",
        }

    def test_training_and_validation_artifacts_are_separate_and_validation_selects(self):
        first_runtime = ValidationSensitiveRuntime()
        first = develop_candidates(
            self.training, self.validation, self.config, runtime=first_runtime,
            output_root=self.root / "first", limits=SearchLimits(seed=41),
            clock=lambda: 0.0,
        )
        changed_validation = [dict(row, weight=row["weight"] + 5) for row in self.validation]
        second_runtime = ValidationSensitiveRuntime()
        second = develop_candidates(
            self.training, changed_validation, self.config, runtime=second_runtime,
            output_root=self.root / "second", limits=SearchLimits(seed=41),
            clock=lambda: 0.0,
        )

        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "completed")
        self.assertNotEqual(
            first.locked_candidate.contract["fixed_training_counts"].get(
                "ensemble_neural_network_weight"
            ),
            second.locked_candidate.contract["fixed_training_counts"].get(
                "ensemble_neural_network_weight"
            ),
        )
        expected_training = tuple(
            (value, value * 1000, value * 100, value * 1100, value, value, 0.1)
            for value in range(1, 7)
        )
        expected_validation = tuple(
            (value, value * 1000, value * 100, value * 1100, value, value, 0.1)
            for value in range(20, 24)
        )
        for runtime in (first_runtime, second_runtime):
            self.assertTrue(runtime.fit_calls)
            self.assertTrue(all(call[2] == expected_training for call in runtime.fit_calls))
            self.assertTrue(all(call[3] == expected_validation for call in runtime.fit_calls))
            self.assertTrue(all(len(features) == 7 for call in runtime.fit_calls
                                for partition in call[2:4] for features in partition))
            self.assertEqual(len(runtime.refit_calls), 1)
            self.assertEqual(len(runtime.refit_calls[0][1]), len(self.training))
        contract = first.locked_candidate.contract
        self.assertEqual(contract["development_contract"], "explicit_train_validation")
        self.assertEqual(contract["refit_partition"], "training_records_only")
        self.assertFalse(contract["test_input_accessed"])
        self.assertEqual(
            contract["development_data_usage"]["preprocessing"], "training_records_only"
        )
        self.assertEqual(
            contract["development_data_usage"]["candidate_selection"],
            "validation_records_only",
        )
        self.assertIn("training_input_fingerprint", contract["development_evidence"])
        self.assertIn("validation_input_fingerprint", contract["development_evidence"])
        self.assertEqual(
            contract["development_evidence"]["validation_grouping_contract"],
            "source-grouped-validation-v1",
        )
        self.assertEqual(len(contract["development_source_groups"]), 1)
        self.assertNotIn("private-source", json.dumps(first.to_dict()))

    def test_unequal_validation_source_groups_remain_distinct_for_every_gate(self):
        validation = []
        value = 1_000
        for source, count in (("validation-a", 200), ("validation-b", 600),
                              ("validation-c", 600), ("validation-d", 600)):
            for _ in range(count):
                validation.append(dict(self.row(f"validation-{value}", value),
                                       anonymous_source_group=source))
                value += 1
        shifted = {row["weight"] for row in validation[:5]}

        result = develop_candidates(
            self.training, validation, self.config,
            runtime=SelectiveTailRuntime(shifted),
            output_root=self.root / "source-balanced",
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed_no_candidate")
        self.assertTrue(result.initial_results)
        for run in result.initial_results:
            self.assertEqual(run.metrics["source_count"], 4)
            self.assertAlmostEqual(run.metrics["pooled_above_5g_fraction"], 0.0025)
            self.assertAlmostEqual(
                run.metrics["source_balanced_above_5g_fraction"], 0.00625
            )
            self.assertAlmostEqual(
                run.metrics["maximum_qualifying_source_above_5g_fraction"], 0.025
            )
            self.assertFalse(run.eligible)
            self.assertEqual(len(run.source_reports), 4)
        self.assertNotIn("source_reports", result.to_dict(public=True))
        self.assertNotIn("validation-a", json.dumps(result.to_dict()))

    def test_pre_correction_explicit_lock_is_rejected_even_with_valid_checksums(self):
        runtime = ValidationSensitiveRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "current-lock", limits=SearchLimits(seed=41),
            clock=lambda: 0.0,
        )
        assert result.locked_candidate is not None
        lock = result.locked_candidate.directory
        contract_path = lock / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["development_evidence"].pop("validation_grouping_contract")
        contract_path.write_text(json.dumps(contract, sort_keys=True))
        manifest_path = lock / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["candidate-contract.json"] = sha256(
            contract_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        blockers, _, _ = verify_locked_candidate_files(
            lock, runtime.dependency_versions
        )

        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_cross_fitted_gate_uses_training_oof_evidence_and_locks_source_neutral_state(self):
        def canonical(identity, value):
            return {
                **self.row(identity, value),
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        training = [canonical(f"train-gate-{value}", value) for value in range(1, 21)]
        validation = [
            canonical(f"validation-gate-{value}", value)
            for value in (3, 8, 13, 18)
        ]
        runtime = ComplementaryGeometryRuntime()

        result = develop_candidates(
            training, validation, self.config, runtime=runtime,
            output_root=self.root / "cross-fitted-gate",
            limits=SearchLimits.for_plan(41, "cross_fitted_geometry_gate"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        gates = [run for run in result.initial_results
                 if run.candidate.family == "geometry_gate"]
        self.assertEqual(len(gates), 3)
        self.assertTrue(all(run.eligible for run in gates))
        self.assertTrue(all(
            run.fit_metadata[0]["gate_training"] == {
                "partition": "training_oof_predictions_only",
                "cross_fit_folds": 5,
                "oof_prediction_count": len(training),
                "source_metadata_used": False,
            }
            for run in gates
        ))
        self.assertTrue(any(
            0 < len(call[2]) < len(training) and 0 < len(call[3]) < len(training)
            for call in runtime.fit_calls
        ))
        self.assertIsNotNone(result.locked_candidate)
        contract = result.locked_candidate.contract
        self.assertEqual(contract["candidate"]["family"], "geometry_gate")
        self.assertEqual(
            contract["development_data_usage"]["gate_fitting"],
            "training_oof_predictions_only",
        )
        self.assertEqual(contract["runtime_configuration"]["combination"],
                         "geometry_conditioned_clipped_weight")
        fixed_counts = contract["fixed_training_counts"]
        cross_fit_refits = [
            call for call in runtime.refit_calls if len(call[1]) < len(training)
        ]
        self.assertEqual(len(cross_fit_refits), 10)
        self.assertTrue(all(
            call[2] in (
                {"neural_network_epochs": fixed_counts["neural_network_epochs"]},
                {"xgboost_trees": fixed_counts["xgboost_trees"]},
            )
            for call in cross_fit_refits
        ))
        self.assertEqual(contract["model_specification"]["model_kind"],
                         "geometry_gate")
        self.assertIn("gate-state.json", result.locked_candidate.manifest["files"])
        reloaded = load_locked_candidate(result.locked_candidate.directory, runtime)
        self.assertEqual(reloaded.candidate, result.locked_candidate.candidate)
        gate_path = result.locked_candidate.directory / "gate-state.json"
        gate_state = json.loads(gate_path.read_text())
        gate_state["version"] = "tampered-gate"
        gate_path.write_text(json.dumps(gate_state, sort_keys=True))
        manifest_path = result.locked_candidate.directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["gate-state.json"] = sha256(gate_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        blockers, _, _ = verify_locked_candidate_files(
            result.locked_candidate.directory, runtime.dependency_versions
        )
        self.assertIn("locked_candidate_contract_mismatch", blockers)
        public = json.dumps(result.to_dict(public=True), sort_keys=True)
        self.assertNotIn("private-source", public)
        self.assertNotIn("anonymous_source_group", public)

    def test_nonlinear_stack_cross_fits_refits_and_checksum_binds_the_lock(self):
        runtime = StackRecordingRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "nonlinear-stack",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 6)
        self.assertEqual(result.resource_use["model_fits"], 54)
        self.assertEqual(len(runtime.refit_calls), 54)
        self.assertTrue(result.to_dict()["second_seed_comparison"]["complete"])
        cross_fit_calls = [call for call in runtime.refit_calls
                           if len(call[1]) < len(self.training)]
        self.assertEqual(len(cross_fit_calls), 40)
        meta_calls = [call for call in runtime.refit_calls
                      if call[0].endswith("-meta")]
        self.assertEqual(len(meta_calls), 6)
        self.assertTrue(all(len(row) == 8 for call in meta_calls for row in call[1]))
        self.assertTrue(all(run.fit_metadata[0]["stack_training"] == {
            "partition": "training_oof_predictions_only",
            "oof_prediction_count_per_base": [len(self.training)] * 4,
            "source_metadata_used": False,
            "validation_labels_used": False,
        } for run in (*result.initial_results, *result.second_seed_results)))
        self.assertTrue(all(len(run.fit_metadata[0]["cross_fit_metadata"]) == 20
                            for run in result.initial_results))
        assert result.locked_candidate is not None
        contract = result.locked_candidate.contract
        self.assertEqual(contract["candidate"]["family"], "nonlinear_oof_stack")
        self.assertEqual(contract["training_count_rule"], "predeclared_fixed_counts")
        self.assertEqual(contract["development_data_usage"]["stack_fitting"],
                         "training_oof_predictions_only")
        self.assertIn("training_record_identity_fingerprint",
                      contract["development_evidence"])
        self.assertIn("validation_record_identity_fingerprint",
                      contract["development_evidence"])
        self.assertEqual(contract["oof_assignment"]["version"],
                         "stable-record-identity-sha256-round-robin-v1")
        self.assertEqual(set(contract["oof_assignment"]["fingerprints_by_seed"]),
                         {"41", "42"})
        self.assertTrue(all(
            len(value) == 64
            for value in contract["oof_assignment"]["fingerprints_by_seed"].values()
        ))
        self.assertEqual(contract["model_specification"]["model_kind"],
                         "nonlinear_oof_stack")
        self.assertEqual(len(contract["model_specification"]["members"]["bases"]), 4)
        blockers, _, _ = verify_locked_candidate_files(
            result.locked_candidate.directory, runtime.dependency_versions
        )
        self.assertEqual(blockers, ())

        contract_path = result.locked_candidate.directory / "candidate-contract.json"
        original_contract = contract_path.read_text()
        manifest_path = result.locked_candidate.directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        impossible_metrics = (
            {"source_balanced_mae_g": -1.0},
            {"source_count": contract["selected_combined_development_evidence"][
                "metrics"]["sample_count"] + 1},
            {"pooled_within_2g_fraction": 1.0,
             "pooled_above_5g_fraction": 0.01},
        )
        for mutation in impossible_metrics:
            tampered_contract = json.loads(original_contract)
            tampered_contract["selected_combined_development_evidence"][
                "metrics"
            ].update(mutation)
            contract_path.write_text(json.dumps(tampered_contract, sort_keys=True))
            manifest["files"]["candidate-contract.json"] = sha256(
                contract_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            blockers, _, _ = verify_locked_candidate_files(
                result.locked_candidate.directory, runtime.dependency_versions
            )
            self.assertIn("locked_candidate_contract_mismatch", blockers)
        contract_path.write_text(original_contract)
        manifest["files"]["candidate-contract.json"] = sha256(
            contract_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        state_path = result.locked_candidate.directory / "stack-state.json"
        state = json.loads(state_path.read_text())
        state["version"] = "tampered-stack"
        state_path.write_text(json.dumps(state, sort_keys=True))
        manifest_path = result.locked_candidate.directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["stack-state.json"] = sha256(state_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        blockers, _, _ = verify_locked_candidate_files(
            result.locked_candidate.directory, runtime.dependency_versions
        )
        self.assertIn("locked_candidate_contract_mismatch", blockers)

        artifact = next(path for path in result.locked_candidate.directory.iterdir()
                        if path.name.startswith("meta-") and path.is_file())
        artifact.write_bytes(b"tampered")
        blockers, _, _ = verify_locked_candidate_files(
            result.locked_candidate.directory, runtime.dependency_versions
        )
        self.assertIn("locked_candidate_checksum_mismatch", blockers)

    def test_guarded_residual_stack_cross_fits_bounds_corrections_and_locks_shift_evidence(self):
        runtime = StackRecordingRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "guarded-residual-stack",
            limits=SearchLimits.for_plan(41, "guarded_residual_stacking"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 6)
        self.assertEqual(result.resource_use["model_fits"], 50)
        self.assertEqual(len(runtime.refit_calls), 48)
        self.assertTrue(result.to_dict()["second_seed_comparison"]["complete"])
        self.assertTrue(all(
            run.fit_metadata[0]["residual_training"] == {
                "partition": "training_oof_predictions_only",
                "oof_prediction_count_per_base": [len(self.training)] * 4,
                "source_metadata_used": False,
                "validation_labels_used": False,
            }
            for run in (*result.initial_results, *result.second_seed_results)
        ))
        self.assertEqual(
            [run.fit_metadata[0]["correction_scale"]
             for run in result.initial_results],
            [0.0, 0.5, 1.0],
        )
        self.assertTrue(all(
            run.fit_metadata[0]["maximum_absolute_correction_g"]
            <= run.fit_metadata[0]["correction_bound_g"]
            for run in (*result.initial_results, *result.second_seed_results)
        ))
        assert result.locked_candidate is not None
        contract = result.locked_candidate.contract
        self.assertEqual(contract["candidate"]["family"], "guarded_residual_stack")
        self.assertEqual(contract["development_data_usage"]["residual_fitting"],
                         "training_oof_predictions_only")
        self.assertEqual(contract["runtime_configuration"]["combination"],
                         "fixed_mean_anchor_plus_bounded_training_oof_residual")
        self.assertEqual(contract["model_specification"]["model_kind"],
                         "guarded_residual_stack")
        self.assertIn("oof_full_fit_shift", contract["development_evidence"])
        self.assertIn("guarded-residual-state.json",
                      result.locked_candidate.manifest["files"])
        blockers, _, _ = verify_locked_candidate_files(
            result.locked_candidate.directory, runtime.dependency_versions
        )
        self.assertEqual(blockers, ())
        reloaded = load_locked_candidate(result.locked_candidate.directory, runtime)
        self.assertEqual(reloaded.candidate, result.locked_candidate.candidate)

        run_manifest = json.loads(
            (self.root / "guarded-residual-stack" / "manifest.json").read_text()
        )
        self.assertTrue(run_manifest["create_only"])
        self.assertFalse(run_manifest["publication_performed"])
        self.assertIn("locked-candidate/lock-manifest.json", run_manifest["artifacts"])

        directory = result.locked_candidate.directory
        state_path = directory / "guarded-residual-state.json"
        state = json.loads(state_path.read_text())
        state["correction_scale"] = 9.0
        state_path.write_text(json.dumps(state, sort_keys=True))
        manifest_path = directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][state_path.name] = sha256(state_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        blockers, _, _ = verify_locked_candidate_files(
            directory, runtime.dependency_versions
        )
        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_nonlinear_stack_rejects_coordinated_oof_fingerprint_tampering(self):
        runtime = StackRecordingRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "nonlinear-stack-oof-tamper",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )
        assert result.locked_candidate is not None
        directory = result.locked_candidate.directory
        fake = {"41": "0" * 64, "42": "1" * 64}
        contract_path = directory / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["oof_assignment"]["fingerprints_by_seed"] = fake
        contract_path.write_text(json.dumps(contract, sort_keys=True))
        state_path = directory / "stack-state.json"
        state = json.loads(state_path.read_text())
        state["oof_assignment_fingerprints"] = fake
        state_path.write_text(json.dumps(state, sort_keys=True))
        preprocessing_path = directory / "preprocessing-state.json"
        preprocessing = json.loads(preprocessing_path.read_text())
        preprocessing["stack_state"]["oof_assignment_fingerprints"] = fake
        preprocessing_path.write_text(json.dumps(preprocessing, sort_keys=True))
        manifest_path = directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for path in (contract_path, state_path, preprocessing_path):
            manifest["files"][path.name] = sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        blockers, _, _ = verify_locked_candidate_files(
            directory, runtime.dependency_versions
        )
        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_nonlinear_stack_rejects_tampered_contract_and_nonfinite_oof(self):
        candidate = _nonlinear_oof_stack_candidate(0)
        tampered = replace(candidate, parameters={**candidate.parameters,
                                                  "cross_fit_folds": 4})
        from minires.modeling.tuning import _validate_nonlinear_oof_stack_candidate
        with self.assertRaisesRegex(ValueError, "invalid_candidate_configuration"):
            _validate_nonlinear_oof_stack_candidate(tampered)
        with self.assertRaisesRegex(ValueError, "invalid_candidate_configuration"):
            _validate_nonlinear_oof_stack_candidate(
                replace(candidate, candidate_id="forged-stack-id")
            )

        class NaNStackRuntime(StackRecordingRuntime):
            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                fitted = super().refit(candidate, seed, features, targets,
                                       fixed_training_counts)
                return replace(fitted, predictor=lambda rows: [math.nan] * len(rows))

        result = develop_candidates(
            self.training, self.validation, self.config, runtime=NaNStackRuntime(),
            output_root=self.root / "nonlinear-stack-nan",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("candidate_runtime_failed",))
        self.assertIsNone(result.locked_candidate)

    def test_nonlinear_stack_preserves_completed_and_failed_candidate_outcomes(self):
        class SecondMetaFailingRuntime(StackRecordingRuntime):
            def __init__(self):
                super().__init__()
                self.meta_calls = 0

            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                if candidate.candidate_id.endswith("-meta"):
                    self.meta_calls += 1
                    if self.meta_calls == 2:
                        raise RuntimeError("private meta failure")
                return super().refit(
                    candidate, seed, features, targets, fixed_training_counts
                )

        result = develop_candidates(
            self.training, self.validation, self.config,
            runtime=SecondMetaFailingRuntime(),
            output_root=self.root / "nonlinear-stack-partial",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("candidate_runtime_failed",))
        self.assertEqual(result.run_count, 2)
        self.assertEqual([run.status for run in result.initial_results],
                         ["completed", "failed"])
        report = result.to_dict()
        self.assertFalse(report["second_seed_comparison"]["complete"])
        self.assertEqual(len(report["candidate_history"]), 6)
        self.assertEqual([item["status"] for item in report["candidate_history"][:2]],
                         ["completed", "failed"])
        self.assertEqual(
            [item["candidate_id"] for item in report["skipped_candidates"]
             if item["phase"] == "initial"],
            [_nonlinear_oof_stack_candidate(2).candidate_id],
        )
        self.assertTrue(all(item["status"] == "skipped"
                            for item in report["candidate_history"][2:]))
        self.assertNotIn("stack-base", json.dumps(report["skipped_candidates"]))
        self.assertIsNone(result.locked_candidate)

    def test_nonlinear_stack_seed_two_interruption_keeps_canonical_skipped_id(self):
        class SeedTwoMetaFailingRuntime(StackRecordingRuntime):
            def __init__(self):
                super().__init__()
                self.meta_calls = 0

            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                if candidate.candidate_id.endswith("-meta"):
                    self.meta_calls += 1
                    if self.meta_calls == 5:
                        raise RuntimeError("private second-seed meta failure")
                return super().refit(
                    candidate, seed, features, targets, fixed_training_counts
                )

        result = develop_candidates(
            self.training, self.validation, self.config,
            runtime=SeedTwoMetaFailingRuntime(),
            output_root=self.root / "nonlinear-stack-seed-two-partial",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.run_count, 5)
        self.assertEqual([run.status for run in result.second_seed_results],
                         ["completed", "failed"])
        skipped = result.to_dict()["skipped_candidates"]
        self.assertEqual(skipped, [{
            "candidate_id": _nonlinear_oof_stack_candidate(2).candidate_id,
            "phase": "second_seed",
            "reason": "candidate_runtime_failed",
        }])

    def test_nonlinear_stack_repeats_all_slots_when_only_one_is_eligible(self):
        class OneEligibleStackRuntime(StackRecordingRuntime):
            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                fitted = super().refit(
                    candidate, seed, features, targets, fixed_training_counts
                )
                if candidate.candidate_id.endswith("-meta") and not candidate.candidate_id.startswith("stack-01-"):
                    return replace(fitted, predictor=lambda rows: [row[0] + 6.0 for row in rows])
                return fitted

        result = develop_candidates(
            self.training, self.validation, self.config,
            runtime=OneEligibleStackRuntime(),
            output_root=self.root / "nonlinear-stack-one-eligible",
            limits=SearchLimits.for_plan(41, "nonlinear_oof_stacking"),
            clock=lambda: 0.0,
        )

        comparison = result.to_dict()["second_seed_comparison"]
        self.assertEqual(comparison["eligible_initial_candidates"], 1)
        self.assertEqual(comparison["repeated_candidates"], 3)
        self.assertEqual(comparison["shortfall"], 0)
        self.assertTrue(comparison["complete"])
        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.locked_candidate)

    def test_legacy_geometry_augmentation_appends_four_canonical_terms(self):
        def canonical(row):
            value = float(row["weight"])
            return {
                **row,
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        runtime = AugmentedGeometryRecordingRuntime()
        result = develop_candidates(
            [canonical(row) for row in self.training],
            [canonical(row) for row in self.validation],
            self.config, runtime=runtime, output_root=self.root / "augmented-geometry",
            limits=SearchLimits.for_plan(41, "legacy_geometry_augmentation"),
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        control_calls = [call for call in runtime.feature_calls if call[0] == "control"]
        candidate_calls = [call for call in runtime.feature_calls if call[0] != "control"]
        self.assertTrue(control_calls)
        self.assertTrue(candidate_calls)
        self.assertTrue(all(len(row) == 7 for call in control_calls for rows in call[1:]
                            for row in rows))
        self.assertTrue(all(len(row) == 11 for call in candidate_calls for rows in call[1:]
                            for row in rows))
        first = candidate_calls[0][1][0]
        self.assertEqual(first[:7], (1.0, 1000.0, 100.0, 1100.0, 1.0, 1.0, 0.1))
        self.assertAlmostEqual(first[7], 1000.0 / 1100.0)
        self.assertAlmostEqual(first[8], 100.0 / 1100.0)
        self.assertAlmostEqual(first[9], math.log1p(1000.0) ** 2)
        self.assertAlmostEqual(first[10], 4.0 / 2.0)
        self.assertEqual(
            result.locked_candidate.contract["feature_contract"]["ordered_features"],
            list(result.plan.prediction_features),
        )
        reloaded = load_locked_candidate(result.locked_candidate.directory, runtime)
        final = []
        for source_index in range(3):
            for index in range(200):
                value = source_index * 200 + index + 1
                row = canonical(CandidateTuningTests.row(
                    f"unseen-{source_index}", f"family-{source_index}-{index}", value
                ))
                final.append({**row, "slicing_conditions": {"layer_height_mm": 0.05}})
        legacy_predictor = lambda rows: [row[1] / 1000.0 for row in rows]
        legacy = LegacyReference.from_predictors(
            neural_network=legacy_predictor, xgboost=legacy_predictor,
            neural_network_weight=0.2, provenance=LegacyProvenance.unknown(),
        )

        assessment = assess_locked_candidate(
            final, EvaluationConfig(None, "mm3", True), reloaded, legacy,
            output_root=self.root / "augmented-geometry-assessment", runtime=runtime,
        )

        self.assertNotEqual(assessment.status, "blocked")
        self.assertEqual(set(runtime.loaded_feature_widths), {11})

    def test_legacy_geometry_augmentation_blocks_invalid_geometry_before_fitting(self):
        def canonical(row):
            value = float(row["weight"])
            return {
                **row,
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        base_training = [canonical(row) for row in self.training]
        base_validation = [canonical(row) for row in self.validation]
        invalid_cases = {
            "missing": lambda row: (
                row.pop("bounding_box_z_mm"), row.pop("bbox_z")
            ),
            "not-float32-volume": lambda row: row.update(
                {"bounding_box_volume_mm3": 1e100, "bbox_area": 1e100}
            ),
            "not-float32-dimensions": lambda row: row.update({
                "bounding_box_x_mm": 1e100, "bbox_x": 1e100,
                "bounding_box_y_mm": 1e100, "bbox_y": 1e100,
                "bounding_box_z_mm": 1e100, "bbox_z": 1e100,
            }),
        }
        for name, mutate in invalid_cases.items():
            with self.subTest(name=name):
                training = [dict(row) for row in base_training]
                validation = [dict(row) for row in base_validation]
                mutate(validation[0])
                runtime = AugmentedGeometryRecordingRuntime()

                result = develop_candidates(
                    training, validation, self.config, runtime=runtime,
                    output_root=self.root / f"augmented-{name}",
                    limits=SearchLimits.for_plan(41, "legacy_geometry_augmentation"),
                )

                self.assertEqual(result.status, "blocked")
                self.assertEqual(result.blockers, ("invalid_candidate_feature_data",))
                self.assertEqual(runtime.feature_calls, [])
                self.assertEqual(runtime.refit_calls, [])

    def test_legacy_geometry_augmentation_lock_rejects_feature_contract_tampering(self):
        def canonical(row):
            value = float(row["weight"])
            return {
                **row,
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        runtime = AugmentedGeometryRecordingRuntime()
        result = develop_candidates(
            [canonical(row) for row in self.training],
            [canonical(row) for row in self.validation], self.config,
            runtime=runtime, output_root=self.root / "augmented-tampering",
            limits=SearchLimits.for_plan(41, "legacy_geometry_augmentation"),
            clock=lambda: 0.0,
        )
        assert result.locked_candidate is not None
        lock = result.locked_candidate.directory
        contract_path = lock / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        changed_version = "changed-formulas"
        contract["candidate"]["feature_transformation_version"] = changed_version
        for member in ("neural_network", "xgboost"):
            if member in contract["candidate"]["parameters"]:
                contract["candidate"]["parameters"][member][
                    "feature_transformation_version"
                ] = changed_version
        contract["feature_contract"]["transformation_version"] = changed_version

        def refresh_specification(specification):
            if specification["model_kind"] == "ensemble":
                refresh_specification(specification["ensemble"]["neural_network"])
                refresh_specification(specification["ensemble"]["xgboost"])
            else:
                specification["version"] = (
                    f"minires-model-definition-v1:{changed_version}"
                )
            payload = {
                key: value for key, value in specification.items()
                if key != "stable_identity"
            }
            digest = sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()[:16]
            specification["stable_identity"] = (
                f"model-{specification['model_kind']}-{digest}"
            )

        refresh_specification(contract["model_specification"])
        contract_path.write_text(json.dumps(contract, sort_keys=True))
        manifest_path = lock / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["candidate-contract.json"] = sha256(
            contract_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        blockers, _, _ = verify_locked_candidate_files(
            lock, runtime.dependency_versions
        )

        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_geometry_regime_uses_only_richer_canonical_geometry_for_candidates(self):
        def canonical(row):
            value = float(row["weight"])
            return {
                **row,
                "kb": 900_000 + value,
                "scale": 800_000 + value,
                "surface_volume_ratio": 700_000 + value,
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        runtime = GeometryRecordingRuntime()
        result = develop_candidates(
            [canonical(row) for row in self.training],
            [canonical(row) for row in self.validation],
            self.config, runtime=runtime, output_root=self.root / "geometry",
            limits=SearchLimits.for_plan(41, "geometry_regime"), clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        control_calls = [call for call in runtime.feature_calls if call[0] == "control"]
        candidate_calls = [call for call in runtime.feature_calls if call[0] != "control"]
        self.assertTrue(control_calls)
        self.assertTrue(candidate_calls)
        self.assertTrue(all(len(row) == 7 for call in control_calls for rows in call[1:]
                            for row in rows))
        self.assertTrue(all(len(row) == 16 for call in candidate_calls for rows in call[1:]
                            for row in rows))
        first = candidate_calls[0][1][0]
        self.assertEqual(first[:7], (1000.0, 100.0, 2.0, 3.0, 4.0, 1100.0, 1.0))
        self.assertAlmostEqual(first[7], math.log1p(1000.0))
        self.assertAlmostEqual(first[13], math.log(1000.0 / 1100.0))
        self.assertAlmostEqual(first[14], math.log(100.0 / 1000.0))
        self.assertAlmostEqual(first[15], math.log(4.0 / 2.0))
        self.assertTrue(all(value < 700_000 for value in first))
        self.assertEqual(
            result.locked_candidate.contract["feature_contract"]["ordered_features"],
            list(result.plan.prediction_features),
        )
        reloaded = load_locked_candidate(result.locked_candidate.directory, runtime)
        self.assertEqual(reloaded.candidate, result.locked_candidate.candidate)
        final = []
        for source_index in range(3):
            for index in range(200):
                value = source_index * 200 + index + 1
                row = canonical(CandidateTuningTests.row(
                    f"unseen-{source_index}", f"family-{source_index}-{index}", value
                ))
                final.append({**row, "slicing_conditions": {"layer_height_mm": 0.05}})
        legacy_predictor = lambda rows: [row[1] / 1000.0 for row in rows]
        legacy = LegacyReference.from_predictors(
            neural_network=legacy_predictor, xgboost=legacy_predictor,
            neural_network_weight=0.2, provenance=LegacyProvenance.unknown(),
        )
        assessment = assess_locked_candidate(
            final, EvaluationConfig(None, "mm3", True), reloaded, legacy,
            output_root=self.root / "geometry-assessment", runtime=runtime,
        )
        self.assertNotEqual(assessment.status, "blocked")
        self.assertEqual(set(runtime.loaded_feature_widths), {16})
        serialized = json.dumps(result.to_dict(public=True), sort_keys=True)
        self.assertNotIn("private-source", serialized)
        self.assertNotIn("private-join", serialized)

    def test_geometry_regime_lock_rejects_feature_transformation_version_tampering(self):
        def canonical(row):
            value = float(row["weight"])
            return {
                **row,
                "volume_mm3": value * 1000,
                "surface_area_mm2": value * 100,
                "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                "bounding_box_volume_mm3": value * 1100,
            }

        runtime = GeometryRecordingRuntime()
        result = develop_candidates(
            [canonical(row) for row in self.training],
            [canonical(row) for row in self.validation],
            self.config, runtime=runtime, output_root=self.root / "geometry-version",
            limits=SearchLimits.for_plan(41, "geometry_regime"), clock=lambda: 0.0,
        )
        assert result.locked_candidate is not None
        lock = result.locked_candidate.directory
        contract_path = lock / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["feature_contract"]["transformation_version"] = "changed-formulas"
        contract_path.write_text(json.dumps(contract, sort_keys=True))
        manifest_path = lock / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["candidate-contract.json"] = sha256(
            contract_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

        blockers, _, _ = verify_locked_candidate_files(
            lock, runtime.dependency_versions
        )

        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_geometry_regime_missing_measurement_blocks_before_fitting(self):
        training = [dict(row) for row in self.training]
        validation = [dict(row) for row in self.validation]
        for rows in (training, validation):
            for row in rows:
                value = float(row["weight"])
                row.update({
                    "volume_mm3": value * 1000,
                    "surface_area_mm2": value * 100,
                    "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                    "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                    "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                    "bounding_box_volume_mm3": value * 1100,
                })
        validation[0].pop("bounding_box_z_mm")
        validation[0].pop("bbox_z")
        runtime = GeometryRecordingRuntime()

        result = develop_candidates(
            training, validation, self.config, runtime=runtime,
            output_root=self.root / "geometry-missing",
            limits=SearchLimits.for_plan(41, "geometry_regime"),
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("invalid_candidate_feature_data",))
        self.assertEqual(runtime.feature_calls, [])
        self.assertEqual(runtime.refit_calls, [])

    def test_geometry_regime_blocks_values_not_representable_as_float32(self):
        training = [dict(row) for row in self.training]
        validation = [dict(row) for row in self.validation]
        for rows in (training, validation):
            for row in rows:
                value = float(row["weight"])
                row.update({
                    "volume_mm3": value * 1000,
                    "surface_area_mm2": value * 100,
                    "bounding_box_x_mm": value * 2, "bbox_x": value * 2,
                    "bounding_box_y_mm": value * 4, "bbox_y": value * 4,
                    "bounding_box_z_mm": value * 3, "bbox_z": value * 3,
                    "bounding_box_volume_mm3": value * 1100,
                })
        validation[0]["volume_mm3"] = 1e100
        validation[0]["volume"] = 1e100
        runtime = GeometryRecordingRuntime()

        result = develop_candidates(
            training, validation, self.config, runtime=runtime,
            output_root=self.root / "geometry-float32",
            limits=SearchLimits.for_plan(41, "geometry_regime"),
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("invalid_candidate_feature_data",))
        self.assertEqual(runtime.feature_calls, [])

    def test_invalid_or_overlapping_partition_identities_block_before_fitting(self):
        invalid_cases = {
            "overlap": [dict(self.validation[0], _id=self.training[0]["_id"])],
            "missing-source": [
                {key: value for key, value in self.validation[0].items()
                 if key != "anonymous_source_group"}
            ],
        }
        for name, validation in invalid_cases.items():
            with self.subTest(name=name):
                runtime = ValidationSensitiveRuntime()
                result = develop_candidates(
                    self.training, validation, self.config, runtime=runtime,
                    output_root=self.root / name, limits=SearchLimits(seed=41),
                )

                self.assertEqual(result.status, "blocked")
                self.assertEqual(
                    result.blockers, ("invalid_or_overlapping_partition_identities",)
                )
                self.assertEqual(runtime.fit_calls, [])
                self.assertEqual(runtime.refit_calls, [])


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
            self.assertEqual(combined["seed_eligibility"], [True, True])
            self.assertEqual(combined["equal_seed_weight"], 0.5)
            self.assertAlmostEqual(combined["metrics"]["pooled_mae_g"], 1.0)

    def test_final_candidate_must_pass_the_fixed_gates_under_both_seeds(self):
        records = [
            self.row(f"source-{source}", f"family-{source}-{row}", source * 200 + row)
            for source in range(3) for row in range(200)
        ]

        result = tune_candidates(
            records, self.config, runtime=SeedTailRuntime(), output_root=self.root,
            limits=SearchLimits(seed=41), clock=lambda: 0.0,
        )

        self.assertTrue(all(not run.eligible for run in result.second_seed_results))
        self.assertTrue(all(not item["eligible"] for item in result.combined_results))
        self.assertTrue(all(item["seed_eligibility"] == [True, False]
                            for item in result.combined_results))
        self.assertTrue(all(
            0.0 < item["metrics"]["pooled_above_5g_fraction"] <= 0.01
            for item in result.combined_results
        ))
        self.assertEqual(result.status, "completed_no_candidate")
        self.assertIsNone(result.locked_candidate)

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

    def test_cli_creates_private_result_from_explicit_partitions_without_a_test_argument(self):
        training = Path(self.temp.name) / "training.json"
        validation = Path(self.temp.name) / "validation.json"
        training.write_text(json.dumps([
            dict(row, _id=f"training-{index}") for index, row in enumerate(self.records[:4])
        ]))
        validation.write_text(json.dumps([
            dict(row, _id=f"validation-{index}") for index, row in enumerate(self.records[4:])
        ]))
        output = io.StringIO()
        with patch(
            "minires.modeling.tuning.TensorflowXGBoostCandidateRuntime",
            return_value=RecordingTuningRuntime(),
        ), contextlib.redirect_stdout(output):
            code = tuning_main([
                "--training-records", str(training),
                "--validation-records", str(validation),
                "--output-root", str(self.root),
                "--volume-unit", "mm3", "--scope-confirmed", "--seed", "41",
            ])

        status = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["run_count"], 20)
        self.assertNotIn(str(training), output.getvalue())
        self.assertNotIn(str(validation), output.getvalue())
        self.assertNotIn("source_reports", output.getvalue())
        self.assertTrue((self.root / "tuning-result.json").exists())
        help_text = tuning_parser().format_help()
        self.assertIn("--training-records", help_text)
        self.assertIn("--validation-records", help_text)
        self.assertNotIn("--test-records", help_text)

    def test_cli_runs_each_predeclared_expanded_plan(self):
        cases = (
            ("tail_aware_expanded", 40),
            ("large_batch_extended", 80),
            ("geometry_regime", 40),
        )
        for plan_kind, maximum_runs in cases:
            with self.subTest(plan_kind=plan_kind):
                training = Path(self.temp.name) / f"training-{plan_kind}.json"
                validation = Path(self.temp.name) / f"validation-{plan_kind}.json"
                records = self.records
                runtime = RecordingTuningRuntime()
                if plan_kind == "geometry_regime":
                    records = [
                        {
                            **row,
                            "bbox_x": row["weight"] * 2,
                            "bbox_y": row["weight"] * 4,
                            "bbox_z": row["weight"] * 3,
                        }
                        for row in records
                    ]
                    runtime = GeometryRecordingRuntime()
                training.write_text(json.dumps([
                    dict(row, _id=f"training-{index}")
                    for index, row in enumerate(records[:4])
                ]))
                validation.write_text(json.dumps([
                    dict(row, _id=f"validation-{index}")
                    for index, row in enumerate(records[4:])
                ]))
                output_root = self.root.parent / plan_kind
                output = io.StringIO()

                with patch(
                    "minires.modeling.tuning.TensorflowXGBoostCandidateRuntime",
                    return_value=runtime,
                ), contextlib.redirect_stdout(output):
                    code = tuning_main([
                        "--training-records", str(training),
                        "--validation-records", str(validation),
                        "--output-root", str(output_root),
                        "--volume-unit", "mm3", "--scope-confirmed", "--seed", "41",
                        "--plan-kind", plan_kind,
                    ])

                status = json.loads(output.getvalue())
                plan = json.loads((output_root / "search-plan.json").read_text())
                self.assertEqual(code, 0)
                self.assertEqual(status["run_count"], maximum_runs)
                self.assertEqual(plan["generator"]["plan_kind"], plan_kind)
                self.assertEqual(
                    plan["resource_limits"]["maximum_candidate_runs"], maximum_runs
                )
                if plan_kind == "geometry_regime":
                    self.assertEqual(len(plan["prediction_features"]), 16)
                    self.assertNotIn("anonymous_source_group", plan["prediction_features"])

    def test_predeclared_geometry_cli_rejects_unverified_development_artifacts(self):
        dataset = Path(self.temp.name) / "dataset"
        dataset.mkdir()
        training = dataset / "train.jsonl"
        validation = dataset / "validation.jsonl"
        training.write_text("{}\n")
        validation.write_text("{}\n")
        (dataset / "manifest.json").write_text(json.dumps({
            "artifacts": {
                "train.jsonl": "0" * 64,
                "validation.jsonl": sha256(validation.read_bytes()).hexdigest(),
            },
        }))

        for plan_kind in (
            "cross_fitted_geometry_gate", "legacy_geometry_augmentation",
        ):
            with self.subTest(plan_kind=plan_kind):
                runtime = RecordingTuningRuntime()
                output_root = self.root.parent / plan_kind
                with patch(
                    "minires.modeling.tuning.TensorflowXGBoostCandidateRuntime",
                    return_value=runtime,
                ), self.assertRaisesRegex(
                    SystemExit, "development_artifact_checksum_mismatch"
                ):
                    tuning_main([
                        "--training-records", str(training),
                        "--validation-records", str(validation),
                        "--output-root", str(output_root),
                        "--volume-unit", "mm3", "--scope-confirmed", "--seed", "41",
                        "--plan-kind", plan_kind,
                    ])

                self.assertEqual(runtime.fit_calls, [])
                self.assertFalse(output_root.exists())

    def test_guarded_residual_cli_rejects_self_consistent_alternate_artifacts(self):
        dataset = Path(self.temp.name) / "alternate-dataset"
        dataset.mkdir()
        training = dataset / "train.jsonl"
        validation = dataset / "validation.jsonl"
        training.write_text("{}\n")
        validation.write_text("{}\n")
        (dataset / "manifest.json").write_text(json.dumps({
            "artifacts": {
                "train.jsonl": sha256(training.read_bytes()).hexdigest(),
                "validation.jsonl": sha256(validation.read_bytes()).hexdigest(),
            },
        }))
        runtime = RecordingTuningRuntime()
        output_root = self.root.parent / "guarded-alternate"

        with patch(
            "minires.modeling.tuning.TensorflowXGBoostCandidateRuntime",
            return_value=runtime,
        ), self.assertRaisesRegex(
            SystemExit, "development_artifact_checksum_mismatch"
        ):
            tuning_main([
                "--training-records", str(training),
                "--validation-records", str(validation),
                "--output-root", str(output_root),
                "--volume-unit", "mm3", "--scope-confirmed", "--seed", "41",
                "--plan-kind", "guarded_residual_stacking",
            ])

        self.assertEqual(runtime.fit_calls, [])
        self.assertFalse(output_root.exists())

    def test_invalid_unbounded_worker_plan_is_rejected_before_runtime_fitting(self):
        runtime = RecordingTuningRuntime()
        loaded, input_identity = load_records(self.records)
        del loaded
        code_identity = code_fingerprint()
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
            "minires.evaluation.assessment.TensorflowXGBoostCandidateRuntime",
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

        with patch("minires.evaluation.assessment.load_records") as final_loader:
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
