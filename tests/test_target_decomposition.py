from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import unittest

from minires.evaluation import EvaluationConfig
from minires.ingestion import CanonicalRow, fingerprint
from minires.modeling import target_decomposition, target_decomposition_search
from minires.modeling.tuning import (
    LockedFit, SearchLimits, develop_candidates, generate_search_plan,
    verify_locked_candidate_files,
)


class DecompositionRuntime:
    dependency_versions = {"runtime": "synthetic-1"}
    startup_blockers = ()

    def __init__(self, *, improve=True):
        self.refit_calls = []
        self.improve = improve

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.refit_calls.append((candidate.candidate_id, seed, tuple(targets)))
        decomposed = candidate.parameters.get("target_representation") == (
            target_decomposition.TARGET_UNIT
        )
        predictor = (
            (lambda rows: [1.0 / 1.21 + (0.0 if self.improve else 8.0) for _ in rows])
            if decomposed
            else (lambda rows: [row[1] / 1000.0 + (8.0 if self.improve else 0.0)
                                for row in rows])
        )
        preprocessing = (
            {"mean": [0.0] * 7, "variance": [1.0] * 7}
            if candidate.family == "neural_network"
            else {"xgboost": "unnormalized_float32"}
        )
        return LockedFit(
            predictor, preprocessing, {"model.bin": candidate.candidate_id.encode()},
            {
                "seed": seed,
                "validation_used": False,
                ("selected_epochs" if candidate.family == "neural_network"
                 else "selected_trees"): (
                    87 if candidate.family == "neural_network" else 1091
                ),
                "fitted_parameter_fingerprint": candidate.candidate_id.ljust(64, "0")[:64],
            },
        )


class TargetDecompositionTests(unittest.TestCase):
    @staticmethod
    def row(*, mass_g=2.2, bounding_box_volume_mm3=4000.0, density_g_per_ml=1.1):
        return CanonicalRow(
            row_index=0,
            features={
                "volume_mm3": 1000.0,
                "surface_area_mm2": 100.0,
                "bounding_box_x_mm": 10.0,
                "bounding_box_y_mm": 20.0,
                "bounding_box_z_mm": 20.0,
                "bounding_box_volume_mm3": bounding_box_volume_mm3,
                "euler_number": 2.0,
            },
            sliced_resin_mass_g=mass_g,
            outcome="included",
            reasons=(),
            warnings=(),
            metadata={
                "resin_density_g_per_ml": density_g_per_ml,
                "legacy_kb_unit_unknown": 1.0,
                "legacy_scale_unit_unknown": 1.0,
                "legacy_surface_volume_ratio": 0.1,
            },
        )

    def test_training_matrix_factors_out_declared_bounding_box_mass_scale(self):
        features, targets = target_decomposition.occupancy_training_matrix((self.row(),))

        self.assertEqual(len(features), 1)
        self.assertAlmostEqual(targets[0], 0.5)

    def test_reconstruction_returns_grams_from_unbounded_factor(self):
        features, _ = target_decomposition.occupancy_training_matrix((self.row(),))

        predictions = target_decomposition.reconstruct_mass_predictions(
            features * 2, (0.5, 1.25)
        )

        self.assertEqual(predictions, (2.2, 5.5))

    def test_zero_target_is_valid_and_factor_is_not_bounded_to_one(self):
        _, targets = target_decomposition.occupancy_training_matrix((
            self.row(mass_g=0.0), self.row(mass_g=8.8),
        ))

        self.assertEqual(targets, (0.0, 2.0))

    def test_decomposed_bases_accept_the_runtime_per_kind_fixed_count_contract(self):
        decomposed_neural, decomposed_xgboost = (
            target_decomposition_search.base_candidates()[2:]
        )

        neural = target_decomposition_search.base_specification(
            decomposed_neural, {"neural_network_epochs": 87}
        )
        xgboost = target_decomposition_search.base_specification(
            decomposed_xgboost, {"xgboost_trees": 1091}
        )

        self.assertEqual(neural.output_unit, target_decomposition.TARGET_UNIT)
        self.assertEqual(xgboost.output_unit, target_decomposition.TARGET_UNIT)
        with self.assertRaisesRegex(ValueError, "invalid_locked_candidate_training_counts"):
            target_decomposition_search.base_specification(
                decomposed_neural, target_decomposition_search.FIXED_COUNTS
            )

    def test_plan_is_fixed_finite_and_source_neutral(self):
        plan = generate_search_plan(
            SearchLimits.for_plan(41, "bounding_box_target_decomposition"),
            input_fingerprint="input",
            code_fingerprint="code",
            dependency_versions={"runtime": "synthetic-1"},
        )

        self.assertEqual((plan.seed, plan.second_seed), (41, 42))
        self.assertEqual(len(plan.component_trials), 4)
        self.assertEqual(len(plan.ensemble_rules), 2)
        self.assertEqual(
            [rule["scored_kind"] for rule in plan.ensemble_rules],
            ["decomposed_ensemble", "closest_anchor_control"],
        )
        self.assertEqual(plan.resource_limits["maximum_model_fits"], 24)
        self.assertEqual(plan.resource_limits["maximum_candidate_runs"], 4)
        self.assertEqual(plan.second_seed_rule["selection"], "both_fixed_target_candidates")
        self.assertEqual(
            target_decomposition_search.specification(
                target_decomposition_search.candidate("decomposed_ensemble")
            ).output_unit,
            target_decomposition.TARGET_UNIT,
        )
        serialized = str(plan.to_dict())
        self.assertNotIn("anonymous_source_group", serialized)
        self.assertNotIn("miniature_family", serialized)

    @staticmethod
    def record(identity, value):
        return {
            "_id": identity,
            "kb": value,
            "volume": value * 1000,
            "surface_area": value * 100,
            "bbox_area": value * 1100,
            "euler_number": value,
            "scale": value,
            "surface_volume_ratio": 0.1,
            "weight": value,
            "resin_density_g_per_ml": 1.1,
            "anonymous_source_group": "private-source",
        }

    def test_governed_lifecycle_scores_both_seeds_and_locks_scored_seed_42_state(self):
        runtime = DecompositionRuntime()
        training = [self.record(f"train-{index}", index + 1) for index in range(6)]
        validation = [self.record(f"validation-{index}", index + 20) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            result = develop_candidates(
                training,
                validation,
                EvaluationConfig(1.1, "mm3", True, seed=41),
                runtime=runtime,
                output_root=Path(temporary) / "private" / "run",
                limits=SearchLimits.for_plan(41, "bounding_box_target_decomposition"),
                clock=lambda: 0.0,
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 4)
        self.assertEqual(len(runtime.refit_calls), 24)
        self.assertEqual(result.resource_use["model_fit_attempts"], 24)
        self.assertEqual([run.seed for run in result.initial_results], [41, 41])
        self.assertEqual([run.seed for run in result.second_seed_results], [42, 42])
        self.assertIsNotNone(result.locked_candidate)
        runtime_metadata = result.locked_candidate.contract["runtime_metadata"]
        self.assertEqual(runtime_metadata["seed"], 42)
        self.assertEqual(len(runtime_metadata["component_fits"]), 2)
        self.assertTrue(all(item["model_seed"] == 42
                            for item in runtime_metadata["component_fits"]))
        scored_metadata = result.locked_candidate.contract["seed42_scored_fit_metadata"]
        self.assertEqual(len(scored_metadata), 1)
        self.assertEqual(
            scored_metadata[0]["scored_state_fingerprint"],
            runtime_metadata["scored_state_fingerprint"],
        )
        self.assertEqual(
            scored_metadata[0]["metrics_fingerprint"],
            fingerprint(result.locked_candidate.contract["seed42_scored_metrics"]),
        )
        self.assertEqual(len(runtime.refit_calls), 24, "locking must not refit")

    def test_lock_rejects_coordinated_fixed_count_metadata_tampering(self):
        runtime = DecompositionRuntime()
        training = [self.record(f"train-{index}", index + 1) for index in range(6)]
        validation = [self.record(f"validation-{index}", index + 20) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private" / "run"
            result = develop_candidates(
                training,
                validation,
                EvaluationConfig(1.1, "mm3", True, seed=41),
                runtime=runtime,
                output_root=root,
                limits=SearchLimits.for_plan(41, "bounding_box_target_decomposition"),
                clock=lambda: 0.0,
            )
            self.assertIsNotNone(result.locked_candidate)
            lock = root / "locked-candidate"
            contract_path = lock / "candidate-contract.json"
            contract = json.loads(contract_path.read_text())
            contract["runtime_metadata"]["component_fits"][0]["fixed_count"] += 1
            contract_path.write_text(json.dumps(contract))
            manifest_path = lock / "lock-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["candidate-contract.json"] = sha256(
                contract_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))

            blockers, _, _ = verify_locked_candidate_files(
                lock, runtime.dependency_versions
            )

        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_failed_training_prerequisite_stops_before_validation(self):
        runtime = DecompositionRuntime(improve=False)
        training = [self.record(f"train-{index}", index + 1) for index in range(6)]
        validation = [self.record(f"validation-{index}", index + 20) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            result = develop_candidates(
                training,
                validation,
                EvaluationConfig(1.1, "mm3", True, seed=41),
                runtime=runtime,
                output_root=Path(temporary) / "private" / "run",
                limits=SearchLimits.for_plan(41, "bounding_box_target_decomposition"),
                clock=lambda: 0.0,
            )

        self.assertEqual(result.status, "training_evidence_rejected")
        self.assertEqual(
            result.blockers,
            ("target_decomposition_training_prerequisite_failed",),
        )
        self.assertEqual(len(runtime.refit_calls), 16)
        self.assertEqual(result.run_count, 0)
        self.assertEqual(result.resource_use["validation_candidate_evaluations"], 0)

    def test_invalid_scale_or_prediction_fails_without_clipping(self):
        for row in (
            self.row(bounding_box_volume_mm3=0.0),
            self.row(density_g_per_ml=1.0),
        ):
            with self.subTest(row=row), self.assertRaisesRegex(
                ValueError, "invalid_target_decomposition_data"
            ):
                target_decomposition.occupancy_training_matrix((row,))

        features, _ = target_decomposition.occupancy_training_matrix((self.row(),))
        for prediction in (-0.1, math.inf, math.nan):
            with self.subTest(prediction=prediction), self.assertRaisesRegex(
                ValueError, "invalid_target_decomposition_prediction"
            ):
                target_decomposition.reconstruct_mass_predictions(features, (prediction,))


if __name__ == "__main__":
    unittest.main()
