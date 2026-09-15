"""Governed search and checksum loading at the existing public lifecycle seams."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minires import EvaluationConfig
from minires.modeling.tuning import (
    LockedFit, SearchLimits, develop_candidates, generate_search_plan,
    TensorflowXGBoostCandidateRuntime, load_locked_candidate, verify_locked_candidate_files,
    tune_candidates,
)
from minires.modeling.definitions import (
    FittedModel, ModelKind, TailCorrectionModelSpecification, TensorflowXGBoostBackend,
)
from minires.ingestion import InputError, fingerprint, normalize


def preprocessing(family, mean=0.0):
    return ({"mean": [mean] * 7, "variance": [1.0] * 7}
            if family == "neural_network" else {"xgboost": "unnormalized_float32"})


def row(identity, value):
    return {
        "_id": identity, "kb": value, "volume": value * 1000,
        "surface_area": value * 100, "bbox_area": value * 1100,
        "euler_number": value, "scale": value, "surface_volume_ratio": 0.1,
        "weight": value, "anonymous_source_group": "synthetic-group",
    }


class SyntheticRuntime:
    dependency_versions = {"runtime": "synthetic-1"}

    def __init__(self, shift=0.0):
        self.shift = shift
        self.fits = []
        self.predicted_values = []

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.fits.append((seed, candidate, tuple(features), tuple(targets), fixed_training_counts))
        def predict(rows):
            self.predicted_values.extend(value[0] for value in rows)
            return [value[1] / 1000.0 + self.shift for value in rows]
        return LockedFit(predict, preprocessing(candidate.family),
                         {"model.bin": b"synthetic"}, {"seed": seed})

    def fit_fold(self, *args, **kwargs):
        raise AssertionError("early stopping is forbidden")


class SyntheticBackend:
    """Fake external model backend; real ModelRuntime and candidate loader remain in use."""
    dependency_versions = {"backend": "synthetic-1"}

    def __init__(self):
        self.fits = []
        self.loads = []

    def fit_component(self, specification, training, validation, seed):
        if validation is not None:
            raise AssertionError("no early stopping partition may reach a base")
        self.fits.append((specification, training, seed))
        offset = (1.0 if seed == 41 else 1.5)
        fitted_preprocessing = preprocessing(specification.model_kind.value, offset)
        state = {"offset": offset, "preprocessing": fitted_preprocessing}
        return FittedModel(
            specification, lambda rows: [value[1] / 1000.0 + offset for value in rows],
            fitted_preprocessing, {"seed": seed}, state,
        )

    def serialize_component(self, fitted):
        name = "model.keras" if fitted.specification.model_kind is ModelKind.NEURAL_NETWORK else "model.json"
        return {name: json.dumps(dict(fitted.backend_state)).encode()}

    def load_component(self, specification, artifacts, preprocessing_state):
        self.loads.append((specification, artifacts, preprocessing_state))
        state = json.loads(next(iter(artifacts.values())))
        if state["preprocessing"] != preprocessing_state:
            raise ValueError("state mismatch")
        return lambda rows: [value[1] / 1000.0 + state["offset"] for value in rows]


class SerializedLoadingBackend(TensorflowXGBoostBackend):
    """Real production loading logic with only the external frameworks replaced."""

    def __init__(self):
        import numpy as np
        class Normalization:
            def __init__(self, state):
                self.mean = SimpleNamespace(numpy=lambda: np.asarray(state["mean"]))
                self.variance = SimpleNamespace(numpy=lambda: np.asarray(state["variance"]))

        class Neural:
            def __init__(self, path):
                self.state = json.loads(Path(path).read_text())
                self.input_shape = (None, self.state.get("width", 7))
                self.layers = [Normalization(self.state["preprocessing"])]

            def __call__(self, rows, training=False):
                return np.asarray([value[1] / 1000.0 + self.state["offset"] for value in rows])

        class XGBoost:
            def load_model(self, path):
                self.state = json.loads(Path(path).read_text())
                self.n_features_in_ = self.state.get("width", 7)

            def predict(self, rows):
                return np.asarray([value[1] / 1000.0 + self.state["offset"] for value in rows])

        self.np = np
        self.tf = SimpleNamespace(keras=SimpleNamespace(
            layers=SimpleNamespace(Normalization=Normalization),
            models=SimpleNamespace(load_model=lambda path, compile=False: Neural(path)),
        ))
        self.xgboost = SimpleNamespace(XGBRegressor=XGBoost)
        self.dependency_versions = SyntheticBackend.dependency_versions


class TailFocusedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private"
        self.training = [row(f"train-{i}", i + 1) for i in range(50)]
        self.validation = [row(f"validation-{i}", i + 101) for i in range(20)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def test_invalid_fitted_base_preprocessing_blocks_before_prediction(self):
        class InvalidPreprocessingRuntime(SyntheticRuntime):
            def refit(self, *args):
                return replace(super().refit(*args), preprocessing_state={})
        runtime = InvalidPreprocessingRuntime(1.0)
        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=runtime, output_root=self.root / "invalid-fitted-state",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                                    clock=lambda: 0.0)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("candidate_runtime_failed",))
        self.assertEqual(result.resource_use["model_fits"], 1)
        self.assertEqual(runtime.predicted_values, [])
        self.assertIsNone(result.locked_candidate)

    def test_production_loader_matches_preprocessing_and_width_to_serialized_bases(self):
        with patch("minires.modeling.tuning.TensorflowXGBoostBackend", return_value=SyntheticBackend()):
            fitting_runtime = TensorflowXGBoostCandidateRuntime()
        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=fitting_runtime, output_root=self.root / "serialized",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                                    clock=lambda: 0.0)
        directory = result.locked_candidate.directory
        with patch("minires.modeling.tuning.TensorflowXGBoostBackend", return_value=SerializedLoadingBackend()):
            loading_runtime = TensorflowXGBoostCandidateRuntime()
        loaded = load_locked_candidate(directory, loading_runtime)
        features = ((7.0, 7000.0, 700.0, 7700.0, 7.0, 7.0, 0.1),)
        self.assertEqual(loaded.predictor(features), result.locked_candidate.predictor(features))
        manifest_path = directory / "lock-manifest.json"
        original_manifest = json.loads(manifest_path.read_text())
        for filename, mutation in (
            ("preprocessing-state.json", lambda state: state["base_1"].update(mean=[8.0] * 7)),
            ("base-01-model.keras", lambda state: state.update(width=6)),
            ("base-02-model.json", lambda state: state.update(width=6)),
        ):
            with self.subTest(filename=filename):
                path = directory / filename
                original = path.read_bytes()
                state = json.loads(original)
                mutation(state)
                path.write_text(json.dumps(state))
                manifest = json.loads(json.dumps(original_manifest))
                manifest["files"][filename] = sha256(path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(InputError):
                    load_locked_candidate(directory, loading_runtime)
                path.write_bytes(original)
                manifest_path.write_text(json.dumps(original_manifest))

    def test_lock_rejects_incomplete_or_invalid_base_preprocessing_with_updated_checksums(self):
        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=SyntheticRuntime(1.0), output_root=self.root / "preprocessing",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                                    clock=lambda: 0.0)
        directory = result.locked_candidate.directory
        state_path = directory / "preprocessing-state.json"
        original = json.loads(state_path.read_text())
        manifest_path = directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        invalid_states = (
            ("base_1", {}), ("base_2", {}),
            ("base_1", {"mean": [0.0] * 6, "variance": [1.0] * 7}),
            ("base_1", {"mean": [float("nan")] * 7, "variance": [1.0] * 7}),
            ("base_1", {"mean": [4e38] * 7, "variance": [1.0] * 7}),
            ("base_1", {"mean": [0.0] * 7, "variance": [-1.0] * 7}),
            ("base_2", {"xgboost": "normalized_float32"}),
        )
        for base, invalid in invalid_states:
            with self.subTest(base=base, invalid=invalid):
                state_path.write_text(json.dumps({**original, base: invalid}))
                manifest["files"][state_path.name] = sha256(state_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                blockers, _, _ = verify_locked_candidate_files(directory, SyntheticRuntime.dependency_versions)
                self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_unfavorable_second_seed_is_retained_and_cannot_lock(self):
        class UnfavorableRuntime(SyntheticRuntime):
            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                fitted = super().refit(candidate, seed, features, targets, fixed_training_counts)
                return replace(fitted, predictor=lambda rows: [
                    value[1] / 1000.0 + (7.0 if seed == 42 and value[0] > 100 else 1.0)
                    for value in rows])
        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=UnfavorableRuntime(), output_root=self.root / "unfavorable",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                                    clock=lambda: 0.0)
        self.assertEqual(result.status, "completed_no_candidate")
        self.assertEqual(result.resource_use["model_fits"], 52)
        self.assertEqual(result.run_count, 4)
        self.assertTrue(all(run.eligible for run in result.initial_results))
        self.assertFalse(any(run.eligible for run in result.second_seed_results))
        self.assertFalse(any(run["eligible"] for run in result.combined_results))
        self.assertIsNone(result.locked_candidate)
        self.assertEqual(result.to_dict()["second_seed_comparison"]["shortfall"], 0)

    def test_expired_final_correction_fit_stops_before_second_seed_validation_scoring(self):
        class Clock:
            now = 0.0
            final_base_predictions = 0

            def __call__(self):
                observed = self.now
                # Both final base prediction vectors are ready. The next fitting
                # boundary starts the final numerical correction; time expires
                # before control returns from that fit, not before it starts.
                if self.final_base_predictions == 2:
                    self.now = 7201.0
                return observed

        clock = Clock()
        class ExpiringRuntime(SyntheticRuntime):
            def refit(self, candidate, seed, features, targets, fixed_training_counts):
                fitted = super().refit(candidate, seed, features, targets, fixed_training_counts)
                final_base = seed == 42 and len(features) == 50 and candidate.family == "xgboost"
                def predict(rows):
                    if final_base:
                        clock.final_base_predictions += 1
                    return [value[1] / 1000.0 + (7.0 if value[0] > 100 else 1.0) for value in rows]
                return replace(fitted, predictor=predict)

        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=ExpiringRuntime(), output_root=self.root / "expired-final-fit",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"), clock=clock)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("candidate_search_deadline_reached",))
        self.assertEqual(result.resource_use["model_fits"], 52)
        self.assertEqual(result.run_count, 2)
        self.assertEqual(result.second_seed_results, ())
        self.assertFalse(any(run.eligible for run in result.initial_results))
        self.assertIsNone(result.locked_candidate)

    def test_deadline_and_invalid_base_predictions_stop_without_recycling(self):
        class InvalidRuntime(SyntheticRuntime):
            def refit(self, *args):
                fitted = super().refit(*args)
                return replace(fitted, predictor=lambda rows: [float("nan")] * len(rows))
        for name, runtime, clock, expected in (
            ("invalid", InvalidRuntime(), lambda: 0.0, "candidate_runtime_failed"),
            ("deadline", SyntheticRuntime(), iter((0.0, 7200.0, 7200.0)).__next__,
             "candidate_search_deadline_reached"),
        ):
            result = develop_candidates(self.training, self.validation, self.config, runtime=runtime,
                                        output_root=self.root / name,
                                        limits=SearchLimits.for_plan(41, "tail_focused_correction"), clock=clock)
            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.blockers, (expected,))
            self.assertLessEqual(result.resource_use["model_fits"], 1)
            self.assertEqual(result.run_count, 0)
            self.assertEqual(len(result.to_dict()["skipped_candidates"]), 4)
            self.assertIsNone(result.locked_candidate)

    def test_coherently_checksummed_incomplete_shift_evidence_is_rejected(self):
        result = develop_candidates(self.training, self.validation, self.config,
                                    runtime=SyntheticRuntime(1.0), output_root=self.root / "incomplete",
                                    limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                                    clock=lambda: 0.0)
        directory = result.locked_candidate.directory
        shift_path = directory / "oof-full-fit-shift.json"
        shift = json.loads(shift_path.read_text())
        shift["seeds"]["41"] = {}
        shift_path.write_text(json.dumps(shift))
        contract_path = directory / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["development_evidence"]["oof_full_fit_shift_fingerprint"] = fingerprint(shift)
        contract_path.write_text(json.dumps(contract))
        manifest_path = directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for path in (shift_path, contract_path):
            manifest["files"][path.name] = sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        blockers, _, _ = verify_locked_candidate_files(directory, SyntheticRuntime.dependency_versions)
        self.assertIn("locked_candidate_contract_mismatch", blockers)

    def test_plan_is_fixed_and_historical_source_holdout_cannot_execute_it(self):
        limits = SearchLimits.for_plan(41, "tail_focused_correction")
        plan = generate_search_plan(limits, input_fingerprint="synthetic", code_fingerprint="code",
                                    dependency_versions=SyntheticRuntime.dependency_versions)
        self.assertEqual(plan, generate_search_plan(
            limits, input_fingerprint="synthetic", code_fingerprint="code",
            dependency_versions=SyntheticRuntime.dependency_versions))
        self.assertEqual(len(plan.component_trials), 2)
        self.assertEqual(len(plan.ensemble_rules), 2)
        self.assertEqual(plan.resource_limits["maximum_model_fits"], 52)
        self.assertEqual(plan.resource_limits["maximum_candidate_runs"], 4)
        self.assertEqual(plan.eligibility_gates, {
            "pooled_above_5g_fraction_maximum": 0.01,
            "source_balanced_above_5g_fraction_maximum": 0.01,
            "per_source_above_5g_fraction_maximum": 0.02,
            "per_source_minimum_accepted_records": 200,
        })
        self.assertEqual(plan.component_trials[0].parameters["maximum_epochs"], 100)
        self.assertEqual(plan.component_trials[1].parameters["n_estimators"], 1200)
        self.assertEqual(plan.ensemble_rules[0]["fixed_training_counts"],
                         {"neural_network_epochs": 87, "xgboost_trees": 1091})
        with self.assertRaisesRegex(InputError, "explicit_train_validation_required"):
            tune_candidates(self.training, self.config, runtime=SyntheticRuntime(),
                            output_root=self.root / "forbidden", limits=limits)
        self.assertFalse((self.root / "forbidden").exists())
        for changed in (replace(limits, seed=42), replace(limits, maximum_candidate_runs=5),
                        replace(limits, second_seed=43)):
            with self.assertRaises(ValueError):
                generate_search_plan(changed, input_fingerprint="synthetic", code_fingerprint="code",
                                     dependency_versions=SyntheticRuntime.dependency_versions)

    def test_outer_targets_cannot_influence_any_honest_fit_or_correction_prediction(self):
        normalized = normalize(self.training, self.config, contract="legacy")
        ordered = sorted(range(50), key=lambda i: sha256(
            f"41:{normalized[i].metadata['record_identity']}".encode()).hexdigest())
        held = set(ordered[::5])
        changed = [dict(record, weight=record["weight"] + 1000) if i in held else dict(record)
                   for i, record in enumerate(self.training)]
        original_runtime, changed_runtime = SyntheticRuntime(1.0), SyntheticRuntime(1.0)
        evidence = []
        for name, records, runtime in (("original", self.training, original_runtime),
                                        ("poisoned", changed, changed_runtime)):
            develop_candidates(records, self.validation, self.config, runtime=runtime,
                               output_root=self.root / name,
                               limits=SearchLimits.for_plan(41, "tail_focused_correction"),
                               clock=lambda: 0.0)
            evidence.append(json.loads((self.root / name / "honest-training-evidence.json").read_text()))
        self.assertEqual(original_runtime.fits[:12], changed_runtime.fits[:12])
        held_values = {self.training[i]["kb"] for i in held}
        for fit in original_runtime.fits[:12]:
            self.assertFalse(held_values & {features[0] for features in fit[2]})
        self.assertEqual(evidence[0]["seeds"]["41"]["prediction_distributions"],
                         evidence[1]["seeds"]["41"]["prediction_distributions"])
        self.assertNotEqual(evidence[0]["seeds"]["41"]["anchor"]["prediction_loss"],
                            evidence[1]["seeds"]["41"]["anchor"]["prediction_loss"])

    def test_lock_round_trip_at_model_backend_and_rechecks_modified_feature_contract(self):
        backend = SyntheticBackend()
        with patch("minires.modeling.tuning.TensorflowXGBoostBackend", return_value=backend):
            runtime = TensorflowXGBoostCandidateRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "locked",
            limits=SearchLimits.for_plan(41, "tail_focused_correction"), clock=lambda: 0.0,
        )
        self.assertEqual(result.status, "completed")
        locked = result.locked_candidate
        self.assertIsNotNone(locked)
        rows = ((7.0, 7000.0, 700.0, 7700.0, 7.0, 7.0, 0.1),)
        reloaded = load_locked_candidate(locked.directory, runtime)
        self.assertEqual(reloaded.predictor(rows), locked.predictor(rows))
        self.assertEqual(reloaded.candidate, locked.candidate)
        specification = locked.specification
        self.assertIsInstance(specification, TailCorrectionModelSpecification)
        self.assertEqual(specification, reloaded.specification)
        self.assertEqual(specification.to_dict(), locked.contract["model_specification"])
        self.assertEqual(specification.stable_identity, reloaded.specification.stable_identity)
        self.assertEqual(specification.correction_scale, 1.0)
        self.assertEqual(len(specification.bases), 2)
        self.assertIn("tail_focused_correction", specification.describe())
        self.assertEqual(backend.loads[0][2]["mean"], [1.5] * 7)
        self.assertEqual(len(backend.fits), 48)
        for specification, _, _ in backend.fits:
            if specification.model_kind is ModelKind.NEURAL_NETWORK:
                self.assertEqual(specification.training_parameters["maximum_epochs"], 87)
                self.assertEqual(specification.preprocessing.normalization, "fit_on_training_records")
            else:
                self.assertEqual(specification.architecture_parameters["n_estimators"], 1091)
                self.assertEqual(specification.preprocessing.normalization, "none")
        artifact = locked.directory / "base-01-model.keras"
        original_artifact = artifact.read_bytes()
        artifact.write_bytes(b"modified")
        with self.assertRaises(InputError):
            load_locked_candidate(locked.directory, runtime)
        artifact.write_bytes(original_artifact)
        contract_path = locked.directory / "candidate-contract.json"
        contract = json.loads(contract_path.read_text())
        contract["feature_contract"]["base_transformation_version"] = "changed"
        contract_path.write_text(json.dumps(contract))
        manifest_path = locked.directory / "lock-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][contract_path.name] = sha256(contract_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(InputError):
            load_locked_candidate(locked.directory, runtime)

    def test_qualified_honest_evidence_runs_both_variants_under_both_seeds(self):
        runtime = SyntheticRuntime(shift=1.0)
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "qualified",
            limits=SearchLimits.for_plan(41, "tail_focused_correction"), clock=lambda: 0.0,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_count, 4)
        self.assertEqual(result.resource_use["model_fits"], 52)
        self.assertEqual([run.seed for run in result.initial_results], [41, 41])
        self.assertEqual([run.seed for run in result.second_seed_results], [42, 42])
        self.assertIsNotNone(result.locked_candidate)
        self.assertEqual(result.locked_candidate.contract["runtime_metadata"]["seed"], 42)
        self.assertEqual(result.locked_candidate.candidate.parameters["correction_scale"], 1.0)
        self.assertTrue(result.to_dict()["second_seed_comparison"]["complete"])
        self.assertEqual(result.to_dict()["second_seed_comparison"]["shortfall"], 0)

    def test_honest_rejection_stops_before_production_fits_and_validation_scoring(self):
        runtime = SyntheticRuntime()
        result = develop_candidates(
            self.training, self.validation, self.config, runtime=runtime,
            output_root=self.root / "rejected",
            limits=SearchLimits.for_plan(41, "tail_focused_correction"), clock=lambda: 0.0,
        )
        self.assertEqual(result.status, "training_evidence_rejected")
        self.assertEqual(result.blockers, ("honest_training_correction_gate_failed",))
        self.assertEqual(result.resource_use["model_fits"], 26)
        self.assertEqual(result.run_count, 0)
        self.assertIsNone(result.locked_candidate)
        self.assertEqual({fit[0] for fit in runtime.fits}, {41, 42})
        self.assertTrue(all(value <= 50 for value in runtime.predicted_values))
        evidence = json.loads((self.root / "rejected" / "honest-training-evidence.json").read_text())
        self.assertEqual(set(evidence["seeds"]), {"41", "42"})
        self.assertTrue(all(not seed["qualified"] for seed in evidence["seeds"].values()))
        self.assertEqual(len(result.to_dict()["skipped_candidates"]), 4)


if __name__ == "__main__":
    unittest.main()
