import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from minires_evaluation import EvaluationConfig, evaluate_records
from minires_evaluation.legacy import (
    LEGACY_FEATURES,
    PINNED_ARTIFACTS,
    LegacyProvenance,
    LegacyReference,
    PinnedArtifact,
    load_legacy_reference,
    prepare_legacy_features,
    resolve_legacy_artifacts,
)


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = EvaluationConfig(
            resin_density_g_per_ml=None,
            volume_unit="mm3",
            scope_confirmed=True,
            seed=34,
        )
        self.record = {
            "kb": 12.9,
            "volume": 10.01,
            "surface_area": 25.06,
            "bbox_area": 100.04,
            "euler_number": -1,
            "scale": 76.76,
            "surface_volume_ratio": 2.503,
            "weight": 5.0,
        }

    def test_preserves_released_inference_order_without_assuming_training_notebook_parity(self):
        self.assertEqual(
            LEGACY_FEATURES,
            ("kb", "volume", "surface_area", "bbox_area", "euler_number", "scale", "surface_volume_ratio"),
        )
        self.assertEqual(
            prepare_legacy_features(self.record),
            (12.9, 10.01, 25.06, 100.04, -1.0, 76.76, 2.503),
        )

    def test_legacy_aliases_win_when_canonical_fields_are_also_present(self):
        seen = []

        def predictor(rows):
            seen.extend(rows)
            return [5.0 for _ in rows]

        model = LegacyReference.from_predictors(
            neural_network=predictor,
            xgboost=predictor,
            neural_network_weight=0.2,
            provenance=LegacyProvenance.unknown(),
        )
        record = dict(
            self.record,
            volume_mm3=999.0,
            surface_area_mm2=888.0,
            bounding_box_volume_mm3=777.0,
        )

        evaluate_records([record], self.config, model)

        self.assertEqual(seen[0], prepare_legacy_features(self.record))

    def test_runs_all_components_through_shared_diagnostics(self):
        model = LegacyReference.from_predictors(
            neural_network=lambda rows: [4.0 for _ in rows],
            xgboost=lambda rows: [6.0 for _ in rows],
            neural_network_weight=0.2,
            provenance=LegacyProvenance.unknown(),
        )

        result = evaluate_records([self.record], self.config, model)

        self.assertEqual(result.status, "completed")
        self.assertAlmostEqual(result.predictions[0].predicted_sliced_resin_mass_g, 5.6)
        self.assertEqual(set(result.model_diagnostics), {"neural_network", "xgboost", "ensemble"})
        self.assertEqual(result.model_diagnostics["ensemble"].within_tolerance_fraction, 1.0)
        self.assertEqual(result.model_diagnostics["ensemble"].within_tolerance_percent, 100.0)
        self.assertEqual(result.provenance_classification, "legacy_reference_training_provenance_unknown")
        self.assertEqual(result.run_metadata.baseline, "legacy_ensemble")

    def test_empty_scoring_set_does_not_call_inference_runtime(self):
        def must_not_run(rows):
            raise AssertionError("empty inference call")

        model = LegacyReference.from_predictors(
            neural_network=must_not_run,
            xgboost=must_not_run,
            neural_network_weight=0.2,
            provenance=LegacyProvenance.unknown(),
        )
        result = evaluate_records(
            [dict(self.record, weight=None)], self.config, model
        )

        self.assertEqual(result.status, "needs_review")
        self.assertNotIn("legacy_inference_failed", result.blockers)
        self.assertEqual(result.model_diagnostics["ensemble"].sample_count, 0)

    def test_source_holdout_requires_evidence_and_overlap_is_never_called_holdout(self):
        blocked = LegacyReference.from_predictors(
            neural_network=lambda rows: [5.0 for _ in rows],
            xgboost=lambda rows: [5.0 for _ in rows],
            neural_network_weight=0.2,
            provenance=LegacyProvenance.source_held_out(),
        )
        overlap = LegacyReference.from_predictors(
            neural_network=lambda rows: [5.0 for _ in rows],
            xgboost=lambda rows: [5.0 for _ in rows],
            neural_network_weight=0.2,
            provenance=LegacyProvenance.overlap(),
        )

        blocked_result = evaluate_records([self.record], self.config, blocked)
        overlap_result = evaluate_records([self.record], self.config, overlap)

        self.assertEqual(blocked_result.status, "blocked")
        self.assertIn("source_holdout_evidence_required", blocked_result.blockers)
        self.assertEqual(overlap_result.provenance_classification, "legacy_reference_training_overlap")
        self.assertNotIn("held_out", overlap_result.provenance_classification)

        held_out = LegacyReference.from_predictors(
            neural_network=lambda rows: [5.0 for _ in rows],
            xgboost=lambda rows: [5.0 for _ in rows],
            neural_network_weight=0.2,
            provenance=LegacyProvenance.source_held_out(["private-source"], "audit evidence"),
        )
        held_out_result = evaluate_records(
            [dict(self.record, anonymous_source_group="private-source")], self.config, held_out
        )
        private_report = held_out_result.to_dict()
        public_report = held_out_result.to_dict(public=True)
        self.assertEqual(held_out_result.provenance_classification, "legacy_reference_source_held_out")
        self.assertIn("evidence_fingerprint", private_report["provenance_evidence"])
        self.assertNotIn("provenance_evidence", public_report)

        mismatch = evaluate_records(
            [dict(self.record, anonymous_source_group="different-source")], self.config, held_out
        )
        self.assertEqual(mismatch.status, "blocked")
        self.assertIn("source_holdout_evidence_mismatch", mismatch.blockers)
        self.assertEqual(
            mismatch.provenance_classification, "legacy_reference_source_holdout_unverified"
        )

    def test_missing_artifacts_and_metadata_are_actionable_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = resolve_legacy_artifacts(directory)
            self.assertEqual(missing.status, "blocked")
            self.assertIn("legacy_artifact_missing_minires_keras", missing.blockers)

            root = Path(directory)
            for artifact in PINNED_ARTIFACTS:
                (root / artifact.filename).write_bytes(b"wrong")
            invalid = resolve_legacy_artifacts(root)
            self.assertEqual(invalid.status, "blocked")
            self.assertIn("legacy_artifact_checksum_mismatch", invalid.blockers)

    def test_mismatched_metadata_and_missing_dependencies_return_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = b'{"w_nn":0.7,"features":[]}'
            specs = []
            for name, content in (
                ("minires.keras", b"nn"),
                ("minires_xgb.json", b"xgb"),
                ("minires_meta.json", metadata),
            ):
                (root / name).write_bytes(content)
                specs.append(PinnedArtifact(name, hashlib.sha256(content).hexdigest(), len(content)))
            with patch("minires_evaluation.legacy.PINNED_ARTIFACTS", tuple(specs)):
                mismatch = resolve_legacy_artifacts(root)
            self.assertIn("legacy_metadata_mismatch", mismatch.blockers)

            valid_metadata = json.dumps({"w_nn": 0.2, "features": list(LEGACY_FEATURES)}).encode()
            (root / "minires_meta.json").write_bytes(valid_metadata)
            specs[-1] = PinnedArtifact(
                "minires_meta.json", hashlib.sha256(valid_metadata).hexdigest(), len(valid_metadata)
            )
            with patch("minires_evaluation.legacy.PINNED_ARTIFACTS", tuple(specs)), patch.dict(
                "sys.modules", {"numpy": None}
            ):
                loaded = load_legacy_reference(root)
            self.assertEqual(loaded.blockers, ("legacy_inference_dependencies_required",))

    def test_loaded_neural_network_inference_avoids_keras_dataset_pipeline(self):
        import numpy as np

        class FakeNeuralNetwork:
            def predict(self, *_args, **_kwargs):
                raise AssertionError("Keras predict dataset pipeline must not be used")

            def __call__(self, rows, *, training):
                self.training = training
                return np.full((len(rows), 1), 5.0, dtype=np.float32)

        class FakeXGBoost:
            def load_model(self, _path):
                pass

            def predict(self, rows):
                return np.full(len(rows), 4.0, dtype=np.float32)

        neural_model = FakeNeuralNetwork()
        tensorflow = types.ModuleType("tensorflow")
        tensorflow_keras = types.ModuleType("tensorflow.keras")
        tensorflow_models = types.ModuleType("tensorflow.keras.models")
        tensorflow_models.load_model = lambda *_args, **_kwargs: neural_model
        xgboost = types.ModuleType("xgboost")
        xgboost.XGBRegressor = FakeXGBoost

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "minires.keras": b"nn",
                "minires_xgb.json": b"xgb",
                "minires_meta.json": json.dumps(
                    {"w_nn": 0.2, "features": list(LEGACY_FEATURES)}
                ).encode(),
            }
            specs = []
            for name, content in contents.items():
                (root / name).write_bytes(content)
                specs.append(PinnedArtifact(name, hashlib.sha256(content).hexdigest(), len(content)))
            modules = {
                "tensorflow": tensorflow,
                "tensorflow.keras": tensorflow_keras,
                "tensorflow.keras.models": tensorflow_models,
                "xgboost": xgboost,
            }
            with patch("minires_evaluation.legacy.PINNED_ARTIFACTS", tuple(specs)), patch.dict(
                sys.modules, modules
            ):
                loaded = load_legacy_reference(root)

        assert loaded.neural_network is not None
        predictions = loaded.neural_network([(1.0,) * len(LEGACY_FEATURES)])
        self.assertEqual(predictions, [5.0])
        self.assertFalse(neural_model.training)

    def test_pinned_manifest_records_immutable_provenance_and_real_checksums(self):
        self.assertEqual(
            {artifact.filename: artifact.sha256 for artifact in PINNED_ARTIFACTS},
            {
                "minires.keras": "369cbb70097ab12ca08cad89c4ad356e115f85feacebe1ebddc5222d92b4fe85",
                "minires_xgb.json": "928bb7f87c704de47cc54d387a78e5eb74584c9ad117b24855a3abbf018f80ed",
                "minires_meta.json": "12d17f5538081f8f47de7c752b3117e96a0da60edab37cfd073209b9c45d90f2",
            },
        )
        self.assertTrue(all(len(artifact.revision) == 40 for artifact in PINNED_ARTIFACTS))

    def test_public_report_and_failures_do_not_echo_private_values(self):
        canary = dict(self.record, artist="identifying-canary", file="/private/canary.stl")
        model = LegacyReference.blocked(("legacy_dependency_required",), LegacyProvenance.unknown())

        result = evaluate_records([canary], self.config, model)
        rendered = json.dumps(result.to_dict(public=True))

        self.assertEqual(result.status, "blocked")
        self.assertNotIn("identifying-canary", rendered)
        self.assertNotIn("canary.stl", rendered)
        self.assertNotIn("artifact_paths", rendered)


if __name__ == "__main__":
    unittest.main()
