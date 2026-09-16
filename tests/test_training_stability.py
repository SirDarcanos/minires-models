"""Training-only crossed split/model-seed stability diagnosis at its public seam."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minires import EvaluationConfig
from minires.modeling.training_stability import (
    _verify_predeclared_output_root,
    run_training_stability_diagnostic,
)
from minires.modeling.tuning import LockedFit


def row(identity: str, value: float) -> dict[str, object]:
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
        "anonymous_source_group": "synthetic-group",
    }


class RecordingRuntime:
    dependency_versions = {"runtime": "synthetic-1"}

    def __init__(self) -> None:
        self.fits: list[tuple[int, str, tuple[tuple[float, ...], ...]]] = []

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.fits.append((seed, candidate.family, tuple(features)))
        offset = 1.0 if seed == 41 else 1.5
        preprocessing = (
            {"mean": [0.0] * 7, "variance": [1.0] * 7}
            if candidate.family == "neural_network"
            else {"xgboost": "unnormalized_float32"}
        )
        return LockedFit(
            lambda rows: [value[1] / 1000.0 + offset for value in rows],
            preprocessing,
            {"model.bin": b"synthetic"},
            {"seed": seed},
        )


class TrainingStabilityDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "private" / "run-015"
        self.records = [row(f"train-{index}", index + 1) for index in range(50)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def test_rejects_any_unpredeclared_normalization_configuration(self):
        for config in (
            EvaluationConfig(None, "cm3", True, seed=17),
            EvaluationConfig(None, "mm3", None, seed=17),
            EvaluationConfig(None, "mm3", True, seed=41),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(Exception, "invalid_training_stability_configuration"):
                    run_training_stability_diagnostic(
                        self.records, config, runtime=RecordingRuntime(), output_root=self.output,
                        clock=lambda: 0.0,
                    )
        self.assertFalse(self.output.exists())

    def test_rejects_any_output_root_other_than_the_predeclared_run(self):
        with self.assertRaisesRegex(Exception, "training_stability_output_root_mismatch"):
            _verify_predeclared_output_root(self.output)

    def test_crosses_two_fixed_splits_with_two_model_seeds_without_emitting_private_metadata(self):
        runtime = RecordingRuntime()

        result = run_training_stability_diagnostic(
            self.records, self.config, runtime=runtime, output_root=self.output,
            clock=lambda: 0.0,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.resource_use["fits"], 52)
        self.assertEqual(len(runtime.fits), 48)
        self.assertEqual({seed for seed, _, _ in runtime.fits}, {41, 42})
        self.assertEqual(set(result.evidence["cells"]), {"split-101-model-41", "split-101-model-42",
                                                            "split-202-model-41", "split-202-model-42"})
        self.assertEqual(set(result.evidence["model_seed_contrasts"]), {"split-101", "split-202"})
        self.assertEqual(set(result.evidence["split_contrasts"]), {"model-41", "model-42"})
        self.assertTrue(all(cell["held_out_record_count"] == 10
                            for cell in result.evidence["cells"].values()))
        self.assertTrue(all(cell["validation_labels_used"] is False
                            for cell in result.evidence["cells"].values()))
        self.assertTrue(all(cell["qualified"] is None for cell in result.evidence["cells"].values()))
        self.assertTrue((self.output / "diagnostic-plan.json").is_file())
        self.assertTrue((self.output / "training-stability-evidence.json").is_file())
        self.assertTrue((self.output / "manifest.json").is_file())
        for path in self.output.iterdir():
            self.assertNotIn("train-", path.read_text())
            self.assertNotIn("synthetic-group", path.read_text())
        plan = json.loads((self.output / "diagnostic-plan.json").read_text())
        self.assertEqual(plan["model_seeds"], [41, 42])
        self.assertEqual(plan["outer_split_seeds"], [101, 202])
        self.assertEqual(plan["maximum_fits"], 52)


if __name__ == "__main__":
    unittest.main()
