import math
import unittest

from minires_evaluation import (
    EvaluationConfig,
    PhysicalBaseline,
    evaluate_records,
)


class EvaluationInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.config = EvaluationConfig(
            resin_density_g_per_ml=1.1,
            volume_unit="mm3",
            scope_confirmed=True,
            seed=17,
        )

    def test_evaluates_physical_baseline_and_correct_metrics(self):
        result = evaluate_records(
            records=[
                {"volume": 1000.0, "weight": 1.1, "artist": "private-a"},
                {"volume": 2000.0, "weight": 3.3, "artist": "private-b"},
            ],
            config=self.config,
            baseline=PhysicalBaseline(),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.split_status, "not_applicable")
        self.assertEqual(result.metrics.sample_count, 2)
        self.assertAlmostEqual(result.metrics.mae_g, 0.55)
        self.assertAlmostEqual(result.metrics.rmse_g, math.sqrt(0.605))
        self.assertAlmostEqual(result.metrics.signed_error_g, -0.55)
        self.assertEqual(result.metrics.within_tolerance_fraction, 1.0)
        self.assertEqual(result.metrics.within_tolerance_percent, 100.0)
        self.assertEqual(result.metrics.underestimation_count, 1)
        self.assertAlmostEqual(result.metrics.mean_underestimation_g, 1.1)
        self.assertEqual(result.data_quality.accepted_count, 2)
        self.assertEqual(result.data_quality.needs_review_count, 0)

    def test_tolerance_boundary_is_inclusive_and_diagnostics_are_declared(self):
        result = evaluate_records(
            records=[
                {"volume": 1000.0, "weight": 3.1},
                {"volume": 6000.0, "weight": 6.6},
            ],
            config=self.config,
            baseline=PhysicalBaseline(),
        )

        self.assertEqual(result.metrics.within_tolerance_fraction, 1.0)
        self.assertEqual(
            [item.label for item in result.metrics.absolute_error_bins],
            ["[0.0, 2.0]", "(2.0, 5.0]", "(5.0, inf)"],
        )
        self.assertEqual(sum(item.count for item in result.metrics.absolute_error_bins), 2)
        self.assertEqual(sum(item.count for item in result.metrics.volume_bins), 2)

    def test_marks_invalid_records_and_missing_scope_confirmation_for_review(self):
        result = evaluate_records(
            records=[
                {"volume": 0, "weight": 1},
                {"volume": float("nan"), "weight": 1},
                {"volume": 1000, "weight": None},
                {"volume": 1000, "weight": float("nan")},
                {"volume": 1000, "weight": -1},
                {"volume": 1000, "sliced_resin_mass_g": 1.1},
            ],
            config=EvaluationConfig(
                resin_density_g_per_ml=1.1,
                volume_unit="mm3",
                scope_confirmed=None,
            ),
            baseline=PhysicalBaseline(),
        )

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.metrics.sample_count, 0)
        self.assertEqual(result.data_quality.accepted_count, 0)
        self.assertEqual(result.data_quality.reasons, {
            "invalid_volume": 1,
            "non_finite_volume_mm3": 1,
            "missing_target_sliced_resin_mass": 1,
            "invalid_target_sliced_resin_mass": 1,
            "non_finite_target_sliced_resin_mass": 1,
            "scope_confirmation_required": 1,
        })

    def test_blocks_unknown_density_or_units_without_inventing_defaults(self):
        missing_density = evaluate_records(
            records=[{"volume": 1000, "weight": 1.1}],
            config=EvaluationConfig(
                resin_density_g_per_ml=None,
                volume_unit="mm3",
                scope_confirmed=True,
            ),
            baseline=PhysicalBaseline(),
        )
        unsupported_unit = evaluate_records(
            records=[{"volume": 1000, "weight": 1.1}],
            config=EvaluationConfig(
                resin_density_g_per_ml=1.1,
                volume_unit="in3",
                scope_confirmed=True,
            ),
            baseline=PhysicalBaseline(),
        )

        self.assertEqual(missing_density.status, "blocked")
        self.assertEqual(missing_density.blockers, ("resin_density_required",))
        self.assertEqual(unsupported_unit.status, "blocked")
        self.assertEqual(unsupported_unit.blockers, ("unsupported_volume_unit",))

    def test_public_output_does_not_disclose_unknown_input_fields_or_review_values(self):
        result = evaluate_records(
            records=[
                {
                    "volume": 1000,
                    "weight": 1.1,
                    "artist": "identifying-canary",
                    "file": "/private/path/canary.stl",
                },
                {"volume": 0, "weight": 1.1, "file": "/private/path/bad.stl"},
            ],
            config=self.config,
            baseline=PhysicalBaseline(),
        )
        public = result.to_dict(public=True)
        rendered = str(public)

        self.assertNotIn("identifying-canary", rendered)
        self.assertNotIn("/private/path", rendered)
        self.assertNotIn("normalized_records", public)
        self.assertEqual(public["data_quality"]["reasons"], {"invalid_volume": 1})

    def test_repeat_runs_are_deterministic_and_metadata_has_safe_fingerprint(self):
        records = [{"volume": 1000, "weight": 1.1, "artist": "canary"}]
        first = evaluate_records(records, self.config, PhysicalBaseline())
        second = evaluate_records(records, self.config, PhysicalBaseline())

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.run_metadata.seed, 17)
        self.assertEqual(first.run_metadata.volume_unit, "mm3")
        self.assertNotIn("canary", first.run_metadata.input_fingerprint)

    def test_prefers_canonical_unit_bearing_target_key(self):
        result = evaluate_records(
            records=[
                {"volume": 1000, "weight": 9.9, "sliced_resin_mass_g": 1.1},
            ],
            config=self.config,
            baseline=PhysicalBaseline(),
        )

        self.assertAlmostEqual(result.predictions[0].actual_sliced_resin_mass_g, 1.1)


if __name__ == "__main__":
    unittest.main()
