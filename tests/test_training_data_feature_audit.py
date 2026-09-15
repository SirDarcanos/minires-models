"""Synthetic public-seam tests for the bounded training data/feature audit."""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires import EvaluationConfig
from minires.modeling import training_data_feature_audit as audit
from minires.modeling import training_stability as stability
from minires.modeling.tuning import LockedFit


def row(identity: str, value: float, *, target: float | None = None) -> dict[str, object]:
    x, y, z = value * 10.0 + 1.0, value * 10.0 + 2.0, value * 10.0 + 3.0
    volume = value * 1000.0
    surface = value * 100.0
    bbox = x * y * z
    ratio = surface / volume
    target_value = value if target is None else target
    return {
        "_id": identity,
        "anonymous_source_group": "private-synthetic-source",
        "file_size_kib": value,
        "kb": value,
        "volume_mm3": volume,
        "volume": volume,
        "surface_area_mm2": surface,
        "surface_area": surface,
        "bounding_box_x_mm": x,
        "bbox_x": x,
        "bounding_box_y_mm": y,
        "bbox_y": y,
        "bounding_box_z_mm": z,
        "bbox_z": z,
        "bounding_box_volume_mm3": bbox,
        "bbox_area": bbox,
        "mesh_mass_at_unit_density": volume,
        "mass": volume,
        "euler_characteristic": 2,
        "euler_number": 2,
        "mesh_scale_mm": value,
        "scale": value,
        "surface_to_volume_ratio_per_mm": ratio,
        "surface_volume_ratio": ratio,
        "sliced_resin_mass_g": target_value,
        "weight": target_value,
        "volume_unit": "mm3",
        "resin_density_g_per_ml": 1.1,
        "scope_confirmed": True,
        "slicing_conditions": {"layer_height_mm": 0.05, "slicer_added_supports": False},
    }


class AuditRuntime:
    dependency_versions = {"runtime": "synthetic-1"}

    def __init__(self) -> None:
        self.fits: list[tuple[int, str, int]] = []

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        self.fits.append((seed, candidate.family, len(features)))
        preprocessing = (
            {"mean": [0.0] * 7, "variance": [1.0] * 7}
            if candidate.family == "neural_network"
            else {"xgboost": "unnormalized_float32"}
        )
        return LockedFit(
            lambda values: [float(item[1]) / 1000.0 for item in values],
            preprocessing,
            {"model.bin": b"synthetic"},
            {"seed": seed},
        )


class TrainingDataFeatureAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "private" / "synthetic-run-019"
        self.records = [row(f"private-row-{index}", index + 1.0) for index in range(150)]
        self.config = EvaluationConfig(None, "mm3", True, seed=17)

    def run_audit(self, runtime=None, **kwargs):
        return audit.run_training_data_feature_audit(
            self.records,
            self.config,
            runtime=runtime or AuditRuntime(),
            output_root=self.output,
            clock=kwargs.pop("clock", lambda: 0.0),
            **kwargs,
        )

    def test_runs_four_frozen_anchor_cells_with_eight_fits_and_aggregate_only_artifacts(self):
        runtime = AuditRuntime()

        result = self.run_audit(runtime)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.resource_use["fits_started"], 8)
        self.assertEqual(len(runtime.fits), 8)
        self.assertEqual({seed for seed, _, _ in runtime.fits}, {41, 42})
        self.assertEqual(set(result.evidence["cells"]), {
            "split-101-model-41", "split-101-model-42",
            "split-202-model-41", "split-202-model-42",
        })
        self.assertEqual(result.evidence["decision"]["route"], "inconclusive_collect_evidence")
        self.assertFalse(result.evidence["source_groups_used"])
        self.assertEqual(result.evidence["rows_removed_or_repaired"], 0)
        plan = json.loads((self.output / "audit-plan.json").read_text())
        self.assertEqual(plan["maximum_fits"], 8)
        self.assertEqual(plan["fit_allocation"], {
            "outer_partition_neural_network": 4,
            "outer_partition_xgboost": 4,
        })
        self.assertEqual(plan["required_environment"]["python"], "3.13")
        self.assertEqual(plan["frozen_anchor"]["fixed_training_counts"], {
            "neural_network_epochs": 87,
            "xgboost_trees": 1091,
        })
        self.assertEqual(set(path.name for path in self.output.iterdir()), {
            "audit-plan.json", "training-data-feature-audit.json", "manifest.json",
        })
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()).hexdigest(), digest)
        serialized = "".join(path.read_text() for path in self.output.iterdir())
        self.assertNotIn("private-row-", serialized)
        self.assertNotIn("private-synthetic-source", serialized)
        self.assertNotIn('"row_ids"', serialized)
        self.assertNotIn('"per_row_predictions"', serialized)
        self.assertNotIn('"per_row_targets"', serialized)

    def test_explicit_alias_contradiction_has_priority_and_is_not_repaired(self):
        self.records[0]["volume_mm3"] = float(self.records[0]["volume_mm3"]) + 1.0
        original = dict(self.records[0])

        result = self.run_audit()

        contract = result.evidence["contract_and_target_audit"]
        self.assertEqual(contract["alias_reconciliation"]["volume"]["contradictory"], 1)
        self.assertTrue(contract["data_contract_trigger"])
        self.assertEqual(result.evidence["decision"]["route"], "investigate_data_contracts")
        self.assertEqual(self.records[0], original)
        self.assertEqual(result.evidence["rows_removed_or_repaired"], 0)

    def test_supported_exact_geometry_target_heterogeneity_routes_to_better_inputs(self):
        template = dict(self.records[0])
        for index in range(20):
            identity = self.records[index]["_id"]
            self.records[index] = dict(template)
            self.records[index]["_id"] = identity
            target = 1.0 if index % 2 == 0 else 11.0
            self.records[index]["sliced_resin_mass_g"] = target
            self.records[index]["weight"] = target

        result = self.run_audit()

        exact = result.evidence["contract_and_target_audit"]["target_similarity"]["exact_geometry"]
        self.assertTrue(exact["support_sufficient"])
        self.assertTrue(exact["trigger"])
        self.assertEqual(result.evidence["decision"]["route"], "obtain_better_inputs")

    def test_stable_supported_geometry_error_enrichment_routes_to_base_predictor(self):
        class TailErrorRuntime(AuditRuntime):
            def refit(inner, candidate, seed, features, targets, fixed_training_counts):
                fitted = super().refit(candidate, seed, features, targets, fixed_training_counts)
                return LockedFit(
                    lambda values: [float(item[1]) / 1000.0 + (6.0 if float(item[1]) > 120000 else 0.0) for item in values],
                    fitted.preprocessing_state,
                    fitted.artifacts,
                    fitted.metadata,
                )

        with patch.object(audit, "MINIMUM_REGIME_SUPPORT", 5):
            result = self.run_audit(TailErrorRuntime())

        self.assertEqual(result.evidence["decision"]["route"], "improve_base_predictor")
        self.assertIn(
            "mesh_volume:q5",
            result.evidence["decision"]["stable_supported_enriched_geometry_regimes"],
        )

    def test_missing_slicing_attributes_are_unknown_not_contract_contradictions(self):
        for item in self.records:
            item.pop("resin_density_g_per_ml")
            item["slicing_conditions"] = {}

        result = self.run_audit()

        contract = result.evidence["contract_and_target_audit"]
        self.assertFalse(contract["data_contract_trigger"])
        self.assertEqual(contract["missing_required_slicing_attribute_count"], len(self.records))
        self.assertEqual(
            contract["slicing_attribute_coverage"]["resin_density_g_per_ml"]["missing_unknown"],
            len(self.records),
        )
        self.assertEqual(contract["historical_measurement_contract_status"], "unknown_not_a_contradiction")

    def test_malformed_explicit_slicing_conditions_route_to_data_contract_investigation(self):
        self.records[0]["slicing_conditions"] = "not-json"

        result = self.run_audit()

        contract = result.evidence["contract_and_target_audit"]
        self.assertEqual(
            contract["slicing_attribute_coverage"]["slicing_conditions"]["invalid_explicit"],
            1,
        )
        self.assertTrue(contract["data_contract_trigger"])
        self.assertEqual(result.evidence["decision"]["route"], "investigate_data_contracts")

    def test_out_of_float32_frozen_fit_input_blocks_before_any_fit(self):
        self.records[0]["file_size_kib"] = 1e39
        self.records[0]["kb"] = 1e39
        runtime = AuditRuntime()

        result = self.run_audit(runtime)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 0)
        self.assertEqual(runtime.fits, [])
        self.assertEqual(result.evidence["decision"]["route"], "blocked_no_decision")

    def test_out_of_float32_target_blocks_before_any_fit(self):
        self.records[0]["sliced_resin_mass_g"] = 1e39
        self.records[0]["weight"] = 1e39
        runtime = AuditRuntime()

        result = self.run_audit(runtime)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.resource_use["fits_started"], 0)
        self.assertEqual(runtime.fits, [])

    def test_create_only_and_deadline_failure_preserve_bounded_evidence_without_retry(self):
        runtime = AuditRuntime()
        result = self.run_audit(runtime, clock=lambda: 7200.0 if runtime.fits else 0.0)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.blockers, ("training_data_feature_audit_deadline_reached",))
        self.assertEqual(result.resource_use["fits_started"], 1)
        self.assertEqual(len(runtime.fits), 1)
        self.assertEqual(result.evidence["decision"]["route"], "blocked_no_decision")
        snapshot = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaisesRegex(Exception, "private_output_directory_unavailable"):
            self.run_audit()
        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_cli_exposes_only_the_fixed_training_only_arguments_and_checks_before_runtime(self):
        parameters = inspect.signature(audit.run_training_data_feature_audit).parameters
        self.assertNotIn("validation_records", parameters)
        self.assertNotIn("test_records", parameters)
        parser = audit.build_parser()
        self.assertEqual({action.dest for action in parser._actions}, {
            "help", "training_records", "output_root", "volume_unit", "scope_confirmed",
        })
        for flag in ("--validation-records", "--test-records", "--seed", "--maximum-fits", "--threshold", "--lock"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--training-records", "unused", "--output-root", "unused", flag, "unused"])
        with patch.object(audit, "PREDECLARED_OUTPUT_ROOT", self.output), \
             patch.object(stability, "_verify_predeclared_training_artifact") as verify, \
             patch.object(audit.t, "TensorflowXGBoostCandidateRuntime", return_value=AuditRuntime()), \
             patch.object(audit.t, "_verify_predeclared_environment", side_effect=ValueError("blocked")):
            with self.assertRaisesRegex(SystemExit, "training_data_feature_audit_failed"):
                audit.main(["--training-records", "unused", "--output-root", str(self.output), "--scope-confirmed"])
            verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
