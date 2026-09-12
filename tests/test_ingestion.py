import csv
import json
from pathlib import Path
import tempfile
import unittest

from minires import EvaluationConfig, PhysicalBaseline, evaluate_records


class LocalIngestionTests(unittest.TestCase):
    def setUp(self):
        self.config = EvaluationConfig(1.1, "mm3", True, seed=17)

    def test_supported_files_share_precision_preserving_normalization(self):
        row = {"volume": 26671.4841620408, "weight": 26.895716,
               "bbox_area": 208020.653476093, "base_mm": "unknown"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export.jsonl"
            export.write_text(json.dumps({k: {"$numberDouble": str(v)} if isinstance(v, float) else v
                                          for k, v in row.items()}) + "\n")
            tabular = root / "table.csv"
            with tabular.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            before = [p.read_bytes() for p in (export, tabular)]
            results = [evaluate_records(p, self.config, PhysicalBaseline()) for p in (export, tabular)]
            self.assertEqual(before, [p.read_bytes() for p in (export, tabular)])
        self.assertEqual(results[0].normalized_records, results[1].normalized_records)
        for result in results:
            record = result.canonical_rows[0]
            self.assertEqual(record.features["volume_mm3"], 26671.4841620408)
            self.assertEqual(record.features["bounding_box_volume_mm3"], 208020.653476093)
            self.assertEqual(record.sliced_resin_mass_g, 26.895716)
            self.assertEqual(record.outcome, "included")
            self.assertIn("invalid_optional_base_mm", record.warnings)
            self.assertIsNone(record.metadata["geometry_valid"])

    def test_every_row_has_distinct_outcome_and_large_examples_survive(self):
        records = [
            {"volume": 1e12, "weight": 1e9, "base_mm": None},
            {"volume": 0, "weight": 1},
            {"volume": float("inf"), "weight": 1},
            {"volume": 1000, "weight": None},
            {"volume": 1000, "weight": 1, "scope_confirmed": False},
            {"volume": 1000, "weight": 1, "scope_confirmed": None},
            {"volume": 1000, "weight": 1, "resin_density_g_per_ml": None},
            {"volume": 1000, "weight": 1, "slicing_conditions": {"layer_height_mm": 0.05}},
        ]
        result = evaluate_records(records, self.config, PhysicalBaseline())
        self.assertEqual([r.outcome for r in result.canonical_rows],
                         ["included", "excluded", "excluded", "needs_review", "needs_review",
                          "needs_review", "needs_review", "included"])
        self.assertEqual([r.reasons for r in result.canonical_rows[1:7]], [
            ("invalid_volume",), ("non_finite_volume_mm3",),
            ("missing_target_sliced_resin_mass",), ("unsupported_scope",),
            ("scope_confirmation_required",), ("resin_density_required",)])
        self.assertEqual(result.metrics.sample_count, 2)
        self.assertEqual(result.data_quality.excluded_count, 2)
        self.assertEqual(result.data_quality.needs_review_count, 4)
        self.assertEqual(result.canonical_rows[0].features["volume_mm3"], 1e12)
        self.assertEqual(result.canonical_rows[-1].metadata["slicing_conditions"]["layer_height_mm"], 0.05)

    def test_reconciliation_uses_identity_evidence_not_counts_or_equal_features(self):
        primary = [
            {"_id": {"$oid": "record-1"}, "volume": 1234.56789, "weight": 1.23456789},
            {"volume": 1000, "weight": 1},
            {"_id": "repeated", "volume": 2000, "weight": 2},
            {"_id": "repeated", "volume": 2000, "weight": 2},
        ]
        other = [
            {"_id": {"$oid": "record-1"}, "volume": "1234.6", "weight": "1.2"},
            {"volume": 1000, "weight": 1},
            {"_id": "repeated", "volume": 2000, "weight": 2},
            {"_id": "different", "volume": 2000, "weight": 2},
        ]
        result = evaluate_records(primary, self.config, PhysicalBaseline(), reconcile_with=[other])
        report = result.reconciliations[0]
        self.assertFalse(report["dataset_equivalence_proven"])
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["unmatched_primary_indices"], [1])
        self.assertEqual(report["ambiguous_primary_indices"], [2, 3])
        self.assertEqual(report["unmatched_comparison_indices"], [1, 3])
        self.assertEqual(report["ambiguous_comparison_indices"], [2])
        difference = report["matches"][0]
        self.assertEqual(difference["evidence"], "record_identity")
        self.assertEqual(difference["differences"]["volume_mm3"], [1234.56789, 1234.6])
        self.assertEqual(difference["differences"]["sliced_resin_mass_g"], [1.23456789, 1.2])
        self.assertEqual(len(report["comparison_rows"]), 4)

    def test_private_artifacts_are_typed_separated_and_never_overwrite_inputs(self):
        import pyarrow.parquet as pq
        import hashlib
        row = {"volume": 1234.56789012345, "weight": 1.23456789012345,
               "artist": "identity-canary", "file": "/location-canary/model.stl",
               "miniature_family": "family-canary", "base_mm": {"unexpected": "identity-canary"},
               "slicing_conditions": {"profile": "identity-canary", "layer_height_mm": 0.05}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private" / "run"
            result = evaluate_records([row], self.config, PhysicalBaseline(), output_dir=output)
            features = pq.read_table(output / "features.parquet")
            metadata = pq.read_table(output / "evaluation_metadata.parquet")
            self.assertEqual(features.schema.field("volume_mm3").type.bit_width, 64)
            self.assertEqual(features.to_pylist()[0]["volume_mm3"], 1234.56789012345)
            self.assertNotIn("sliced_resin_mass_g", features.column_names)
            self.assertNotIn("anonymous_source_group", features.column_names)
            self.assertIn("anonymous_source_group", metadata.column_names)
            self.assertEqual(metadata.to_pylist()[0]["sliced_resin_mass_g"], 1.23456789012345)
            self.assertIn(b"transformation_version", features.schema.metadata)
            report = (output / "report.json").read_text()
            for canary in ("identity-canary", "location-canary", "family-canary"):
                self.assertNotIn(canary, report)
                self.assertNotIn(canary, str(metadata.to_pylist()))
                self.assertNotIn(canary, str(result.to_dict(public=True)))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["artifacts"]["features.parquet"],
                             hashlib.sha256((output / "features.parquet").read_bytes()).hexdigest())
            before = (output / "report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "private_output_directory_unavailable"):
                evaluate_records([row], self.config, PhysicalBaseline(), output_dir=output)
            self.assertEqual(before, (output / "report.json").read_bytes())

    def test_row_provenance_and_units_override_only_when_explicit(self):
        result = evaluate_records([
            {"volume": "1.23456789", "volume_unit": "cm3", "weight": "1.2",
             "scope_confirmed": "true", "resin_density_g_per_ml": "1.2"},
            {"volume_mm3": 1234.56789, "weight": 1.2, "scope_confirmed": True,
             "resin_density_g_per_ml": 1.2},
        ], EvaluationConfig(None, None, None), PhysicalBaseline())
        self.assertEqual(result.metrics.sample_count, 2)
        self.assertAlmostEqual(result.predictions[0].predicted_sliced_resin_mass_g, 1.481481468)
        self.assertAlmostEqual(result.normalized_records[0].volume_mm3,
                               result.normalized_records[1].volume_mm3, places=10)
        self.assertIsNone(result.canonical_rows[0].metadata["support_presence"])

    def test_malformed_lines_are_accounted_and_fingerprints_cover_rejected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text('{"volume":1000,"weight":1}\nnot-json\nnull\n')
            first = evaluate_records(path, self.config, PhysicalBaseline())
            self.assertEqual(first.data_quality.input_count, 3)
            self.assertEqual(first.data_quality.excluded_count, 2)
            self.assertEqual(first.data_quality.reasons, {"invalid_record": 2})
            path.write_text('{"volume":1000,"weight":1}\ndifferent-bad-json\nnull\n')
            second = evaluate_records(path, self.config, PhysicalBaseline())
            self.assertNotEqual(first.run_metadata.input_fingerprint, second.run_metadata.input_fingerprint)

    def test_nonfinite_predictions_and_configuration_serialize_without_disclosure(self):
        for density, volume, weight in [(1e308, 1e308, 1), (float("nan"), 1000, 1)]:
            result = evaluate_records([{"volume": volume, "weight": weight}],
                                      EvaluationConfig(density, "mm3", True), PhysicalBaseline())
            self.assertEqual(result.metrics.sample_count, 0)
            json.dumps(result.to_dict(), allow_nan=False)
        result = evaluate_records([{"volume": 1000, "weight": 1}],
                                  EvaluationConfig(1.1, "unit-canary", True),
                                  PhysicalBaseline(name="name-canary"))
        self.assertNotIn("canary", json.dumps(result.to_dict()))

    def test_empty_parquet_and_conflicting_identity_evidence_remain_explicit(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private" / "empty"
            result = evaluate_records([], self.config, PhysicalBaseline(), output_dir=output)
            self.assertEqual(result.metrics.sample_count, 0)
            self.assertIsNone(result.metrics.mae_g)
            self.assertEqual(pq.read_table(output / "features.parquet").num_rows, 0)
        result = evaluate_records([
            {"_id": "a", "file": "same-location", "volume": 1000, "weight": 1}
        ], self.config, PhysicalBaseline(), reconcile_with=[[
            {"_id": "b", "file": "same-location", "volume": 1000, "weight": 1}
        ]])
        self.assertEqual(result.reconciliations[0]["matched_count"], 0)
        self.assertEqual(result.reconciliations[0]["ambiguous_primary_indices"], [0])
        self.assertEqual(result.reconciliations[0]["ambiguous_comparison_indices"], [0])

    def test_jsonl_unicode_separators_inside_strings_do_not_create_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            row = {"volume": 1000, "weight": 1, "ignored": "canary\u2028text\u2029end"}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n\n")
            result = evaluate_records(path, self.config, PhysicalBaseline())
        self.assertEqual(result.data_quality.input_count, 2)
        self.assertEqual(result.data_quality.accepted_count, 1)
        self.assertEqual(result.data_quality.excluded_count, 1)

    def test_one_sided_repeated_ids_cannot_be_resolved_by_weaker_keys(self):
        result = evaluate_records([
            {"_id": "repeat", "file": "location-a", "volume": 1000, "weight": 1},
            {"_id": "repeat", "file": "location-b", "volume": 1000, "weight": 1},
        ], self.config, PhysicalBaseline(), reconcile_with=[[
            {"file": "location-a", "volume": 1000, "weight": 1},
        ]])
        report = result.reconciliations[0]
        self.assertEqual(report["matched_count"], 0)
        self.assertEqual(report["ambiguous_primary_indices"], [0, 1])
        self.assertEqual(report["ambiguous_comparison_indices"], [0])
        self.assertEqual(report["unmatched_primary_indices"], [])

    def test_parquet_is_private_during_writing_even_with_permissive_umask(self):
        import os
        import stat
        import pyarrow.parquet as pq
        from unittest.mock import patch
        original_write = pq.write_table
        observed_modes = []

        def observe_write(table, destination, *args, **kwargs):
            original_write(table, destination, *args, **kwargs)
            info = os.fstat(destination.fileno()) if hasattr(destination, "fileno") else Path(destination).stat()
            observed_modes.append(stat.S_IMODE(info.st_mode))

        with tempfile.TemporaryDirectory() as directory:
            old_umask = os.umask(0)
            try:
                with patch("pyarrow.parquet.write_table", side_effect=observe_write):
                    evaluate_records([{"volume": 1000, "weight": 1}], self.config, PhysicalBaseline(),
                                     output_dir=Path(directory) / "private" / "run")
            finally:
                os.umask(old_umask)
        self.assertEqual(observed_modes, [0o600, 0o600])


if __name__ == "__main__":
    unittest.main()
