import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandLineInterfaceTests(unittest.TestCase):
    def test_runs_synthetic_records_without_network_and_can_emit_public_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.json"
            output_path = Path(directory) / "summary.json"
            records_path.write_text(
                json.dumps(
                    [
                        {
                            "volume": 1000,
                            "weight": 1.1,
                            "artist": "identifying-canary",
                            "file": "/private/canary.stl",
                        }
                    ]
                )
            )
            command = [
                sys.executable,
                "-m",
                "minires_evaluation",
                "--records",
                str(records_path),
                "--density-g-per-ml",
                "1.1",
                "--volume-unit",
                "mm3",
                "--scope-confirmed",
                "--public",
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            summary = json.loads(output_path.read_text())

        self.assertEqual(completed.stdout, "")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["metrics"]["sample_count"], 1)
        self.assertNotIn("normalized_records", summary)
        self.assertNotIn("identifying-canary", json.dumps(summary))
        self.assertNotIn("/private", json.dumps(summary))

    def test_ingests_jsonl_and_csv_and_writes_private_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export.jsonl"
            export.write_text('{"volume":{"$numberDouble":"1000"},"weight":1.1}\n')
            table = root / "table.csv"
            table.write_text("volume,weight\n1000,1.1\n")
            output = root / "private" / "run"
            completed = subprocess.run([
                sys.executable, "-m", "minires_evaluation", "--records", str(export),
                "--reconcile", str(table), "--private-dir", str(output),
                "--volume-unit", "mm3"], text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["data_quality"]["input_count"], 1)
            self.assertTrue((output / "features.parquet").exists())
            self.assertEqual(report["reconciliations"][0]["comparison_count"], 1)

    def test_errors_never_echo_input_paths_or_values(self):
        completed = subprocess.run([
            sys.executable, "-m", "minires_evaluation", "--records", "/private/location-canary.json"],
            text=True, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("location-canary", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_legacy_mode_reports_missing_pinned_artifacts_without_echoing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.json"
            canary_artifacts = root / "private-artifact-canary"
            records.write_text(json.dumps([{
                "kb": 1, "volume": 1000, "surface_area": 10, "bbox_area": 20,
                "euler_number": 1, "scale": 30, "weight": 2,
            }]))
            completed = subprocess.run([
                sys.executable, "-m", "minires_evaluation", "--records", str(records),
                "--volume-unit", "mm3", "--scope-confirmed", "--legacy-artifacts",
                str(canary_artifacts), "--public",
            ], text=True, capture_output=True, check=True)
            report = json.loads(completed.stdout)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["provenance_classification"], "legacy_reference_training_provenance_unknown")
        self.assertIn("legacy_artifact_missing_minires_keras", report["blockers"])
        self.assertNotIn("private-artifact-canary", completed.stdout + completed.stderr)

    def test_json_is_private_at_creation_without_post_write_chmod(self):
        import os
        import stat
        from unittest.mock import patch
        from minires_evaluation.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.json"
            records.write_text('[{"volume":1000,"weight":1}]')
            private = root / "private"
            private.mkdir()
            output = private / "report.json"
            old_umask = os.umask(0)
            try:
                with patch("pathlib.Path.chmod", side_effect=PermissionError):
                    self.assertEqual(main(["--records", str(records), "--output", str(output)]), 0)
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            finally:
                os.umask(old_umask)


if __name__ == "__main__":
    unittest.main()
