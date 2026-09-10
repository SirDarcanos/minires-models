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


if __name__ == "__main__":
    unittest.main()
