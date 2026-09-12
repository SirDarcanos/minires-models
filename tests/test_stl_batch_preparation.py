from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from minires import EvaluationConfig, PhysicalBaseline, evaluate_records
from minires.preparation.prepare_batch import main as prepare_batch_main
from minires.preparation.stl import ProcessResult


class ScenarioRunner:
    """Fake only the external process boundary; the real batch command does the rest."""

    def __init__(
        self, *, failures=None, version="2.9.6", interrupt_digest=None,
        mutate_sources=None, delay=0.0,
    ):
        self.failures = failures or {}
        self.version = version
        self.interrupt_digest = interrupt_digest
        self.mutate_sources = mutate_sources or {}
        self.delay = delay
        self.events = []
        self.generated = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self._workspace_digests = {}

    def _digest(self, args, cwd):
        stls = [Path(value) for value in args if str(value).endswith(".stl")]
        if stls:
            digest = sha256(stls[0].read_bytes()).hexdigest()
            self._workspace_digests[str(stls[0].parent)] = digest
            return digest
        return self._workspace_digests[str(cwd)]

    def run(self, args, *, timeout_s, cwd=None):
        args = tuple(str(value) for value in args)
        if args[:2] == ("prusa-slicer", "--help"):
            return ProcessResult(
                0, f"PrusaSlicer-{self.version} based on Slic3r (with GUI support)\n", ""
            )
        if args[:2] == ("UVtoolsCmd", "--core-version"):
            return ProcessResult(0, "6.2.0\n", "")
        if "--version" in args:
            return ProcessResult(0, "trimesh 4.10.1\n", "")
        digest = self._digest(args, cwd)
        phase = "probe" if "--probe" in args else "slice" if args[0] == "prusa-slicer" else "properties"
        with self._lock:
            self.events.append((digest, phase))
        if phase == "probe":
            if digest == self.interrupt_digest:
                raise KeyboardInterrupt
            if digest in self.mutate_sources:
                self.mutate_sources[digest].write_bytes(b"changed externally")
            if self.failures.get(digest) == "corrupt_geometry":
                return ProcessResult(2, "", "private tool diagnostic")
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self.active -= 1
            measurements = {
                "volume": 1234.567890123456,
                "surface_area": 456.789012345678,
                "bbox_x": 10.1,
                "bbox_y": 20.2,
                "bbox_z": 30.3,
                "bbox_area": 6181.806,
                "mass": 1234.567890123456,
                "euler_number": 2,
                "scale": 37.78888725538237,
                "surface_volume_ratio": 0.37000000000000005,
            }
            return ProcessResult(0, json.dumps(measurements), "")
        if phase == "slice":
            output = Path(args[args.index("--output") + 1])
            output.write_bytes(b"private sliced output")
            self.generated.append(output)
            return ProcessResult(0, "", "")
        return ProcessResult(
            1,
            "Opening file sliced-output.pwmx:\n"
            "Done in 0.44s\n"
            "-------------------------\n"
            "HeaderSettings: TableName: HEADER, TableLength: 80, "
            "LayerHeight: 0.05, WeightG: 1.23456789012345, Price: 0.007\n"
            "FileType: Binary\n"
            "ManufacturingProcess: mSLA\n",
            "",
        )


class StlBatchCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "source-canary"
        self.inputs.mkdir()
        self.private = self.root / "private" / "batch"
        self.output = self.private / "result.json"
        self.checkpoints = self.private / "checkpoints"

    def add_stl(self, name, content):
        path = self.inputs / name
        path.write_bytes(content)
        return path

    def run_command(self, runner, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = [
            "--input-directory", str(self.inputs),
            "--private-output", str(self.output),
            "--private-checkpoints", str(self.checkpoints),
            "--scope-confirmed",
            *extra,
        ]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = prepare_batch_main(argv, runner=runner)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_successful_batch_persists_inventory_schema_accounting_and_provenance(self):
        originals = {
            self.add_stl("private-a.stl", b"solid a"): b"solid a",
            self.add_stl("private-b.STL", b"solid b"): b"solid b",
        }
        runner = ScenarioRunner()

        status, stdout, stderr = self.run_command(runner)

        self.assertEqual((status, stdout, stderr), (0, "completed\n", ""))
        report = json.loads(self.output.read_text())
        self.assertEqual(report["outcome"], "completed")
        self.assertEqual(report["accounting"]["inventory_count"], 2)
        self.assertEqual(report["accounting"]["inventory_count"],
                         report["accounting"]["accepted_count"] + report["accounting"]["rejected_count"])
        self.assertEqual(len(report["inventory"]), 2)
        self.assertTrue(all(item["byte_count"] > 0 and len(item["sha256"]) == 64
                            for item in report["inventory"]))
        self.assertEqual(len(report["accepted_records"]), 2)
        record = report["accepted_records"][0]["record"]
        self.assertEqual(record["volume_mm3"], 1234.567890123456)
        self.assertEqual(record["volume"], 1234.567890123456)
        self.assertEqual(record["weight"], 1.23456789012345)
        self.assertEqual(evaluate_records([record], EvaluationConfig(1.1, "mm3", True),
                                          PhysicalBaseline()).data_quality.accepted_count, 1)
        self.assertEqual(report["provenance"]["worker_count"], 1)
        self.assertIn("contract_fingerprint", report["provenance"])
        checkpoint_payloads = [
            json.loads(path.read_text()) for path in self.checkpoints.glob("*.json")
        ]
        self.assertEqual(len(checkpoint_payloads), 2)
        self.assertTrue(all(payload["complete"] for payload in checkpoint_payloads))
        self.assertTrue(all(
            payload["contract_fingerprint"] == report["provenance"]["contract_fingerprint"]
            and len(payload["input_sha256"]) == 64
            for payload in checkpoint_payloads
        ))
        self.assertEqual(set(report["provenance"]["tool_versions"]),
                         {"geometry", "prusaslicer", "python", "uvtools"})
        self.assertGreaterEqual(report["resources"]["elapsed_seconds"], 0)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        self.assertTrue(all(not path.exists() for path in runner.generated))
        self.assertFalse(any(self.inputs.glob("*.pwmx")))

    def test_mixed_failures_and_unsupported_entries_are_bounded_and_reconciled(self):
        good = self.add_stl("good.stl", b"good")
        bad = self.add_stl("bad.stl", b"bad")
        self.add_stl("empty.stl", b"")
        self.add_stl("notes.txt", b"not an stl")
        linked_directory = self.root / "linked-directory"
        linked_directory.mkdir()
        (self.inputs / "directory-link").symlink_to(linked_directory, target_is_directory=True)
        bad_digest = sha256(bad.read_bytes()).hexdigest()

        status, stdout, stderr = self.run_command(
            ScenarioRunner(failures={bad_digest: "corrupt_geometry"})
        )

        self.assertEqual((status, stdout, stderr), (0, "completed\n", ""))
        report = json.loads(self.output.read_text())
        self.assertEqual(report["accounting"]["inventory_count"], 5)
        self.assertEqual(report["accounting"]["accepted_count"], 1)
        self.assertEqual(report["accounting"]["rejected_count"], 4)
        self.assertEqual(report["accounting"]["rejection_reasons"], {
            "corrupt_geometry": 1, "unsupported_inventory_entry": 2, "zero_length": 1,
        })
        self.assertEqual(good.read_bytes(), b"good")
        self.assertNotIn("private tool diagnostic", stdout + stderr)
        self.assertTrue(all(item["reason"] in {
            "corrupt_geometry", "unsupported_inventory_entry", "zero_length"
        } for item in report["rejected_entries"]))

    def test_source_change_during_processing_is_a_bounded_rejection(self):
        source = self.add_stl("changing.stl", b"original")
        digest = sha256(source.read_bytes()).hexdigest()

        status, stdout, stderr = self.run_command(
            ScenarioRunner(mutate_sources={digest: source})
        )

        self.assertEqual((status, stdout, stderr), (0, "completed\n", ""))
        report = json.loads(self.output.read_text())
        self.assertEqual(report["accounting"]["rejection_reasons"],
                         {"source_checksum_changed": 1})
        self.assertEqual(report["accepted_records"], [])

    def test_matching_checkpoints_resume_without_reprocessing_or_duplicate_rows(self):
        self.add_stl("a.stl", b"a")
        first = ScenarioRunner()
        self.assertEqual(self.run_command(first)[0], 0)
        first_report = json.loads(self.output.read_text())
        second = ScenarioRunner()

        self.assertEqual(self.run_command(second)[0], 0)

        report = json.loads(self.output.read_text())
        self.assertEqual(second.events, [])
        self.assertEqual(report["accounting"]["reused_unique_outcomes"], 1)
        self.assertEqual(len(report["accepted_records"]), 1)
        self.assertEqual(report["accepted_records"], first_report["accepted_records"])

        checkpoint = next(self.checkpoints.glob("*.json"))
        incomplete = json.loads(checkpoint.read_text())
        incomplete["complete"] = False
        checkpoint.write_text(json.dumps(incomplete))
        recomputed = ScenarioRunner()
        self.assertEqual(self.run_command(recomputed)[0], 0)
        self.assertTrue(recomputed.events)

    def test_changed_input_and_contract_invalidate_stale_checkpoints(self):
        path = self.add_stl("a.stl", b"old")
        self.assertEqual(self.run_command(ScenarioRunner(version="2.9.6"))[0], 0)
        path.write_bytes(b"new")
        changed_input = ScenarioRunner(version="2.9.6")
        self.assertEqual(self.run_command(changed_input)[0], 0)
        self.assertTrue(changed_input.events)
        changed_contract = ScenarioRunner(version="2.9.7")
        self.assertEqual(self.run_command(changed_contract)[0], 0)
        self.assertTrue(changed_contract.events)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["accepted_records"][0]["input_sha256"], sha256(b"new").hexdigest())

    def test_duplicates_share_evidence_and_are_processed_once(self):
        self.add_stl("copy-a.stl", b"same")
        self.add_stl("copy-b.stl", b"same")
        runner = ScenarioRunner()
        self.assertEqual(self.run_command(runner)[0], 0)
        report = json.loads(self.output.read_text())
        digest = sha256(b"same").hexdigest()
        self.assertEqual(sum(event == (digest, "probe") for event in runner.events), 1)
        self.assertEqual(report["accounting"]["inventory_count"], 2)
        self.assertEqual(report["accounting"]["accepted_count"], 2)
        groups = {entry["duplicate_group"] for entry in report["accepted_records"]}
        self.assertEqual(len(groups), 1)
        self.assertNotIn(None, groups)

    def test_interruption_keeps_completed_checkpoint_and_resume_is_exactly_once(self):
        self.add_stl("a.stl", b"a")
        self.add_stl("b.stl", b"b")
        digest_b = sha256(b"b").hexdigest()
        interrupted = ScenarioRunner(interrupt_digest=digest_b)

        status, stdout, stderr = self.run_command(interrupted)

        self.assertEqual((status, stdout, stderr), (2, "", "interrupted\n"))
        self.assertFalse(self.output.exists())
        resumed = ScenarioRunner()
        self.assertEqual(self.run_command(resumed)[0], 0)
        report = json.loads(self.output.read_text())
        self.assertEqual(len(report["accepted_records"]), 2)
        self.assertNotIn((sha256(b"a").hexdigest(), "probe"), resumed.events)
        self.assertIn((digest_b, "probe"), resumed.events)

    def test_worker_bounds_default_and_explicit_concurrency(self):
        for index in range(3):
            self.add_stl(f"{index}.stl", f"mesh-{index}".encode())
        sequential = ScenarioRunner(delay=0.02)
        self.assertEqual(self.run_command(sequential)[0], 0)
        self.assertEqual(sequential.max_active, 1)
        # A changed contract forces fresh work with the explicitly bounded worker count.
        concurrent = ScenarioRunner(version="2.9.7", delay=0.05)
        self.assertEqual(self.run_command(concurrent, "--workers", "2")[0], 0)
        self.assertEqual(concurrent.max_active, 2)
        self.assertLessEqual(concurrent.max_active, 2)
        for invalid in ("0", "5"):
            fresh_output = self.private / f"invalid-{invalid}.json"
            self.output = fresh_output
            runner = ScenarioRunner()
            status, stdout, stderr = self.run_command(runner, "--workers", invalid)
            self.assertEqual(status, 2)
            self.assertEqual(stdout, "")
            self.assertEqual(stderr, "invalid_worker_count\n")
            self.assertEqual(runner.events, [])

    def test_command_output_is_private_and_non_private_destinations_fail_early(self):
        source = self.add_stl("filename-canary.stl", b"content-canary")
        digest = sha256(source.read_bytes()).hexdigest()
        status, stdout, stderr = self.run_command(ScenarioRunner())
        self.assertEqual(status, 0)
        for canary in (source.name, str(source), digest, "1234.567890123456", "1.23456789012345"):
            self.assertNotIn(canary, stdout + stderr)
        self.output = Path.cwd() / "public-batch-canary.json"
        runner = ScenarioRunner()
        status, stdout, stderr = self.run_command(runner)
        self.assertEqual((status, stdout, stderr), (2, "", "private_output_required\n"))
        self.assertEqual(runner.events, [])


if __name__ == "__main__":
    unittest.main()
