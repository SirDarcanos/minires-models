from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires_evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records
from minires_evaluation.prepare_one import main as prepare_one_main
from minires_evaluation.stl_preparation import (
    EBMINIMANAGER_REVISION,
    PROFILE_RELATIVE_PATH,
    PROFILE_SHA256,
    ProcessResult,
    SubprocessRunner,
    prepare_stl,
)


PROFILE = """material_density = 1.1
layer_height = 0.05
supports_enable = 0
"""


class FakeRunner:
    def __init__(self, failure=None, source=None):
        self.failure = failure
        self.source = source
        self.calls = []
        self.generated_path = None

    def run(self, args, *, timeout_s, cwd=None):
        args = tuple(str(value) for value in args)
        self.calls.append((args, timeout_s, cwd))
        command = args[0]
        if command == "git":
            return ProcessResult(0, EBMINIMANAGER_REVISION + "\n", "")
        if any(value.endswith(".stl_probe") for value in args):
            if "--version" in args:
                if self.failure == "missing_geometry_dependency":
                    return ProcessResult(1, "", "dependency unavailable")
                return ProcessResult(0, "trimesh 4.10.1\n", "")
            if self.failure == "geometry_timeout":
                raise TimeoutError
            if self.failure == "corrupt_geometry":
                return ProcessResult(2, "", "corrupt")
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
            if self.failure == "invalid_measurements":
                measurements["volume"] = 0
            if self.failure == "fractional_euler_number":
                measurements["euler_number"] = 1.5
            return ProcessResult(0, json.dumps(measurements), "")
        if command == "prusa-slicer" and "--version" in args:
            if self.failure == "missing_prusaslicer":
                raise FileNotFoundError
            if self.failure == "malformed_prusaslicer_version":
                return ProcessResult(0, "available\n", "")
            return ProcessResult(0, "PrusaSlicer 2.6.0\n", "")
        if command == "UVtoolsCmd" and "--version" in args:
            if self.failure == "missing_uvtools":
                raise FileNotFoundError
            return ProcessResult(0, "UVtoolsCmd 4.0.0\n", "")
        if command == "prusa-slicer":
            if self.failure == "slicing_timeout":
                raise TimeoutError
            output = Path(args[args.index("--output") + 1])
            self.generated_path = output
            if self.failure != "sliced_output_absent":
                output.write_bytes(b"private sliced output")
            if self.failure == "source_checksum_changed":
                self.source.write_bytes(b"changed")
            if self.failure == "slicing_failed":
                return ProcessResult(7, "", "failed")
            return ProcessResult(0, "", "")
        if command == "UVtoolsCmd":
            if self.failure == "uvtools_timeout":
                raise TimeoutError
            if self.failure == "uvtools_failed":
                return ProcessResult(3, "", "failed")
            if self.failure == "weight_g_missing":
                return ProcessResult(0, "LayerCount: 10\n", "")
            if self.failure == "invalid_weight_g":
                return ProcessResult(0, "WeightG: 0\n", "")
            return ProcessResult(0, "WeightG: 1.23456789012345\n", "")
        raise AssertionError(args)


class StlPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        profile = self.checkout / PROFILE_RELATIVE_PATH
        profile.parent.mkdir(parents=True)
        profile.write_text(PROFILE)
        self.stl = self.root / "private-source-canary.stl"
        self.stl.write_bytes(b"solid private source canary")

    def prepare(self, runner):
        profile_digest = sha256((self.checkout / PROFILE_RELATIVE_PATH).read_bytes()).hexdigest()
        with patch("minires_evaluation.stl_preparation.PROFILE_SHA256", profile_digest):
            return prepare_stl(
                self.stl,
                ebminimanager_dir=self.checkout,
                runner=runner,
                scope_confirmed=True,
            )

    def test_success_produces_full_precision_legacy_record_and_cleans_workspace(self):
        runner = FakeRunner()
        original = self.stl.read_bytes()

        result = self.prepare(runner)

        self.assertEqual(result.outcome, "prepared")
        self.assertIsNone(result.rejection)
        self.assertEqual(result.record["sliced_resin_mass_g"], 1.23456789012345)
        self.assertEqual(result.record["volume"], 1234.567890123456)
        self.assertEqual(result.record["volume_mm3"], 1234.567890123456)
        self.assertEqual(result.record["bounding_box_volume_mm3"], 6181.806)
        self.assertEqual(result.record["surface_to_volume_ratio_per_mm"], 0.37000000000000005)
        self.assertEqual(
            tuple(result.record[field] for field in (
                "kb", "volume", "surface_area", "bbox_area",
                "euler_number", "scale", "surface_volume_ratio",
            )),
            (
                len(original) / 1024,
                1234.567890123456,
                456.789012345678,
                6181.806,
                2,
                37.78888725538237,
                0.37000000000000005,
            ),
        )
        self.assertEqual(result.contract["ebminimanager_revision"], EBMINIMANAGER_REVISION)
        self.assertEqual(result.contract["resin_density_g_per_ml"], 1.1)
        self.assertEqual(result.contract["layer_height_mm"], 0.05)
        self.assertFalse(result.contract["slicer_added_supports"])
        self.assertEqual(result.versions["prusaslicer"], "PrusaSlicer 2.6.0")
        self.assertIn("python", result.versions)
        self.assertEqual(
            result.record["sliced_output_inventory"]["sha256"],
            sha256(b"private sliced output").hexdigest(),
        )
        self.assertEqual(self.stl.read_bytes(), original)
        self.assertFalse(runner.generated_path.exists())
        self.assertFalse(runner.generated_path.parent.exists())
        self.assertTrue(all(call[1] > 0 for call in runner.calls))
        self.assertTrue(all(isinstance(call[0], tuple) for call in runner.calls))
        evaluated = evaluate_records(
            [result.record],
            EvaluationConfig(1.1, "mm3", True),
            PhysicalBaseline(),
        )
        self.assertEqual(evaluated.data_quality.accepted_count, 1)

    def test_pinned_profile_checksum_is_verified(self):
        result = prepare_stl(
            self.stl,
            ebminimanager_dir=self.checkout,
            runner=FakeRunner(),
            scope_confirmed=True,
        )
        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(result.rejection, "profile_checksum_mismatch")
        self.assertEqual(PROFILE_SHA256, "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e")

    def test_preflight_checks_every_dependency_before_geometry_or_slicing(self):
        for failure in ("missing_prusaslicer", "missing_uvtools", "missing_geometry_dependency"):
            with self.subTest(failure=failure):
                runner = FakeRunner(failure)
                result = self.prepare(runner)
                self.assertEqual(result.rejection, failure)
                self.assertFalse(any("--probe" in call[0] for call in runner.calls))
                self.assertFalse(any(call[0][0] == "prusa-slicer" and "--sla" in call[0]
                                     for call in runner.calls))

    def test_each_processing_failure_has_a_distinct_bounded_rejection(self):
        failures = (
            "geometry_timeout",
            "corrupt_geometry",
            "invalid_measurements",
            "slicing_failed",
            "slicing_timeout",
            "sliced_output_absent",
            "uvtools_timeout",
            "uvtools_failed",
            "weight_g_missing",
            "invalid_weight_g",
        )
        original = self.stl.read_bytes()
        for failure in failures:
            with self.subTest(failure=failure):
                runner = FakeRunner(failure)
                result = self.prepare(runner)
                self.assertEqual(result.outcome, "rejected")
                self.assertEqual(result.rejection, failure)
                self.assertIsNone(result.record)
                self.assertEqual(self.stl.read_bytes(), original)
                if runner.generated_path is not None:
                    self.assertFalse(runner.generated_path.parent.exists())

    def test_fractional_euler_number_is_not_a_compatible_record(self):
        result = self.prepare(FakeRunner("fractional_euler_number"))
        self.assertEqual(result.rejection, "invalid_measurements")
        self.assertIsNone(result.record)

    def test_scope_confirmation_and_version_output_fail_closed(self):
        result = prepare_stl(
            self.stl,
            ebminimanager_dir=self.checkout,
            runner=FakeRunner(),
        )
        self.assertEqual(result.rejection, "scope_confirmation_required")
        result = self.prepare(FakeRunner("malformed_prusaslicer_version"))
        self.assertEqual(result.rejection, "prusaslicer_version_unavailable")

    def test_workspace_creation_failure_is_bounded(self):
        with patch("minires_evaluation.stl_preparation.tempfile.mkdtemp", side_effect=OSError):
            result = self.prepare(FakeRunner())
        self.assertEqual(result.rejection, "workspace_failure")
        self.assertEqual(self.stl.read_bytes(), b"solid private source canary")

    def test_detects_source_change_without_returning_a_record(self):
        result = self.prepare(FakeRunner("source_checksum_changed", self.stl))
        self.assertEqual(result.rejection, "source_checksum_changed")
        self.assertIsNone(result.record)

    def test_command_output_is_bounded_and_identity_free(self):
        result = self.prepare(FakeRunner())
        private_dir = self.root / "private"
        private_dir.mkdir()
        output = private_dir / "result.json"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("minires_evaluation.prepare_one.prepare_stl", return_value=result):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = prepare_one_main([
                    "--stl", str(self.stl),
                    "--ebminimanager-dir", str(self.checkout),
                    "--private-output", str(output),
                    "--scope-confirmed",
                ])

        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue(), "prepared\n")
        self.assertEqual(stderr.getvalue(), "")
        command_output = stdout.getvalue() + stderr.getvalue()
        for private_value in (
            self.stl.name,
            str(self.stl),
            result.record["input_inventory"]["sha256"],
            str(result.record["volume"]),
            str(result.record["sliced_resin_mass_g"]),
        ):
            self.assertNotIn(private_value, command_output)
        persisted = json.loads(output.read_text())
        self.assertEqual(persisted["record"]["sliced_resin_mass_g"], 1.23456789012345)

    def test_default_runner_never_uses_shell_interpolation(self):
        completed = __import__("subprocess").CompletedProcess([], 0, "ok", "")
        with patch("minires_evaluation.stl_preparation.subprocess.run", return_value=completed) as run:
            result = SubprocessRunner().run(("tool", "$(private-canary)"), timeout_s=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_args.args[0], ["tool", "$(private-canary)"])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 1)

    def test_profile_contract_and_revision_fail_closed(self):
        runner = FakeRunner()
        profile = self.checkout / PROFILE_RELATIVE_PATH
        profile.write_text("material_density = 1.0\nlayer_height = 0.05\nsupports_enable = 0\n")
        result = self.prepare(runner)
        self.assertEqual(result.rejection, "profile_contract_mismatch")

        profile.write_text(PROFILE)
        runner = FakeRunner()
        original_run = runner.run
        runner.run = lambda args, **kwargs: (
            ProcessResult(0, "wrong-revision\n", "")
            if args[0] == "git" else original_run(args, **kwargs)
        )
        result = self.prepare(runner)
        self.assertEqual(result.rejection, "ebminimanager_revision_mismatch")


if __name__ == "__main__":
    unittest.main()
