from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minires import EvaluationConfig, PhysicalBaseline, evaluate_records
from minires.preparation.diagnose_toolchain import main as diagnose_toolchain_main
from minires.preparation.prepare_one import main as prepare_one_main
from minires.preparation.slicing_contract import BUNDLED_PROFILE_RESOURCE
from minires.preparation.stl import (
    PROFILE_SHA256,
    ProcessResult,
    SubprocessRunner,
    prepare_stl,
)


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
        if command == "prusa-slicer" and "--help" in args:
            if self.failure == "missing_prusaslicer":
                raise FileNotFoundError
            if self.failure == "prusaslicer_timeout":
                raise TimeoutError
            if self.failure == "malformed_prusaslicer_version":
                return ProcessResult(0, "PrusaSlicer available\n", "")
            if self.failure == "unsupported_prusaslicer_version":
                return ProcessResult(0, "PrusaSlicer-2.8.1 based on Slic3r\n", "")
            if self.failure == "contaminated_prusaslicer_version":
                return ProcessResult(
                    0, "Error: fallback output\nPrusaSlicer-2.9.6 based on Slic3r\n", ""
                )
            return ProcessResult(
                0,
                "PrusaSlicer-2.9.6 based on Slic3r (with GUI support)\n"
                "Usage: prusa-slicer [ INPUT ] [ OPTIONS ]\n",
                "",
            )
        if command == "UVtoolsCmd" and "--core-version" in args:
            if self.failure == "missing_uvtools":
                raise FileNotFoundError
            if self.failure == "uvtools_version_timeout":
                raise TimeoutError
            if self.failure == "malformed_uvtools_version":
                return ProcessResult(0, "UVtools core available\n", "")
            if self.failure == "unsupported_uvtools_version":
                return ProcessResult(0, "6.1.0\n", "")
            if self.failure == "contaminated_uvtools_version":
                return ProcessResult(0, "Error: fallback output\n6.2.0\n", "")
            return ProcessResult(0, "6.2.0\n", "")
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
            if self.failure == "uvtools_missing_file":
                return ProcessResult(1, "Description: MSLA/DLP file analysis\nUsage: UVtoolsCmd\n", "")
            header = (
                "HeaderSettings: TableName: HEADER, TableLength: 80, "
                "LayerHeight: 0.05, WeightG: 1.23456789012345, Price: 0.007\n"
            )
            if self.failure == "weight_g_missing":
                header = (
                    "HeaderSettings: TableName: HEADER, TableLength: 80, "
                    "LayerHeight: 0.05, Price: 0.007\n"
                )
            if self.failure == "similarly_named_weight":
                header = (
                    "HeaderSettings: TableName: HEADER, TableLength: 80, "
                    "LayerHeight: 0.05, DryWeightG: 1.2, Price: 0.007\n"
                )
            if self.failure == "invalid_weight_g":
                header = header.replace("1.23456789012345", "0")
            if self.failure == "non_finite_weight_g":
                header = header.replace("1.23456789012345", "NaN")
            if self.failure == "invalid_weight_syntax":
                header = header.replace("1.23456789012345", "1_2")
            if self.failure == "malformed_properties":
                return ProcessResult(1, header, "")
            output = (
                "Opening file sliced-output.pwmx:\n"
                "Done in 0.44s\n"
                "-------------------------\n"
                f"{header}"
                "FileType: Binary\n"
                "ManufacturingProcess: mSLA\n"
            )
            if self.failure == "help_contaminated_properties":
                output += "Usage: UVtoolsCmd [command] [options]\n"
            stderr = "Error: unexpected diagnostic\n" if self.failure == "stderr_properties" else ""
            return ProcessResult(1, output, stderr)
        raise AssertionError(args)


class StlPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / "config-anycubic-mono.ini"
        self.profile.write_bytes(BUNDLED_PROFILE_RESOURCE.read_bytes())
        self.stl = self.root / "private-source-canary.stl"
        self.stl.write_bytes(b"solid private source canary")

    def prepare(self, runner):
        return prepare_stl(
            self.stl,
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
        self.assertEqual(result.contract["profile_sha256"], PROFILE_SHA256)
        self.assertEqual(result.contract["resin_density_g_per_ml"], 1.1)
        self.assertEqual(result.contract["layer_height_mm"], 0.05)
        self.assertFalse(result.contract["slicer_added_supports"])
        self.assertEqual(
            result.versions["prusaslicer"],
            "PrusaSlicer-2.9.6 based on Slic3r (with GUI support)",
        )
        self.assertEqual(result.versions["uvtools"], "6.2.0")
        self.assertIn("python", result.versions)
        self.assertIn((("prusa-slicer", "--help"), 120.0, None), runner.calls)
        self.assertIn((("UVtoolsCmd", "--core-version"), 120.0, None), runner.calls)
        self.assertEqual(
            result.record["sliced_output_inventory"]["sha256"],
            sha256(b"private sliced output").hexdigest(),
        )
        self.assertEqual(self.stl.read_bytes(), original)
        self.assertFalse(runner.generated_path.exists())
        self.assertFalse(runner.generated_path.parent.exists())
        self.assertTrue(all(call[1] > 0 for call in runner.calls))
        self.assertTrue(all(isinstance(call[0], tuple) for call in runner.calls))
        self.assertFalse(any(call[0][0] == "git" for call in runner.calls))
        evaluated = evaluate_records(
            [result.record],
            EvaluationConfig(1.1, "mm3", True),
            PhysicalBaseline(),
        )
        self.assertEqual(evaluated.data_quality.accepted_count, 1)

    def test_pinned_profile_checksum_is_verified(self):
        self.profile.write_text("modified profile")
        with patch("minires.preparation.stl.BUNDLED_PROFILE_RESOURCE", self.profile):
            result = prepare_stl(
                self.stl,
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
            "uvtools_missing_file",
            "weight_g_missing",
            "similarly_named_weight",
            "invalid_weight_g",
            "non_finite_weight_g",
            "invalid_weight_syntax",
            "malformed_properties",
            "help_contaminated_properties",
            "stderr_properties",
        )
        original = self.stl.read_bytes()
        for failure in failures:
            with self.subTest(failure=failure):
                runner = FakeRunner(failure)
                result = self.prepare(runner)
                self.assertEqual(result.outcome, "rejected")
                expected = {
                    "uvtools_missing_file": "uvtools_failed",
                    "similarly_named_weight": "weight_g_missing",
                    "non_finite_weight_g": "invalid_weight_g",
                    "invalid_weight_syntax": "invalid_weight_g",
                    "malformed_properties": "uvtools_failed",
                    "help_contaminated_properties": "uvtools_failed",
                    "stderr_properties": "uvtools_failed",
                }.get(failure, failure)
                self.assertEqual(result.rejection, expected)
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
            runner=FakeRunner(),
        )
        self.assertEqual(result.rejection, "scope_confirmation_required")
        cases = {
            "prusaslicer_timeout": "missing_prusaslicer",
            "malformed_prusaslicer_version": "prusaslicer_version_unavailable",
            "unsupported_prusaslicer_version": "unsupported_prusaslicer_version",
            "contaminated_prusaslicer_version": "prusaslicer_version_unavailable",
            "uvtools_version_timeout": "missing_uvtools",
            "malformed_uvtools_version": "uvtools_version_unavailable",
            "unsupported_uvtools_version": "unsupported_uvtools_version",
            "contaminated_uvtools_version": "uvtools_version_unavailable",
        }
        for failure, expected in cases.items():
            with self.subTest(failure=failure):
                result = self.prepare(FakeRunner(failure))
                self.assertEqual(result.rejection, expected)
                self.assertIsNone(result.record)

    def test_workspace_creation_failure_is_bounded(self):
        with patch("minires.preparation.stl.tempfile.mkdtemp", side_effect=OSError):
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
        with patch("minires.preparation.prepare_one.prepare_stl", return_value=result):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = prepare_one_main([
                    "--stl", str(self.stl),
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

    def test_synthetic_diagnostic_exercises_adapter_without_scope_claim(self):
        runner = FakeRunner()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = diagnose_toolchain_main([], runner=runner)

        self.assertEqual((status, stdout.getvalue(), stderr.getvalue()), (0, "compatible\n", ""))
        self.assertTrue(any("--probe" in call[0] for call in runner.calls))
        self.assertTrue(any(call[0][0] == "prusa-slicer" and "--sla" in call[0]
                            for call in runner.calls))
        self.assertTrue(any(call[0][:2] == ("UVtoolsCmd", "print-properties")
                            for call in runner.calls))
        self.assertFalse(runner.generated_path.exists())
        self.assertNotIn("scope_confirmed", stdout.getvalue() + stderr.getvalue())

    def test_default_runner_never_uses_shell_interpolation(self):
        completed = __import__("subprocess").CompletedProcess([], 0, "ok", "")
        with patch("minires.preparation.stl.subprocess.run", return_value=completed) as run:
            result = SubprocessRunner().run(("tool", "$(private-canary)"), timeout_s=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_args.args[0], ["tool", "$(private-canary)"])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 1)

    def test_profile_contract_fails_closed(self):
        self.profile.write_text(
            "material_density = 1.0\nlayer_height = 0.05\nsupports_enable = 0\n"
        )
        profile_digest = sha256(self.profile.read_bytes()).hexdigest()
        with (
            patch("minires.preparation.stl.BUNDLED_PROFILE_RESOURCE", self.profile),
            patch("minires.preparation.stl.PROFILE_SHA256", profile_digest),
        ):
            result = self.prepare(FakeRunner())
        self.assertEqual(result.rejection, "profile_contract_mismatch")


if __name__ == "__main__":
    unittest.main()
