"""Prepare one STL as an identity-free MiniRes record under the pinned slicer contract."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence

EBMINIMANAGER_REVISION = "1a841195813136ee3b380ab1d192727f385f7a55"
PROFILE_RELATIVE_PATH = Path("prediction/config-anycubic-mono.ini")
PROFILE_SHA256 = "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e"
DENSITY_G_PER_ML = 1.1
LAYER_HEIGHT_MM = 0.05
DEFAULT_TIMEOUT_S = 120.0

_MEASUREMENT_FIELDS = (
    "volume",
    "surface_area",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "bbox_area",
    "mass",
    "euler_number",
    "scale",
    "surface_volume_ratio",
)
_POSITIVE_FIELDS = (
    "volume",
    "surface_area",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "bbox_area",
    "mass",
    "scale",
    "surface_volume_ratio",
)
_WEIGHT_PATTERN = re.compile(
    r"(?:^|\n)\s*WeightG\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:\n|$)"
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    """Replaceable boundary for every external application."""

    def run(
        self,
        args: Sequence[str | Path],
        *,
        timeout_s: float,
        cwd: Path | None = None,
    ) -> ProcessResult: ...


class SubprocessRunner:
    """Run argument vectors directly, without a shell."""

    def run(
        self,
        args: Sequence[str | Path],
        *,
        timeout_s: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        completed = subprocess.run(
            [str(value) for value in args],
            cwd=cwd,
            timeout=timeout_s,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class StlPreparationResult:
    outcome: str
    rejection: str | None
    record: Mapping[str, Any] | None
    contract: Mapping[str, Any]
    versions: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "rejection": self.rejection,
            "record": dict(self.record) if self.record is not None else None,
            "contract": dict(self.contract),
            "versions": dict(self.versions),
        }


def _contract(profile_digest: str | None = None) -> dict[str, Any]:
    return {
        "ebminimanager_revision": EBMINIMANAGER_REVISION,
        "profile_relative_path": PROFILE_RELATIVE_PATH.as_posix(),
        "profile_sha256": profile_digest or PROFILE_SHA256,
        "resin_density_g_per_ml": DENSITY_G_PER_ML,
        "layer_height_mm": LAYER_HEIGHT_MM,
        "slicer_added_supports": False,
        "label_source": "UVtools print-properties WeightG",
    }


def _rejected(
    reason: str,
    *,
    versions: Mapping[str, str] | None = None,
    profile_digest: str | None = None,
) -> StlPreparationResult:
    return StlPreparationResult(
        outcome="rejected",
        rejection=reason,
        record=None,
        contract=_contract(profile_digest),
        versions=dict(versions or {}),
    )


def _file_inventory(path: Path) -> dict[str, Any]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {"byte_count": byte_count, "sha256": digest.hexdigest()}


def _profile_values(profile: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in profile.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _version_line(*outputs: str) -> str:
    for output in outputs:
        for line in output.splitlines():
            if line.strip():
                return line.strip()[:200]
    return "available_version_unreported"


def _preflight(
    checkout: Path,
    runner: ProcessRunner,
    timeout_s: float,
) -> tuple[str | None, dict[str, str], Path, str | None]:
    versions: dict[str, str] = {}
    try:
        revision = runner.run(
            ("git", "-C", checkout, "rev-parse", "HEAD"),
            timeout_s=timeout_s,
        )
    except (OSError, TimeoutError, subprocess.TimeoutExpired):
        return "ebminimanager_revision_unavailable", versions, checkout / PROFILE_RELATIVE_PATH, None
    if revision.returncode != 0:
        return "ebminimanager_revision_unavailable", versions, checkout / PROFILE_RELATIVE_PATH, None
    if revision.stdout.strip() != EBMINIMANAGER_REVISION:
        return "ebminimanager_revision_mismatch", versions, checkout / PROFILE_RELATIVE_PATH, None

    profile_path = checkout / PROFILE_RELATIVE_PATH
    try:
        profile = profile_path.read_bytes()
    except OSError:
        return "profile_unavailable", versions, profile_path, None
    digest = sha256(profile).hexdigest()
    if digest != PROFILE_SHA256:
        return "profile_checksum_mismatch", versions, profile_path, digest
    try:
        values = _profile_values(profile)
        contract_matches = (
            float(values.get("material_density", "nan")) == DENSITY_G_PER_ML
            and float(values.get("layer_height", "nan")) == LAYER_HEIGHT_MM
            and values.get("supports_enable") == "0"
        )
    except (UnicodeDecodeError, ValueError):
        contract_matches = False
    if not contract_matches:
        return "profile_contract_mismatch", versions, profile_path, digest

    probes = (
        ("prusaslicer", ("prusa-slicer", "--version"), "missing_prusaslicer"),
        ("uvtools", ("UVtoolsCmd", "--version"), "missing_uvtools"),
        (
            "geometry",
            (sys.executable, "-m", "minires_evaluation.stl_probe", "--version"),
            "missing_geometry_dependency",
        ),
    )
    for name, command, missing_reason in probes:
        try:
            result = runner.run(command, timeout_s=timeout_s)
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            return missing_reason, versions, profile_path, digest
        if result.returncode != 0:
            return missing_reason, versions, profile_path, digest
        versions[name] = _version_line(result.stdout, result.stderr)
    versions["ebminimanager"] = EBMINIMANAGER_REVISION
    versions["python"] = platform.python_version()
    return None, versions, profile_path, digest


def _measurements(output: str) -> dict[str, int | float] | None:
    try:
        raw = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict) or any(field not in raw for field in _MEASUREMENT_FIELDS):
        return None
    parsed: dict[str, int | float] = {}
    for field in _MEASUREMENT_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        if field in _POSITIVE_FIELDS and value <= 0:
            return None
        parsed[field] = value
    return parsed


def _process_copy(
    source_copy: Path,
    profile_path: Path,
    runner: ProcessRunner,
    timeout_s: float,
    inventory: Mapping[str, Any],
    versions: Mapping[str, str],
    profile_digest: str,
) -> StlPreparationResult:
    try:
        geometry = runner.run(
            (sys.executable, "-m", "minires_evaluation.stl_probe", "--probe", source_copy),
            timeout_s=timeout_s,
        )
    except (TimeoutError, subprocess.TimeoutExpired):
        return _rejected("geometry_timeout", versions=versions, profile_digest=profile_digest)
    except OSError:
        return _rejected("corrupt_geometry", versions=versions, profile_digest=profile_digest)
    if geometry.returncode != 0:
        return _rejected("corrupt_geometry", versions=versions, profile_digest=profile_digest)
    measurements = _measurements(geometry.stdout)
    if measurements is None:
        return _rejected("invalid_measurements", versions=versions, profile_digest=profile_digest)

    sliced_output = source_copy.parent / "sliced-output.pwmx"
    try:
        sliced = runner.run(
            (
                "prusa-slicer", "--load", profile_path, "--sla",
                "--output", sliced_output, source_copy,
            ),
            timeout_s=timeout_s,
            cwd=source_copy.parent,
        )
    except (TimeoutError, subprocess.TimeoutExpired):
        return _rejected("slicing_timeout", versions=versions, profile_digest=profile_digest)
    except OSError:
        return _rejected("slicing_failed", versions=versions, profile_digest=profile_digest)
    if sliced.returncode != 0:
        return _rejected("slicing_failed", versions=versions, profile_digest=profile_digest)
    if not sliced_output.is_file():
        return _rejected("sliced_output_absent", versions=versions, profile_digest=profile_digest)
    try:
        sliced_inventory = _file_inventory(sliced_output)
    except OSError:
        return _rejected("sliced_output_unreadable", versions=versions, profile_digest=profile_digest)

    try:
        properties = runner.run(
            ("UVtoolsCmd", "print-properties", sliced_output),
            timeout_s=timeout_s,
            cwd=source_copy.parent,
        )
    except (TimeoutError, subprocess.TimeoutExpired):
        return _rejected("uvtools_timeout", versions=versions, profile_digest=profile_digest)
    except OSError:
        return _rejected("uvtools_failed", versions=versions, profile_digest=profile_digest)
    if properties.returncode != 0:
        return _rejected("uvtools_failed", versions=versions, profile_digest=profile_digest)
    match = _WEIGHT_PATTERN.search(properties.stdout)
    if match is None:
        return _rejected("weight_g_missing", versions=versions, profile_digest=profile_digest)
    weight = float(match.group(1))
    if not math.isfinite(weight) or weight <= 0:
        return _rejected("invalid_weight_g", versions=versions, profile_digest=profile_digest)

    file_size_kib = inventory["byte_count"] / 1024
    record: dict[str, Any] = {
        "file_size_kib": file_size_kib,
        "mesh_volume_mm3": measurements["volume"],
        "surface_area_mm2": measurements["surface_area"],
        "bounding_box_x_mm": measurements["bbox_x"],
        "bounding_box_y_mm": measurements["bbox_y"],
        "bounding_box_z_mm": measurements["bbox_z"],
        "bounding_box_volume_mm3": measurements["bbox_area"],
        "mesh_mass_at_unit_density": measurements["mass"],
        "euler_characteristic": measurements["euler_number"],
        "mesh_scale_mm": measurements["scale"],
        "surface_to_volume_ratio_per_mm": measurements["surface_volume_ratio"],
        "kb": file_size_kib,
        **measurements,
        "sliced_resin_mass_g": weight,
        "weight": weight,
        "volume_unit": "mm3",
        "resin_density_g_per_ml": DENSITY_G_PER_ML,
        "scope_confirmed": True,
        "slicing_conditions": {
            "layer_height_mm": LAYER_HEIGHT_MM,
            "slicer_added_supports": False,
        },
        "input_inventory": dict(inventory),
        "sliced_output_inventory": sliced_inventory,
    }
    return StlPreparationResult(
        outcome="prepared",
        rejection=None,
        record=record,
        contract=_contract(profile_digest),
        versions=dict(versions),
    )


def prepare_stl(
    source_stl: str | Path,
    *,
    ebminimanager_dir: str | Path,
    runner: ProcessRunner | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> StlPreparationResult:
    """Prepare one STL, returning a record or a named rejection without exposing identity."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return _rejected("invalid_timeout")
    process_runner = runner or SubprocessRunner()
    source = Path(source_stl)
    try:
        inventory = _file_inventory(source)
    except OSError:
        return _rejected("input_unavailable")
    if inventory["byte_count"] == 0 or source.suffix.lower() != ".stl":
        return _rejected("invalid_input")

    rejection, versions, profile_path, profile_digest = _preflight(
        Path(ebminimanager_dir), process_runner, timeout_s
    )
    if rejection is not None:
        return _rejected(rejection, versions=versions, profile_digest=profile_digest)
    assert profile_digest is not None

    workspace = Path(tempfile.mkdtemp(prefix="minires-stl-"))
    result: StlPreparationResult
    try:
        source_copy = workspace / "input.stl"
        shutil.copyfile(source, source_copy)
        if _file_inventory(source_copy) != inventory:
            result = _rejected(
                "source_checksum_changed", versions=versions, profile_digest=profile_digest
            )
        else:
            result = _process_copy(
                source_copy,
                profile_path,
                process_runner,
                timeout_s,
                inventory,
                versions,
                profile_digest,
            )
    except OSError:
        result = _rejected("workspace_failure", versions=versions, profile_digest=profile_digest)
    finally:
        try:
            shutil.rmtree(workspace)
        except OSError:
            result = _rejected("cleanup_failed", versions=versions, profile_digest=profile_digest)

    try:
        unchanged = _file_inventory(source) == inventory
    except OSError:
        unchanged = False
    if not unchanged:
        return _rejected("source_checksum_changed", versions=versions, profile_digest=profile_digest)
    return result
