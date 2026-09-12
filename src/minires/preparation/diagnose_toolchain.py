"""Check the installed STL toolchain with generated geometry outside validated scope."""

from __future__ import annotations

import argparse
from importlib.resources import as_file
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Sequence

from .stl import (
    BUNDLED_PROFILE_RESOURCE,
    DEFAULT_TIMEOUT_S,
    ProcessRunner,
    SubprocessRunner,
    _measurements,
    _preflight,
    _weight_from_properties,
)

_TRIANGLES = (
    ((-5, -5, 5), (-5, 5, 5), (-5, -5, -5)),
    ((5, -5, -5), (-5, -5, 5), (-5, -5, -5)),
    ((-5, -5, -5), (-5, 5, 5), (-5, 5, -5)),
    ((-5, 5, -5), (5, -5, -5), (-5, -5, -5)),
    ((-5, -5, 5), (5, 5, 5), (-5, 5, 5)),
    ((5, -5, 5), (-5, -5, 5), (5, -5, -5)),
    ((5, -5, 5), (5, 5, 5), (-5, -5, 5)),
    ((-5, 5, 5), (5, 5, 5), (-5, 5, -5)),
    ((5, 5, -5), (5, -5, -5), (-5, 5, -5)),
    ((-5, 5, -5), (5, 5, 5), (5, 5, -5)),
    ((5, 5, -5), (5, -5, 5), (5, -5, -5)),
    ((5, 5, 5), (5, -5, 5), (5, 5, -5)),
)


def _synthetic_cube_stl() -> bytes:
    header = struct.pack("<80sI", b"MiniRes synthetic toolchain diagnostic", len(_TRIANGLES))
    facets = []
    for triangle in _TRIANGLES:
        coordinates = tuple(float(value) for vertex in triangle for value in vertex)
        facets.append(struct.pack("<12fH", 0.0, 0.0, 0.0, *coordinates, 0))
    return header + b"".join(facets)


def diagnose_toolchain(
    *, runner: ProcessRunner | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> str | None:
    """Return a bounded rejection, or None when synthetic interoperability succeeds.

    The generated cube checks tool interoperability only. It is not a pre-supported
    miniature and this function makes no validated-scope claim.
    """
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        return "invalid_timeout"
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return "invalid_timeout"
    process_runner = runner or SubprocessRunner()
    try:
        with as_file(BUNDLED_PROFILE_RESOURCE) as profile_path:
            rejection, _, _, _ = _preflight(process_runner, float(timeout_s), profile_path)
            if rejection is not None:
                return rejection
            with tempfile.TemporaryDirectory(prefix="minires-toolchain-") as directory:
                workspace = Path(directory)
                source = workspace / "synthetic-cube.stl"
                sliced_output = workspace / "synthetic-output.pwmx"
                source.write_bytes(_synthetic_cube_stl())
                try:
                    geometry = process_runner.run(
                        (
                            sys.executable,
                            "-m",
                            "minires.preparation.stl_probe",
                            "--probe",
                            source,
                        ),
                        timeout_s=float(timeout_s),
                    )
                except (OSError, TimeoutError, subprocess.TimeoutExpired):
                    return "geometry_diagnostic_failed"
                if geometry.returncode != 0 or _measurements(geometry.stdout) is None:
                    return "geometry_diagnostic_failed"
                try:
                    sliced = process_runner.run(
                        (
                            "prusa-slicer",
                            "--load",
                            profile_path,
                            "--sla",
                            "--output",
                            sliced_output,
                            source,
                        ),
                        timeout_s=float(timeout_s),
                        cwd=workspace,
                    )
                except (OSError, TimeoutError, subprocess.TimeoutExpired):
                    return "slicing_diagnostic_failed"
                if sliced.returncode != 0 or not sliced_output.is_file():
                    return "slicing_diagnostic_failed"
                try:
                    properties = process_runner.run(
                        ("UVtoolsCmd", "print-properties", sliced_output),
                        timeout_s=float(timeout_s),
                        cwd=workspace,
                    )
                except (OSError, TimeoutError, subprocess.TimeoutExpired):
                    return "uvtools_diagnostic_failed"
                weight_rejection, _ = _weight_from_properties(properties)
                if weight_rejection is not None:
                    return "uvtools_diagnostic_failed"
    except OSError:
        return "toolchain_diagnostic_failed"
    return None


def main(
    argv: Sequence[str] | None = None, *, runner: ProcessRunner | None = None
) -> int:
    parser = argparse.ArgumentParser(
        description="Check STL tool interoperability with a temporary synthetic cube."
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    rejection = diagnose_toolchain(runner=runner, timeout_s=args.timeout_seconds)
    if rejection is not None:
        print(rejection, file=sys.stderr)
        return 2
    print("compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
