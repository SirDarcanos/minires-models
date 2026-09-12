"""Synthetic interoperability diagnostic for the installed STL toolchain."""

from __future__ import annotations

from importlib.resources import as_file
import math
from pathlib import Path
import struct
import tempfile

from .stl import (
    BUNDLED_PROFILE_RESOURCE,
    DEFAULT_TIMEOUT_S,
    ProcessRunner,
    SubprocessRunner,
    _preflight,
    _run_toolchain,
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
                source = Path(directory) / "synthetic-cube.stl"
                source.write_bytes(_synthetic_cube_stl())
                rejection, _ = _run_toolchain(
                    source, profile_path, process_runner, float(timeout_s)
                )
                if rejection is not None:
                    return "toolchain_diagnostic_failed"
    except OSError:
        return "toolchain_diagnostic_failed"
    return None
