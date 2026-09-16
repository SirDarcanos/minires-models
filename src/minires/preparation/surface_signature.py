"""Deterministic print-axis surface signatures for pre-supported STL geometry.

The representation is additive triangle-surface data. It is deliberately not an
occupied-volume or cross-sectional-area approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

BIN_COUNT = 32
FACE_CHUNK_SIZE = 100_000
VERSION = "minires-print-axis-surface-signature-v1"
SEMANTICS = "additive_triangle_surface_not_occupied_volume"


@dataclass(frozen=True)
class SurfaceSignature:
    surface_area_by_normalized_z: tuple[float, ...]
    absolute_xy_projected_area_by_normalized_z: tuple[float, ...]
    semantics: str = SEMANTICS
    version: str = VERSION


def _normalized(values: Any) -> tuple[float, ...]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum(dtype=np.float64))
    if (
        array.shape != (BIN_COUNT,)
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or not math.isfinite(total)
        or total <= 0.0
    ):
        raise ValueError("invalid_surface_signature_geometry")
    normalized = array / total
    if not np.isfinite(normalized).all():
        raise ValueError("invalid_surface_signature_geometry")
    return tuple(float(value) for value in normalized)


def extract(path: str | Path) -> SurfaceSignature:
    """Load one STL without processing and return its fixed surface signature."""
    import trimesh

    try:
        mesh = trimesh.load(Path(path), process=False)
    except Exception:
        raise ValueError("invalid_surface_signature_geometry") from None
    return from_mesh(mesh)


def from_mesh(mesh: object) -> SurfaceSignature:
    """Return the fixed source-neutral signature without repairing the mesh."""
    import numpy as np
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("invalid_surface_signature_geometry")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or (faces < 0).any()
        or (faces >= len(vertices)).any()
    ):
        raise ValueError("invalid_surface_signature_geometry")
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    z_span = z_max - z_min
    if not math.isfinite(z_span) or z_span <= 0.0:
        raise ValueError("invalid_surface_signature_geometry")

    surface_bins = np.zeros(BIN_COUNT, dtype=np.float64)
    projected_bins = np.zeros(BIN_COUNT, dtype=np.float64)
    for start in range(0, len(faces), FACE_CHUNK_SIZE):
        triangles = vertices[faces[start:start + FACE_CHUNK_SIZE]]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas = np.linalg.norm(crosses, axis=1) * 0.5
        projected = np.abs(crosses[:, 2]) * 0.5
        normalized_z = (triangles[:, :, 2].mean(axis=1) - z_min) / z_span
        bins = np.minimum(
            (normalized_z * BIN_COUNT).astype(np.int64), BIN_COUNT - 1
        )
        if (
            not np.isfinite(crosses).all()
            or not np.isfinite(areas).all()
            or not np.isfinite(projected).all()
            or not np.isfinite(normalized_z).all()
            or (normalized_z < 0.0).any()
            or (normalized_z > 1.0).any()
            or (areas < 0.0).any()
            or (projected < 0.0).any()
        ):
            raise ValueError("invalid_surface_signature_geometry")
        surface_bins += np.bincount(bins, weights=areas, minlength=BIN_COUNT)
        projected_bins += np.bincount(bins, weights=projected, minlength=BIN_COUNT)

    return SurfaceSignature(
        _normalized(surface_bins),
        _normalized(projected_bins),
    )
