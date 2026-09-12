"""Isolated trimesh probe used by the one-STL preparation workflow."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Sequence


def _trimesh_version() -> str:
    try:
        return version("trimesh")
    except PackageNotFoundError:
        raise RuntimeError("geometry_dependency_unavailable") from None


def probe(path: Path) -> dict[str, int | float]:
    try:
        import trimesh
        mesh = trimesh.load(path)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError("invalid mesh")
        bbox_x, bbox_y, bbox_z = (float(value) for value in mesh.bounding_box.extents)
        volume = float(mesh.volume)
        surface_area = float(mesh.area)
        return {
            "volume": volume,
            "surface_area": surface_area,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_z": bbox_z,
            "bbox_area": bbox_x * bbox_y * bbox_z,
            "mass": float(mesh.mass),
            "euler_number": int(mesh.euler_number),
            "scale": float(mesh.scale),
            "surface_volume_ratio": surface_area / volume,
        }
    except Exception:
        raise RuntimeError("geometry_probe_failed") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", action="store_true")
    group.add_argument("--probe", type=Path)
    args = parser.parse_args(argv)
    try:
        dependency_version = _trimesh_version()
        if args.version:
            print(f"trimesh {dependency_version}")
            return 0
        print(json.dumps(probe(args.probe), allow_nan=False, separators=(",", ":")))
        return 0
    except (RuntimeError, TypeError, ValueError, ZeroDivisionError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
