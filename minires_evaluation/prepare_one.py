"""Command-line entry point for preparing one private pre-supported STL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .private_io import PrivateArgumentParser, write_private_json
from .stl_preparation import DEFAULT_TIMEOUT_S, prepare_stl


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Prepare one private pre-supported STL under the pinned MiniRes contract."
    )
    parser.add_argument("--stl", required=True, type=Path, help="Private source STL")
    parser.add_argument(
        "--ebminimanager-dir",
        required=True,
        type=Path,
        help="Checkout of the pinned EBMiniManager revision",
    )
    parser.add_argument(
        "--private-output",
        required=True,
        type=Path,
        help="New result JSON path beneath private/",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--scope-confirmed",
        action="store_true",
        help="Confirm that the input is a pre-supported miniature in the validated scope",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if "private" not in args.private_output.resolve().parts:
        print("private_output_required", file=sys.stderr)
        return 2
    result = prepare_stl(
        args.stl,
        ebminimanager_dir=args.ebminimanager_dir,
        timeout_s=args.timeout_seconds,
        scope_confirmed=args.scope_confirmed,
    )
    try:
        write_private_json(args.private_output, result.to_dict())
    except (OSError, TypeError, ValueError):
        print("private_output_unavailable", file=sys.stderr)
        return 2
    if result.outcome == "prepared":
        print("prepared")
        return 0
    print(result.rejection or "stl_preparation_failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
