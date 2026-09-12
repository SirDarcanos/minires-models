"""Command-line entry point for resumable private STL batch preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..private_io import PrivateArgumentParser, is_private_path
from .batch import (
    MAX_BATCH_WORKERS,
    BatchPreparationError,
    prepare_stl_batch,
    write_batch_result,
)
from .stl import DEFAULT_TIMEOUT_S, ProcessRunner


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description=(
            "Prepare a directory of private pre-supported STLs with durable "
            "checksum-and-contract checkpoints."
        )
    )
    parser.add_argument(
        "--input-directory", required=True, type=Path,
        help="Private directory whose entries will be inventoried",
    )
    parser.add_argument(
        "--private-output", required=True, type=Path,
        help="Atomically replaced private batch result JSON",
    )
    parser.add_argument(
        "--private-checkpoints", required=True, type=Path,
        help="Private durable checkpoint directory used for resumption",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=f"Bounded processing concurrency (default 1, maximum {MAX_BATCH_WORKERS})",
    )
    parser.add_argument(
        "--scope-confirmed", action="store_true",
        help="Confirm every candidate is a pre-supported miniature in validated scope",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: ProcessRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if not is_private_path(args.private_output) or not is_private_path(args.private_checkpoints):
        print("private_output_required", file=sys.stderr)
        return 2
    try:
        result = prepare_stl_batch(
            args.input_directory,
            checkpoint_dir=args.private_checkpoints,
            runner=runner,
            timeout_s=args.timeout_seconds,
            scope_confirmed=args.scope_confirmed,
            workers=args.workers,
        )
        write_batch_result(result, args.private_output)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 2
    except BatchPreparationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError):
        print("private_output_unavailable", file=sys.stderr)
        return 2
    print("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
