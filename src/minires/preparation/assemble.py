"""Command-line entry point for the complete four-source dataset assembly."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..ingestion import InputError
from ..private_io import PrivateArgumentParser, is_private_path
from .assembly import assemble_expanded_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Assemble historical measurements and a completed new-STL batch into the current private dataset."
    )
    parser.add_argument("--historical-records", required=True, type=Path)
    parser.add_argument("--exclude-historical-source", required=True)
    parser.add_argument("--new-batch-result", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--private-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not is_private_path(args.private_dir):
        print("private_output_required", file=sys.stderr)
        return 2
    try:
        assemble_expanded_dataset(
            args.historical_records,
            excluded_historical_source=args.exclude_historical_source,
            new_batch_result=args.new_batch_result,
            output_dir=args.private_dir,
            seed=args.seed,
        )
    except (InputError, OSError, TypeError, ValueError):
        print("dataset_assembly_failed", file=sys.stderr)
        return 2
    print("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
