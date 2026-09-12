"""Command-line entry point for private source-balanced partitioning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .ingestion import InputError
from .partitioning import partition_private_dataset
from .private_io import PrivateArgumentParser


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Create deterministic private train, validation, and test record artifacts."
    )
    parser.add_argument("--records", required=True, type=Path, help="Harmonized local JSON, JSONL, or CSV records")
    parser.add_argument("--seed", required=True, type=int, help="Fixed allocation seed")
    parser.add_argument(
        "--exclude-source", required=True,
        help="Private anonymous source identity to omit (never persisted)",
    )
    parser.add_argument(
        "--private-dir", required=True, type=Path,
        help="Current output directory beneath private/; complete sets are replaced",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        partition_private_dataset(
            args.records,
            excluded_source=args.exclude_source,
            output_dir=args.private_dir,
            seed=args.seed,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, TypeError, ValueError):
        raise SystemExit("private_partitioning_failed") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
