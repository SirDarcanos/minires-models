"""Private dataset preparation command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Never, Sequence

from .ingestion import InputError
from .preparation import prepare_private_dataset


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.exit(2, "Invalid command arguments; use --help.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Prepare identity-redacted records and grouping evidence beneath private/."
    )
    parser.add_argument("--records", required=True, type=Path, help="Retained local JSONL export")
    parser.add_argument(
        "--reconcile", type=Path, action="append", default=[],
        help="Existing labeled CSV or export to reconcile (repeatable)",
    )
    parser.add_argument(
        "--private-dir", required=True, type=Path,
        help="New output directory beneath private/",
    )
    parser.add_argument("--seed", type=int, default=0, help="Frozen-fold allocation seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_private_dataset(
            args.records,
            comparison_records=args.reconcile,
            output_dir=args.private_dir,
            seed=args.seed,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, TypeError, ValueError):
        raise SystemExit("private_preparation_failed") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
