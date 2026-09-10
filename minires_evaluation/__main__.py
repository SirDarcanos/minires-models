"""Command-line entry point for local physical-baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Never, Sequence
from uuid import uuid4

from .ingestion import InputError
from .private_io import create_private_file
from .evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.exit(2, "Invalid command arguments; use --help.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(
        description="Evaluate local JSON, Extended JSONL, or CSV records without database access."
    )
    parser.add_argument("--records", required=True, type=Path, help="Local .json array, .jsonl export, or .csv table")
    parser.add_argument(
        "--density-g-per-ml", type=float, help="Explicit resin density in grams per millilitre"
    )
    parser.add_argument("--volume-unit", help="Explicit volume unit: mm3, cm3, or ml")
    parser.add_argument("--reconcile", type=Path, action="append", default=[], help="Comparison input (repeatable)")
    parser.add_argument("--private-dir", type=Path, help="New run directory beneath a private/ directory")
    parser.add_argument(
        "--scope-confirmed",
        action="store_true",
        help="Confirm the inputs are within the intended pre-supported-miniature scope",
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded reproducibility seed")
    parser.add_argument(
        "--public", action="store_true", help="Write only the public allowlisted summary"
    )
    parser.add_argument("--output", type=Path, help="Write JSON result to this path instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    private_dir = args.private_dir
    if not args.public and not args.output and private_dir is None:
        private_dir = Path("private") / ("run-" + uuid4().hex)
    try:
        result = evaluate_records(
            records=args.records,
            config=EvaluationConfig(
                resin_density_g_per_ml=args.density_g_per_ml,
                volume_unit=args.volume_unit,
                scope_confirmed=True if args.scope_confirmed else None,
                seed=args.seed,
            ),
            baseline=PhysicalBaseline(),
            reconcile_with=args.reconcile,
            output_dir=private_dir,
        )
        serialized = json.dumps(result.to_dict(public=args.public), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            if not args.public and "private" not in args.output.resolve().parts:
                raise InputError("private_output_directory_required")
            with create_private_file(args.output) as stream:
                stream.write(serialized.encode("utf-8"))
        elif args.public:
            print(serialized, end="")
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("local_evaluation_failed") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
