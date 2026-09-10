"""Command-line entry point for local physical-baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a volume-and-density sliced-resin-mass baseline from local JSON records."
    )
    parser.add_argument("--records", required=True, type=Path, help="JSON array of local records")
    parser.add_argument(
        "--density-g-per-ml", type=float, help="Explicit resin density in grams per millilitre"
    )
    parser.add_argument("--volume-unit", help="Volume unit; currently only mm3 is supported")
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
    try:
        records = json.loads(args.records.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read records JSON: {error.__class__.__name__}") from error
    if not isinstance(records, list):
        raise SystemExit("Records JSON must contain an array.")
    if not all(isinstance(record, dict) for record in records):
        raise SystemExit("Each record must be a JSON object.")

    result = evaluate_records(
        records=records,
        config=EvaluationConfig(
            resin_density_g_per_ml=args.density_g_per_ml,
            volume_unit=args.volume_unit,
            scope_confirmed=True if args.scope_confirmed else None,
            seed=args.seed,
        ),
        baseline=PhysicalBaseline(),
    )
    serialized = json.dumps(result.to_dict(public=args.public), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
