"""Command-line adapter for the synthetic STL toolchain diagnostic."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .stl import DEFAULT_TIMEOUT_S, ProcessRunner
from .toolchain import diagnose_toolchain


def main(
    argv: Sequence[str] | None = None, *, runner: ProcessRunner | None = None
) -> int:
    parser = argparse.ArgumentParser(
        description="Check STL tool interoperability with a temporary synthetic cube."
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    rejection = diagnose_toolchain(runner=runner, timeout_s=args.timeout_seconds)
    if rejection is not None:
        print(rejection, file=sys.stderr)
        return 2
    print("compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
