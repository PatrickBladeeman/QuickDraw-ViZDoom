"""Command-line entry point for QuickDraw-ViZDoom quality gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .runner import run_profile


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, emit one JSON report, and return its tri-state exit code."""

    parser = argparse.ArgumentParser(prog="python -m quickdraw_vizdoom.qa")
    run_parser = parser.add_subparsers(dest="command", required=True).add_parser("run")
    run_parser.add_argument("--contract", required=True)
    run_parser.add_argument("--profile", required=True)
    arguments = parser.parse_args(argv)
    report = run_profile(arguments.contract, arguments.profile)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return report.outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
