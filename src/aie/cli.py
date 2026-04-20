"""Command-line entry point (``aie``)."""

from __future__ import annotations

import argparse
import sys

from aie.pipeline import run_from_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aie", description="Atmospheric Intelligence Engine CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["data", "train", "evaluate", "scenarios", "all"],
        help="Which pipeline stage to run.",
    )
    args = parser.parse_args(argv)
    run_from_yaml(args.config, args.stage)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
