"""Thin wrapper around :func:`aie.pipeline.run_from_yaml`."""

from __future__ import annotations

import argparse

from aie.pipeline import run_from_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIE pipeline from YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["data", "train", "evaluate", "scenarios", "all"],
    )
    args = parser.parse_args()
    run_from_yaml(args.config, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
