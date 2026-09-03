#!/usr/bin/env python
"""Backward-compatible entry point for the original script interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdp_uq.cli import main as cli_main


def main() -> int:
    """Translate the historic arguments to the packaged CLI."""
    parser = argparse.ArgumentParser(
        description="Correct phase-detection probe measurements and quantify uncertainty."
    )
    parser.add_argument("-dx", type=float, required=True)
    parser.add_argument("-dy", type=float, required=True)
    parser.add_argument("-Np", type=int, required=True)
    parser.add_argument("path_to_file", type=Path)
    arguments = parser.parse_args()
    return cli_main(
        [
            "predict",
            str(arguments.path_to_file),
            "--dx",
            str(arguments.dx),
            "--dy",
            str(arguments.dy),
            "--particles-per-window",
            str(arguments.Np),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
