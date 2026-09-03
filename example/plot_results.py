#!/usr/bin/env python
"""Create the published-style application profile plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdp_uq.plotting import plot_profile_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Corrected CSV produced by pdp-uq.")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = plot_profile_file(arguments.input, arguments.output)
    print(output)


if __name__ == "__main__":
    main()
