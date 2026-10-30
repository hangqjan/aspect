#!/usr/bin/env python3
"""Convert the native self-gravitating statistics record into an objective include."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FIELDS = ("surface topography", "CMB topography", "surface geoid", "CMB geoid")
COMPONENTS = ("cosine", "sine")
HEADER = re.compile(r"#\s*(\d+):\s*(.*)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    lines = args.statistics.read_text().splitlines()
    columns: dict[str, int] = {}
    for line in lines:
        match = HEADER.fullmatch(line)
        if match:
            columns[match.group(2)] = int(match.group(1)) - 1
    data = [line for line in lines if line and not line.startswith("#")]
    if not data:
        raise RuntimeError(f"no statistics record in {args.statistics}")
    values = data[-1].split()

    result: dict[tuple[str, str], list[str]] = {}
    for field in FIELDS:
        for component in COMPONENTS:
            series = []
            for order in range(args.degree + 1):
                name = (f"Self-gravitating geoid {field} l={args.degree} "
                        f"m={order} {component}")
                if name not in columns:
                    raise RuntimeError(f"missing native self-gravity column: {name}")
                series.append(values[columns[name]])
            result[field, component] = series

    output = ["# Generated from the native self-gravitating reference statistics.",
              "subsection Adjoint", "  subsection Self-gravitating geoid"]
    for field in FIELDS:
        for component in COMPONENTS:
            output.append("    set Observed " + field + " " + component
                          + " coefficients = " + ", ".join(result[field, component]))
    output.extend(["    set Coefficient weights = 1, 1, 1",
                   "    set Surface density above = 0",
                   "    set CMB density below = 9900",
                   "  end", "end", ""])
    args.output.write_text("\n".join(output))


if __name__ == "__main__":
    main()
