#!/usr/bin/env python3
"""Extract TD-GADOPT truth CSV files from normal ASPECT VTU output.

This script is intentionally outside ASPECT's C++ runtime path.  It converts
ordinary visualization files into portable benchmark evidence and never
changes an adjoint objective, optimizer control, or trajectory archive.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vtk_io import read_vtu


def write_temperature(path: Path, data: dict) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("x", "y", "T"))
        for point, value in zip(data["Points"], data["T"].reshape(-1), strict=True):
            writer.writerow((f"{point[0]:.17g}", f"{point[1]:.17g}", f"{value:.17g}"))


def write_velocity(path: Path, data: dict) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("x", "y", "u_x", "u_y", "u_z"))
        for point, value in zip(data["Points"], data["velocity"], strict=True):
            writer.writerow((f"{point[0]:.17g}", f"{point[1]:.17g}",
                             *(f"{component:.17g}" for component in value)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="ASPECT reference output directory containing solution/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination diagnostics directory")
    arguments = parser.parse_args()
    # ASPECT's visualization lifecycle is the authoritative producer of the
    # fields.  Reading from ``solution/`` rather than an in-memory harness
    # makes the benchmark extraction reproducible from durable files.
    solution_dir = arguments.reference_dir / "solution"
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(70, 81):
        data = read_vtu(solution_dir / f"solution-{step:05d}.0000.vtu")
        if step in (70, 80):
            write_temperature(arguments.output_dir / f"reference_T{step}.csv", data)
        if step < 80:
            write_velocity(arguments.output_dir / f"reference_u{step}.csv", data)
    print(f"Wrote T70, T80, and u70...u79 CSV observations to {arguments.output_dir}")


if __name__ == "__main__":
    main()
