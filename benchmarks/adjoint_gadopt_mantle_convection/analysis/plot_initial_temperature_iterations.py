#!/usr/bin/env python3
"""Visualize true, initial, and accepted recovered TD-GADOPT temperatures."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vtk_io import scalar_grid


def accepted_fields(output: Path) -> list[tuple[str, Path, str]]:
    directory = output / "diagnostics/accepted_controls"
    entries = [("True T70", directory / "truth_T70.vtu", "truth_T70")]
    entries.extend((path.stem.replace("initial_temperature_", "accepted "), path, "initial_temperature")
                   for path in sorted(directory.glob("initial_temperature_iter_*.vtu")))
    return entries


def panel_figure(entries: list[tuple[str, Path, str]], output: Path, error: bool) -> None:
    truth = scalar_grid(entries[0][1], entries[0][2])[2]
    data = [(title, scalar_grid(path, field)[2]) for title, path, field in entries]
    if error:
        data = [(title, grid - truth) for title, grid in data[1:]]
        maximum = max(float(np.max(np.abs(grid))) for _, grid in data)
        vmin, vmax, cmap, name, heading = (-maximum, maximum, "coolwarm",
                                            "initial_temperature_error.png",
                                            "Recovered initial-temperature error")
    else:
        vmin, vmax, cmap, name, heading = (0.0, 1.0, "viridis",
                                            "initial_temperature_iterations.png",
                                            "Recovered initial temperature")
    columns = min(3, len(data))
    rows = math.ceil(len(data) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.7 * rows),
                                layout="constrained", squeeze=False)
    image = None
    for axis, (title, grid) in zip(axes.flat, data):
        image = axis.imshow(grid, origin="lower", extent=(0, 1, 0, 1),
                            vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    for axis in axes.flat[len(data):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes.flat[:len(data)], shrink=0.85,
                    label="T" if not error else r"$T_{ic}-T_{70}$")
    figure.suptitle(heading)
    figure.savefig(output / "diagnostics/plots" / name, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fields = accepted_fields(args.output_dir)
    panel_figure(fields, args.output_dir, False)
    panel_figure(fields, args.output_dir, True)


if __name__ == "__main__":
    main()
