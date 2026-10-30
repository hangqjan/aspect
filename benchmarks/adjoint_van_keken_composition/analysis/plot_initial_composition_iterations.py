#!/usr/bin/env python3
"""Plot analytic and accepted initial-composition controls for the twin experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vtk_io import scalar_grid


X_EXTENT = 0.9142


def reference_initial_composition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The smooth van Keken initial chemical field in reference_forward.prm."""
    xx, yy = np.meshgrid(x, y)
    return 0.5 * (1.0 + np.tanh((0.2 + 0.02 * np.cos(np.pi * xx / X_EXTENT) - yy) / 0.02))


def accepted_fields(output: Path) -> list[tuple[str, Path]]:
    directory = output / "diagnostics/accepted_controls"
    fields = sorted(directory.glob("initial_composition_iter_*.vtu"))
    if not fields:
        raise FileNotFoundError(f"No accepted-control VTU files in {directory}")
    return [(f"iteration {int(path.stem.rsplit('_', 1)[1])}", path) for path in fields]


def panel_figure(entries: list[tuple[str, Path]], plot_directory: Path, error: bool) -> None:
    x, y, first = scalar_grid(entries[0][1], "chemical")
    truth = reference_initial_composition(x, y)
    data = [("reference initial C", truth), (entries[0][0], first)]
    data.extend((title, scalar_grid(path, "chemical")[2]) for title, path in entries[1:])
    if error:
        data = [(title, grid - truth) for title, grid in data[1:]]
        maximum = max(float(np.max(np.abs(grid))) for _, grid in data)
        vmin, vmax, cmap, filename, label, heading = (
            -maximum, maximum, "coolwarm", "initial_composition_error.png",
            r"$C_{ic} - C_{reference}$", "Recovered initial-composition error")
    else:
        vmin, vmax, cmap, filename, label, heading = (
            0.0, 1.0, "viridis", "initial_composition_iterations.png", "C",
            "Recovered initial composition")
    columns = min(3, len(data))
    rows = math.ceil(len(data) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.7 * rows),
                                layout="constrained", squeeze=False)
    image = None
    for axis, (title, grid) in zip(axes.flat, data):
        image = axis.imshow(grid, origin="lower", extent=(0, X_EXTENT, 0, 1),
                            vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("z")
    for axis in axes.flat[len(data):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes.flat[:len(data)], shrink=0.85, label=label)
    figure.suptitle(heading)
    plot_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_directory / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="ASPECT inverse output directory")
    parser.add_argument("--plot-dir", type=Path,
                        help="directory for generated PNG diagnostics")
    args = parser.parse_args()
    fields = accepted_fields(args.output_dir)
    plot_directory = args.plot_dir or args.output_dir / "diagnostics/plots"
    panel_figure(fields, plot_directory, False)
    panel_figure(fields, plot_directory, True)


if __name__ == "__main__":
    main()
