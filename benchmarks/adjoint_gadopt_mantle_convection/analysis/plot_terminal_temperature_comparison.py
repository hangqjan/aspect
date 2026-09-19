#!/usr/bin/env python3
"""Compare observed, initial, and final terminal-temperature fields."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vtk_io import scalar_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir / "diagnostics/terminal_predictions"
    paths = sorted(directory.glob("terminal_temperature_iter_*.vtu"))
    if not paths:
        raise RuntimeError("No accepted terminal predictions were found")
    observed = scalar_grid(directory / "terminal_temperature_observed_T80.vtu", "terminal_temperature")[2]
    initial = scalar_grid(paths[0], "terminal_temperature")[2]
    final = scalar_grid(paths[-1], "terminal_temperature")[2]
    residual_limit = max(float(np.max(np.abs(initial-observed))),
                         float(np.max(np.abs(final-observed))))
    panels = [("Observed T80", observed, "viridis", 0.0, 1.0),
              ("Initial prediction", initial, "viridis", 0.0, 1.0),
              ("Final prediction", final, "viridis", 0.0, 1.0),
              ("Initial − observed", initial-observed, "coolwarm", -residual_limit, residual_limit),
              ("Final − observed", final-observed, "coolwarm", -residual_limit, residual_limit)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axis, (title, grid, cmap, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(grid, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
        figure.colorbar(image, ax=axis, shrink=0.8)
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    axes.flat[-1].set_visible(False)
    figure.suptitle("TD-GADOPT terminal-temperature prediction comparison")
    figure.savefig(args.output_dir / "diagnostics/plots/terminal_temperature_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
