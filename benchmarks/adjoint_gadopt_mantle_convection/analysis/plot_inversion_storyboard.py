#!/usr/bin/env python3
"""Create a paper-style TD-GADOPT initial/final reconstruction storyboard.

The figure uses only durable ASPECT VTU output produced by the validated
terminal-temperature inversion.  It deliberately lives outside ASPECT's C++
solver path: plotting must neither alter an optimization iterate nor require a
benchmark-specific postprocessor in the core executable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vtk_io import scalar_grid


SELECTED_ITERATIONS = (0, 2, 4, 6, 8, 12)


def field(path: Path, name: str) -> np.ndarray:
    """Read a scalar VTU field and fail early if archived evidence is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required ASPECT field does not exist: {path}")
    return scalar_grid(path, name)[2]


def add_panel(axis: plt.Axes, values: np.ndarray, *, cmap: str,
              vmin: float, vmax: float) -> plt.AxesImage:
    """Draw one square unit-box field with a quiet, publication-style frame."""
    image = axis.imshow(values, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                        interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_aspect("equal")
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.70")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reference and selected accepted TD-GADOPT iterates.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Archived ASPECT inverse-output directory")
    parser.add_argument("--iterations", type=int, nargs="+", default=SELECTED_ITERATIONS,
                        help="Accepted iterations shown below the reference row")
    arguments = parser.parse_args()

    diagnostics = arguments.output_dir / "diagnostics"
    initial_directory = diagnostics / "accepted_controls"
    terminal_directory = diagnostics / "terminal_predictions"

    # The reference fields define a common temperature colour scale. Every
    # reconstruction is shown against the same [0,1] interval, so visual
    # improvement cannot be mistaken for an iteration-specific rescaling.
    true_initial = field(initial_directory / "truth_T70.vtu", "truth_T70")
    true_final = field(terminal_directory / "terminal_temperature_observed_T80.vtu",
                       "terminal_temperature")

    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for iteration in arguments.iterations:
        initial = field(initial_directory / f"initial_temperature_iter_{iteration:03d}.vtu",
                        "initial_temperature")
        final = field(terminal_directory / f"terminal_temperature_iter_{iteration:03d}.vtu",
                      "terminal_temperature")
        # Absolute error answers the presentation question directly: darker
        # gray means a larger pointwise mismatch, independent of its sign.
        rows.append((f"Iteration {iteration}", initial, np.abs(initial - true_initial),
                     final, np.abs(final - true_final)))

    maximum_misfit = max(float(np.max(values))
                         for row in rows for values in (row[2], row[4]))
    # Avoid a degenerate colourbar if a future exact reconstruction is plotted.
    maximum_misfit = max(maximum_misfit, np.finfo(float).eps)

    figure = plt.figure(figsize=(10.0, 2.22 * (len(rows) + 1) + 0.9),
                         layout="constrained")
    grid = figure.add_gridspec(len(rows) + 2, 4,
                               height_ratios=[1.0] * (len(rows) + 1) + [0.10])
    axes = np.empty((len(rows) + 1, 4), dtype=object)

    headings = ("Initial condition", "Initial misfit",
                "Final condition", "Final misfit")
    for column, heading in enumerate(headings):
        axis = figure.add_subplot(grid[0, column])
        axes[0, column] = axis
        axis.set_title(heading, fontsize=12, pad=12,
                       bbox={"facecolor": "white", "edgecolor": "black", "pad": 3.5})

    initial_image = add_panel(axes[0, 0], true_initial, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[0, 1].axis("off")
    final_image = add_panel(axes[0, 2], true_final, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[0, 3].axis("off")
    axes[0, 0].set_ylabel("Reference", rotation=90, labelpad=30, fontsize=11,
                          bbox={"facecolor": "white", "edgecolor": "black", "pad": 4})

    misfit_image = None
    for row_index, (label, initial, initial_misfit, final, final_misfit) in enumerate(rows, start=1):
        row_axes = [figure.add_subplot(grid[row_index, column]) for column in range(4)]
        axes[row_index, :] = row_axes
        initial_image = add_panel(row_axes[0], initial, cmap="turbo", vmin=0.0, vmax=1.0)
        misfit_image = add_panel(row_axes[1], initial_misfit, cmap="Greys",
                                 vmin=0.0, vmax=maximum_misfit)
        final_image = add_panel(row_axes[2], final, cmap="turbo", vmin=0.0, vmax=1.0)
        misfit_image = add_panel(row_axes[3], final_misfit, cmap="Greys",
                                 vmin=0.0, vmax=maximum_misfit)
        row_axes[0].set_ylabel(label, rotation=90, labelpad=30, fontsize=11,
                                bbox={"facecolor": "white", "edgecolor": "black", "pad": 4})

    # Two shared colourbars preserve comparability across all rows: temperature
    # is physical T in [0,1], whereas misfit is |T-T_reference| in [0,max].
    temperature_bar = figure.colorbar(initial_image, cax=figure.add_subplot(grid[-1, :2]),
                                      orientation="horizontal")
    temperature_bar.set_label("Temperature, $T$")
    misfit_bar = figure.colorbar(misfit_image, cax=figure.add_subplot(grid[-1, 2:]),
                                 orientation="horizontal")
    misfit_bar.set_label(r"Absolute misfit, $|T-T_{reference}|$")

    figure.suptitle("TD-GADOPT terminal-temperature inversion: accepted reconstructions",
                    fontsize=14)
    destination = diagnostics / "plots" / "inversion_storyboard.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
