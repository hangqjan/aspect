#!/usr/bin/env python3
"""Create a four-column composition inversion storyboard from durable output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adjoint_van_keken_composition" / "analysis"))
from vtk_io import scalar_grid  # noqa: E402


def read_observation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    data = np.asarray([[float(value) for value in row] for row in rows if row[0] != "x"])
    x, y = np.unique(data[:, 0]), np.unique(data[:, 1])
    if len(data) != len(x) * len(y):
        raise ValueError(f"{path}: incomplete structured observation")
    return x, y, data[:, 2].reshape((len(y), len(x)))


def resample(x: np.ndarray, y: np.ndarray, values: np.ndarray,
             target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    along_x = np.asarray([np.interp(target_x, x, row) for row in values])
    return np.asarray([np.interp(target_y, y, along_x[:, column])
                       for column in range(len(target_x))]).T


def scalar_field(path: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"required ASPECT field does not exist: {path}")
    return scalar_grid(path, name)


def add_panel(axis: plt.Axes, values: np.ndarray, *, cmap: str,
              vmin: float, vmax: float) -> plt.AxesImage:
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
        description="Plot reference and accepted composition reconstructions.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-initial-vtu", type=Path, required=True)
    parser.add_argument("--reference-observation", type=Path, required=True)
    parser.add_argument("--field", default="chemical")
    parser.add_argument("--control-stem", default="composition")
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--title", required=True)
    arguments = parser.parse_args()

    diagnostics = arguments.output_dir / "inverse" / "diagnostics"
    controls = diagnostics / "accepted_controls"
    reference_x, reference_y, reference_initial = scalar_field(
        arguments.reference_initial_vtu, arguments.field)
    observation_x, observation_y, reference_terminal = read_observation(arguments.reference_observation)

    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for iteration in arguments.iterations:
        control = controls / f"initial_{arguments.control_stem}_iter_{iteration:03d}.vtu"
        initial_x, initial_y, initial = scalar_field(control, arguments.field)
        true_initial = resample(reference_x, reference_y, reference_initial, initial_x, initial_y)
        terminal = (arguments.output_dir / "inverse" / "replays" /
                    f"accepted_{iteration:03d}" / "terminal_composition.txt")
        terminal_x, terminal_y, final = read_observation(terminal)
        true_final = resample(observation_x, observation_y, reference_terminal,
                              terminal_x, terminal_y)
        rows.append((f"Iteration {iteration}", initial, np.abs(initial - true_initial),
                     final, np.abs(final - true_final)))

    maximum_misfit = max(float(np.max(values))
                         for row in rows for values in (row[2], row[4]))
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

    initial_image = add_panel(axes[0, 0], reference_initial, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[0, 1].axis("off")
    final_image = add_panel(axes[0, 2], reference_terminal, cmap="turbo", vmin=0.0, vmax=1.0)
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

    composition_bar = figure.colorbar(initial_image, cax=figure.add_subplot(grid[-1, :2]),
                                      orientation="horizontal")
    composition_bar.set_label("Chemical composition, $C$")
    misfit_bar = figure.colorbar(misfit_image, cax=figure.add_subplot(grid[-1, 2:]),
                                 orientation="horizontal")
    misfit_bar.set_label(r"Absolute misfit, $|C-C_{reference}|$")
    figure.suptitle(arguments.title, fontsize=14)

    destination = diagnostics / "plots" / "inversion_storyboard.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
