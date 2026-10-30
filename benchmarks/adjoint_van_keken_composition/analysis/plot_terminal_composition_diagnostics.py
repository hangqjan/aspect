#!/usr/bin/env python3
"""Plot final terminal-composition agreement and objective convergence."""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vtk_io import read_vtu


OBJECTIVE = re.compile(r"Time-dependent terminal objective 'composition field' = ([^\s]+)")


def structured_observation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the benchmark's x-y-chemical structured observation file."""
    rows = np.loadtxt(path, comments="#", skiprows=3)
    x = np.unique(rows[:, 0])
    y = np.unique(rows[:, 1])
    if rows.shape[0] != len(x) * len(y):
        raise ValueError(f"{path}: not a complete structured observation")
    return x, y, rows[:, 2].reshape((len(y), len(x)))


def distributed_scalar_grid(pvtu: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a scalar point grid from an ASPECT parallel VTU collection."""
    root = ET.parse(pvtu).getroot()
    sources = [piece.attrib["Source"] for piece in root.findall(".//Piece")]
    if not sources:
        raise ValueError(f"{pvtu}: contains no VTU pieces")

    points = []
    values = []
    for source in sources:
        data = read_vtu(pvtu.parent / source)
        points.append(data["Points"])
        values.append(data[field].reshape(-1))
    coordinates = np.concatenate(points)
    scalars = np.concatenate(values)
    x = np.unique(coordinates[:, 0])
    y = np.unique(coordinates[:, 1])
    grid = np.full((len(y), len(x)), np.nan)
    ix = {round(float(value), 12): index for index, value in enumerate(x)}
    iy = {round(float(value), 12): index for index, value in enumerate(y)}
    for point, value in zip(coordinates, scalars, strict=True):
        row, column = iy[round(float(point[1]), 12)], ix[round(float(point[0]), 12)]
        previous = grid[row, column]
        if not math.isnan(previous) and not math.isclose(previous, value, abs_tol=2e-6):
            raise ValueError(f"{pvtu}: inconsistent shared value at {point[:2]}")
        grid[row, column] = value
    if np.isnan(grid).any():
        raise ValueError(f"{pvtu}: {field} is not a complete point grid")
    return x, y, grid


def sample_structured_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray,
                           target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a regular scalar grid onto a second regular grid."""
    along_x = np.vstack([np.interp(target_x, x, row) for row in grid])
    return np.vstack([np.interp(target_y, y, along_x[:, column])
                      for column in range(len(target_x))]).T


def final_solution_pvtu(output: Path) -> Path:
    """Return the last standard ASPECT parallel-VTU collection in an output directory."""
    candidates = list((output / "solution").glob("solution-*.pvtu"))
    if not candidates:
        raise FileNotFoundError(f"No parallel VTU solution files in {output / 'solution'}")
    return max(candidates, key=lambda path: int(path.stem.rsplit("-", 1)[1]))


def terminal_figure(output: Path, observation: Path, plot_directory: Path) -> None:
    observation_x, observation_y, observed = structured_observation(observation)
    x, y, predicted = distributed_scalar_grid(final_solution_pvtu(output), "chemical")
    observed_on_solution_grid = sample_structured_grid(observation_x, observation_y, observed, x, y)
    error = predicted - observed_on_solution_grid
    maximum = max(float(np.max(np.abs(error))), 1e-12)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), layout="constrained")
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    image = axes[0].imshow(observed, origin="lower",
                           extent=(float(observation_x[0]), float(observation_x[-1]),
                                   float(observation_y[0]), float(observation_y[-1])),
                           vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title("terminal observation")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    image = axes[1].imshow(predicted, origin="lower", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set_title("final terminal prediction")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    error_image = axes[2].imshow(error, origin="lower", extent=extent, vmin=-maximum, vmax=maximum,
                                 cmap="coolwarm")
    axes[2].set_title("prediction − observation")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("z")
    figure.colorbar(image, ax=axes[:2], shrink=0.85, label="chemical composition")
    figure.colorbar(error_image, ax=axes[2], shrink=0.85, label="chemical-composition error")
    plot_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_directory / "terminal_composition_comparison.png", dpi=180)
    plt.close(figure)


def objective_figure(output: Path, plot_directory: Path) -> None:
    values = [float(match.group(1)) for match in OBJECTIVE.finditer((output / "log.txt").read_text())]
    if len(values) < 2:
        raise ValueError(f"{output}/log.txt: fewer than two optimizer objective values")
    figure, axis = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    iterations = np.arange(len(values))
    axis.semilogy(iterations, values, marker="o")
    axis.set_xlabel("accepted forward evaluation")
    axis.set_ylabel("terminal-composition objective")
    axis.set_title("Objective convergence")
    axis.grid(True, which="both", alpha=0.3)
    plot_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_directory / "terminal_composition_objective.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True, help="ASPECT inverse output directory")
    parser.add_argument("--observation", type=Path, required=True, help="structured terminal-composition observation")
    parser.add_argument("--plot-dir", type=Path,
                        help="directory for generated PNG diagnostics")
    args = parser.parse_args()
    plot_directory = args.plot_dir or args.output_dir / "diagnostics/plots"
    terminal_figure(args.output_dir, args.observation, plot_directory)
    objective_figure(args.output_dir, plot_directory)


if __name__ == "__main__":
    main()
