#!/usr/bin/env python3
"""Plot Cartesian cellwise adjoint and finite-difference kernels as heat maps."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


KERNEL_COLUMNS = [
    "objective",
    "term",
    "property",
    "active_cell_index",
    "cell_volume",
    "value",
]

FD_CHECK_COLUMNS = [
    "objective",
    "control",
    "pattern",
    "active_cell_index",
    "cell_group",
    "step",
    "base_objective",
    "perturbed_objective",
    "primary_finite_difference_derivative",
    "frozen_forward_derivative",
    "finite_difference_derivative",
    "baseline_material_volume_derivative",
    "volume_split_difference",
    "adjoint_derivative",
    "full_control_gradient_derivative",
    "included_adjoint_term_derivative",
    "control_gradient_breakdown_difference",
    "absolute_error",
    "objective_scaled_error",
    "relative_error",
]


def read_kernel_file(path: Path, control: str, objective: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            row = dict(zip(KERNEL_COLUMNS, line.rstrip("\n").split("\t")))
            if row["objective"] != objective or row["property"] != control:
                continue
            cell = int(row["active_cell_index"])
            contribution = float(row["value"]) * float(row["cell_volume"])
            values[cell] = values.get(cell, 0.0) + contribution
    return values


def read_fd_checks_file(
    path: Path,
    control: str,
    objective: str,
    fd_column: str,
    adjoint_column: str,
) -> tuple[dict[int, float], dict[int, float]]:
    finite_difference: dict[int, float] = {}
    adjoint: dict[int, float] = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            row = dict(zip(FD_CHECK_COLUMNS, line.rstrip("\n").split("\t")))
            if row["objective"] != objective or row["control"] != control:
                continue
            cell = int(row["active_cell_index"])
            finite_difference[cell] = float(row[fd_column])
            adjoint[cell] = float(row[adjoint_column])
    return finite_difference, adjoint


def read_validation_csv(
    path: Path,
    control: str,
    fd_column: str,
    adjoint_column: str,
) -> tuple[dict[int, float], dict[int, float]]:
    finite_difference: dict[int, float] = {}
    adjoint: dict[int, float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["control"] != control:
                continue
            cell = int(row["active_cell_index"])
            finite_difference[cell] = float(row[fd_column])
            adjoint[cell] = float(row[adjoint_column])
    return finite_difference, adjoint


def grid_shape(cell_count: int, requested: int | None) -> tuple[int, int]:
    if requested is not None:
        return requested, requested
    side = math.isqrt(cell_count)
    if side * side != cell_count:
        raise ValueError(
            f"Cannot infer a square grid from {cell_count} cells; pass --cells-per-side."
        )
    return side, side


def cell_to_xy(cell: int, nx: int, ny: int, order: str) -> tuple[int, int]:
    if order == "row-major":
        return cell % nx, cell // nx

    if nx != ny or nx & (nx - 1) != 0:
        raise ValueError("--index-order dealii-morton requires a square power-of-two grid.")

    levels = int(math.log2(nx))
    x = 0
    y = 0
    remaining = cell
    for level in range(levels):
        child = remaining % 4
        remaining //= 4
        bit = 1 << (levels - level - 1)
        x += (child & 1) * bit
        y += ((child >> 1) & 1) * bit
    return x, y


def values_to_grid(values: dict[int, float], nx: int, ny: int, order: str) -> np.ndarray:
    grid = np.full((ny, nx), np.nan)
    for cell, value in values.items():
        x, y = cell_to_xy(cell, nx, ny, order)
        if 0 <= x < nx and 0 <= y < ny:
            grid[y, x] = value
    return grid


def symmetric_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite_values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if finite_values.size == 0:
        return -1.0, 1.0
    limit = float(np.nanmax(np.abs(finite_values)))
    return -limit, limit


def paired_values(adjoint_grid: np.ndarray, fd_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(adjoint_grid) & np.isfinite(fd_grid)
    return adjoint_grid[mask], fd_grid[mask]


def comparison_metrics(adjoint_grid: np.ndarray, fd_grid: np.ndarray) -> dict[str, float]:
    adjoint, finite_difference = paired_values(adjoint_grid, fd_grid)
    if adjoint.size == 0:
        return {
            "count": 0,
            "correlation": np.nan,
            "relative_l2_error": np.nan,
            "max_relative_error": np.nan,
            "median_relative_error": np.nan,
            "max_absolute_error": np.nan,
        }

    difference = adjoint - finite_difference
    scale = np.maximum.reduce([
        np.abs(adjoint),
        np.abs(finite_difference),
        np.full_like(adjoint, 1.0e-300),
    ])
    relative_error = np.abs(difference) / scale
    denominator = np.linalg.norm(finite_difference)
    if denominator > 0.0:
        relative_l2_error = np.linalg.norm(difference) / denominator
    else:
        relative_l2_error = 0.0

    if adjoint.size > 1 and np.std(adjoint) > 0.0 and np.std(finite_difference) > 0.0:
        correlation = float(np.corrcoef(adjoint, finite_difference)[0, 1])
    else:
        correlation = 1.0

    return {
        "count": float(adjoint.size),
        "correlation": correlation,
        "relative_l2_error": float(relative_l2_error),
        "max_relative_error": float(np.max(relative_error)),
        "median_relative_error": float(np.median(relative_error)),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def relative_error_grid(adjoint_grid: np.ndarray, fd_grid: np.ndarray) -> np.ndarray:
    scale = np.maximum.reduce([
        np.abs(adjoint_grid),
        np.abs(fd_grid),
        np.full_like(adjoint_grid, 1.0e-300),
    ])
    return np.abs(adjoint_grid - fd_grid) / scale


def plot_pair(
    adjoint_grid: np.ndarray,
    fd_grid: np.ndarray,
    output: Path,
    title: str,
    colormap: str,
    error_output: Path | None,
) -> None:
    vmin, vmax = symmetric_limits(adjoint_grid, fd_grid)
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad("#eeeeee")
    metrics = comparison_metrics(adjoint_grid, fd_grid)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.23, top=0.78, wspace=0.25)

    for axis, grid, label in [
        (axes[0], adjoint_grid, "Adjoint / model"),
        (axes[1], fd_grid, "Finite difference"),
    ]:
        image = axis.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(0, grid.shape[1], 0, grid.shape[0]),
        )
        axis.set_title(label)
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")
        axis.set_xticks(np.arange(0, grid.shape[1] + 1, max(1, grid.shape[1] // 4)))
        axis.set_yticks(np.arange(0, grid.shape[0] + 1, max(1, grid.shape[0] // 4)))
        axis.set_xticks(np.arange(0, grid.shape[1] + 1), minor=True)
        axis.set_yticks(np.arange(0, grid.shape[0] + 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.45)
        axis.tick_params(which="minor", length=0)
        axis.set_aspect("equal")

    fig.suptitle(title, y=0.96)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("cell-integrated derivative")
    metric_text = (
        f"n = {int(metrics['count'])}    "
        f"r = {metrics['correlation']:.6f}    "
        f"L2 rel. err. = {metrics['relative_l2_error']:.3e}    "
        f"median rel. err. = {metrics['median_relative_error']:.3e}    "
        f"max rel. err. = {metrics['max_relative_error']:.3e}    "
        f"max abs. err. = {metrics['max_absolute_error']:.3e}"
    )
    fig.text(0.5, 0.055, metric_text, ha="center", va="bottom", fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())

    if error_output is not None:
        error = relative_error_grid(adjoint_grid, fd_grid)
        error_cmap = plt.get_cmap("magma").copy()
        error_cmap.set_bad("#eeeeee")
        finite_error = error[np.isfinite(error)]
        error_max = float(np.nanmax(finite_error)) if finite_error.size else 1.0
        fig_error, axis = plt.subplots(1, 1, figsize=(4.9, 4.4), constrained_layout=True)
        error_image = axis.imshow(
            error,
            origin="lower",
            cmap=error_cmap,
            vmin=0.0,
            vmax=error_max,
            interpolation="nearest",
            extent=(0, error.shape[1], 0, error.shape[0]),
        )
        axis.set_title("Relative error")
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")
        axis.set_xticks(np.arange(0, error.shape[1] + 1, max(1, error.shape[1] // 4)))
        axis.set_yticks(np.arange(0, error.shape[0] + 1, max(1, error.shape[0] // 4)))
        axis.set_xticks(np.arange(0, error.shape[1] + 1), minor=True)
        axis.set_yticks(np.arange(0, error.shape[0] + 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.45)
        axis.tick_params(which="minor", length=0)
        axis.set_aspect("equal")
        fig_error.colorbar(error_image, ax=axis, shrink=0.86, pad=0.02, label="relative error")
        error_output.parent.mkdir(parents=True, exist_ok=True)
        fig_error.savefig(error_output, dpi=220, facecolor=fig_error.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-file", type=Path)
    parser.add_argument("--fd-checks-file", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--control", default="density", choices=["density", "viscosity"])
    parser.add_argument("--objective", default="dynamic topography")
    parser.add_argument("--cells-per-side", type=int, default=16)
    parser.add_argument(
        "--index-order",
        choices=["dealii-morton", "row-major"],
        default="dealii-morton",
    )
    parser.add_argument(
        "--fd-column",
        default="primary_finite_difference_derivative",
        help="Column used from --fd-checks-file or --validation-csv for the finite-difference panel.",
    )
    parser.add_argument(
        "--adjoint-column",
        default="full_control_gradient_derivative",
        help="Column used from --fd-checks-file when --kernel-file is not supplied.",
    )
    parser.add_argument("--output", type=Path, default=Path("scratch/adjoint_kernel_grid/kernel_grid.png"))
    parser.add_argument(
        "--error-output",
        type=Path,
        help="Optional path for an additional relative-error heatmap.",
    )
    parser.add_argument("--colormap", default="RdBu_r")
    args = parser.parse_args()

    fd_values: dict[int, float] = {}
    fd_adjoint_values: dict[int, float] = {}
    if args.fd_checks_file is not None:
        fd_values, fd_adjoint_values = read_fd_checks_file(
            args.fd_checks_file,
            args.control,
            args.objective,
            args.fd_column,
            args.adjoint_column,
        )
    elif args.validation_csv is not None:
        fd_values, fd_adjoint_values = read_validation_csv(
            args.validation_csv,
            args.control,
            "finite_difference" if args.fd_column == "primary_finite_difference_derivative" else args.fd_column,
            "adjoint" if args.adjoint_column == "full_control_gradient_derivative" else args.adjoint_column,
        )

    if args.kernel_file is not None:
        adjoint_values = read_kernel_file(args.kernel_file, args.control, args.objective)
    else:
        adjoint_values = fd_adjoint_values

    if not adjoint_values:
        raise ValueError("No adjoint/model values were found.")
    if not fd_values:
        raise ValueError("No finite-difference values were found.")

    nx, ny = grid_shape(max(max(adjoint_values), max(fd_values)) + 1, args.cells_per_side)
    adjoint_grid = values_to_grid(adjoint_values, nx, ny, args.index_order)
    fd_grid = values_to_grid(fd_values, nx, ny, args.index_order)
    plot_pair(
        adjoint_grid,
        fd_grid,
        args.output,
        f"{args.objective}: {args.control} kernel on {nx}x{ny} Cartesian grid",
        args.colormap,
        args.error_output,
    )
    print(f"Wrote {args.output}")
    if args.error_output is not None:
        print(f"Wrote {args.error_output}")


if __name__ == "__main__":
    main()
