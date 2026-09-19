#!/usr/bin/env python3
"""Plot cellwise adjoint/FD heatmaps from run_fd_validation.py CSV output."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, float | str] = {"control": row["control"]}
            for key, value in row.items():
                if key == "control":
                    continue
                parsed[key] = int(value) if key == "active_cell_index" else float(value)
            rows.append(parsed)
    return rows


def parse_shape(value: str, cell_count: int) -> tuple[int, int]:
    if "x" in value:
        rows_text, cols_text = value.lower().split("x", 1)
        nrows = int(rows_text)
        ncols = int(cols_text)
        if nrows * ncols != cell_count:
            raise ValueError(f"--shape {value} does not hold {cell_count} cells.")
        return nrows, ncols

    ncols = int(value)
    if cell_count % ncols != 0:
        raise ValueError(f"{cell_count} cells cannot be reshaped into {ncols} columns.")
    return cell_count // ncols, ncols


def infer_shape(cell_count: int) -> tuple[int, int]:
    root = int(math.sqrt(cell_count))
    candidates = [(cell_count // divisor, divisor) for divisor in range(1, root + 1) if cell_count % divisor == 0]
    return min(candidates, key=lambda shape: abs(shape[0] - shape[1]))


def metrics(rows: list[dict[str, float | str]], control: str) -> dict[str, float]:
    subset = [row for row in rows if row["control"] == control]
    fd = np.array([float(row["finite_difference"]) for row in subset])
    adjoint = np.array([float(row["adjoint"]) for row in subset])
    relative_error = np.array([float(row["relative_error"]) for row in subset])
    absolute_error = np.array([float(row["absolute_error"]) for row in subset])
    corr = float(np.corrcoef(fd, adjoint)[0, 1]) if fd.size > 1 and np.std(fd) > 0.0 and np.std(adjoint) > 0.0 else 1.0
    l2 = float(np.linalg.norm(adjoint - fd) / np.linalg.norm(fd)) if np.linalg.norm(fd) > 0.0 else 0.0
    return {
        "count": float(fd.size),
        "correlation": corr,
        "relative_l2_error": l2,
        "median_relative_error": float(np.median(relative_error)),
        "max_relative_error": float(np.max(relative_error)),
        "max_absolute_error": float(np.max(absolute_error)),
    }


def values_to_grid(rows: list[dict[str, float | str]], control: str, column: str, shape: tuple[int, int]) -> np.ndarray:
    nrows, ncols = shape
    grid = np.full((nrows, ncols), np.nan)
    for row in rows:
        if row["control"] != control:
            continue
        cell = int(row["active_cell_index"])
        grid[cell // ncols, cell % ncols] = float(row[column])
    return grid


def local_morton_x(local_cell: int, cells_per_coarse_sector: int) -> int:
    side = int(math.sqrt(cells_per_coarse_sector))
    if side * side != cells_per_coarse_sector:
        raise ValueError("cells per coarse angular sector must be a square number.")

    levels = int(math.log2(side))
    x = 0
    remaining = local_cell
    for level in range(levels):
        child = remaining % 4
        remaining //= 4
        bit = 1 << (levels - level - 1)
        x += (child & 1) * bit
    return x


def expected_spherical_cell_volumes(inner_radius: float, outer_radius: float, radial_cells: int, angular_cells: int) -> list[float]:
    radial_edges = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    angular_width = 2.0 * math.pi / angular_cells
    return [
        0.5 * (radial_edges[index + 1] ** 2 - radial_edges[index] ** 2) * angular_width
        for index in range(radial_cells)
    ]


def spherical_values_to_grid(
    rows: list[dict[str, float | str]],
    control: str,
    column: str,
    inner_radius: float,
    outer_radius: float,
    radial_cells: int,
    angular_cells: int,
) -> np.ndarray:
    grid = np.full((radial_cells, angular_cells), np.nan)
    volumes = expected_spherical_cell_volumes(inner_radius, outer_radius, radial_cells, angular_cells)
    cells_per_coarse_sector = radial_cells * radial_cells
    angular_children_per_coarse_sector = radial_cells

    for row in rows:
        if row["control"] != control:
            continue
        cell = int(row["active_cell_index"])
        volume = float(row["cell_volume"])
        radial_index = int(np.argmin([abs(volume - expected) for expected in volumes]))
        coarse_sector = cell // cells_per_coarse_sector
        local_cell = cell % cells_per_coarse_sector
        angular_index = coarse_sector * angular_children_per_coarse_sector + local_morton_x(local_cell, cells_per_coarse_sector)
        if 0 <= angular_index < angular_cells:
            grid[radial_index, angular_index] = float(row[column])
    return grid


def plot_control(rows: list[dict[str, float | str]], control: str, shape: tuple[int, int], output: Path) -> None:
    adjoint = values_to_grid(rows, control, "adjoint", shape)
    fd = values_to_grid(rows, control, "finite_difference", shape)
    finite = np.concatenate([adjoint[np.isfinite(adjoint)], fd[np.isfinite(fd)]])
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    summary = metrics(rows, control)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.22, top=0.78, wspace=0.25)
    fig.patch.set_facecolor("white")

    for axis, grid, title in [
        (axes[0], adjoint, "Adjoint / model"),
        (axes[1], fd, "Finite difference"),
    ]:
        image = axis.imshow(grid, origin="lower", cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("logical cell column")
        axis.set_ylabel("logical cell row")
        axis.set_aspect("equal")
        axis.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35)
        axis.tick_params(which="minor", length=0)

    fig.suptitle(f"2D spherical benchmark: {control} cellwise kernel", y=0.96)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("cell-integrated derivative")
    metric_text = (
        f"n = {int(summary['count'])}    "
        f"r = {summary['correlation']:.6f}    "
        f"L2 rel. err. = {summary['relative_l2_error']:.3e}    "
        f"median rel. err. = {summary['median_relative_error']:.3e}    "
        f"max rel. err. = {summary['max_relative_error']:.3e}    "
        f"max abs. err. = {summary['max_absolute_error']:.3e}"
    )
    fig.text(0.5, 0.055, metric_text, ha="center", va="bottom", fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())


def color_limit(adjoint: np.ndarray, fd: np.ndarray, percentile: float) -> float:
    finite = np.concatenate([adjoint[np.isfinite(adjoint)], fd[np.isfinite(fd)]])
    if not finite.size:
        return 1.0
    if not 0.0 < percentile <= 100.0:
        raise ValueError("--symmetric-percentile must be in (0, 100].")
    limit = float(np.percentile(np.abs(finite), percentile))
    return limit if limit > 0.0 else float(np.max(np.abs(finite)))


def plot_spherical_control(
    rows: list[dict[str, float | str]],
    control: str,
    inner_radius: float,
    outer_radius: float,
    radial_cells: int,
    angular_cells: int,
    symmetric_percentile: float,
    output: Path,
) -> None:
    adjoint = spherical_values_to_grid(rows, control, "adjoint", inner_radius, outer_radius, radial_cells, angular_cells)
    fd = spherical_values_to_grid(rows, control, "finite_difference", inner_radius, outer_radius, radial_cells, angular_cells)
    limit = color_limit(adjoint, fd, symmetric_percentile)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    summary = metrics(rows, control)

    theta_edges = np.linspace(0.0, 2.0 * math.pi, angular_cells + 1)
    radial_edges = np.linspace(inner_radius, outer_radius, radial_cells + 1)
    theta_grid, radial_grid = np.meshgrid(theta_edges, radial_edges)
    x_grid = radial_grid * np.cos(theta_grid)
    y_grid = radial_grid * np.sin(theta_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.18, top=0.84, wspace=0.18)
    fig.patch.set_facecolor("white")

    for axis, grid, title in [
        (axes[0], adjoint, "Adjoint / model"),
        (axes[1], fd, "Finite difference"),
    ]:
        image = axis.pcolormesh(x_grid, y_grid, grid, cmap=cmap, vmin=-limit, vmax=limit, shading="flat")
        axis.add_patch(Circle((0.0, 0.0), inner_radius, fill=False, color="#333333", linewidth=0.7))
        axis.add_patch(Circle((0.0, 0.0), outer_radius, fill=False, color="#333333", linewidth=0.7))
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-outer_radius, outer_radius)
        axis.set_ylim(-outer_radius, outer_radius)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(6, 6))
        axis.grid(color="#eeeeee", linewidth=0.45)

    fig.suptitle(f"2D spherical benchmark: {control} cellwise kernel", y=0.98)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.04)
    colorbar.set_label("cell-integrated derivative")
    scale_text = "full symmetric range" if symmetric_percentile == 100.0 else f"symmetric {symmetric_percentile:g}th percentile range"
    metric_text = (
        f"{scale_text}    "
        f"n = {int(summary['count'])}    "
        f"r = {summary['correlation']:.6f}    "
        f"L2 rel. err. = {summary['relative_l2_error']:.3e}    "
        f"median rel. err. = {summary['median_relative_error']:.3e}    "
        f"max rel. err. = {summary['max_relative_error']:.3e}    "
        f"max abs. err. = {summary['max_absolute_error']:.3e}"
    )
    fig.text(0.5, 0.04, metric_text, ha="center", va="bottom", fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--geometry", choices=["logical", "spherical"], default="logical")
    parser.add_argument("--shape", help="Logical heatmap shape as rowsxcols. Defaults to closest factor pair.")
    parser.add_argument("--inner-radius", type=float, default=3480000.0)
    parser.add_argument("--outer-radius", type=float, default=6336000.0)
    parser.add_argument("--radial-cells", type=int, default=4)
    parser.add_argument("--angular-cells", type=int, default=48)
    parser.add_argument("--symmetric-percentile", type=float, default=100.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scratch/adjoint_kernel_grid/spherical_all_cell"))
    args = parser.parse_args()

    rows = read_rows(args.csv)
    controls = sorted({str(row["control"]) for row in rows})

    for control in controls:
        output = args.output_prefix.with_name(f"{args.output_prefix.name}_{control}_heatmaps.png")
        if args.geometry == "spherical":
            plot_spherical_control(
                rows,
                control,
                args.inner_radius,
                args.outer_radius,
                args.radial_cells,
                args.angular_cells,
                args.symmetric_percentile,
                output,
            )
        else:
            cell_count = max(int(row["active_cell_index"]) for row in rows) + 1
            shape = parse_shape(args.shape, cell_count) if args.shape is not None else infer_shape(cell_count)
            plot_control(rows, control, shape, output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
