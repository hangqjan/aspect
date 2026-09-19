#!/usr/bin/env python3
"""Run all-cell finite-difference validation and plot scatter plus cell heatmaps."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
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


def replace_parameter(text: str, parameter: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*set\s+{re.escape(parameter)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not find unique parameter line for '{parameter}'.")
    return updated


def insert_plugin_material_model(text: str, plugin: Path) -> str:
    if "set Additional shared libraries" not in text:
        text = f"set Additional shared libraries            = {plugin}\n" + text
    else:
        text = replace_parameter(text, "Additional shared libraries", str(plugin))

    material_header = "subsection Material model\n  set Model name = simple"
    if material_header not in text:
        raise ValueError("Could not find the simple material model block.")
    return text.replace(
        material_header,
        "subsection Material model\n  set Model name = adjoint cellwise perturbation",
        1,
    )


def write_case_prm(base_text: str, prm_path: Path, output_directory: Path) -> None:
    text = replace_parameter(base_text, "Output directory", str(output_directory))
    prm_path.write_text(text)


def run_aspect(
    aspect: Path,
    prm_path: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
    skip_existing_output: bool = False,
) -> None:
    output_directory = prm_path.parent / "output"
    if skip_existing_output and (output_directory / "adjoint_kernels_rank_00000.txt").exists():
        return

    env = os.environ.copy()
    if environment is not None:
        env.update(environment)

    aspect_executable = aspect.resolve() if aspect.exists() else aspect
    with log_path.open("w") as log:
        result = subprocess.run(
            [str(aspect_executable), str(prm_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-50:])
        raise RuntimeError(f"ASPECT failed for {prm_path}:\n{tail}")


def read_objective(output_directory: Path, objective_name: str) -> float:
    kernel_file = output_directory / "adjoint_kernels_rank_00000.txt"
    with kernel_file.open() as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 3 and fields[0] == "# objective_value" and fields[1] == objective_name:
                return float(fields[2])
    raise ValueError(f"Could not find objective '{objective_name}' in {kernel_file}.")


def read_property_kernels(output_directory: Path, objective_name: str) -> list[dict[str, float | str]]:
    values: dict[tuple[str, int], dict[str, float | str]] = {}
    kernel_file = output_directory / "adjoint_kernels_rank_00000.txt"
    with kernel_file.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            row = dict(zip(KERNEL_COLUMNS, line.rstrip("\n").split("\t")))
            if row["objective"] != objective_name or row["property"] not in {"density", "viscosity"}:
                continue

            control = row["property"]
            cell = int(row["active_cell_index"])
            key = (control, cell)
            if key not in values:
                values[key] = {
                    "control": control,
                    "active_cell_index": cell,
                    "cell_volume": float(row["cell_volume"]),
                    "adjoint": 0.0,
                }
            values[key]["adjoint"] = float(values[key]["adjoint"]) + float(row["value"]) * float(row["cell_volume"])
    return sorted(values.values(), key=lambda row: (str(row["control"]), int(row["active_cell_index"])))


def l2_relative_error(fd_values: list[float], adjoint_values: list[float]) -> float:
    numerator = sum((adjoint - fd) * (adjoint - fd) for fd, adjoint in zip(fd_values, adjoint_values))
    denominator = sum(fd * fd for fd in fd_values)
    return math.sqrt(numerator / denominator) if denominator > 0.0 else 0.0


def pearson_correlation(fd_values: list[float], adjoint_values: list[float]) -> float:
    if len(fd_values) < 2:
        return 1.0
    if np.std(fd_values) == 0.0 or np.std(adjoint_values) == 0.0:
        return 1.0
    return float(np.corrcoef(fd_values, adjoint_values)[0, 1])


def cell_to_xy(cell: int, nx: int, order: str) -> tuple[int, int]:
    if order == "row-major":
        return cell % nx, cell // nx

    if nx & (nx - 1) != 0:
        raise ValueError("--index-order dealii-morton requires a power-of-two grid.")

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


def rows_to_grid(rows: list[dict[str, float | str]], control: str, column: str, cells_per_side: int, order: str) -> np.ndarray:
    grid = np.full((cells_per_side, cells_per_side), np.nan)
    for row in rows:
        if row["control"] != control:
            continue
        x, y = cell_to_xy(int(row["active_cell_index"]), cells_per_side, order)
        grid[y, x] = float(row[column])
    return grid


def comparison_metrics(rows: list[dict[str, float | str]], control: str) -> dict[str, float]:
    subset = [row for row in rows if row["control"] == control]
    fd_values = [float(row["finite_difference"]) for row in subset]
    adjoint_values = [float(row["adjoint"]) for row in subset]
    relative_errors = [float(row["relative_error"]) for row in subset]
    absolute_errors = [float(row["absolute_error"]) for row in subset]
    return {
        "count": float(len(subset)),
        "correlation": pearson_correlation(fd_values, adjoint_values),
        "relative_l2_error": l2_relative_error(fd_values, adjoint_values),
        "median_relative_error": float(np.median(relative_errors)) if relative_errors else float("nan"),
        "max_relative_error": max(relative_errors) if relative_errors else float("nan"),
        "max_absolute_error": max(absolute_errors) if absolute_errors else float("nan"),
    }


def plot_scatter(rows: list[dict[str, float | str]], controls: list[str], output: Path) -> None:
    colors = {"density": "#2a9d8f", "viscosity": "#e76f51"}
    fig, axes = plt.subplots(1, len(controls), figsize=(4.6 * len(controls), 4.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for axis, control in zip(axes, controls):
        subset = [row for row in rows if row["control"] == control]
        fd_values = [float(row["finite_difference"]) for row in subset]
        adjoint_values = [float(row["adjoint"]) for row in subset]
        metrics = comparison_metrics(rows, control)

        axis.set_facecolor("#fbfbfa")
        axis.grid(True, color="#e7e3dd", linewidth=0.7)
        axis.axhline(0.0, color="#c8c4bd", linewidth=0.8)
        axis.axvline(0.0, color="#c8c4bd", linewidth=0.8)
        axis.scatter(
            fd_values,
            adjoint_values,
            s=24,
            color=colors.get(control, "#4c78a8"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.45,
        )

        if fd_values and adjoint_values:
            bound = max(max(abs(value) for value in fd_values + adjoint_values), 1.0e-300)
            lower = -1.08 * bound
            upper = 1.08 * bound
            axis.plot([lower, upper], [lower, upper], color="#202020", linewidth=1.1, label="1:1")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")

        annotation = (
            f"n = {int(metrics['count'])}\n"
            f"r = {metrics['correlation']:.6f}\n"
            f"L2 rel. err. = {metrics['relative_l2_error']:.2e}\n"
            f"median rel. err. = {metrics['median_relative_error']:.2e}\n"
            f"max rel. err. = {metrics['max_relative_error']:.2e}"
        )
        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "#ded8cf",
                "linewidth": 0.7,
                "alpha": 0.94,
            },
        )
        axis.set_title(control.capitalize())
        axis.set_xlabel("Finite difference")
        axis.set_ylabel("Adjoint kernel prediction")
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3), useOffset=False)
        axis.legend(frameon=False, loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())


def plot_heatmaps(rows: list[dict[str, float | str]], control: str, cells_per_side: int, order: str, output: Path) -> None:
    adjoint = rows_to_grid(rows, control, "adjoint", cells_per_side, order)
    finite_difference = rows_to_grid(rows, control, "finite_difference", cells_per_side, order)
    finite = np.concatenate([adjoint[np.isfinite(adjoint)], finite_difference[np.isfinite(finite_difference)]])
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    metrics = comparison_metrics(rows, control)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.23, top=0.78, wspace=0.25)
    fig.patch.set_facecolor("white")

    for axis, grid, title in [
        (axes[0], adjoint, "Adjoint / model"),
        (axes[1], finite_difference, "Finite difference"),
    ]:
        image = axis.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            extent=(0, cells_per_side, 0, cells_per_side),
        )
        axis.set_title(title)
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")
        axis.set_xticks(np.arange(0, cells_per_side + 1, max(1, cells_per_side // 4)))
        axis.set_yticks(np.arange(0, cells_per_side + 1, max(1, cells_per_side // 4)))
        axis.set_xticks(np.arange(0, cells_per_side + 1), minor=True)
        axis.set_yticks(np.arange(0, cells_per_side + 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35)
        axis.tick_params(which="minor", length=0)
        axis.set_aspect("equal")

    fig.suptitle(f"{control} all-cell kernel on {cells_per_side}x{cells_per_side} Cartesian grid", y=0.96)
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


def parse_controls(value: str) -> list[str]:
    controls = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(controls) - {"density", "viscosity"})
    if invalid:
        raise ValueError(f"Unsupported controls: {', '.join(invalid)}")
    return controls


def parse_cells(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", type=Path, default=Path("build-release/aspect-release"))
    parser.add_argument(
        "--plugin",
        type=Path,
        default=Path("build-release/adjoint_dynamic_topography_2d_spherical_plugin/libadjoint_dynamic_topography_2d_spherical.release.so"),
    )
    parser.add_argument(
        "--prm",
        type=Path,
        default=Path("scratch/adjoint_kernel_grid/adjoint_dynamic_topography_2d_box_gaussian_16x16.prm"),
    )
    parser.add_argument("--workdir", type=Path, default=Path("build-release/fd_validation_all_cells_2d_box_gaussian_16x16"))
    parser.add_argument("--objective", default="dynamic topography")
    parser.add_argument("--controls", default="viscosity", help="Comma-separated subset of density,viscosity.")
    parser.add_argument("--cells", help="Optional comma-separated active cell indices. Defaults to every cell.")
    parser.add_argument("--density-step", type=float, default=1.0)
    parser.add_argument("--viscosity-step", type=float, default=1.0e18)
    parser.add_argument("--cells-per-side", type=int, default=16)
    parser.add_argument("--index-order", choices=["dealii-morton", "row-major"], default="dealii-morton")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv", type=Path, default=Path("scratch/adjoint_kernel_grid/all_cell_fd_validation.csv"))
    parser.add_argument("--scatter", type=Path, default=Path("scratch/adjoint_kernel_grid/all_cell_fd_validation_1to1.png"))
    parser.add_argument("--heatmap-prefix", type=Path, default=Path("scratch/adjoint_kernel_grid/all_cell_fd_validation"))
    args = parser.parse_args()

    controls = parse_controls(args.controls)
    explicit_cells = parse_cells(args.cells)
    plugin = args.plugin.resolve()
    if not plugin.exists():
        raise FileNotFoundError(f"Cellwise perturbation plugin not found: {plugin}")

    base_text = insert_plugin_material_model(args.prm.read_text(), plugin)
    args.workdir.mkdir(parents=True, exist_ok=True)

    baseline_dir = args.workdir / "baseline"
    if baseline_dir.exists() and not args.resume:
        shutil.rmtree(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_case_prm(base_text, baseline_dir / "baseline.prm", baseline_dir / "output")
    run_aspect(args.aspect, baseline_dir / "baseline.prm", baseline_dir / "aspect.log", skip_existing_output=args.resume)

    kernels = [
        row for row in read_property_kernels(baseline_dir / "output", args.objective)
        if row["control"] in controls
        and (explicit_cells is None or int(row["active_cell_index"]) in explicit_cells)
    ]
    if not kernels:
        raise ValueError("No baseline kernel rows selected.")

    rows: list[dict[str, float | str]] = []
    for kernel in kernels:
        control = str(kernel["control"])
        cell = int(kernel["active_cell_index"])
        step = args.density_step if control == "density" else args.viscosity_step
        objectives: dict[str, float] = {}

        for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
            case_name = f"{control}_{cell}_{sign_name}"
            case_dir = args.workdir / case_name
            if case_dir.exists() and not args.resume:
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            write_case_prm(base_text, case_dir / f"{case_name}.prm", case_dir / "output")
            run_aspect(
                args.aspect,
                case_dir / f"{case_name}.prm",
                case_dir / "aspect.log",
                {
                    "ASPECT_ADJOINT_FD_CELL": str(cell),
                    "ASPECT_ADJOINT_FD_PROPERTY": control,
                    "ASPECT_ADJOINT_FD_AMPLITUDE": f"{sign * step:.16g}",
                },
                skip_existing_output=args.resume,
            )
            objectives[sign_name] = read_objective(case_dir / "output", args.objective)

        finite_difference = (objectives["plus"] - objectives["minus"]) / (2.0 * step)
        adjoint = float(kernel["adjoint"])
        scale = max(abs(finite_difference), abs(adjoint), 1.0e-300)
        rows.append(
            {
                "control": control,
                "active_cell_index": cell,
                "cell_volume": float(kernel["cell_volume"]),
                "step": step,
                "objective_minus": objectives["minus"],
                "objective_plus": objectives["plus"],
                "finite_difference": finite_difference,
                "adjoint": adjoint,
                "absolute_error": abs(finite_difference - adjoint),
                "relative_error": abs(finite_difference - adjoint) / scale,
            }
        )
        print(
            f"{control} cell {cell}: FD={finite_difference:.8e}, "
            f"adjoint={adjoint:.8e}, rel={rows[-1]['relative_error']:.3e}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_scatter(rows, controls, args.scatter)
    for control in controls:
        heatmap_output = args.heatmap_prefix.with_name(f"{args.heatmap_prefix.name}_{control}_heatmaps.png")
        plot_heatmaps(
            rows,
            control,
            args.cells_per_side,
            args.index_order,
            heatmap_output,
        )
        print(f"Wrote {heatmap_output}")

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.scatter}")


if __name__ == "__main__":
    main()
