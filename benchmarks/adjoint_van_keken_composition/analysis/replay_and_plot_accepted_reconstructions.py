#!/usr/bin/env python3
"""Replay selected accepted controls and plot four-column twin reconstructions."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_terminal_composition_diagnostics import structured_observation
from vtk_io import scalar_grid


def reference_initial_composition(x: np.ndarray, y: np.ndarray, truth: str) -> np.ndarray:
    """Return the analytic initial composition for a supported twin benchmark."""
    xx, yy = np.meshgrid(x, y)
    if truth == "van-keken":
        return 0.5 * (1.0 + np.tanh((0.2 + 0.02 * np.cos(np.pi * xx / 0.9142) - yy) / 0.02))
    if truth == "gadopt":
        erf = np.vectorize(math.erf)
        return (0.5 * (erf((1.0 - yy) * 3.0) + erf(-3.0 * yy) + 1.0)
                + 0.1 * np.exp(-0.5 * ((xx - 0.5) ** 2 + (yy - 0.2) ** 2) / (0.1 ** 2)))
    raise ValueError(f"Unknown analytic truth: {truth}")


def resample_structured_grid(source_x: np.ndarray, source_y: np.ndarray, values: np.ndarray,
                             target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Linearly resample a rectilinear reference grid onto the control grid."""
    along_x = np.asarray([np.interp(target_x, source_x, row) for row in values])
    return np.asarray([np.interp(target_y, source_y, along_x[:, column])
                       for column in range(len(target_x))]).T


def write_ascii_control(vtu: Path, destination: Path, vtu_field_name: str,
                        ascii_field_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an accepted ASPECT VTU control into its ascii-data input form."""
    x, y, control = scalar_grid(vtu, vtu_field_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as output:
        output.write(f"# Accepted initial {ascii_field_name} control exported for forward replay.\n")
        output.write(f"# POINTS: {len(x)} {len(y)}\n")
        output.write(f"# Columns: x y {ascii_field_name}\n")
        for row, coordinate_y in enumerate(y):
            for column, coordinate_x in enumerate(x):
                output.write(f"{coordinate_x:.17g} {coordinate_y:.17g} {control[row, column]:.17g}\n")
    return x, y, control


def write_replay_prm(reference_prm: Path, output: Path, control: Path, destination: Path,
                     end_step: int | None, initial_model_kind: str,
                     terminal_postprocessor: str) -> None:
    """Create a forward-only replay input without overwriting original results."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    termination = ("subsection Termination criteria\n"
                   f"  set End step = {end_step}\n"
                   "end\n\n" if end_step is not None else "")
    initial_model = "Initial composition model" if initial_model_kind == "composition" else "Initial temperature model"
    destination.write_text(
        f"include {reference_prm.resolve()}\n\n"
        f"set Output directory = {output.resolve()}\n\n"
        f"subsection {initial_model}\n"
        "  set Model name = ascii data\n"
        "  subsection Ascii data model\n"
        f"    set Data directory = {control.parent.resolve()}\n"
        f"    set Data file name = {control.name}\n"
        "  end\n"
        "end\n\n"
        + termination
        +
        "subsection Postprocess\n"
        f"  set List of postprocessors = {terminal_postprocessor}\n"
        "end\n"
    )


def replay_controls(aspect: Path, inverse_output: Path, reference_prm: Path,
                    work_directory: Path, iterations: list[int], end_step: int | None,
                    vtu_field_name: str, control_name: str, ascii_field_name: str,
                    initial_model_kind: str, terminal_postprocessor: str,
                    observation_file_name: str) -> list[tuple[int, Path, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Replay each requested control only when its terminal observation is absent."""
    controls = inverse_output / "diagnostics/accepted_controls"
    result = []
    for iteration in iterations:
        vtu = controls / f"initial_{control_name}_iter_{iteration:03d}.vtu"
        if not vtu.is_file():
            raise FileNotFoundError(vtu)
        iteration_directory = work_directory / f"accepted_{iteration:03d}"
        observation = iteration_directory / observation_file_name
        control = work_directory / "controls" / f"accepted_{iteration:03d}.txt"
        prm = work_directory / "inputs" / f"accepted_{iteration:03d}.prm"
        field = write_ascii_control(vtu, control, vtu_field_name, ascii_field_name)
        if not observation.is_file():
            if iteration_directory.exists():
                raise RuntimeError(f"Incomplete replay output exists: {iteration_directory}")
            write_replay_prm(reference_prm, iteration_directory, control, prm, end_step,
                             initial_model_kind, terminal_postprocessor)
            with (work_directory / f"accepted_{iteration:03d}.log").open("w") as log:
                subprocess.run([str(aspect.resolve()), str(prm.resolve())],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
        result.append((iteration, observation, field))
    return result


def plot_reconstructions(replays: list[tuple[int, Path, tuple[np.ndarray, np.ndarray, np.ndarray]]],
                         reference_observation: Path, destination: Path, label: str, truth: str,
                         reference_initial_vtu: Path | None, vtu_field_name: str,
                         field_label: str, title: str | None) -> None:
    """Render reference plus accepted controls in the requested four-column layout."""
    if not replays:
        raise ValueError("At least one accepted control is required")
    observation_x, observation_y, terminal_reference = structured_observation(reference_observation)
    initial_x, initial_y, _ = replays[0][2]
    if reference_initial_vtu is None:
        initial_reference = reference_initial_composition(initial_x, initial_y, truth)
    else:
        reference_x, reference_y, initial_reference = scalar_grid(reference_initial_vtu, vtu_field_name)
        if not (np.array_equal(reference_x, initial_x) and np.array_equal(reference_y, initial_y)):
            initial_reference = resample_structured_grid(reference_x, reference_y, initial_reference,
                                                         initial_x, initial_y)
    initial_errors = []
    terminal_fields = []
    terminal_errors = []
    for _, observation, (x, y, control) in replays:
        if not (np.array_equal(x, initial_x) and np.array_equal(y, initial_y)):
            raise ValueError("Accepted controls use inconsistent initial grids")
        initial_errors.append(np.abs(control - initial_reference))
        terminal_x, terminal_y, terminal = structured_observation(observation)
        if not (np.array_equal(terminal_x, observation_x) and np.array_equal(terminal_y, observation_y)):
            raise ValueError(f"{observation}: terminal grid differs from the reference observation")
        terminal_fields.append(terminal)
        terminal_errors.append(np.abs(terminal - terminal_reference))
    initial_error_max = max(float(np.max(error)) for error in initial_errors)
    terminal_error_max = max(float(np.max(error)) for error in terminal_errors)
    rows = len(replays) + 1
    figure = plt.figure(figsize=(12.4, 2.75 * rows), layout="constrained")
    # Reserve a dedicated top row for the figure title so it cannot overlap
    # the boxed column headings.
    grid_spec = figure.add_gridspec(rows + 2, 4,
                                    height_ratios=[0.35] + [1.0] * rows + [0.045])
    axes = np.asarray([[figure.add_subplot(grid_spec[row + 1, column]) for column in range(4)]
                       for row in range(rows)])
    titles = ("Initial condition", "Initial absolute misfit", "Final condition", "Final absolute misfit")
    for axis, heading in zip(axes[0], titles, strict=True):
        axis.set_title(heading, fontsize=12, pad=12,
                       bbox={"facecolor": "white", "edgecolor": "black", "pad": 3.5})
    initial_extent = (float(initial_x[0]), float(initial_x[-1]), float(initial_y[0]), float(initial_y[-1]))
    terminal_extent = (float(observation_x[0]), float(observation_x[-1]),
                       float(observation_y[0]), float(observation_y[-1]))
    initial_image = axes[0, 0].imshow(initial_reference, origin="lower",
                                      extent=initial_extent, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].imshow(terminal_reference, origin="lower", extent=terminal_extent,
                      cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 0].set_ylabel("Reference")
    for row, ((iteration, _, (_, _, control)), initial_error, terminal, terminal_error) in enumerate(
            zip(replays, initial_errors, terminal_fields, terminal_errors, strict=True), start=1):
        axes[row, 0].imshow(control, origin="lower", extent=initial_extent, cmap="viridis", vmin=0.0, vmax=1.0)
        error_image = axes[row, 1].imshow(initial_error, origin="lower", extent=initial_extent,
                                          cmap="Greys", vmin=0.0, vmax=initial_error_max)
        axes[row, 2].imshow(terminal, origin="lower", extent=terminal_extent, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[row, 3].imshow(terminal_error, origin="lower", extent=terminal_extent,
                            cmap="Greys", vmin=0.0, vmax=terminal_error_max)
        axes[row, 0].set_ylabel(f"Iteration {iteration}")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_xticks(())
        axis.set_yticks(())
    for axis in (axes[0, 1], axes[0, 3]):
        for spine in axis.spines.values():
            spine.set_visible(False)
    composition_colorbar = figure.add_subplot(grid_spec[-1, :2])
    misfit_colorbar = figure.add_subplot(grid_spec[-1, 2:])
    figure.colorbar(initial_image, cax=composition_colorbar, orientation="horizontal",
                    label=field_label)
    figure.colorbar(error_image, cax=misfit_colorbar, orientation="horizontal",
                    label=f"absolute {field_label} misfit")
    figure_title = ("van Keken terminal-composition inversion: accepted reconstructions "
             f"({label})"
             if truth == "van-keken"
             else "G-ADOPT composition terminal-composition inversion: "
                  "accepted reconstructions")
    figure.suptitle(title if title is not None else figure_title, fontsize=14, y=0.995)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", type=Path, required=True)
    parser.add_argument("--inverse-output", type=Path, required=True)
    parser.add_argument("--reference-prm", type=Path, required=True)
    parser.add_argument("--reference-observation", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--truth", choices=("van-keken", "gadopt"), default="van-keken")
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    parser.add_argument("--end-step", type=int,
                        help="Override the reference input's end step in replay inputs.")
    parser.add_argument("--reference-initial-vtu", type=Path,
                        help="VTU field used for the reference initial-condition panel.")
    parser.add_argument("--vtu-field-name", default="chemical",
                        help="Scalar field name in accepted-control and reference VTU files.")
    parser.add_argument("--control-name", default="chemical",
                        help="Control name used in accepted-control filenames.")
    parser.add_argument("--ascii-field-name", default="chemical",
                        help="Scalar column name written in replay initial-condition files.")
    parser.add_argument("--initial-model-kind", choices=("composition", "temperature"),
                        default="composition", help="Initial model overridden in forward replay inputs.")
    parser.add_argument("--field-label", default="chemical composition",
                        help="Label used for the reconstruction colour bars.")
    parser.add_argument("--terminal-postprocessor", default="terminal composition observation",
                        help="Postprocessor retained in generated replay inputs.")
    parser.add_argument("--replay-observation-file", default="terminal_composition.txt",
                        help="Filename written by the terminal postprocessor in each replay directory.")
    parser.add_argument("--title", help="Optional complete figure title.")
    args = parser.parse_args()
    replays = replay_controls(args.aspect, args.inverse_output, args.reference_prm,
                              args.work_dir, args.iterations, args.end_step, args.vtu_field_name,
                              args.control_name, args.ascii_field_name, args.initial_model_kind,
                              args.terminal_postprocessor, args.replay_observation_file)
    plot_reconstructions(replays, args.reference_observation, args.plot, args.label, args.truth,
                         args.reference_initial_vtu, args.vtu_field_name, args.field_label, args.title)


if __name__ == "__main__":
    main()
