#!/usr/bin/env python3
"""Generate durable plots for a physical full-horizon scalar-field twin."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adjoint_van_keken_composition" / "analysis"))
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


def controls(output: Path, stem: str, field: str) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    directory = output / "inverse/diagnostics/accepted_controls"
    found = []
    for path in sorted(directory.glob(f"initial_{stem}_iter_*.vtu")):
        iteration = int(path.stem.rsplit("_", 1)[1])
        x, y, value = scalar_grid(path, field)
        found.append((iteration, x, y, value))
    if not found:
        raise RuntimeError(f"No accepted controls found under {directory}")
    return found


def save_initial_plots(output: Path, truth_vtu: Path, field: str, stem: str,
                       label: str, title: str, plot_dir: Path) -> None:
    accepted = controls(output, stem, field)
    truth_x, truth_y, truth = scalar_grid(truth_vtu, field)
    reference = resample(truth_x, truth_y, truth, accepted[0][1], accepted[0][2])
    values = [("Reference", reference)] + [(f"Iteration {iteration}", value)
                                              for iteration, _, _, value in accepted]
    errors = [(name, value - reference) for name, value in values[1:]]
    columns = 4
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.15 * rows),
                                layout="constrained", squeeze=False)
    image = None
    for axis, (name, value) in zip(axes.flat, values):
        image = axis.imshow(value, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=1)
        axis.set(title=name, aspect="equal", xticks=(), yticks=())
    for axis in axes.flat[len(values):]: axis.set_visible(False)
    figure.colorbar(image, ax=axes.flat[:len(values)], shrink=0.82, label=label)
    figure.suptitle(f"{title}: accepted initial controls")
    figure.savefig(plot_dir / f"initial_{stem}_iterations.png", dpi=180)
    plt.close(figure)
    maximum = max(float(np.max(np.abs(value))) for _, value in errors)
    figure, axes = plt.subplots(int(np.ceil(len(errors) / columns)), columns,
                                figsize=(3.4 * columns, 3.15 * int(np.ceil(len(errors) / columns))),
                                layout="constrained", squeeze=False)
    for axis, (name, value) in zip(axes.flat, errors):
        image = axis.imshow(value, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axis.set(title=name, aspect="equal", xticks=(), yticks=())
    for axis in axes.flat[len(errors):]: axis.set_visible(False)
    figure.colorbar(image, ax=axes.flat[:len(errors)], shrink=0.82, label=f"{label} error")
    figure.suptitle(f"{title}: initial-control error")
    figure.savefig(plot_dir / f"initial_{stem}_error.png", dpi=180)
    plt.close(figure)


def save_terminal_plot(output: Path, observation: Path, observation_name: str,
                       label: str, title: str, plot_dir: Path, filename: str) -> None:
    _, _, observed = read_observation(observation)
    replays = sorted((output / "inverse/replays").glob(f"accepted_*/{observation_name}"))
    if not replays: raise RuntimeError("No replay terminal observations found")
    _, _, initial = read_observation(replays[0]); _, _, final = read_observation(replays[-1])
    limit = max(float(np.max(np.abs(initial-observed))), float(np.max(np.abs(final-observed))))
    panels = [("Observed terminal field", observed, "viridis", 0, 1),
              ("Initial prediction", initial, "viridis", 0, 1),
              ("Final prediction", final, "viridis", 0, 1),
              ("Initial minus observed", initial-observed, "coolwarm", -limit, limit),
              ("Final minus observed", final-observed, "coolwarm", -limit, limit)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axis, (name, value, cmap, lower, upper) in zip(axes.flat, panels):
        image = axis.imshow(value, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=lower, vmax=upper)
        figure.colorbar(image, ax=axis, shrink=.80); axis.set(title=name, aspect="equal", xticks=(), yticks=())
    axes.flat[-1].set_visible(False); figure.suptitle(f"{title}: terminal-field comparison")
    figure.savefig(plot_dir / filename, dpi=180); plt.close(figure)


def save_objective_plot(output: Path, objective: str, title: str, plot_dir: Path,
                        filename: str, linear_filename: str) -> None:
    pattern = re.compile(rf"Time-dependent terminal objective '{re.escape(objective)}' = ([0-9.eE+-]+)")
    values = [float(match.group(1)) for match in pattern.finditer((output / "inverse/log.txt").read_text())]
    if not values: raise RuntimeError("No terminal objective values found")

    def save(path: Path, *, logarithmic: bool) -> None:
        figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        draw = axis.semilogy if logarithmic else axis.plot
        draw(range(len(values)), values, "o-", color="#1769aa")
        scale = "logarithmic" if logarithmic else "linear"
        axis.set(xlabel="objective evaluation", ylabel="terminal objective",
                 title=f"{title}: objective history ({scale} scale)")
        axis.grid(True, which="both" if logarithmic else "major", alpha=.3)
        figure.savefig(path, dpi=180)
        plt.close(figure)

    save(plot_dir / filename, logarithmic=True)
    save(plot_dir / linear_filename, logarithmic=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-initial-vtu", type=Path, required=True)
    parser.add_argument("--reference-observation", type=Path, required=True)
    parser.add_argument("--replay-observation-file", required=True)
    parser.add_argument("--field", required=True); parser.add_argument("--control-stem", required=True)
    parser.add_argument("--objective", required=True); parser.add_argument("--label", required=True)
    parser.add_argument("--title", required=True); parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--terminal-plot", default="terminal_field_comparison.png")
    parser.add_argument("--objective-plot", default="objective_history.png")
    parser.add_argument("--objective-linear-plot", default="objective_history_linear.png")
    args = parser.parse_args(); args.plot_dir.mkdir(parents=True, exist_ok=True)
    save_initial_plots(args.output_dir, args.reference_initial_vtu, args.field, args.control_stem,
                       args.label, args.title, args.plot_dir)
    save_terminal_plot(args.output_dir, args.reference_observation, args.replay_observation_file,
                       args.label, args.title, args.plot_dir, args.terminal_plot)
    save_objective_plot(args.output_dir, args.objective, args.title, args.plot_dir,
                        args.objective_plot, args.objective_linear_plot)


if __name__ == "__main__": main()
