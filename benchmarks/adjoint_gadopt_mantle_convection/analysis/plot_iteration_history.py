#!/usr/bin/env python3
"""Plot accepted TD-GADOPT objective, errors, and reduced L2 gradient norm."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_common import read_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    history = args.output_dir / "diagnostics/statistics_adjoint.csv"
    if not history.exists():
        history = args.output_dir / "diagnostics/iteration_history.csv"
    rows = [row for row in read_history(history)
            if row["accepted"] == 1.0]
    if not rows:
        raise RuntimeError("iteration_history.csv has no accepted rows")
    iteration = [row["accepted_iteration"] for row in rows]
    panels = [("J_over_J0", r"$J_T/J_T(0)$"),
              ("terminal_T_error_rel", r"$E_T^{rel}$"),
              ("control_error_L2_rel", r"$E_{control}^{rel}$"),
              ("gradient_norm_L2", r"$\|g\|_{L^2}$")]
    def save(path: Path, *, logarithmic: bool) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for axis, (column, label) in zip(axes.flat, panels, strict=True):
            draw = axis.semilogy if logarithmic else axis.plot
            draw(iteration, [row[column] for row in rows], "o-", color="#1769aa")
            axis.set_xlabel("accepted iteration")
            axis.set_ylabel(label)
            axis.grid(True, which="both" if logarithmic else "major", alpha=0.3)
        scale = "logarithmic" if logarithmic else "linear"
        figure.suptitle(f"TD-GADOPT terminal-temperature inversion history ({scale} scale)")
        figure.savefig(path, dpi=180)
        plt.close(figure)

    plot_directory = args.output_dir / "diagnostics/plots"
    save(plot_directory / "iteration_history.png", logarithmic=True)
    save(plot_directory / "iteration_history_linear.png", logarithmic=False)


if __name__ == "__main__":
    main()
