#!/usr/bin/env python3
"""Analyze FD/adjoint errors by signal magnitude for cellwise validation CSVs."""

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


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            fd = float(row["finite_difference"])
            adjoint = float(row["adjoint"])
            signal = max(abs(fd), abs(adjoint))
            rows.append(
                {
                    "control": row["control"],
                    "active_cell_index": int(row["active_cell_index"]),
                    "finite_difference": fd,
                    "adjoint": adjoint,
                    "signal": signal,
                    "absolute_error": abs(fd - adjoint),
                    "relative_error": float(row["relative_error"]),
                    "opposite_sign": fd * adjoint < 0.0,
                }
            )
    return rows


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def write_summary(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "control",
        "log10_signal_bin",
        "count",
        "signal_median",
        "absolute_error_median",
        "absolute_error_p95",
        "relative_error_median",
        "relative_error_p90",
        "relative_error_p95",
        "relative_error_max",
        "opposite_sign_count",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for control in sorted({str(row["control"]) for row in rows}):
            subset = [row for row in rows if row["control"] == control]
            signals = np.array([float(row["signal"]) for row in subset])
            lo = math.floor(math.log10(float(np.min(signals))))
            hi = math.ceil(math.log10(float(np.max(signals))))
            for exponent in range(lo, hi):
                lower = 10.0**exponent
                upper = 10.0 ** (exponent + 1)
                bin_rows = [row for row in subset if lower <= float(row["signal"]) < upper]
                if not bin_rows:
                    continue
                rel = np.array([float(row["relative_error"]) for row in bin_rows])
                abs_err = np.array([float(row["absolute_error"]) for row in bin_rows])
                signal = np.array([float(row["signal"]) for row in bin_rows])
                writer.writerow(
                    {
                        "control": control,
                        "log10_signal_bin": f"[{exponent},{exponent + 1})",
                        "count": len(bin_rows),
                        "signal_median": f"{np.median(signal):.16e}",
                        "absolute_error_median": f"{np.median(abs_err):.16e}",
                        "absolute_error_p95": f"{np.quantile(abs_err, 0.95):.16e}",
                        "relative_error_median": f"{np.median(rel):.16e}",
                        "relative_error_p90": f"{np.quantile(rel, 0.90):.16e}",
                        "relative_error_p95": f"{np.quantile(rel, 0.95):.16e}",
                        "relative_error_max": f"{np.max(rel):.16e}",
                        "opposite_sign_count": sum(bool(row["opposite_sign"]) for row in bin_rows),
                    }
                )


def plot(rows: list[dict[str, float | str]], output: Path) -> None:
    controls = sorted({str(row["control"]) for row in rows})
    fig, axes = plt.subplots(len(controls), 2, figsize=(10.8, 4.2 * len(controls)), constrained_layout=True)
    if len(controls) == 1:
        axes = np.array([axes])

    for row_index, control in enumerate(controls):
        subset = [row for row in rows if row["control"] == control]
        signal = np.array([float(row["signal"]) for row in subset])
        rel = np.array([float(row["relative_error"]) for row in subset])
        abs_err = np.array([float(row["absolute_error"]) for row in subset])
        cell = np.array([int(row["active_cell_index"]) for row in subset])
        opposite = np.array([bool(row["opposite_sign"]) for row in subset])

        ax = axes[row_index, 0]
        ax.scatter(signal[~opposite], rel[~opposite], s=22, alpha=0.8, label="same sign")
        if np.any(opposite):
            ax.scatter(signal[opposite], rel[opposite], s=42, color="#d62728", marker="x", label="opposite sign")
            for x_value, y_value, cell_id in zip(signal[opposite], rel[opposite], cell[opposite]):
                ax.annotate(str(cell_id), (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("signal = max(|FD|, |adjoint|)")
        ax.set_ylabel("relative error")
        ax.set_title(f"{control}: relative error")
        ax.grid(True, which="both", linewidth=0.35, color="#dddddd")
        ax.legend(loc="best", fontsize=8)

        ax = axes[row_index, 1]
        ax.scatter(signal, abs_err, s=22, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("signal = max(|FD|, |adjoint|)")
        ax.set_ylabel("absolute error")
        ax.set_title(f"{control}: absolute error")
        ax.grid(True, which="both", linewidth=0.35, color="#dddddd")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("scratch/adjoint_kernel_grid/spherical_all_cell_fd_validation.csv"))
    parser.add_argument("--summary", type=Path, default=Path("scratch/adjoint_kernel_grid/spherical_error_by_signal_summary.csv"))
    parser.add_argument("--figure", type=Path, default=Path("scratch/adjoint_kernel_grid/spherical_error_by_signal.png"))
    args = parser.parse_args()

    rows = read_rows(args.csv)
    write_summary(rows, args.summary)
    plot(rows, args.figure)
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
