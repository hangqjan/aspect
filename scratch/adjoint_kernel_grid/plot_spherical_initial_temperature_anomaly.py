#!/usr/bin/env python3
"""Plot the initial temperature anomaly from the 2D spherical benchmark."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


INNER_RADIUS = 3_480_000.0
OUTER_RADIUS = 6_336_000.0
BACKGROUND_TEMPERATURE = 1600.0
ANOMALY_TEMPERATURE = 1700.0
ANOMALY_CENTER = (0.0, 6_100_000.0)
ANOMALY_RADIUS = 250_000.0


def initial_anomaly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radius = np.hypot(x, y)
    in_shell = (INNER_RADIUS <= radius) & (radius <= OUTER_RADIUS)
    in_anomaly = np.hypot(x - ANOMALY_CENTER[0], y - ANOMALY_CENTER[1]) < ANOMALY_RADIUS
    anomaly = np.full_like(x, np.nan, dtype=float)
    anomaly[in_shell] = 0.0
    anomaly[in_shell & in_anomaly] = ANOMALY_TEMPERATURE - BACKGROUND_TEMPERATURE
    return anomaly


def main() -> None:
    output = Path("scratch/adjoint_kernel_grid/spherical_initial_temperature_anomaly.png")
    points = 900
    axis = np.linspace(-OUTER_RADIUS, OUTER_RADIUS, points)
    x, y = np.meshgrid(axis, axis)
    anomaly = initial_anomaly(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    for ax, title, xlim, ylim in [
        (
            axes[0],
            "physical scale",
            (-OUTER_RADIUS, OUTER_RADIUS),
            (-OUTER_RADIUS, OUTER_RADIUS),
        ),
        (
            axes[1],
            "top anomaly zoom",
            (-650_000.0, 650_000.0),
            (5_620_000.0, 6_420_000.0),
        ),
    ]:
        image = ax.imshow(
            anomaly,
            origin="lower",
            extent=(-OUTER_RADIUS, OUTER_RADIUS, -OUTER_RADIUS, OUTER_RADIUS),
            cmap=cmap,
            vmin=0.0,
            vmax=ANOMALY_TEMPERATURE - BACKGROUND_TEMPERATURE,
            interpolation="nearest",
        )
        ax.add_patch(Circle((0.0, 0.0), INNER_RADIUS, fill=False, color="#333333", linewidth=0.9))
        ax.add_patch(Circle((0.0, 0.0), OUTER_RADIUS, fill=False, color="#333333", linewidth=0.9))
        ax.add_patch(Circle(ANOMALY_CENTER, ANOMALY_RADIUS, fill=False, color="#00a6d6", linewidth=1.2))
        ax.scatter([ANOMALY_CENTER[0]], [ANOMALY_CENTER[1]], s=18, color="#00a6d6", zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(6, 6))
        ax.set_title(title)

    fig.suptitle("2D spherical benchmark initial temperature anomaly")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.03)
    colorbar.set_label("T - 1600 K")
    fig.text(
        0.5,
        0.02,
        "center = (0, 6.1e6 m), radius = 2.5e5 m, anomaly = +100 K",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
