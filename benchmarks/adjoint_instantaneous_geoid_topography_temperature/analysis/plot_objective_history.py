#!/usr/bin/env python3
"""Plot accepted reduced-objective histories for the three observation cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (("Dynamic topography", "inverse-dynamic-topography"),
           ("Self-gravitating geoid", "inverse-geoid"),
           ("Combined", "inverse-combined"))


def load_history(root: Path, directory: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / directory / "diagnostics/accepted_controls/objective_history.txt"
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != 2 or values.shape[0] == 0:
        raise RuntimeError(f"invalid accepted-objective history: {path}")
    return values[:, 0], values[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing inverse-* output directories.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for label, directory in METHODS:
        iteration, objective = load_history(args.root, directory)
        axis.semilogy(iteration, objective / objective[0], marker="o", label=label)
    axis.set_xlabel("Accepted optimization iteration")
    axis.set_ylabel(r"Reduced objective $J/J_0$")
    axis.set_title("Accepted objective reduction")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
