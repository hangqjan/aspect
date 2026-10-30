#!/usr/bin/env python3
"""Run the shared full-horizon scalar-twin plotting pipeline for TD-GADOPT."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    shared = (Path(__file__).resolve().parents[2] / "adjoint_gadopt_composition_convection"
              / "analysis" / "plot_0to80_twin.py")
    runpy.run_path(str(shared), run_name="__main__")
