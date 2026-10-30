#!/usr/bin/env python3
"""Use the shared scalar-twin plotting implementation for the 70-to-80 case."""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("plot_0to80_twin.py")), run_name="__main__")
