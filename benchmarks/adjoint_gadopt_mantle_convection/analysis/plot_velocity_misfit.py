#!/usr/bin/env python3
"""Reserved for the running top-surface velocity objective; skips cleanly for terminal-only runs."""
from __future__ import annotations
import argparse
from pathlib import Path
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print("Skipping velocity plot: the running Stokes velocity objective is not enabled yet.")
if __name__ == "__main__": main()
