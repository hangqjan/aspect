#!/usr/bin/env python3
"""Generate every TD-GADOPT plot supported by the available diagnostic data."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def run(script: str, output_dir: Path, required: Path | None = None) -> None:
    if required is not None and not required.exists():
        print(f"Skipping {script}: required data is absent: {required}")
        return
    command = [sys.executable, str(Path(__file__).with_name(script)), "--output-dir", str(output_dir)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        print(f"Skipping {script}: {completed.stderr.strip() or completed.stdout.strip()}")
    else:
        print(completed.stdout.strip() or f"Generated {script}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); diagnostics = args.output_dir / "diagnostics"
    history = diagnostics / "statistics_adjoint.csv"
    if not history.exists():
        history = diagnostics / "iteration_history.csv"
    run("plot_iteration_history.py", args.output_dir, history)
    run("plot_initial_temperature_iterations.py", args.output_dir,
        diagnostics / "accepted_controls/truth_T70.vtu")
    run("plot_terminal_temperature_comparison.py", args.output_dir,
        diagnostics / "terminal_predictions/terminal_temperature_observed_T80.vtu")
    run("plot_inversion_storyboard.py", args.output_dir,
        diagnostics / "terminal_predictions/terminal_temperature_observed_T80.vtu")
    run("plot_velocity_misfit.py", args.output_dir)

if __name__ == "__main__": main()
