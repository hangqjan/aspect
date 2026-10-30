#!/usr/bin/env python3
"""Regenerate every durable figure for the physical TD-GADOPT 0-to-80 twin."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
benchmark = root / "benchmarks/adjoint_gadopt_mantle_convection"
shared = root / "benchmarks/adjoint_van_keken_composition/analysis/replay_and_plot_accepted_reconstructions.py"

def run(args: list[str]) -> None: subprocess.run([sys.executable, *args], check=True)

if __name__ == "__main__":
    out = benchmark / "results_0to80"
    run([str(Path(__file__).with_name("plot_0to80_twin.py")), "--output-dir", str(out),
         "--reference-initial-vtu", str(out / "reference/solution/solution-00000.0000.vtu"),
         "--reference-observation", str(out / "reference/terminal_temperature.txt"),
         "--replay-observation-file", "terminal_temperature.txt", "--field", "T",
         "--control-stem", "temperature", "--objective", "temperature field",
         "--label", "temperature, T", "--title", "TD-GADOPT physical 0-to-80 inversion",
         "--plot-dir", str(out / "plots"), "--terminal-plot", "terminal_temperature_comparison.png",
         "--objective-plot", "iteration_history.png",
         "--objective-linear-plot", "iteration_history_linear.png"])
    run([str(shared), "--aspect", str(root / "build-release/aspect-release"), "--inverse-output", str(out / "inverse"),
         "--reference-prm", str(benchmark / "reference_forward_0to80.prm"),
         "--reference-observation", str(out / "reference/terminal_temperature.txt"),
         "--reference-initial-vtu", str(out / "reference/solution/solution-00000.0000.vtu"),
         "--work-dir", str(out / "inverse/replays"), "--plot", str(out / "inverse/accepted_reconstructions.png"),
         "--label", "TD-GADOPT", "--truth", "gadopt", "--vtu-field-name", "T", "--control-name", "temperature",
         "--ascii-field-name", "temperature", "--initial-model-kind", "temperature",
         "--terminal-postprocessor", "terminal temperature observation", "--replay-observation-file", "terminal_temperature.txt",
         "--field-label", "temperature, T", "--title", "TD-GADOPT 0-to-80 terminal-temperature inversion: accepted reconstructions",
         "--iterations", "0", "2", "4", "6", "8", "9"])
