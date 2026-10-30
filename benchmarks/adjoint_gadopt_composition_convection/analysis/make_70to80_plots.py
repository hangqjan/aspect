#!/usr/bin/env python3
"""Regenerate every figure for the G-ADOPT composition 70-to-80 twin."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
benchmark = root / "benchmarks/adjoint_gadopt_composition_convection"
shared = root / "benchmarks/adjoint_van_keken_composition/analysis/replay_and_plot_accepted_reconstructions.py"

def run(args: list[str]) -> None: subprocess.run([sys.executable, *args], check=True)

if __name__ == "__main__":
    out = benchmark / "results_70to80"
    run([str(shared), "--aspect", str(root / "build-release/aspect-release"), "--inverse-output", str(out / "inverse"),
         "--reference-prm", str(benchmark / "reference_forward_70to80.prm"),
         "--reference-observation", str(out / "reference/terminal_composition.txt"),
         "--reference-initial-vtu", str(out / "reference/solution/solution-00070.0000.vtu"),
         "--work-dir", str(out / "inverse/replays"), "--plot", str(out / "inverse/accepted_reconstructions.png"),
         "--label", "G-ADOPT composition", "--truth", "gadopt", "--control-name", "composition",
         "--end-step", "10", "--field-label", "chemical composition",
         "--title", "G-ADOPT composition 70-to-80 terminal-composition inversion: accepted reconstructions",
         "--iterations", "0", "2", "4", "6", "8", "10", "12"])
    run([str(Path(__file__).with_name("plot_70to80_twin.py")), "--output-dir", str(out),
         "--reference-initial-vtu", str(out / "reference/solution/solution-00070.0000.vtu"),
         "--reference-observation", str(out / "reference/terminal_composition.txt"),
         "--replay-observation-file", "terminal_composition.txt", "--field", "chemical",
         "--control-stem", "composition", "--objective", "composition field",
         "--label", "chemical composition", "--title", "G-ADOPT composition 70-to-80 inversion",
         "--plot-dir", str(out / "plots"), "--terminal-plot", "terminal_composition_comparison.png",
         "--objective-plot", "iteration_history.png",
         "--objective-linear-plot", "iteration_history_linear.png"])
    run([str(Path(__file__).with_name("plot_inversion_storyboard.py")), "--output-dir", str(out),
         "--reference-initial-vtu", str(out / "reference/solution/solution-00070.0000.vtu"),
         "--reference-observation", str(out / "reference/terminal_composition.txt"),
         "--field", "chemical", "--control-stem", "composition",
         "--iterations", "0", "2", "4", "6", "8", "10", "12",
         "--title", "G-ADOPT terminal-composition inversion: accepted reconstructions (70 to 80)"])
