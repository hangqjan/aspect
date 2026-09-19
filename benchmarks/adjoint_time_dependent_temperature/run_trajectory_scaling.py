#!/usr/bin/env python3
"""Run reproducible load/scaling measurements for the adjoint trajectory archive.

The runner changes only archive mode, accepted-step count, dimension, and
initial global refinement.  It retains the fixed-mesh benchmark's material
model, timestep cap, nonlinear scheme, and solver tolerance.  Each run gets a
separate output directory and its complete ASPECT screen output is retained.
"""

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PRM = (ROOT / "benchmarks" / "adjoint_time_dependent_temperature"
               / "trajectory_archive_benchmark.prm")

ARCHIVE_RE = re.compile(
    r"Adjoint trajectory archive \[reverse complete\]: mode = (?P<mode>.+?), "
    r"records = (?P<records>\d+), total bytes = (?P<total>\d+), "
    r"mean bytes/record = (?P<mean>\d+) "
    r"\(metadata (?P<metadata>\d+), T (?P<temperature>\d+), "
    r"C (?P<composition>\d+), u (?P<velocity>\d+), p (?P<pressure>\d+)\)")
MEMORY_RE = re.compile(
    r"Adjoint trajectory memory \[reverse complete\]: .*?"
    r"metadata object bytes = (?P<metadata_memory>\d+), "
    r"current loaded record bytes = (?P<current_record_memory>\d+), "
    r"current recovered Stokes-state bytes = (?P<recovered_stokes_memory>\d+), "
    r"reverse working-cache bytes = (?P<cache_memory>\d+), "
    r"VmPeak = (?P<vm_peak>\d+) kB, VmRSS = (?P<vm_rss>\d+) kB, "
    r"Stokes recovery solves = (?P<recovery_solves>\d+)")
TIMER_RE = re.compile(
    r"^\|\s*(?P<section>.*?)\s*\|\s*\d+\s*\|\s*"
    r"(?P<seconds>[0-9.eE+-]+)s\s*\|", re.MULTILINE)
TOTAL_TIME_RE = re.compile(
    r"Total wallclock time elapsed since start\s*\|\s*([0-9.eE+-]+)s")
DOF_RE = re.compile(r"Number of degrees of freedom: (?P<total>[\d,]+) \((?P<blocks>[^)]+)\)")
# ASPECT's /proc-derived VmPeak/VmRSS diagnostics are unavailable on macOS.
# `/usr/bin/time -l` supplies the corresponding process-level measurements
# there. The values reported by the macOS implementation are byte counts.
EXTERNAL_MAX_RSS_RE = re.compile(
    r"^\s*(?P<bytes>\d+)\s+maximum resident set size$", re.MULTILINE)
EXTERNAL_PEAK_FOOTPRINT_RE = re.compile(
    r"^\s*(?P<bytes>\d+)\s+peak memory footprint$", re.MULTILINE)


def parse_int_list(values: str) -> List[int]:
    result = [int(value) for value in values.split(",") if value.strip()]
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("Expected one or more positive integers.")
    return result


def replace_entry(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^\s*set\s+{re.escape(name)}\s*=.*$", re.MULTILINE)
    replacement = f"set {name} = {value}"
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace parameter '{name}'.")
    return text


def write_input(base_text: str,
                path: pathlib.Path,
                output_dir: pathlib.Path,
                mode: str,
                steps: int,
                dimension: int,
                refinement: int) -> None:
    # The base has dt_max=0.01. Preserve it and adjust end time together with
    # the explicit end-step criterion so each load has the intended record
    # count, without invoking a new CFL schedule during replay.
    text = replace_entry(base_text, "Dimension", str(dimension))
    text = replace_entry(text, "End time", f"{steps * 0.01:.16g}")
    text = replace_entry(text, "Output directory", str(output_dir))
    text = replace_entry(text, "Trajectory storage mode", mode)
    text = replace_entry(text, "Run trajectory Stokes recovery diagnostic", "false")
    text = replace_entry(text, "End step", str(steps))
    text = replace_entry(text, "Initial global refinement", str(refinement))
    if dimension == 3:
        # The 2-D source expressions are z-independent, but ParsedFunction
        # still requires all three coordinate variables to be declared.
        text = re.sub(r"set Variable names\s*=\s*x,y\s*$",
                      "set Variable names = x,y,z", text, flags=re.MULTILINE)
        text = replace_entry(text, "Tangential velocity boundary indicators",
                             "left, right, front, back, bottom, top")
    path.write_text(text)


def parse_gradient_norm(path: pathlib.Path) -> float:
    total = 0.0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        total += float(line.split()[1]) ** 2
    return total ** 0.5


def read_gradient(path: pathlib.Path) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            values[int(fields[0])] = float(fields[1])
    return values


def relative_gradient_difference(first: pathlib.Path, second: pathlib.Path) -> float:
    first_values = read_gradient(first)
    second_values = read_gradient(second)
    if first_values.keys() != second_values.keys():
        raise RuntimeError("Gradient DoF sets differ between archive modes.")
    diff_sq = sum((first_values[index] - second_values[index]) ** 2
                  for index in first_values)
    ref_sq = sum(value ** 2 for value in second_values.values())
    return diff_sq ** 0.5 / max(ref_sq ** 0.5, 1.0e-300)


def last_match(pattern: re.Pattern, text: str) -> re.Match:
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError(f"Could not find expected output matching {pattern.pattern!r}.")
    return matches[-1]


def timer_values(text: str) -> Dict[str, float]:
    return {match.group("section").strip(): float(match.group("seconds"))
            for match in TIMER_RE.finditer(text)}


def run_case(binary: pathlib.Path,
             input_path: pathlib.Path,
             run_dir: pathlib.Path,
             measure_rss: bool) -> Tuple[Dict[str, object], pathlib.Path]:
    command = [str(binary), str(input_path)]
    if measure_rss:
        command = ["/usr/bin/time", "-l", *command]
    completed = subprocess.run(command, cwd=ROOT,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (run_dir / "screen-output").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"ASPECT failed for {input_path}; inspect {run_dir / 'screen-output'}")

    archive = last_match(ARCHIVE_RE, completed.stdout).groupdict()
    memory = last_match(MEMORY_RE, completed.stdout).groupdict()
    dofs = last_match(DOF_RE, completed.stdout).groupdict()
    timers = timer_values(completed.stdout)
    output_dir = pathlib.Path(re.search(
        r"^set Output directory = (.+)$", input_path.read_text(), re.MULTILINE).group(1))
    gradient = output_dir / "initial_temperature_gradient_rank_00000.txt"
    if not gradient.exists():
        raise RuntimeError(f"Missing temperature gradient {gradient}.")

    result: Dict[str, object] = {
        "records": int(archive["records"]),
        "archive_bytes": int(archive["total"]),
        "bytes_per_record": int(archive["mean"]),
        "metadata_bytes": int(archive["metadata"]),
        "temperature_bytes": int(archive["temperature"]),
        "composition_bytes": int(archive["composition"]),
        "velocity_bytes": int(archive["velocity"]),
        "pressure_bytes": int(archive["pressure"]),
        "metadata_object_bytes": int(memory["metadata_memory"]),
        "current_record_memory_bytes": int(memory["current_record_memory"]),
        "recovered_stokes_memory_bytes": int(memory["recovered_stokes_memory"]),
        "reverse_cache_memory_bytes": int(memory["cache_memory"]),
        "vm_peak_kb": int(memory["vm_peak"]),
        "vm_rss_kb": int(memory["vm_rss"]),
        "external_max_rss_bytes": (
            int(last_match(EXTERNAL_MAX_RSS_RE, completed.stdout).group("bytes"))
            if measure_rss else None),
        "external_peak_footprint_bytes": (
            int(last_match(EXTERNAL_PEAK_FOOTPRINT_RE, completed.stdout).group("bytes"))
            if measure_rss else None),
        "stokes_recovery_solves": int(memory["recovery_solves"]),
        "dofs": int(dofs["total"].replace(",", "")),
        "dof_blocks": dofs["blocks"],
        "gradient_l2_norm": parse_gradient_norm(gradient),
        "write_seconds": timers.get("Adjoint trajectory write", 0.0),
        "read_seconds": timers.get("Adjoint trajectory read", 0.0),
        "recovery_seconds": timers.get("Adjoint trajectory Stokes recovery", 0.0),
        "reconstruction_seconds": timers.get("Adjoint trajectory state reconstruction", 0.0),
        "reverse_seconds": timers.get("Adjoint trajectory reverse sweep", 0.0),
        "total_seconds": float(last_match(TOTAL_TIME_RE, completed.stdout).group(1)),
        "gradient_path": str(gradient),
    }
    return result, gradient


def write_results(results: List[Dict[str, object]], results_dir: pathlib.Path) -> None:
    json_path = results_dir / "summary.json"
    csv_path = results_dir / "summary.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def benchmark_levels(args: argparse.Namespace) -> Iterable[Tuple[str, int, int, int]]:
    if args.spatial_refinements:
        for refinement in args.spatial_refinements:
            yield (f"S{refinement}", args.spatial_steps, refinement, args.dimension)
    else:
        for steps in args.temporal_steps:
            yield (f"T{steps}", steps, args.refinement, args.dimension)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build-debug" / "aspect-debug"))
    parser.add_argument("--base-prm", default=str(DEFAULT_PRM))
    parser.add_argument("--results-dir", required=True,
                        help="New directory for inputs, logs, and summary.{json,csv}.")
    parser.add_argument("--temporal-steps", type=parse_int_list,
                        default=parse_int_list("20,100,250,500,1000"))
    parser.add_argument("--spatial-refinements", type=parse_int_list,
                        help="Run a spatial hierarchy instead of temporal levels.")
    parser.add_argument("--spatial-steps", type=int, default=20)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--measure-rss", action="store_true",
                        help="Capture macOS process RSS and physical footprint with /usr/bin/time -l.")
    args = parser.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    base_prm = pathlib.Path(args.base_prm).resolve()
    results_dir = pathlib.Path(args.results_dir).resolve()
    if not binary.exists() or not base_prm.exists():
        raise RuntimeError("The ASPECT executable or base parameter file is missing.")
    if results_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing results directory {results_dir}.")
    results_dir.mkdir(parents=True)
    base_text = base_prm.read_text()
    results: List[Dict[str, object]] = []

    for label, steps, refinement, dimension in benchmark_levels(args):
        gradients: Dict[str, pathlib.Path] = {}
        level_results: Dict[str, Dict[str, object]] = {}
        for mode in ("minimal", "stokes cached"):
            mode_label = mode.replace(" ", "_")
            run_dir = results_dir / f"{label}_{mode_label}"
            run_dir.mkdir()
            output_dir = run_dir / "output"
            input_path = run_dir / "input.prm"
            write_input(base_text, input_path, output_dir, mode, steps,
                        dimension, refinement)
            result, gradients[mode] = run_case(binary, input_path, run_dir,
                                                args.measure_rss)
            result.update({"label": label, "mode": mode, "steps": steps,
                           "dimension": dimension, "initial_global_refinement": refinement,
                           "mpi_ranks": 1})
            level_results[mode] = result
            results.append(result)

        minimal = level_results["minimal"]
        cached = level_results["stokes cached"]
        gradient_difference = relative_gradient_difference(gradients["minimal"],
                                                            gradients["stokes cached"])
        for result in (minimal, cached):
            result["minimal_vs_cached_gradient_relative_l2"] = gradient_difference
        minimal["storage_saving"] = 1.0 - minimal["archive_bytes"] / cached["archive_bytes"]
        minimal["reverse_time_penalty"] = ((minimal["reverse_seconds"] - cached["reverse_seconds"])
                                            / max(cached["reverse_seconds"], 1.0e-300))
        minimal["stokes_recovery_fraction"] = (minimal["recovery_seconds"]
                                                / max(minimal["reverse_seconds"], 1.0e-300))
        for result in (minimal, cached):
            result["io_fraction"] = ((result["read_seconds"] + result["write_seconds"])
                                      / max(result["total_seconds"], 1.0e-300))
            result["write_bandwidth_bytes_per_second"] = (result["archive_bytes"]
                                                            / max(result["write_seconds"], 1.0e-300))
            result["read_bandwidth_bytes_per_second"] = (result["archive_bytes"]
                                                           / max(result["read_seconds"], 1.0e-300))

        print(f"{label}: records={minimal['records']}; gradient rel-L2={gradient_difference:.3e}; "
              f"minimal/cached bytes={minimal['archive_bytes']}/{cached['archive_bytes']}; "
              f"minimal/cached reverse s={minimal['reverse_seconds']:.6g}/{cached['reverse_seconds']:.6g}")

    write_results(results, results_dir)
    print(f"Wrote {results_dir / 'summary.csv'} and {results_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
