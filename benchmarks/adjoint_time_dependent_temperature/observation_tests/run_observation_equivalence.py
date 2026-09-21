#!/usr/bin/env python3
"""Run terminal volume-observation representation checks in serial or MPI.

The driver deliberately compares rank-owned (block, index) seed entries, not
rank ordering, so a two-rank run can be checked against serial output.
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "adjoint_time_dependent_temperature_composition.prm"
OBJECTIVE_RE = re.compile(r"Time-dependent terminal objective '([^']+)' = ([0-9eE+.-]+)")
SEED_RE = re.compile(r"ASPECT_ADJOINT_TERMINAL_SEED field=(\S+) (?:block=(\d+) l2=|full_l2=)([0-9eE+.-]+)")

def make_prm(case, field, source, work, plugin):
    text = BASE.read_text()
    text = "set Additional shared libraries = " + str(plugin) + "\n" + text
    text = text.replace("set Output directory                       = output-adjoint-time-dependent-temperature-composition",
                        "set Output directory = " + str(work / "output"))
    text = text.replace("set Function expression = 0.45 + 0.05 * sin(2*pi*x) * sin(pi*y)",
                        "set Function expression = 0.5")
    text = text.replace("subsection Postprocess\n  set List of postprocessors =\nend",
                        "subsection Postprocess\n  set List of postprocessors = volume observation test\n"
                        "  subsection Volume observation test\n"
                        f"    set Mode = {'in memory' if source in ('memory', 'copy', 'bad-layout') else 'none'}\n"
                        f"    set Target kind = {'copy' if source == 'copy' else 'constant'}\n"
                        f"    set Field = {field}\n"
                        f"    set Offset = {0. if source == 'copy' else (0.5 if field == 'temperature' else 0.2)}\n"
                        f"    set Suppress exception stack trace = {'true' if case == 'malformed' else 'false'}\n"
                        "  end\nend")
    if source == "bad-layout":
        text = text.replace("set Mode = in memory", "set Mode = invalid layout")
    if field == "temperature":
        if source == "file":
            filename = "malformed_volume.txt" if case == "malformed" else "constant_temperature.txt"
            text = text.replace("set Coordinate system = cartesian",
              "set Coordinate system = cartesian\n    set Observation source = file\n"
              "    subsection Observation file\n"
              f"      set Data directory = {HERE}\n      set Data file name = {filename}\n"
              "    end")
    else:
        text = text.replace("set List of objectives = temperature field", "set List of objectives = composition field")
        source_lines = "set Observation source = function"
        if source == "file":
            source_lines = ("set Observation source = file\n    subsection Observation file\n"
                            f"      set Data directory = {HERE}\n      set Data file name = constant_llsvp.txt\n    end")
        text += ("\nsubsection Adjoint\n  subsection Composition field\n"
                 "    set Field name = llsvp\n    set Coordinate system = cartesian\n"
                 f"    {source_lines}\n"
                 "    subsection Target function\n      set Variable names = x,y\n"
                 "      set Function expression = 0.2\n    end\n"
                 "    subsection Weight function\n      set Variable names = x,y\n"
                 "      set Function expression = 1\n    end\n  end\nend\n")
    text = text.replace("set Output MPI adjoint memory diagnostics = false",
                        "set Output MPI adjoint memory diagnostics = false\n    set Output terminal seed diagnostics = true")
    # The diagnostics subsection may not otherwise mention this new setting.
    if "Output terminal seed diagnostics" not in text:
        text = text.replace("subsection Diagnostics", "subsection Diagnostics\n    set Output terminal seed diagnostics = true", 1)
    prm = work / f"{case}.prm"
    prm.write_text(text)
    return prm

def parse_result(log, output, field):
    objectives = {n: float(v) for n, v in OBJECTIVE_RE.findall(log)}
    values = [v for n, v in objectives.items() if (field == "temperature") == (n == "temperature field")]
    if len(values) != 1:
        raise AssertionError(f"missing objective for {field}: {objectives}")
    seeds = {}
    for name, block, value in SEED_RE.findall(log):
        if name == ("temperature" if field == "temperature" else "composition_llsvp"):
            seeds["full" if not block else int(block)] = float(value)
    prefix = "terminal_seed_temperature" if field == "temperature" else "terminal_seed_composition_llsvp"
    vector = {}
    for path in output.glob(prefix + "_rank_*.txt"):
        for line in path.read_text().splitlines():
            if line and not line.startswith("#"):
                b, i, v = line.split()
                vector[(int(b), int(i))] = float(v)
    if not vector:
        raise AssertionError("terminal seed vector was not written")
    return values[0], seeds, vector

def close(a, b, atol=2e-11, rtol=2e-10):
    return abs(a-b) <= atol + rtol * max(abs(a), abs(b))

def compare(a, b, label):
    if not close(a[0], b[0]):
        raise AssertionError(f"{label}: objective {a[0]} != {b[0]}")
    if a[2].keys() != b[2].keys():
        raise AssertionError(f"{label}: distributed vector ownership differs")
    diff = max((abs(a[2][k]-b[2][k]) for k in a[2]), default=0.)
    scale = max((abs(v) for v in a[2].values()), default=0.)
    if diff > 2e-11 + 2e-10 * scale:
        raise AssertionError(f"{label}: terminal RHS max difference {diff}")
    return diff

def run(command, cwd, expect_ok=True, needle=None):
    p = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    (cwd / "run.log").write_text(p.stdout)
    if expect_ok and p.returncode:
        raise RuntimeError(p.stdout)
    if not expect_ok and (p.returncode == 0 or needle not in re.sub(r"\s+", " ", p.stdout)):
        raise AssertionError(f"negative case did not fail with {needle!r}:\n{p.stdout}")
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", required=True, type=Path)
    ap.add_argument("--plugin", required=True, type=Path)
    ap.add_argument("--mpi", type=int, default=1)
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--positive-only", action="store_true",
                    help="Run representation and truth cases, but not intentional failures.")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    results = {}
    for field in ("temperature", "llsvp"):
        for source in ("function", "file", "memory"):
            d = args.work / f"{field}-{source}"
            shutil.rmtree(d, ignore_errors=True); d.mkdir()
            prm = make_prm(source, field, source, d, args.plugin)
            cmd = [str(args.aspect), str(prm)]
            if args.mpi > 1: cmd = ["mpirun", "-np", str(args.mpi)] + cmd
            log = run(cmd, d)
            results[field, source] = parse_result(log, d / "output", field)
            if results[field, source][0] <= 0:
                raise AssertionError(f"{field}/{source}: expected J > 0")
        compare(results[field, "function"], results[field, "file"], field + " function/file")
        compare(results[field, "file"], results[field, "memory"], field + " file/memory")
        # Equal in-memory target proves the capture path itself contributes no misfit.
        d = args.work / f"{field}-truth"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
        prm = make_prm("truth", field, "copy", d, args.plugin)
        cmd = [str(args.aspect), str(prm)]
        if args.mpi > 1: cmd = ["mpirun", "-np", str(args.mpi)] + cmd
        truth = parse_result(run(cmd, d), d / "output", field)
        if abs(truth[0]) > 2e-12:
            raise AssertionError(f"{field}: truth objective {truth[0]}")
    if args.positive_only:
        print("OBSERVATION_EQUIVALENCE_POSITIVE_OK")
        return
    # Stable manager errors: incompatible vector layout and missing compatible objective.
    d = args.work / "negative-layout"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    run([str(args.aspect), str(make_prm("negative", "temperature", "bad-layout", d, args.plugin))], d,
        False, "terminal observation is incompatible with the current DoF layout")
    d = args.work / "negative-unavailable"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    prm = make_prm("negative", "llsvp", "memory", d, args.plugin)
    # Remove the composition objective after the plugin has been set to request llsvp.
    prm.write_text(prm.read_text().replace("set List of objectives = composition field", "set List of objectives = temperature field"))
    run([str(args.aspect), str(prm)], d, False, "A compatible terminal scalar-field objective is required")
    d = args.work / "negative-malformed-file"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    run([str(args.aspect), str(make_prm("malformed", "temperature", "file", d, args.plugin))], d,
        False, "has not found the expected number of data values")
    print("OBSERVATION_EQUIVALENCE_OK")

if __name__ == "__main__":
    main()
