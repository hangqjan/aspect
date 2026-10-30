#!/usr/bin/env python3
"""Focused ASPECT-native geoid surface-observation checks."""
import argparse
import math
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "adjoint_geoid_density.prm"
VALUE = re.compile(r"ASPECT_GEOID_SURFACE_OBSERVATION rank=(\d+) value=([0-9eE+.-]+)")


def make_prm(work, plugin, observation_file, longitude, latitude, suppress=False):
    text = BASE.read_text()
    text = "set Additional shared libraries = " + str(plugin) + "\n" + text
    text = text.replace("set Output directory                       = output-adjoint-geoid-density",
                        "set Output directory = " + str(work / "output"))
    text = text.replace("set List of postprocessors = geoid, adjoint kernels",
                        "set List of postprocessors = geoid, adjoint kernels, geoid surface observation test\n"
                        "\n  subsection Geoid surface observation test\n"
                        "    set Mode = geoid\n"
                        f"    set Longitude = {longitude}\n"
                        f"    set Latitude = {latitude}\n"
                        f"    set Suppress exception stack trace = {'true' if suppress else 'false'}\n"
                        "    subsection Observation\n"
                        "      set Observation source = file\n"
                        "      set Observation format = ASPECT native\n"
                        f"      set Observation file = {observation_file}\n"
                        "    end\n"
                        "  end")
    prm = work / "input.prm"
    prm.write_text(text)
    return prm


def run(aspect, prm, work, mpi=1, expect_ok=True, needle=None):
    command = [str(aspect), str(prm)]
    if mpi > 1:
        command = ["mpirun", "-np", str(mpi)] + command
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, cwd=work)
    (work / "run.log").write_text(completed.stdout)
    if expect_ok:
        if completed.returncode:
            raise RuntimeError(completed.stdout)
    elif completed.returncode == 0 or needle not in completed.stdout:
        raise AssertionError(f"expected failure containing {needle!r}:\n{completed.stdout}")
    return completed.stdout


def values(log):
    found = {int(rank): float(value) for rank, value in VALUE.findall(log)}
    if not found:
        raise AssertionError("no geoid surface observation diagnostic")
    if max(found.values()) - min(found.values()) > 1e-14:
        raise AssertionError(f"MPI ranks disagree: {found}")
    return next(iter(found.values()))


def write(work, name, content):
    path = work / name
    path.write_text(content)
    return path


def case(aspect, plugin, root, name, content, longitude, latitude,
         expected=None, mpi=1, needle=None):
    work = root / name
    work.mkdir()
    data = write(work, "reference.txt", content)
    prm = make_prm(work, plugin, data, longitude, latitude, suppress=needle is not None)
    log = run(aspect, prm, work, mpi, expect_ok=needle is None, needle=needle)
    if expected is None:
        return None
    value = values(log)
    if not math.isfinite(value) or abs(value - expected) > 1e-10:
        raise AssertionError(f"{name}: value {value} != {expected}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=HERE / "work")
    args = parser.parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)

    cartesian = "# x y z geoid_anomaly\n1 0 0 11\n0 1 0 7\n0 0 1 3\n"
    case(args.aspect, args.plugin, args.work, "cartesian", cartesian, 0, 0, 11)

    geographic = "# longitude latitude geoid_anomaly\n-170 20 7\n10 -40 2\n120 50 4\n"
    case(args.aspect, args.plugin, args.work, "geographic", geographic, -170, 20, 7)

    wrap = "# longitude latitude geoid_anomaly\n-179 0 10\n179 0 20\n"
    left = case(args.aspect, args.plugin, args.work, "wrap_left", wrap, -180, 0, 15)
    right = case(args.aspect, args.plugin, args.work, "wrap_right", wrap, 180, 0, 15)
    if abs(left - right) > 1e-12:
        raise AssertionError("longitude wraparound is discontinuous")

    polar = "# longitude latitude geoid_anomaly\n0 89 4\n180 89 6\n"
    case(args.aspect, args.plugin, args.work, "polar", polar, 90, 90, 5)

    coarse_smooth = "# longitude latitude geoid_anomaly\n-10 0 0.984807753012208\n10 0 0.984807753012208\n"
    fine_smooth = coarse_smooth + "0 0 1\n"
    coarse_value = case(args.aspect, args.plugin, args.work, "coarse_smooth", coarse_smooth,
                        0, 0, 0.984807753012208)
    fine_value = case(args.aspect, args.plugin, args.work, "fine_smooth", fine_smooth,
                      0, 0, 1)
    if abs(fine_value - 1) >= abs(coarse_value - 1):
        raise AssertionError("denser smooth samples did not improve interpolation")

    case(args.aspect, args.plugin, args.work, "cartesian_mpi", cartesian, 0, 0, 11, mpi=2)
    case(args.aspect, args.plugin, args.work, "bad_header",
         "# x y z not_geoid_anomaly\n1 0 0 1\n", 0, 0,
         needle="ASPECT-native geoid observation expected final column")
    case(args.aspect, args.plugin, args.work, "wrong_physical_file",
         "# x y z dynamic_topography_top\n1 0 0 1\n", 0, 0,
         needle="ASPECT-native geoid observation expected final column")
    case(args.aspect, args.plugin, args.work, "empty_file", "", 0, 0,
         needle="must start with a commented column header")
    print("GEOID_SURFACE_OBSERVATION_OK")


if __name__ == "__main__":
    main()
