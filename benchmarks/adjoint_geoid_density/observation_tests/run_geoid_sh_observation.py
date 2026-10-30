#!/usr/bin/env python3
"""Negative parser checks for ASPECT-native geoid harmonic observations."""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "adjoint_geoid_density.prm"


def make_prm(work, stacktrace_plugin, observation_file):
    text = BASE.read_text()
    text = "set Additional shared libraries = " + str(stacktrace_plugin) + "\n" + text
    text = text.replace("set Output directory                       = output-adjoint-geoid-density",
                        "set Output directory = " + str(work / "output"))
    text = text.replace("set Observed cosine coefficients = 0, 0, 0, 0, 0, 0\n"
                        "    set Observed sine coefficients   = 0, 0, 0, 0, 0, 0",
                        "set Observation source = file")
    text = text.replace("set Coefficient weights          = 1, 2, 3, 4, 5, 6",
                        "subsection Observation\n"
                        "      set Observation format = ASPECT native\n"
                        f"      set Observation file = {observation_file}\n"
                        "    end\n"
                        "    set Coefficient weights          = 1, 2, 3, 4, 5, 6")
    text = text.replace("set List of postprocessors = geoid, adjoint kernels",
                        "set List of postprocessors = geoid, adjoint kernels, geoid surface observation test\n"
                        "\n  subsection Geoid surface observation test\n"
                        "    set Mode = none\n"
                        "    set Suppress exception stack trace = true\n"
                        "  end")
    prm = work / "input.prm"
    prm.write_text(text)
    return prm


def run(aspect, prm, work, needle):
    completed = subprocess.run([str(aspect), str(prm)], cwd=work, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / "run.log").write_text(completed.stdout)
    if completed.returncode == 0 or needle not in completed.stdout:
        raise AssertionError(f"expected failure containing {needle!r}:\n{completed.stdout}")


def test(aspect, plugin, root, name, content, needle):
    work = root / name
    work.mkdir()
    observation = work / "observation.txt"
    observation.write_text(content)
    run(aspect, make_prm(work, plugin, observation), work, needle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", required=True, type=Path)
    parser.add_argument("--stacktrace-plugin", required=True, type=Path)
    parser.add_argument("--dt-file", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=HERE / "sh-work")
    args = parser.parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)

    header = "# degree order cosine_coefficient sine_coefficient\n"
    all_rows = ("0 0 0 0\n1 0 0 0\n1 1 0 0\n2 0 0 0\n"
                "2 1 0 0\n2 2 0 0\n")
    test(args.aspect, args.stacktrace_plugin, args.work, "bad_header",
         "# degree order C S\n" + all_rows, "canonical degree/order/cosine/sine header")
    test(args.aspect, args.stacktrace_plugin, args.work, "bad_row",
         header + "0 0 not-a-number 0\n", "Malformed harmonic coefficient row")
    test(args.aspect, args.stacktrace_plugin, args.work, "duplicate",
         header + all_rows + "2 2 0 0\n", "Duplicate harmonic coefficient")
    test(args.aspect, args.stacktrace_plugin, args.work, "invalid_order",
         header + "0 1 0 0\n", "Harmonic order exceeds degree")
    test(args.aspect, args.stacktrace_plugin, args.work, "missing",
         header + all_rows.rsplit("2 2", 1)[0], "missing one or more required")
    test(args.aspect, args.stacktrace_plugin, args.work, "unexpected_degree",
         header + all_rows + "3 0 0 0\n", "outside the configured observation range")
    work = args.work / "wrong_physical_file"
    work.mkdir()
    run(args.aspect, make_prm(work, args.stacktrace_plugin, args.dt_file), work,
        "canonical degree/order/cosine/sine header")
    print("GEOID_SH_OBSERVATION_NEGATIVE_OK")


if __name__ == "__main__":
    main()
