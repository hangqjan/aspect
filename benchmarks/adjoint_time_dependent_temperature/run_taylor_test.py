#!/usr/bin/env python3

import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import argparse
from typing import Optional


ROOT = pathlib.Path(__file__).resolve().parents[2]
OBJECTIVE_RE = re.compile(r"Time-dependent (?:temperature|terminal) objective '[^']+' = ([0-9.eE+-]+)")


def read_prm_with_local_includes(path: pathlib.Path) -> str:
    text = path.read_text()
    expanded_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("include "):
            include_target = stripped.split(None, 1)[1]
            include_target = include_target.replace("$ASPECT_SOURCE_DIR", str(ROOT))
            include_path = pathlib.Path(include_target)
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            expanded_lines.append(read_prm_with_local_includes(include_path))
        else:
            expanded_lines.append(line)
    return "\n".join(expanded_lines) + "\n"


def write_prm(base_prm: pathlib.Path,
              run_dir: pathlib.Path,
              epsilon: float,
              end_step: int,
              constant_name: str,
              trajectory_storage_mode: Optional[str]) -> pathlib.Path:
    prm_text = read_prm_with_local_includes(base_prm)
    prm_text = prm_text.replace(f"{constant_name}=0",
                                f"{constant_name}={epsilon:.16e}")
    prm_text = re.sub(r"set End step\s*=.*",
                      f"set End step = {end_step}",
                      prm_text,
                      count=1)
    prm_text = re.sub(r"set Output directory\s*=.*",
                      f"set Output directory = {run_dir / 'output'}",
                      prm_text)
    if trajectory_storage_mode is not None:
        # Appending a second Adjoint subsection intentionally overrides the
        # base benchmark without changing its physics or included files.
        prm_text += ("\nsubsection Adjoint\n"
                     f"  set Trajectory storage mode = {trajectory_storage_mode}\n"
                     "end\n")
    prm_path = run_dir / "input.prm"
    prm_path.write_text(prm_text)
    return prm_path


def run_model(binary: pathlib.Path,
              base_prm: pathlib.Path,
              work_dir: pathlib.Path,
              epsilon: float,
              end_step: int,
              constant_name: str,
              trajectory_storage_mode: Optional[str]):
    run_dir = work_dir / f"eps_{epsilon:.1e}".replace("+", "")
    run_dir.mkdir()
    prm_path = write_prm(base_prm, run_dir, epsilon, end_step, constant_name,
                         trajectory_storage_mode)
    completed = subprocess.run([str(binary), str(prm_path)],
                               cwd=ROOT,
                               text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               check=False)
    (run_dir / "screen-output").write_text(completed.stdout)
    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"ASPECT failed for epsilon={epsilon} with exit code {completed.returncode}")

    match = OBJECTIVE_RE.search(completed.stdout)
    if match is None:
        print(completed.stdout)
        raise RuntimeError(f"Could not find objective value for epsilon={epsilon}")

    return float(match.group(1)), run_dir / "output"


def perturbation_value(x: float, y: float) -> float:
    return math.sin(2.0 * math.pi * x) * math.sin(math.pi * y)


def adjoint_directional_derivative(output_dir: pathlib.Path, gradient_field: str) -> float:
    if gradient_field == "temperature":
        gradient_file = output_dir / "initial_temperature_gradient_rank_00000.txt"
    elif gradient_field.startswith("composition:"):
        composition_name = gradient_field.split(":", 1)[1]
        gradient_file = output_dir / f"initial_composition_gradient_{composition_name}_rank_00000.txt"
    else:
        raise RuntimeError(f"Unsupported gradient field: {gradient_field}")

    total = 0.0
    with gradient_file.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            gradient = float(fields[1])
            x = float(fields[2])
            y = float(fields[3])
            total += gradient * perturbation_value(x, y)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a finite-difference Taylor test for the time-dependent temperature adjoint benchmark.")
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build-debug" / "aspect-debug"),
                        help="Path to the ASPECT executable.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temporary run directories for inspection.")
    parser.add_argument("--end-step", type=int, default=1,
                        help="Termination end step used for every ASPECT run.")
    parser.add_argument("--base-prm", default=str(ROOT / "benchmarks" / "adjoint_time_dependent_temperature" / "adjoint_time_dependent_temperature.prm"),
                        help="Base parameter file to perturb.")
    parser.add_argument("--epsilons", default="1e-2,5e-3,2.5e-3,1.25e-3,6.25e-4",
                        help="Comma-separated perturbation amplitudes.")
    parser.add_argument("--tolerance", type=float, default=5.0e-2,
                        help="Maximum allowed relative slope error on the finest epsilon.")
    parser.add_argument("--absolute-slope-tolerance", type=float, default=1.0e-12,
                        help="Treat slopes below this absolute scale as zero when deciding pass/fail.")
    parser.add_argument("--constant-name", default="eps",
                        help="Function constant that is replaced by the perturbation amplitude.")
    parser.add_argument("--gradient-field", default="temperature",
                        help="Gradient file to test: 'temperature' or 'composition:<field_name>'.")
    parser.add_argument("--trajectory-storage-mode",
                        choices=("minimal", "stokes cached"),
                        help="Override the archive mode in every temporary input.")
    args = parser.parse_args()

    binary = pathlib.Path(args.binary)
    if not binary.exists():
        raise RuntimeError(f"ASPECT binary does not exist: {binary}")
    base_prm = pathlib.Path(args.base_prm)
    if not base_prm.exists():
        raise RuntimeError(f"Base parameter file does not exist: {base_prm}")

    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="aspect-adjoint-taylor-"))
    try:
        j0, output0 = run_model(binary, base_prm, tmp_root, 0.0, args.end_step,
                                 args.constant_name, args.trajectory_storage_mode)
        prediction = adjoint_directional_derivative(output0, args.gradient_field)

        epsilons = [float(value) for value in args.epsilons.split(",") if value.strip()]
        if len(epsilons) == 0:
            raise RuntimeError("At least one epsilon is required.")

        errors = []
        remainders = []
        previous_epsilon = None
        print(f"end step = {args.end_step}")
        print(f"base prm = {base_prm}")
        print(f"gradient field = {args.gradient_field}")
        print(f"trajectory storage mode = {args.trajectory_storage_mode or 'base input'}")
        print(f"J(0) = {j0:.16e}")
        print(f"adjoint directional derivative = {prediction:.16e}")
        print("# epsilon finite_difference relative_slope_error taylor_remainder observed_order")
        for epsilon in epsilons:
            jplus, _ = run_model(binary, base_prm, tmp_root, epsilon, args.end_step,
                                  args.constant_name, args.trajectory_storage_mode)
            jminus, _ = run_model(binary, base_prm, tmp_root, -epsilon, args.end_step,
                                   args.constant_name, args.trajectory_storage_mode)
            finite_difference = (jplus - jminus) / (2.0 * epsilon)
            slope_error = (
                abs(finite_difference - prediction)
                / max(abs(finite_difference), abs(prediction), 1.0e-30)
            )
            remainder = max(abs(jplus - j0 - epsilon * prediction),
                            abs(jminus - j0 + epsilon * prediction))
            errors.append(slope_error)
            remainders.append(remainder)
            observed_order = float("nan")
            if previous_epsilon is not None and remainder > 0.0 and remainders[-2] > 0.0:
                observed_order = math.log(remainders[-2] / remainder) / math.log(previous_epsilon / epsilon)
            print(f"{epsilon:.8e} {finite_difference:.16e} {slope_error:.16e} {remainder:.16e} {observed_order:.8e}")
            previous_epsilon = epsilon

        finest_scale = max(abs(finite_difference), abs(prediction))
        if errors[-1] > args.tolerance and finest_scale > args.absolute_slope_tolerance:
            print(f"FAIL: finest relative slope error {errors[-1]:.3e} exceeds {args.tolerance:.3e}")
            return 1

        print("PASS: finite-difference slope agrees with adjoint prediction on the finest epsilon.")
        return 0
    finally:
        if args.keep:
            print(f"Kept temporary runs in {tmp_root}")
        else:
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    sys.exit(main())
