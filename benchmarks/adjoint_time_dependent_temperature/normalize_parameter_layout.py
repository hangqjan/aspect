#!/usr/bin/env python3
"""Normalize adjoint benchmark parameters into production/diagnostic groups.

The benchmark matrix intentionally uses small overlay files. This utility
keeps those overlays deterministic while the canonical ASPECT parameter tree
evolves. It moves only known direct children of ``subsection Adjoint`` and
leaves objective-specific nested subsections and numerical values untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


TIME_DEPENDENT_PARAMETERS = {
    "Trajectory storage mode",
    "Trajectory cleanup policy",
    "Trajectory records per segment",
    "Trajectory anchor spacing",
    "Adjoint checkpoint interval",
    "Advection preconditioner",
}

DIAGNOSTIC_PARAMETERS = {
    "Run trajectory replay diagnostic",
    "Trajectory replay tolerance",
    "Run sparse trajectory replay validation",
    "Stop after adjoint checkpoint",
    "Lifecycle failure injection",
    "Checkpoint transaction failure injection",
    "Checkpoint transaction failure rank",
    "Checkpoint transaction failure next reverse step",
    "Trajectory cleanup failure injection",
    "Trajectory cleanup failure rank",
    "Trajectory cleanup failure next reverse step",
    "Trajectory append failure injection",
    "Trajectory append failure rank",
    "Trajectory append failure step",
    "Run trajectory Stokes recovery diagnostic",
    "Run matrix-free cold rebuild diagnostic",
    "Output MPI adjoint memory diagnostics",
    "Trajectory Stokes recovery tolerance",
    "Run advection preconditioner comparison",
    "Run material model adjoint VJP test",
    "Material model adjoint VJP test epsilon",
    "Run C to Stokes RHS transpose test",
    "C to Stokes RHS transpose test epsilon",
    "Run direct Stokes viscosity derivative test",
    "Direct Stokes viscosity derivative test epsilon",
    "Run transposed advection constraint test",
    "Run artificial viscosity coefficient seed test",
    "Artificial viscosity coefficient seed test epsilon",
    "Run artificial viscosity estimator transpose test",
    "Artificial viscosity estimator transpose test epsilon",
    "Run local entropy residual transpose test",
    "Local entropy residual transpose test epsilon",
    "Run global entropy denominator scalar test",
    "Global entropy denominator scalar test epsilon",
    "Run global entropy variation transpose test",
    "Global entropy variation transpose test epsilon",
}

SET_PATTERN = re.compile(r"^\s*set\s+([^=]+?)\s*=")
SUBSECTION_PATTERN = re.compile(r"^\s*subsection\s+(.+?)\s*$")
END_PATTERN = re.compile(r"^\s*end\s*$")


def _group(lines: list[str], name: str) -> list[str]:
    result = [f"  subsection {name}\n"]
    result.extend(f"    {line.lstrip()}" for line in lines)
    result.append("  end\n")
    return result


def normalize(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    in_adjoint = False
    relative_depth = 0
    time_dependent: list[str] = []
    diagnostics: list[str] = []

    for line in lines:
        subsection = SUBSECTION_PATTERN.match(line)
        if not in_adjoint and subsection and subsection.group(1) == "Adjoint":
            in_adjoint = True
            relative_depth = 0
            time_dependent = []
            diagnostics = []
            output.append(line)
            continue

        if in_adjoint:
            if END_PATTERN.match(line) and relative_depth == 0:
                if time_dependent or diagnostics:
                    if output and output[-1].strip():
                        output.append("\n")
                    if time_dependent:
                        output.extend(_group(time_dependent, "Time dependent"))
                    if diagnostics:
                        if time_dependent:
                            output.append("\n")
                        output.extend(_group(diagnostics, "Diagnostics"))
                output.append(line)
                in_adjoint = False
                continue

            if relative_depth == 0:
                setting = SET_PATTERN.match(line)
                if setting:
                    parameter = setting.group(1).strip()
                    if parameter in TIME_DEPENDENT_PARAMETERS:
                        time_dependent.append(line)
                        continue
                    if parameter in DIAGNOSTIC_PARAMETERS:
                        diagnostics.append(line)
                        continue

            if subsection:
                relative_depth += 1
            elif END_PATTERN.match(line):
                relative_depth -= 1

        output.append(line)

    normalized = "".join(output)

    def indent_group(match: re.Match[str]) -> str:
        body = "".join(
            f"    {line.lstrip()}" if line.strip() else line
            for line in match.group(2).splitlines(keepends=True)
        )
        return match.group(1) + body + match.group(3)

    return re.sub(
        r"(^  subsection (?:Time dependent|Diagnostics)\n)(.*?)(^  end\n)",
        indent_group,
        normalized,
        flags=re.MULTILINE | re.DOTALL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    paths = arguments.paths or sorted(Path(__file__).parent.glob("*.prm"))
    changed: list[Path] = []
    for path in paths:
        original = path.read_text()
        normalized = normalize(original)
        if normalized != original:
            changed.append(path)
            if not arguments.check:
                path.write_text(normalized)

    if arguments.check and changed:
        for path in changed:
            print(path)
        return 1
    print(f"Normalized {len(changed)} parameter files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
