"""Tiny SVG/PNG plotting helpers with no third-party plotting dependency."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np


def colour(value: float, low: float, high: float, symmetric: bool = False) -> str:
    if symmetric:
        scale = max(abs(low), abs(high), 1.0e-30)
        q = max(0.0, min(1.0, 0.5 + 0.5 * value / scale))
        return f"rgb({int(255*q)},{int(255*(1-abs(2*q-1)))},{int(255*(1-q))})"
    q = max(0.0, min(1.0, (value-low) / max(high-low, 1.0e-30)))
    return f"rgb({int(40+210*q)},{int(20+180*q)},{int(150-110*q)})"


def heatmap(grid: np.ndarray, x: float, y: float, width: float, height: float,
            low: float, high: float, symmetric: bool = False) -> str:
    rows, columns = grid.shape
    cells: list[str] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            value = float(np.mean(grid[row:row+2, column:column+2]))
            left = x + width * column / (columns - 1)
            bottom = y + height * (rows - 2 - row) / (rows - 1)
            cells.append(f'<rect x="{left:.2f}" y="{bottom:.2f}" width="{width/(columns-1)+.2:.2f}" '
                         f'height="{height/(rows-1)+.2:.2f}" fill="{colour(value, low, high, symmetric)}"/>')
    return "".join(cells)


def svg_document(width: int, height: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111} .small{font-size:11px}'
            '.title{font-size:14px;font-weight:bold}</style>' + body + '</svg>')


def write_svg_png(svg: str, png: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png.with_suffix(".svg")
    svg_path.write_text(svg)
    result = subprocess.run(["/usr/bin/qlmanage", "-t", "-s", "1200", "-o", str(png.parent), str(svg_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quick Look could not create {png}: {result.stderr.strip()}")
    rendered = png.parent / (svg_path.name + ".png")
    if not rendered.exists():
        raise RuntimeError(f"Quick Look did not produce {rendered}")
    rendered.replace(png)
    svg_path.unlink()


def read_history(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append({key: (float(value) if key not in ("update_type",) and value else value)
                         for key, value in row.items()})
    return rows
