#!/usr/bin/env python3
"""Plot true y=0 meridional temperature reconstructions from accepted controls."""

from __future__ import annotations

import argparse
import base64
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


METHODS = (("Dynamic topography", "inverse-dynamic-topography"),
           ("Self-gravitating geoid", "inverse-geoid"),
           ("Combined", "inverse-combined"))
INNER_RADIUS = 3.480e6
OUTER_RADIUS = 6.336e6


def decode(array: ET.Element) -> np.ndarray:
    text = "".join((array.text or "").split())
    if array.attrib.get("format") == "ascii":
        return np.fromstring(text, sep=" ")
    header = base64.b64decode(text[:24])
    blocks, block_size, last_size, compressed_size = struct.unpack("<4I", header)
    if blocks != 1 or block_size != last_size:
        raise RuntimeError("plotter currently expects one VTK compressed block")
    payload = zlib.decompress(base64.b64decode(text[24:]))
    if len(payload) != last_size or compressed_size == 0:
        raise RuntimeError("invalid VTK compressed payload")
    dtype = {"Float32": "<f4", "Float64": "<f8"}[array.attrib["type"]]
    return np.frombuffer(payload, dtype=dtype)


def field_piece(piece: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    arrays = piece.findall(".//DataArray")
    points = piece.find("./Points/DataArray")
    if points is None:
        raise RuntimeError("no point coordinates in VTK piece")
    temperature = next(item for item in arrays if item.attrib.get("Name") == "T")
    return decode(points).reshape((-1, 3)), decode(temperature)


def field_vtu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    pieces = root.findall(".//Piece")
    if not pieces:
        raise RuntimeError(f"no VTK pieces in {path}")
    values = [field_piece(piece) for piece in pieces]
    return np.concatenate([item[0] for item in values]), np.concatenate([item[1] for item in values])


def field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".vtu":
        return field_vtu(path)
    root = ET.parse(path).getroot()
    pieces = [path.parent / item.attrib["Source"]
              for item in root.findall(".//Piece")]
    if not pieces:
        raise RuntimeError(f"no VTK pieces in {path}")
    values = [field_vtu(piece) for piece in pieces]
    return np.concatenate([item[0] for item in values]), np.concatenate([item[1] for item in values])


def control_files(directory: Path) -> list[Path]:
    controls = directory / "diagnostics/accepted_controls"
    parallel = sorted(controls.glob("initial_temperature_iter_*.pvtu"))
    if parallel:
        return parallel
    serial = sorted(controls.glob("initial_temperature_iter_*.vtu"))
    if not serial:
        raise RuntimeError(f"no accepted temperature controls in {controls}")
    return serial


def meridional_field(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    on_plane = np.isclose(points[:, 1], 0.0, atol=1e-5)
    points, values = points[on_plane], values[on_plane]
    if len(points) == 0:
        raise RuntimeError("output has no y=0 meridional-plane points")
    keys = np.round(points[:, (0, 2)], 3)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique[:, 0] / 1e6, unique[:, 1] / 1e6, sums / counts


def triangulation(x: np.ndarray, z: np.ndarray) -> mtri.Triangulation:
    mesh = mtri.Triangulation(x, z)
    centroid_x = x[mesh.triangles].mean(axis=1) * 1e6
    centroid_z = z[mesh.triangles].mean(axis=1) * 1e6
    radius = np.hypot(centroid_x, centroid_z)
    mesh.set_mask((radius < INNER_RADIUS) | (radius > OUTER_RADIUS))
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(len(METHODS), 4, figsize=(15, 12), constrained_layout=True)
    field_artist = error_artist = None
    for row, (label, directory) in enumerate(METHODS):
        files = control_files(args.root / directory)
        start_points, start_temperature = field(files[0])
        final_points, final_temperature = field(files[-1])
        start_x, start_z, start_temperature = meridional_field(start_points, start_temperature)
        final_x, final_z, final_temperature = meridional_field(final_points, final_temperature)
        if not (np.array_equal(start_x, final_x) and np.array_equal(start_z, final_z)):
            raise RuntimeError(f"initial/final mesh mismatch for {label}")
        reference_temperature = 1600 + 100 * np.exp(
            -((start_x * 1e6) ** 2 + (start_z * 1e6 - 6100000) ** 2)
            / (2 * (250000) ** 2))
        mesh = triangulation(start_x, start_z)
        panels = ((reference_temperature - 1600, "Reference anomaly", "magma", 0, 100),
                  (start_temperature - 1600, "Initial/background anomaly", "magma", 0, 100),
                  (final_temperature - 1600, "Final reconstruction anomaly", "magma", 0, 100),
                  (np.abs(final_temperature - reference_temperature), "Absolute error", "Greys", 0, 100))
        for column, (values, title, cmap, vmin, vmax) in enumerate(panels):
            artist = axes[row, column].tripcolor(mesh, values, shading="gouraud",
                                                  cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(label if column == 0 else "")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
            if column < 3:
                field_artist = artist
            else:
                error_artist = artist
    figure.colorbar(field_artist, ax=axes[:, :3], orientation="horizontal",
                    pad=0.04, shrink=0.75, label="Temperature anomaly (K)")
    figure.colorbar(error_artist, ax=axes[:, 3], orientation="horizontal",
                    pad=0.04, shrink=0.85, label="Absolute error (K)")
    figure.suptitle("Meridional temperature reconstruction: y = 0", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
