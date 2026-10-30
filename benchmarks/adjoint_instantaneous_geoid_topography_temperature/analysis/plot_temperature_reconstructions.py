#!/usr/bin/env python3
"""Plot a common meridional slice of accepted steady temperature controls."""

from __future__ import annotations

import argparse
import base64
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def decode(array: ET.Element) -> np.ndarray:
    """Read ASPECT's inline, zlib-compressed VTK XML DataArray."""
    text = "".join((array.text or "").split())
    if array.attrib.get("format") == "ascii":
        return np.fromstring(text, sep=" ")
    # ASPECT writes an inline base64 header followed by a separately encoded
    # zlib payload. All acceptance files here have one compressed block.
    header = base64.b64decode(text[:24])
    blocks, block_size, last_size, compressed_size = struct.unpack("<4I", header)
    if blocks != 1 or block_size != last_size:
        raise RuntimeError("plotter currently expects one VTK compressed block")
    payload = zlib.decompress(base64.b64decode(text[24:]))
    if len(payload) != last_size or compressed_size == 0:
        raise RuntimeError("invalid VTK compressed payload")
    dtype = {"Float32": "<f4", "Float64": "<f8"}[array.attrib["type"]]
    return np.frombuffer(payload, dtype=dtype)


def field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    arrays = root.findall(".//DataArray")
    points = root.find(".//Points/DataArray")
    if points is None:
        raise RuntimeError(f"no point coordinates in {path}")
    temperature = next(item for item in arrays if item.attrib.get("Name") == "T")
    return decode(points).reshape((-1, 3)), decode(temperature)


def slice_mask(points: np.ndarray) -> np.ndarray:
    # The coarsest spherical-shell mesh contains several azimuthal planes.
    # Keep the nearest meridional slab, common to reference and every control.
    return np.abs(points[:, 1]) <= np.quantile(np.abs(points[:, 1]), 0.12)


def align_to_reference(points: np.ndarray, values: np.ndarray,
                       reference_points: np.ndarray) -> np.ndarray:
    """Match a deterministic ASPECT output ordering by rounded coordinates."""
    lookup = {tuple(np.round(point, 2)): value for point, value in zip(points, values, strict=True)}
    try:
        return np.asarray([lookup[tuple(np.round(point, 2))] for point in reference_points])
    except KeyError as error:
        raise RuntimeError("control mesh coordinates differ from reference") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = (("Dynamic topography", "inverse-dynamic-topography"),
               ("Self-gravitating geoid", "inverse-geoid"),
               ("Combined", "inverse-combined"))
    first_control = sorted((args.root / methods[0][1] / "diagnostics/accepted_controls").glob("*.vtu"))[0]
    reference, _ = field(first_control)
    # This analytic profile is the reference initial-temperature model in
    # reference.prm. Evaluating it at the accepted-control points avoids an
    # incidental visualization patch-order dependency.
    reference_temperature = 1600 + 100 * np.exp(
        -(reference[:, 0] ** 2 + reference[:, 1] ** 2
          + (reference[:, 2] - 6100000) ** 2) / (2 * 250000 ** 2))
    mask = slice_mask(reference)
    reference_xy = reference[mask][:, (0, 2)] / 1e6
    reference_temperature = reference_temperature[mask]
    figure, axes = plt.subplots(len(methods) + 1, 3, figsize=(12, 14),
                                constrained_layout=True)
    scatter = axes[0, 0].scatter(reference_xy[:, 0], reference_xy[:, 1],
                                 c=reference_temperature, s=10, cmap="inferno",
                                 vmin=1600, vmax=1700)
    axes[0, 0].set_title("Reference temperature")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    for row, (label, directory) in enumerate(methods, start=1):
        controls = sorted((args.root / directory / "diagnostics/accepted_controls").glob("*.vtu"))
        if not controls:
            raise RuntimeError(f"no accepted controls for {label}")
        start_points, start_temperature = field(controls[0])
        final_points, final_temperature = field(controls[-1])
        start_temperature = align_to_reference(start_points, start_temperature, reference)
        final_temperature = align_to_reference(final_points, final_temperature, reference)
        for column, (values, heading, cmap, vmin, vmax) in enumerate(
                ((start_temperature[mask], "Initial/background", "inferno", 1600, 1700),
                 (final_temperature[mask], "Final reconstruction", "inferno", 1600, 1700),
                 (np.abs(final_temperature[mask] - reference_temperature), "Final absolute error", "Greys", 0, 100))):
            axes[row, column].scatter(reference_xy[:, 0], reference_xy[:, 1], c=values,
                                      s=10, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].set_title(heading if row == 1 else "")
            axes[row, column].set_ylabel(label if column == 0 else "")
    for axis in axes.flat:
        if axis.has_data():
            axis.set_aspect("equal")
            axis.set_xticks(())
            axis.set_yticks(())
    figure.colorbar(scatter, ax=axes[:, :2], shrink=0.65, label="Temperature (K)")
    figure.suptitle("Instantaneous temperature inversion from surface observations", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
