#!/usr/bin/env python3
"""Small dependency-free reader for ASPECT's serial compressed VTU output."""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np


_DTYPES = {
    "Float32": ("f", np.float32),
    "Float64": ("d", np.float64),
    "Int32": ("i", np.int32),
    "UInt8": ("B", np.uint8),
}


def _decode_binary(text: str, vtk_type: str) -> np.ndarray:
    """Decode VTK XML's vtkZLibDataCompressor block representation."""
    encoded = "".join(text.split())
    first = base64.b64decode(encoded[:8])
    blocks = struct.unpack("<I", first[:4])[0]
    header_bytes = (blocks + 3) * 4
    header_chars = 4 * ((header_bytes + 2) // 3)
    header = base64.b64decode(encoded[:header_chars])
    values = struct.unpack("<" + "I" * (blocks + 3), header)
    compressed_sizes = values[3:]
    payload = base64.b64decode(encoded[header_chars:])
    offset = 0
    raw = bytearray()
    for size in compressed_sizes:
        raw.extend(zlib.decompress(payload[offset:offset + size]))
        offset += size
    code, dtype = _DTYPES[vtk_type]
    del code
    return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).copy()


def read_vtu(path: str | Path) -> dict[str, np.ndarray]:
    """Return Points and PointData arrays from one serial ASPECT VTU file."""
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError(f"{path}: missing UnstructuredGrid/Piece")
    arrays: dict[str, np.ndarray] = {}
    point_array = piece.find("./Points/DataArray")
    if point_array is None:
        raise ValueError(f"{path}: missing Points/DataArray")
    npoints = int(piece.attrib["NumberOfPoints"])
    points = _decode_binary(point_array.text or "", point_array.attrib["type"])
    arrays["Points"] = points.reshape((npoints, int(point_array.attrib.get("NumberOfComponents", "3"))))
    point_data = piece.find("PointData")
    if point_data is not None:
        for array in point_data.findall("DataArray"):
            name = array.attrib.get("Name")
            if name and array.attrib.get("format") == "binary":
                data = _decode_binary(array.text or "", array.attrib["type"])
                arrays[name] = data.reshape((npoints, int(array.attrib.get("NumberOfComponents", "1"))))
    return arrays


def scalar_grid(path: str | Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y and a structured scalar grid from a regular ASPECT box VTU."""
    data = read_vtu(path)
    points = data["Points"]
    values = data[field].reshape(-1)
    x = np.unique(points[:, 0])
    y = np.unique(points[:, 1])
    grid = np.full((len(y), len(x)), np.nan)
    ix = {round(float(value), 12): index for index, value in enumerate(x)}
    iy = {round(float(value), 12): index for index, value in enumerate(y)}
    for point, value in zip(points, values, strict=True):
        grid[iy[round(float(point[1]), 12)], ix[round(float(point[0]), 12)]] = value
    if np.isnan(grid).any():
        raise ValueError(f"{path}: {field} is not a complete structured point grid")
    return x, y, grid
