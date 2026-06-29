from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np


def plot_scalar_field(
    path: str | Path,
    S: np.ndarray,
    Y: np.ndarray,
    scalar: np.ndarray,
    solid_mask: np.ndarray,
    title: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "fluent",
) -> None:
    del S, Y, title, colorbar_label
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(scalar, dtype=np.float64)
    solid = solid_mask.astype(bool)
    fluid_values = values[~solid]
    if vmin is None:
        vmin = float(np.nanmin(fluid_values)) if fluid_values.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(fluid_values)) if fluid_values.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = _colormap(normalized, cmap)
    rgb[solid] = np.array([255, 255, 255], dtype=np.uint8)
    _write_rgb_png(path, _scale_nearest(rgb, 3))


def plot_geometry_overlay(
    path: str | Path, fluid_mask: np.ndarray, solid_mask: np.ndarray, near_solid_mask: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fluid = fluid_mask.astype(bool)
    solid = solid_mask.astype(bool)
    near_solid = near_solid_mask.astype(bool)
    rgb = np.zeros((*fluid.shape, 3), dtype=np.uint8)
    rgb[fluid] = np.array([219, 240, 250], dtype=np.uint8)
    rgb[near_solid] = np.array([255, 210, 130], dtype=np.uint8)
    rgb[solid] = np.array([255, 255, 255], dtype=np.uint8)
    _write_rgb_png(path, _scale_nearest(rgb, 3))


def plot_history(
    path: str | Path,
    rows: list[dict[str, float]],
    series: Iterable[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1000, 360
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    if not rows:
        _write_rgb_png(path, rgb)
        return
    x_values = np.array([row.get("step", index) for index, row in enumerate(rows)], dtype=np.float64)
    all_values = []
    for key in series:
        all_values.extend([row.get(key, 0.0) for row in rows])
    y_values = np.asarray(all_values, dtype=np.float64)
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        y_values = np.array([0.0, 1.0])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_max <= y_min:
        y_max = y_min + 1.0
    _draw_axes(rgb)
    colors = [
        np.array([220, 30, 30], dtype=np.uint8),
        np.array([20, 90, 220], dtype=np.uint8),
        np.array([20, 150, 80], dtype=np.uint8),
    ]
    for index, key in enumerate(series):
        points = [
            _map_point(
                x_values[i],
                float(row.get(key, 0.0)),
                float(x_values[0]),
                float(x_values[-1]),
                y_min,
                y_max,
                width,
                height,
            )
            for i, row in enumerate(rows)
        ]
        _draw_polyline(rgb, points, colors[index % len(colors)])
    _write_rgb_png(path, rgb)


def _colormap(normalized: np.ndarray, cmap: str) -> np.ndarray:
    if cmap == "diverging":
        stops = np.array(
            [
                [32, 76, 180],
                [235, 245, 255],
                [205, 34, 34],
            ],
            dtype=np.float64,
        )
    else:
        stops = np.array(
            [
                [35, 45, 210],
                [40, 180, 230],
                [50, 220, 120],
                [250, 230, 60],
                [225, 40, 20],
            ],
            dtype=np.float64,
        )
    x = normalized * (len(stops) - 1)
    lower = np.floor(x).astype(int)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    fraction = (x - lower)[..., np.newaxis]
    rgb = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    return rgb.astype(np.uint8)


def _scale_nearest(rgb: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def _draw_axes(rgb: np.ndarray) -> None:
    rgb[30:-30, 70] = 0
    rgb[-40, 70:-30] = 0


def _map_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    x_span = max(x_max - x_min, 1.0e-12)
    y_span = max(y_max - y_min, 1.0e-12)
    px = int(70 + (x - x_min) / x_span * (width - 100))
    py = int((height - 40) - (y - y_min) / y_span * (height - 80))
    return px, py


def _draw_polyline(
    rgb: np.ndarray, points: list[tuple[int, int]], color: np.ndarray
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        _draw_line(rgb, start, end, color)


def _draw_line(
    rgb: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: np.ndarray
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 * (1.0 - t) + x1 * t))
        y = int(round(y0 * (1.0 - t) + y1 * t))
        if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
            rgb[max(y - 1, 0) : min(y + 2, rgb.shape[0]), max(x - 1, 0) : min(x + 2, rgb.shape[1])] = color


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG writer expects uint8 RGB data")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw)

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)

    path.write_bytes(
        b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", compressed),
                chunk(b"IEND", b""),
            ]
        )
    )
