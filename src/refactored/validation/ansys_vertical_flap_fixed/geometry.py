from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def build_geometry_mask(config: dict) -> dict[str, np.ndarray]:
    grid = config["grid"]
    geometry = config["geometry"]
    ns = int(grid["ns"])
    ny = int(grid["ny"])
    duct_length = float(geometry["duct_length"])
    duct_height = float(geometry["duct_height"])
    flap_center_s = float(geometry["flap_center_s"])
    flap_thickness = float(geometry["flap_thickness"])
    gap_height = float(geometry["gap_height"])

    s = np.linspace(0.0, duct_length, ns, dtype=np.float64)
    y = np.linspace(-0.5 * duct_height, 0.5 * duct_height, ny, dtype=np.float64)
    S, Y = np.meshgrid(s, y)
    Z = -S

    solid = np.zeros((ny, ns), dtype=bool)
    wall = np.zeros_like(solid)
    if bool(geometry["wall_cells_are_solid"]):
        wall[0, :] = True
        wall[-1, :] = True
        solid |= wall

    flap_band = np.abs(S - flap_center_s) <= 0.5 * flap_thickness
    upper_flap = flap_band & (Y >= 0.5 * gap_height)
    lower_flap = flap_band & (Y <= -0.5 * gap_height)
    gap = flap_band & (np.abs(Y) < 0.5 * gap_height)
    flap = upper_flap | lower_flap
    solid |= flap
    fluid = ~solid

    return {
        "s": s,
        "y": y,
        "S": S,
        "Y": Y,
        "Z": Z,
        "solid_mask": solid,
        "fluid_mask": fluid,
        "wall_mask": wall,
        "flap_mask": flap,
        "upper_flap_mask": upper_flap,
        "lower_flap_mask": lower_flap,
        "gap_mask": gap & fluid,
    }


def save_geometry_mask(path: Path, geometry: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **geometry)


def plot_geometry_preview(path: Path, geometry: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _plot_geometry_preview_matplotlib(path, geometry)
    except ModuleNotFoundError:
        _plot_geometry_preview_png(path, geometry)


def _plot_geometry_preview_matplotlib(
    path: Path, geometry: dict[str, np.ndarray]
) -> None:
    import matplotlib.pyplot as plt

    rgb = _geometry_rgb(geometry).astype(np.float64) / 255.0
    extent = [
        float(geometry["s"][0]),
        float(geometry["s"][-1]),
        float(geometry["y"][0]),
        float(geometry["y"][-1]),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    ax.imshow(rgb, origin="lower", extent=extent, aspect="auto")
    ax.set_xlabel("display streamwise coordinate s (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("ANSYS vertical flap fixed-flow geometry mask")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_geometry_preview_png(path: Path, geometry: dict[str, np.ndarray]) -> None:
    rgb = np.repeat(_geometry_rgb(geometry), 3, axis=0)
    rgb = np.repeat(rgb, 3, axis=1)
    _write_rgb_png(path, rgb)


def _geometry_rgb(geometry: dict[str, np.ndarray]) -> np.ndarray:
    fluid = geometry["fluid_mask"].astype(bool)
    wall = geometry["wall_mask"].astype(bool)
    upper = geometry["upper_flap_mask"].astype(bool)
    lower = geometry["lower_flap_mask"].astype(bool)
    gap = geometry["gap_mask"].astype(bool)

    rgb = np.zeros((*fluid.shape, 3), dtype=np.uint8)
    rgb[fluid] = np.array([225, 243, 255], dtype=np.uint8)
    rgb[wall] = np.array([42, 51, 64], dtype=np.uint8)
    rgb[upper | lower] = np.array([235, 238, 244], dtype=np.uint8)
    rgb[gap] = np.array([176, 224, 255], dtype=np.uint8)
    return rgb


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG fallback expects uint8 RGB data")

    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw)

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)

    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)
