"""Build squid run GIFs in the project's two reference styles.

Styles are replicated from src/render_visit_side_section_gif.py:
  * panel: two-panel XZ/YZ turbo sections (matches rendered_results frames)
  * publication: dark centerline view (matches the publication video style)

Usage (Anaconda interpreter):
  python tools_build_squid_gif.py <run_dir> <out_prefix> [frame_stride]
Writes <out_prefix>_panel.gif and <out_prefix>_publication.gif.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

XZ_Y_M = 0.0155
YZ_X_M = -0.0300
XZ_X_RANGE_M = (-0.090, 0.030)
YZ_Y_RANGE_M = (-0.080, 0.120)
Z_RANGE_M = (0.955, 1.095)
PUBLICATION_X_RANGE_M = (-0.090, 0.030)
PUBLICATION_Z_RANGE_M = (0.925, 1.095)
PUBLICATION_CENTER_X_M = -0.0300
PUBLICATION_PROBES = (
    ("lip", 0.9678),
    ("outlet", 0.9565),
    ("downstream", 0.9415),
)
MARKER_SLAB_HALF_M = 0.0018

run_dir = Path(sys.argv[1])
out_prefix = Path(sys.argv[2])
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

snapshot_paths = sorted((run_dir / "snapshots").glob("snapshot_*.npz"))[::stride]
if not snapshot_paths:
    raise SystemExit(f"no snapshots under {run_dir / 'snapshots'}")


def load_static_obstacle_yz() -> np.ndarray | None:
    candidates = sorted(run_dir.glob("sharp_failure_step_*_fluid.vti"))
    if not candidates:
        return None
    text = candidates[-1].read_text(encoding="utf-8", errors="replace")
    extent = re.search(r'WholeExtent="([^"]+)"', text).group(1)
    x0, x1, y0, y1, z0, z1 = (int(value) for value in extent.split())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    match = re.search(r'Name="obstacle"[^>]*>\s*([^<]+)<', text, re.DOTALL)
    if match is None:
        return None
    obstacle = np.fromstring(match.group(1), sep=" ", dtype=np.float64)
    obstacle = obstacle.reshape((nz, ny, nx)).transpose(2, 1, 0)
    return obstacle[nx // 2, :, :]


static_obstacle_yz = load_static_obstacle_yz()

with np.load(snapshot_paths[0]) as data:
    base_positions = np.asarray(data["marker_positions_m"], dtype=np.float32)

speed_samples: list[np.ndarray] = []
for path in snapshot_paths:
    with np.load(path) as data:
        speed_samples.append(np.asarray(data["speed_xz"], dtype=np.float32).ravel())
        speed_samples.append(np.asarray(data["speed_yz"], dtype=np.float32).ravel())
merged = np.concatenate(speed_samples)
panel_vmax = max(float(np.nanpercentile(merged, 99.0)), 1.0e-6)
publication_vmax = max(float(np.nanpercentile(merged, 99.95)), 1.0e-6)

publication_cmap = mcolors.LinearSegmentedColormap.from_list(
    "publication_speed",
    ["#24143b", "#30205f", "#1f4f85", "#15938c", "#a9dc4e", "#f07a25", "#8f120e"],
)


def panel_frame(data: dict[str, np.ndarray]) -> Image.Image:
    speed_xz = data["speed_xz"]
    speed_yz = data["speed_yz"]
    obstacle_xz = data["obstacle_xz"]
    positions = data["marker_positions_m"]
    cx = data["cell_center_x_m"]
    cy = data["cell_center_y_m"]
    cz = data["cell_center_z_m"]
    displacement = np.linalg.norm(positions - base_positions, axis=1)
    displacement_vmax = max(float(np.nanmax(displacement)), 1.0e-6)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), dpi=110)
    panels = (
        (
            axes[0],
            np.where(obstacle_xz > 0.5, np.nan, speed_xz),
            cx,
            XZ_X_RANGE_M,
            0,
            f"XZ section at y={XZ_Y_M:.4f} m",
        ),
        (
            axes[1],
            speed_yz,
            cy,
            YZ_Y_RANGE_M,
            1,
            f"YZ section at x={YZ_X_M:.4f} m",
        ),
    )
    mesh = None
    for axis, field, h_centers, h_range, h_index, title in panels:
        mesh = axis.imshow(
            field.T,
            origin="lower",
            extent=(
                float(h_centers[0]),
                float(h_centers[-1]),
                float(cz[0]),
                float(cz[-1]),
            ),
            aspect="equal",
            cmap="turbo",
            vmin=0.0,
            vmax=panel_vmax,
            interpolation="nearest",
        )
        slab_value = XZ_Y_M if h_index == 0 else YZ_X_M
        slab_axis = 1 if h_index == 0 else 0
        slab = np.abs(positions[:, slab_axis] - slab_value) <= MARKER_SLAB_HALF_M
        if slab.any():
            axis.scatter(
                positions[slab, h_index],
                positions[slab, 2],
                c=displacement[slab],
                cmap="magma",
                s=5,
                linewidths=0.0,
                alpha=0.80,
                vmin=0.0,
                vmax=displacement_vmax,
            )
        axis.set_title(title)
        axis.set_xlim(*h_range)
        axis.set_ylim(*Z_RANGE_M)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#cbd5e1", linewidth=0.4, alpha=0.45)
        axis.set_xlabel("x (m)" if h_index == 0 else "y (m)")
        axis.set_ylabel("z (m)")
    colorbar = figure.colorbar(
        mesh, ax=axes.ravel().tolist(), shrink=0.90, pad=0.01
    )
    colorbar.set_label("fluid speed (m/s)")
    figure.suptitle(
        "FSI side sections from sharp snapshots | "
        f"step {int(data['step']):06d}  t={float(data['time_s']):.4f}s",
        fontsize=13,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def publication_frame(data: dict[str, np.ndarray]) -> Image.Image:
    speed_xz = data["speed_xz"]
    obstacle_xz = data["obstacle_xz"]
    positions = data["marker_positions_m"]
    cx = data["cell_center_x_m"]
    cz = data["cell_center_z_m"]
    figure, axis = plt.subplots(figsize=(6.4, 8.0), dpi=110)
    figure.patch.set_facecolor("#24143b")
    axis.set_facecolor("#24143b")
    axis.set_xlim(*PUBLICATION_X_RANGE_M)
    axis.set_ylim(*PUBLICATION_Z_RANGE_M)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.linspace(*PUBLICATION_X_RANGE_M, 4))
    axis.set_yticks(np.linspace(*PUBLICATION_Z_RANGE_M, 5))
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axis.grid(color="#d7cceb", linestyle=":", linewidth=0.6, alpha=0.18)
    for spine in axis.spines.values():
        spine.set_visible(False)
    field = np.where(obstacle_xz > 0.5, np.nan, speed_xz)
    axis.imshow(
        field.T,
        origin="lower",
        extent=(float(cx[0]), float(cx[-1]), float(cz[0]), float(cz[-1])),
        aspect="equal",
        cmap=publication_cmap,
        norm=mcolors.PowerNorm(gamma=0.72, vmin=0.0, vmax=publication_vmax),
        interpolation="bilinear",
        alpha=0.88,
    )
    axis.scatter(
        positions[:, 0],
        positions[:, 2],
        s=1.2,
        c="#d8c5ff",
        linewidths=0.0,
        alpha=0.24,
    )
    guide_color = "#f04b2f"
    axis.axvline(PUBLICATION_CENTER_X_M, color=guide_color, linewidth=1.2, alpha=0.92)
    for label, z_m in PUBLICATION_PROBES:
        axis.plot(
            [PUBLICATION_CENTER_X_M - 0.0030, PUBLICATION_CENTER_X_M + 0.0030],
            [z_m, z_m],
            color=guide_color,
            linewidth=1.8,
            solid_capstyle="round",
        )
        axis.text(
            PUBLICATION_CENTER_X_M + 0.0042,
            z_m,
            label,
            color=guide_color,
            fontsize=7,
            ha="left",
            va="center",
            fontweight="bold",
        )
    axis.text(
        0.03,
        0.985,
        f"step {int(data['step']):06d}  t={float(data['time_s']):.4f}s",
        transform=axis.transAxes,
        va="top",
        ha="left",
        color="#e5d7ff",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="#24143b", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


panel_frames: list[Image.Image] = []
publication_frames: list[Image.Image] = []
for path in snapshot_paths:
    with np.load(path) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    panel_frames.append(panel_frame(data))
    publication_frames.append(publication_frame(data))

for frames, suffix in (
    (panel_frames, "_panel.gif"),
    (publication_frames, "_publication.gif"),
):
    out_path = out_prefix.parent / (out_prefix.name + suffix)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out_path} with {len(frames)} frames")
