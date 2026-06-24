from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


XZ_Y_M = 0.0155
PUBLICATION_X_RANGE_M = (-0.090, 0.030)
PUBLICATION_Z_RANGE_M = (0.925, 1.095)
PUBLICATION_CENTER_X_M = -0.0300
PUBLICATION_PROBES: tuple[tuple[str, float], ...] = (
    ("lip", 0.9678),
    ("outlet", 0.9565),
    ("downstream", 0.9415),
)


def _complete_records(visit_dir: Path, min_age_s: float) -> list[dict[str, Any]]:
    manifest = json.loads((visit_dir / "manifest.json").read_text(encoding="utf-8"))
    now_ns = time.time_ns()
    records: list[dict[str, Any]] = []
    for record in manifest.get("records") or []:
        files = record.get("files") or {}
        required = [
            files.get("fluid_vti"),
            files.get("markers_vtp"),
            files.get("surface_vtp"),
        ]
        if not all(required):
            continue
        paths = [visit_dir / str(name) for name in required]
        if not all(path.exists() and path.stat().st_size > 0 for path in paths):
            continue
        newest_ns = max(path.stat().st_mtime_ns for path in paths)
        if min_age_s > 0.0 and (now_ns - newest_ns) < min_age_s * 1.0e9:
            continue
        records.append(record)
    return sorted(records, key=lambda item: int(item.get("step_index", -1)))


def _slice_arrays(
    fluid: Any,
    *,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sliced = fluid.slice(normal=normal, origin=origin)
    points = np.asarray(sliced.points, dtype=np.float64)
    speed = np.asarray(sliced["speed_mps"], dtype=np.float64)
    velocity = np.asarray(sliced["velocity_mps"], dtype=np.float64)
    excluded = np.zeros(speed.shape, dtype=bool)
    array_names = set(sliced.array_names)
    if "visual_excluded_mask" in array_names:
        excluded |= np.asarray(sliced["visual_excluded_mask"], dtype=np.float64) > 0.5
    else:
        for name in ("solid_extension_mask", "solid_obstacle_full_mask", "inactive_water_mask"):
            if name in array_names:
                excluded |= np.asarray(sliced[name], dtype=np.float64) > 0.5
    if "active_water_mask" in array_names:
        active = np.asarray(sliced["active_water_mask"], dtype=np.float64) > 0.5
    else:
        active = ~excluded
    physical = active & ~excluded & np.isfinite(speed)
    return points, speed, velocity, physical


def _point_slab(polydata: Any, *, axis: int, value: float, half_thickness_m: float) -> np.ndarray:
    points = np.asarray(polydata.points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    mask = np.abs(points[:, axis] - value) <= half_thickness_m
    return points[mask]


def _frame_scalar(
    *,
    speed: np.ndarray,
    velocity: np.ndarray,
    scalar: str,
) -> np.ndarray:
    if scalar == "downward":
        return np.maximum(0.0, -velocity[:, 2])
    if scalar == "speed":
        return speed
    raise ValueError(f"Unsupported scalar: {scalar}")


def _mask_nonphysical_and_floor_values(
    values: np.ndarray,
    *,
    physical: np.ndarray,
    vmin_mps: float,
) -> np.ndarray:
    masked = np.where(physical, values, np.nan)
    finite = np.isfinite(masked)
    return np.where(finite & (masked < vmin_mps), 0.0, masked)


def _draw_centerline_frame(
    ax: Any,
    *,
    points: np.ndarray,
    values: np.ndarray,
    physical: np.ndarray,
    surface_points: np.ndarray,
    marker_points: np.ndarray,
    vmin_mps: float,
    vmax_mps: float,
    scalar_label: str,
) -> Any:
    import matplotlib.colors as mcolors
    import matplotlib.tri as mtri

    x = points[:, 0]
    z = points[:, 2]
    in_bounds = (
        (x >= PUBLICATION_X_RANGE_M[0])
        & (x <= PUBLICATION_X_RANGE_M[1])
        & (z >= PUBLICATION_Z_RANGE_M[0])
        & (z <= PUBLICATION_Z_RANGE_M[1])
    )

    ax.set_facecolor("#24143b")
    ax.set_xlim(*PUBLICATION_X_RANGE_M)
    ax.set_ylim(*PUBLICATION_Z_RANGE_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(PUBLICATION_X_RANGE_M[0], PUBLICATION_X_RANGE_M[1], 4))
    ax.set_yticks(np.linspace(PUBLICATION_Z_RANGE_M[0], PUBLICATION_Z_RANGE_M[1], 5))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(color="#d7cceb", linestyle=":", linewidth=0.6, alpha=0.16)
    for spine in ax.spines.values():
        spine.set_visible(False)

    mesh = None
    if np.any(in_bounds):
        view_idx = np.flatnonzero(in_bounds)
        triang = mtri.Triangulation(x[view_idx], z[view_idx])
        view_physical = physical[view_idx] & np.isfinite(values[view_idx])
        triangle_physical = view_physical[triang.triangles].all(axis=1)

        triangle_points = np.column_stack((x[view_idx], z[view_idx]))
        tri = triang.triangles
        edge01 = np.linalg.norm(triangle_points[tri[:, 0]] - triangle_points[tri[:, 1]], axis=1)
        edge12 = np.linalg.norm(triangle_points[tri[:, 1]] - triangle_points[tri[:, 2]], axis=1)
        edge20 = np.linalg.norm(triangle_points[tri[:, 2]] - triangle_points[tri[:, 0]], axis=1)
        finite_edges = np.concatenate((edge01, edge12, edge20))
        finite_edges = finite_edges[np.isfinite(finite_edges) & (finite_edges > 0.0)]
        if finite_edges.size:
            max_edge = 3.5 * float(np.nanmedian(finite_edges))
            triangle_physical &= (edge01 <= max_edge) & (edge12 <= max_edge) & (edge20 <= max_edge)
        triang.set_mask(~triangle_physical)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "jet_check",
            [
                "#24143b",
                "#25174d",
                "#16345f",
                "#125f76",
                "#0f9588",
                "#c8d94a",
                "#f27a24",
                "#9a110e",
            ],
        )
        view_values = _mask_nonphysical_and_floor_values(
            values[view_idx],
            physical=view_physical,
            vmin_mps=vmin_mps,
        )
        mesh = ax.tripcolor(
            triang,
            view_values,
            shading="flat",
            cmap=cmap,
            norm=mcolors.PowerNorm(gamma=0.92, vmin=0.0, vmax=vmax_mps),
            rasterized=True,
            alpha=0.92,
        )

    def draw_points(points_3d: np.ndarray, *, color: str, size: float, alpha: float) -> None:
        if points_3d.size == 0:
            return
        px = points_3d[:, 0]
        pz = points_3d[:, 2]
        mask = (
            (px >= PUBLICATION_X_RANGE_M[0])
            & (px <= PUBLICATION_X_RANGE_M[1])
            & (pz >= PUBLICATION_Z_RANGE_M[0])
            & (pz <= PUBLICATION_Z_RANGE_M[1])
        )
        if np.any(mask):
            ax.scatter(px[mask], pz[mask], s=size, c=color, linewidths=0.0, alpha=alpha)

    draw_points(surface_points, color="#e5d7ff", size=1.0, alpha=0.20)
    draw_points(marker_points, color="#d8c5ff", size=1.0, alpha=0.18)

    guide_color = "#f04b2f"
    ax.axvline(PUBLICATION_CENTER_X_M, color=guide_color, linewidth=1.2, alpha=0.92)
    tick_left = PUBLICATION_CENTER_X_M - 0.0030
    tick_right = PUBLICATION_CENTER_X_M + 0.0030
    text_x = PUBLICATION_CENTER_X_M + 0.0042
    for label, z_m in PUBLICATION_PROBES:
        ax.plot([tick_left, tick_right], [z_m, z_m], color=guide_color, linewidth=1.8, solid_capstyle="round")
        ax.text(
            text_x,
            z_m,
            label,
            color=guide_color,
            fontsize=7,
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax.text(
        PUBLICATION_X_RANGE_M[0] + 0.003,
        PUBLICATION_Z_RANGE_M[0] + 0.004,
        scalar_label,
        color="#e5d7ff",
        fontsize=6,
        ha="left",
        va="bottom",
        alpha=0.70,
    )
    return mesh


def _save_gif(
    png_paths: list[Path],
    gif_path: Path,
    *,
    duration_ms: int,
    interp_frames: int,
) -> int:
    from PIL import Image

    if not png_paths:
        raise RuntimeError("No PNG frames rendered")
    raw_images = [Image.open(path).convert("RGBA") for path in png_paths]
    frames: list[Image.Image] = []
    try:
        for index, image in enumerate(raw_images):
            frames.append(image.copy())
            if interp_frames <= 0 or index == len(raw_images) - 1:
                continue
            nxt = raw_images[index + 1]
            for substep in range(1, interp_frames + 1):
                alpha = substep / float(interp_frames + 1)
                frames.append(Image.blend(image, nxt, alpha))
        paletted = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=160) for frame in frames]
        try:
            paletted[0].save(
                gif_path,
                save_all=True,
                append_images=paletted[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
        finally:
            for frame in paletted:
                frame.close()
    finally:
        for image in raw_images:
            image.close()
        for frame in frames:
            frame.close()
    return len(frames)


def render(
    visit_dir: Path,
    output_dir: Path,
    *,
    min_age_s: float,
    duration_ms: int,
    scalar: str,
    vmin_mps: float,
    vmax_mps: float,
    interp_frames: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    records = _complete_records(visit_dir, min_age_s=min_age_s)
    if not records:
        raise RuntimeError(f"No complete VisIt records found in {visit_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    max_observed = 0.0
    p99_observed = 0.0
    values_for_stats: list[np.ndarray] = []

    scalar_label = "downward jet velocity max(0, -u_z)" if scalar == "downward" else "physical water speed |u|"
    for index, record in enumerate(records):
        step = int(record["step_index"])
        files = record["files"]
        fluid = pv.read(visit_dir / files["fluid_vti"])
        markers = pv.read(visit_dir / files["markers_vtp"])
        surface = pv.read(visit_dir / files["surface_vtp"])

        points, speed, velocity, physical = _slice_arrays(
            fluid,
            normal=(0.0, 1.0, 0.0),
            origin=(0.0, XZ_Y_M, 0.0),
        )
        values = _frame_scalar(speed=speed, velocity=velocity, scalar=scalar)
        finite = values[physical & np.isfinite(values)]
        if finite.size:
            values_for_stats.append(finite)
            max_observed = max(max_observed, float(np.nanmax(finite)))

        marker_points = _point_slab(markers, axis=1, value=XZ_Y_M, half_thickness_m=0.0018)
        surface_points = _point_slab(surface, axis=1, value=XZ_Y_M, half_thickness_m=0.0018)

        fig = plt.figure(figsize=(7.20, 5.40), dpi=100, facecolor="black")
        ax = fig.add_axes([0.235, 0.0, 0.530, 1.0])
        _draw_centerline_frame(
            ax,
            points=points,
            values=values,
            physical=physical,
            surface_points=surface_points,
            marker_points=marker_points,
            vmin_mps=vmin_mps,
            vmax_mps=vmax_mps,
            scalar_label=scalar_label,
        )
        frame_path = frame_dir / f"squid_jet_style_{step:06d}.png"
        fig.savefig(frame_path, facecolor="black", dpi=100)
        plt.close(fig)
        png_paths.append(frame_path)
        print(f"[render-jet] {index + 1}/{len(records)} {frame_path}", flush=True)

    if values_for_stats:
        merged = np.concatenate(values_for_stats)
        p99_observed = float(np.nanpercentile(merged, 99.0))

    gif_path = output_dir / (
        f"squid_jet_style_{scalar}_steps_{int(records[0]['step_index']):06d}_"
        f"{int(records[-1]['step_index']):06d}.gif"
    )
    gif_frame_count = _save_gif(
        png_paths,
        gif_path,
        duration_ms=duration_ms,
        interp_frames=interp_frames,
    )
    summary = {
        "visit_dir": str(visit_dir),
        "output_dir": str(output_dir),
        "raw_frame_count": len(png_paths),
        "gif_frame_count": gif_frame_count,
        "first_step": int(records[0]["step_index"]),
        "last_step": int(records[-1]["step_index"]),
        "gif": str(gif_path),
        "style": "jet_check_centerline",
        "scalar": scalar,
        "scalar_description": scalar_label,
        "vmin_mps": vmin_mps,
        "vmax_mps": vmax_mps,
        "observed_p99_mps": p99_observed,
        "observed_max_mps": max_observed,
        "visual_interpolation_frames_per_gap": interp_frames,
        "view": {
            "xz_y_m": XZ_Y_M,
            "centerline_x_m": PUBLICATION_CENTER_X_M,
            "x_range_m": list(PUBLICATION_X_RANGE_M),
            "z_range_m": list(PUBLICATION_Z_RANGE_M),
            "probe_z_m": {label: z_m for label, z_m in PUBLICATION_PROBES},
        },
    }
    (output_dir / "squid_jet_style_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render squid nozzle jet-style centerline GIF from VisIt VTK output.")
    parser.add_argument("visit_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-age-s", type=float, default=5.0)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--scalar", choices=("downward", "speed"), default="downward")
    parser.add_argument("--vmin-mps", type=float, default=0.005)
    parser.add_argument("--vmax-mps", type=float, default=0.12)
    parser.add_argument("--interp-frames", type=int, default=3)
    args = parser.parse_args()
    summary = render(
        args.visit_dir.resolve(),
        args.out.resolve(),
        min_age_s=args.min_age_s,
        duration_ms=args.duration_ms,
        scalar=args.scalar,
        vmin_mps=args.vmin_mps,
        vmax_mps=args.vmax_mps,
        interp_frames=args.interp_frames,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
