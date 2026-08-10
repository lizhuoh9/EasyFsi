"""Render periodic Turek-Hron ``flow_snapshots`` without running the solver.

The case stores the streamwise coordinate on solver z in the opposite
direction to the benchmark coordinate.  This module makes that mapping
explicit: every plotted field uses ``physical x = channel_length - solver z``.
All snapshots are loaded and validated before any PNG is written, and one
global velocity range is then used for the complete animation.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..ansys_vertical_flap_fsi.native_fine_rendering import build_gif


DEFAULT_CHANNEL_LENGTH_M = 2.5
VELOCITY_VMIN_MPS = 0.0
_STEP_FILE_PATTERN = re.compile(r"step_(\d{6})\.npz")
REQUIRED_FLOW_SNAPSHOT_FIELDS = (
    "y_centers_m",
    "z_centers_m",
    "velocity_magnitude_yz_mps",
    "velocity_y_yz_mps",
    "velocity_z_yz_mps",
    "pressure_yz_pa",
    "obstacle_mask_yz",
    "span_index",
    "grid_nodes",
    "cell_spacing_m",
    "beam_marker_current_xyz_m",
    "beam_marker_rest_xyz_m",
    "beam_marker_displacement_xyz_m",
    "time_s",
)


class FlowSnapshotContractError(ValueError):
    """Raised when periodic flow snapshots cannot form an honest animation."""


@dataclass(frozen=True)
class FlowSnapshotFrame:
    """Validated, physical-coordinate view of one solver snapshot."""

    source_path: Path
    step: int
    time_s: float
    physical_x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    velocity_magnitude_mps: np.ndarray
    velocity_y_mps: np.ndarray
    velocity_x_mps: np.ndarray
    pressure_pa: np.ndarray
    obstacle_mask: np.ndarray
    grid_nodes: np.ndarray
    span_index: int
    cell_spacing_m: np.ndarray
    beam_current_xyz_m: np.ndarray
    beam_rest_xyz_m: np.ndarray
    beam_displacement_xyz_m: np.ndarray
    beam_current_xy_m: np.ndarray
    beam_rest_xy_m: np.ndarray


@dataclass(frozen=True)
class TurekHronRenderResult:
    """Artifacts and shared scale produced by :func:`render_turek_hron_flow_gif`."""

    gif_path: Path
    frame_paths: tuple[Path, ...]
    source_paths: tuple[Path, ...]
    steps: tuple[int, ...]
    velocity_vmin_mps: float
    velocity_vmax_mps: float


def discover_flow_snapshot_paths(
    snapshot_dir: str | Path,
    *,
    expected_step_interval: int | None = None,
) -> list[Path]:
    """Return ordered snapshots after enforcing an unbroken periodic sequence.

    When the interval is omitted, a complete case directory is assumed and
    the first saved step is used as the periodic interval (for example,
    ``2, 4, 6`` or ``100, 200, 300``).  Pass ``expected_step_interval`` when
    rendering a suffix copied from a longer run.
    """

    directory = Path(snapshot_dir)
    if not directory.is_dir():
        raise FlowSnapshotContractError(
            f"flow snapshot directory does not exist: {directory}"
        )
    candidates = sorted(directory.glob("step_*.npz"))
    if not candidates:
        raise FlowSnapshotContractError(f"no step_XXXXXX.npz files found in {directory}")

    parsed: list[tuple[int, Path]] = []
    for path in candidates:
        match = _STEP_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise FlowSnapshotContractError(
                f"invalid flow snapshot filename (expected step_XXXXXX.npz): {path.name}"
            )
        step = int(match.group(1))
        if step <= 0:
            raise FlowSnapshotContractError(
                f"flow snapshot step must be positive: {path.name}"
            )
        parsed.append((step, path))
    parsed.sort(key=lambda item: item[0])

    interval = int(expected_step_interval) if expected_step_interval is not None else parsed[0][0]
    if interval <= 0:
        raise FlowSnapshotContractError("expected_step_interval must be positive")
    for (previous, _), (actual, path) in zip(parsed, parsed[1:]):
        expected = previous + interval
        if actual != expected:
            raise FlowSnapshotContractError(
                "non-contiguous flow snapshot sequence: "
                f"expected step {expected} after step {previous}, found step {actual} "
                f"({path.name})"
            )
    return [path for _, path in parsed]


def load_flow_snapshot(
    path: str | Path,
    *,
    channel_length_m: float = DEFAULT_CHANNEL_LENGTH_M,
) -> FlowSnapshotFrame:
    """Load one NPZ, validate its export schema, and map it to physical x."""

    source_path = Path(path)
    step_match = _STEP_FILE_PATTERN.fullmatch(source_path.name)
    if step_match is None:
        raise FlowSnapshotContractError(
            f"invalid flow snapshot filename: {source_path.name}"
        )
    length_m = float(channel_length_m)
    if not np.isfinite(length_m) or length_m <= 0.0:
        raise FlowSnapshotContractError("channel_length_m must be finite and positive")

    try:
        with np.load(source_path, allow_pickle=False) as payload:
            missing = [key for key in REQUIRED_FLOW_SNAPSHOT_FIELDS if key not in payload.files]
            if missing:
                raise FlowSnapshotContractError(
                    f"flow snapshot {source_path.name} is missing required field(s): "
                    + ", ".join(missing)
                )
            arrays = {key: np.asarray(payload[key]).copy() for key in REQUIRED_FLOW_SNAPSHOT_FIELDS}
    except FlowSnapshotContractError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise FlowSnapshotContractError(
            f"could not read flow snapshot {source_path}: {exc}"
        ) from exc

    y = _finite_vector(arrays["y_centers_m"], field="y_centers_m", path=source_path)
    z = _finite_vector(arrays["z_centers_m"], field="z_centers_m", path=source_path)
    if not np.all(np.diff(y) > 0.0) or not np.all(np.diff(z) > 0.0):
        raise FlowSnapshotContractError(
            f"y_centers_m and z_centers_m must be strictly increasing in {source_path.name}"
        )
    if z[0] < 0.0 or z[-1] > length_m:
        raise FlowSnapshotContractError(
            f"z_centers_m lies outside channel length {length_m} m in {source_path.name}"
        )

    field_shape = (y.size, z.size)
    field_arrays: dict[str, np.ndarray] = {}
    for key in (
        "velocity_magnitude_yz_mps",
        "velocity_y_yz_mps",
        "velocity_z_yz_mps",
        "pressure_yz_pa",
    ):
        values = np.asarray(arrays[key], dtype=np.float64)
        _require_shape(values, field_shape, field=key, path=source_path)
        _require_finite(values, field=key, path=source_path)
        field_arrays[key] = values
    if np.any(field_arrays["velocity_magnitude_yz_mps"] < 0.0):
        raise FlowSnapshotContractError(
            f"velocity_magnitude_yz_mps contains a negative value in {source_path.name}"
        )

    obstacle_raw = np.asarray(arrays["obstacle_mask_yz"])
    _require_shape(obstacle_raw, field_shape, field="obstacle_mask_yz", path=source_path)
    if not np.all(np.isin(obstacle_raw, (0, 1, False, True))):
        raise FlowSnapshotContractError(
            f"obstacle_mask_yz must contain only 0/1 values in {source_path.name}"
        )
    obstacle = obstacle_raw.astype(bool, copy=False)

    grid_nodes = np.asarray(arrays["grid_nodes"], dtype=np.int64)
    _require_shape(grid_nodes, (3,), field="grid_nodes", path=source_path)
    if np.any(grid_nodes <= 0) or tuple(grid_nodes[1:]) != field_shape:
        raise FlowSnapshotContractError(
            f"grid_nodes does not match the y-z field shape in {source_path.name}"
        )
    span_index = _integer_scalar(arrays["span_index"], field="span_index", path=source_path)
    if not 0 <= span_index < int(grid_nodes[0]):
        raise FlowSnapshotContractError(
            f"span_index is outside grid_nodes[0] in {source_path.name}"
        )
    spacing = np.asarray(arrays["cell_spacing_m"], dtype=np.float64)
    _require_shape(spacing, (2,), field="cell_spacing_m", path=source_path)
    _require_finite(spacing, field="cell_spacing_m", path=source_path)
    if np.any(spacing <= 0.0):
        raise FlowSnapshotContractError(
            f"cell_spacing_m must be positive in {source_path.name}"
        )

    current = _beam_array(
        arrays["beam_marker_current_xyz_m"],
        "beam_marker_current_xyz_m",
        source_path,
    )
    rest = _beam_array(arrays["beam_marker_rest_xyz_m"], "beam_marker_rest_xyz_m", source_path)
    displacement = _beam_array(
        arrays["beam_marker_displacement_xyz_m"],
        "beam_marker_displacement_xyz_m",
        source_path,
    )
    if rest.shape != current.shape or displacement.shape != current.shape:
        raise FlowSnapshotContractError(
            f"beam marker arrays must share one (N, 3) shape in {source_path.name}"
        )
    if not np.allclose(displacement, current - rest, rtol=0.0, atol=1.0e-12):
        raise FlowSnapshotContractError(
            f"beam_marker_displacement_xyz_m is inconsistent in {source_path.name}"
        )
    time_s = _float_scalar(arrays["time_s"], field="time_s", path=source_path)
    if time_s < 0.0:
        raise FlowSnapshotContractError(f"time_s must be non-negative in {source_path.name}")

    physical_x = length_m - z[::-1]
    current_xy = np.column_stack((length_m - current[:, 2], current[:, 1]))
    rest_xy = np.column_stack((length_m - rest[:, 2], rest[:, 1]))
    return FlowSnapshotFrame(
        source_path=source_path,
        step=int(step_match.group(1)),
        time_s=time_s,
        physical_x_centers_m=_readonly(physical_x),
        y_centers_m=_readonly(y),
        velocity_magnitude_mps=_readonly(field_arrays["velocity_magnitude_yz_mps"][:, ::-1]),
        velocity_y_mps=_readonly(field_arrays["velocity_y_yz_mps"][:, ::-1]),
        velocity_x_mps=_readonly(-field_arrays["velocity_z_yz_mps"][:, ::-1]),
        pressure_pa=_readonly(field_arrays["pressure_yz_pa"][:, ::-1]),
        obstacle_mask=_readonly(obstacle[:, ::-1]),
        grid_nodes=_readonly(grid_nodes),
        span_index=span_index,
        cell_spacing_m=_readonly(spacing),
        beam_current_xyz_m=_readonly(current),
        beam_rest_xyz_m=_readonly(rest),
        beam_displacement_xyz_m=_readonly(displacement),
        beam_current_xy_m=_readonly(current_xy),
        beam_rest_xy_m=_readonly(rest_xy),
    )


def render_turek_hron_flow_gif(
    snapshot_dir: str | Path,
    output_gif: str | Path,
    *,
    channel_length_m: float = DEFAULT_CHANNEL_LENGTH_M,
    expected_step_interval: int | None = None,
    velocity_vmax_mps: float | None = None,
    frame_dir: str | Path | None = None,
    duration_ms: int = 100,
    max_width_px: int = 1200,
    dpi: int = 140,
    show_marker_points: bool = False,
    show_rest_geometry: bool = False,
) -> TurekHronRenderResult:
    """Validate all periodic NPZs, render fixed-scale PNGs, and write a GIF."""

    source_paths = discover_flow_snapshot_paths(
        snapshot_dir,
        expected_step_interval=expected_step_interval,
    )
    frames = [
        load_flow_snapshot(path, channel_length_m=channel_length_m)
        for path in source_paths
    ]
    _validate_frame_sequence(frames)
    vmax = _velocity_vmax(frames, requested=velocity_vmax_mps)

    gif_path = Path(output_gif)
    frames_directory = (
        Path(frame_dir)
        if frame_dir is not None
        else gif_path.parent / f"{gif_path.stem}_frames"
    )
    frames_directory.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for frame in frames:
        output_path = frames_directory / f"velocity_step_{frame.step:06d}.png"
        _render_velocity_frame(
            frame,
            output_path,
            channel_length_m=float(channel_length_m),
            velocity_vmax_mps=vmax,
            dpi=int(dpi),
            show_marker_points=bool(show_marker_points),
            show_rest_geometry=bool(show_rest_geometry),
        )
        rendered.append(output_path)
    built_gif = build_gif(
        rendered,
        gif_path,
        duration_ms=int(duration_ms),
        max_width_px=int(max_width_px),
    )
    return TurekHronRenderResult(
        gif_path=built_gif,
        frame_paths=tuple(rendered),
        source_paths=tuple(source_paths),
        steps=tuple(frame.step for frame in frames),
        velocity_vmin_mps=VELOCITY_VMIN_MPS,
        velocity_vmax_mps=vmax,
    )


def _validate_frame_sequence(frames: Sequence[FlowSnapshotFrame]) -> None:
    if not frames:
        raise FlowSnapshotContractError("no flow snapshots were loaded")
    first = frames[0]
    for previous, frame in zip(frames, frames[1:]):
        if frame.time_s <= previous.time_s:
            raise FlowSnapshotContractError(
                f"flow snapshot time_s is not strictly increasing at {frame.source_path.name}"
            )
        for field, actual, expected in (
            ("physical x grid", frame.physical_x_centers_m, first.physical_x_centers_m),
            ("y grid", frame.y_centers_m, first.y_centers_m),
            ("grid_nodes", frame.grid_nodes, first.grid_nodes),
            ("cell_spacing_m", frame.cell_spacing_m, first.cell_spacing_m),
            ("beam rest geometry", frame.beam_rest_xyz_m, first.beam_rest_xyz_m),
        ):
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=1.0e-12
            ):
                raise FlowSnapshotContractError(
                    f"{field} changes across snapshots at {frame.source_path.name}"
                )


def _velocity_vmax(
    frames: Sequence[FlowSnapshotFrame],
    *,
    requested: float | None,
) -> float:
    if requested is not None:
        vmax = float(requested)
        if not np.isfinite(vmax) or vmax <= VELOCITY_VMIN_MPS:
            raise FlowSnapshotContractError("velocity_vmax_mps must be finite and positive")
        return vmax
    observed = max(
        (
            float(np.max(frame.velocity_magnitude_mps[~frame.obstacle_mask]))
            if np.any(~frame.obstacle_mask)
            else 0.0
        )
        for frame in frames
    )
    return observed if observed > 0.0 else 1.0


def _render_velocity_frame(
    frame: FlowSnapshotFrame,
    output_path: Path,
    *,
    channel_length_m: float,
    velocity_vmax_mps: float,
    dpi: int,
    show_marker_points: bool,
    show_rest_geometry: bool,
) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(12.0, 2.8), constrained_layout=True)
    cmap = _velocity_colormap()
    velocity = np.ma.masked_where(
        frame.obstacle_mask,
        frame.velocity_magnitude_mps,
    )
    plot = axis.pcolormesh(
        frame.physical_x_centers_m,
        frame.y_centers_m,
        velocity,
        shading="auto",
        cmap=cmap,
        vmin=VELOCITY_VMIN_MPS,
        vmax=velocity_vmax_mps,
    )
    _overlay_deformed_beam(
        axis,
        current_xyz_m=frame.beam_current_xyz_m,
        rest_xyz_m=frame.beam_rest_xyz_m,
        channel_length_m=channel_length_m,
        show_marker_points=show_marker_points,
        show_rest_geometry=show_rest_geometry,
    )
    axis.set_xlim(0.0, channel_length_m)
    axis.set_ylim(_center_limits(frame.y_centers_m))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Turek-Hron physical x (m)")
    axis.set_ylabel("physical y (m)")
    axis.set_title(
        "Turek-Hron velocity magnitude, "
        f"step {frame.step:06d}, t={frame.time_s:.6f} s"
    )
    fig.colorbar(plot, ax=axis, label="speed (m/s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _overlay_deformed_beam(
    axis: Any,
    *,
    current_xyz_m: np.ndarray,
    rest_xyz_m: np.ndarray,
    channel_length_m: float,
    show_marker_points: bool = False,
    show_rest_geometry: bool = False,
) -> None:
    """Draw the deformed beam as a black face; point clouds stay opt-in."""

    current = np.asarray(current_xyz_m, dtype=np.float64)
    rest = np.asarray(rest_xyz_m, dtype=np.float64)
    polygon = _beam_polygon(current, rest, channel_length_m=float(channel_length_m))
    if polygon.shape[0] >= 3:
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label="deformed beam",
        )
    else:
        axis.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="black",
            linewidth=1.2,
            zorder=5,
            label="deformed beam",
        )
    if show_rest_geometry:
        rest_polygon = _beam_polygon(rest, rest, channel_length_m=float(channel_length_m))
        closed = np.vstack((rest_polygon, rest_polygon[:1]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color="0.35",
            linestyle="--",
            linewidth=0.7,
            zorder=4,
            label="rest beam",
        )
    if show_marker_points:
        axis.scatter(
            float(channel_length_m) - current[:, 2],
            current[:, 1],
            s=3.0,
            color="white",
            edgecolors="black",
            linewidths=0.25,
            zorder=6,
            label="beam particles",
        )


def _beam_polygon(
    current_xyz_m: np.ndarray,
    rest_xyz_m: np.ndarray,
    *,
    channel_length_m: float,
) -> np.ndarray:
    rest_z = np.asarray(rest_xyz_m[:, 2], dtype=np.float64)
    current_x = channel_length_m - np.asarray(current_xyz_m[:, 2], dtype=np.float64)
    current_y = np.asarray(current_xyz_m[:, 1], dtype=np.float64)
    section_z, section_ids = np.unique(rest_z, return_inverse=True)
    sections: list[tuple[float, float, float, float]] = []
    for section_index, solver_z in enumerate(section_z):
        selected = section_ids == section_index
        x_values = current_x[selected]
        y_values = current_y[selected]
        sections.append(
            (
                channel_length_m - float(solver_z),
                float(np.mean(x_values)),
                float(np.min(y_values)),
                float(np.max(y_values)),
            )
        )
    sections.sort(key=lambda item: item[0])
    lower = np.asarray([[item[1], item[2]] for item in sections], dtype=np.float64)
    upper = np.asarray([[item[1], item[3]] for item in reversed(sections)], dtype=np.float64)
    return np.vstack((lower, upper))


def _center_limits(centers: np.ndarray) -> tuple[float, float]:
    if centers.size == 1:
        return float(centers[0] - 0.5), float(centers[0] + 0.5)
    lower = float(centers[0] - 0.5 * (centers[1] - centers[0]))
    upper = float(centers[-1] + 0.5 * (centers[-1] - centers[-2]))
    return lower, upper


def _velocity_colormap() -> Any:
    """Return the velocity map with obstacle/masked cells fixed to black."""

    cmap = _pyplot().get_cmap("turbo").copy()
    cmap.set_bad("black")
    return cmap


def _finite_vector(values: np.ndarray, *, field: str, path: Path) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise FlowSnapshotContractError(
            f"{field} must be a non-empty 1-D array in {path.name}"
        )
    _require_finite(result, field=field, path=path)
    return result


def _beam_array(values: np.ndarray, field: str, path: Path) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise FlowSnapshotContractError(
            f"{field} must have non-empty shape (N, 3) in {path.name}"
        )
    _require_finite(result, field=field, path=path)
    return result


def _require_shape(
    values: np.ndarray,
    expected: tuple[int, ...],
    *,
    field: str,
    path: Path,
) -> None:
    if values.shape != expected:
        raise FlowSnapshotContractError(
            f"{field} has shape {values.shape}, expected {expected}, in {path.name}"
        )


def _require_finite(values: np.ndarray, *, field: str, path: Path) -> None:
    if not np.all(np.isfinite(values)):
        raise FlowSnapshotContractError(
            f"{field} contains a non-finite value in {path.name}"
        )


def _float_scalar(values: np.ndarray, *, field: str, path: Path) -> float:
    array = np.asarray(values)
    if array.size != 1:
        raise FlowSnapshotContractError(f"{field} must be scalar in {path.name}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise FlowSnapshotContractError(f"{field} must be finite in {path.name}")
    return result


def _integer_scalar(values: np.ndarray, *, field: str, path: Path) -> int:
    array = np.asarray(values)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise FlowSnapshotContractError(f"{field} must be an integer scalar in {path.name}")
    return int(array.reshape(-1)[0])


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for rendering an existing ``flow_snapshots`` folder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("output_gif", type=Path)
    parser.add_argument("--channel-length-m", type=float, default=DEFAULT_CHANNEL_LENGTH_M)
    parser.add_argument("--step-interval", type=int, default=None)
    parser.add_argument("--velocity-vmax-mps", type=float, default=None)
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument("--max-width-px", type=int, default=1200)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--show-marker-points", action="store_true")
    parser.add_argument("--show-rest-geometry", action="store_true")
    args = parser.parse_args(argv)
    result = render_turek_hron_flow_gif(
        args.snapshot_dir,
        args.output_gif,
        channel_length_m=args.channel_length_m,
        expected_step_interval=args.step_interval,
        velocity_vmax_mps=args.velocity_vmax_mps,
        duration_ms=args.duration_ms,
        max_width_px=args.max_width_px,
        dpi=args.dpi,
        show_marker_points=args.show_marker_points,
        show_rest_geometry=args.show_rest_geometry,
    )
    print(
        f"rendered {len(result.frame_paths)} frames to {result.gif_path} "
        f"with fixed speed scale [{result.velocity_vmin_mps}, "
        f"{result.velocity_vmax_mps}] m/s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the function API
    raise SystemExit(main())
