"""Rendering helpers for native fine-grid comparison artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .official_fluent_parity import load_solver_npz


VELOCITY_VMIN_MPS = 0.0
GEOMETRY_ALIAS_ATOL = 1.0e-10

SOLID_SCALAR_FIELDS = (
    "solid_x_m",
    "solid_y_m",
    "solid_rest_x_m",
    "solid_rest_y_m",
    "solid_vx_mps",
    "solid_vy_mps",
)
SOLID_VECTOR_FIELDS = (
    "solid_position_m",
    "solid_velocity_mps",
    "solid_rest_position_m",
)
SOLID_MASK_FIELDS = ("solid_fixed_mask", "solid_tip_mask")
MARKER_SCALAR_FIELDS = ("marker_x_m", "marker_y_m", "marker_area_m2")
MARKER_VECTOR_FIELDS = (
    "marker_position_m",
    "marker_velocity_mps",
    "marker_normal",
)
MARKER_ID_FIELDS = ("marker_region_id",)
DEFORMED_GEOMETRY_FIELDS = (
    *SOLID_SCALAR_FIELDS,
    *SOLID_VECTOR_FIELDS,
    *SOLID_MASK_FIELDS,
    *MARKER_SCALAR_FIELDS,
    *MARKER_VECTOR_FIELDS,
    *MARKER_ID_FIELDS,
)


class DeformedGeometryContractError(ValueError):
    """Raised when a solver frame lacks trustworthy moving-geometry data."""


def load_solver_frame_with_geometry(path: str | Path) -> dict[str, np.ndarray]:
    """Load parity fields together with the strict deformed-geometry payload."""

    fields = load_solver_npz(path)
    fields.update(_load_deformed_geometry(path))
    return fields


def validate_deformed_geometry_frames(
    frame_paths: Sequence[Path],
    *,
    expected_reverse_streamwise_axis: bool | None = None,
    expected_streamwise_length_m: float | None = None,
    expected_streamwise_velocity_sign: float | None = None,
) -> dict[str, Any]:
    """Validate all moving-solid/marker payloads before creating outputs."""

    if not frame_paths:
        raise DeformedGeometryContractError("no solver frames were provided")
    solid_counts: set[int] = set()
    marker_counts: set[int] = set()
    peak_displacement_m = 0.0
    peak_displacement_step = 0
    per_frame_max_displacement_m: list[float] = []
    first_rest_position: np.ndarray | None = None
    first_fixed_mask: np.ndarray | None = None
    first_tip_mask: np.ndarray | None = None
    first_marker_region_id: np.ndarray | None = None
    streamwise_mapping: dict[str, Any] | None = None
    for step, path in enumerate(frame_paths, start=1):
        geometry = _load_deformed_geometry(path)
        if streamwise_mapping is None:
            streamwise_mapping = _infer_streamwise_mapping(geometry, path=path)
            _validate_expected_streamwise_mapping(
                streamwise_mapping,
                expected_reverse_streamwise_axis=expected_reverse_streamwise_axis,
                expected_streamwise_length_m=expected_streamwise_length_m,
                expected_streamwise_velocity_sign=expected_streamwise_velocity_sign,
                path=path,
            )
        _validate_scalar_geometry_aliases(
            geometry,
            mapping=streamwise_mapping,
            path=path,
        )
        solid_counts.add(int(geometry["solid_x_m"].size))
        marker_counts.add(int(geometry["marker_x_m"].size))
        rest_position = geometry["solid_rest_position_m"]
        fixed_mask = geometry["solid_fixed_mask"]
        tip_mask = geometry["solid_tip_mask"]
        marker_region_id = geometry["marker_region_id"]
        if first_rest_position is None:
            first_rest_position = rest_position.copy()
            first_fixed_mask = fixed_mask.copy()
            first_tip_mask = tip_mask.copy()
            first_marker_region_id = marker_region_id.copy()
        else:
            if rest_position.shape != first_rest_position.shape:
                raise DeformedGeometryContractError(
                    f"solid point count changes at frame {path.name}: "
                    f"{rest_position.shape[0]} vs {first_rest_position.shape[0]}"
                )
            if not np.allclose(
                rest_position,
                first_rest_position,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise DeformedGeometryContractError(
                    f"solid rest positions change at frame {path.name}"
                )
            if not np.array_equal(fixed_mask, first_fixed_mask):
                raise DeformedGeometryContractError(
                    f"solid fixed mask changes at frame {path.name}"
                )
            if not np.array_equal(tip_mask, first_tip_mask):
                raise DeformedGeometryContractError(
                    f"solid tip mask changes at frame {path.name}"
                )
            if not np.array_equal(marker_region_id, first_marker_region_id):
                raise DeformedGeometryContractError(
                    f"marker region ids change at frame {path.name}"
                )
        displacement = np.linalg.norm(
            geometry["solid_position_m"] - rest_position,
            axis=1,
        )
        frame_max = float(np.max(displacement))
        per_frame_max_displacement_m.append(frame_max)
        if frame_max > peak_displacement_m:
            peak_displacement_m = frame_max
            peak_displacement_step = step
    if len(solid_counts) != 1:
        raise DeformedGeometryContractError(
            f"solid point count changes across frames: {sorted(solid_counts)}"
        )
    if len(marker_counts) != 1:
        raise DeformedGeometryContractError(
            f"marker point count changes across frames: {sorted(marker_counts)}"
        )
    if peak_displacement_m <= 1.0e-18:
        raise DeformedGeometryContractError(
            "all solver frames contain zero solid deformation; a true moving-solid "
            "overlay cannot be claimed"
        )
    assert first_fixed_mask is not None
    assert first_tip_mask is not None
    assert streamwise_mapping is not None
    return {
        "schema": "our_solver_step_deformed_geometry_v1",
        "frame_count": len(frame_paths),
        "solid_point_count": next(iter(solid_counts)),
        "marker_point_count": next(iter(marker_counts)),
        "fixed_solid_point_count": int(np.count_nonzero(first_fixed_mask)),
        "tip_solid_point_count": int(np.count_nonzero(first_tip_mask)),
        "peak_particle_displacement_m": peak_displacement_m,
        "peak_particle_displacement_step": peak_displacement_step,
        "final_particle_displacement_m": per_frame_max_displacement_m[-1],
        "observed_nonzero_deformation": True,
        "true_deformed_geometry_overlay": True,
        "scalar_alias_cross_check": "passed",
        "streamwise_mapping": streamwise_mapping,
        "overlay_layers": ["solid_deformed"],
        "hibm_markers_rendered": False,
        "rest_positions_rendered": False,
        "required_fields": list(DEFORMED_GEOMETRY_FIELDS),
    }


def _validate_expected_streamwise_mapping(
    mapping: Mapping[str, Any],
    *,
    expected_reverse_streamwise_axis: bool | None,
    expected_streamwise_length_m: float | None,
    expected_streamwise_velocity_sign: float | None,
    path: Path,
) -> None:
    if (
        expected_reverse_streamwise_axis is not None
        and mapping["reverse_streamwise_axis"] is not expected_reverse_streamwise_axis
    ):
        raise DeformedGeometryContractError(
            f"frame aliases disagree with runner reverse_streamwise_axis in {path.name}"
        )
    if expected_streamwise_velocity_sign is not None and not np.isclose(
        float(mapping["streamwise_velocity_sign"]),
        float(expected_streamwise_velocity_sign),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise DeformedGeometryContractError(
            f"frame aliases disagree with runner streamwise_velocity_sign in {path.name}"
        )
    if (
        bool(mapping["reverse_streamwise_axis"])
        and expected_streamwise_length_m is not None
        and not np.isclose(
            float(mapping["streamwise_length_m"]),
            float(expected_streamwise_length_m),
            rtol=0.0,
            atol=1.0e-8,
        )
    ):
        raise DeformedGeometryContractError(
            f"frame aliases disagree with runner streamwise_length_m in {path.name}"
        )


def _infer_streamwise_mapping(
    geometry: Mapping[str, np.ndarray],
    *,
    path: Path,
) -> dict[str, Any]:
    rest_x = geometry["solid_rest_x_m"]
    rest_z = geometry["solid_rest_position_m"][:, 2]
    if np.allclose(rest_x, rest_z, rtol=0.0, atol=GEOMETRY_ALIAS_ATOL):
        reverse_streamwise_axis = False
        streamwise_length_m = 0.0
    else:
        offsets = rest_x + rest_z
        streamwise_length_m = float(np.mean(offsets))
        if streamwise_length_m <= 0.0 or not np.allclose(
            offsets,
            streamwise_length_m,
            rtol=0.0,
            atol=GEOMETRY_ALIAS_ATOL,
        ):
            raise DeformedGeometryContractError(
                "scalar alias solid_rest_x_m does not follow the runner streamwise "
                f"mapping in {path.name}"
            )
        reverse_streamwise_axis = True

    velocity_z = geometry["solid_velocity_mps"][:, 2]
    velocity_x = geometry["solid_vx_mps"]
    positive_matches = np.allclose(
        velocity_x, velocity_z, rtol=0.0, atol=GEOMETRY_ALIAS_ATOL
    )
    negative_matches = np.allclose(
        velocity_x, -velocity_z, rtol=0.0, atol=GEOMETRY_ALIAS_ATOL
    )
    if positive_matches and not negative_matches:
        streamwise_velocity_sign = 1.0
    elif negative_matches and not positive_matches:
        streamwise_velocity_sign = -1.0
    elif positive_matches and negative_matches:
        streamwise_velocity_sign = -1.0 if reverse_streamwise_axis else 1.0
    else:
        raise DeformedGeometryContractError(
            "scalar alias solid_vx_mps does not follow the runner velocity mapping "
            f"in {path.name}"
        )
    return {
        "reverse_streamwise_axis": reverse_streamwise_axis,
        "streamwise_length_m": streamwise_length_m,
        "streamwise_velocity_sign": streamwise_velocity_sign,
    }


def _validate_scalar_geometry_aliases(
    geometry: Mapping[str, np.ndarray],
    *,
    mapping: Mapping[str, Any],
    path: Path,
) -> None:
    reverse = bool(mapping["reverse_streamwise_axis"])
    length_m = float(mapping["streamwise_length_m"])
    velocity_sign = float(mapping["streamwise_velocity_sign"])

    def streamwise(values: np.ndarray) -> np.ndarray:
        return length_m - values if reverse else values

    expected_by_alias = {
        "solid_x_m": streamwise(geometry["solid_position_m"][:, 2]),
        "solid_y_m": geometry["solid_position_m"][:, 1],
        "solid_rest_x_m": streamwise(geometry["solid_rest_position_m"][:, 2]),
        "solid_rest_y_m": geometry["solid_rest_position_m"][:, 1],
        "solid_vx_mps": velocity_sign * geometry["solid_velocity_mps"][:, 2],
        "solid_vy_mps": geometry["solid_velocity_mps"][:, 1],
        "marker_x_m": streamwise(geometry["marker_position_m"][:, 2]),
        "marker_y_m": geometry["marker_position_m"][:, 1],
    }
    for alias, expected in expected_by_alias.items():
        actual = geometry[alias]
        if not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=GEOMETRY_ALIAS_ATOL,
        ):
            max_error = float(np.max(np.abs(actual - expected)))
            raise DeformedGeometryContractError(
                f"scalar alias {alias} is stale or inconsistent with its original "
                f"3-D array in {path.name}; max_abs_error={max_error}"
            )


def _load_deformed_geometry(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the non-pickled moving-geometry frame contract."""

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [field for field in DEFORMED_GEOMETRY_FIELDS if field not in data.files]
            if missing:
                raise DeformedGeometryContractError(
                    "deformed solid geometry is missing required field(s) in "
                    f"{path.name}: {', '.join(missing)}"
                )
            marker_ids = np.asarray(data["marker_region_id"])
            if not np.issubdtype(marker_ids.dtype, np.integer):
                raise DeformedGeometryContractError(
                    f"marker_region_id must be integral in {path.name}"
                )
            geometry = {
                field: np.asarray(data[field], dtype=np.float64)
                for field in (*SOLID_SCALAR_FIELDS, *SOLID_VECTOR_FIELDS)
            }
            geometry.update(
                {
                    field: np.asarray(data[field], dtype=bool)
                    for field in SOLID_MASK_FIELDS
                }
            )
            geometry.update(
                {
                    field: np.asarray(data[field], dtype=np.float64)
                    for field in (*MARKER_SCALAR_FIELDS, *MARKER_VECTOR_FIELDS)
                }
            )
            geometry["marker_region_id"] = marker_ids.astype(np.int64, copy=False)
    except DeformedGeometryContractError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise DeformedGeometryContractError(
            f"could not read deformed solid geometry from {path}: {exc}"
        ) from exc

    _validate_deformed_geometry_payload(geometry, path=path)
    return geometry


def _validate_deformed_geometry_payload(
    geometry: Mapping[str, np.ndarray],
    *,
    path: Path,
) -> None:
    solid_count = int(geometry["solid_x_m"].size)
    marker_count = int(geometry["marker_x_m"].size)
    if solid_count <= 0:
        raise DeformedGeometryContractError(
            f"deformed solid geometry is empty in {path.name}"
        )
    if marker_count <= 0:
        raise DeformedGeometryContractError(
            f"deformed marker geometry is empty in {path.name}"
        )

    for field in SOLID_SCALAR_FIELDS:
        _require_shape(geometry[field], (solid_count,), field=field, path=path)
    for field in SOLID_VECTOR_FIELDS:
        _require_shape(geometry[field], (solid_count, 3), field=field, path=path)
    for field in SOLID_MASK_FIELDS:
        _require_shape(geometry[field], (solid_count,), field=field, path=path)
    for field in MARKER_SCALAR_FIELDS:
        _require_shape(geometry[field], (marker_count,), field=field, path=path)
    for field in MARKER_VECTOR_FIELDS:
        _require_shape(geometry[field], (marker_count, 3), field=field, path=path)
    for field in MARKER_ID_FIELDS:
        _require_shape(geometry[field], (marker_count,), field=field, path=path)

    for field in (
        *SOLID_SCALAR_FIELDS,
        *SOLID_VECTOR_FIELDS,
        *MARKER_SCALAR_FIELDS,
        *MARKER_VECTOR_FIELDS,
    ):
        if not np.all(np.isfinite(geometry[field])):
            raise DeformedGeometryContractError(
                f"deformed geometry field {field} contains non-finite values in {path.name}"
            )
    if not np.any(geometry["solid_fixed_mask"]):
        raise DeformedGeometryContractError(
            f"solid_fixed_mask has no anchored particles in {path.name}"
        )
    if not np.any(geometry["solid_tip_mask"]):
        raise DeformedGeometryContractError(
            f"solid_tip_mask has no tip particles in {path.name}"
        )
    if np.any(geometry["marker_area_m2"] <= 0.0):
        raise DeformedGeometryContractError(
            f"marker_area_m2 must be strictly positive in {path.name}"
        )
    marker_normal_norm = np.linalg.norm(geometry["marker_normal"], axis=1)
    if np.any(marker_normal_norm <= 1.0e-12):
        raise DeformedGeometryContractError(
            f"marker_normal contains a zero vector in {path.name}"
        )


def _require_shape(
    values: np.ndarray,
    expected: tuple[int, ...],
    *,
    field: str,
    path: Path,
) -> None:
    if values.shape != expected:
        raise DeformedGeometryContractError(
            f"deformed geometry field {field} has shape {values.shape}, "
            f"expected {expected}, in {path.name}"
        )


def render_solver_velocity_frames(
    frame_paths: Sequence[Path],
    output_dir: str | Path,
    *,
    dt_s: float,
    velocity_vmax_mps: float,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for step, frame_path in enumerate(frame_paths, start=1):
        fields = load_solver_frame_with_geometry(frame_path)
        _validate_solver_fields(fields)
        target = output_dir / f"our_velocity_step_{step:04d}.png"
        _render_solver_velocity_frame(
            fields,
            target,
            step=step,
            time_s=step * dt_s,
            velocity_vmax_mps=velocity_vmax_mps,
        )
        rendered.append(target)
    return rendered


def build_gif(
    frame_paths: Sequence[Path],
    output_path: str | Path,
    *,
    duration_ms: int,
    max_width_px: int,
) -> Path:
    if not frame_paths:
        raise ValueError("cannot build GIF without velocity frames")
    duration = int(duration_ms)
    max_width = int(max_width_px)
    if duration <= 0 or max_width <= 0:
        raise ValueError("GIF duration and maximum width must be positive")
    try:
        from PIL import GifImagePlugin, Image
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("Pillow is required to build the velocity GIF") from exc

    expected_size: tuple[int, int] | None = None
    for path in frame_paths:
        with Image.open(path) as source:
            size = _gif_frame_size(source.size, max_width=max_width)
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError("velocity GIF frame dimensions are inconsistent")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        for index, path in enumerate(frame_paths):
            with Image.open(path) as source:
                rgb = source.convert("RGB")
            try:
                if rgb.size != expected_size:
                    resized = rgb.resize(expected_size, Image.Resampling.LANCZOS)
                    rgb.close()
                    rgb = resized
                paletted = rgb.convert("P", palette=Image.Palette.ADAPTIVE)
            finally:
                rgb.close()
            try:
                if index == 0:
                    header, _ = GifImagePlugin.getheader(
                        paletted,
                        info={"loop": 0},
                    )
                    for block in header:
                        stream.write(block)
                for block in GifImagePlugin.getdata(
                    paletted,
                    duration=duration,
                    disposal=2,
                    include_color_table=index > 0,
                ):
                    stream.write(block)
            finally:
                paletted.close()
        stream.write(b";")
    return output_path


def _gif_frame_size(size: tuple[int, int], *, max_width: int) -> tuple[int, int]:
    width, height = size
    if width <= max_width:
        return width, height
    return max_width, max(1, round(height * max_width / width))


def render_final_field_comparison(
    solver_fields: Mapping[str, np.ndarray],
    fluent_fields: Mapping[str, np.ndarray],
    sampled: Mapping[str, np.ndarray],
    velocity_path: str | Path,
    pressure_path: str | Path,
    *,
    velocity_vmax_mps: float,
) -> None:
    plt = _pyplot()
    valid = np.asarray(sampled["valid"], dtype=bool)
    difference_speed = np.abs(
        np.asarray(sampled["speed"])[valid] - np.asarray(fluent_fields["speed"])[valid]
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    solver_display_mask = _display_fluid_mask(solver_fields)
    velocity_cmap = plt.get_cmap("turbo").copy()
    velocity_cmap.set_bad("black")
    solver_plot = axes[0].pcolormesh(
        solver_fields["s"],
        solver_fields["y"],
        np.ma.masked_where(~solver_display_mask, solver_fields["speed"]),
        shading="auto",
        cmap=velocity_cmap,
        vmin=VELOCITY_VMIN_MPS,
        vmax=velocity_vmax_mps,
    )
    axes[0].set_title("Our solver, final speed")
    _overlay_deformed_solid(axes[0], solver_fields)
    axes[1].scatter(
        fluent_fields["x"],
        fluent_fields["y"],
        c=fluent_fields["speed"],
        s=4,
        linewidths=0,
        cmap="turbo",
        vmin=VELOCITY_VMIN_MPS,
        vmax=velocity_vmax_mps,
        rasterized=True,
    )
    axes[1].set_title("Native Fluent, final speed")
    error_plot = axes[2].scatter(
        np.asarray(fluent_fields["x"])[valid],
        np.asarray(fluent_fields["y"])[valid],
        c=difference_speed,
        s=4,
        linewidths=0,
        cmap="magma",
        rasterized=True,
    )
    axes[2].set_title("Absolute speed difference")
    for axis in axes:
        axis.set_xlabel("streamwise x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
    fig.colorbar(solver_plot, ax=axes[:2], label="speed (m/s)", shrink=0.9)
    fig.colorbar(error_plot, ax=axes[2], label="absolute error (m/s)", shrink=0.9)
    Path(velocity_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(velocity_path, dpi=180)
    plt.close(fig)

    solver_pressure = np.asarray(solver_fields["p"])[solver_fields["fluid_mask"]]
    fluent_pressure = np.asarray(fluent_fields["p"])
    common_min = float(min(np.min(solver_pressure), np.min(fluent_pressure)))
    common_max = float(max(np.max(solver_pressure), np.max(fluent_pressure)))
    pressure_error = np.asarray(sampled["p"])[valid] - fluent_pressure[valid]
    error_limit = float(np.max(np.abs(pressure_error))) if pressure_error.size else 1.0
    error_limit = max(error_limit, np.finfo(np.float64).eps)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    pressure_cmap = plt.get_cmap("coolwarm").copy()
    pressure_cmap.set_bad("black")
    solver_plot = axes[0].pcolormesh(
        solver_fields["s"],
        solver_fields["y"],
        np.ma.masked_where(~solver_display_mask, solver_fields["p"]),
        shading="auto",
        cmap=pressure_cmap,
        vmin=common_min,
        vmax=common_max,
    )
    axes[0].set_title("Our solver, final pressure")
    _overlay_deformed_solid(axes[0], solver_fields)
    axes[1].scatter(
        fluent_fields["x"],
        fluent_fields["y"],
        c=fluent_pressure,
        s=4,
        linewidths=0,
        cmap="coolwarm",
        vmin=common_min,
        vmax=common_max,
        rasterized=True,
    )
    axes[1].set_title("Native Fluent, final pressure")
    error_plot = axes[2].scatter(
        np.asarray(fluent_fields["x"])[valid],
        np.asarray(fluent_fields["y"])[valid],
        c=pressure_error,
        s=4,
        linewidths=0,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        rasterized=True,
    )
    axes[2].set_title("Pressure difference (our - Fluent)")
    for axis in axes:
        axis.set_xlabel("streamwise x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
    fig.colorbar(solver_plot, ax=axes[:2], label="pressure (Pa)", shrink=0.9)
    fig.colorbar(error_plot, ax=axes[2], label="signed error (Pa)", shrink=0.9)
    Path(pressure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pressure_path, dpi=180)
    plt.close(fig)


def render_displacement_comparison(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    plt = _pyplot()
    time_s = np.asarray([float(row["time_s"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(
        time_s,
        [row["our_tip_mean_vector_norm_m"] for row in rows],
        label="our top-row mean-vector norm",
    )
    axes[0].plot(
        time_s,
        [row["fluent_tip_mean_vector_norm_m"] for row in rows],
        label="Fluent 4-node mean-vector norm",
    )
    axes[0].set_title("Tip displacement diagnostic")
    axes[1].plot(
        time_s,
        [row["our_solid_max_displacement_m"] for row in rows],
        label="our solid maximum",
    )
    axes[1].plot(
        time_s,
        [row["fluent_solid_max_displacement_m"] for row in rows],
        label="Fluent solid maximum",
    )
    axes[1].set_title("Whole-solid maximum displacement")
    for axis in axes:
        axis.set_xlabel("time (s)")
        axis.set_ylabel("displacement (m)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _render_solver_velocity_frame(
    fields: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    step: int,
    time_s: float,
    velocity_vmax_mps: float,
) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    display_fluid_mask = _display_fluid_mask(fields)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    plot = axis.pcolormesh(
        fields["s"],
        fields["y"],
        np.ma.masked_where(~display_fluid_mask, fields["speed"]),
        shading="auto",
        cmap=cmap,
        vmin=VELOCITY_VMIN_MPS,
        vmax=velocity_vmax_mps,
    )
    _overlay_deformed_solid(axis, fields)
    axis.set_title(
        f"Our solver velocity magnitude, step {step:04d}, t={time_s:.6f} s"
    )
    axis.set_xlabel("streamwise x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    fig.colorbar(plot, ax=axis, label="speed (m/s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _validate_solver_fields(fields: Mapping[str, np.ndarray]) -> None:
    shape = np.asarray(fields["speed"]).shape
    fluid_mask = np.asarray(fields["fluid_mask"], dtype=bool)
    display_fluid_mask = _display_fluid_mask(fields)
    if len(shape) != 2 or fluid_mask.shape != shape:
        raise ValueError("solver field arrays must share a 2-D grid")
    if display_fluid_mask.shape != shape:
        raise ValueError("solver display mask must share the 2-D field grid")
    for key in ("u", "v", "p", "speed"):
        values = np.asarray(fields[key])
        if values.shape != shape or not np.all(
            np.isfinite(values[display_fluid_mask])
        ):
            raise ValueError(f"invalid solver field array: {key}")


def _display_fluid_mask(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the plot-only fluid mask without changing parity metrics.

    New artifacts carry the exact mask derived from ``obstacle == 0``.  For
    older artifacts, a boundary surrogate was excluded only for comparison,
    so adding it back reconstructs the display domain without opening true
    obstacle cells.
    """
    strict = np.asarray(fields["fluid_mask"], dtype=bool)
    if "display_fluid_mask" in fields:
        return np.asarray(fields["display_fluid_mask"], dtype=bool)
    surrogate = np.asarray(
        fields.get("boundary_surrogate_mask", np.zeros_like(strict)),
        dtype=bool,
    )
    return strict | surrogate


def _overlay_deformed_solid(
    axis: Any,
    fields: Mapping[str, np.ndarray],
    *,
    show_hibm_markers: bool = False,
) -> None:
    solid_x = np.asarray(fields["solid_x_m"], dtype=np.float64)
    solid_y = np.asarray(fields["solid_y_m"], dtype=np.float64)
    axis.scatter(
        solid_x,
        solid_y,
        s=5,
        marker=".",
        color="black",
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        label="deformed solid",
    )
    if bool(show_hibm_markers):
        marker_x = np.asarray(fields["marker_x_m"], dtype=np.float64)
        marker_y = np.asarray(fields["marker_y_m"], dtype=np.float64)
        axis.scatter(
            marker_x,
            marker_y,
            s=9,
            marker="o",
            facecolors="none",
            edgecolors="#00ffff",
            linewidths=0.45,
            alpha=0.9,
            rasterized=True,
            label="deformed HIBM markers",
        )
        axis.legend(loc="upper right", fontsize=6, framealpha=0.75)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
