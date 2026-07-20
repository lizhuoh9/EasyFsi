from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FIVE_PERCENT = 0.05
DEFAULT_MIN_TEMPORAL_OVERLAP_FRACTION = 0.9
PRESSURE_QUANTITY = "static_gauge_pressure_pa"
PRESSURE_REFERENCE = "outlet_0_pa"
PRESSURE_SEMANTIC_KEYS = ("pressure_quantity", "pressure_reference")


def load_pressure_semantics(path: str | Path) -> dict[str, str | None]:
    """Load optional pressure semantics without weakening NPZ type checks."""

    path = Path(path)
    semantics: dict[str, str | None] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in PRESSURE_SEMANTIC_KEYS:
            if key not in data.files:
                semantics[key] = None
                continue
            value = np.asarray(data[key])
            if value.shape != () or value.dtype.kind != "U":
                raise ValueError(
                    f"{path}: pressure semantics {key!r} must be a scalar Unicode string"
                )
            item = value.item()
            if not isinstance(item, str):
                raise ValueError(
                    f"{path}: pressure semantics {key!r} must be a scalar Unicode string"
                )
            semantics[key] = item
    return semantics


def load_fluent_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "x": data["x"].astype(np.float64),
            "y": data["y"].astype(np.float64),
            "u": data["u"].astype(np.float64),
            "v": data["v"].astype(np.float64),
            "p": data["p"].astype(np.float64),
            "speed": data["speed"].astype(np.float64),
            "cell_ids": data["cell_ids"].astype(np.int64),
        }


def load_solver_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        fluid_mask = data["fluid_mask"].astype(bool)
        solid_mask = (
            data["solid_mask"].astype(bool)
            if "solid_mask" in data.files
            else ~fluid_mask
        )
        boundary_surrogate_mask = (
            data["boundary_surrogate_mask"].astype(bool)
            if "boundary_surrogate_mask" in data.files
            else np.zeros_like(fluid_mask, dtype=bool)
        )
        display_fluid_mask = (
            data["display_fluid_mask"].astype(bool)
            if "display_fluid_mask" in data.files
            else fluid_mask | boundary_surrogate_mask
        )
        display_obstacle_mask = (
            data["display_obstacle_mask"].astype(bool)
            if "display_obstacle_mask" in data.files
            else ~display_fluid_mask
        )
        return {
            "s": data["s"].astype(np.float64),
            "y": data["y"].astype(np.float64),
            "u": data["u"].astype(np.float64),
            "v": data["v"].astype(np.float64),
            "p": data["p"].astype(np.float64),
            "speed": data["speed"].astype(np.float64),
            "fluid_mask": fluid_mask,
            "solid_mask": solid_mask,
            "boundary_surrogate_mask": boundary_surrogate_mask,
            "display_fluid_mask": display_fluid_mask,
            "display_obstacle_mask": display_obstacle_mask,
        }


def save_solver_npz_from_flow_snapshot(
    path: str | Path,
    snapshot: dict[str, Any],
    *,
    streamwise_velocity_sign: float = -1.0,
    reverse_streamwise_axis: bool = True,
    span_reduction: str = "mean",
    exclude_velocity_dirichlet_rows: bool = True,
    physical_solid_bounds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Persist a runner final_flow_field_snapshot in parity-comparison format."""

    velocity = np.asarray(snapshot["velocity"], dtype=np.float64)
    pressure = np.asarray(snapshot["pressure"], dtype=np.float64)
    obstacle = np.asarray(snapshot["obstacle"]) != 0
    dirichlet_active = np.asarray(
        snapshot.get("velocity_dirichlet_boundary_active", np.zeros_like(obstacle)),
        dtype=np.int32,
    ) != 0
    dirichlet_weight = np.asarray(
        snapshot.get(
            "velocity_dirichlet_boundary_projection_weight",
            np.zeros_like(pressure),
        ),
        dtype=np.float64,
    )
    cell_center_y = np.asarray(snapshot["cell_center_y_m"], dtype=np.float64)
    cell_center_z = np.asarray(snapshot["cell_center_z_m"], dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("snapshot velocity must have shape (nx, ny, nz, 3)")
    if pressure.shape != velocity.shape[:3] or obstacle.shape != velocity.shape[:3]:
        raise ValueError("snapshot pressure/obstacle shapes must match velocity grid")
    if dirichlet_active.shape != velocity.shape[:3]:
        raise ValueError("snapshot velocity_dirichlet_boundary_active shape must match velocity grid")
    if dirichlet_weight.shape != velocity.shape[:3]:
        raise ValueError("snapshot velocity_dirichlet_boundary_projection_weight shape must match velocity grid")

    boundary_surrogate_3d = dirichlet_active & (dirichlet_weight > 0.0)
    fluid_3d = ~obstacle
    comparison_3d = fluid_3d & ~boundary_surrogate_3d if exclude_velocity_dirichlet_rows else fluid_3d
    cell_center_velocity = _cell_center_velocity_from_solver_faces(velocity)
    u_3d = float(streamwise_velocity_sign) * cell_center_velocity[:, :, :, 2]
    v_3d = cell_center_velocity[:, :, :, 1]
    if span_reduction == "mean":
        u = _masked_span_mean(u_3d, fluid_3d)
        v = _masked_span_mean(v_3d, fluid_3d)
        p = _masked_span_mean(pressure, fluid_3d)
        fluid_mask = np.any(comparison_3d, axis=0)
        display_fluid_mask = np.any(fluid_3d, axis=0)
        boundary_surrogate_mask = np.any(boundary_surrogate_3d, axis=0)
    elif span_reduction == "center":
        center = velocity.shape[0] // 2
        u = u_3d[center, :, :]
        v = v_3d[center, :, :]
        p = pressure[center, :, :]
        fluid_mask = comparison_3d[center, :, :]
        display_fluid_mask = fluid_3d[center, :, :]
        boundary_surrogate_mask = boundary_surrogate_3d[center, :, :]
    else:
        raise ValueError(f"unsupported span_reduction: {span_reduction!r}")

    s = _grid_axis_from_centers(cell_center_z, axis=2)
    y = _grid_axis_from_centers(cell_center_y, axis=1)
    if reverse_streamwise_axis:
        s_upper_face = _outer_axis_face_from_centers(s)
        s = s_upper_face - s[::-1]
        u = u[:, ::-1]
        v = v[:, ::-1]
        p = p[:, ::-1]
        fluid_mask = fluid_mask[:, ::-1]
        display_fluid_mask = display_fluid_mask[:, ::-1]
        boundary_surrogate_mask = boundary_surrogate_mask[:, ::-1]
    display_obstacle_mask = ~display_fluid_mask
    solid_mask = (
        physical_solid_mask_from_bounds(s, y, physical_solid_bounds)
        if physical_solid_bounds is not None
        else display_obstacle_mask
    )
    speed = np.sqrt(u * u + v * v)
    if not (
        np.all(np.isfinite(s))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(u[fluid_mask]))
        and np.all(np.isfinite(v[fluid_mask]))
        and np.all(np.isfinite(p[fluid_mask]))
        and np.all(np.isfinite(speed[fluid_mask]))
    ):
        raise ValueError("non-finite values in solver flow snapshot")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        s=s,
        y=y,
        u=u,
        v=v,
        p=p,
        speed=speed,
        fluid_mask=fluid_mask.astype(bool),
        solid_mask=solid_mask.astype(bool),
        boundary_surrogate_mask=boundary_surrogate_mask.astype(bool),
        display_fluid_mask=display_fluid_mask.astype(bool),
        display_obstacle_mask=display_obstacle_mask.astype(bool),
        pressure_quantity=np.asarray(PRESSURE_QUANTITY),
        pressure_reference=np.asarray(PRESSURE_REFERENCE),
    )
    return {
        "path": str(path),
        "shape": list(u.shape),
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "max_speed": float(np.nanmax(speed[fluid_mask])) if np.any(fluid_mask) else 0.0,
        "fluid_cell_count": int(np.count_nonzero(fluid_mask)),
        "solid_cell_count": int(np.count_nonzero(solid_mask)),
        "boundary_surrogate_cell_count": int(np.count_nonzero(boundary_surrogate_mask)),
        "display_fluid_cell_count": int(np.count_nonzero(display_fluid_mask)),
        "display_obstacle_cell_count": int(np.count_nonzero(display_obstacle_mask)),
        "exclude_velocity_dirichlet_rows": bool(exclude_velocity_dirichlet_rows),
        "span_reduction": span_reduction,
        "streamwise_velocity_sign": float(streamwise_velocity_sign),
        "reverse_streamwise_axis": bool(reverse_streamwise_axis),
        "physical_solid_bounds": dict(physical_solid_bounds or {}),
    }


def physical_solid_mask_from_bounds(
    s_grid: np.ndarray,
    y_grid: np.ndarray,
    bounds: Mapping[str, float],
) -> np.ndarray:
    """Build a 2D display mask for the physical Fluent flap geometry."""

    s = np.asarray(s_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    s_min = _bound(bounds, "streamwise_min_m", "x_min", "s_min")
    s_max = _bound(bounds, "streamwise_max_m", "x_max", "s_max")
    y_min = _bound(bounds, "y_min_m", "y_min", default=0.0)
    y_max = _bound(bounds, "y_max_m", "y_max")
    if s_min >= s_max or y_min >= y_max:
        raise ValueError("physical solid bounds must have increasing extents")
    s_tol = _axis_tolerance(s)
    y_tol = _axis_tolerance(y)
    s_mask = (s >= s_min - s_tol) & (s < s_max + s_tol)
    y_mask = (y >= y_min - y_tol) & (y < y_max + y_tol)
    return y_mask[:, None] & s_mask[None, :]


def compare_solver_to_fluent_field(
    solver_fields: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    *,
    prefix: str = "",
    throat_x: float = 0.050,
    downstream_x: tuple[float, ...] = (0.054, 0.060, 0.070, 0.090),
    tolerance: float = FIVE_PERCENT,
) -> dict[str, Any]:
    """Compare a structured solver field to Fluent cell-centered samples."""

    samples = sample_structured_solver_at_fluent_points(solver_fields, fluent_fields)
    valid = samples["valid"]
    if not np.any(valid):
        raise ValueError("no overlapping valid solver/Fluent sampling points")

    reference_speed = fluent_fields["speed"][valid]
    reference_u = fluent_fields["u"][valid]
    solver_speed = samples["speed"][valid]
    solver_u = samples["u"][valid]

    metrics = {
        _key(prefix, "speed_max_rel_error"): _relative_error(
            np.max(solver_speed), np.max(reference_speed)
        ),
        _key(prefix, "u_max_rel_error"): _relative_error(
            np.max(solver_u), np.max(reference_u)
        ),
        _key(prefix, "speed_mean_rel_error"): _relative_error(
            np.mean(solver_speed), np.mean(reference_speed)
        ),
        _key(prefix, "pressure_range_rel_error"): _relative_error(
            np.max(samples["p"][valid]) - np.min(samples["p"][valid]),
            np.max(fluent_fields["p"][valid]) - np.min(fluent_fields["p"][valid]),
        ),
        _key(prefix, "pressure_drop_rel_error"): pressure_drop_rel_error(
            solver_samples=samples,
            fluent_fields=fluent_fields,
            valid=valid,
        ),
        _key(prefix, "mass_flow_inlet_rel_error"): mass_flow_proxy_rel_error(
            solver_samples=samples,
            fluent_fields=fluent_fields,
            valid=valid,
            inlet=True,
        ),
        _key(prefix, "mass_flow_outlet_rel_error"): mass_flow_proxy_rel_error(
            solver_samples=samples,
            fluent_fields=fluent_fields,
            valid=valid,
            inlet=False,
        ),
    }

    centerline_mask = _profile_band(
        fluent_fields["y"],
        target=float(np.max(fluent_fields["y"])),
    )
    throat_mask = _profile_band(fluent_fields["x"], target=throat_x)
    downstream_masks = [
        _profile_band(fluent_fields["x"], target=x_value) for x_value in downstream_x
    ]
    metrics[_key(prefix, "centerline_u_nrmse")] = _profile_nrmse(
        samples["u"], fluent_fields["u"], valid & centerline_mask
    )
    metrics[_key(prefix, "throat_u_nrmse")] = _profile_nrmse(
        samples["u"], fluent_fields["u"], valid & throat_mask
    )
    metrics[_key(prefix, "downstream_u_nrmse")] = _mean_metric(
        _profile_nrmse(samples["u"], fluent_fields["u"], valid & mask)
        for mask in downstream_masks
    )
    backflow_diagnostics = _downstream_near_wall_backflow_diagnostics(
        samples,
        fluent_fields,
        valid,
        downstream_x=downstream_x,
    )
    metrics[_key(prefix, "downstream_near_wall_backflow_fraction_abs_error")] = (
        backflow_diagnostics["negative_u_fraction_abs_error"]
    )
    metrics[_key(prefix, "downstream_near_wall_min_u_abs_error_mps")] = (
        backflow_diagnostics["min_u_abs_error_mps"]
    )
    metrics[_key(prefix, "downstream_near_wall_u_nrmse")] = (
        backflow_diagnostics["u_nrmse"]
    )

    diagnostic_metric_names = {
        _key(prefix, "pressure_range_rel_error"),
        _key(prefix, "downstream_near_wall_backflow_fraction_abs_error"),
        _key(prefix, "downstream_near_wall_min_u_abs_error_mps"),
        _key(prefix, "downstream_near_wall_u_nrmse"),
    }
    gate_results = {
        name: {
            "value": float(value),
            "threshold": float(tolerance),
            "status": "diagnostic"
            if name in diagnostic_metric_names
            else ("passed" if float(value) <= tolerance else "failed"),
        }
        for name, value in metrics.items()
    }
    return {
        "status": "passed"
        if all(
            item["status"] in {"passed", "diagnostic"}
            for item in gate_results.values()
        )
        else "failed",
        "threshold": tolerance,
        "metrics": metrics,
        "gates": gate_results,
        "diagnostics": _field_diagnostics(
            samples,
            fluent_fields,
            valid,
            downstream_near_wall_backflow=backflow_diagnostics,
        ),
        "sample_count": int(np.count_nonzero(valid)),
        "profile_sample_counts": {
            "centerline": int(np.count_nonzero(valid & centerline_mask)),
            "throat": int(np.count_nonzero(valid & throat_mask)),
            "downstream": [
                int(np.count_nonzero(valid & mask)) for mask in downstream_masks
            ],
        },
    }


def sample_structured_solver_at_fluent_points(
    solver_fields: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    x = fluent_fields["x"]
    y = _map_fluent_y_to_solver_y(solver_fields["y"], fluent_fields["y"])
    s_grid = solver_fields["s"]
    y_grid = solver_fields["y"]
    fluid_mask = solver_fields["fluid_mask"].astype(bool)
    sampled = {"valid": np.ones_like(x, dtype=bool)}
    fluid_weight = None
    for key in ("u", "v", "p", "speed"):
        values, weights = _fluid_weighted_bilinear_sample(
            s_grid,
            y_grid,
            solver_fields[key],
            fluid_mask,
            x,
            y,
        )
        sampled[key] = values
        sampled["valid"] &= np.isfinite(values) & (weights > 1.0e-12)
        fluid_weight = weights if fluid_weight is None else np.minimum(fluid_weight, weights)
    sampled["fluid_mask"] = (
        fluid_weight if fluid_weight is not None else np.zeros_like(x, dtype=np.float64)
    )
    return sampled


def _field_diagnostics(
    samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    valid: np.ndarray,
    *,
    downstream_near_wall_backflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pressure_extrema": _pressure_extrema_diagnostics(
            samples,
            fluent_fields,
            valid,
        ),
        "downstream_near_wall_backflow": downstream_near_wall_backflow
        if downstream_near_wall_backflow is not None
        else _downstream_near_wall_backflow_diagnostics(
            samples,
            fluent_fields,
            valid,
            downstream_x=(0.054, 0.060, 0.070, 0.090),
        ),
    }


def _downstream_near_wall_backflow_diagnostics(
    samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    valid: np.ndarray,
    *,
    downstream_x: Iterable[float],
) -> dict[str, Any]:
    x = np.asarray(fluent_fields["x"], dtype=np.float64)
    y = np.asarray(fluent_fields["y"], dtype=np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    near_wall_limit = y_min + 0.35 * max(y_max - y_min, 1.0e-12)
    downstream_mask = np.zeros_like(valid, dtype=bool)
    for x_value in downstream_x:
        downstream_mask |= _profile_band(x, target=float(x_value))
    mask = valid & downstream_mask & (y <= near_wall_limit)
    sample_count = int(np.count_nonzero(mask))
    if sample_count == 0:
        return {
            "sample_count": 0,
            "near_wall_y_max_m": near_wall_limit,
            "reference_negative_u_fraction": math.inf,
            "solver_negative_u_fraction": math.inf,
            "negative_u_fraction_abs_error": math.inf,
            "reference_min_u_mps": math.inf,
            "solver_min_u_mps": math.inf,
            "min_u_abs_error_mps": math.inf,
            "u_nrmse": math.inf,
        }

    reference_u = np.asarray(fluent_fields["u"], dtype=np.float64)[mask]
    solver_u = np.asarray(samples["u"], dtype=np.float64)[mask]
    reference_negative_fraction = float(np.mean(reference_u < 0.0))
    solver_negative_fraction = float(np.mean(solver_u < 0.0))
    reference_min_u = float(np.min(reference_u))
    solver_min_u = float(np.min(solver_u))
    return {
        "sample_count": sample_count,
        "near_wall_y_max_m": near_wall_limit,
        "reference_negative_u_fraction": reference_negative_fraction,
        "solver_negative_u_fraction": solver_negative_fraction,
        "negative_u_fraction_abs_error": abs(
            solver_negative_fraction - reference_negative_fraction
        ),
        "reference_min_u_mps": reference_min_u,
        "solver_min_u_mps": solver_min_u,
        "min_u_abs_error_mps": abs(solver_min_u - reference_min_u),
        "u_nrmse": _nrmse(solver_u, reference_u),
    }


def _pressure_extrema_diagnostics(
    samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    valid: np.ndarray,
) -> dict[str, Any]:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return {}
    reference_pressure = np.asarray(fluent_fields["p"], dtype=np.float64)
    solver_pressure = np.asarray(samples["p"], dtype=np.float64)
    reference_min_index = int(valid_indices[int(np.argmin(reference_pressure[valid]))])
    reference_max_index = int(valid_indices[int(np.argmax(reference_pressure[valid]))])
    solver_min_index = int(valid_indices[int(np.argmin(solver_pressure[valid]))])
    solver_max_index = int(valid_indices[int(np.argmax(solver_pressure[valid]))])
    return {
        "reference_pressure_min_point": _sample_pair_diagnostic(
            reference_min_index,
            samples,
            fluent_fields,
        ),
        "reference_pressure_max_point": _sample_pair_diagnostic(
            reference_max_index,
            samples,
            fluent_fields,
        ),
        "solver_pressure_min_point": _sample_pair_diagnostic(
            solver_min_index,
            samples,
            fluent_fields,
        ),
        "solver_pressure_max_point": _sample_pair_diagnostic(
            solver_max_index,
            samples,
            fluent_fields,
        ),
    }


def _sample_pair_diagnostic(
    index: int,
    samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
) -> dict[str, float | int]:
    return {
        "sample_index": int(index),
        "x_m": float(fluent_fields["x"][index]),
        "y_m": float(fluent_fields["y"][index]),
        "reference_u_mps": float(fluent_fields["u"][index]),
        "solver_u_mps": float(samples["u"][index]),
        "reference_v_mps": float(fluent_fields["v"][index]),
        "solver_v_mps": float(samples["v"][index]),
        "reference_speed_mps": float(fluent_fields["speed"][index]),
        "solver_speed_mps": float(samples["speed"][index]),
        "reference_pressure_pa": float(fluent_fields["p"][index]),
        "solver_pressure_pa": float(samples["p"][index]),
    }


def fluent_to_solver_y(
    solver_y_grid: np.ndarray,
    fluent_y: np.ndarray,
) -> np.ndarray:
    return _map_fluent_y_to_solver_y(solver_y_grid, fluent_y)


def compare_structure_monitor(
    solver_rows: list[dict[str, Any]],
    fluent_rows: list[dict[str, Any]],
    *,
    solver_displacement_key: str,
    fluent_displacement_key: str = "monitor_avg_total_col0_col6_m",
    solver_solid_max_key: str | None = None,
    fluent_solid_max_key: str = "solid_max_total_col0_col6_m",
    tolerance: float = FIVE_PERCENT,
    min_temporal_overlap_fraction: float = DEFAULT_MIN_TEMPORAL_OVERLAP_FRACTION,
) -> dict[str, Any]:
    if not solver_rows:
        raise ValueError("solver structure rows are empty")
    if not fluent_rows:
        raise ValueError("Fluent structure rows are empty")
    solver_series = np.array([float(row[solver_displacement_key]) for row in solver_rows])
    fluent_series = np.array([float(row[fluent_displacement_key]) for row in fluent_rows])
    solver_length = int(solver_series.size)
    fluent_length = int(fluent_series.size)
    count = min(solver_length, fluent_length)
    if count <= 0:
        raise ValueError("structure series do not overlap")
    # A shorter series silently truncated against a much longer one (e.g. 1
    # solver sample vs 50 Fluent samples) discards nearly all of the
    # reference data and can trivially "pass" on whatever the first row
    # happens to be. Require the shorter series to cover most of the longer
    # one before treating the truncated comparison as meaningful.
    longer_length = max(solver_length, fluent_length)
    overlap_fraction = count / longer_length if longer_length > 0 else 0.0
    if overlap_fraction < float(min_temporal_overlap_fraction):
        return {
            "status": "failed",
            "reason": (
                f"insufficient temporal overlap: {solver_length} vs {fluent_length}"
            ),
            "solver_sample_count": solver_length,
            "fluent_sample_count": fluent_length,
            "overlap_fraction": overlap_fraction,
            "min_temporal_overlap_fraction": float(min_temporal_overlap_fraction),
            "metrics": {},
            "gates": {},
            "sample_count": count,
        }
    solver_series = solver_series[:count]
    fluent_series = fluent_series[:count]
    solver_peak_index = int(np.argmax(solver_series))
    fluent_peak_index = int(np.argmax(fluent_series))
    solver_peak_step = int(solver_rows[solver_peak_index].get("step", solver_peak_index + 1))
    fluent_peak_step = int(fluent_rows[fluent_peak_index].get("step", fluent_peak_index + 1))
    solver_peak_time = float(solver_rows[solver_peak_index].get("time_s", math.nan))
    fluent_peak_time = float(fluent_rows[fluent_peak_index].get("time_s", math.nan))

    metrics = {
        "monitor_displacement_peak_abs_error_m": abs(
            float(np.max(solver_series)) - float(np.max(fluent_series))
        ),
        "monitor_displacement_peak_rel_error": _relative_error(
            np.max(solver_series), np.max(fluent_series)
        ),
        "monitor_peak_step_error": abs(solver_peak_step - fluent_peak_step),
        "monitor_peak_time_error_s": abs(solver_peak_time - fluent_peak_time),
        "monitor_final_displacement_abs_error_m": abs(
            float(solver_series[-1]) - float(fluent_series[-1])
        ),
        "monitor_final_displacement_rel_error": _relative_error(
            solver_series[-1], fluent_series[-1]
        ),
        "monitor_timeseries_nrmse": _nrmse(solver_series, fluent_series),
    }
    if solver_solid_max_key:
        solver_solid = float(solver_rows[count - 1][solver_solid_max_key])
        fluent_solid = float(fluent_rows[count - 1][fluent_solid_max_key])
        metrics["solid_max_final_displacement_rel_error"] = _relative_error(
            solver_solid, fluent_solid
        )
    gate_limits = {
        "monitor_displacement_peak_rel_error": tolerance,
        "monitor_peak_step_error": 1,
        "monitor_peak_time_error_s": 5.0e-4,
        "monitor_final_displacement_rel_error": tolerance,
        "monitor_timeseries_nrmse": tolerance,
        "solid_max_final_displacement_rel_error": tolerance,
    }
    gates = {}
    for name, limit in gate_limits.items():
        if name in metrics:
            gates[name] = {
                "value": float(metrics[name]),
                "threshold": float(limit),
                "status": "passed" if float(metrics[name]) <= float(limit) else "failed",
            }
    return {
        "status": "passed"
        if all(item["status"] == "passed" for item in gates.values())
        else "failed",
        "metrics": metrics,
        "gates": gates,
        "sample_count": count,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: _parse(value) for key, value in row.items()})
        return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bilinear_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    result = np.full_like(x, np.nan, dtype=np.float64)
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])
    x_lower_tol = 0.5 * abs(float(x_grid[1] - x_grid[0])) if len(x_grid) > 1 else 0.0
    x_upper_tol = 0.5 * abs(float(x_grid[-1] - x_grid[-2])) if len(x_grid) > 1 else 0.0
    y_lower_tol = 0.5 * abs(float(y_grid[1] - y_grid[0])) if len(y_grid) > 1 else 0.0
    y_upper_tol = 0.5 * abs(float(y_grid[-1] - y_grid[-2])) if len(y_grid) > 1 else 0.0
    inside = (
        (x >= x_min - x_lower_tol)
        & (x <= x_max + x_upper_tol)
        & (y >= y_min - y_lower_tol)
        & (y <= y_max + y_upper_tol)
    )
    if not np.any(inside):
        return result
    sample_x = np.clip(x[inside], x_min, x_max)
    sample_y = np.clip(y[inside], y_min, y_max)
    x_idx = np.searchsorted(x_grid, sample_x, side="right") - 1
    y_idx = np.searchsorted(y_grid, sample_y, side="right") - 1
    x_idx = np.clip(x_idx, 0, len(x_grid) - 2)
    y_idx = np.clip(y_idx, 0, len(y_grid) - 2)
    x0 = x_grid[x_idx]
    x1 = x_grid[x_idx + 1]
    y0 = y_grid[y_idx]
    y1 = y_grid[y_idx + 1]
    tx = _safe_fraction(sample_x - x0, x1 - x0)
    ty = _safe_fraction(sample_y - y0, y1 - y0)
    v00 = values[y_idx, x_idx]
    v10 = values[y_idx, x_idx + 1]
    v01 = values[y_idx + 1, x_idx]
    v11 = values[y_idx + 1, x_idx + 1]
    result[inside] = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return result


def _fluid_weighted_bilinear_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    fluid_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full_like(x, np.nan, dtype=np.float64)
    fluid_weight = np.zeros_like(x, dtype=np.float64)
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])
    x_lower_tol = 0.5 * abs(float(x_grid[1] - x_grid[0])) if len(x_grid) > 1 else 0.0
    x_upper_tol = 0.5 * abs(float(x_grid[-1] - x_grid[-2])) if len(x_grid) > 1 else 0.0
    y_lower_tol = 0.5 * abs(float(y_grid[1] - y_grid[0])) if len(y_grid) > 1 else 0.0
    y_upper_tol = 0.5 * abs(float(y_grid[-1] - y_grid[-2])) if len(y_grid) > 1 else 0.0
    inside = (
        (x >= x_min - x_lower_tol)
        & (x <= x_max + x_upper_tol)
        & (y >= y_min - y_lower_tol)
        & (y <= y_max + y_upper_tol)
    )
    if not np.any(inside):
        return result, fluid_weight
    sample_x = np.clip(x[inside], x_min, x_max)
    sample_y = np.clip(y[inside], y_min, y_max)
    x_idx = np.searchsorted(x_grid, sample_x, side="right") - 1
    y_idx = np.searchsorted(y_grid, sample_y, side="right") - 1
    x_idx = np.clip(x_idx, 0, len(x_grid) - 2)
    y_idx = np.clip(y_idx, 0, len(y_grid) - 2)
    x0 = x_grid[x_idx]
    x1 = x_grid[x_idx + 1]
    y0 = y_grid[y_idx]
    y1 = y_grid[y_idx + 1]
    tx = _safe_fraction(sample_x - x0, x1 - x0)
    ty = _safe_fraction(sample_y - y0, y1 - y0)
    weights = (
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    )
    values_at_corners = (
        values[y_idx, x_idx],
        values[y_idx, x_idx + 1],
        values[y_idx + 1, x_idx],
        values[y_idx + 1, x_idx + 1],
    )
    masks_at_corners = (
        fluid_mask[y_idx, x_idx],
        fluid_mask[y_idx, x_idx + 1],
        fluid_mask[y_idx + 1, x_idx],
        fluid_mask[y_idx + 1, x_idx + 1],
    )
    weighted = np.zeros_like(sample_x, dtype=np.float64)
    denominator = np.zeros_like(sample_x, dtype=np.float64)
    for weight, value, mask in zip(weights, values_at_corners, masks_at_corners):
        active_weight = weight * mask.astype(np.float64)
        weighted += active_weight * value
        denominator += active_weight
    sampled = np.full_like(sample_x, np.nan, dtype=np.float64)
    valid = denominator > 1.0e-12
    sampled[valid] = weighted[valid] / denominator[valid]
    result[inside] = sampled
    fluid_weight[inside] = denominator
    return result, fluid_weight


def _map_fluent_y_to_solver_y(
    solver_y_grid: np.ndarray, fluent_y: np.ndarray
) -> np.ndarray:
    solver_min = float(np.min(solver_y_grid))
    solver_max = float(np.max(solver_y_grid))
    fluent_min = float(np.min(fluent_y))
    fluent_max = float(np.max(fluent_y))
    solver_crosses_center = solver_min < 0.0 < solver_max
    fluent_is_positive_domain = fluent_min >= -1.0e-12
    if solver_crosses_center and fluent_is_positive_domain:
        # The official Fluent tutorial mesh is a half domain with wall at y=0
        # and symmetry at y=max. The centered solver's symmetry line is y=0.
        return fluent_y - fluent_max
    return fluent_y


def _cell_center_velocity_from_solver_faces(velocity: np.ndarray) -> np.ndarray:
    """Reconstruct cell-centered velocity from the solver's face-like storage."""

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("velocity must have shape (nx, ny, nz, 3)")
    centered = np.array(velocity, copy=True, dtype=np.float64)
    for axis in range(3):
        component = velocity[..., axis]
        upper = np.array(component, copy=True, dtype=np.float64)
        if component.shape[axis] > 1:
            lower_slices = [slice(None)] * 3
            upper_slices = [slice(None)] * 3
            lower_slices[axis] = slice(0, -1)
            upper_slices[axis] = slice(1, None)
            upper[tuple(lower_slices)] = component[tuple(upper_slices)]
        centered[..., axis] = 0.5 * (component + upper)
    return centered


def pressure_drop_rel_error(
    *,
    solver_samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    valid: np.ndarray,
) -> float:
    x = fluent_fields["x"]
    inlet = valid & _profile_band(x, target=float(np.min(x)))
    outlet = valid & _profile_band(x, target=float(np.max(x)))
    if np.count_nonzero(inlet) == 0 or np.count_nonzero(outlet) == 0:
        return math.inf
    solver_drop = float(np.mean(solver_samples["p"][inlet]) - np.mean(solver_samples["p"][outlet]))
    fluent_drop = float(np.mean(fluent_fields["p"][inlet]) - np.mean(fluent_fields["p"][outlet]))
    return _relative_error(solver_drop, fluent_drop)


def mass_flow_proxy_rel_error(
    *,
    solver_samples: dict[str, np.ndarray],
    fluent_fields: dict[str, np.ndarray],
    valid: np.ndarray,
    inlet: bool,
) -> float:
    x = fluent_fields["x"]
    target = float(np.min(x) if inlet else np.max(x))
    band = valid & _profile_band(x, target=target)
    if np.count_nonzero(band) == 0:
        return math.inf
    # Same sample points are used for both codes, so mean streamwise velocity is
    # a deterministic flow-rate proxy without inventing face areas.
    return _relative_error(np.mean(solver_samples["u"][band]), np.mean(fluent_fields["u"][band]))


def _profile_band(values: np.ndarray, *, target: float) -> np.ndarray:
    distances = np.abs(values - target)
    if distances.size == 0:
        return np.zeros_like(distances, dtype=bool)
    quantile_cutoff = float(np.quantile(distances, 0.02))
    nearest = np.sort(distances)[: max(16, int(0.01 * distances.size))]
    nearest_cutoff = float(np.max(nearest)) if nearest.size else 0.0
    cutoff = max(quantile_cutoff, nearest_cutoff)
    return distances <= max(cutoff, 1.0e-12)


def _profile_nrmse(
    solver_values: np.ndarray, reference_values: np.ndarray, mask: np.ndarray
) -> float:
    if np.count_nonzero(mask) == 0:
        return math.inf
    return _nrmse(solver_values[mask], reference_values[mask])


def _nrmse(values: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.shape != reference.shape:
        raise ValueError("NRMSE arrays must have matching shapes")
    rms = math.sqrt(float(np.mean((values - reference) ** 2)))
    denom = max(float(np.max(np.abs(reference))), 1.0e-12)
    return rms / denom


def _relative_error(value: float | np.ndarray, reference: float | np.ndarray) -> float:
    value_float = float(value)
    reference_float = float(reference)
    return abs(value_float - reference_float) / max(abs(reference_float), 1.0e-12)


def _mean_metric(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.inf
    return float(np.mean(finite))


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=np.abs(denominator) > 1.0e-15)
    return out


def _axis_tolerance(axis_values: np.ndarray) -> float:
    axis = np.asarray(axis_values, dtype=np.float64)
    if axis.size < 2:
        return 1.0e-7
    finite = axis[np.isfinite(axis)]
    if finite.size < 2:
        return 1.0e-7
    spacing = np.diff(np.sort(finite))
    positive = spacing[spacing > 0.0]
    if positive.size == 0:
        return 1.0e-7
    return max(1.0e-7, 1.0e-6 * float(np.min(positive)))


def _bound(
    bounds: Mapping[str, float],
    *names: str,
    default: float | None = None,
) -> float:
    for name in names:
        if name in bounds:
            return float(bounds[name])
    if default is not None:
        return float(default)
    expected = ", ".join(names)
    raise KeyError(f"missing physical solid bound; expected one of: {expected}")


def _masked_span_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weighted = np.where(mask, values, 0.0)
    counts = np.count_nonzero(mask, axis=0).astype(np.float64)
    summed = np.sum(weighted, axis=0)
    mean = np.zeros_like(summed, dtype=np.float64)
    np.divide(summed, counts, out=mean, where=counts > 0.0)
    return mean


def _grid_axis_from_centers(centers: np.ndarray, *, axis: int) -> np.ndarray:
    if centers.ndim == 1:
        return np.asarray(centers, dtype=np.float64)
    if centers.ndim != 3:
        raise ValueError("cell-center coordinate arrays must be 1D or 3D")
    reduced = np.nanmean(centers, axis=tuple(i for i in range(3) if i != axis))
    return np.asarray(reduced, dtype=np.float64)


def _outer_axis_face_from_centers(axis_values: np.ndarray) -> float:
    axis = np.asarray(axis_values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("axis coordinate array must be a non-empty 1D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError("axis coordinate array contains non-finite values")
    if axis.size == 1:
        return float(axis[0])
    return float(axis[-1] + 0.5 * (axis[-1] - axis[-2]))


def _key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _parse(value: str) -> float | int | str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number
