from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
    ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING,
    VerticalFlapFsiConfig,
    thin_wall_pressure_probe_max_multiplier,
    with_local_surface_force_support,
)
from benchmarks.official.solid_mpm_fsi_runner import (  # noqa: E402
    PRIMARY_REGION_ID,
    SECONDARY_UNUSED_REGION_ID,
    _lame_parameters,
)
from simulation_core.fluid import CartesianFluidSolver, FluidDomainSpec  # noqa: E402
from simulation_core.hibm_mpm import (  # noqa: E402
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    advance_hibm_mpm_sharp_mpm_step,
)
from simulation_core.neo_hookean_mpm import NeoHookeanMpmState  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "true_sharp_fsi.log"
PROCESS_PATH = ROOT / "true_sharp_fsi_process.json"
REPORT_PATH = ROOT / "true_sharp_fsi_report.json"
HISTORY_PATH = ROOT / "true_sharp_fsi_history.csv"
FIELDS_PATH = ROOT / "true_sharp_fsi_fields.npz"

GRID_NODES = (4, 128, 256)
SOLID_PARTICLE_COUNTS = (1, 80, 24)
MARKERS_PER_FACE = 48
SOLID_SUBSTEPS = 200
PREFLOW_STEPS = 0
FLAP_NAMES = ("lower", "upper")


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    try:
        print(message, flush=True)
    except OSError:
        pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _config(
    step_count: int,
    grid_nodes: tuple[int, int, int] = GRID_NODES,
    solid_particle_counts: tuple[int, int, int] = SOLID_PARTICLE_COUNTS,
    markers_per_face: int = MARKERS_PER_FACE,
    solid_substeps: int = SOLID_SUBSTEPS,
) -> VerticalFlapFsiConfig:
    return with_local_surface_force_support(VerticalFlapFsiConfig(
        step_count=int(step_count),
        grid_nodes=grid_nodes,
        solid_particle_counts=solid_particle_counts,
        marker_count=4 * int(markers_per_face),
        solid_substeps=int(solid_substeps),
        flow_pressure_solver="fv_cg",
        flow_cg_tolerance=1.0e-6,
    ))


def _full_bounds(config: VerticalFlapFsiConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.0, 0.0, 0.0), (
        float(config.span_m),
        float(config.duct_height_m),
        float(config.duct_length_m),
    )


def _solid_bounds(config: VerticalFlapFsiConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dy = float(config.duct_height_m) / float(config.grid_nodes[1])
    pad_y = 3.0 * dy
    return (0.0, -pad_y, 0.0), (
        float(config.span_m),
        float(config.duct_height_m) + pad_y,
        float(config.duct_length_m),
    )


def _flap_boxes(config: VerticalFlapFsiConfig) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    center_z = 0.5 * float(config.duct_length_m)
    half_t = 0.5 * float(config.flap_thickness_m)
    x_min = 0.0
    x_max = float(config.span_m)
    return {
        "lower": (
            (x_min, 0.0, center_z - half_t),
            (x_max, float(config.flap_height_m), center_z + half_t),
        ),
        "upper": (
            (x_min, float(config.duct_height_m) - float(config.flap_height_m), center_z - half_t),
            (x_max, float(config.duct_height_m), center_z + half_t),
        ),
    }


def _build_fluid(config: VerticalFlapFsiConfig, runtime: TaichiRuntimeConfig) -> CartesianFluidSolver:
    bounds_min, bounds_max = _full_bounds(config)
    fluid = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=config.grid_nodes,
            density_kgm3=config.air_density_kgm3,
            viscosity_pa_s=config.air_viscosity_pa_s,
            dt_s=config.dt_s,
        ),
        runtime=runtime,
    )
    velocity = np.zeros((*config.grid_nodes, 3), dtype=np.float32)
    velocity[..., 2] = -float(config.inlet_velocity_mps)
    fluid.velocity.from_numpy(velocity)
    fluid.velocity_prev.from_numpy(velocity)
    fluid.pressure.from_numpy(np.zeros(config.grid_nodes, dtype=np.float32))
    fluid.obstacle.from_numpy(np.zeros(config.grid_nodes, dtype=np.int32))
    fluid.clear_volume_source()
    fluid.add_zmax_velocity_inlet_volume_source(
        normal_velocity_mps=-float(config.inlet_velocity_mps),
    )
    return fluid


def _single_box_solid(
    config: VerticalFlapFsiConfig,
    runtime: TaichiRuntimeConfig,
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> NeoHookeanMpmState:
    bounds_min, bounds_max = _solid_bounds(config)
    count = math.prod(config.solid_particle_counts)
    solid = NeoHookeanMpmState(
        particle_capacity=count,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        grid_nodes=config.grid_nodes,
        runtime=runtime,
    )
    solid.initialize_box(
        particle_counts=config.solid_particle_counts,
        box_min_m=box[0],
        box_max_m=box[1],
        density_kgm3=config.solid_density_kgm3,
    )
    return solid


def _build_solid(config: VerticalFlapFsiConfig, runtime: TaichiRuntimeConfig) -> tuple[NeoHookeanMpmState, dict[str, np.ndarray]]:
    boxes = _flap_boxes(config)
    bounds_min, bounds_max = _solid_bounds(config)
    per_flap = math.prod(config.solid_particle_counts)
    capacity = 2 * per_flap
    combined = NeoHookeanMpmState(
        particle_capacity=capacity,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        grid_nodes=config.grid_nodes,
        runtime=runtime,
    )
    parts = [_single_box_solid(config, runtime, boxes[name]) for name in FLAP_NAMES]
    combined.particle_count = capacity
    for field_name in ("x", "rest_x", "v", "mass_kg", "volume_m3", "C", "F"):
        values = [getattr(part, field_name).to_numpy()[:per_flap] for part in parts]
        getattr(combined, field_name).from_numpy(np.concatenate(values, axis=0))
    combined.saved_x.from_numpy(combined.x.to_numpy())
    combined.saved_v.from_numpy(combined.v.to_numpy())
    combined.saved_C.from_numpy(combined.C.to_numpy())
    combined.saved_F.from_numpy(combined.F.to_numpy())

    rest = combined.rest_x.to_numpy()
    fixed = np.zeros(capacity, dtype=np.int32)
    region = np.full(capacity, PRIMARY_REGION_ID, dtype=np.int32)
    normals = np.zeros((capacity, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    areas = np.full(capacity, float(config.flap_height_m * config.span_m) / float(capacity), dtype=np.float32)
    row_h = float(config.flap_height_m) / float(config.solid_particle_counts[1])
    lower_idx = np.arange(0, per_flap)
    upper_idx = np.arange(per_flap, capacity)
    fixed[lower_idx[rest[lower_idx, 1] <= boxes["lower"][0][1] + 1.01 * row_h]] = 1
    fixed[upper_idx[rest[upper_idx, 1] >= boxes["upper"][1][1] - 1.01 * row_h]] = 1
    combined.fixed_particle.from_numpy(fixed)
    combined.region_id.from_numpy(region)
    combined.surface_normal.from_numpy(normals)
    combined.rest_surface_normal.from_numpy(normals)
    combined.area_weight_m2.from_numpy(areas)
    combined.rest_area_weight_m2.from_numpy(areas)
    combined.external_force_n.from_numpy(np.zeros((capacity, 3), dtype=np.float32))
    combined._update_rest_center()
    masks = {
        "fixed": fixed.astype(bool),
        "lower": lower_idx,
        "upper": upper_idx,
        "lower_tip": lower_idx[rest[lower_idx, 1] >= boxes["lower"][1][1] - 1.01 * row_h],
        "upper_tip": upper_idx[rest[upper_idx, 1] <= boxes["upper"][0][1] + 1.01 * row_h],
    }
    return combined, masks


def _build_markers(
    config: VerticalFlapFsiConfig,
    runtime: TaichiRuntimeConfig,
    markers_per_face: int,
) -> HibmMpmSurfaceMarkers:
    boxes = _flap_boxes(config)
    dz = float(config.duct_length_m) / float(config.grid_nodes[2])
    x_center = 0.5 * (boxes["lower"][0][0] + boxes["lower"][1][0])
    area = float(config.flap_height_m * (boxes["lower"][1][0] - boxes["lower"][0][0])) / float(markers_per_face)
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    areas: list[float] = []
    for name in FLAP_NAMES:
        y_min = boxes[name][0][1]
        y_max = boxes[name][1][1]
        segment = (y_max - y_min) / float(markers_per_face)
        faces = (
            ((0.0, 0.0, 1.0), boxes[name][1][2] + 0.51 * dz),
            ((0.0, 0.0, -1.0), boxes[name][0][2] - 0.51 * dz),
        )
        for normal, z in faces:
            for marker in range(markers_per_face):
                y = y_min + (float(marker) + 0.5) * segment
                positions.append((x_center, y, z))
                normals.append(normal)
                areas.append(area)
    markers = HibmMpmSurfaceMarkers(marker_capacity=len(positions), runtime=runtime)
    markers.load_markers(
        positions_m=positions,
        velocities_mps=[(0.0, 0.0, 0.0)] * len(positions),
        normals=normals,
        areas_m2=areas,
        region_ids=[PRIMARY_REGION_ID] * len(positions),
    )
    return markers


def _marker_face_diagnostics(
    markers: HibmMpmSurfaceMarkers,
    markers_per_face: int,
) -> dict[str, dict[str, Any]]:
    valid = markers._stress_pressure_valid.to_numpy()[: markers.marker_count]
    tractions = markers.t_gamma_pa.to_numpy()[: markers.marker_count]
    forces = markers.F_gamma_n.to_numpy()[: markers.marker_count]
    marker_diagnostics = tuple(markers.stress_marker_diagnostics())
    face_ranges = {
        "lower_plus_z": (0, markers_per_face),
        "lower_minus_z": (markers_per_face, 2 * markers_per_face),
        "upper_plus_z": (2 * markers_per_face, 3 * markers_per_face),
        "upper_minus_z": (3 * markers_per_face, 4 * markers_per_face),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (start, stop) in face_ranges.items():
        face_valid = valid[start:stop] != 0
        face_tractions = tractions[start:stop]
        face_forces = forces[start:stop]
        face_marker_diagnostics = marker_diagnostics[start:stop]
        invalid_reason_counts: dict[str, int] = {}
        for diagnostic in face_marker_diagnostics:
            if diagnostic.get("valid", False):
                continue
            reason = str(diagnostic.get("invalid_reason", "unknown"))
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
        traction_norm = np.linalg.norm(face_tractions, axis=1)
        result[name] = {
            "marker_count": int(stop - start),
            "valid_marker_count": int(np.count_nonzero(face_valid)),
            "invalid_marker_count": int((stop - start) - np.count_nonzero(face_valid)),
            "invalid_reason_counts": invalid_reason_counts,
            "base_pressure_found_count": int(
                sum(
                    1
                    for item in face_marker_diagnostics
                    if item.get("base_pressure_found", False)
                )
            ),
            "inside_pressure_found_count": int(
                sum(
                    1
                    for item in face_marker_diagnostics
                    if item.get("inside_pressure_found", False)
                )
            ),
            "outside_pressure_found_count": int(
                sum(
                    1
                    for item in face_marker_diagnostics
                    if item.get("outside_pressure_found", False)
                )
            ),
            "pressure_anchor_available_count": int(
                sum(
                    1
                    for item in face_marker_diagnostics
                    if item.get("pressure_anchor_available", False)
                )
            ),
            "max_abs_traction_pa": float(traction_norm.max(initial=0.0)),
            "force_n": tuple(float(v) for v in face_forces.sum(axis=0)),
        }
    return result


def _solid_displacement(solid: NeoHookeanMpmState, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    current = solid.x.to_numpy()[: solid.particle_count]
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    disp = current - rest
    norm = np.linalg.norm(disp, axis=1)
    result: dict[str, Any] = {
        "max_displacement_m": float(norm.max(initial=0.0)),
        "fixed_root_max_displacement_m": float(norm[masks["fixed"]].max(initial=0.0)),
    }
    for name in FLAP_NAMES:
        tip = masks[f"{name}_tip"]
        tip_mean = disp[tip].mean(axis=0)
        result[f"{name}_tip_mean_displacement_m"] = tuple(float(v) for v in tip_mean)
    return result


def _fluid_speed_metrics(fluid: CartesianFluidSolver) -> dict[str, float]:
    velocity = fluid.velocity.to_numpy()
    speed = np.linalg.norm(velocity, axis=-1)
    return {
        "fluid_speed_max_mps": float(speed.max(initial=0.0)),
        "fluid_speed_p99_mps": float(np.percentile(speed, 99.0)),
        "fluid_speed_p999_mps": float(np.percentile(speed, 99.9)),
    }


def _reference_comparison(
    *,
    config: VerticalFlapFsiConfig,
    max_displacement_m: float,
    fluid_speed_max_mps: float,
    fluid_speed_p999_mps: float,
) -> dict[str, Any]:
    target_displacement = float(
        ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS["max_displacement_m"]
    )
    velocity_range = tuple(
        float(value)
        for value in ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS[
            "local_velocity_peak_range_mps"
        ]
    )
    displacement_relative_error = abs(max_displacement_m - target_displacement) / max(
        target_displacement,
        1.0e-30,
    )
    return {
        "reference_max_displacement_m": target_displacement,
        "max_displacement_relative_error": displacement_relative_error,
        "displacement_within_tolerance": (
            displacement_relative_error <= float(config.displacement_tolerance)
        ),
        "reference_local_velocity_peak_range_mps": velocity_range,
        "fluid_speed_max_mps": float(fluid_speed_max_mps),
        "fluid_speed_p999_mps": float(fluid_speed_p999_mps),
        "velocity_peak_metric": "fluid_speed_p999_mps",
        "velocity_peak_within_reference_range": (
            velocity_range[0]
            <= float(fluid_speed_p999_mps)
            <= velocity_range[1] * (1.0 + float(config.velocity_peak_tolerance))
        ),
    }


def _write_history(history: list[dict[str, Any]]) -> None:
    fields = [
        "step",
        "time_s",
        "stress_valid_marker_count",
        "stress_invalid_marker_count",
        "marker_force_z_n",
        "scatter_active_particle_count",
        "projection_l2",
        "projection_max_abs",
        "max_displacement_m",
        "fixed_root_max_displacement_m",
        "fluid_speed_max_mps",
        "fluid_speed_p99_mps",
        "fluid_speed_p999_mps",
        "velocity_dirichlet_max_abs_velocity_mps",
        "velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps",
        "velocity_dirichlet_boundary_velocity_only_rows",
        "pressure_neumann_max_abs_rhs",
        "no_slip_max_residual_mps",
        "next_velocity_dirichlet_max_abs_velocity_mps",
        "next_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps",
        "next_velocity_dirichlet_boundary_velocity_only_rows",
        "next_pressure_neumann_max_abs_rhs",
        "post_solid_no_slip_max_residual_mps",
        "surface_feedback_max_marker_speed_mps",
    ]
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in fields})


def _write_field_snapshot(
    *,
    path: Path,
    config: VerticalFlapFsiConfig,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    markers: HibmMpmSurfaceMarkers,
    initial_marker_position_m: np.ndarray,
) -> None:
    marker_count = int(markers.marker_count)
    particle_count = int(solid.particle_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid_nodes=np.asarray(config.grid_nodes, dtype=np.int32),
        duct_length_m=np.asarray(float(config.duct_length_m), dtype=np.float64),
        duct_height_m=np.asarray(float(config.duct_height_m), dtype=np.float64),
        span_m=np.asarray(float(config.span_m), dtype=np.float64),
        flap_height_m=np.asarray(float(config.flap_height_m), dtype=np.float64),
        flap_thickness_m=np.asarray(float(config.flap_thickness_m), dtype=np.float64),
        pressure_pa=fluid.pressure.to_numpy(),
        velocity_mps=fluid.velocity.to_numpy(),
        obstacle=fluid.obstacle.to_numpy(),
        solid_position_m=solid.x.to_numpy()[:particle_count],
        solid_rest_position_m=solid.rest_x.to_numpy()[:particle_count],
        solid_fixed=solid.fixed_particle.to_numpy()[:particle_count],
        solid_region_id=solid.region_id.to_numpy()[:particle_count],
        solid_external_force_n=solid.external_force_n.to_numpy()[:particle_count],
        marker_position_m=markers.x_gamma_m.to_numpy()[:marker_count],
        marker_initial_position_m=initial_marker_position_m[:marker_count],
        marker_normal=markers.n_gamma.to_numpy()[:marker_count],
        marker_force_n=markers.F_gamma_n.to_numpy()[:marker_count],
        marker_traction_pa=markers.t_gamma_pa.to_numpy()[:marker_count],
        marker_valid=markers._stress_pressure_valid.to_numpy()[:marker_count],
    )


def run(
    step_count: int,
    projection_iterations: int,
    grid_nodes: tuple[int, int, int] = GRID_NODES,
    solid_particle_counts: tuple[int, int, int] = SOLID_PARTICLE_COUNTS,
    markers_per_face: int = MARKERS_PER_FACE,
    pressure_solve_failure_policy: str = "raise",
    convert_internal_nodes_to_obstacles: bool = True,
    fluid_advection_scheme: str = "rk2",
    fluid_substeps: int = 1,
    solid_substeps: int = SOLID_SUBSTEPS,
    preflow_steps: int = PREFLOW_STEPS,
    stop_on_speed_mps: float | None = None,
) -> dict[str, Any]:
    config = _config(
        step_count,
        grid_nodes,
        solid_particle_counts,
        markers_per_face,
        solid_substeps,
    )
    speed_stop_threshold = None if stop_on_speed_mps is None else float(stop_on_speed_mps)
    if speed_stop_threshold is not None and speed_stop_threshold <= 0.0:
        speed_stop_threshold = None
    pressure_sampling_policy = dict(ANSYS_VERTICAL_FLAP_THIN_WALL_PRESSURE_SAMPLING)
    pressure_sampling_model = str(pressure_sampling_policy.get("model", ""))
    if pressure_sampling_model == "one-sided-reference-pressure":
        one_sided_pressure_region_id = PRIMARY_REGION_ID
        one_sided_reference_pressure_pa = float(
            pressure_sampling_policy["reference_pressure_pa"]
        )
    elif pressure_sampling_model == "two-sided-fluid-pressure":
        one_sided_pressure_region_id = -1
        one_sided_reference_pressure_pa = 0.0
    else:
        raise ValueError(f"unsupported thin-wall pressure model: {pressure_sampling_model}")
    one_sided_probe_max_multiplier = float(
        pressure_sampling_policy["probe_max_multiplier"]
    )
    two_sided_probe_max_multiplier = thin_wall_pressure_probe_max_multiplier(config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = _build_fluid(config, runtime)
    solid, masks = _build_solid(config, runtime)
    markers = _build_markers(config, runtime, markers_per_face)
    initial_marker_position_m = markers.x_gamma_m.to_numpy()[: markers.marker_count].copy()
    bounds_min, bounds_max = _full_bounds(config)
    search = HibmMpmIbNodeSearch(
        grid_nodes=config.grid_nodes,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        marker_capacity=markers.marker_count,
        runtime=runtime,
    )
    boundary = HibmMpmIbBoundaryConditions(
        grid_nodes=config.grid_nodes,
        marker_capacity=markers.marker_count,
        runtime=runtime,
    )
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_dt = float(config.dt_s) / float(config.solid_substeps)
    solid_damping = float(config.velocity_damping) ** (1.0 / float(config.solid_substeps))
    max_spacing = max(
        float(config.span_m) / float(config.grid_nodes[0]),
        float(config.duct_height_m) / float(config.grid_nodes[1]),
        float(config.duct_length_m) / float(config.grid_nodes[2]),
    )
    history: list[dict[str, Any]] = []
    preflow_history: list[dict[str, Any]] = []
    latest_report = None
    latest_preflow_load = None
    terminated_early = False
    termination_reason = ""
    preflow_count = max(0, int(preflow_steps))

    def solid_step() -> Any:
        latest = None
        for _ in range(config.solid_substeps):
            latest = solid.step(
                dt_s=solid_substep_dt,
                mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_UNUSED_REGION_ID,
                velocity_damping=solid_damping,
            )
            if config.enforce_plane_strain_x:
                solid.enforce_rest_x_plane()
        return latest

    def fixed_solid_step() -> dict[str, Any]:
        return {"preflow_fixed_solid": True}

    def stationary_fluid_load_step() -> Any:
        report = advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=fixed_solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=boundary.marker_pressure_neumann_gradient_field,
            search_radius_m=3.0 * max_spacing,
            interior_probe_distance_m=2.0 * max_spacing,
            mpm_support_radius_m=config.mpm_support_radius_m,
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_UNUSED_REGION_ID,
            fluid_dt_s=config.dt_s,
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=True,
            pressure_neumann_density_kgm3=config.air_density_kgm3,
            pressure_neumann_dt_s=float(config.dt_s),
            pressure_outlet_zmin=True,
            reset_pressure=False,
            pressure_solver=config.flow_pressure_solver,
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            cg_tolerance=config.flow_cg_tolerance,
            fluid_advection_scheme=str(fluid_advection_scheme),
            fluid_substeps=int(fluid_substeps),
            convert_internal_nodes_to_obstacles=bool(
                convert_internal_nodes_to_obstacles
            ),
            two_sided_probe_max_multiplier=two_sided_probe_max_multiplier,
            one_sided_pressure_region_id=one_sided_pressure_region_id,
            one_sided_reference_pressure_pa=one_sided_reference_pressure_pa,
            one_sided_probe_max_multiplier=one_sided_probe_max_multiplier,
            post_dirichlet_consistency_projection_iterations=1,
            update_surface_geometry_from_mpm=False,
            interpolate_velocity_dirichlet_with_interior=False,
        )
        return report.fluid_to_mpm_loads

    start = time.perf_counter()
    _write_json(
        PROCESS_PATH,
        {
            "status": "running",
            "requested_steps": step_count,
            "completed_steps": 0,
            "preflow_steps": preflow_count,
            "completed_preflow_steps": 0,
            "solver_path": "advance_hibm_mpm_sharp_mpm_step",
            "grid_nodes": list(config.grid_nodes),
            "solid_particle_counts": list(config.solid_particle_counts),
            "solid_substeps": int(config.solid_substeps),
            "mpm_support_radius_m": float(config.mpm_support_radius_m),
            "markers_per_face": int(markers_per_face),
        },
    )
    for preflow_index in range(preflow_count):
        preflow_start = time.perf_counter()
        latest_preflow_load = stationary_fluid_load_step()
        preflow_projection = latest_preflow_load.fluid_projection
        preflow_speed = _fluid_speed_metrics(fluid)
        preflow_velocity_dirichlet = latest_preflow_load.velocity_dirichlet
        preflow_pressure_neumann = latest_preflow_load.pressure_neumann
        preflow_no_slip = latest_preflow_load.no_slip_residual
        preflow_row = {
            "preflow_step": preflow_index + 1,
            "physical_time_s": 0.0,
            "stress_valid_marker_count": latest_preflow_load.fluid_stress.valid_marker_count,
            "stress_invalid_marker_count": latest_preflow_load.fluid_stress.invalid_marker_count,
            "marker_force_z_n": latest_preflow_load.marker_forces.total_marker_force_n[2],
            "scatter_active_particle_count": latest_preflow_load.mpm_force_scatter.active_particle_count,
            "projection_l2": float(preflow_projection.get("l2", 0.0)),
            "projection_max_abs": float(preflow_projection.get("max_abs", 0.0)),
            **preflow_speed,
            "velocity_dirichlet_max_abs_velocity_mps": (
                preflow_velocity_dirichlet.max_abs_velocity_mps
            ),
            "velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
                preflow_velocity_dirichlet.raw_reconstructed_max_abs_velocity_mps
            ),
            "velocity_dirichlet_boundary_velocity_only_rows": (
                preflow_velocity_dirichlet.boundary_velocity_only_row_count
            ),
            "pressure_neumann_max_abs_rhs": preflow_pressure_neumann.max_abs_rhs,
            "no_slip_max_residual_mps": preflow_no_slip.max_no_slip_residual_mps,
            "marker_face_diagnostics": _marker_face_diagnostics(markers, markers_per_face),
        }
        preflow_history.append(preflow_row)
        _write_json(
            PROCESS_PATH,
            {
                "status": "preflow",
                "requested_steps": step_count,
                "completed_steps": 0,
                "preflow_steps": preflow_count,
                "completed_preflow_steps": preflow_index + 1,
                "elapsed_s": time.perf_counter() - start,
                "last_preflow_step_elapsed_s": time.perf_counter() - preflow_start,
                "latest_preflow": preflow_row,
                "solver_path": "stationary_assemble_hibm_mpm_sharp_fluid_to_mpm_loads",
                "grid_nodes": list(config.grid_nodes),
                "solid_particle_counts": list(config.solid_particle_counts),
                "solid_substeps": int(config.solid_substeps),
                "mpm_support_radius_m": float(config.mpm_support_radius_m),
                "markers_per_face": int(markers_per_face),
            },
        )
        _log(
            f"preflow {preflow_index + 1}/{preflow_count} "
            f"valid={preflow_row['stress_valid_marker_count']}/{markers.marker_count} "
            f"Fz={preflow_row['marker_force_z_n']:.6e} "
            f"p999={preflow_row['fluid_speed_p999_mps']:.6e} "
            f"umax={preflow_row['fluid_speed_max_mps']:.6e} "
            f"vd={preflow_row['velocity_dirichlet_max_abs_velocity_mps']:.6e}"
        )
    for step_index in range(config.step_count):
        step_start = time.perf_counter()
        latest_report = advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=boundary.marker_pressure_neumann_gradient_field,
            search_radius_m=3.0 * max_spacing,
            interior_probe_distance_m=2.0 * max_spacing,
            mpm_support_radius_m=config.mpm_support_radius_m,
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_UNUSED_REGION_ID,
            fluid_dt_s=config.dt_s,
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=True,
            pressure_neumann_density_kgm3=config.air_density_kgm3,
            pressure_neumann_dt_s=float(config.dt_s),
            pressure_outlet_zmin=True,
            reset_pressure=False,
            pressure_solver=config.flow_pressure_solver,
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            cg_tolerance=config.flow_cg_tolerance,
            fluid_advection_scheme=str(fluid_advection_scheme),
            fluid_substeps=int(fluid_substeps),
            convert_internal_nodes_to_obstacles=bool(
                convert_internal_nodes_to_obstacles
            ),
            two_sided_probe_max_multiplier=two_sided_probe_max_multiplier,
            one_sided_pressure_region_id=one_sided_pressure_region_id,
            one_sided_reference_pressure_pa=one_sided_reference_pressure_pa,
            one_sided_probe_max_multiplier=one_sided_probe_max_multiplier,
            post_dirichlet_consistency_projection_iterations=1,
            update_surface_geometry_from_mpm=False,
            interpolate_velocity_dirichlet_with_interior=False,
        )
        load = latest_report.fluid_to_mpm_loads
        displacement = _solid_displacement(solid, masks)
        face_diagnostics = _marker_face_diagnostics(markers, markers_per_face)
        projection = load.fluid_projection
        field_speed = _fluid_speed_metrics(fluid)
        velocity_dirichlet = load.velocity_dirichlet
        pressure_neumann = load.pressure_neumann
        no_slip = load.no_slip_residual
        next_velocity_dirichlet = latest_report.next_velocity_dirichlet
        next_pressure_neumann = latest_report.next_pressure_neumann
        post_solid_no_slip = latest_report.post_solid_no_slip_residual
        surface_feedback = latest_report.surface_feedback
        row = {
            "step": step_index + 1,
            "time_s": (step_index + 1) * float(config.dt_s),
            "stress_valid_marker_count": load.fluid_stress.valid_marker_count,
            "stress_invalid_marker_count": load.fluid_stress.invalid_marker_count,
            "marker_force_z_n": load.marker_forces.total_marker_force_n[2],
            "scatter_active_particle_count": load.mpm_force_scatter.active_particle_count,
            "projection_l2": float(projection.get("l2", 0.0)),
            "projection_max_abs": float(projection.get("max_abs", 0.0)),
            **displacement,
            **field_speed,
            "velocity_dirichlet_max_abs_velocity_mps": (
                velocity_dirichlet.max_abs_velocity_mps
            ),
            "velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
                velocity_dirichlet.raw_reconstructed_max_abs_velocity_mps
            ),
            "velocity_dirichlet_boundary_velocity_only_rows": (
                velocity_dirichlet.boundary_velocity_only_row_count
            ),
            "pressure_neumann_max_abs_rhs": pressure_neumann.max_abs_rhs,
            "no_slip_max_residual_mps": no_slip.max_no_slip_residual_mps,
            "next_velocity_dirichlet_max_abs_velocity_mps": (
                next_velocity_dirichlet.max_abs_velocity_mps
            ),
            "next_velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps": (
                next_velocity_dirichlet.raw_reconstructed_max_abs_velocity_mps
            ),
            "next_velocity_dirichlet_boundary_velocity_only_rows": (
                next_velocity_dirichlet.boundary_velocity_only_row_count
            ),
            "next_pressure_neumann_max_abs_rhs": next_pressure_neumann.max_abs_rhs,
            "post_solid_no_slip_max_residual_mps": (
                0.0
                if post_solid_no_slip is None
                else post_solid_no_slip.max_no_slip_residual_mps
            ),
            "surface_feedback_max_marker_speed_mps": (
                surface_feedback.max_marker_speed_mps
            ),
            "marker_face_diagnostics": face_diagnostics,
        }
        history.append(row)
        _write_history(history)
        _write_json(
            PROCESS_PATH,
            {
                "status": "running",
                "requested_steps": step_count,
                "completed_steps": step_index + 1,
                "preflow_steps": preflow_count,
                "completed_preflow_steps": len(preflow_history),
                "elapsed_s": time.perf_counter() - start,
                "last_step_elapsed_s": time.perf_counter() - step_start,
                "latest_preflow": preflow_history[-1] if preflow_history else None,
                "latest": row,
                "solver_path": "advance_hibm_mpm_sharp_mpm_step",
                "grid_nodes": list(config.grid_nodes),
                "solid_particle_counts": list(config.solid_particle_counts),
                "solid_substeps": int(config.solid_substeps),
                "mpm_support_radius_m": float(config.mpm_support_radius_m),
                "markers_per_face": int(markers_per_face),
            },
        )
        _log(
            f"step {step_index + 1}/{config.step_count} "
            f"valid={row['stress_valid_marker_count']}/{markers.marker_count} "
            f"Fz={row['marker_force_z_n']:.6e} "
            f"disp={row['max_displacement_m']:.6e} "
            f"umax={row['fluid_speed_max_mps']:.6e} "
            f"vd={row['velocity_dirichlet_max_abs_velocity_mps']:.6e} "
            f"raw_vd={row['velocity_dirichlet_raw_reconstructed_max_abs_velocity_mps']:.6e}"
        )
        if (
            speed_stop_threshold is not None
            and float(row["fluid_speed_max_mps"]) > speed_stop_threshold
        ):
            terminated_early = True
            termination_reason = (
                "fluid_speed_max_mps_exceeded_threshold:"
                f"{float(row['fluid_speed_max_mps']):.6g}>{speed_stop_threshold:.6g}"
            )
            _log(f"stopping early: {termination_reason}")
            break

    if latest_report is None:
        raise RuntimeError("true sharp FSI run did not advance")
    final_displacement = _solid_displacement(solid, masks)
    final_speed = _fluid_speed_metrics(fluid)
    reference_comparison = _reference_comparison(
        config=config,
        max_displacement_m=float(final_displacement["max_displacement_m"]),
        fluid_speed_max_mps=float(final_speed["fluid_speed_max_mps"]),
        fluid_speed_p999_mps=float(final_speed["fluid_speed_p999_mps"]),
    )
    report = {
        "case": "full-domain-two-flap-true-sharp-fsi",
        "config": asdict(config),
        "solver_path": "advance_hibm_mpm_sharp_mpm_step",
        "full_domain_two_flap": True,
        "flap_count": 2,
        "mpm_support_radius_m": float(config.mpm_support_radius_m),
        "thin_wall_pressure_sampling": {
            **pressure_sampling_policy,
            "one_sided_pressure_region_id": one_sided_pressure_region_id,
            "two_sided_probe_max_multiplier": two_sided_probe_max_multiplier,
        },
        "fixed_end_definition": {
            "lower": "bottom root row fixed",
            "upper": "top root row fixed",
        },
        "preflow_steps": preflow_count,
        "preflow_history": preflow_history,
        "latest_preflow_load_report": latest_preflow_load,
        "history": history,
        "completed_steps": len(history),
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "latest_step_report": latest_report,
        "marker_face_diagnostics": _marker_face_diagnostics(markers, markers_per_face),
        "field_snapshot_npz": str(FIELDS_PATH),
        "field_speed_metrics": final_speed,
        "reference_comparison": reference_comparison,
        **final_displacement,
    }
    _write_field_snapshot(
        path=FIELDS_PATH,
        config=config,
        fluid=fluid,
        solid=solid,
        markers=markers,
        initial_marker_position_m=initial_marker_position_m,
    )
    REPORT_PATH.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_json(
        PROCESS_PATH,
        {
            "status": "stopped_physical_failure" if terminated_early else "completed",
            "requested_steps": step_count,
            "completed_steps": len(history),
            "preflow_steps": preflow_count,
            "completed_preflow_steps": len(preflow_history),
            "terminated_early": terminated_early,
            "termination_reason": termination_reason,
            "elapsed_s": time.perf_counter() - start,
            "report_json": str(REPORT_PATH),
            "history_csv": str(HISTORY_PATH),
            "field_snapshot_npz": str(FIELDS_PATH),
            "solver_path": "advance_hibm_mpm_sharp_mpm_step",
            "grid_nodes": list(config.grid_nodes),
            "solid_particle_counts": list(config.solid_particle_counts),
            "solid_substeps": int(config.solid_substeps),
            "mpm_support_radius_m": float(config.mpm_support_radius_m),
            "markers_per_face": int(markers_per_face),
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--projection-iterations", type=int, default=120)
    parser.add_argument("--grid-y", type=int, default=GRID_NODES[1])
    parser.add_argument("--grid-z", type=int, default=GRID_NODES[2])
    parser.add_argument("--solid-y", type=int, default=SOLID_PARTICLE_COUNTS[1])
    parser.add_argument("--solid-z", type=int, default=SOLID_PARTICLE_COUNTS[2])
    parser.add_argument("--solid-substeps", type=int, default=SOLID_SUBSTEPS)
    parser.add_argument("--preflow-steps", type=int, default=PREFLOW_STEPS)
    parser.add_argument("--markers-per-face", type=int, default=MARKERS_PER_FACE)
    parser.add_argument(
        "--pressure-solve-failure-policy",
        choices=("raise", "report"),
        default="raise",
    )
    parser.add_argument(
        "--disable-internal-obstacles",
        action="store_true",
        help="Diagnostic only: leave HIBM internal nodes fluid instead of solid obstacles.",
    )
    parser.add_argument("--advection-scheme", choices=("euler", "rk2"), default="rk2")
    parser.add_argument("--fluid-substeps", type=int, default=1)
    parser.add_argument(
        "--stop-on-speed-mps",
        type=float,
        default=0.0,
        help="Stop after writing diagnostics when the global speed exceeds this threshold.",
    )
    args = parser.parse_args(argv)
    try:
        LOG_PATH.write_text("", encoding="utf-8")
        report = run(
            args.steps,
            args.projection_iterations,
            grid_nodes=(GRID_NODES[0], args.grid_y, args.grid_z),
            solid_particle_counts=(SOLID_PARTICLE_COUNTS[0], args.solid_y, args.solid_z),
            markers_per_face=args.markers_per_face,
            pressure_solve_failure_policy=args.pressure_solve_failure_policy,
            convert_internal_nodes_to_obstacles=not args.disable_internal_obstacles,
            fluid_advection_scheme=args.advection_scheme,
            fluid_substeps=args.fluid_substeps,
            solid_substeps=args.solid_substeps,
            preflow_steps=args.preflow_steps,
            stop_on_speed_mps=args.stop_on_speed_mps,
        )
        latest = report["history"][-1]
        reference = report["reference_comparison"]
        failed_reasons = []
        if latest["stress_valid_marker_count"] <= 0 or abs(latest["marker_force_z_n"]) <= 0.0:
            failed_reasons.append("failed_no_valid_hibm_force")
        if report.get("terminated_early", False):
            failed_reasons.append(str(report.get("termination_reason", "failed_early_stop")))
        if not reference["velocity_peak_within_reference_range"]:
            failed_reasons.append("failed_reference_velocity_peak")
        if not reference["displacement_within_tolerance"]:
            failed_reasons.append("failed_reference_displacement")
        failed = bool(failed_reasons)
        if failed:
            _write_json(
                PROCESS_PATH,
                {
                    **json.loads(PROCESS_PATH.read_text(encoding="utf-8")),
                    "physical_status": "failed",
                    "physical_failure_reasons": failed_reasons,
                    "reference_comparison": reference,
                },
            )
        else:
            _write_json(
                PROCESS_PATH,
                {
                    **json.loads(PROCESS_PATH.read_text(encoding="utf-8")),
                    "physical_status": "passed",
                    "reference_comparison": reference,
                },
            )
        return 2 if failed else 0
    except Exception as exc:
        _write_json(
            PROCESS_PATH,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "preflow_steps": int(getattr(args, "preflow_steps", PREFLOW_STEPS)),
            },
        )
        _log(f"failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
