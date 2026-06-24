from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from simulation_core.fluid import CartesianFluidSolver, FluidDomainSpec
from simulation_core.hibm_mpm import HibmMpmSurfaceMarkers
from simulation_core.neo_hookean_mpm import NeoHookeanMpmState
from simulation_core.runtime import TaichiRuntimeConfig


PRIMARY_REGION_ID = 101
SECONDARY_UNUSED_REGION_ID = 202
STREAMWISE_AXIS_INDEX = 2
OUT_OF_PLANE_AXIS_INDEX = 0
AXIS_NAMES = ("x", "y", "z")
FLOW_SOLUTION_MODE = "computed_projection"


def run_rectangular_solid_marker_mpm_fsi_smoke(
    *,
    case_id: str,
    case_metadata: Mapping[str, Any],
    boundary_conditions: Mapping[str, Any],
    reference_results: Mapping[str, Any],
    config: Any,
) -> dict[str, object]:
    """Run a generic Cartesian fluid to rectangular solid MPM marker-FSI smoke."""
    _validate_rectangular_solid_config(config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = _build_fluid(config, runtime)
    flow_report = _solve_computed_flow(fluid, config)
    local_velocity_peak_mps = float(flow_report["local_velocity_peak_mps"])
    markers = _build_markers(config, runtime)
    solid = _build_solid(config, runtime)
    fixed_mask, tip_mask = _solid_masks(solid, config)
    mu_pa, lambda_pa = _lame_parameters(config)

    latest_stress_report = None
    latest_force_report = None
    latest_scatter_report = None
    latest_solid_report = None
    latest_feedback_report = None
    latest_feedback_flow_report = flow_report
    initial_flow_report = flow_report
    fluid_recompute_steps: list[int] = []
    history: list[dict[str, object]] = []

    for step_index in range(config.step_count):
        latest_stress_report = _sample_stress_to_marker_forces(markers, fluid)
        latest_force_report = markers.aggregate_region_forces(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_UNUSED_REGION_ID,
        )
        markers.clear_mpm_external_forces(
            solid.external_force_n,
            particle_count=solid.particle_count,
        )
        latest_scatter_report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=config.mpm_support_radius_m,
        )
        solid_substep_dt_s = config.dt_s / float(config.solid_substeps)
        solid_substep_velocity_damping = config.velocity_damping ** (
            1.0 / float(config.solid_substeps)
        )
        for _solid_substep in range(config.solid_substeps):
            latest_solid_report = solid.step(
                dt_s=solid_substep_dt_s,
                mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_UNUSED_REGION_ID,
                velocity_damping=solid_substep_velocity_damping,
            )
            if config.enforce_plane_strain_x:
                solid.enforce_rest_x_plane()
        latest_feedback_report = markers.update_surface_feedback_from_mpm_surface_particles(
            solid.x,
            solid.v,
            solid.surface_normal,
            solid.area_weight_m2,
            particle_count=solid.particle_count,
            support_radius_m=config.mpm_support_radius_m,
            dt_s=config.dt_s,
        )
        latest_feedback_flow_report = _recompute_flow_after_solid_feedback(
            fluid,
            solid,
            config,
        )
        fluid_recompute_steps.append(step_index + 1)
        step_displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
        history.append(
            {
                "step": step_index + 1,
                "stress_valid_marker_count": latest_stress_report.valid_marker_count,
                "scatter_invalid_marker_count": (
                    latest_scatter_report.invalid_marker_count
                ),
                "feedback_invalid_marker_count": (
                    latest_feedback_report.invalid_marker_count
                ),
                "total_marker_force_n": latest_force_report.total_marker_force_n,
                "mpm_external_force_n": latest_solid_report.external_force_n,
                "max_displacement_m": step_displacement["max_displacement_m"],
                "tip_mean_displacement_m": step_displacement[
                    "tip_mean_displacement_m"
                ],
                "fluid_recomputed_after_feedback": True,
                "fluid_recompute_step": step_index + 1,
                "post_feedback_local_velocity_peak_mps": (
                    latest_feedback_flow_report["local_velocity_peak_mps"]
                ),
                "post_feedback_pressure_min_pa": (
                    latest_feedback_flow_report["pressure_min_pa"]
                ),
                "post_feedback_pressure_max_pa": (
                    latest_feedback_flow_report["pressure_max_pa"]
                ),
                "post_feedback_obstacle_cell_count": (
                    latest_feedback_flow_report["obstacle_cell_count"]
                ),
                "post_feedback_fluid_cell_count": (
                    latest_feedback_flow_report["fluid_cell_count"]
                ),
            }
        )

    if (
        latest_stress_report is None
        or latest_force_report is None
        or latest_scatter_report is None
        or latest_solid_report is None
        or latest_feedback_report is None
    ):
        raise RuntimeError("rectangular solid marker-MPM FSI smoke did not advance")

    displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
    reference_displacement = float(reference_results["max_displacement_m"])
    reference_velocity_peak = float(reference_results["local_velocity_peak_mps"])
    max_displacement = float(displacement["max_displacement_m"])
    displacement_relative_error = (
        abs(max_displacement - reference_displacement) / reference_displacement
    )
    final_flow_report = latest_feedback_flow_report
    local_velocity_peak_mps = float(final_flow_report["local_velocity_peak_mps"])
    velocity_relative_error = (
        abs(local_velocity_peak_mps - reference_velocity_peak) / reference_velocity_peak
    )

    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        "computed_result_sources": {
            "pressure_pa": "fluid.pressure",
            "local_velocity_peak_mps": "max(norm(fluid.velocity))",
            "fluid_interface_force_n": "HIBM marker traction integral",
            "max_displacement_m": "solid.x-rest_x",
        },
        "boundary_conditions": dict(boundary_conditions),
        "reference_results": dict(reference_results),
        "flow_projection_report": final_flow_report["projection_report"],
        "initial_flow_projection_report": initial_flow_report["projection_report"],
        "final_flow_projection_report": final_flow_report["projection_report"],
        "fluid_recomputed_after_feedback": bool(fluid_recompute_steps),
        "fluid_recompute_count": len(fluid_recompute_steps),
        "fluid_recompute_steps": list(fluid_recompute_steps),
        "fluid_feedback_coupling_mode": "solid-particle-obstacle-reprojection",
        "flow_obstacle_cell_count": final_flow_report["obstacle_cell_count"],
        "flow_fluid_cell_count": final_flow_report["fluid_cell_count"],
        "computed_pressure_min_pa": final_flow_report["pressure_min_pa"],
        "computed_pressure_max_pa": final_flow_report["pressure_max_pa"],
        "pressure_sign_convention": final_flow_report["pressure_sign_convention"],
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "local_velocity_peak_relative_error": velocity_relative_error,
        "velocity_peak_tolerance": config.velocity_peak_tolerance,
        "stress_valid_marker_count": latest_stress_report.valid_marker_count,
        "stress_invalid_marker_count": latest_stress_report.invalid_marker_count,
        "two_sided_pressure_marker_count": (
            latest_stress_report.two_sided_pressure_marker_count
        ),
        "max_abs_traction_pa": latest_stress_report.max_abs_traction_pa,
        "total_marker_force_n": latest_force_report.total_marker_force_n,
        "fluid_reaction_force_n": latest_force_report.fluid_reaction_force_n,
        "marker_action_reaction_residual_n": (
            latest_force_report.action_reaction_residual_n
        ),
        "scatter_invalid_marker_count": latest_scatter_report.invalid_marker_count,
        "scatter_active_marker_count": latest_scatter_report.active_marker_count,
        "scatter_active_particle_count": latest_scatter_report.active_particle_count,
        "scatter_action_reaction_residual_n": (
            latest_scatter_report.action_reaction_residual_n
        ),
        "mpm_external_force_n": latest_solid_report.external_force_n,
        "surface_feedback_updated_marker_count": (
            latest_feedback_report.updated_marker_count
        ),
        "surface_feedback_invalid_marker_count": (
            latest_feedback_report.invalid_marker_count
        ),
        "surface_feedback_max_marker_displacement_m": (
            latest_feedback_report.max_marker_displacement_m
        ),
        "history": history,
        "max_displacement_m": max_displacement,
        "reference_max_displacement_m": reference_displacement,
        "max_displacement_relative_error": displacement_relative_error,
        "displacement_tolerance": config.displacement_tolerance,
        **displacement,
    }


def _validate_rectangular_solid_config(config: Any) -> None:
    if config.step_count <= 0:
        raise ValueError("step_count must be positive")
    if min(config.grid_nodes) < 4:
        raise ValueError("grid_nodes must be at least 4 in each direction")
    if min(config.solid_particle_counts) <= 0:
        raise ValueError("solid_particle_counts must be positive")
    if config.marker_count <= 0:
        raise ValueError("marker_count must be positive")
    if config.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if config.solid_substeps <= 0:
        raise ValueError("solid_substeps must be positive")
    if config.flow_projection_iterations <= 0:
        raise ValueError("flow_projection_iterations must be positive")
    if config.flow_cg_tolerance < 0.0:
        raise ValueError("flow_cg_tolerance must be non-negative")
    if config.flow_divergence_cleanup_iterations < 0:
        raise ValueError("flow_divergence_cleanup_iterations must be non-negative")
    if config.displacement_tolerance <= 0.0:
        raise ValueError("displacement_tolerance must be positive")
    if config.velocity_peak_tolerance <= 0.0:
        raise ValueError("velocity_peak_tolerance must be positive")
    if not (0.0 < config.poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in (0, 0.5)")


def _domain_bounds(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (0.0, 0.0, 0.0),
        (config.span_m, 0.5 * config.duct_height_m, config.duct_length_m),
    )


def _solid_box(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center_z = 0.5 * config.duct_length_m
    root_y = 0.0
    return (
        (
            0.15 * config.span_m,
            root_y,
            center_z - 0.5 * config.flap_thickness_m,
        ),
        (
            0.85 * config.span_m,
            root_y + config.flap_height_m,
            center_z + 0.5 * config.flap_thickness_m,
        ),
    )


def _lame_parameters(config: Any) -> tuple[float, float]:
    young = float(config.young_modulus_pa)
    nu = float(config.poisson_ratio)
    mu = young / (2.0 * (1.0 + nu))
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _build_fluid(config: Any, runtime: TaichiRuntimeConfig) -> CartesianFluidSolver:
    bounds_min, bounds_max = _domain_bounds(config)
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
    fluid.obstacle.from_numpy(_solid_obstacle(config))
    return fluid


def _cell_interval_overlaps(
    cell_min: float,
    cell_max: float,
    box_min: float,
    box_max: float,
) -> bool:
    return cell_min < box_max and cell_max > box_min


def _solid_obstacle(config: Any) -> np.ndarray:
    solid_min, solid_max = _solid_box(config)
    return _solid_obstacle_from_box(config, solid_min, solid_max)


def _solid_obstacle_from_box(
    config: Any,
    solid_min: tuple[float, float, float],
    solid_max: tuple[float, float, float],
) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    bounds_min, bounds_max = _domain_bounds(config)
    dx = (bounds_max[0] - bounds_min[0]) / nx
    dy = (bounds_max[1] - bounds_min[1]) / ny
    dz = (bounds_max[2] - bounds_min[2]) / nz
    obstacle = np.zeros((nx, ny, nz), dtype=np.int32)
    for i in range(nx):
        x_min = bounds_min[0] + i * dx
        x_max = x_min + dx
        x_overlaps = _cell_interval_overlaps(x_min, x_max, solid_min[0], solid_max[0])
        for j in range(ny):
            y_min = bounds_min[1] + j * dy
            y_max = y_min + dy
            y_overlaps = _cell_interval_overlaps(y_min, y_max, solid_min[1], solid_max[1])
            for k in range(nz):
                z_min = bounds_min[2] + k * dz
                z_max = z_min + dz
                if (
                    x_overlaps
                    and y_overlaps
                    and _cell_interval_overlaps(z_min, z_max, solid_min[2], solid_max[2])
                ):
                    obstacle[i, j, k] = 1
    return obstacle


def _solid_obstacle_from_current_particles(
    solid: NeoHookeanMpmState,
    config: Any,
) -> np.ndarray:
    positions = solid.x.to_numpy()[: solid.particle_count]
    if positions.size == 0:
        raise RuntimeError("cannot rebuild fluid obstacle from an empty solid")
    current_min = positions.min(axis=0)
    current_max = positions.max(axis=0)
    solid_min, solid_max = _solid_box(config)
    half_padding = np.asarray(
        [
            0.5 * (solid_max[0] - solid_min[0]) / float(config.solid_particle_counts[0]),
            0.5 * (solid_max[1] - solid_min[1]) / float(config.solid_particle_counts[1]),
            0.5 * (solid_max[2] - solid_min[2]) / float(config.solid_particle_counts[2]),
        ],
        dtype=np.float64,
    )
    bounds_min, bounds_max = _domain_bounds(config)
    padded_min = np.maximum(current_min - half_padding, np.asarray(bounds_min))
    padded_max = np.minimum(current_max + half_padding, np.asarray(bounds_max))
    return _solid_obstacle_from_box(
        config,
        tuple(float(v) for v in padded_min),
        tuple(float(v) for v in padded_max),
    )


def _initialize_inlet_flow(
    fluid: CartesianFluidSolver,
    config: Any,
) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    obstacle = fluid.obstacle.to_numpy()
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    velocity[:, :, :, STREAMWISE_AXIS_INDEX] = -float(config.inlet_velocity_mps)
    velocity[obstacle != 0] = 0.0
    fluid.velocity.from_numpy(velocity)
    fluid.velocity_prev.from_numpy(velocity)

    active = np.zeros((nx, ny, nz), dtype=np.int32)
    active[:, :, nz - 1] = 1
    values = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    values[:, :, nz - 1, STREAMWISE_AXIS_INDEX] = -float(config.inlet_velocity_mps)
    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(
        active.astype(np.float32)
    )
    fluid.pressure.from_numpy(np.zeros((nx, ny, nz), dtype=np.float32))
    return obstacle


def _solve_computed_flow(
    fluid: CartesianFluidSolver,
    config: Any,
) -> dict[str, object]:
    obstacle = _initialize_inlet_flow(fluid, config)
    projection_report = fluid.project(
        iterations=config.flow_projection_iterations,
        pressure_outlet_zmin=True,
        reset_pressure=True,
        pressure_solver=config.flow_pressure_solver,
        cg_tolerance=config.flow_cg_tolerance,
        divergence_cleanup_iterations=config.flow_divergence_cleanup_iterations,
    )
    velocity = fluid.velocity.to_numpy()
    pressure = fluid.pressure.to_numpy()
    non_obstacle = obstacle == 0
    speed = np.linalg.norm(velocity, axis=3)
    return {
        "mode": FLOW_SOLUTION_MODE,
        "projection_report": projection_report,
        "obstacle_cell_count": int(obstacle.sum()),
        "fluid_cell_count": int(non_obstacle.sum()),
        "local_velocity_peak_mps": float(speed[non_obstacle].max(initial=0.0)),
        "pressure_min_pa": float(pressure[non_obstacle].min(initial=0.0)),
        "pressure_max_pa": float(pressure[non_obstacle].max(initial=0.0)),
        "pressure_sign_convention": "fluid.pressure projection field is sampled directly",
    }


def _recompute_flow_after_solid_feedback(
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: Any,
) -> dict[str, object]:
    fluid.obstacle.from_numpy(_solid_obstacle_from_current_particles(solid, config))
    return _solve_computed_flow(fluid, config)


def _build_markers(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> HibmMpmSurfaceMarkers:
    markers = HibmMpmSurfaceMarkers(marker_capacity=config.marker_count, runtime=runtime)
    solid_min, solid_max = _solid_box(config)
    x_center = 0.5 * (solid_min[0] + solid_max[0])
    segment = config.flap_height_m / config.marker_count
    area = config.flap_height_m * (solid_max[0] - solid_min[0]) / config.marker_count
    positions = []
    velocities = []
    normals = []
    areas = []
    regions = []
    for marker in range(config.marker_count):
        y = solid_min[1] + (float(marker) + 0.5) * segment
        positions.append((x_center, y, solid_max[2]))
        velocities.append((0.0, 0.0, 0.0))
        normals.append((0.0, 0.0, 1.0))
        areas.append(area)
        regions.append(PRIMARY_REGION_ID)
    markers.load_markers(
        positions_m=positions,
        velocities_mps=velocities,
        normals=normals,
        areas_m2=areas,
        region_ids=regions,
    )
    return markers


def _build_solid(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> NeoHookeanMpmState:
    bounds_min, bounds_max = _domain_bounds(config)
    capacity = math.prod(config.solid_particle_counts)
    solid = NeoHookeanMpmState(
        particle_capacity=capacity,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        grid_nodes=config.grid_nodes,
        runtime=runtime,
    )
    solid_min, solid_max = _solid_box(config)
    solid.initialize_box(
        particle_counts=config.solid_particle_counts,
        box_min_m=solid_min,
        box_max_m=solid_max,
        density_kgm3=config.solid_density_kgm3,
    )
    _configure_solid_fields(solid, config)
    return solid


def _configure_solid_fields(
    solid: NeoHookeanMpmState,
    config: Any,
) -> None:
    particle_count = int(solid.particle_count)
    positions = solid.x.to_numpy()
    normals = np.zeros((solid.particle_capacity, 3), dtype=np.float32)
    areas = np.zeros((solid.particle_capacity,), dtype=np.float32)
    region_ids = np.zeros((solid.particle_capacity,), dtype=np.int32)
    fixed = np.zeros((solid.particle_capacity,), dtype=np.int32)

    solid_min, solid_max = _solid_box(config)
    root_row_height = config.flap_height_m / float(config.solid_particle_counts[1])
    root_limit = solid_min[1] + 1.01 * root_row_height
    particle_area = config.flap_height_m * (solid_max[0] - solid_min[0]) / max(
        float(particle_count),
        1.0,
    )
    for particle in range(particle_count):
        region_ids[particle] = PRIMARY_REGION_ID
        normals[particle] = (0.0, 0.0, 1.0)
        areas[particle] = particle_area
        if positions[particle, 1] <= root_limit:
            fixed[particle] = 1

    solid.region_id.from_numpy(region_ids)
    solid.fixed_particle.from_numpy(fixed)
    solid.surface_normal.from_numpy(normals)
    solid.rest_surface_normal.from_numpy(normals)
    solid.area_weight_m2.from_numpy(areas)
    solid.rest_area_weight_m2.from_numpy(areas)


def _sample_stress_to_marker_forces(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
) -> Any:
    report = markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        fluid.pressure,
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        fluid.grid.grid_nodes,
        viscosity_pa_s=0.0,
        two_sided_pressure=True,
    )
    markers.compute_marker_forces()
    return report


def _solid_displacement_report(
    solid: NeoHookeanMpmState,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
) -> dict[str, object]:
    positions = solid.x.to_numpy()[: solid.particle_count]
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    displacement = positions - rest
    norms = np.linalg.norm(displacement, axis=1)
    tip_displacement = displacement[tip_mask]
    if tip_displacement.size == 0:
        raise RuntimeError("tip particle mask is empty")
    root_norms = norms[fixed_mask]
    return {
        "max_displacement_m": float(norms.max(initial=0.0)),
        "tip_mean_displacement_m": tuple(float(v) for v in tip_displacement.mean(axis=0)),
        "tip_displacement_norm_m": float(np.linalg.norm(tip_displacement.mean(axis=0))),
        "root_max_displacement_m": float(root_norms.max(initial=0.0)),
    }


def _solid_masks(
    solid: NeoHookeanMpmState,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    fixed = solid.fixed_particle.to_numpy()[: solid.particle_count] != 0
    _, solid_max = _solid_box(config)
    tip_row_height = config.flap_height_m / float(config.solid_particle_counts[1])
    tip_mask = rest[:, 1] >= solid_max[1] - 1.01 * tip_row_height
    return fixed, tip_mask
