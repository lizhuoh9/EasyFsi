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
DEFAULT_SOLID_CFL_TARGET = 0.5


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
    _initialize_computed_flow(fluid, config)
    markers = _build_markers(config, runtime)
    solid = _build_solid(config, runtime)
    fixed_mask, tip_mask = _solid_masks(solid, config)
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_cfl = solid_substep_cfl_report(config)
    solid_substeps = int(solid_substep_cfl["solid_substeps_selected"])

    latest_stress_report = None
    latest_force_report = None
    latest_scatter_report = None
    latest_solid_report = None
    latest_feedback_report = None
    latest_flow_report = None
    latest_feedback_constraint_report = None
    fluid_projection_count = 0
    fluid_projection_after_feedback_count = 0
    fluid_projection_consumed_feedback_count = 0
    feedback_available_for_projection = False
    feedback_constraint_cells: set[tuple[int, int, int]] = set()
    history: list[dict[str, object]] = []

    for step_index in range(config.step_count):
        feedback_available_before_projection = feedback_available_for_projection
        latest_feedback_constraint_report = _apply_marker_feedback_to_fluid(
            markers,
            fluid,
            config,
            feedback_available=feedback_available_before_projection,
            previous_feedback_constraint_cells=feedback_constraint_cells,
        )
        feedback_constraint_cells = latest_feedback_constraint_report["_feedback_constraint_cells"]
        latest_flow_report = _project_current_flow(
            fluid,
            config,
            reset_pressure=(step_index == 0),
        )
        latest_feedback_constraint_report[
            "no_slip_projected_residual_after_projection_mps"
        ] = _measure_projected_no_slip_residual(
            markers,
            fluid,
            config,
            feedback_consumed=bool(
                latest_feedback_constraint_report[
                    "fluid_projection_consumed_feedback"
                ]
            ),
        )
        fluid_projection_count += 1
        if feedback_available_before_projection:
            fluid_projection_after_feedback_count += 1
        if latest_feedback_constraint_report["fluid_projection_consumed_feedback"]:
            fluid_projection_consumed_feedback_count += 1
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
        solid_substep_dt_s = config.dt_s / float(solid_substeps)
        solid_substep_velocity_damping = config.velocity_damping ** (
            1.0 / float(solid_substeps)
        )
        for _solid_substep in range(solid_substeps):
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
        feedback_available_for_projection = True
        step_displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
        history.append(
            {
                "step": step_index + 1,
                "fluid_recomputed": True,
                "fluid_recomputed_after_feedback": (
                    feedback_available_before_projection
                ),
                "feedback_available_before_projection": (
                    feedback_available_before_projection
                ),
                "fluid_projection_consumed_feedback": (
                    latest_feedback_constraint_report[
                        "fluid_projection_consumed_feedback"
                    ]
                ),
                "fluid_feedback_constraint_marker_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_marker_count"
                    ]
                ),
                "fluid_feedback_constraint_active_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_active_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_cleared_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_cleared_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_obstacle_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_obstacle_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_non_obstacle_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_non_obstacle_cell_count"
                    ]
                ),
                "fluid_feedback_constraint_projection_participating_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_feedback_constraint_projection_participating_cell_count"
                    ]
                ),
                "no_slip_residual_before_mps": latest_feedback_constraint_report[
                    "no_slip_residual_before_mps"
                ],
                "no_slip_residual_after_mps": latest_feedback_constraint_report[
                    "no_slip_residual_after_mps"
                ],
                "no_slip_target_residual_after_assembly_mps": (
                    latest_feedback_constraint_report[
                        "no_slip_target_residual_after_assembly_mps"
                    ]
                ),
                "no_slip_projected_residual_after_projection_mps": (
                    latest_feedback_constraint_report[
                        "no_slip_projected_residual_after_projection_mps"
                    ]
                ),
                "local_velocity_peak_mps": latest_flow_report[
                    "local_velocity_peak_mps"
                ],
                "pressure_min_pa": latest_flow_report["pressure_min_pa"],
                "pressure_max_pa": latest_flow_report["pressure_max_pa"],
                "flow_projection_report": latest_flow_report["projection_report"],
                "solid_substeps_selected": solid_substeps,
                "solid_estimated_cfl": solid_substep_cfl["solid_estimated_cfl"],
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
            }
        )

    if (
        latest_stress_report is None
        or latest_force_report is None
        or latest_scatter_report is None
        or latest_solid_report is None
        or latest_feedback_report is None
        or latest_flow_report is None
        or latest_feedback_constraint_report is None
    ):
        raise RuntimeError("rectangular solid marker-MPM FSI smoke did not advance")

    displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
    reference_displacement = float(reference_results["max_displacement_m"])
    reference_velocity_peak = float(reference_results["local_velocity_peak_mps"])
    max_displacement = float(displacement["max_displacement_m"])
    local_velocity_peak_mps = float(latest_flow_report["local_velocity_peak_mps"])
    displacement_relative_error = (
        abs(max_displacement - reference_displacement) / reference_displacement
    )
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
        "official_half_domain": _is_official_half_domain(case_metadata),
        "full_domain_two_flap": False,
        "flap_count_modeled": 1,
        "flap_count_displayed_after_symmetry_mirror": (
            2 if _is_official_half_domain(case_metadata) else 1
        ),
        "modeled_grid_nodes": list(config.grid_nodes),
        "display_grid_after_symmetry_mirror": _display_grid_after_symmetry_mirror(
            config,
            case_metadata,
        ),
        "flap_box_m": {
            "min": list(_solid_box(config)[0]),
            "max": list(_solid_box(config)[1]),
        },
        "marker_face_count": 2,
        "marker_count_per_face": int(config.marker_count),
        "marker_count_actual": int(markers.marker_count),
        "flow_projection_iterations_actual": int(config.flow_projection_iterations),
        "solid_substep_cfl_report": solid_substep_cfl,
        "solid_substeps_requested": solid_substep_cfl["solid_substeps_requested"],
        "solid_substeps_selected": solid_substep_cfl["solid_substeps_selected"],
        "solid_substeps_cfl_minimum": solid_substep_cfl[
            "solid_substeps_cfl_minimum"
        ],
        "solid_estimated_cfl": solid_substep_cfl["solid_estimated_cfl"],
        "solid_elastic_wave_speed_mps": solid_substep_cfl[
            "solid_elastic_wave_speed_mps"
        ],
        "solid_min_grid_spacing_m": solid_substep_cfl["solid_min_grid_spacing_m"],
        "solid_cfl_target": solid_substep_cfl["solid_cfl_target"],
        "computed_result_sources": {
            "pressure_pa": "fluid.pressure",
            "local_velocity_peak_mps": "max(norm(fluid.velocity))",
            "fluid_interface_force_n": "HIBM marker traction integral",
            "max_displacement_m": "solid.x-rest_x",
        },
        "boundary_conditions": dict(boundary_conditions),
        "reference_results": dict(reference_results),
        "flow_projection_report": latest_flow_report["projection_report"],
        "flow_obstacle_cell_count": latest_flow_report["obstacle_cell_count"],
        "flow_fluid_cell_count": latest_flow_report["fluid_cell_count"],
        "computed_pressure_min_pa": latest_flow_report["pressure_min_pa"],
        "computed_pressure_max_pa": latest_flow_report["pressure_max_pa"],
        "pressure_sign_convention": latest_flow_report["pressure_sign_convention"],
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "local_velocity_peak_relative_error": velocity_relative_error,
        "velocity_peak_tolerance": config.velocity_peak_tolerance,
        "fluid_recomputed_after_feedback": (
            fluid_projection_after_feedback_count > 0
        ),
        "feedback_closure_status": (
            "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK"
            if fluid_projection_after_feedback_count > 0
            else "OPEN_LOOP_OR_PREFEEDBACK_ONLY"
        ),
        "fluid_recompute_count": fluid_projection_count,
        "fluid_projection_count": fluid_projection_count,
        "fluid_projection_after_feedback_count": (
            fluid_projection_after_feedback_count
        ),
        "fluid_projection_consumed_feedback_count": (
            fluid_projection_consumed_feedback_count
        ),
        "fluid_projection_consumed_feedback": latest_feedback_constraint_report[
            "fluid_projection_consumed_feedback"
        ],
        "fluid_feedback_constraint_marker_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_marker_count"
            ]
        ),
        "fluid_feedback_constraint_active_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_active_cell_count"
            ]
        ),
        "fluid_feedback_constraint_cleared_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_cleared_cell_count"
            ]
        ),
        "fluid_feedback_constraint_obstacle_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_obstacle_cell_count"
            ]
        ),
        "fluid_feedback_constraint_non_obstacle_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_non_obstacle_cell_count"
            ]
        ),
        "fluid_feedback_constraint_projection_participating_cell_count": (
            latest_feedback_constraint_report[
                "fluid_feedback_constraint_projection_participating_cell_count"
            ]
        ),
        "no_slip_residual_before_mps": latest_feedback_constraint_report[
            "no_slip_residual_before_mps"
        ],
        "no_slip_residual_after_mps": latest_feedback_constraint_report[
            "no_slip_residual_after_mps"
        ],
        "no_slip_target_residual_after_assembly_mps": (
            latest_feedback_constraint_report[
                "no_slip_target_residual_after_assembly_mps"
            ]
        ),
        "no_slip_projected_residual_after_projection_mps": (
            latest_feedback_constraint_report[
                "no_slip_projected_residual_after_projection_mps"
            ]
        ),
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
    if float(getattr(config, "solid_cfl_target", DEFAULT_SOLID_CFL_TARGET)) <= 0.0:
        raise ValueError("solid_cfl_target must be positive")
    flap_streamwise_min_m = getattr(config, "flap_streamwise_min_m", None)
    flap_streamwise_max_m = getattr(config, "flap_streamwise_max_m", None)
    if (flap_streamwise_min_m is None) != (flap_streamwise_max_m is None):
        raise ValueError("flap streamwise bounds must be configured as a pair")
    if flap_streamwise_min_m is not None:
        if (
            float(flap_streamwise_min_m) < 0.0
            or float(flap_streamwise_max_m) > float(config.duct_length_m)
            or float(flap_streamwise_min_m) >= float(flap_streamwise_max_m)
        ):
            raise ValueError("flap streamwise bounds must lie inside the duct")
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
    z_min = getattr(config, "flap_streamwise_min_m", None)
    z_max = getattr(config, "flap_streamwise_max_m", None)
    if z_min is None or z_max is None:
        z_min = center_z - 0.5 * config.flap_thickness_m
        z_max = center_z + 0.5 * config.flap_thickness_m
    root_y = 0.0
    return (
        (
            0.0,
            root_y,
            float(z_min),
        ),
        (
            config.span_m,
            root_y + config.flap_height_m,
            float(z_max),
        ),
    )


def _lame_parameters(config: Any) -> tuple[float, float]:
    young = float(config.young_modulus_pa)
    nu = float(config.poisson_ratio)
    mu = young / (2.0 * (1.0 + nu))
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def solid_substep_cfl_report(config: Any) -> dict[str, object]:
    mu, lam = _lame_parameters(config)
    wave_speed_mps = math.sqrt(
        (lam + 2.0 * mu) / float(config.solid_density_kgm3)
    )
    min_spacing_m = min(_grid_spacing_m(config))
    cfl_target = float(getattr(config, "solid_cfl_target", DEFAULT_SOLID_CFL_TARGET))
    requested_substeps = int(config.solid_substeps)
    cfl_minimum = max(
        1,
        int(
            math.ceil(
                wave_speed_mps
                * float(config.dt_s)
                / (cfl_target * min_spacing_m)
            )
        ),
    )
    selected_substeps = max(requested_substeps, cfl_minimum)
    substep_dt_s = float(config.dt_s) / float(selected_substeps)
    estimated_cfl = wave_speed_mps * substep_dt_s / min_spacing_m
    return {
        "solid_substeps_requested": requested_substeps,
        "solid_substeps_cfl_minimum": cfl_minimum,
        "solid_substeps_selected": selected_substeps,
        "solid_substeps_auto_applied": selected_substeps != requested_substeps,
        "solid_elastic_wave_speed_mps": wave_speed_mps,
        "solid_min_grid_spacing_m": min_spacing_m,
        "solid_cfl_target": cfl_target,
        "solid_estimated_cfl": estimated_cfl,
        "solid_substep_dt_s": substep_dt_s,
    }


def _grid_spacing_m(config: Any) -> tuple[float, float, float]:
    bounds_min, bounds_max = _domain_bounds(config)
    return tuple(
        (float(bounds_max[axis]) - float(bounds_min[axis]))
        / float(config.grid_nodes[axis])
        for axis in range(3)
    )


def _is_official_half_domain(case_metadata: Mapping[str, Any]) -> bool:
    geometry = case_metadata.get("geometry", {})
    if not isinstance(geometry, Mapping):
        return False
    return geometry.get("modeled_domain") == "lower-symmetry-half"


def _display_grid_after_symmetry_mirror(
    config: Any,
    case_metadata: Mapping[str, Any],
) -> list[int]:
    grid = list(config.grid_nodes)
    if _is_official_half_domain(case_metadata):
        grid[1] *= 2
    return grid


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
    nx, ny, nz = config.grid_nodes
    bounds_min, bounds_max = _domain_bounds(config)
    solid_min, solid_max = _solid_box(config)
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


def _initialize_computed_flow(
    fluid: CartesianFluidSolver,
    config: Any,
) -> np.ndarray:
    return _initialize_inlet_flow(fluid, config)


def _project_current_flow(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    reset_pressure: bool,
) -> dict[str, object]:
    projection_report = fluid.project(
        iterations=config.flow_projection_iterations,
        pressure_outlet_zmin=True,
        reset_pressure=reset_pressure,
        pressure_solver=config.flow_pressure_solver,
        cg_tolerance=config.flow_cg_tolerance,
        divergence_cleanup_iterations=config.flow_divergence_cleanup_iterations,
    )
    return _flow_state_report(fluid, projection_report)


def _apply_marker_feedback_to_fluid(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_available: bool,
    previous_feedback_constraint_cells: set[tuple[int, int, int]],
) -> dict[str, object]:
    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    weights = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()

    cleared_cell_count = 0
    for i, j, k in previous_feedback_constraint_cells:
        active[i, j, k] = 0
        values[i, j, k] = 0.0
        weights[i, j, k] = 0.0
        cleared_cell_count += 1

    if not feedback_available:
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        return _empty_feedback_constraint_report(cleared_cell_count)

    marker_count = int(markers.marker_count)
    if marker_count <= 0:
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        return _empty_feedback_constraint_report(cleared_cell_count)

    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_velocities = markers.v_gamma_mps.to_numpy()[:marker_count]
    velocity = fluid.velocity.to_numpy()

    marker_cells = _marker_grid_cells(marker_positions, config)

    target_sum: dict[tuple[int, int, int], np.ndarray] = {}
    target_count: dict[tuple[int, int, int], int] = {}
    before_residuals: list[float] = []
    for cell, marker_velocity in zip(marker_cells, marker_velocities):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        key = (i, j, k)
        target_sum[key] = target_sum.get(key, np.zeros(3, dtype=np.float64)) + np.asarray(
            marker_velocity,
            dtype=np.float64,
        )
        target_count[key] = target_count.get(key, 0) + 1
        before_residuals.append(float(np.linalg.norm(velocity[i, j, k] - marker_velocity)))

    for (i, j, k), summed_velocity in target_sum.items():
        active[i, j, k] = 1
        values[i, j, k] = summed_velocity / float(target_count[(i, j, k)])
        weights[i, j, k] = 1.0

    after_residuals: list[float] = []
    for cell, marker_velocity in zip(marker_cells, marker_velocities):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        after_residuals.append(float(np.linalg.norm(values[i, j, k] - marker_velocity)))

    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)

    obstacle = fluid.obstacle.to_numpy()
    active_cell_count = len(target_sum)
    obstacle_cell_count = sum(1 for i, j, k in target_sum if obstacle[i, j, k] != 0)
    non_obstacle_cell_count = active_cell_count - obstacle_cell_count
    return {
        "fluid_projection_consumed_feedback": active_cell_count > 0,
        "fluid_feedback_constraint_marker_count": marker_count,
        "fluid_feedback_constraint_active_cell_count": active_cell_count,
        "fluid_feedback_constraint_cleared_cell_count": cleared_cell_count,
        "fluid_feedback_constraint_obstacle_cell_count": obstacle_cell_count,
        "fluid_feedback_constraint_non_obstacle_cell_count": non_obstacle_cell_count,
        "fluid_feedback_constraint_projection_participating_cell_count": (
            non_obstacle_cell_count
        ),
        "no_slip_residual_before_mps": max(before_residuals, default=0.0),
        "no_slip_residual_after_mps": max(after_residuals, default=0.0),
        "no_slip_target_residual_after_assembly_mps": max(
            after_residuals,
            default=0.0,
        ),
        "no_slip_projected_residual_after_projection_mps": 0.0,
        "_feedback_constraint_cells": set(target_sum),
    }


def _empty_feedback_constraint_report(
    cleared_cell_count: int = 0,
) -> dict[str, object]:
    return {
        "fluid_projection_consumed_feedback": False,
        "fluid_feedback_constraint_marker_count": 0,
        "fluid_feedback_constraint_active_cell_count": 0,
        "fluid_feedback_constraint_cleared_cell_count": cleared_cell_count,
        "fluid_feedback_constraint_obstacle_cell_count": 0,
        "fluid_feedback_constraint_non_obstacle_cell_count": 0,
        "fluid_feedback_constraint_projection_participating_cell_count": 0,
        "no_slip_residual_before_mps": "",
        "no_slip_residual_after_mps": "",
        "no_slip_target_residual_after_assembly_mps": "",
        "no_slip_projected_residual_after_projection_mps": 0.0,
        "_feedback_constraint_cells": set(),
    }


def _measure_projected_no_slip_residual(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_consumed: bool,
) -> float:
    if not feedback_consumed:
        return 0.0

    marker_count = int(markers.marker_count)
    if marker_count <= 0:
        return 0.0

    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_velocities = markers.v_gamma_mps.to_numpy()[:marker_count]
    marker_cells = _marker_grid_cells(marker_positions, config)
    velocity = fluid.velocity.to_numpy()

    residuals = []
    for cell, marker_velocity in zip(marker_cells, marker_velocities):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        residuals.append(float(np.linalg.norm(velocity[i, j, k] - marker_velocity)))
    return max(residuals, default=0.0)


def _marker_grid_cells(
    marker_positions: np.ndarray,
    config: Any,
) -> np.ndarray:
    bounds_min, bounds_max = _domain_bounds(config)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    grid_nodes = np.asarray(config.grid_nodes, dtype=np.int32)
    cell_width = (upper - lower) / grid_nodes.astype(np.float64)
    marker_cells = np.floor((marker_positions - lower) / cell_width).astype(np.int32)
    return np.clip(marker_cells, 0, grid_nodes - 1)


def _flow_state_report(
    fluid: CartesianFluidSolver,
    projection_report: Any,
) -> dict[str, object]:
    obstacle = fluid.obstacle.to_numpy()
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


def _build_markers(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> HibmMpmSurfaceMarkers:
    markers_per_face = int(config.marker_count)
    markers = HibmMpmSurfaceMarkers(
        marker_capacity=2 * markers_per_face,
        runtime=runtime,
    )
    solid_min, solid_max = _solid_box(config)
    x_center = 0.5 * (solid_min[0] + solid_max[0])
    segment = config.flap_height_m / markers_per_face
    area = config.flap_height_m * (solid_max[0] - solid_min[0]) / markers_per_face
    dz = _grid_spacing_m(config)[2]
    face_specs = (
        (solid_max[2] + 0.51 * dz, (0.0, 0.0, 1.0)),
        (solid_min[2] - 0.51 * dz, (0.0, 0.0, -1.0)),
    )
    positions = []
    velocities = []
    normals = []
    areas = []
    regions = []
    for z, normal in face_specs:
        for marker in range(markers_per_face):
            y = solid_min[1] + (float(marker) + 0.5) * segment
            positions.append((x_center, y, z))
            velocities.append((0.0, 0.0, 0.0))
            normals.append(normal)
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
    mid_z = 0.5 * (solid_min[2] + solid_max[2])
    particle_area = config.flap_height_m * (solid_max[0] - solid_min[0]) / max(
        float(particle_count),
        1.0,
    )
    for particle in range(particle_count):
        region_ids[particle] = PRIMARY_REGION_ID
        normals[particle] = (
            0.0,
            0.0,
            -1.0 if positions[particle, 2] < mid_z else 1.0,
        )
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
