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
SECONDARY_REGION_ID = 202
SECONDARY_UNUSED_REGION_ID = SECONDARY_REGION_ID
STREAMWISE_AXIS_INDEX = 2
OUT_OF_PLANE_AXIS_INDEX = 0
AXIS_NAMES = ("x", "y", "z")
FLOW_SOLUTION_MODE = "computed_projection"
DEFAULT_SOLID_CFL_TARGET = 0.5
FLOW_DRIVER_PROJECTION_ONLY = "projection_only"
FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC = "reinitialize_inlet_each_step_diagnostic"
FLOW_DRIVER_SUSTAINED_BOUNDARY = "sustained_boundary_inlet"
FLOW_DRIVER_SUSTAINED_SOURCE = "sustained_volume_source_inlet"
FLOW_DRIVER_SUSTAINED_PREDICTOR = "sustained_inlet_predictor"
FLOW_DRIVER_SHARP_REFERENCE = "sharp_hibm_mpm_reference"
FLOW_INLET_SOURCE_PROFILES = {"constant", "linear_ramp"}
FLOW_INLET_SOURCE_SCHEDULE_SCOPES = {"global", "phase_local"}
FLOW_OUTLET_BALANCE_POLICIES = {"report_only"}
TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES = "dual_physical_faces"
TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE = "single_mid_surface"
TRACTION_MARKER_LAYOUTS = {
    TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES,
    TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE,
}
TRACTION_PRESSURE_TWO_SIDED = "two_sided_pressure_jump"
TRACTION_PRESSURE_ONE_SIDED = "one_sided_surface_pressure"
TRACTION_PRESSURE_SAMPLING_MODES = {
    TRACTION_PRESSURE_TWO_SIDED,
    TRACTION_PRESSURE_ONE_SIDED,
}
TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION = "marker_position"
TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET = "physical_face_offset"
TRACTION_PRESSURE_PROBE_ORIGIN_MODES = {
    TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION,
    TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET,
}
TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL = "current_normal_cell_ladder"
TRACTION_PRESSURE_PROBE_LADDER_MODES = {
    TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL,
}
TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER = "independent_ladder"
TRACTION_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR = "symmetric_cell_pair"
TRACTION_PRESSURE_PAIR_POLICIES = {
    TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
    TRACTION_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR,
}
TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX = 4.0
SUPPORTED_FORMAL_FLOW_DRIVER_MODES = {
    FLOW_DRIVER_PROJECTION_ONLY,
    FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC,
    FLOW_DRIVER_SUSTAINED_BOUNDARY,
    FLOW_DRIVER_SUSTAINED_SOURCE,
    FLOW_DRIVER_SUSTAINED_PREDICTOR,
    FLOW_DRIVER_SHARP_REFERENCE,
}
FLOW_SOURCE_REPORT_KEYS = (
    "source_volume_flux_m3s",
    "positive_source_volume_flux_m3s",
    "abs_source_volume_flux_m3s",
    "zmin_pressure_outlet_flux_m3s",
    "zmin_velocity_outlet_flux_m3s",
    "zmin_pressure_outlet_to_source_ratio",
    "zmin_velocity_outlet_to_source_ratio",
    "zmin_pressure_outlet_to_net_source_ratio",
    "zmin_velocity_outlet_to_net_source_ratio",
    "zmin_pressure_outlet_to_positive_source_ratio",
    "zmin_velocity_outlet_to_positive_source_ratio",
    "zmin_pressure_outlet_to_abs_source_ratio",
    "zmin_velocity_outlet_to_abs_source_ratio",
    "pressure_outlet_flux_ratio",
    "velocity_outlet_flux_ratio",
)


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
    preflow_report = _run_fixed_solid_preflow(markers, fluid, solid, config)
    preflow_history = preflow_report["preflow_history"]

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
    apply_feedback = bool(getattr(config, "apply_marker_feedback_to_fluid", True))
    flow_driver_mode = _effective_flow_driver_mode(config)

    for step_index in range(config.step_count):
        if _flow_driver_requires_full_field_reinitialize(flow_driver_mode):
            _initialize_computed_flow(fluid, config)
            feedback_constraint_cells = set()
        feedback_available_before_projection = (
            feedback_available_for_projection and apply_feedback
        )
        latest_feedback_constraint_report = _apply_marker_feedback_to_fluid(
            markers,
            fluid,
            config,
            feedback_available=feedback_available_before_projection,
            previous_feedback_constraint_cells=feedback_constraint_cells,
        )
        feedback_constraint_cells = latest_feedback_constraint_report["_feedback_constraint_cells"]
        latest_flow_report = _flow_advance_current_step(
            fluid,
            config,
            flow_phase="fsi",
            step_index_local=step_index,
            step_index_global=len(preflow_history) + step_index,
            preflow_history=preflow_history,
            reset_pressure=(
                bool(getattr(config, "flow_reset_pressure_each_step", False))
                or (step_index == 0 and not preflow_history)
            ),
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
        latest_stress_report = _sample_stress_to_marker_forces(
            markers,
            fluid,
            config,
        )
        latest_force_report = markers.aggregate_region_forces(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
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
                secondary_region_id=SECONDARY_REGION_ID,
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
                "apply_marker_feedback_to_fluid": apply_feedback,
                "flow_driver_mode": latest_flow_report["flow_driver_mode"],
                "flow_driver_diagnostic_only": latest_flow_report[
                    "flow_driver_diagnostic_only"
                ],
                "flow_driver_uses_full_velocity_reset": latest_flow_report[
                    "flow_driver_uses_full_velocity_reset"
                ],
                "flow_full_field_reinitialized": latest_flow_report[
                    "flow_full_field_reinitialized"
                ],
                "flow_inlet_boundary_reapplied": latest_flow_report[
                    "flow_inlet_boundary_reapplied"
                ],
                "flow_volume_source_applied": latest_flow_report[
                    "flow_volume_source_applied"
                ],
                "flow_inlet_source_strength": float(
                    getattr(config, "flow_inlet_source_strength", 1.0)
                ),
                "flow_inlet_source_profile": str(
                    getattr(config, "flow_inlet_source_profile", "constant")
                ),
                "flow_inlet_source_ramp_steps": int(
                    getattr(config, "flow_inlet_source_ramp_steps", 0)
                ),
                "flow_inlet_source_schedule_scope": str(
                    getattr(config, "flow_inlet_source_schedule_scope", "global")
                ),
                "flow_inlet_source_factor": latest_flow_report[
                    "flow_inlet_source_factor"
                ],
                "flow_inlet_source_normal_velocity_mps": latest_flow_report[
                    "flow_inlet_source_normal_velocity_mps"
                ],
                "flow_pressure_outlet_enabled": bool(
                    getattr(config, "flow_pressure_outlet_enabled", True)
                ),
                "flow_outlet_balance_policy": str(
                    getattr(config, "flow_outlet_balance_policy", "report_only")
                ),
                "flow_predictor_applied": latest_flow_report[
                    "flow_predictor_applied"
                ],
                "flow_predictor_note": latest_flow_report["flow_predictor_note"],
                "flow_inlet_boundary_active_cell_count": latest_flow_report[
                    "flow_inlet_boundary_active_cell_count"
                ],
                "flow_inlet_boundary_obstacle_cell_count": latest_flow_report[
                    "flow_inlet_boundary_obstacle_cell_count"
                ],
                "flow_phase": latest_flow_report["flow_phase"],
                "flow_step_index_local": latest_flow_report[
                    "flow_step_index_local"
                ],
                "flow_step_index_global": latest_flow_report[
                    "flow_step_index_global"
                ],
                "flow_source_schedule_step_index": latest_flow_report[
                    "flow_source_schedule_step_index"
                ],
                "flow_source_schedule_scope": latest_flow_report[
                    "flow_source_schedule_scope"
                ],
                "flow_source_ramp_restarted_after_preflow": latest_flow_report[
                    "flow_source_ramp_restarted_after_preflow"
                ],
                "flow_reset_pressure_each_step": bool(
                    getattr(config, "flow_reset_pressure_each_step", False)
                ),
                "flow_pressure_reset_applied": latest_flow_report[
                    "flow_pressure_reset_applied"
                ],
                "flow_reinitialize_inlet_each_step": bool(
                    getattr(config, "flow_reinitialize_inlet_each_step", False)
                ),
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
                "fluid_speed_p99_mps": latest_flow_report["fluid_speed_p99_mps"],
                "fluid_speed_p999_mps": latest_flow_report["fluid_speed_p999_mps"],
                "pressure_min_pa": latest_flow_report["pressure_min_pa"],
                "pressure_max_pa": latest_flow_report["pressure_max_pa"],
                "flow_projection_report": latest_flow_report["projection_report"],
                **_flow_source_report_fields(latest_flow_report),
                "solid_substeps_selected": solid_substeps,
                "solid_estimated_cfl": solid_substep_cfl["solid_estimated_cfl"],
                "stress_valid_marker_count": latest_stress_report.valid_marker_count,
                "stress_invalid_marker_count": (
                    latest_stress_report.invalid_marker_count
                ),
                "scatter_invalid_marker_count": (
                    latest_scatter_report.invalid_marker_count
                ),
                "feedback_invalid_marker_count": (
                    latest_feedback_report.invalid_marker_count
                ),
                "total_marker_force_n": latest_force_report.total_marker_force_n,
                **_marker_force_report_fields(latest_force_report),
                **_stress_sampling_report_fields(latest_stress_report),
                **_marker_traction_report_fields(markers),
                **_scatter_report_fields(latest_scatter_report),
                "mpm_external_force_n": latest_solid_report.external_force_n,
                "max_displacement_m": step_displacement["max_displacement_m"],
                "root_max_displacement_m": step_displacement[
                    "root_max_displacement_m"
                ],
                "tip_mean_displacement_m": step_displacement[
                    "tip_mean_displacement_m"
                ],
            }
        )

    if config.step_count == 0 and preflow_history:
        return _preflow_only_report(
            case_id=case_id,
            case_metadata=case_metadata,
            boundary_conditions=boundary_conditions,
            reference_results=reference_results,
            config=config,
            markers=markers,
            solid=solid,
            fixed_mask=fixed_mask,
            tip_mask=tip_mask,
            solid_substep_cfl=solid_substep_cfl,
            preflow_report=preflow_report,
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
        **preflow_report,
        "apply_marker_feedback_to_fluid": apply_feedback,
        "flow_driver_mode": flow_driver_mode,
        "flow_driver_diagnostic_only": (
            flow_driver_mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
        ),
        "flow_inlet_source_strength": float(
            getattr(config, "flow_inlet_source_strength", 1.0)
        ),
        "flow_inlet_source_profile": str(
            getattr(config, "flow_inlet_source_profile", "constant")
        ),
        "flow_inlet_source_ramp_steps": int(
            getattr(config, "flow_inlet_source_ramp_steps", 0)
        ),
        "flow_inlet_source_schedule_scope": str(
            getattr(config, "flow_inlet_source_schedule_scope", "global")
        ),
        "flow_pressure_outlet_enabled": bool(
            getattr(config, "flow_pressure_outlet_enabled", True)
        ),
        "flow_outlet_balance_policy": str(
            getattr(config, "flow_outlet_balance_policy", "report_only")
        ),
        "flow_reset_pressure_each_step": bool(
            getattr(config, "flow_reset_pressure_each_step", False)
        ),
        "flow_pressure_reset_applied": latest_flow_report[
            "flow_pressure_reset_applied"
        ],
        "flow_reinitialize_inlet_each_step": bool(
            getattr(config, "flow_reinitialize_inlet_each_step", False)
        ),
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
        "marker_face_count": _traction_marker_face_count(config),
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
        "flow_phase": latest_flow_report["flow_phase"],
        "flow_step_index_local": latest_flow_report["flow_step_index_local"],
        "flow_step_index_global": latest_flow_report["flow_step_index_global"],
        "flow_source_schedule_step_index": latest_flow_report[
            "flow_source_schedule_step_index"
        ],
        "flow_source_schedule_scope": latest_flow_report["flow_source_schedule_scope"],
        "flow_source_ramp_restarted_after_preflow": latest_flow_report[
            "flow_source_ramp_restarted_after_preflow"
        ],
        **_flow_source_report_fields(latest_flow_report),
        "flow_obstacle_cell_count": latest_flow_report["obstacle_cell_count"],
        "flow_fluid_cell_count": latest_flow_report["fluid_cell_count"],
        "computed_pressure_min_pa": latest_flow_report["pressure_min_pa"],
        "computed_pressure_max_pa": latest_flow_report["pressure_max_pa"],
        "pressure_sign_convention": latest_flow_report["pressure_sign_convention"],
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "fluid_speed_p99_mps": latest_flow_report["fluid_speed_p99_mps"],
        "fluid_speed_p999_mps": latest_flow_report["fluid_speed_p999_mps"],
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
        **_marker_force_report_fields(latest_force_report),
        "scatter_invalid_marker_count": latest_scatter_report.invalid_marker_count,
        "scatter_active_marker_count": latest_scatter_report.active_marker_count,
        "scatter_active_particle_count": latest_scatter_report.active_particle_count,
        **_scatter_report_fields(latest_scatter_report),
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


def _preflow_only_report(
    *,
    case_id: str,
    case_metadata: Mapping[str, Any],
    boundary_conditions: Mapping[str, Any],
    reference_results: Mapping[str, Any],
    config: Any,
    markers: HibmMpmSurfaceMarkers,
    solid: NeoHookeanMpmState,
    fixed_mask: np.ndarray,
    tip_mask: np.ndarray,
    solid_substep_cfl: Mapping[str, object],
    preflow_report: Mapping[str, object],
) -> dict[str, object]:
    preflow_history = list(preflow_report["preflow_history"])
    latest_preflow = dict(preflow_history[-1])
    projection_report = latest_preflow["flow_projection_report"]
    displacement = _solid_displacement_report(solid, fixed_mask, tip_mask)
    marker_force = tuple(latest_preflow["total_marker_force_n"])
    reference_velocity_peak = float(reference_results["local_velocity_peak_mps"])
    local_velocity_peak_mps = float(latest_preflow["local_velocity_peak_mps"])
    velocity_relative_error = (
        abs(local_velocity_peak_mps - reference_velocity_peak) / reference_velocity_peak
    )
    flow_driver_mode = _effective_flow_driver_mode(config)
    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        **preflow_report,
        "apply_marker_feedback_to_fluid": bool(
            getattr(config, "apply_marker_feedback_to_fluid", True)
        ),
        "flow_driver_mode": flow_driver_mode,
        "flow_driver_diagnostic_only": (
            flow_driver_mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
        ),
        "flow_inlet_source_strength": float(
            getattr(config, "flow_inlet_source_strength", 1.0)
        ),
        "flow_inlet_source_profile": str(
            getattr(config, "flow_inlet_source_profile", "constant")
        ),
        "flow_inlet_source_ramp_steps": int(
            getattr(config, "flow_inlet_source_ramp_steps", 0)
        ),
        "flow_inlet_source_schedule_scope": str(
            getattr(config, "flow_inlet_source_schedule_scope", "global")
        ),
        "flow_pressure_outlet_enabled": bool(
            getattr(config, "flow_pressure_outlet_enabled", True)
        ),
        "flow_outlet_balance_policy": str(
            getattr(config, "flow_outlet_balance_policy", "report_only")
        ),
        "flow_reset_pressure_each_step": bool(
            getattr(config, "flow_reset_pressure_each_step", False)
        ),
        "flow_pressure_reset_applied": latest_preflow["flow_pressure_reset_applied"],
        "flow_reinitialize_inlet_each_step": bool(
            getattr(config, "flow_reinitialize_inlet_each_step", False)
        ),
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
        "marker_face_count": _traction_marker_face_count(config),
        "marker_count_per_face": int(config.marker_count),
        "marker_count_actual": int(markers.marker_count),
        "flow_projection_iterations_actual": int(config.flow_projection_iterations),
        "solid_substep_cfl_report": dict(solid_substep_cfl),
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
        "flow_projection_report": projection_report,
        "flow_phase": latest_preflow["flow_phase"],
        "flow_step_index_local": latest_preflow["flow_step_index_local"],
        "flow_step_index_global": latest_preflow["flow_step_index_global"],
        "flow_source_schedule_step_index": latest_preflow[
            "flow_source_schedule_step_index"
        ],
        "flow_source_schedule_scope": latest_preflow["flow_source_schedule_scope"],
        "flow_source_ramp_restarted_after_preflow": latest_preflow[
            "flow_source_ramp_restarted_after_preflow"
        ],
        **_flow_source_report_fields(latest_preflow),
        "computed_pressure_min_pa": latest_preflow["pressure_min_pa"],
        "computed_pressure_max_pa": latest_preflow["pressure_max_pa"],
        "pressure_sign_convention": "fluid.pressure projection field is sampled directly",
        "local_velocity_peak_mps": local_velocity_peak_mps,
        "fluid_speed_p99_mps": latest_preflow["fluid_speed_p99_mps"],
        "fluid_speed_p999_mps": latest_preflow["fluid_speed_p999_mps"],
        "local_velocity_peak_relative_error": velocity_relative_error,
        "velocity_peak_tolerance": config.velocity_peak_tolerance,
        "fluid_recomputed_after_feedback": False,
        "feedback_closure_status": "PREFLOW_ONLY_FIXED_SOLID",
        "fluid_recompute_count": int(preflow_report["preflow_steps_completed"]),
        "fluid_projection_count": int(preflow_report["preflow_steps_completed"]),
        "fluid_projection_after_feedback_count": 0,
        "fluid_projection_consumed_feedback_count": 0,
        "fluid_projection_consumed_feedback": False,
        "fluid_feedback_constraint_marker_count": 0,
        "fluid_feedback_constraint_active_cell_count": 0,
        "fluid_feedback_constraint_cleared_cell_count": 0,
        "fluid_feedback_constraint_obstacle_cell_count": 0,
        "fluid_feedback_constraint_non_obstacle_cell_count": 0,
        "fluid_feedback_constraint_projection_participating_cell_count": 0,
        "no_slip_residual_before_mps": "",
        "no_slip_residual_after_mps": "",
        "no_slip_target_residual_after_assembly_mps": "",
        "no_slip_projected_residual_after_projection_mps": 0.0,
        "stress_valid_marker_count": latest_preflow["stress_valid_marker_count"],
        "stress_invalid_marker_count": latest_preflow["stress_invalid_marker_count"],
        "two_sided_pressure_marker_count": latest_preflow[
            "two_sided_pressure_marker_count"
        ],
        "max_abs_traction_pa": latest_preflow.get("max_abs_traction_pa", ""),
        "one_sided_pressure_marker_count": latest_preflow.get(
            "one_sided_pressure_marker_count",
            "",
        ),
        "total_marker_force_n": marker_force,
        "fluid_reaction_force_n": tuple(latest_preflow["fluid_reaction_force_n"]),
        "fluid_reaction_force_z_N": latest_preflow["fluid_reaction_force_z_N"],
        "marker_force_z_N": latest_preflow["marker_force_z_N"],
        "marker_action_reaction_residual_n": latest_preflow[
            "marker_action_reaction_residual_n"
        ],
        "marker_action_reaction_residual_N": latest_preflow[
            "marker_action_reaction_residual_N"
        ],
        "primary_face_force_n": tuple(latest_preflow["primary_face_force_n"]),
        "secondary_face_force_n": tuple(latest_preflow["secondary_face_force_n"]),
        "primary_face_force_z_N": latest_preflow["primary_face_force_z_N"],
        "secondary_face_force_z_N": latest_preflow["secondary_face_force_z_N"],
        "primary_face_marker_count": latest_preflow["primary_face_marker_count"],
        "secondary_face_marker_count": latest_preflow["secondary_face_marker_count"],
        "primary_face_valid_marker_count": latest_preflow[
            "primary_face_valid_marker_count"
        ],
        "secondary_face_valid_marker_count": latest_preflow[
            "secondary_face_valid_marker_count"
        ],
        "primary_face_invalid_marker_count": latest_preflow[
            "primary_face_invalid_marker_count"
        ],
        "secondary_face_invalid_marker_count": latest_preflow[
            "secondary_face_invalid_marker_count"
        ],
        "scatter_invalid_marker_count": latest_preflow["scatter_invalid_marker_count"],
        "scatter_active_marker_count": latest_preflow["scatter_active_marker_count"],
        "scatter_active_particle_count": latest_preflow[
            "scatter_active_particle_count"
        ],
        "scatter_action_reaction_residual_n": latest_preflow[
            "scatter_action_reaction_residual_n"
        ],
        "scatter_action_reaction_residual_N": latest_preflow[
            "scatter_action_reaction_residual_N"
        ],
        "mpm_external_force_n": tuple(latest_preflow["mpm_external_force_n"]),
        "surface_feedback_updated_marker_count": 0,
        "surface_feedback_invalid_marker_count": 0,
        "surface_feedback_max_marker_displacement_m": 0.0,
        "history": [],
        "max_displacement_m": displacement["max_displacement_m"],
        "reference_max_displacement_m": float(reference_results["max_displacement_m"]),
        "max_displacement_relative_error": 1.0,
        "displacement_tolerance": config.displacement_tolerance,
        **displacement,
    }


def _traction_marker_layout(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_marker_layout",
            TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES,
        )
    )


def _traction_pressure_sampling_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_sampling_mode",
            TRACTION_PRESSURE_TWO_SIDED,
        )
    )


def _traction_marker_face_offset_cells(config: Any) -> float:
    return float(getattr(config, "traction_marker_face_offset_cells", 0.51))


def _traction_pressure_probe_origin_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_probe_origin_mode",
            TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION,
        )
    )


def _traction_pressure_probe_origin_offset_cells(config: Any) -> float | None:
    value = getattr(config, "traction_pressure_probe_origin_offset_cells", None)
    if value is None:
        return None
    return float(value)


def _traction_pressure_probe_start_offset_cells(config: Any) -> float | None:
    value = getattr(config, "traction_pressure_probe_start_offset_cells", None)
    if value is None:
        return None
    return float(value)


def _traction_pressure_probe_ladder_spacing_cells(config: Any) -> float:
    return float(getattr(config, "traction_pressure_probe_ladder_spacing_cells", 0.5))


def _traction_pressure_probe_ladder_rung_count(config: Any) -> int:
    return int(getattr(config, "traction_pressure_probe_ladder_rung_count", 5))


def _traction_pressure_probe_ladder_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_probe_ladder_mode",
            TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL,
        )
    )


def _traction_pressure_pair_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_pair_policy",
            TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
        )
    )


def _traction_pressure_pair_max_cell_delta(config: Any) -> int:
    return int(getattr(config, "traction_pressure_pair_max_cell_delta", 1))


def _traction_pressure_pair_require_opposite_sides(config: Any) -> bool:
    return bool(getattr(config, "traction_pressure_pair_require_opposite_sides", True))


def _traction_marker_face_count(config: Any) -> int:
    if _traction_marker_layout(config) == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE:
        return 1
    return 2


def _traction_include_viscous(config: Any) -> bool:
    return bool(getattr(config, "traction_include_viscous", False))


def _is_default_traction_formulation(config: Any) -> bool:
    return (
        _traction_marker_layout(config) == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and _traction_pressure_sampling_mode(config) == TRACTION_PRESSURE_TWO_SIDED
        and math.isclose(_traction_marker_face_offset_cells(config), 0.51)
        and not _traction_include_viscous(config)
        and _traction_pressure_probe_origin_mode(config)
        == TRACTION_PRESSURE_PROBE_ORIGIN_MARKER_POSITION
        and _traction_pressure_probe_origin_offset_cells(config) is None
        and _traction_pressure_probe_start_offset_cells(config) is None
        and _traction_pressure_probe_ladder_mode(config)
        == TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
        and _traction_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER
        and _traction_pressure_pair_max_cell_delta(config) == 1
        and _traction_pressure_pair_require_opposite_sides(config)
    )


def _traction_viscosity_pa_s(config: Any) -> float:
    if not _traction_include_viscous(config):
        return 0.0
    configured = float(
        getattr(
            config,
            "traction_viscosity_pa_s",
            getattr(config, "air_viscosity_pa_s", 0.0),
        )
    )
    if configured == 0.0:
        return float(getattr(config, "air_viscosity_pa_s", 0.0))
    return configured


def traction_formulation_supported(config: Any) -> tuple[bool, str]:
    marker_layout = _traction_marker_layout(config)
    pressure_sampling_mode = _traction_pressure_sampling_mode(config)
    if (
        marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
    ):
        return (
            False,
            "dual-face one-sided pressure needs per-face one-sided region support; "
            "current core exposes one one_sided_pressure_region_id",
        )
    if (
        marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE
        and pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
    ):
        return (
            False,
            "single-mid one-sided pressure has ambiguous fluid side without "
            "explicit one_sided_fluid_side_normal_sign",
        )
    return True, "supported"


def _validate_rectangular_solid_config(config: Any) -> None:
    flow_driver_mode = str(
        getattr(config, "flow_driver_mode", FLOW_DRIVER_PROJECTION_ONLY)
    )
    if flow_driver_mode not in SUPPORTED_FORMAL_FLOW_DRIVER_MODES:
        raise ValueError(f"unsupported flow_driver_mode: {flow_driver_mode!r}")
    if flow_driver_mode == FLOW_DRIVER_SHARP_REFERENCE:
        raise ValueError(
            "sharp_hibm_mpm_reference is reserved for a later sharp-path runner"
        )
    source_strength = float(getattr(config, "flow_inlet_source_strength", 1.0))
    if not math.isfinite(source_strength) or source_strength < 0.0:
        raise ValueError("flow_inlet_source_strength must be finite and non-negative")
    source_profile = str(getattr(config, "flow_inlet_source_profile", "constant"))
    if source_profile not in FLOW_INLET_SOURCE_PROFILES:
        raise ValueError(f"unsupported flow_inlet_source_profile: {source_profile!r}")
    source_scope = str(getattr(config, "flow_inlet_source_schedule_scope", "global"))
    if source_scope not in FLOW_INLET_SOURCE_SCHEDULE_SCOPES:
        raise ValueError(
            f"unsupported flow_inlet_source_schedule_scope: {source_scope!r}"
        )
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if ramp_steps < 0:
        raise ValueError("flow_inlet_source_ramp_steps must be non-negative")
    outlet_policy = str(getattr(config, "flow_outlet_balance_policy", "report_only"))
    if outlet_policy not in FLOW_OUTLET_BALANCE_POLICIES:
        raise ValueError(f"unsupported flow_outlet_balance_policy: {outlet_policy!r}")
    marker_layout = _traction_marker_layout(config)
    if marker_layout not in TRACTION_MARKER_LAYOUTS:
        raise ValueError(f"unsupported traction_marker_layout: {marker_layout!r}")
    pressure_sampling_mode = _traction_pressure_sampling_mode(config)
    if pressure_sampling_mode not in TRACTION_PRESSURE_SAMPLING_MODES:
        raise ValueError(
            f"unsupported traction_pressure_sampling_mode: {pressure_sampling_mode!r}"
        )
    formulation_supported, formulation_reason = traction_formulation_supported(config)
    if not formulation_supported:
        raise ValueError(f"unsupported traction formulation: {formulation_reason}")
    marker_face_offset_cells = _traction_marker_face_offset_cells(config)
    if not math.isfinite(marker_face_offset_cells) or marker_face_offset_cells < 0.0:
        raise ValueError(
            "traction_marker_face_offset_cells must be finite and non-negative"
        )
    if marker_face_offset_cells > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX:
        raise ValueError(
            "traction_marker_face_offset_cells is outside the fixed-solid "
            "diagnostic range"
        )
    probe_origin_mode = _traction_pressure_probe_origin_mode(config)
    if probe_origin_mode not in TRACTION_PRESSURE_PROBE_ORIGIN_MODES:
        raise ValueError(
            f"unsupported traction_pressure_probe_origin_mode: {probe_origin_mode!r}"
        )
    probe_origin_offset_cells = _traction_pressure_probe_origin_offset_cells(config)
    if probe_origin_offset_cells is not None:
        if (
            not math.isfinite(probe_origin_offset_cells)
            or probe_origin_offset_cells < 0.0
        ):
            raise ValueError(
                "traction_pressure_probe_origin_offset_cells must be finite "
                "and non-negative"
            )
        if (
            probe_origin_offset_cells
            > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX
        ):
            raise ValueError(
                "traction_pressure_probe_origin_offset_cells is outside the "
                "diagnostic range"
            )
    if (
        probe_origin_mode == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
        and probe_origin_offset_cells is None
    ):
        raise ValueError(
            "traction_pressure_probe_origin_offset_cells is required for "
            "physical_face_offset probe origins"
        )
    probe_start_offset_cells = _traction_pressure_probe_start_offset_cells(config)
    if probe_start_offset_cells is not None:
        if (
            not math.isfinite(probe_start_offset_cells)
            or probe_start_offset_cells < 0.0
        ):
            raise ValueError(
                "traction_pressure_probe_start_offset_cells must be finite "
                "and non-negative"
            )
        if (
            probe_start_offset_cells
            > TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX
        ):
            raise ValueError(
                "traction_pressure_probe_start_offset_cells is outside the "
                "diagnostic range"
            )
    probe_ladder_spacing_cells = _traction_pressure_probe_ladder_spacing_cells(config)
    if (
        not math.isfinite(probe_ladder_spacing_cells)
        or probe_ladder_spacing_cells <= 0.0
    ):
        raise ValueError(
            "traction_pressure_probe_ladder_spacing_cells must be finite and positive"
        )
    probe_ladder_rung_count = _traction_pressure_probe_ladder_rung_count(config)
    if probe_ladder_rung_count <= 0:
        raise ValueError("traction_pressure_probe_ladder_rung_count must be positive")
    probe_ladder_mode = _traction_pressure_probe_ladder_mode(config)
    if probe_ladder_mode not in TRACTION_PRESSURE_PROBE_LADDER_MODES:
        raise ValueError(
            f"unsupported traction_pressure_probe_ladder_mode: {probe_ladder_mode!r}"
        )
    pressure_pair_policy = _traction_pressure_pair_policy(config)
    if pressure_pair_policy not in TRACTION_PRESSURE_PAIR_POLICIES:
        raise ValueError(
            f"unsupported traction_pressure_pair_policy: {pressure_pair_policy!r}"
        )
    pressure_pair_max_cell_delta = _traction_pressure_pair_max_cell_delta(config)
    if pressure_pair_max_cell_delta < 0:
        raise ValueError("traction_pressure_pair_max_cell_delta must be non-negative")
    traction_viscosity = _traction_viscosity_pa_s(config)
    if not math.isfinite(traction_viscosity) or traction_viscosity < 0.0:
        raise ValueError("traction viscosity must be finite and non-negative")
    if config.step_count < 0:
        raise ValueError("step_count must be non-negative")
    if config.step_count > 0 and not _is_default_traction_formulation(config):
        raise ValueError(
            "non-default traction formulations are fixed-solid diagnostics only"
        )
    if config.step_count == 0 and int(getattr(config, "preflow_steps", 0)) <= 0:
        raise ValueError("step_count=0 is only valid for preflow-only diagnostics")
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
    if int(getattr(config, "preflow_steps", 0)) < 0:
        raise ValueError("preflow_steps must be non-negative")
    if float(getattr(config, "preflow_convergence_tolerance", 0.0)) < 0.0:
        raise ValueError("preflow_convergence_tolerance must be non-negative")
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
    projection_report = dict(
        fluid.project(
            iterations=config.flow_projection_iterations,
            pressure_outlet_zmin=bool(
                getattr(config, "flow_pressure_outlet_enabled", True)
            ),
            reset_pressure=reset_pressure,
            pressure_solver=config.flow_pressure_solver,
            cg_tolerance=config.flow_cg_tolerance,
            divergence_cleanup_iterations=config.flow_divergence_cleanup_iterations,
        )
    )
    projection_report.update(
        fluid.pressure_outlet_fv_flux_report(dt_s=float(config.dt_s))
    )
    return _flow_state_report(fluid, projection_report)


def _flow_advance_current_step(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    flow_phase: str,
    step_index_local: int,
    step_index_global: int,
    preflow_history: list[dict[str, object]],
    reset_pressure: bool,
) -> dict[str, object]:
    source_schedule_scope = _flow_source_schedule_scope(config)
    source_schedule_step_index = _flow_source_schedule_step_index(
        config,
        step_index_local=step_index_local,
        step_index_global=step_index_global,
    )
    mode = _effective_flow_driver_mode(config)
    fluid.clear_volume_source()
    driver_report = _flow_driver_report(
        mode=mode,
        full_field_reinitialized=_flow_driver_requires_full_field_reinitialize(mode),
        inlet_boundary_report={},
        volume_source_applied=False,
    )

    if mode == FLOW_DRIVER_PROJECTION_ONLY:
        pass
    elif mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC:
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=True,
            inlet_boundary_report=_zmax_inlet_boundary_report(fluid),
            volume_source_applied=False,
        )
    elif mode == FLOW_DRIVER_SUSTAINED_BOUNDARY:
        boundary_report = _refresh_zmax_inlet_boundary(fluid, config)
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=False,
            inlet_boundary_report=boundary_report,
            volume_source_applied=False,
        )
    elif mode in {FLOW_DRIVER_SUSTAINED_SOURCE, FLOW_DRIVER_SUSTAINED_PREDICTOR}:
        boundary_report = _refresh_zmax_inlet_boundary(fluid, config)
        source_factor = _flow_inlet_source_factor(config, source_schedule_step_index)
        source_normal_velocity_mps = -float(config.inlet_velocity_mps) * source_factor
        fluid.add_zmax_velocity_inlet_volume_source(
            normal_velocity_mps=source_normal_velocity_mps,
        )
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=False,
            inlet_boundary_report=boundary_report,
            volume_source_applied=True,
            source_factor=source_factor,
            source_normal_velocity_mps=source_normal_velocity_mps,
            predictor_applied=False,
            predictor_note=(
                "diagnostic source-driven projection path; no separate "
                "predictor/advection step is applied yet"
                if mode == FLOW_DRIVER_SUSTAINED_PREDICTOR
                else ""
            ),
        )
    elif mode == FLOW_DRIVER_SHARP_REFERENCE:
        raise RuntimeError(
            "sharp_hibm_mpm_reference is reserved for a sharp-path runner"
        )
    else:  # pragma: no cover - protected by config validation.
        raise RuntimeError(f"unsupported flow_driver_mode: {mode!r}")

    flow_report = _project_current_flow(
        fluid,
        config,
        reset_pressure=reset_pressure,
    )
    flow_report.update(driver_report)
    flow_report["flow_phase"] = str(flow_phase)
    flow_report["flow_step_index"] = int(step_index_local)
    flow_report["flow_step_index_local"] = int(step_index_local)
    flow_report["flow_step_index_global"] = int(step_index_global)
    flow_report["flow_pressure_reset_applied"] = bool(reset_pressure)
    flow_report["flow_source_schedule_step_index"] = int(source_schedule_step_index)
    flow_report["flow_source_schedule_scope"] = source_schedule_scope
    flow_report["flow_source_ramp_restarted_after_preflow"] = (
        _flow_source_ramp_restarted_after_preflow(
            config,
            flow_phase=flow_phase,
            step_index_local=step_index_local,
            step_index_global=step_index_global,
            source_schedule_step_index=source_schedule_step_index,
            preflow_history=preflow_history,
        )
    )
    flow_report["flow_preflow_history_rows"] = len(preflow_history)
    return flow_report


def _effective_flow_driver_mode(config: Any) -> str:
    if bool(getattr(config, "flow_reinitialize_inlet_each_step", False)):
        return FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
    return str(getattr(config, "flow_driver_mode", FLOW_DRIVER_PROJECTION_ONLY))


def _flow_driver_requires_full_field_reinitialize(mode: str) -> bool:
    return mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC


def _flow_inlet_source_factor(config: Any, step_index: int) -> float:
    strength = float(getattr(config, "flow_inlet_source_strength", 1.0))
    profile = str(getattr(config, "flow_inlet_source_profile", "constant"))
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if profile == "constant" or ramp_steps <= 0:
        return strength
    if profile == "linear_ramp":
        ramp_fraction = min(1.0, max(0.0, float(step_index + 1) / float(ramp_steps)))
        return strength * ramp_fraction
    raise ValueError(f"unsupported flow_inlet_source_profile: {profile!r}")


def _flow_source_schedule_scope(config: Any) -> str:
    return str(getattr(config, "flow_inlet_source_schedule_scope", "global"))


def _flow_source_schedule_step_index(
    config: Any,
    *,
    step_index_local: int,
    step_index_global: int,
) -> int:
    if _flow_source_schedule_scope(config) == "global":
        return int(step_index_global)
    return int(step_index_local)


def _flow_source_ramp_restarted_after_preflow(
    config: Any,
    *,
    flow_phase: str,
    step_index_local: int,
    step_index_global: int,
    source_schedule_step_index: int,
    preflow_history: list[dict[str, object]],
) -> bool:
    if str(flow_phase) != "fsi" or not preflow_history:
        return False
    if _flow_source_schedule_scope(config) != "phase_local":
        return False
    if str(getattr(config, "flow_inlet_source_profile", "constant")) != "linear_ramp":
        return False
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if ramp_steps <= 0:
        return False
    return (
        int(step_index_global) >= ramp_steps
        and int(source_schedule_step_index) < ramp_steps
        and int(step_index_local) == int(source_schedule_step_index)
    )


def _flow_driver_report(
    *,
    mode: str,
    full_field_reinitialized: bool,
    inlet_boundary_report: Mapping[str, object],
    volume_source_applied: bool,
    source_factor: float = 0.0,
    source_normal_velocity_mps: float = 0.0,
    predictor_applied: bool = False,
    predictor_note: str = "",
) -> dict[str, object]:
    inlet_reapplied = bool(inlet_boundary_report)
    return {
        "flow_driver_mode": mode,
        "flow_driver_diagnostic_only": mode == FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC,
        "flow_driver_uses_full_velocity_reset": bool(full_field_reinitialized),
        "flow_full_field_reinitialized": bool(full_field_reinitialized),
        "flow_inlet_boundary_reapplied": inlet_reapplied,
        "flow_volume_source_applied": bool(volume_source_applied),
        "flow_inlet_boundary_active_cell_count": int(
            inlet_boundary_report.get("flow_inlet_boundary_active_cell_count", 0)
        ),
        "flow_inlet_boundary_obstacle_cell_count": int(
            inlet_boundary_report.get("flow_inlet_boundary_obstacle_cell_count", 0)
        ),
        "flow_inlet_source_factor": float(source_factor),
        "flow_inlet_source_normal_velocity_mps": float(source_normal_velocity_mps),
        "flow_predictor_applied": bool(predictor_applied),
        "flow_predictor_note": str(predictor_note),
    }


def _refresh_zmax_inlet_boundary(
    fluid: CartesianFluidSolver,
    config: Any,
) -> dict[str, object]:
    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    weights = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
    obstacle = fluid.obstacle.to_numpy()
    k = int(config.grid_nodes[2]) - 1
    fluid_mask = obstacle[:, :, k] == 0

    active[:, :, k] = fluid_mask.astype(np.int32)
    values[:, :, k, :] = 0.0
    values[:, :, k, STREAMWISE_AXIS_INDEX] = (
        -float(config.inlet_velocity_mps) * fluid_mask.astype(np.float32)
    )
    weights[:, :, k] = fluid_mask.astype(np.float32)

    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
    return _zmax_inlet_boundary_report(fluid)


def _zmax_inlet_boundary_report(
    fluid: CartesianFluidSolver,
) -> dict[str, object]:
    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    obstacle = fluid.obstacle.to_numpy()
    k = active.shape[2] - 1
    active_slice = active[:, :, k] != 0
    obstacle_slice = obstacle[:, :, k] != 0
    return {
        "flow_inlet_boundary_active_cell_count": int(active_slice.sum()),
        "flow_inlet_boundary_obstacle_cell_count": int(
            np.logical_and(active_slice, obstacle_slice).sum()
        ),
    }


def _run_fixed_solid_preflow(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: Any,
) -> dict[str, object]:
    requested_steps = int(getattr(config, "preflow_steps", 0))
    tolerance = float(getattr(config, "preflow_convergence_tolerance", 0.0))
    history: list[dict[str, object]] = []
    previous_row: dict[str, object] | None = None
    converged = requested_steps == 0
    stop_reason = "not_requested" if requested_steps == 0 else "max_steps"

    for preflow_index in range(requested_steps):
        if _flow_driver_requires_full_field_reinitialize(
            _effective_flow_driver_mode(config)
        ):
            _initialize_computed_flow(fluid, config)
        flow_report = _flow_advance_current_step(
            fluid,
            config,
            flow_phase="preflow",
            step_index_local=preflow_index,
            step_index_global=preflow_index,
            preflow_history=history,
            reset_pressure=(
                bool(getattr(config, "flow_reset_pressure_each_step", False))
                or preflow_index == 0
            ),
        )
        stress_report = _sample_stress_to_marker_forces(markers, fluid, config)
        force_report = markers.aggregate_region_forces(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
        )
        markers.clear_mpm_external_forces(
            solid.external_force_n,
            particle_count=solid.particle_count,
        )
        scatter_report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=config.mpm_support_radius_m,
        )
        row = {
            "preflow_step": preflow_index + 1,
            "fluid_recomputed": True,
            "flow_driver_mode": flow_report["flow_driver_mode"],
            "flow_driver_diagnostic_only": flow_report["flow_driver_diagnostic_only"],
            "flow_driver_uses_full_velocity_reset": flow_report[
                "flow_driver_uses_full_velocity_reset"
            ],
            "flow_full_field_reinitialized": flow_report[
                "flow_full_field_reinitialized"
            ],
            "flow_inlet_boundary_reapplied": flow_report[
                "flow_inlet_boundary_reapplied"
            ],
            "flow_volume_source_applied": flow_report["flow_volume_source_applied"],
            "flow_inlet_source_strength": float(
                getattr(config, "flow_inlet_source_strength", 1.0)
            ),
            "flow_inlet_source_profile": str(
                getattr(config, "flow_inlet_source_profile", "constant")
            ),
            "flow_inlet_source_ramp_steps": int(
                getattr(config, "flow_inlet_source_ramp_steps", 0)
            ),
            "flow_inlet_source_schedule_scope": str(
                getattr(config, "flow_inlet_source_schedule_scope", "global")
            ),
            "flow_inlet_source_factor": flow_report["flow_inlet_source_factor"],
            "flow_inlet_source_normal_velocity_mps": flow_report[
                "flow_inlet_source_normal_velocity_mps"
            ],
            "flow_pressure_outlet_enabled": bool(
                getattr(config, "flow_pressure_outlet_enabled", True)
            ),
            "flow_outlet_balance_policy": str(
                getattr(config, "flow_outlet_balance_policy", "report_only")
            ),
            "flow_predictor_applied": flow_report["flow_predictor_applied"],
            "flow_predictor_note": flow_report["flow_predictor_note"],
            "flow_inlet_boundary_active_cell_count": flow_report[
                "flow_inlet_boundary_active_cell_count"
            ],
            "flow_inlet_boundary_obstacle_cell_count": flow_report[
                "flow_inlet_boundary_obstacle_cell_count"
            ],
            "flow_phase": flow_report["flow_phase"],
            "flow_step_index_local": flow_report["flow_step_index_local"],
            "flow_step_index_global": flow_report["flow_step_index_global"],
            "flow_source_schedule_step_index": flow_report[
                "flow_source_schedule_step_index"
            ],
            "flow_source_schedule_scope": flow_report["flow_source_schedule_scope"],
            "flow_source_ramp_restarted_after_preflow": flow_report[
                "flow_source_ramp_restarted_after_preflow"
            ],
            "flow_pressure_reset_applied": flow_report["flow_pressure_reset_applied"],
            "solid_fixed": True,
            "solid_advanced": False,
            "local_velocity_peak_mps": flow_report["local_velocity_peak_mps"],
            "fluid_speed_p99_mps": flow_report["fluid_speed_p99_mps"],
            "fluid_speed_p999_mps": flow_report["fluid_speed_p999_mps"],
            "pressure_min_pa": flow_report["pressure_min_pa"],
            "pressure_max_pa": flow_report["pressure_max_pa"],
            "flow_projection_report": flow_report["projection_report"],
            **_flow_source_report_fields(flow_report),
            "stress_valid_marker_count": stress_report.valid_marker_count,
            "stress_invalid_marker_count": stress_report.invalid_marker_count,
            "two_sided_pressure_marker_count": (
                stress_report.two_sided_pressure_marker_count
            ),
            "total_marker_force_n": force_report.total_marker_force_n,
            "mpm_external_force_n": scatter_report.total_mpm_external_force_n,
            "scatter_invalid_marker_count": scatter_report.invalid_marker_count,
            "scatter_active_marker_count": scatter_report.active_marker_count,
            "scatter_active_particle_count": scatter_report.active_particle_count,
            **_marker_force_report_fields(force_report),
            **_stress_sampling_report_fields(stress_report),
            **_marker_traction_report_fields(markers),
            **_scatter_report_fields(scatter_report),
        }
        if previous_row is not None:
            row["velocity_peak_relative_delta"] = _relative_delta(
                row["local_velocity_peak_mps"],
                previous_row["local_velocity_peak_mps"],
            )
            row["pressure_range_relative_delta"] = _relative_delta(
                _pressure_range(row),
                _pressure_range(previous_row),
            )
            if tolerance > 0.0 and (
                float(row["velocity_peak_relative_delta"]) <= tolerance
                and float(row["pressure_range_relative_delta"]) <= tolerance
            ):
                converged = True
                stop_reason = "converged"
                history.append(row)
                break
        else:
            row["velocity_peak_relative_delta"] = ""
            row["pressure_range_relative_delta"] = ""
        history.append(row)
        previous_row = row

    return {
        "preflow_steps_requested": requested_steps,
        "preflow_steps_completed": len(history),
        "preflow_convergence_tolerance": tolerance,
        "preflow_converged": converged,
        "preflow_status": stop_reason,
        "preflow_stop_reason": stop_reason,
        "preflow_history": history,
        "final_stress_marker_diagnostics": (
            markers.stress_marker_diagnostics() if history else []
        ),
        "final_stress_face_diagnostics": (
            markers.stress_face_diagnostics(
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_REGION_ID,
            )
            if history
            else {}
        ),
        "final_flow_field_snapshot": (
            _flow_field_snapshot(fluid)
            if history and bool(getattr(config, "export_final_flow_snapshot", False))
            else {}
        ),
    }


def _pressure_range(row: Mapping[str, object]) -> float:
    return float(row["pressure_max_pa"]) - float(row["pressure_min_pa"])


def _relative_delta(current: object, previous: object) -> float:
    current_value = float(current)
    previous_value = float(previous)
    scale = max(abs(current_value), abs(previous_value), 1.0e-30)
    return abs(current_value - previous_value) / scale


def _flow_field_snapshot(fluid: CartesianFluidSolver) -> dict[str, np.ndarray]:
    snapshot = {
        "pressure": fluid.pressure.to_numpy(),
        "velocity": fluid.velocity.to_numpy(),
        "obstacle": fluid.obstacle.to_numpy(),
        "cell_face_x_m": fluid.cell_face_x_m.to_numpy(),
        "cell_face_y_m": fluid.cell_face_y_m.to_numpy(),
        "cell_face_z_m": fluid.cell_face_z_m.to_numpy(),
        "cell_center_x_m": fluid.cell_center_x_m.to_numpy(),
        "cell_center_y_m": fluid.cell_center_y_m.to_numpy(),
        "cell_center_z_m": fluid.cell_center_z_m.to_numpy(),
        "cell_width_x_m": fluid.cell_width_x_m.to_numpy(),
        "cell_width_y_m": fluid.cell_width_y_m.to_numpy(),
        "cell_width_z_m": fluid.cell_width_z_m.to_numpy(),
    }
    sampling_obstacle = getattr(fluid, "sampling_obstacle", None)
    if sampling_obstacle is not None:
        snapshot["sampling_obstacle"] = sampling_obstacle.to_numpy()
    return snapshot


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
    active_speed = speed[non_obstacle]
    if active_speed.size:
        speed_p99 = float(np.percentile(active_speed, 99.0))
        speed_p999 = float(np.percentile(active_speed, 99.9))
    else:
        speed_p99 = 0.0
        speed_p999 = 0.0
    return {
        "mode": FLOW_SOLUTION_MODE,
        "projection_report": projection_report,
        "obstacle_cell_count": int(obstacle.sum()),
        "fluid_cell_count": int(non_obstacle.sum()),
        "local_velocity_peak_mps": float(active_speed.max(initial=0.0)),
        "fluid_speed_p99_mps": speed_p99,
        "fluid_speed_p999_mps": speed_p999,
        "pressure_min_pa": float(pressure[non_obstacle].min(initial=0.0)),
        "pressure_max_pa": float(pressure[non_obstacle].max(initial=0.0)),
        "pressure_sign_convention": "fluid.pressure projection field is sampled directly",
        **_flow_source_report_fields(projection_report),
    }


def _flow_source_report_fields(report: Any) -> dict[str, object]:
    if not isinstance(report, Mapping):
        return {key: "" for key in FLOW_SOURCE_REPORT_KEYS}
    fields = {key: report.get(key, "") for key in FLOW_SOURCE_REPORT_KEYS}
    if fields["pressure_outlet_flux_ratio"] == "":
        fields["pressure_outlet_flux_ratio"] = report.get(
            "zmin_pressure_outlet_to_abs_source_ratio",
            report.get("zmin_pressure_outlet_to_positive_source_ratio", ""),
        )
    if fields["velocity_outlet_flux_ratio"] == "":
        fields["velocity_outlet_flux_ratio"] = report.get(
            "zmin_velocity_outlet_to_abs_source_ratio",
            report.get("zmin_velocity_outlet_to_positive_source_ratio", ""),
        )
    return fields


def _marker_force_report_fields(report: Any) -> dict[str, object]:
    primary_force = tuple(report.primary_marker_force_n)
    secondary_force = tuple(report.secondary_marker_force_n)
    total_force = tuple(report.total_marker_force_n)
    fluid_reaction = tuple(report.fluid_reaction_force_n)
    primary_plus_secondary_z = float(primary_force[2]) + float(secondary_force[2])
    total_z = float(total_force[2])
    return {
        "primary_face_force_n": primary_force,
        "secondary_face_force_n": secondary_force,
        "primary_face_force_z_N": float(primary_force[2]),
        "secondary_face_force_z_N": float(secondary_force[2]),
        "primary_plus_secondary_force_z_N": primary_plus_secondary_z,
        "force_decomposition_residual_N": abs(primary_plus_secondary_z - total_z),
        "marker_force_z_N": float(total_force[2]),
        "fluid_reaction_force_n": fluid_reaction,
        "fluid_reaction_force_z_N": float(fluid_reaction[2]),
        "marker_action_reaction_residual_n": float(
            report.action_reaction_residual_n
        ),
        "marker_action_reaction_residual_N": float(
            report.action_reaction_residual_n
        ),
        "primary_face_marker_count": int(report.primary_marker_count),
        "secondary_face_marker_count": int(report.secondary_marker_count),
        "total_marker_count": int(report.total_marker_count),
        "primary_face_valid_marker_count": int(
            report.primary_stress_valid_marker_count
        ),
        "secondary_face_valid_marker_count": int(
            report.secondary_stress_valid_marker_count
        ),
        "primary_face_invalid_marker_count": int(
            report.primary_stress_invalid_marker_count
        ),
        "secondary_face_invalid_marker_count": int(
            report.secondary_stress_invalid_marker_count
        ),
        "primary_face_force_norm_sum_N": float(
            report.primary_marker_force_norm_sum_n
        ),
        "secondary_face_force_norm_sum_N": float(
            report.secondary_marker_force_norm_sum_n
        ),
        "total_marker_force_norm_sum_N": float(
            report.total_marker_force_norm_sum_n
        ),
        "primary_face_force_norm_max_N": float(
            report.primary_marker_force_norm_max_n
        ),
        "secondary_face_force_norm_max_N": float(
            report.secondary_marker_force_norm_max_n
        ),
        "total_marker_force_norm_max_N": float(
            report.total_marker_force_norm_max_n
        ),
    }


def _stress_sampling_report_fields(report: Any) -> dict[str, object]:
    return {
        "max_abs_traction_pa": float(report.max_abs_traction_pa),
        "two_sided_pressure_marker_count": int(
            report.two_sided_pressure_marker_count
        ),
        "one_sided_pressure_marker_count": int(
            report.one_sided_pressure_marker_count
        ),
        "two_sided_extended_marker_count": int(
            getattr(report, "two_sided_extended_marker_count", 0)
        ),
        "one_sided_extended_marker_count": int(
            getattr(report, "one_sided_extended_marker_count", 0)
        ),
    }


def _marker_traction_report_fields(markers: HibmMpmSurfaceMarkers) -> dict[str, object]:
    return markers.stress_face_diagnostics(
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
        streamwise_axis_index=STREAMWISE_AXIS_INDEX,
    )


def _scatter_report_fields(report: Any) -> dict[str, object]:
    return {
        "scatter_action_reaction_residual_n": float(
            report.action_reaction_residual_n
        ),
        "scatter_action_reaction_residual_N": float(
            report.action_reaction_residual_n
        ),
    }


def _build_markers(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> HibmMpmSurfaceMarkers:
    markers_per_face = int(config.marker_count)
    marker_layout = _traction_marker_layout(config)
    marker_capacity = (
        markers_per_face
        if marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE
        else 2 * markers_per_face
    )
    markers = HibmMpmSurfaceMarkers(
        marker_capacity=marker_capacity,
        runtime=runtime,
    )
    solid_min, solid_max = _solid_box(config)
    x_center = 0.5 * (solid_min[0] + solid_max[0])
    segment = config.flap_height_m / markers_per_face
    area = config.flap_height_m * (solid_max[0] - solid_min[0]) / markers_per_face
    dz = _grid_spacing_m(config)[2]
    offset = _traction_marker_face_offset_cells(config) * dz
    probe_origin_mode = _traction_pressure_probe_origin_mode(config)
    probe_origin_offset_cells = _traction_pressure_probe_origin_offset_cells(config)
    probe_origin_offset = (
        0.0 if probe_origin_offset_cells is None else probe_origin_offset_cells * dz
    )
    if marker_layout == TRACTION_MARKER_LAYOUT_SINGLE_MID_SURFACE:
        face_specs = (
            (
                0.5 * (solid_min[2] + solid_max[2]),
                0.5 * (solid_min[2] + solid_max[2]),
                (0.0, 0.0, 1.0),
                PRIMARY_REGION_ID,
            ),
        )
    else:
        face_specs = (
            (
                solid_max[2] + offset,
                solid_max[2],
                (0.0, 0.0, 1.0),
                PRIMARY_REGION_ID,
            ),
            (
                solid_min[2] - offset,
                solid_min[2],
                (0.0, 0.0, -1.0),
                SECONDARY_REGION_ID,
            ),
        )
    positions = []
    probe_origins = []
    velocities = []
    normals = []
    areas = []
    regions = []
    for z, physical_face_z, normal, region_id in face_specs:
        for marker in range(markers_per_face):
            y = solid_min[1] + (float(marker) + 0.5) * segment
            positions.append((x_center, y, z))
            if (
                probe_origin_mode
                == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
            ):
                probe_origin_z = physical_face_z + probe_origin_offset * normal[2]
                probe_origins.append((x_center, y, probe_origin_z))
            velocities.append((0.0, 0.0, 0.0))
            normals.append(normal)
            areas.append(area)
            regions.append(region_id)
    markers.load_markers(
        positions_m=positions,
        velocities_mps=velocities,
        normals=normals,
        areas_m2=areas,
        region_ids=regions,
        pressure_probe_origins_m=(
            probe_origins
            if probe_origin_mode == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
            else None
        ),
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
    config: Any | None = None,
) -> Any:
    pressure_sampling_mode = (
        TRACTION_PRESSURE_TWO_SIDED
        if config is None
        else _traction_pressure_sampling_mode(config)
    )
    one_sided_region_id = (
        PRIMARY_REGION_ID
        if pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
        else -1
    )
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
        viscosity_pa_s=0.0 if config is None else _traction_viscosity_pa_s(config),
        two_sided_pressure=True,
        one_sided_pressure_region_id=one_sided_region_id,
        one_sided_reference_pressure_pa=0.0,
        pressure_probe_ladder_start_offset_cells=(
            None
            if config is None
            else _traction_pressure_probe_start_offset_cells(config)
        ),
        pressure_probe_ladder_spacing_cells=(
            0.5
            if config is None
            else _traction_pressure_probe_ladder_spacing_cells(config)
        ),
        pressure_probe_ladder_rung_count=(
            5 if config is None else _traction_pressure_probe_ladder_rung_count(config)
        ),
        pressure_probe_ladder_mode=(
            TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
            if config is None
            else _traction_pressure_probe_ladder_mode(config)
        ),
        pressure_pair_policy=(
            TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER
            if config is None
            else _traction_pressure_pair_policy(config)
        ),
        pressure_pair_max_cell_delta=(
            1 if config is None else _traction_pressure_pair_max_cell_delta(config)
        ),
        pressure_pair_require_opposite_sides=(
            True
            if config is None
            else _traction_pressure_pair_require_opposite_sides(config)
        ),
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
