from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.coupling.hibm_mpm import (
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
)
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState
from simulation_core.coupling.pressure_sample_pairs import (
    PressureSamplePairMap,
    RuntimeAnchoredCellPairProvider,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig


PRIMARY_REGION_ID = 101
SECONDARY_REGION_ID = 202
SECONDARY_UNUSED_REGION_ID = SECONDARY_REGION_ID
STREAMWISE_AXIS_INDEX = 2
OUT_OF_PLANE_AXIS_INDEX = 0
AXIS_NAMES = ("x", "y", "z")
OUT_OF_PLANE_BOUNDARY_POLICY = "finite_slab_x_faces_no_periodic_or_slip"
OUT_OF_PLANE_BOUNDARY_NOTE = (
    "The official case is conceptual 2D. This runner extrudes it into a finite "
    "3D slab and does not yet impose a strict periodic/slip condition on the "
    "out-of-plane x faces, so depth-normalized quantities are diagnostic rather "
    "than a full Fluent parity claim."
)
FLOW_SOLUTION_MODE = "computed_projection"
DEFAULT_SOLID_CFL_TARGET = 0.5
FLOW_DRIVER_PROJECTION_ONLY = "projection_only"
FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC = "reinitialize_inlet_each_step_diagnostic"
FLOW_DRIVER_SUSTAINED_BOUNDARY = "sustained_boundary_inlet"
FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR = "sustained_boundary_predictor"
FLOW_DRIVER_SUSTAINED_SOURCE = "sustained_volume_source_inlet"
FLOW_DRIVER_SUSTAINED_PREDICTOR = "sustained_inlet_predictor"
FLOW_DRIVER_SHARP_REFERENCE = "sharp_hibm_mpm_reference"
FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS = "cell_obstacle_layers"
FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS = "hibm_sharp_marker_rows"
FLOW_SOLID_BOUNDARY_MODES = {
    FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
    FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS,
}
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
TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR = (
    "baseline_anchored_cell_pair"
)
TRACTION_PRESSURE_PAIR_POLICIES = {
    TRACTION_PRESSURE_PAIR_POLICY_INDEPENDENT_LADDER,
    TRACTION_PRESSURE_PAIR_POLICY_SYMMETRIC_CELL_PAIR,
    TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR,
}
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED = "disabled"
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR = (
    "runtime_anchored_cell_pair"
)
TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDERS = {
    TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED,
    TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR,
}
TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED = "disabled"
TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED = "per_face_mirrored"
TRACTION_ONE_SIDED_PRESSURE_POLICIES = {
    TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED,
    TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED,
}
TRACTION_MARKER_FACE_OFFSET_CELLS_DIAGNOSTIC_MAX = 4.0
SUPPORTED_FORMAL_FLOW_DRIVER_MODES = {
    FLOW_DRIVER_PROJECTION_ONLY,
    FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC,
    FLOW_DRIVER_SUSTAINED_BOUNDARY,
    FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
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
FLOW_OBSTACLE_NORMAL_VELOCITY_POLICIES = {"face_clamp", "cell_zero_only"}
FLOW_PRESSURE_OUTLET_BACKFLOW_POLICIES = {"clamp", "allow"}
FLOW_PROJECTION_REPORT_KEYS = (
    "pressure_solver_requested",
    "pressure_solver",
    "pressure_outlet_backflow_policy",
    "obstacle_normal_velocity_policy",
    "pressure_solver_forced_to_fv_cg",
    "pressure_solver_force_reason",
    "pressure_solve_failed",
    "pressure_solve_failure_action",
    "pressure_projection_physical_failure",
    "pressure_projection_physical_failure_reason",
    "pressure_projection_physical_failure_action",
    "pressure_nullspace_policy",
    "l2",
    "max_abs",
    "pre_projection_l2",
    "pre_projection_max_abs",
    "projection_l2",
    "projection_max_abs",
    "post_boundary_l2",
    "post_boundary_max_abs",
    "cg_project_calls",
    "cg_iterations_max",
    "cg_relative_residual_max",
    "cg_converged_all",
    "cg_breakdown_count",
    "cg_breakdown_code",
    "cg_breakdown",
    "flow_symmetry_domain_walls",
    "fsi_pressure_snapshot_updated",
)
FLOW_ADVECTION_SCHEMES = {"euler", "rk2"}
FLOW_PREDICTOR_NO_SLIP_WALL_INDEX = {
    "xmin": 0,
    "xmax": 1,
    "ymin": 2,
    "ymax": 3,
    "zmin": 4,
    "zmax": 5,
}
SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN = "3d_neo_hookean"
SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR = "plane_stress_linear_elastic"
SOLID_CONSTITUTIVE_MODELS = {
    SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
    SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR,
}


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
    anchor_install_report = _install_selected_pressure_pair_anchor_markers(
        markers,
        config,
    )
    pressure_pair_anchor_pair_map = dict(
        anchor_install_report.pop("pressure_pair_anchor_pair_map", {})
    )
    solid = _build_solid(config, runtime)
    fixed_mask, tip_mask = _solid_masks(solid, config)
    # cache the constant rest positions once so the per-step displacement report
    # does not re-fetch the whole rest array from the device every step
    rest_positions_m = solid.rest_x.to_numpy()[: solid.particle_count]
    mu_pa, lambda_pa = _lame_parameters(config)
    solid_substep_cfl = solid_substep_cfl_report(config)
    solid_substeps = int(solid_substep_cfl["solid_substeps_selected"])
    solid_seeding = _enforce_solid_seeding_limit(config)
    preflow_report = _run_fixed_solid_preflow(markers, fluid, solid, config)
    preflow_history = preflow_report["preflow_history"]

    latest_stress_report = None
    latest_force_report = None
    latest_scatter_report = None
    latest_solid_report = None
    latest_feedback_report = None
    latest_dynamic_obstacle_report = _fluid_obstacle_update_disabled_report()
    latest_flow_report = None
    latest_feedback_constraint_report = None
    fluid_projection_count = 0
    fluid_projection_after_feedback_count = 0
    fluid_projection_consumed_feedback_count = 0
    feedback_available_for_projection = False
    feedback_constraint_cells: set[tuple[int, int, int]] = set()
    history: list[dict[str, object]] = []
    apply_feedback = bool(getattr(config, "apply_marker_feedback_to_fluid", True))
    flow_driver_mode = _effective_flow_driver_mode(config, flow_phase="fsi")
    sharp_boundary_cache: dict[str, object] = {}

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
            markers=markers,
            sharp_boundary_cache=sharp_boundary_cache,
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
        solid_substep_velocity_damping = _solid_substep_velocity_damping(
            config,
            solid_substeps=solid_substeps,
        )
        for _solid_substep in range(solid_substeps):
            latest_solid_report = solid.step(
                dt_s=solid_substep_dt_s,
                mu_pa=mu_pa,
                lambda_pa=lambda_pa,
                primary_region_id=PRIMARY_REGION_ID,
                secondary_region_id=SECONDARY_REGION_ID,
                velocity_damping=solid_substep_velocity_damping,
                fixed_node_lock_policy=str(
                    getattr(config, "fixed_node_lock_policy", "any_fixed_particle")
                ),
                constitutive_model=str(
                    getattr(
                        config,
                        "solid_constitutive_model",
                        SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
                    )
                ),
                velocity_transfer_flip_blend=float(
                    getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
                ),
                # Read the per-substep report (a device->host snapshot with the
                # out-of-bounds safety check) only on the final substep. The
                # caller keeps just the last report (used at the per-step history
                # assembly), so this turns solid_substeps host round-trips per
                # step into 1 with bit-for-bit identical results.
                read_report=(_solid_substep == solid_substeps - 1),
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
            preserve_marker_area=bool(
                getattr(config, "preserve_marker_area_during_surface_feedback", False)
            ),
        )
        latest_dynamic_obstacle_report = _update_fluid_obstacle_from_solid(
            fluid,
            solid,
            config,
        )
        feedback_available_for_projection = True
        step_displacement = _solid_displacement_report(
            solid, fixed_mask, tip_mask, rest=rest_positions_m
        )
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
                "flow_predictor_kinematic_viscosity_m2_s": latest_flow_report[
                    "flow_predictor_kinematic_viscosity_m2_s"
                ],
                "flow_predictor_no_slip_domain_walls": latest_flow_report[
                    "flow_predictor_no_slip_domain_walls"
                ],
                "flow_obstacle_no_slip_layers": latest_flow_report[
                    "flow_obstacle_no_slip_layers"
                ],
                "flow_obstacle_no_slip_weight": latest_flow_report[
                    "flow_obstacle_no_slip_weight"
                ],
                "flow_solid_boundary_mode": latest_flow_report[
                    "flow_solid_boundary_mode"
                ],
                "flow_obstacle_normal_velocity_policy": latest_flow_report[
                    "flow_obstacle_normal_velocity_policy"
                ],
                "flow_pressure_outlet_backflow_policy": latest_flow_report[
                    "flow_pressure_outlet_backflow_policy"
                ],
                "hibm_sharp_marker_boundary_enabled": latest_flow_report[
                    "hibm_sharp_marker_boundary_enabled"
                ],
                "hibm_sharp_marker_boundary_search_reused": latest_flow_report[
                    "hibm_sharp_marker_boundary_search_reused"
                ],
                "hibm_sharp_marker_boundary_near_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_near_node_count"
                ],
                "hibm_sharp_marker_boundary_external_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_external_node_count"
                ],
                "hibm_sharp_marker_boundary_internal_node_count": latest_flow_report[
                    "hibm_sharp_marker_boundary_internal_node_count"
                ],
                "hibm_sharp_marker_boundary_internal_obstacle_cell_count": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_internal_obstacle_cell_count"
                    ]
                ),
                "hibm_sharp_marker_boundary_no_slip_rows": latest_flow_report[
                    "hibm_sharp_marker_boundary_no_slip_rows"
                ],
                "hibm_sharp_marker_boundary_pressure_neumann_rows": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_pressure_neumann_rows"
                    ]
                ),
                "hibm_sharp_marker_boundary_pressure_gradient_updated": (
                    latest_flow_report[
                        "hibm_sharp_marker_boundary_pressure_gradient_updated"
                    ]
                ),
                "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_velocity_dirichlet_count"
                    ]
                ),
                "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count"
                    ]
                ),
                "hibm_pressure_neumann_skipped_obstacle_owner_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_skipped_obstacle_owner_count"
                    ]
                ),
                "hibm_pressure_neumann_relocated_obstacle_owner_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_relocated_obstacle_owner_count"
                    ]
                ),
                "hibm_pressure_neumann_duplicate_owner_count": latest_flow_report[
                    "hibm_pressure_neumann_duplicate_owner_count"
                ],
                "hibm_pressure_neumann_invalid_reconstruction_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_reconstruction_count"
                    ]
                ),
                "hibm_pressure_neumann_invalid_unreconstructable_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_unreconstructable_count"
                    ]
                ),
                "hibm_pressure_neumann_invalid_bad_marker_count": latest_flow_report[
                    "hibm_pressure_neumann_invalid_bad_marker_count"
                ],
                "hibm_pressure_neumann_invalid_nonpositive_volume_count": (
                    latest_flow_report[
                        "hibm_pressure_neumann_invalid_nonpositive_volume_count"
                    ]
                ),
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
                "fluid_marker_velocity_constraints_enabled": (
                    latest_feedback_constraint_report[
                        "fluid_marker_velocity_constraints_enabled"
                    ]
                ),
                "fluid_marker_velocity_constraint_active_cell_count": (
                    latest_feedback_constraint_report[
                        "fluid_marker_velocity_constraint_active_cell_count"
                    ]
                ),
                **latest_dynamic_obstacle_report,
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
                **_flow_projection_report_fields(latest_flow_report),
                **_flow_source_report_fields(latest_flow_report),
                "solid_substeps_selected": solid_substeps,
                "solid_constitutive_model": str(
                    getattr(
                        config,
                        "solid_constitutive_model",
                        SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
                    )
                ),
                "solid_fixed_node_lock_policy": str(
                    getattr(config, "fixed_node_lock_policy", "any_fixed_particle")
                ),
                "solid_velocity_transfer_flip_blend": float(
                    getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
                ),
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
                "surface_feedback_preserve_marker_area": bool(
                    getattr(
                        config,
                        "preserve_marker_area_during_surface_feedback",
                        False,
                    )
                ),
                "surface_feedback_geometry_updated_marker_count": (
                    latest_feedback_report.geometry_updated_marker_count
                ),
                "surface_feedback_max_area_change_m2": (
                    latest_feedback_report.max_marker_area_change_m2
                ),
                "total_marker_force_n": latest_force_report.total_marker_force_n,
                **_marker_force_report_fields(latest_force_report),
                **_stress_sampling_report_fields(latest_stress_report),
                **_marker_traction_report_fields(
                    markers, include_face_diagnostics=False
                ),
                **anchor_install_report,
                **_scatter_report_fields(latest_scatter_report),
                "mpm_external_force_n": latest_solid_report.external_force_n,
                "mpm_primary_mean_velocity_mps": (
                    latest_solid_report.primary_mean_velocity_mps
                ),
                "mpm_secondary_mean_velocity_mps": (
                    latest_solid_report.secondary_mean_velocity_mps
                ),
                "mpm_primary_mean_displacement_m": (
                    latest_solid_report.primary_mean_displacement_m
                ),
                "mpm_secondary_mean_displacement_m": (
                    latest_solid_report.secondary_mean_displacement_m
                ),
                "mpm_active_grid_nodes": latest_solid_report.active_grid_nodes,
                "mpm_grid_out_of_bounds_particle_count": (
                    latest_solid_report.grid_out_of_bounds_particle_count
                ),
                "mpm_max_speed_mps": latest_solid_report.max_speed_mps,
                "mpm_deformation_clamp_count": (
                    latest_solid_report.deformation_clamp_count
                ),
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
    pressure_force_source = (
        "total_marker_force_n_pressure_only"
        if not _traction_include_viscous(config)
        else "total_marker_force_n_pressure_plus_viscous"
    )
    slab_diagnostics = slab_equivalence_diagnostics(
        config,
        interface_force_total_n=latest_force_report.total_marker_force_n,
        pressure_force_total_n=latest_force_report.total_marker_force_n,
        marker_total_area_m2=_marker_total_area_m2(markers),
        solid_mass_total_kg=latest_solid_report.total_mass_kg,
        max_displacement_m=max_displacement,
        pressure_force_source=pressure_force_source,
    )

    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        **slab_diagnostics,
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
        "solid_seeding_report": solid_seeding,
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
            "pressure_pa": "fluid.fsi_pressure",
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
        "fluid_marker_velocity_constraints_enabled": (
            latest_feedback_constraint_report[
                "fluid_marker_velocity_constraints_enabled"
            ]
        ),
        "fluid_marker_velocity_constraint_active_cell_count": (
            latest_feedback_constraint_report[
                "fluid_marker_velocity_constraint_active_cell_count"
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
        **_stress_sampling_report_fields(latest_stress_report),
        **_marker_traction_report_fields(markers, include_face_diagnostics=True),
        **anchor_install_report,
        "scatter_invalid_marker_count": latest_scatter_report.invalid_marker_count,
        "scatter_active_marker_count": latest_scatter_report.active_marker_count,
        "scatter_active_particle_count": latest_scatter_report.active_pair_count,
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
        "final_stress_marker_diagnostics": markers.stress_marker_diagnostics(),
        "final_stress_face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
            streamwise_axis_index=STREAMWISE_AXIS_INDEX,
            include_face_diagnostics=True,
        ),
        "pressure_pair_anchor_pair_map": pressure_pair_anchor_pair_map,
        "history": history,
        "max_displacement_m": max_displacement,
        "reference_max_displacement_m": reference_displacement,
        "max_displacement_relative_error": displacement_relative_error,
        "displacement_tolerance": config.displacement_tolerance,
        "final_flow_field_snapshot": (
            _flow_field_snapshot(fluid)
            if history and bool(getattr(config, "export_final_flow_snapshot", False))
            else {}
        ),
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
    slab_diagnostics = slab_equivalence_diagnostics(
        config,
        interface_force_total_n=marker_force,
        pressure_force_total_n=marker_force,
        marker_total_area_m2=_marker_total_area_m2(markers),
        max_displacement_m=displacement["max_displacement_m"],
        pressure_force_source="preflow_total_marker_force_n_pressure_only",
    )
    return {
        "case": case_id,
        "case_metadata": dict(case_metadata),
        "config": asdict(config),
        "flow_solution_mode": FLOW_SOLUTION_MODE,
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        **slab_diagnostics,
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
            "pressure_pa": "fluid.fsi_pressure",
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
        "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
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
        "fluid_projection_consumed_feedback_count": int(
            bool(latest_preflow["fluid_marker_velocity_constraints_enabled"])
        ),
        "fluid_projection_consumed_feedback": bool(
            latest_preflow["fluid_marker_velocity_constraints_enabled"]
        ),
        "fluid_feedback_constraint_marker_count": latest_preflow[
            "fluid_feedback_constraint_marker_count"
        ],
        "fluid_feedback_constraint_active_cell_count": latest_preflow[
            "fluid_feedback_constraint_active_cell_count"
        ],
        "fluid_feedback_constraint_cleared_cell_count": latest_preflow[
            "fluid_feedback_constraint_cleared_cell_count"
        ],
        "fluid_feedback_constraint_obstacle_cell_count": latest_preflow[
            "fluid_feedback_constraint_obstacle_cell_count"
        ],
        "fluid_feedback_constraint_non_obstacle_cell_count": latest_preflow[
            "fluid_feedback_constraint_non_obstacle_cell_count"
        ],
        "fluid_feedback_constraint_projection_participating_cell_count": latest_preflow[
            "fluid_feedback_constraint_projection_participating_cell_count"
        ],
        "fluid_marker_velocity_constraints_enabled": latest_preflow[
            "fluid_marker_velocity_constraints_enabled"
        ],
        "fluid_marker_velocity_constraint_active_cell_count": latest_preflow[
            "fluid_marker_velocity_constraint_active_cell_count"
        ],
        "no_slip_residual_before_mps": latest_preflow[
            "no_slip_residual_before_mps"
        ],
        "no_slip_residual_after_mps": latest_preflow["no_slip_residual_after_mps"],
        "no_slip_target_residual_after_assembly_mps": latest_preflow[
            "no_slip_target_residual_after_assembly_mps"
        ],
        "no_slip_projected_residual_after_projection_mps": latest_preflow[
            "no_slip_projected_residual_after_projection_mps"
        ],
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


def _traction_one_sided_pressure_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_one_sided_pressure_policy",
            TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED,
        )
    )


def _traction_one_sided_primary_fluid_side_normal_sign(config: Any) -> float | None:
    value = getattr(config, "traction_one_sided_primary_fluid_side_normal_sign", None)
    if value is None:
        return None
    return float(value)


def _traction_one_sided_secondary_fluid_side_normal_sign(config: Any) -> float | None:
    value = getattr(config, "traction_one_sided_secondary_fluid_side_normal_sign", None)
    if value is None:
        return None
    return float(value)


def _traction_one_sided_primary_reference_pressure_pa(config: Any) -> float:
    return float(getattr(config, "traction_one_sided_primary_reference_pressure_pa", 0.0))


def _traction_one_sided_secondary_reference_pressure_pa(config: Any) -> float:
    return float(
        getattr(config, "traction_one_sided_secondary_reference_pressure_pa", 0.0)
    )


def _traction_one_sided_pressure_pair_policy(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_one_sided_pressure_pair_policy",
            TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR,
        )
    )


def _traction_pressure_pair_anchor_markers_json(config: Any) -> str | None:
    value = getattr(config, "traction_pressure_pair_anchor_markers_json", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _traction_pressure_pair_runtime_provider_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "traction_pressure_pair_runtime_provider_mode",
            TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_DISABLED,
        )
    )


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
        and _traction_one_sided_pressure_policy(config)
        == TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        and _traction_one_sided_primary_fluid_side_normal_sign(config) is None
        and _traction_one_sided_secondary_fluid_side_normal_sign(config) is None
        and _traction_one_sided_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and _traction_pressure_pair_anchor_markers_json(config) is None
    )


def _is_selected_traction_formulation_coupled_smoke(config: Any) -> bool:
    if not bool(getattr(config, "allow_selected_traction_formulation_coupled_smoke", False)):
        return False
    max_selected_step_count = (
        50
        if bool(
            getattr(
                config,
                "allow_selected_traction_formulation_coupled_long_validation",
                False,
            )
        )
        else 10
    )
    return (
        0 < int(getattr(config, "step_count", 0)) <= max_selected_step_count
        and _traction_marker_layout(config)
        == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES
        and _traction_pressure_sampling_mode(config) == TRACTION_PRESSURE_ONE_SIDED
        and math.isclose(_traction_marker_face_offset_cells(config), 0.51)
        and not _traction_include_viscous(config)
        and _traction_pressure_probe_origin_mode(config)
        == TRACTION_PRESSURE_PROBE_ORIGIN_PHYSICAL_FACE_OFFSET
        and math.isclose(
            float(_traction_pressure_probe_origin_offset_cells(config) or -1.0),
            0.51,
        )
        and _traction_pressure_probe_start_offset_cells(config) is None
        and _traction_pressure_probe_ladder_mode(config)
        == TRACTION_PRESSURE_PROBE_LADDER_CURRENT_NORMAL_CELL
        and _traction_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and _traction_pressure_pair_max_cell_delta(config) == 1
        and _traction_pressure_pair_require_opposite_sides(config)
        and _traction_one_sided_pressure_policy(config)
        == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
        and _traction_one_sided_primary_fluid_side_normal_sign(config) == 1.0
        and _traction_one_sided_secondary_fluid_side_normal_sign(config) == 1.0
        and _traction_one_sided_pressure_pair_policy(config)
        == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
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
        if (
            _traction_one_sided_pressure_policy(config)
            == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
        ):
            return True, "supported"
        return (
            False,
            "dual-face one-sided pressure requires "
            "traction_one_sided_pressure_policy='per_face_mirrored'",
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
    solid_boundary_mode = _flow_solid_boundary_mode(config)
    if solid_boundary_mode not in FLOW_SOLID_BOUNDARY_MODES:
        raise ValueError(f"unsupported flow_solid_boundary_mode: {solid_boundary_mode!r}")
    for mode_field_name in ("flow_driver_mode", "preflow_flow_driver_mode"):
        flow_driver_mode = getattr(config, mode_field_name, None)
        if flow_driver_mode is None and mode_field_name == "preflow_flow_driver_mode":
            continue
        flow_driver_mode = str(
            FLOW_DRIVER_PROJECTION_ONLY
            if flow_driver_mode is None
            else flow_driver_mode
        )
        if not flow_driver_mode and mode_field_name == "preflow_flow_driver_mode":
            continue
        if flow_driver_mode not in SUPPORTED_FORMAL_FLOW_DRIVER_MODES:
            raise ValueError(f"unsupported {mode_field_name}: {flow_driver_mode!r}")
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
    advection_scheme = str(getattr(config, "flow_advection_scheme", "euler")).lower()
    if advection_scheme not in FLOW_ADVECTION_SCHEMES:
        raise ValueError(f"unsupported flow_advection_scheme: {advection_scheme!r}")
    predictor_substeps = int(getattr(config, "flow_predictor_substeps", 1))
    if predictor_substeps <= 0:
        raise ValueError("flow_predictor_substeps must be positive")
    ymin_no_slip_rows = int(getattr(config, "flow_ymin_no_slip_rows", 0))
    if ymin_no_slip_rows < 0:
        raise ValueError("flow_ymin_no_slip_rows must be non-negative")
    obstacle_no_slip_layers = int(getattr(config, "flow_obstacle_no_slip_layers", 0))
    if obstacle_no_slip_layers < 0:
        raise ValueError("flow_obstacle_no_slip_layers must be non-negative")
    obstacle_no_slip_weight = float(getattr(config, "flow_obstacle_no_slip_weight", 1.0))
    if not math.isfinite(obstacle_no_slip_weight) or not 0.0 <= obstacle_no_slip_weight <= 1.0:
        raise ValueError("flow_obstacle_no_slip_weight must be in [0, 1]")
    obstacle_cap_no_slip_weight = getattr(
        config,
        "flow_obstacle_cap_no_slip_weight",
        None,
    )
    if obstacle_cap_no_slip_weight is not None:
        obstacle_cap_no_slip_weight = float(obstacle_cap_no_slip_weight)
        if (
            not math.isfinite(obstacle_cap_no_slip_weight)
            or not 0.0 <= obstacle_cap_no_slip_weight <= 1.0
        ):
            raise ValueError("flow_obstacle_cap_no_slip_weight must be in [0, 1]")
    obstacle_wake_no_slip_layers = int(
        getattr(config, "flow_obstacle_wake_no_slip_layers", 0)
    )
    if obstacle_wake_no_slip_layers < 0:
        raise ValueError("flow_obstacle_wake_no_slip_layers must be non-negative")
    obstacle_wake_no_slip_weight = float(
        getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5)
    )
    if (
        not math.isfinite(obstacle_wake_no_slip_weight)
        or not 0.0 <= obstacle_wake_no_slip_weight <= 1.0
    ):
        raise ValueError("flow_obstacle_wake_no_slip_weight must be in [0, 1]")
    viscosity_multiplier = float(
        getattr(config, "flow_predictor_kinematic_viscosity_multiplier", 1.0)
    )
    if not math.isfinite(viscosity_multiplier) or viscosity_multiplier < 0.0:
        raise ValueError(
            "flow_predictor_kinematic_viscosity_multiplier must be finite and non-negative"
        )
    _flow_predictor_no_slip_domain_walls(config)
    _flow_symmetry_domain_walls(config)
    constraint_blend = float(getattr(config, "marker_velocity_constraint_blend", 1.0))
    if not math.isfinite(constraint_blend) or not 0.0 <= constraint_blend <= 1.0:
        raise ValueError("marker_velocity_constraint_blend must be in [0, 1]")
    constraint_mobility_ratio = float(
        getattr(config, "marker_velocity_constraint_solid_mobility_ratio", 0.0)
    )
    if not math.isfinite(constraint_mobility_ratio) or constraint_mobility_ratio < 0.0:
        raise ValueError(
            "marker_velocity_constraint_solid_mobility_ratio must be finite and non-negative"
        )
    ramp_steps = int(getattr(config, "flow_inlet_source_ramp_steps", 0))
    if ramp_steps < 0:
        raise ValueError("flow_inlet_source_ramp_steps must be non-negative")
    outlet_policy = str(getattr(config, "flow_outlet_balance_policy", "report_only"))
    if outlet_policy not in FLOW_OUTLET_BALANCE_POLICIES:
        raise ValueError(f"unsupported flow_outlet_balance_policy: {outlet_policy!r}")
    pressure_outlet_backflow_policy = str(
        getattr(config, "flow_pressure_outlet_backflow_policy", "clamp")
    )
    if pressure_outlet_backflow_policy not in FLOW_PRESSURE_OUTLET_BACKFLOW_POLICIES:
        raise ValueError(
            "unsupported flow_pressure_outlet_backflow_policy: "
            f"{pressure_outlet_backflow_policy!r}"
        )
    obstacle_normal_velocity_policy = str(
        getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp")
    )
    if obstacle_normal_velocity_policy not in FLOW_OBSTACLE_NORMAL_VELOCITY_POLICIES:
        raise ValueError(
            "unsupported flow_obstacle_normal_velocity_policy: "
            f"{obstacle_normal_velocity_policy!r}"
        )
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
    runtime_pair_provider = _traction_pressure_pair_runtime_provider_mode(config)
    if runtime_pair_provider not in TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDERS:
        raise ValueError(
            "unsupported traction_pressure_pair_runtime_provider_mode: "
            f"{runtime_pair_provider!r}"
        )
    pressure_pair_max_cell_delta = _traction_pressure_pair_max_cell_delta(config)
    if pressure_pair_max_cell_delta < 0:
        raise ValueError("traction_pressure_pair_max_cell_delta must be non-negative")
    one_sided_policy = _traction_one_sided_pressure_policy(config)
    if one_sided_policy not in TRACTION_ONE_SIDED_PRESSURE_POLICIES:
        raise ValueError(
            f"unsupported traction_one_sided_pressure_policy: {one_sided_policy!r}"
        )
    one_sided_pair_policy = _traction_one_sided_pressure_pair_policy(config)
    if one_sided_pair_policy not in TRACTION_PRESSURE_PAIR_POLICIES:
        raise ValueError(
            "unsupported traction_one_sided_pressure_pair_policy: "
            f"{one_sided_pair_policy!r}"
        )
    primary_side_sign = _traction_one_sided_primary_fluid_side_normal_sign(config)
    secondary_side_sign = _traction_one_sided_secondary_fluid_side_normal_sign(config)
    primary_reference_pressure = _traction_one_sided_primary_reference_pressure_pa(config)
    secondary_reference_pressure = (
        _traction_one_sided_secondary_reference_pressure_pa(config)
    )
    if not math.isfinite(primary_reference_pressure):
        raise ValueError(
            "traction_one_sided_primary_reference_pressure_pa must be finite"
        )
    if not math.isfinite(secondary_reference_pressure):
        raise ValueError(
            "traction_one_sided_secondary_reference_pressure_pa must be finite"
        )
    if one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED:
        if marker_layout != TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
            raise ValueError(
                "per_face_mirrored one-sided pressure requires dual_physical_faces"
            )
        if pressure_sampling_mode != TRACTION_PRESSURE_ONE_SIDED:
            raise ValueError(
                "per_face_mirrored one-sided pressure requires "
                "traction_pressure_sampling_mode='one_sided_surface_pressure'"
            )
        if primary_side_sign not in (-1.0, 1.0):
            raise ValueError(
                "traction_one_sided_primary_fluid_side_normal_sign must be -1.0 or 1.0"
            )
        if secondary_side_sign not in (-1.0, 1.0):
            raise ValueError(
                "traction_one_sided_secondary_fluid_side_normal_sign must be -1.0 or 1.0"
            )
        if one_sided_pair_policy != pressure_pair_policy:
            raise ValueError(
                "traction_one_sided_pressure_pair_policy must match "
                "traction_pressure_pair_policy for per-face diagnostics"
            )
    elif pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED:
        if marker_layout == TRACTION_MARKER_LAYOUT_DUAL_PHYSICAL_FACES:
            raise ValueError(
                "dual-face one-sided pressure requires "
                "traction_one_sided_pressure_policy='per_face_mirrored'"
            )
    traction_viscosity = _traction_viscosity_pa_s(config)
    if not math.isfinite(traction_viscosity) or traction_viscosity < 0.0:
        raise ValueError("traction viscosity must be finite and non-negative")
    if config.step_count < 0:
        raise ValueError("step_count must be non-negative")
    if (
        config.step_count > 0
        and not _is_default_traction_formulation(config)
        and not _is_selected_traction_formulation_coupled_smoke(config)
    ):
        raise ValueError(
            "non-default traction formulations are fixed-solid diagnostics only"
        )
    anchor_markers_json = _traction_pressure_pair_anchor_markers_json(config)
    if (
        _is_selected_traction_formulation_coupled_smoke(config)
        and pressure_pair_policy == TRACTION_PRESSURE_PAIR_POLICY_BASELINE_ANCHORED_CELL_PAIR
        and anchor_markers_json is None
        and runtime_pair_provider
        != TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR
    ):
        raise ValueError(
            "selected coupled smoke requires "
            "traction_pressure_pair_anchor_markers_json"
        )
    if (
        anchor_markers_json is not None
        and not _is_selected_traction_formulation_coupled_smoke(config)
    ):
        raise ValueError(
            "traction_pressure_pair_anchor_markers_json is selected coupled smoke only"
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
    solid_velocity_transfer_flip_blend = float(
        getattr(config, "solid_velocity_transfer_flip_blend", 0.0)
    )
    if not 0.0 <= solid_velocity_transfer_flip_blend <= 1.0:
        raise ValueError("solid_velocity_transfer_flip_blend must be in [0, 1]")
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
    solid_model = str(
        getattr(
            config,
            "solid_constitutive_model",
            SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
        )
    )
    if solid_model not in SOLID_CONSTITUTIVE_MODELS:
        raise ValueError(
            f"solid_constitutive_model must be one of "
            f"{sorted(SOLID_CONSTITUTIVE_MODELS)!r}; got {solid_model!r}"
        )


def _flow_solid_boundary_mode(config: Any) -> str:
    return str(
        getattr(
            config,
            "flow_solid_boundary_mode",
            FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
        )
    )


def _flow_pressure_outlet_backflow_policy(config: Any) -> str:
    return str(getattr(config, "flow_pressure_outlet_backflow_policy", "clamp"))


def _flow_obstacle_normal_velocity_policy(config: Any) -> str:
    return str(getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp"))


def _use_hibm_sharp_marker_boundary(config: Any) -> bool:
    return (
        _flow_solid_boundary_mode(config)
        == FLOW_SOLID_BOUNDARY_HIBM_SHARP_MARKER_ROWS
    )


def _domain_bounds(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (0.0, 0.0, 0.0),
        (config.span_m, 0.5 * config.duct_height_m, config.duct_length_m),
    )


def _official_streamwise_to_solver_z(config: Any, streamwise_m: float) -> float:
    return float(config.duct_length_m) - float(streamwise_m)


def _solid_box(config: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center_z = 0.5 * config.duct_length_m
    z_min = getattr(config, "flap_streamwise_min_m", None)
    z_max = getattr(config, "flap_streamwise_max_m", None)
    if z_min is None or z_max is None:
        z_min = center_z - 0.5 * config.flap_thickness_m
        z_max = center_z + 0.5 * config.flap_thickness_m
    else:
        solver_z_min = _official_streamwise_to_solver_z(config, z_max)
        solver_z_max = _official_streamwise_to_solver_z(config, z_min)
        z_min = min(solver_z_min, solver_z_max)
        z_max = max(solver_z_min, solver_z_max)
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


def _solid_mpm_bounds(
    config: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    domain_min, domain_max = _domain_bounds(config)
    solid_min, solid_max = _solid_box(config)
    base_dy = (float(domain_max[1]) - float(domain_min[1])) / float(
        config.grid_nodes[1]
    )
    pad_y = 3.0 * base_dy
    return (
        (
            float(domain_min[0]),
            min(float(domain_min[1]), float(solid_min[1]) - pad_y),
            float(domain_min[2]),
        ),
        (
            float(domain_max[0]),
            max(float(domain_max[1]), float(solid_max[1]) + pad_y),
            float(domain_max[2]),
        ),
    )


def _solid_mpm_grid_spacing_m(config: Any) -> tuple[float, float, float]:
    bounds_min, bounds_max = _solid_mpm_bounds(config)
    grid_nodes = tuple(int(value) for value in config.grid_nodes)
    return tuple(
        (float(max_value) - float(min_value)) / float(node_count)
        for min_value, max_value, node_count in zip(
            bounds_min,
            bounds_max,
            grid_nodes,
            strict=True,
        )
    )


def _lame_parameters(config: Any) -> tuple[float, float]:
    young = float(config.young_modulus_pa)
    nu = float(config.poisson_ratio)
    mu = young / (2.0 * (1.0 + nu))
    solid_model = str(
        getattr(
            config,
            "solid_constitutive_model",
            SOLID_CONSTITUTIVE_MODEL_3D_NEO_HOOKEAN,
        )
    )
    if solid_model == SOLID_CONSTITUTIVE_MODEL_PLANE_STRESS_LINEAR:
        lam = young * nu / (1.0 - nu * nu)
    else:
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def solid_substep_cfl_report(config: Any) -> dict[str, object]:
    mu, lam = _lame_parameters(config)
    wave_speed_mps = math.sqrt(
        (lam + 2.0 * mu) / float(config.solid_density_kgm3)
    )
    min_spacing_m = min(_solid_mpm_grid_spacing_m(config))
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


def solid_seeding_report(config: Any) -> dict[str, object]:
    """Per-axis MPM particle spacing relative to the solid background grid.

    Explicit MPM loses inter-particle grid connectivity when particle
    spacing exceeds roughly one background cell: the quadratic B-spline
    stencils of adjacent particle layers stop sharing well-supported nodes,
    the body numerically fractures, and a fixed-root cantilever free-falls
    instead of bending (observed on the 2026-07-02 fine flap campaign:
    grid 4x256x320 with solid_particle_counts (1, 64, 12) put ~2 cells
    between wall-normal particle layers and ejected particles by step 30,
    while (1, 256, 20) on the same grid rings stably about the
    Euler-Bernoulli static deflection).

    The span axis (x) is excluded from the guard: the vertical-flap slab is
    a 2D-equivalent extrusion with an x-uniform solution, where a single
    particle column is intentional and empirically sound.
    """
    solid_min, solid_max = _solid_box(config)
    grid_spacing_m = _solid_mpm_grid_spacing_m(config)
    particle_counts = tuple(int(value) for value in config.solid_particle_counts)
    particle_spacing_m = tuple(
        (float(solid_max[axis]) - float(solid_min[axis]))
        / float(max(particle_counts[axis], 1))
        for axis in range(3)
    )
    spacing_cells = tuple(
        particle_spacing_m[axis] / grid_spacing_m[axis] for axis in range(3)
    )
    max_spacing_cells = float(
        getattr(config, "solid_seeding_max_spacing_cells", 1.5)
    )
    guard_enabled = bool(getattr(config, "enforce_solid_seeding_limit", False))
    guarded_axes = (1, 2)  # wall-normal (y) and streamwise (z); x is span
    worst_guarded_spacing_cells = max(
        spacing_cells[axis] for axis in guarded_axes
    )
    return {
        "solid_particle_spacing_m": particle_spacing_m,
        "solid_grid_spacing_m": grid_spacing_m,
        "solid_particle_spacing_cells": spacing_cells,
        "solid_seeding_guarded_axes": guarded_axes,
        "solid_seeding_worst_guarded_spacing_cells": worst_guarded_spacing_cells,
        "solid_seeding_max_spacing_cells": max_spacing_cells,
        "solid_seeding_guard_enabled": guard_enabled,
        "solid_seeding_guard_satisfied": (
            worst_guarded_spacing_cells <= max_spacing_cells
        ),
    }


def _enforce_solid_seeding_limit(config: Any) -> dict[str, object]:
    report = solid_seeding_report(config)
    if report["solid_seeding_guard_enabled"] and not report[
        "solid_seeding_guard_satisfied"
    ]:
        spacing_cells = report["solid_particle_spacing_cells"]
        raise ValueError(
            "solid particle seeding is too sparse for the MPM background "
            f"grid: particle spacing per cell (x, y, z) = "
            f"({spacing_cells[0]:.2f}, {spacing_cells[1]:.2f}, "
            f"{spacing_cells[2]:.2f}) exceeds "
            f"{report['solid_seeding_max_spacing_cells']:.2f} on a guarded "
            "axis (y/z); increase solid_particle_counts so adjacent particle "
            "layers stay within ~1 background cell, or disable "
            "enforce_solid_seeding_limit"
        )
    return report


def _solid_substep_velocity_damping(config: Any, *, solid_substeps: int) -> float:
    substeps = int(solid_substeps)
    if substeps <= 0:
        raise ValueError("solid_substeps must be positive")
    damping = float(getattr(config, "velocity_damping", 1.0))
    if damping < 0.0:
        raise ValueError("velocity_damping must be non-negative")
    if damping == 0.0 or substeps == 1:
        return damping
    return damping ** (1.0 / float(substeps))


def _grid_spacing_m(config: Any) -> tuple[float, float, float]:
    bounds_min, bounds_max = _domain_bounds(config)
    return tuple(
        (float(bounds_max[axis]) - float(bounds_min[axis]))
        / float(config.grid_nodes[axis])
        for axis in range(3)
    )


def _positive_finite(value: object, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _finite_vector3(
    value: tuple[float, float, float] | list[float] | np.ndarray | None,
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _divide_vector3(
    value: tuple[float, float, float],
    denominator: float,
) -> tuple[float, float, float]:
    return tuple(float(component) / denominator for component in value)


def _marker_total_area_m2(markers: HibmMpmSurfaceMarkers) -> float:
    marker_count = int(getattr(markers, "marker_count", 0))
    if marker_count <= 0:
        return 0.0
    return float(np.sum(markers.A_gamma_m2.to_numpy()[:marker_count]))


def slab_equivalence_diagnostics(
    config: Any,
    *,
    interface_force_total_n: tuple[float, float, float]
    | list[float]
    | np.ndarray
    | None = None,
    pressure_force_total_n: tuple[float, float, float]
    | list[float]
    | np.ndarray
    | None = None,
    marker_total_area_m2: float | None = None,
    solid_mass_total_kg: float | None = None,
    max_displacement_m: float | None = None,
    flap_count: int = 1,
    marker_face_count: int | None = None,
    conceptual_coordinate_model: str = "cartesian-2d",
    runtime_discretization_model: str = "cartesian-3d-half-domain",
    out_of_plane_boundary_policy: str = OUT_OF_PLANE_BOUNDARY_POLICY,
    pressure_force_source: str = "marker_traction_pressure_integral",
) -> dict[str, object]:
    extrusion_depth_m = _positive_finite(
        getattr(config, "span_m"),
        field_name="span_m/extrusion_depth_m",
    )
    solid_min, solid_max = _solid_box(config)
    flap_height_m = _positive_finite(
        float(solid_max[1]) - float(solid_min[1]),
        field_name="flap_height_m",
    )
    flap_streamwise_thickness_m = _positive_finite(
        float(solid_max[2]) - float(solid_min[2]),
        field_name="flap_streamwise_thickness_m",
    )
    resolved_flap_count = int(flap_count)
    if resolved_flap_count <= 0:
        raise ValueError("flap_count must be positive")
    resolved_marker_face_count = (
        _traction_marker_face_count(config)
        if marker_face_count is None
        else int(marker_face_count)
    )
    if resolved_marker_face_count <= 0:
        raise ValueError("marker_face_count must be positive")
    expected_marker_total_area_m2 = (
        float(resolved_marker_face_count) * flap_height_m * extrusion_depth_m
    )
    resolved_marker_total_area_m2 = (
        expected_marker_total_area_m2
        if marker_total_area_m2 is None
        else float(marker_total_area_m2)
    )
    if (
        not math.isfinite(resolved_marker_total_area_m2)
        or resolved_marker_total_area_m2 < 0.0
    ):
        raise ValueError("marker_total_area_m2 must be finite and non-negative")

    expected_solid_volume_m3 = (
        float(resolved_flap_count)
        * extrusion_depth_m
        * flap_height_m
        * flap_streamwise_thickness_m
    )
    expected_solid_mass_kg = (
        expected_solid_volume_m3 * float(config.solid_density_kgm3)
    )
    resolved_solid_mass_kg = (
        expected_solid_mass_kg
        if solid_mass_total_kg is None
        else float(solid_mass_total_kg)
    )
    if not math.isfinite(resolved_solid_mass_kg) or resolved_solid_mass_kg < 0.0:
        raise ValueError("solid_mass_total_kg must be finite and non-negative")

    interface_force = _finite_vector3(
        interface_force_total_n,
        field_name="interface_force_total_n",
    )
    pressure_force = _finite_vector3(
        pressure_force_total_n,
        field_name="pressure_force_total_n",
    )
    interface_force_per_depth = _divide_vector3(interface_force, extrusion_depth_m)
    pressure_force_per_depth = _divide_vector3(pressure_force, extrusion_depth_m)

    displacement_value: float | str = (
        "" if max_displacement_m is None else float(max_displacement_m)
    )
    if displacement_value != "" and not math.isfinite(float(displacement_value)):
        raise ValueError("max_displacement_m must be finite")

    return {
        "conceptual_coordinate_model": str(conceptual_coordinate_model),
        "runtime_discretization_model": str(runtime_discretization_model),
        "streamwise_axis": AXIS_NAMES[STREAMWISE_AXIS_INDEX],
        "out_of_plane_axis": AXIS_NAMES[OUT_OF_PLANE_AXIS_INDEX],
        "extrusion_depth_m": extrusion_depth_m,
        "extrusion_depth_source": "VerticalFlapFsiConfig.span_m",
        "span_m": extrusion_depth_m,
        "span_is_extrusion_depth": True,
        "flap_streamwise_thickness_m": flap_streamwise_thickness_m,
        "flap_streamwise_thickness_source": (
            "VerticalFlapFsiConfig.flap_thickness_m"
        ),
        "flap_thickness_is_streamwise_not_extrusion": True,
        "flap_count": resolved_flap_count,
        "marker_face_count": int(resolved_marker_face_count),
        "marker_total_area_m2": resolved_marker_total_area_m2,
        "marker_expected_total_area_m2": expected_marker_total_area_m2,
        "marker_total_area_per_depth_m": (
            resolved_marker_total_area_m2 / extrusion_depth_m
        ),
        "solid_volume_total_m3": expected_solid_volume_m3,
        "solid_volume_per_depth_m2": expected_solid_volume_m3 / extrusion_depth_m,
        "solid_mass_total_kg": resolved_solid_mass_kg,
        "solid_expected_mass_total_kg": expected_solid_mass_kg,
        "solid_mass_per_depth_kgpm": resolved_solid_mass_kg / extrusion_depth_m,
        "interface_force_total_n": interface_force,
        "interface_force_z_N": interface_force[2],
        "interface_force_per_depth_npm": interface_force_per_depth,
        "interface_force_z_per_depth_N_per_m": interface_force_per_depth[2],
        "pressure_force_total_n": pressure_force,
        "pressure_force_z_N": pressure_force[2],
        "pressure_force_per_depth_npm": pressure_force_per_depth,
        "pressure_force_z_per_depth_N_per_m": pressure_force_per_depth[2],
        "pressure_force_source": str(pressure_force_source),
        "max_displacement_m": displacement_value,
        "displacement_depth_scaling_expectation": (
            "depth_invariant_when_force_and_mass_scale_together"
        ),
        "out_of_plane_boundary_policy": str(out_of_plane_boundary_policy),
        "out_of_plane_boundary_residual_modeling_error": (
            str(out_of_plane_boundary_policy) != "strict_periodic_or_slip"
        ),
        "out_of_plane_boundary_note": OUT_OF_PLANE_BOUNDARY_NOTE,
        "fluent_parity_claimed": False,
    }


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
    fluid.obstacle.from_numpy(_initial_fluid_obstacle(config))
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


def _initial_fluid_obstacle(config: Any) -> np.ndarray:
    if _use_hibm_sharp_marker_boundary(config):
        return np.zeros(tuple(int(value) for value in config.grid_nodes), dtype=np.int32)
    return _solid_obstacle(config)


def _fluid_obstacle_update_disabled_report() -> dict[str, object]:
    return {
        "fluid_dynamic_obstacle_update_enabled": False,
        "fluid_dynamic_obstacle_cell_count": "",
        "fluid_dynamic_obstacle_added_cell_count": "",
        "fluid_dynamic_obstacle_removed_cell_count": "",
    }


def _update_fluid_obstacle_from_solid(
    fluid: CartesianFluidSolver,
    solid: NeoHookeanMpmState,
    config: Any,
) -> dict[str, object]:
    if (
        not bool(getattr(config, "update_fluid_obstacle_from_solid", False))
        or _use_hibm_sharp_marker_boundary(config)
    ):
        return _fluid_obstacle_update_disabled_report()

    device_update = getattr(fluid, "update_dynamic_flap_obstacle_from_particles", None)
    row_count = int(config.solid_particle_counts[1])
    if device_update is not None and row_count <= int(config.grid_nodes[1]):
        solid_min, solid_max = _solid_box(config)
        report = device_update(
            solid.x,
            solid.rest_x,
            particle_count=solid.particle_count,
            row_count=row_count,
            solid_min_m=solid_min,
            solid_max_m=solid_max,
            flap_height_m=float(config.flap_height_m),
            min_thickness_m=float(config.flap_thickness_m),
        )
        return {
            "fluid_dynamic_obstacle_update_enabled": True,
            **report,
        }

    previous_obstacle = fluid.obstacle.to_numpy()
    obstacle = _solid_obstacle_from_mpm_particles(solid, config)
    velocity = fluid.velocity.to_numpy()
    velocity_prev = fluid.velocity_prev.to_numpy()
    solid_cells = obstacle != 0
    velocity[solid_cells] = 0.0
    velocity_prev[solid_cells] = 0.0
    fluid.obstacle.from_numpy(obstacle)
    fluid.velocity.from_numpy(velocity)
    fluid.velocity_prev.from_numpy(velocity_prev)
    return {
        "fluid_dynamic_obstacle_update_enabled": True,
        "fluid_dynamic_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "fluid_dynamic_obstacle_added_cell_count": int(
            np.count_nonzero((obstacle != 0) & (previous_obstacle == 0))
        ),
        "fluid_dynamic_obstacle_removed_cell_count": int(
            np.count_nonzero((obstacle == 0) & (previous_obstacle != 0))
        ),
    }


def _solid_obstacle_from_mpm_particles(
    solid: NeoHookeanMpmState,
    config: Any,
) -> np.ndarray:
    nx, ny, nz = config.grid_nodes
    bounds_min, bounds_max = _domain_bounds(config)
    dx = (bounds_max[0] - bounds_min[0]) / nx
    dy = (bounds_max[1] - bounds_min[1]) / ny
    dz = (bounds_max[2] - bounds_min[2]) / nz
    positions = solid.x.to_numpy()[: solid.particle_count]
    rest = solid.rest_x.to_numpy()[: solid.particle_count]
    obstacle = np.zeros((nx, ny, nz), dtype=np.int32)
    if positions.size == 0:
        return obstacle

    solid_min, solid_max = _solid_box(config)
    row_height = float(config.flap_height_m) / float(config.solid_particle_counts[1])
    x_min = float(solid_min[0])
    x_max = float(solid_max[0])
    # Group by rest-y row so the deformed flap remains a continuous thin wall
    # instead of a cloud of isolated obstacle cells on coarse grids.
    row_count = int(config.solid_particle_counts[1])
    row_indices = np.clip(
        np.floor((rest[:, 1] - solid_min[1]) / max(row_height, 1.0e-12)).astype(int),
        0,
        row_count - 1,
    )

    row_particle_count = np.bincount(row_indices, minlength=row_count).astype(np.int32)
    active_rows = row_particle_count > 0
    y_sum = np.bincount(row_indices, weights=positions[:, 1], minlength=row_count)
    y_center = np.zeros(row_count, dtype=np.float64)
    y_center[active_rows] = y_sum[active_rows] / row_particle_count[active_rows]
    y_min = y_center - 0.5 * row_height
    y_max = y_center + 0.5 * row_height

    z_min = np.full(row_count, np.inf, dtype=np.float64)
    z_max = np.full(row_count, -np.inf, dtype=np.float64)
    np.minimum.at(z_min, row_indices, positions[:, 2])
    np.maximum.at(z_max, row_indices, positions[:, 2])
    # Keep at least the physical thickness represented even when all
    # particles in a row compress into the same streamwise cell.
    too_thin = active_rows & (
        (z_max - z_min) < 0.25 * float(config.flap_thickness_m)
    )
    z_mid = 0.5 * (z_min[too_thin] + z_max[too_thin])
    half_thickness = 0.5 * float(config.flap_thickness_m)
    z_min[too_thin] = z_mid - half_thickness
    z_max[too_thin] = z_mid + half_thickness

    x_cell_min = bounds_min[0] + np.arange(nx, dtype=np.float64) * dx
    y_cell_min = bounds_min[1] + np.arange(ny, dtype=np.float64) * dy
    z_cell_min = bounds_min[2] + np.arange(nz, dtype=np.float64) * dz
    x_overlap = (x_cell_min < x_max) & ((x_cell_min + dx) > x_min)
    y_overlap = (
        active_rows[:, None]
        & (y_cell_min[None, :] < y_max[:, None])
        & ((y_cell_min[None, :] + dy) > y_min[:, None])
    )
    z_overlap = (
        active_rows[:, None]
        & (z_cell_min[None, :] < z_max[:, None])
        & ((z_cell_min[None, :] + dz) > z_min[:, None])
    )
    yz_overlap = np.any(y_overlap[:, :, None] & z_overlap[:, None, :], axis=0)
    obstacle[x_overlap, :, :] = yz_overlap.astype(np.int32)
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
    values = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    weights = np.zeros((nx, ny, nz), dtype=np.float32)
    _apply_ymin_no_slip_rows(
        active,
        values,
        weights,
        obstacle,
        config,
    )
    if not _use_hibm_sharp_marker_boundary(config):
        _apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )
    active[:, :, nz - 1] = 1
    values[:, :, nz - 1, STREAMWISE_AXIS_INDEX] = -float(config.inlet_velocity_mps)
    weights[:, :, nz - 1] = 1.0
    active[obstacle != 0] = 0
    values[obstacle != 0] = 0.0
    weights[obstacle != 0] = 0.0
    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
    zero_pressure = np.zeros((nx, ny, nz), dtype=np.float32)
    fluid.pressure.from_numpy(zero_pressure)
    fluid.fsi_pressure.from_numpy(zero_pressure)
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
    preserve_velocity_constraints = bool(
        getattr(config, "preserve_marker_velocity_constraints", True)
    )
    projection_report = dict(
        fluid.project(
            iterations=config.flow_projection_iterations,
            pressure_outlet_zmin=bool(
                getattr(config, "flow_pressure_outlet_enabled", True)
            ),
            pressure_outlet_backflow_policy=str(
                getattr(config, "flow_pressure_outlet_backflow_policy", "clamp")
            ),
            obstacle_normal_velocity_policy=str(
                getattr(config, "flow_obstacle_normal_velocity_policy", "face_clamp")
            ),
            preserve_velocity_constraints=preserve_velocity_constraints,
            velocity_constraint_blend=float(
                getattr(config, "marker_velocity_constraint_blend", 1.0)
            ),
            velocity_constraint_solid_mobility_ratio=float(
                getattr(
                    config,
                    "marker_velocity_constraint_solid_mobility_ratio",
                    0.0,
                )
            ),
            reset_pressure=reset_pressure,
            pressure_solver=config.flow_pressure_solver,
            cg_tolerance=config.flow_cg_tolerance,
            divergence_cleanup_iterations=config.flow_divergence_cleanup_iterations,
            velocity_inlet_zmax=bool(
                getattr(config, "flow_projection_velocity_inlet_zmax", False)
            ),
        )
    )
    symmetry_domain_walls = _flow_symmetry_domain_walls(config)
    if any(symmetry_domain_walls):
        fluid.apply_symmetry_domain_walls(symmetry_domain_walls)
    projection_report["flow_symmetry_domain_walls"] = [
        bool(flag) for flag in symmetry_domain_walls
    ]
    projection_report.update(
        fluid.pressure_outlet_fv_flux_report(dt_s=float(config.dt_s))
    )
    projection_report["fsi_pressure_snapshot_updated"] = bool(
        fluid.snapshot_pressure(preserve_if_current_is_zero=True)
    )
    return _flow_state_report(
        fluid,
        projection_report,
        include_percentiles=bool(getattr(config, "flow_report_include_percentiles", False)),
    )


def _empty_hibm_sharp_marker_boundary_report() -> dict[str, object]:
    return {
        "flow_solid_boundary_mode": FLOW_SOLID_BOUNDARY_CELL_OBSTACLE_LAYERS,
        "hibm_sharp_marker_boundary_enabled": False,
        "hibm_sharp_marker_boundary_search_reused": False,
        "hibm_sharp_marker_boundary_near_node_count": 0,
        "hibm_sharp_marker_boundary_external_node_count": 0,
        "hibm_sharp_marker_boundary_internal_node_count": 0,
        "hibm_sharp_marker_boundary_internal_obstacle_cell_count": 0,
        "hibm_sharp_marker_boundary_no_slip_rows": 0,
        "hibm_sharp_marker_boundary_pressure_neumann_rows": 0,
        "hibm_sharp_marker_boundary_pressure_gradient_updated": False,
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": 0,
        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": 0,
        "hibm_pressure_neumann_skipped_obstacle_owner_count": 0,
        "hibm_pressure_neumann_relocated_obstacle_owner_count": 0,
        "hibm_pressure_neumann_duplicate_owner_count": 0,
        "hibm_pressure_neumann_invalid_reconstruction_count": 0,
        "hibm_pressure_neumann_invalid_unreconstructable_count": 0,
        "hibm_pressure_neumann_invalid_bad_marker_count": 0,
        "hibm_pressure_neumann_invalid_nonpositive_volume_count": 0,
    }


def _hibm_sharp_search_radius_m(config: Any) -> float:
    configured = getattr(config, "flow_hibm_sharp_search_radius_m", None)
    if configured is not None:
        return float(configured)
    return 2.5 * max(_grid_spacing_m(config))


def _hibm_sharp_interior_probe_distance_m(config: Any) -> float:
    configured = getattr(config, "flow_hibm_sharp_interior_probe_distance_m", None)
    if configured is not None:
        return float(configured)
    return 1.5 * max(_grid_spacing_m(config))


def _apply_hibm_sharp_marker_boundary_to_fluid(
    markers: HibmMpmSurfaceMarkers | None,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    update_pressure_gradient: bool,
    boundary_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    if not _use_hibm_sharp_marker_boundary(config):
        return _empty_hibm_sharp_marker_boundary_report()
    if markers is None:
        raise ValueError("hibm_sharp_marker_rows requires surface markers")

    bounds_min, bounds_max = _domain_bounds(config)
    marker_capacity = max(
        int(getattr(markers, "marker_capacity", 0)),
        int(getattr(markers, "marker_count", 0)),
        1,
    )
    search_radius_m = _hibm_sharp_search_radius_m(config)
    interior_probe_distance_m = _hibm_sharp_interior_probe_distance_m(config)
    cache_key = (
        tuple(config.grid_nodes),
        tuple(float(value) for value in bounds_min),
        tuple(float(value) for value in bounds_max),
        int(marker_capacity),
        float(search_radius_m),
        float(interior_probe_distance_m),
    )
    cache_entry = (
        boundary_cache.get("hibm_sharp_marker_boundary")
        if boundary_cache is not None
        else None
    )
    search_reused = bool(
        isinstance(cache_entry, dict) and cache_entry.get("cache_key") == cache_key
    )
    if search_reused:
        ib_search = cache_entry["ib_search"]
        ib_boundary = cache_entry["ib_boundary"]
    else:
        runtime = TaichiRuntimeConfig(arch="cuda")
        ib_search = HibmMpmIbNodeSearch(
            grid_nodes=tuple(config.grid_nodes),
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        ib_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=tuple(config.grid_nodes),
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        if boundary_cache is not None:
            boundary_cache["hibm_sharp_marker_boundary"] = {
                "cache_key": cache_key,
                "ib_search": ib_search,
                "ib_boundary": ib_boundary,
            }
    search_report = ib_search.search_and_classify_grid_fields(
        markers,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        search_radius_m=search_radius_m,
        interior_probe_distance_m=interior_probe_distance_m,
        classify_far_internal_nodes=False,
    )
    internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
        ib_search.node_kind_code,
        internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        convert_internal_nodes=True,
    )
    if update_pressure_gradient:
        markers.update_pressure_neumann_gradient_from_fluid_predictor(
            ib_boundary.marker_pressure_neumann_gradient_field,
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=float(config.air_density_kgm3),
            dt_s=float(config.dt_s),
            probe_distance_m=interior_probe_distance_m,
        )
    ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            ib_boundary.marker_pressure_neumann_gradient_field
        ),
    )
    velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
        fluid.velocity_dirichlet_boundary_active,
        fluid.velocity_dirichlet_boundary_value_mps,
        fluid.velocity_dirichlet_boundary_projection_weight,
        fluid.obstacle,
        fluid.velocity,
        ib_search,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
        velocity_dirichlet_marker_region_id=(
            fluid.velocity_dirichlet_boundary_marker_region_id
        ),
        marker_region_id=markers.region_id,
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
        interpolate_interior_velocity=bool(
            getattr(config, "flow_hibm_sharp_interpolate_velocity_rows", True)
        ),
    )
    fluid.clear_pressure_interface_matrix_terms()
    pressure_report = ib_boundary.assemble_pressure_neumann_matrix_rows(
        fluid.pressure_interface_matrix_diagonal,
        fluid.pressure_interface_matrix_rhs,
        fluid.pressure_interface_coupling_active,
        fluid.pressure_interface_coupling_neighbor,
        fluid.pressure_interface_coupling_coefficient,
        fluid.obstacle,
        fluid.velocity_dirichlet_boundary_active,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        ib_search,
        markers,
        pressure_coupling_extra_neighbor=(
            fluid.pressure_interface_coupling_extra_neighbor
        ),
        pressure_coupling_extra_coefficient=(
            fluid.pressure_interface_coupling_extra_coefficient
        ),
        pressure_interface_row_count=fluid.pressure_interface_row_count,
        pressure_interface_row_owner=fluid.pressure_interface_row_owner,
        pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
        pressure_interface_row_transmissibility=(
            fluid.pressure_interface_row_transmissibility
        ),
        pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
    )
    return {
        "flow_solid_boundary_mode": _flow_solid_boundary_mode(config),
        "hibm_sharp_marker_boundary_enabled": True,
        "hibm_sharp_marker_boundary_search_reused": bool(search_reused),
        "hibm_sharp_marker_boundary_near_node_count": (
            search_report.near_boundary_node_count
        ),
        "hibm_sharp_marker_boundary_external_node_count": (
            search_report.external_ib_node_count
        ),
        "hibm_sharp_marker_boundary_internal_node_count": (
            search_report.internal_node_count
        ),
        "hibm_sharp_marker_boundary_internal_obstacle_cell_count": (
            int(internal_obstacle_cell_count)
        ),
        "hibm_sharp_marker_boundary_no_slip_rows": (
            velocity_report.active_velocity_dirichlet_rows
        ),
        "hibm_sharp_marker_boundary_pressure_neumann_rows": (
            pressure_report.active_pressure_neumann_rows
        ),
        "hibm_sharp_marker_boundary_pressure_gradient_updated": bool(
            update_pressure_gradient
        ),
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
            pressure_report.skipped_velocity_dirichlet_row_count
        ),
        "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
            pressure_report.skipped_pressure_boundary_adjacent_row_count
        ),
        "hibm_pressure_neumann_skipped_obstacle_owner_count": (
            pressure_report.skipped_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_relocated_obstacle_owner_count": (
            pressure_report.relocated_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_duplicate_owner_count": (
            pressure_report.duplicate_owner_row_count
        ),
        "hibm_pressure_neumann_invalid_reconstruction_count": (
            pressure_report.invalid_reconstruction_row_count
        ),
        "hibm_pressure_neumann_invalid_unreconstructable_count": (
            pressure_report.invalid_unreconstructable_row_count
        ),
        "hibm_pressure_neumann_invalid_bad_marker_count": (
            pressure_report.invalid_bad_marker_row_count
        ),
        "hibm_pressure_neumann_invalid_nonpositive_volume_count": (
            pressure_report.invalid_nonpositive_volume_row_count
        ),
    }


def _flow_predictor_kinematic_viscosity_m2_s(config: Any) -> float:
    molecular_nu = float(getattr(config, "air_viscosity_pa_s", 0.0)) / max(
        float(getattr(config, "air_density_kgm3", 1.0)),
        1.0e-30,
    )
    multiplier = float(
        getattr(config, "flow_predictor_kinematic_viscosity_multiplier", 1.0)
    )
    return molecular_nu * multiplier


def _flow_predictor_no_slip_domain_walls(
    config: Any,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    return _flow_domain_wall_flags(
        config,
        field_name="flow_predictor_no_slip_domain_walls",
    )


def _flow_symmetry_domain_walls(
    config: Any,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    return _flow_domain_wall_flags(
        config,
        field_name="flow_symmetry_domain_walls",
    )


def _flow_domain_wall_flags(
    config: Any,
    *,
    field_name: str,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    raw_walls = getattr(config, field_name, ())
    if raw_walls is None:
        names: tuple[str, ...] = ()
    elif isinstance(raw_walls, str):
        names = tuple(
            part.strip().lower()
            for part in raw_walls.split(",")
            if part.strip()
        )
    else:
        names = tuple(str(part).strip().lower() for part in raw_walls if str(part).strip())
    flags = [False, False, False, False, False, False]
    unsupported = sorted(
        {name for name in names if name not in FLOW_PREDICTOR_NO_SLIP_WALL_INDEX}
    )
    if unsupported:
        raise ValueError(
            f"unsupported {field_name} entries: {unsupported!r}"
        )
    for name in names:
        flags[FLOW_PREDICTOR_NO_SLIP_WALL_INDEX[name]] = True
    return tuple(flags)


def _apply_ymin_no_slip_rows(
    active: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    obstacle: np.ndarray,
    config: Any,
) -> None:
    rows = int(getattr(config, "flow_ymin_no_slip_rows", 0))
    if rows <= 0:
        return
    row_count = min(rows, active.shape[1])
    fluid_rows = obstacle[:, :row_count, :] == 0
    active_rows = active[:, :row_count, :]
    values_rows = values[:, :row_count, :, :]
    weights_rows = weights[:, :row_count, :]
    preserved_rows = active_rows != 0
    apply_rows = np.logical_and(fluid_rows, ~preserved_rows)
    clear_rows = np.logical_and(~fluid_rows, ~preserved_rows)

    active_rows[apply_rows] = 1
    values_rows[apply_rows, :] = 0.0
    weights_rows[apply_rows] = 1.0

    active_rows[clear_rows] = 0
    values_rows[clear_rows, :] = 0.0
    weights_rows[clear_rows] = 0.0


def _apply_obstacle_no_slip_rows(
    active: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    obstacle: np.ndarray,
    config: Any,
) -> int:
    layers = int(getattr(config, "flow_obstacle_no_slip_layers", 0))
    wake_layers = int(getattr(config, "flow_obstacle_wake_no_slip_layers", 0))
    if layers <= 0 and wake_layers <= 0:
        return 0
    weight = float(getattr(config, "flow_obstacle_no_slip_weight", 1.0))
    cap_weight_config = getattr(config, "flow_obstacle_cap_no_slip_weight", None)
    cap_weight = weight if cap_weight_config is None else float(cap_weight_config)
    wake_weight = float(getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5))
    fluid_mask = obstacle == 0
    solid_front = obstacle != 0
    selected = np.zeros_like(fluid_mask, dtype=bool)
    row_weights = np.zeros_like(weights, dtype=np.float32)
    for _layer in range(layers):
        x_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        x_adjacent[1:, :, :] |= solid_front[:-1, :, :]
        x_adjacent[:-1, :, :] |= solid_front[1:, :, :]
        y_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        y_adjacent[:, 1:, :] |= solid_front[:, :-1, :]
        y_adjacent[:, :-1, :] |= solid_front[:, 1:, :]
        z_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        z_adjacent[:, :, 1:] |= solid_front[:, :, :-1]
        z_adjacent[:, :, :-1] |= solid_front[:, :, 1:]
        adjacent_by_axis = (x_adjacent, y_adjacent, z_adjacent)
        adjacent = np.zeros_like(fluid_mask, dtype=bool)
        cap_adjacent = np.zeros_like(fluid_mask, dtype=bool)
        for axis, axis_adjacent in enumerate(adjacent_by_axis):
            adjacent |= axis_adjacent
            if axis != STREAMWISE_AXIS_INDEX:
                cap_adjacent |= axis_adjacent
        layer_cells = adjacent & fluid_mask & ~selected
        selected |= layer_cells
        row_weights[layer_cells] = weight
        row_weights[layer_cells & cap_adjacent] = cap_weight
        solid_front = solid_front | layer_cells
    solid_mask = obstacle != 0
    for layer_index in range(1, wake_layers + 1):
        shifted = np.zeros_like(fluid_mask, dtype=bool)
        shifted[:, :, :-layer_index] |= solid_mask[:, :, layer_index:]
        layer_cells = shifted & fluid_mask & ~selected
        selected |= layer_cells
        row_weights[layer_cells] = wake_weight
    constrained = selected & (row_weights > 0.0)
    active[constrained] = 1
    values[constrained] = 0.0
    weights[constrained] = row_weights[constrained]
    return int(np.count_nonzero(constrained))


def _flow_advance_current_step(
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    markers: HibmMpmSurfaceMarkers | None = None,
    sharp_boundary_cache: dict[str, object] | None = None,
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
    mode = _effective_flow_driver_mode(config, flow_phase=flow_phase)
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
    elif mode in {
        FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
        FLOW_DRIVER_SUSTAINED_SOURCE,
        FLOW_DRIVER_SUSTAINED_PREDICTOR,
    }:
        boundary_report = _refresh_zmax_inlet_boundary(fluid, config)
        volume_source_applied = mode in {
            FLOW_DRIVER_SUSTAINED_SOURCE,
            FLOW_DRIVER_SUSTAINED_PREDICTOR,
        }
        if volume_source_applied:
            source_factor = _flow_inlet_source_factor(config, source_schedule_step_index)
            source_normal_velocity_mps = -float(config.inlet_velocity_mps) * source_factor
            fluid.add_zmax_velocity_inlet_volume_source(
                normal_velocity_mps=source_normal_velocity_mps,
            )
        else:
            source_factor = 0.0
            source_normal_velocity_mps = 0.0
        predictor_applied = mode in {
            FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
            FLOW_DRIVER_SUSTAINED_PREDICTOR,
        }
        predictor_note = ""
        predictor_kinematic_viscosity_m2_s = 0.0
        predictor_no_slip_domain_walls = _flow_predictor_no_slip_domain_walls(config)
        if predictor_applied:
            advection_scheme = str(
                getattr(config, "flow_advection_scheme", "euler")
            ).lower()
            predictor_substeps = int(getattr(config, "flow_predictor_substeps", 1))
            predictor_dt_s = float(config.dt_s) / float(predictor_substeps)
            predictor_kinematic_viscosity_m2_s = (
                _flow_predictor_kinematic_viscosity_m2_s(config)
            )
            for _predictor_substep in range(predictor_substeps):
                fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
                fluid.predict(
                    dt_s=predictor_dt_s,
                    advection_scheme=advection_scheme,
                    kinematic_viscosity_m2_s=predictor_kinematic_viscosity_m2_s,
                    no_slip_domain_walls=predictor_no_slip_domain_walls,
                )
            fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
            predictor_note = (
                "core fluid predictor applied before pressure projection "
                f"(advection_scheme={advection_scheme}, "
                f"substeps={predictor_substeps}, "
                f"nu={predictor_kinematic_viscosity_m2_s:g} m^2/s, "
                f"no_slip_domain_walls={predictor_no_slip_domain_walls})"
            )
        driver_report = _flow_driver_report(
            mode=mode,
            full_field_reinitialized=False,
            inlet_boundary_report=boundary_report,
            volume_source_applied=volume_source_applied,
            source_factor=source_factor,
            source_normal_velocity_mps=source_normal_velocity_mps,
            predictor_applied=predictor_applied,
            predictor_note=predictor_note,
            predictor_kinematic_viscosity_m2_s=predictor_kinematic_viscosity_m2_s,
            predictor_no_slip_domain_walls=predictor_no_slip_domain_walls,
        )
    elif mode == FLOW_DRIVER_SHARP_REFERENCE:
        raise RuntimeError(
            "sharp_hibm_mpm_reference is reserved for a sharp-path runner"
        )
    else:  # pragma: no cover - protected by config validation.
        raise RuntimeError(f"unsupported flow_driver_mode: {mode!r}")

    sharp_boundary_report = _apply_hibm_sharp_marker_boundary_to_fluid(
        markers,
        fluid,
        config,
        update_pressure_gradient=True,
        boundary_cache=sharp_boundary_cache,
    )
    flow_report = _project_current_flow(
        fluid,
        config,
        reset_pressure=reset_pressure,
    )
    flow_report.update(sharp_boundary_report)
    flow_report.update(driver_report)
    flow_report["flow_phase"] = str(flow_phase)
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
    flow_report["flow_obstacle_no_slip_layers"] = int(
        getattr(config, "flow_obstacle_no_slip_layers", 0)
    )
    flow_report["flow_obstacle_no_slip_weight"] = float(
        getattr(config, "flow_obstacle_no_slip_weight", 1.0)
    )
    cap_no_slip_weight = getattr(config, "flow_obstacle_cap_no_slip_weight", None)
    flow_report["flow_obstacle_cap_no_slip_weight"] = (
        None if cap_no_slip_weight is None else float(cap_no_slip_weight)
    )
    flow_report["flow_obstacle_wake_no_slip_layers"] = int(
        getattr(config, "flow_obstacle_wake_no_slip_layers", 0)
    )
    flow_report["flow_obstacle_wake_no_slip_weight"] = float(
        getattr(config, "flow_obstacle_wake_no_slip_weight", 0.5)
    )
    flow_report["flow_solid_boundary_mode"] = _flow_solid_boundary_mode(config)
    flow_report["flow_obstacle_normal_velocity_policy"] = (
        _flow_obstacle_normal_velocity_policy(config)
    )
    flow_report["flow_pressure_outlet_backflow_policy"] = (
        _flow_pressure_outlet_backflow_policy(config)
    )
    flow_report["flow_projection_velocity_inlet_zmax"] = bool(
        getattr(config, "flow_projection_velocity_inlet_zmax", False)
    )
    return flow_report


def _effective_flow_driver_mode(config: Any, *, flow_phase: str = "fsi") -> str:
    if bool(getattr(config, "flow_reinitialize_inlet_each_step", False)):
        return FLOW_DRIVER_REINITIALIZE_DIAGNOSTIC
    if str(flow_phase) == "preflow":
        preflow_mode = getattr(config, "preflow_flow_driver_mode", None)
        if preflow_mode is not None and str(preflow_mode):
            return str(preflow_mode)
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
    predictor_kinematic_viscosity_m2_s: float = 0.0,
    predictor_no_slip_domain_walls: tuple[bool, bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
        False,
    ),
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
        "flow_predictor_kinematic_viscosity_m2_s": float(
            predictor_kinematic_viscosity_m2_s
        ),
        "flow_predictor_no_slip_domain_walls": [
            bool(flag) for flag in predictor_no_slip_domain_walls
        ],
    }


def _zmax_inlet_boundary_device_refresh_compatible(config: Any) -> bool:
    if int(getattr(config, "flow_ymin_no_slip_rows", 0)) > 0:
        return False
    if _use_hibm_sharp_marker_boundary(config):
        return True
    return (
        int(getattr(config, "flow_obstacle_no_slip_layers", 0)) <= 0
        and int(getattr(config, "flow_obstacle_wake_no_slip_layers", 0)) <= 0
    )


def _refresh_zmax_inlet_boundary(
    fluid: CartesianFluidSolver,
    config: Any,
) -> dict[str, object]:
    device_refresh = getattr(fluid, "refresh_zmax_inlet_boundary", None)
    if (
        device_refresh is not None
        and _zmax_inlet_boundary_device_refresh_compatible(config)
    ):
        return dict(
            device_refresh(
                inlet_velocity_mps=float(config.inlet_velocity_mps),
                streamwise_axis_index=STREAMWISE_AXIS_INDEX,
            )
        )

    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    weights = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
    obstacle = fluid.obstacle.to_numpy()
    k = int(config.grid_nodes[2]) - 1
    _apply_ymin_no_slip_rows(
        active,
        values,
        weights,
        obstacle,
        config,
    )
    if not _use_hibm_sharp_marker_boundary(config):
        _apply_obstacle_no_slip_rows(
            active,
            values,
            weights,
            obstacle,
            config,
        )
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
    device_report = getattr(fluid, "zmax_inlet_boundary_report", None)
    if device_report is not None:
        return dict(device_report())
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
    previous_feedback_constraint_cells: set[tuple[int, int, int]] = set()
    converged = requested_steps == 0
    stop_reason = "not_requested" if requested_steps == 0 else "max_steps"
    sharp_boundary_cache: dict[str, object] = {}

    for preflow_index in range(requested_steps):
        if _flow_driver_requires_full_field_reinitialize(
            _effective_flow_driver_mode(config, flow_phase="preflow")
        ):
            _initialize_computed_flow(fluid, config)
        feedback_constraint_report = _apply_marker_feedback_to_fluid(
            markers,
            fluid,
            config,
            feedback_available=bool(getattr(config, "apply_marker_feedback_to_fluid", True)),
            previous_feedback_constraint_cells=previous_feedback_constraint_cells,
        )
        previous_feedback_constraint_cells = set(
            feedback_constraint_report.get("_feedback_constraint_cells", set())
        )
        flow_report = _flow_advance_current_step(
            fluid,
            config,
            markers=markers,
            sharp_boundary_cache=sharp_boundary_cache,
            flow_phase="preflow",
            step_index_local=preflow_index,
            step_index_global=preflow_index,
            preflow_history=history,
            reset_pressure=(
                bool(getattr(config, "flow_reset_pressure_each_step", False))
                or preflow_index == 0
            ),
        )
        feedback_constraint_report[
            "no_slip_projected_residual_after_projection_mps"
        ] = _measure_projected_no_slip_residual(
            markers,
            fluid,
            config,
            feedback_consumed=bool(
                feedback_constraint_report[
                    "fluid_marker_velocity_constraints_enabled"
                ]
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
            "flow_predictor_kinematic_viscosity_m2_s": flow_report[
                "flow_predictor_kinematic_viscosity_m2_s"
            ],
            "flow_predictor_no_slip_domain_walls": flow_report[
                "flow_predictor_no_slip_domain_walls"
            ],
            "flow_obstacle_no_slip_layers": flow_report["flow_obstacle_no_slip_layers"],
            "flow_obstacle_no_slip_weight": flow_report["flow_obstacle_no_slip_weight"],
            "flow_solid_boundary_mode": flow_report["flow_solid_boundary_mode"],
            "flow_obstacle_normal_velocity_policy": flow_report[
                "flow_obstacle_normal_velocity_policy"
            ],
            "flow_pressure_outlet_backflow_policy": flow_report[
                "flow_pressure_outlet_backflow_policy"
            ],
            "hibm_sharp_marker_boundary_enabled": flow_report[
                "hibm_sharp_marker_boundary_enabled"
            ],
            "hibm_sharp_marker_boundary_search_reused": flow_report[
                "hibm_sharp_marker_boundary_search_reused"
            ],
            "hibm_sharp_marker_boundary_near_node_count": flow_report[
                "hibm_sharp_marker_boundary_near_node_count"
            ],
            "hibm_sharp_marker_boundary_external_node_count": flow_report[
                "hibm_sharp_marker_boundary_external_node_count"
            ],
            "hibm_sharp_marker_boundary_internal_node_count": flow_report[
                "hibm_sharp_marker_boundary_internal_node_count"
            ],
            "hibm_sharp_marker_boundary_internal_obstacle_cell_count": flow_report[
                "hibm_sharp_marker_boundary_internal_obstacle_cell_count"
            ],
            "hibm_sharp_marker_boundary_no_slip_rows": flow_report[
                "hibm_sharp_marker_boundary_no_slip_rows"
            ],
            "hibm_sharp_marker_boundary_pressure_neumann_rows": flow_report[
                "hibm_sharp_marker_boundary_pressure_neumann_rows"
            ],
            "hibm_sharp_marker_boundary_pressure_gradient_updated": flow_report[
                "hibm_sharp_marker_boundary_pressure_gradient_updated"
            ],
            "hibm_pressure_neumann_skipped_velocity_dirichlet_count": flow_report[
                "hibm_pressure_neumann_skipped_velocity_dirichlet_count"
            ],
            "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
                flow_report[
                    "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count"
                ]
            ),
            "hibm_pressure_neumann_skipped_obstacle_owner_count": flow_report[
                "hibm_pressure_neumann_skipped_obstacle_owner_count"
            ],
            "hibm_pressure_neumann_relocated_obstacle_owner_count": flow_report[
                "hibm_pressure_neumann_relocated_obstacle_owner_count"
            ],
            "hibm_pressure_neumann_duplicate_owner_count": flow_report[
                "hibm_pressure_neumann_duplicate_owner_count"
            ],
            "hibm_pressure_neumann_invalid_reconstruction_count": flow_report[
                "hibm_pressure_neumann_invalid_reconstruction_count"
            ],
            "hibm_pressure_neumann_invalid_unreconstructable_count": flow_report[
                "hibm_pressure_neumann_invalid_unreconstructable_count"
            ],
            "hibm_pressure_neumann_invalid_bad_marker_count": flow_report[
                "hibm_pressure_neumann_invalid_bad_marker_count"
            ],
            "hibm_pressure_neumann_invalid_nonpositive_volume_count": flow_report[
                "hibm_pressure_neumann_invalid_nonpositive_volume_count"
            ],
            "flow_inlet_boundary_active_cell_count": flow_report[
                "flow_inlet_boundary_active_cell_count"
            ],
            "flow_inlet_boundary_obstacle_cell_count": flow_report[
                "flow_inlet_boundary_obstacle_cell_count"
            ],
            "apply_marker_feedback_to_fluid": bool(
                getattr(config, "apply_marker_feedback_to_fluid", True)
            ),
            "fluid_marker_velocity_constraints_enabled": (
                feedback_constraint_report["fluid_marker_velocity_constraints_enabled"]
            ),
            "fluid_marker_velocity_constraint_active_cell_count": (
                feedback_constraint_report[
                    "fluid_marker_velocity_constraint_active_cell_count"
                ]
            ),
            "fluid_feedback_constraint_marker_count": (
                feedback_constraint_report["fluid_feedback_constraint_marker_count"]
            ),
            "fluid_feedback_constraint_active_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_active_cell_count"]
            ),
            "fluid_feedback_constraint_cleared_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_cleared_cell_count"]
            ),
            "fluid_feedback_constraint_obstacle_cell_count": (
                feedback_constraint_report["fluid_feedback_constraint_obstacle_cell_count"]
            ),
            "fluid_feedback_constraint_non_obstacle_cell_count": (
                feedback_constraint_report[
                    "fluid_feedback_constraint_non_obstacle_cell_count"
                ]
            ),
            "fluid_feedback_constraint_projection_participating_cell_count": (
                feedback_constraint_report[
                    "fluid_feedback_constraint_projection_participating_cell_count"
                ]
            ),
            "no_slip_residual_before_mps": (
                feedback_constraint_report["no_slip_residual_before_mps"]
            ),
            "no_slip_residual_after_mps": (
                feedback_constraint_report["no_slip_residual_after_mps"]
            ),
            "no_slip_target_residual_after_assembly_mps": (
                feedback_constraint_report["no_slip_target_residual_after_assembly_mps"]
            ),
            "no_slip_projected_residual_after_projection_mps": (
                feedback_constraint_report[
                    "no_slip_projected_residual_after_projection_mps"
                ]
            ),
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
            **_flow_projection_report_fields(flow_report),
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
            "scatter_active_particle_count": scatter_report.active_pair_count,
            **_marker_force_report_fields(force_report),
            **_stress_sampling_report_fields(stress_report),
            **_marker_traction_report_fields(
                markers, include_face_diagnostics=False
            ),
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
                include_face_diagnostics=True,
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
        "pressure": _fluid_feedback_pressure_numpy(fluid),
        "velocity": fluid.velocity.to_numpy(),
        "obstacle": fluid.obstacle.to_numpy(),
        "velocity_dirichlet_boundary_active": (
            fluid.velocity_dirichlet_boundary_active.to_numpy()
        ),
        "velocity_dirichlet_boundary_projection_weight": (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        ),
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


def _fluid_feedback_pressure_field(fluid: CartesianFluidSolver):
    return getattr(fluid, "fsi_pressure", fluid.pressure)


def _fluid_feedback_pressure_numpy(fluid: CartesianFluidSolver) -> np.ndarray:
    return _fluid_feedback_pressure_field(fluid).to_numpy()


def _apply_marker_feedback_to_fluid(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_available: bool,
    previous_feedback_constraint_cells: set[tuple[int, int, int]],
) -> dict[str, object]:
    apply_device = getattr(fluid, "apply_marker_feedback_constraints", None)
    marker_region_field = getattr(markers, "region_id", None)
    if apply_device is not None and marker_region_field is not None:
        report = apply_device(
            markers.x_gamma_m,
            markers.v_gamma_mps,
            marker_region_field,
            int(markers.marker_count),
            feedback_available=feedback_available,
            preserve_velocity_constraints=bool(
                getattr(config, "preserve_marker_velocity_constraints", True)
            ),
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
        )
        report["_feedback_constraint_cells"] = set()
        return report

    return _apply_marker_feedback_to_fluid_host_fallback(
        markers,
        fluid,
        config,
        feedback_available=feedback_available,
        previous_feedback_constraint_cells=previous_feedback_constraint_cells,
    )


def _apply_marker_feedback_to_fluid_host_fallback(
    markers: HibmMpmSurfaceMarkers,
    fluid: CartesianFluidSolver,
    config: Any,
    *,
    feedback_available: bool,
    previous_feedback_constraint_cells: set[tuple[int, int, int]],
) -> dict[str, object]:
    preserve_velocity_constraints = bool(
        getattr(config, "preserve_marker_velocity_constraints", True)
    )
    _clear_fluid_velocity_constraints(fluid)

    active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    values = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
    weights = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
    row_region_field = getattr(
        fluid,
        "velocity_dirichlet_boundary_marker_region_id",
        None,
    )
    row_regions = (
        row_region_field.to_numpy()
        if row_region_field is not None
        else None
    )

    cleared_cell_count = 0
    for i, j, k in previous_feedback_constraint_cells:
        active[i, j, k] = 0
        values[i, j, k] = 0.0
        weights[i, j, k] = 0.0
        if row_regions is not None:
            row_regions[i, j, k] = -1
        cleared_cell_count += 1

    if not feedback_available:
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        if row_region_field is not None and row_regions is not None:
            row_region_field.from_numpy(row_regions)
        return _empty_feedback_constraint_report(cleared_cell_count)

    marker_count = int(markers.marker_count)
    if marker_count <= 0:
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        if row_region_field is not None and row_regions is not None:
            row_region_field.from_numpy(row_regions)
        return _empty_feedback_constraint_report(cleared_cell_count)

    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_velocities = markers.v_gamma_mps.to_numpy()[:marker_count]
    marker_region_ids = _marker_region_ids(markers, marker_count)
    velocity = fluid.velocity.to_numpy()

    marker_cells = _marker_grid_cells(marker_positions, config)

    target_sum: dict[tuple[int, int, int], np.ndarray] = {}
    target_count: dict[tuple[int, int, int], int] = {}
    target_regions: dict[tuple[int, int, int], set[int]] = {}
    target_primary_sum: dict[tuple[int, int, int], np.ndarray] = {}
    target_primary_count: dict[tuple[int, int, int], int] = {}
    target_secondary_sum: dict[tuple[int, int, int], np.ndarray] = {}
    target_secondary_count: dict[tuple[int, int, int], int] = {}
    before_residuals: list[float] = []
    for marker_index, (cell, marker_velocity) in enumerate(
        zip(marker_cells, marker_velocities)
    ):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        key = (i, j, k)
        target_sum[key] = target_sum.get(key, np.zeros(3, dtype=np.float64)) + np.asarray(
            marker_velocity,
            dtype=np.float64,
        )
        target_count[key] = target_count.get(key, 0) + 1
        marker_region_id = int(marker_region_ids[marker_index])
        target_regions.setdefault(key, set()).add(marker_region_id)
        if marker_region_id == PRIMARY_REGION_ID:
            target_primary_sum[key] = target_primary_sum.get(
                key,
                np.zeros(3, dtype=np.float64),
            ) + np.asarray(marker_velocity, dtype=np.float64)
            target_primary_count[key] = target_primary_count.get(key, 0) + 1
        elif marker_region_id == SECONDARY_REGION_ID:
            target_secondary_sum[key] = target_secondary_sum.get(
                key,
                np.zeros(3, dtype=np.float64),
            ) + np.asarray(marker_velocity, dtype=np.float64)
            target_secondary_count[key] = target_secondary_count.get(key, 0) + 1
        before_residuals.append(float(np.linalg.norm(velocity[i, j, k] - marker_velocity)))

    for (i, j, k), summed_velocity in target_sum.items():
        active[i, j, k] = 1
        values[i, j, k] = summed_velocity / float(target_count[(i, j, k)])
        weights[i, j, k] = 1.0
        if row_regions is not None:
            regions = target_regions[(i, j, k)]
            row_regions[i, j, k] = next(iter(regions)) if len(regions) == 1 else -1

    constraint_active_cell_count = 0
    if preserve_velocity_constraints:
        constraint_active_cell_count = _write_marker_velocity_constraints(
            fluid,
            target_sum=target_sum,
            target_count=target_count,
            target_primary_sum=target_primary_sum,
            target_primary_count=target_primary_count,
            target_secondary_sum=target_secondary_sum,
            target_secondary_count=target_secondary_count,
        )

    after_residuals: list[float] = []
    for cell, marker_velocity in zip(marker_cells, marker_velocities):
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        after_residuals.append(float(np.linalg.norm(values[i, j, k] - marker_velocity)))

    fluid.velocity_dirichlet_boundary_active.from_numpy(active)
    fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
    fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
    if row_region_field is not None and row_regions is not None:
        row_region_field.from_numpy(row_regions)

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
        "fluid_marker_velocity_constraints_enabled": preserve_velocity_constraints,
        "fluid_marker_velocity_constraint_active_cell_count": constraint_active_cell_count,
        "no_slip_residual_before_mps": max(before_residuals, default=0.0),
        "no_slip_residual_after_mps": max(after_residuals, default=0.0),
        "no_slip_target_residual_after_assembly_mps": max(
            after_residuals,
            default=0.0,
        ),
        "no_slip_projected_residual_after_projection_mps": 0.0,
        "_feedback_constraint_cells": set(target_sum),
    }


def _clear_fluid_velocity_constraints(fluid: CartesianFluidSolver) -> None:
    fluid.clear_velocity_constraints()


def _marker_region_ids(
    markers: HibmMpmSurfaceMarkers,
    marker_count: int,
) -> np.ndarray:
    region_field = getattr(markers, "region_id", None)
    if region_field is None:
        return np.full(marker_count, -1, dtype=np.int32)
    return np.asarray(region_field.to_numpy()[:marker_count], dtype=np.int32)


def _write_marker_velocity_constraints(
    fluid: CartesianFluidSolver,
    *,
    target_sum: Mapping[tuple[int, int, int], np.ndarray],
    target_count: Mapping[tuple[int, int, int], int],
    target_primary_sum: Mapping[tuple[int, int, int], np.ndarray],
    target_primary_count: Mapping[tuple[int, int, int], int],
    target_secondary_sum: Mapping[tuple[int, int, int], np.ndarray],
    target_secondary_count: Mapping[tuple[int, int, int], int],
) -> int:
    required_fields = (
        "velocity_constraint_sum",
        "velocity_constraint_weight",
        "velocity_constraint_primary_sum",
        "velocity_constraint_primary_weight",
        "velocity_constraint_secondary_sum",
        "velocity_constraint_secondary_weight",
    )
    if any(getattr(fluid, name, None) is None for name in required_fields):
        return 0

    constraint_sum = fluid.velocity_constraint_sum.to_numpy()
    constraint_weight = fluid.velocity_constraint_weight.to_numpy()
    primary_sum = fluid.velocity_constraint_primary_sum.to_numpy()
    primary_weight = fluid.velocity_constraint_primary_weight.to_numpy()
    secondary_sum = fluid.velocity_constraint_secondary_sum.to_numpy()
    secondary_weight = fluid.velocity_constraint_secondary_weight.to_numpy()

    for (i, j, k), summed_velocity in target_sum.items():
        count = float(target_count[(i, j, k)])
        constraint_sum[i, j, k] = np.asarray(summed_velocity, dtype=np.float64)
        constraint_weight[i, j, k] = count
        if (i, j, k) in target_primary_sum:
            primary_sum[i, j, k] = np.asarray(
                target_primary_sum[(i, j, k)],
                dtype=np.float64,
            )
            primary_weight[i, j, k] = float(target_primary_count[(i, j, k)])
        if (i, j, k) in target_secondary_sum:
            secondary_sum[i, j, k] = np.asarray(
                target_secondary_sum[(i, j, k)],
                dtype=np.float64,
            )
            secondary_weight[i, j, k] = float(target_secondary_count[(i, j, k)])

    fluid.velocity_constraint_sum.from_numpy(constraint_sum)
    fluid.velocity_constraint_weight.from_numpy(constraint_weight)
    fluid.velocity_constraint_primary_sum.from_numpy(primary_sum)
    fluid.velocity_constraint_primary_weight.from_numpy(primary_weight)
    fluid.velocity_constraint_secondary_sum.from_numpy(secondary_sum)
    fluid.velocity_constraint_secondary_weight.from_numpy(secondary_weight)
    return len(target_sum)


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
        "fluid_marker_velocity_constraints_enabled": False,
        "fluid_marker_velocity_constraint_active_cell_count": 0,
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

    measure_device = getattr(fluid, "marker_feedback_projected_residual_mps", None)
    if measure_device is not None:
        return float(
            measure_device(
                markers.x_gamma_m,
                markers.v_gamma_mps,
                marker_count,
            )
        )

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
    *,
    include_percentiles: bool = False,
) -> dict[str, object]:
    device_report = getattr(fluid, "flow_state_report", None)
    if device_report is not None:
        state = dict(
            device_report(
                pressure_field=_fluid_feedback_pressure_field(fluid),
                include_percentiles=bool(include_percentiles),
            )
        )
        return {
            "mode": FLOW_SOLUTION_MODE,
            "projection_report": projection_report,
            "obstacle_cell_count": int(state["obstacle_cell_count"]),
            "fluid_cell_count": int(state["fluid_cell_count"]),
            "local_velocity_peak_mps": float(state["local_velocity_peak_mps"]),
            "fluid_speed_p99_mps": state["fluid_speed_p99_mps"],
            "fluid_speed_p999_mps": state["fluid_speed_p999_mps"],
            "pressure_min_pa": float(state["pressure_min_pa"]),
            "pressure_max_pa": float(state["pressure_max_pa"]),
            "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
            **_flow_source_report_fields(projection_report),
        }

    obstacle = fluid.obstacle.to_numpy()
    velocity = fluid.velocity.to_numpy()
    pressure = _fluid_feedback_pressure_numpy(fluid)
    non_obstacle = obstacle == 0
    speed = np.linalg.norm(velocity, axis=3)
    active_speed = speed[non_obstacle]
    if bool(include_percentiles) and active_speed.size:
        speed_p99 = float(np.percentile(active_speed, 99.0))
        speed_p999 = float(np.percentile(active_speed, 99.9))
    else:
        speed_p99 = "" if active_speed.size else 0.0
        speed_p999 = "" if active_speed.size else 0.0
    active_pressure = pressure[non_obstacle]
    if active_pressure.size:
        pressure_min_pa = float(active_pressure.min())
        pressure_max_pa = float(active_pressure.max())
    else:
        pressure_min_pa = 0.0
        pressure_max_pa = 0.0
    return {
        "mode": FLOW_SOLUTION_MODE,
        "projection_report": projection_report,
        "obstacle_cell_count": int(obstacle.sum()),
        "fluid_cell_count": int(non_obstacle.sum()),
        "local_velocity_peak_mps": float(active_speed.max(initial=0.0)),
        "fluid_speed_p99_mps": speed_p99,
        "fluid_speed_p999_mps": speed_p999,
        "pressure_min_pa": pressure_min_pa,
        "pressure_max_pa": pressure_max_pa,
        "pressure_sign_convention": "fluid.fsi_pressure feedback field is sampled for reports and traction",
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


def _flow_projection_report_fields(report: Any) -> dict[str, object]:
    if not isinstance(report, Mapping):
        return {f"flow_projection_{key}": "" for key in FLOW_PROJECTION_REPORT_KEYS}
    projection_report = report.get("projection_report", report)
    if not isinstance(projection_report, Mapping):
        projection_report = {}
    fields = {
        f"flow_projection_{key}": projection_report.get(key, "")
        for key in FLOW_PROJECTION_REPORT_KEYS
    }
    if fields["flow_projection_fsi_pressure_snapshot_updated"] == "":
        fields["flow_projection_fsi_pressure_snapshot_updated"] = report.get(
            "fsi_pressure_snapshot_updated",
            "",
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


def _marker_traction_report_fields(
    markers: HibmMpmSurfaceMarkers,
    *,
    include_face_diagnostics: bool = True,
) -> dict[str, object]:
    return markers.stress_face_diagnostics(
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
        streamwise_axis_index=STREAMWISE_AXIS_INDEX,
        include_face_diagnostics=include_face_diagnostics,
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


def _install_selected_pressure_pair_anchor_markers(
    markers: HibmMpmSurfaceMarkers,
    config: Any,
) -> dict[str, object]:
    anchor_markers_json = _traction_pressure_pair_anchor_markers_json(config)
    if anchor_markers_json is None:
        if (
            _is_selected_traction_formulation_coupled_smoke(config)
            and _traction_pressure_pair_runtime_provider_mode(config)
            == TRACTION_PRESSURE_PAIR_RUNTIME_PROVIDER_ANCHORED_CELL_PAIR
        ):
            runtime_pair_map = _runtime_pressure_pair_anchor_map(markers, config)
            markers.set_pressure_pair_anchor_cells(
                inside_cells=runtime_pair_map.inside_cells,
                outside_cells=runtime_pair_map.outside_cells,
            )
            return _pressure_pair_anchor_install_report(
                status="installed",
                source="runtime_generated",
                marker_count=int(markers.marker_count),
                active_marker_count=runtime_pair_map.selected_count,
                anchor_map_sha256=runtime_pair_map.pair_map_sha256,
                source_marker_geometry_sha256=(
                    runtime_pair_map.marker_geometry_sha256
                ),
                pair_map_diagnostics=runtime_pair_map.as_diagnostics(),
                fixed_solid_snapshot_policy="runtime_marker_geometry",
            )
        if _is_selected_traction_formulation_coupled_smoke(config):
            raise ValueError(
                "selected coupled smoke requires "
                "traction_pressure_pair_anchor_markers_json"
            )
        return _pressure_pair_anchor_install_report(
            status="not_requested",
            source="unset",
            marker_count=int(markers.marker_count),
        )
    if not _is_selected_traction_formulation_coupled_smoke(config):
        raise ValueError(
            "traction_pressure_pair_anchor_markers_json is selected coupled smoke only"
        )

    (
        marker_payload,
        resolved_markers_json,
        wrapper_payloads,
        wrapper_paths,
    ) = _load_pressure_pair_anchor_marker_payload(Path(anchor_markers_json))
    _assert_pressure_pair_anchor_marker_geometry_matches(markers, marker_payload)
    inside_cells, outside_cells = _pressure_pair_anchor_cells_from_marker_payload(
        marker_payload,
    )
    markers.set_pressure_pair_anchor_cells(
        inside_cells=inside_cells,
        outside_cells=outside_cells,
    )

    metadata_sources = list(wrapper_payloads) + [marker_payload]
    return _pressure_pair_anchor_install_report(
        status="installed",
        source="marker_diagnostics_json",
        marker_count=int(markers.marker_count),
        active_marker_count=len(inside_cells),
        source_json=anchor_markers_json,
        resolved_json=resolved_markers_json.as_posix(),
        wrapper_jsons=[path.as_posix() for path in wrapper_paths],
        wrapper_depth=len(wrapper_paths),
        anchor_map_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_map_sha256",
        ),
        source_flow_snapshot_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_source_flow_snapshot_sha256",
            "flow_snapshot_sha256",
            "new_or_confirmed_flow_snapshot_sha256",
        ),
        source_marker_geometry_sha256=_first_metadata_value(
            metadata_sources,
            "anchor_source_marker_geometry_sha256",
            "marker_geometry_sha256",
        ),
        fixed_solid_snapshot_policy=_first_metadata_value(
            metadata_sources,
            "fixed_solid_snapshot_policy",
        ),
    )


def _runtime_pressure_pair_anchor_map(
    markers: HibmMpmSurfaceMarkers,
    config: Any,
) -> PressureSamplePairMap:
    solid_min, solid_max = _solid_box(config)
    inside_axis_position_m = 0.5 * (
        float(solid_min[STREAMWISE_AXIS_INDEX])
        + float(solid_max[STREAMWISE_AXIS_INDEX])
    )
    provider = RuntimeAnchoredCellPairProvider(
        domain_bounds_m=_domain_bounds(config),
        grid_nodes=tuple(int(value) for value in config.grid_nodes),
        anchor_axis=STREAMWISE_AXIS_INDEX,
        inside_axis_position_m=inside_axis_position_m,
        outside_axis_offset_cells=1,
    )
    return provider.compute_pairs(markers)


def _load_pressure_pair_anchor_marker_payload(
    path: Path,
) -> tuple[dict[str, Any], Path, tuple[dict[str, Any], ...], tuple[Path, ...]]:
    current = path
    wrappers: list[dict[str, Any]] = []
    wrapper_paths: list[Path] = []
    seen: set[str] = set()
    for _depth in range(8):
        current_key = current.resolve().as_posix()
        if current_key in seen:
            raise ValueError("pressure pair anchor marker diagnostics source cycle")
        seen.add(current_key)
        payload = json.loads(current.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("pressure pair anchor marker diagnostics must be an object")
        if isinstance(payload.get("markers"), list):
            return payload, current, tuple(wrappers), tuple(wrapper_paths)
        source = payload.get("source_marker_diagnostics_json")
        if not source:
            raise ValueError(
                "pressure pair anchor marker diagnostics must contain markers "
                "or source_marker_diagnostics_json"
            )
        wrappers.append(payload)
        wrapper_paths.append(current)
        current = Path(str(source))
    raise ValueError("pressure pair anchor marker diagnostics source chain too deep")


def _pressure_pair_anchor_cells_from_marker_payload(
    payload: Mapping[str, Any],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    marker_payloads = _pressure_pair_anchor_marker_entries(payload)
    inside_cells: list[tuple[int, int, int]] = []
    outside_cells: list[tuple[int, int, int]] = []
    for index, marker in enumerate(marker_payloads):
        if not bool(marker.get("pressure_pair_anchor_active", False)):
            raise ValueError(
                "pressure pair anchor marker payload contains inactive marker "
                f"{index}"
            )
        inside_cells.append(
            _pressure_pair_anchor_cell(
                marker.get("pressure_pair_anchor_inside_cell"),
                marker_index=index,
                field_name="pressure_pair_anchor_inside_cell",
            )
        )
        outside_cells.append(
            _pressure_pair_anchor_cell(
                marker.get("pressure_pair_anchor_outside_cell"),
                marker_index=index,
                field_name="pressure_pair_anchor_outside_cell",
            )
        )
    return inside_cells, outside_cells


def _pressure_pair_anchor_marker_entries(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    markers_payload = payload.get("markers")
    if not isinstance(markers_payload, list) or not markers_payload:
        raise ValueError("pressure pair anchor marker payload must contain markers")
    entries: list[Mapping[str, Any]] = []
    for index, marker in enumerate(markers_payload):
        if not isinstance(marker, Mapping):
            raise ValueError(f"pressure pair anchor marker {index} must be an object")
        entries.append(marker)
    declared_count = payload.get("marker_count")
    if declared_count is not None and int(declared_count) != len(entries):
        raise ValueError("pressure pair anchor marker_count does not match markers")
    return entries


def _pressure_pair_anchor_cell(
    value: object,
    *,
    marker_index: int,
    field_name: str,
) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} for marker {marker_index} must have 3 cells")
    cell = tuple(int(component) for component in value)
    if any(component < 0 for component in cell):
        raise ValueError(f"{field_name} for marker {marker_index} must be in-bounds")
    return cell


def _assert_pressure_pair_anchor_marker_geometry_matches(
    markers: HibmMpmSurfaceMarkers,
    payload: Mapping[str, Any],
) -> None:
    marker_payloads = _pressure_pair_anchor_marker_entries(payload)
    marker_count = int(markers.marker_count)
    if len(marker_payloads) != marker_count:
        raise ValueError("pressure pair anchor marker count must match live markers")
    positions = markers.x_gamma_m.to_numpy()[:marker_count]
    normals = markers.n_gamma.to_numpy()[:marker_count]
    regions = markers.region_id.to_numpy()[:marker_count]
    for index, marker in enumerate(marker_payloads):
        marker_index = int(marker.get("marker_index", index))
        if marker_index != index:
            raise ValueError("pressure pair anchor marker indices must be ordered")
        if int(marker.get("region_id", -1)) != int(regions[index]):
            raise ValueError(
                "pressure pair anchor marker region mismatch at marker "
                f"{index}"
            )
        expected_position = _pressure_pair_anchor_vector3(
            marker.get("position_m"),
            marker_index=index,
            field_name="position_m",
        )
        expected_normal = _pressure_pair_anchor_vector3(
            marker.get("normal"),
            marker_index=index,
            field_name="normal",
        )
        if not np.allclose(
            positions[index],
            np.asarray(expected_position, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(
                "pressure pair anchor marker position mismatch at marker "
                f"{index}"
            )
        if not np.allclose(
            normals[index],
            np.asarray(expected_normal, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(
                "pressure pair anchor marker normal mismatch at marker "
                f"{index}"
            )


def _pressure_pair_anchor_vector3(
    value: object,
    *,
    marker_index: int,
    field_name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} for marker {marker_index} must be length 3")
    return tuple(float(component) for component in value)


def _first_metadata_value(
    payloads: list[Mapping[str, Any]],
    *keys: str,
) -> str:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value) != "":
                return str(value)
    return ""


def _pressure_pair_anchor_install_report(
    *,
    status: str,
    source: str,
    marker_count: int,
    active_marker_count: int = 0,
    source_json: str = "",
    resolved_json: str = "",
    wrapper_jsons: list[str] | None = None,
    wrapper_depth: int = 0,
    anchor_map_sha256: str = "",
    source_flow_snapshot_sha256: str = "",
    source_marker_geometry_sha256: str = "",
    pair_map_diagnostics: Mapping[str, Any] | None = None,
    fixed_solid_snapshot_policy: str = "",
) -> dict[str, object]:
    return {
        "pressure_pair_anchor_install_status": status,
        "pressure_pair_anchor_source": source,
        "pressure_pair_anchor_markers_json": source_json,
        "pressure_pair_anchor_resolved_markers_json": resolved_json,
        "pressure_pair_anchor_wrapper_jsons": list(wrapper_jsons or []),
        "pressure_pair_anchor_wrapper_depth": int(wrapper_depth),
        "pressure_pair_anchor_active_marker_count": int(active_marker_count),
        "pressure_pair_anchor_expected_marker_count": int(marker_count),
        "pressure_pair_anchor_map_sha256": anchor_map_sha256,
        "pressure_pair_anchor_source_flow_snapshot_sha256": (
            source_flow_snapshot_sha256
        ),
        "pressure_pair_anchor_source_marker_geometry_sha256": (
            source_marker_geometry_sha256
        ),
        "pressure_pair_anchor_current_marker_geometry_sha256": (
            source_marker_geometry_sha256 if active_marker_count == marker_count else ""
        ),
        "pressure_pair_anchor_pair_map": dict(pair_map_diagnostics or {}),
        "pressure_pair_anchor_fixed_solid_snapshot_policy": (
            fixed_solid_snapshot_policy
        ),
    }


def _build_solid(
    config: Any,
    runtime: TaichiRuntimeConfig,
) -> NeoHookeanMpmState:
    bounds_min, bounds_max = _solid_mpm_bounds(config)
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
    one_sided_policy = (
        TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        if config is None
        else _traction_one_sided_pressure_policy(config)
    )
    one_sided_region_id = (
        PRIMARY_REGION_ID
        if pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
        and one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_DISABLED
        else -1
    )
    per_face_one_sided = (
        pressure_sampling_mode == TRACTION_PRESSURE_ONE_SIDED
        and one_sided_policy == TRACTION_ONE_SIDED_PRESSURE_POLICY_PER_FACE_MIRRORED
    )
    primary_side_sign = (
        0.0
        if config is None
        else _traction_one_sided_primary_fluid_side_normal_sign(config) or 0.0
    )
    secondary_side_sign = (
        0.0
        if config is None
        else _traction_one_sided_secondary_fluid_side_normal_sign(config) or 0.0
    )
    report = markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        _fluid_feedback_pressure_field(fluid),
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
        one_sided_pressure_primary_region_id=(
            PRIMARY_REGION_ID if per_face_one_sided else -1
        ),
        one_sided_pressure_secondary_region_id=(
            SECONDARY_REGION_ID if per_face_one_sided else -1
        ),
        one_sided_primary_reference_pressure_pa=(
            0.0
            if config is None
            else _traction_one_sided_primary_reference_pressure_pa(config)
        ),
        one_sided_secondary_reference_pressure_pa=(
            0.0
            if config is None
            else _traction_one_sided_secondary_reference_pressure_pa(config)
        ),
        one_sided_primary_fluid_side_normal_sign=primary_side_sign,
        one_sided_secondary_fluid_side_normal_sign=secondary_side_sign,
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
    rest: np.ndarray | None = None,
) -> dict[str, object]:
    positions = solid.x.to_numpy()[: solid.particle_count]
    if rest is None:
        # rest positions are constant; per-step callers pass a cached copy so
        # the whole rest array is not re-fetched from the device every step
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
