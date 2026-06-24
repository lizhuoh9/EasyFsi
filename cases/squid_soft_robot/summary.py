from __future__ import annotations

import math
from collections.abc import Mapping


def _context_value(context: Mapping[str, object], name: str):
    try:
        return context[name]
    except KeyError as exc:
        raise KeyError(f"build_final_run_report missing context value: {name}") from exc


def validation_scope_report(
    *,
    requested_steps: int,
    completed_steps: int,
    full_pressure_waveform_steps: int,
    partial_run_stopped: bool,
    partial_run_reason: str = "",
) -> dict[str, object]:
    requested = int(requested_steps)
    completed = int(completed_steps)
    full_steps = int(full_pressure_waveform_steps)
    if requested <= 0 or completed < 0 or full_steps <= 0:
        raise ValueError("step counts must be positive, with completed_steps non-negative")
    if partial_run_stopped or completed < requested:
        reason = str(partial_run_reason or "completed_steps_less_than_requested")
        scope = "wall_time_partial" if reason == "max_wall_time_s" else "partial_requested_steps"
        return {
            "validation_scope": scope,
            "validation_scope_complete": False,
            "validation_scope_reason": reason,
        }
    if requested < full_steps:
        return {
            "validation_scope": "explicit_step_count",
            "validation_scope_complete": False,
            "validation_scope_reason": "explicit_steps_before_full_pressure_waveform",
        }
    return {
        "validation_scope": "full_pressure_waveform",
        "validation_scope_complete": True,
        "validation_scope_reason": None,
    }


def runtime_budget_report(
    *,
    timing_summary: dict[str, object],
    requested_steps: int,
    completed_steps: int,
    full_pressure_waveform_steps: int,
) -> dict[str, object]:
    requested = max(1, int(requested_steps))
    completed = max(0, int(completed_steps))
    full_steps = max(1, int(full_pressure_waveform_steps))
    try:
        mean_step_wall_time_s = float(timing_summary.get("mean_step_wall_time_s", 0.0))
    except (TypeError, ValueError):
        mean_step_wall_time_s = 0.0
    if not math.isfinite(mean_step_wall_time_s) or mean_step_wall_time_s < 0.0:
        mean_step_wall_time_s = 0.0
    try:
        steady_state_mean_step_wall_time_s = float(
            timing_summary.get("steady_state_mean_step_wall_time_s", 0.0)
        )
    except (TypeError, ValueError):
        steady_state_mean_step_wall_time_s = 0.0
    if (
        not math.isfinite(steady_state_mean_step_wall_time_s)
        or steady_state_mean_step_wall_time_s < 0.0
    ):
        steady_state_mean_step_wall_time_s = 0.0
    steady_state_sample_count = max(
        0,
        int(timing_summary.get("steady_state_step_wall_time_sample_count", 0) or 0),
    )
    steady_state_warmup_excluded_steps = max(
        0,
        int(timing_summary.get("steady_state_warmup_excluded_steps", 0) or 0),
    )
    steady_state_available = (
        steady_state_sample_count > 0 and steady_state_mean_step_wall_time_s > 0.0
    )
    requested_remaining = max(0, requested - completed)
    full_remaining = max(0, full_steps - completed)
    report = {
        "basis": "measured_mean_step_wall_time_s",
        "requested_steps": requested,
        "completed_steps": completed,
        "full_pressure_waveform_steps": full_steps,
        "measured_mean_step_wall_time_s": mean_step_wall_time_s,
        "estimated_requested_run_wall_time_s": mean_step_wall_time_s * requested,
        "estimated_requested_remaining_wall_time_s": (
            mean_step_wall_time_s * requested_remaining
        ),
        "estimated_full_pressure_waveform_wall_time_s": mean_step_wall_time_s * full_steps,
        "estimated_full_pressure_waveform_remaining_wall_time_s": (
            mean_step_wall_time_s * full_remaining
        ),
        "note": (
            "Runtime budget only: extrapolated from measured completed-step wall time. "
            "It does not change pressure, velocity, flow, IBM force, FSI coupling, or "
            "validation gates."
        ),
    }
    if steady_state_available:
        report.update(
            {
                "steady_state_estimate_available": True,
                "steady_state_basis": "steady_state_mean_step_wall_time_s",
                "steady_state_mean_step_wall_time_s": steady_state_mean_step_wall_time_s,
                "steady_state_step_wall_time_sample_count": steady_state_sample_count,
                "steady_state_warmup_excluded_steps": steady_state_warmup_excluded_steps,
                "steady_state_estimated_requested_run_wall_time_s": (
                    steady_state_mean_step_wall_time_s * requested
                ),
                "steady_state_estimated_requested_remaining_wall_time_s": (
                    steady_state_mean_step_wall_time_s * requested_remaining
                ),
                "steady_state_estimated_full_pressure_waveform_wall_time_s": (
                    steady_state_mean_step_wall_time_s * full_steps
                ),
                "steady_state_estimated_full_pressure_waveform_remaining_wall_time_s": (
                    steady_state_mean_step_wall_time_s * full_remaining
                ),
                "steady_state_note": (
                    "warmup-excluded runtime budget: ignores the first measured step "
                    "when later completed-step timings are available. It is a reporting "
                    "estimate only and does not change pressure, velocity, flow, IBM "
                    "force, FSI coupling, or validation gates."
                ),
            }
        )
    else:
        report.update(
            {
                "steady_state_estimate_available": False,
                "steady_state_mean_step_wall_time_s": 0.0,
                "steady_state_step_wall_time_sample_count": 0,
                "steady_state_warmup_excluded_steps": 0,
                "steady_state_note": (
                    "warmup-excluded runtime budget is unavailable until at least one "
                    "post-warmup step timing is available."
                ),
            }
        )
    return report


def build_final_run_report(context: Mapping[str, object]) -> dict[str, object]:
    FSI_STABILIZATION_PRESET_CONFLICT_POLICY = _context_value(context, "FSI_STABILIZATION_PRESET_CONFLICT_POLICY")
    _final_row_int = _context_value(context, "_final_row_int")
    _final_row_number = _context_value(context, "_final_row_number")
    _final_row_number_or_none = _context_value(context, "_final_row_number_or_none")
    _required_finite_row_vector = _context_value(context, "_required_finite_row_vector")
    _row_bool = _context_value(context, "_row_bool")
    _rows_max_int = _context_value(context, "_rows_max_int")
    adaptive_fluid_substeps_enabled = _context_value(context, "adaptive_fluid_substeps_enabled")
    args = _context_value(context, "args")
    asdict = _context_value(context, "asdict")
    boundary_drive_compliance_report = _context_value(context, "boundary_drive_compliance_report")
    cad_provenance = _context_value(context, "cad_provenance")
    cg_preconditioner = _context_value(context, "cg_preconditioner")
    cg_tolerance = _context_value(context, "cg_tolerance")
    checks_passed = _context_value(context, "checks_passed")
    count_enabled_unconverged_fsi_rows = _context_value(context, "count_enabled_unconverged_fsi_rows")
    effective_fluid_substep_dt_s = _context_value(context, "effective_fluid_substep_dt_s")
    effective_fluid_substeps = _context_value(context, "effective_fluid_substeps")
    effective_multigrid_cycles = _context_value(context, "effective_multigrid_cycles")
    estimated_solid_particle_spacing_m = _context_value(context, "estimated_solid_particle_spacing_m")
    finite_field_diagnostics = _context_value(context, "finite_field_diagnostics")
    finite_required_row_fields_for_solid_model = _context_value(context, "finite_required_row_fields_for_solid_model")
    first_step = _context_value(context, "first_step")
    fluid_grid_axis_max_spacing_m = _context_value(context, "fluid_grid_axis_max_spacing_m")
    fluid_grid_axis_min_spacing_m = _context_value(context, "fluid_grid_axis_min_spacing_m")
    fluid_grid_resolution = _context_value(context, "fluid_grid_resolution")
    fluid_grid_uniform_spacing_m = _context_value(context, "fluid_grid_uniform_spacing_m")
    fsi_constraint_force_solid_mobility_ratio = _context_value(context, "fsi_constraint_force_solid_mobility_ratio")
    fsi_coupling_adaptive_iterations_cfl_threshold = _context_value(context, "fsi_coupling_adaptive_iterations_cfl_threshold")
    fsi_coupling_adaptive_iterations_max = _context_value(context, "fsi_coupling_adaptive_iterations_max")
    fsi_coupling_adaptive_iterations_residual_threshold_n = _context_value(context, "fsi_coupling_adaptive_iterations_residual_threshold_n")
    fsi_coupling_iterations = _context_value(context, "fsi_coupling_iterations")
    fsi_coupling_max_accepted_residual_n = _context_value(context, "fsi_coupling_max_accepted_residual_n")
    fsi_coupling_mode = _context_value(context, "fsi_coupling_mode")
    fsi_coupling_mode_report = _context_value(context, "fsi_coupling_mode_report")
    fsi_coupling_rejected_trial_backtrack = _context_value(context, "fsi_coupling_rejected_trial_backtrack")
    fsi_coupling_residual_continuation_iterations_max = _context_value(context, "fsi_coupling_residual_continuation_iterations_max")
    fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max = _context_value(context, "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max")
    fsi_coupling_residual_continuation_rebound_secant_factor = _context_value(context, "fsi_coupling_residual_continuation_rebound_secant_factor")
    fsi_coupling_residual_continuation_rebound_secant_from_best = _context_value(context, "fsi_coupling_residual_continuation_rebound_secant_from_best")
    fsi_coupling_residual_continuation_threshold_n = _context_value(context, "fsi_coupling_residual_continuation_threshold_n")
    fsi_coupling_residual_growth_rejection_factor = _context_value(context, "fsi_coupling_residual_growth_rejection_factor")
    fsi_coupling_same_step_rerun_fluid_substep_factor = _context_value(context, "fsi_coupling_same_step_rerun_fluid_substep_factor")
    fsi_coupling_same_step_rerun_iterations_max = _context_value(context, "fsi_coupling_same_step_rerun_iterations_max")
    fsi_coupling_same_step_rerun_residual_threshold_n = _context_value(context, "fsi_coupling_same_step_rerun_residual_threshold_n")
    fsi_coupling_solver = _context_value(context, "fsi_coupling_solver")
    fsi_coupling_target_map_relaxation = _context_value(context, "fsi_coupling_target_map_relaxation")
    fsi_coupling_tolerance_n = _context_value(context, "fsi_coupling_tolerance_n")
    fsi_coupling_trial_interior_divergence_tolerance = _context_value(context, "fsi_coupling_trial_interior_divergence_tolerance")
    fsi_coupling_trust_region_adaptive = _context_value(context, "fsi_coupling_trust_region_adaptive")
    fsi_coupling_trust_region_force_increment_n = _context_value(context, "fsi_coupling_trust_region_force_increment_n")
    fsi_coupling_trust_region_growth_factor = _context_value(context, "fsi_coupling_trust_region_growth_factor")
    fsi_coupling_trust_region_rebound_backtrack = _context_value(context, "fsi_coupling_trust_region_rebound_backtrack")
    fsi_coupling_trust_region_rebound_factor = _context_value(context, "fsi_coupling_trust_region_rebound_factor")
    fsi_coupling_trust_region_rebound_stop_factor = _context_value(context, "fsi_coupling_trust_region_rebound_stop_factor")
    fsi_coupling_trust_region_rebound_stop_max_residual_n = _context_value(context, "fsi_coupling_trust_region_rebound_stop_max_residual_n")
    fsi_coupling_trust_region_shrink_factor = _context_value(context, "fsi_coupling_trust_region_shrink_factor")
    fsi_marker_coupling_tolerance_mps = _context_value(context, "fsi_marker_coupling_tolerance_mps")
    fsi_physical_interface_map_stability_report = _context_value(context, "fsi_physical_interface_map_stability_report")
    fsi_solid_response_dt_s = _context_value(context, "fsi_solid_response_dt_s")
    fsi_solid_response_mobility_coupling = _context_value(context, "fsi_solid_response_mobility_coupling")
    fsi_solid_response_velocity_mobility_coupling = _context_value(context, "fsi_solid_response_velocity_mobility_coupling")
    fsi_stabilization_effective_parameters = _context_value(context, "fsi_stabilization_effective_parameters")
    fsi_stabilization_preset = _context_value(context, "fsi_stabilization_preset")
    fsi_velocity_constraint_blend = _context_value(context, "fsi_velocity_constraint_blend")
    fsi_velocity_constraint_solid_mobility_ratio = _context_value(context, "fsi_velocity_constraint_solid_mobility_ratio")
    fsi_velocity_target_solid_mobility_ratio = _context_value(context, "fsi_velocity_target_solid_mobility_ratio")
    full_pressure_waveform_steps = _context_value(context, "full_pressure_waveform_steps")
    history_path = _context_value(context, "history_path")
    initial_fluid_obstacle_mode = _context_value(context, "initial_fluid_obstacle_mode")
    interface_reaction_aitken = _context_value(context, "interface_reaction_aitken")
    interface_reaction_aitken_lower_bound = _context_value(context, "interface_reaction_aitken_lower_bound")
    interface_reaction_aitken_upper_bound = _context_value(context, "interface_reaction_aitken_upper_bound")
    interface_reaction_passivity_limit = _context_value(context, "interface_reaction_passivity_limit")
    interface_reaction_relaxation = _context_value(context, "interface_reaction_relaxation")
    interface_reaction_robin_impedance_ns_m = _context_value(context, "interface_reaction_robin_impedance_ns_m")
    interface_reaction_robin_matrix_impedance_ns_m = _context_value(context, "interface_reaction_robin_matrix_impedance_ns_m")
    interface_reaction_robin_target_mode = _context_value(context, "interface_reaction_robin_target_mode")
    json = _context_value(context, "json")
    legacy_projected_reduced_fsi_coupling_enabled = _context_value(context, "legacy_projected_reduced_fsi_coupling_enabled")
    material = _context_value(context, "material")
    math = _context_value(context, "math")
    max_wall_time_s = _context_value(context, "max_wall_time_s")
    membrane_thickness_scale = _context_value(context, "membrane_thickness_scale")
    multigrid_cycles = _context_value(context, "multigrid_cycles")
    os = _context_value(context, "os")
    outlet_to_fsi_volume_source_gate_scope = _context_value(context, "outlet_to_fsi_volume_source_gate_scope")
    output_dir = _context_value(context, "output_dir")
    partial_run_reason = _context_value(context, "partial_run_reason")
    partial_run_stopped = _context_value(context, "partial_run_stopped")
    physical_outlet_to_fsi_volume_source_passes = _context_value(context, "physical_outlet_to_fsi_volume_source_passes")
    pressure_boundary_mapping = _context_value(context, "pressure_boundary_mapping")
    pressure_closure_normal = _context_value(context, "pressure_closure_normal")
    pressure_far_side_normal_sign = _context_value(context, "pressure_far_side_normal_sign")
    pressure_flux_trend_report = _context_value(context, "pressure_flux_trend_report")
    pressure_load_direction = _context_value(context, "pressure_load_direction")
    pressure_load_region_id = _context_value(context, "pressure_load_region_id")
    pressure_load_source_region_id = _context_value(context, "pressure_load_source_region_id")
    pressure_outlet_boundary_report = _context_value(context, "pressure_outlet_boundary_report")
    pressure_outlet_source_ratio_passes = _context_value(context, "pressure_outlet_source_ratio_passes")
    pressure_outlet_source_ratio_tolerance = _context_value(context, "pressure_outlet_source_ratio_tolerance")
    pressure_outlet_zmin_enabled = _context_value(context, "pressure_outlet_zmin_enabled")
    pressure_projection_budget = _context_value(context, "pressure_projection_budget")
    pressure_schedule_applied_in_history = _context_value(context, "pressure_schedule_applied_in_history")
    pressure_schedule_input = _context_value(context, "pressure_schedule_input")
    pressure_solver_name = _context_value(context, "pressure_solver_name")
    primary_shell_region_id = _context_value(context, "primary_shell_region_id")
    process_path = _context_value(context, "process_path")
    projection_divergence_cleanup_iterations = _context_value(context, "projection_divergence_cleanup_iterations")
    real_cad_step_binding = _context_value(context, "real_cad_step_binding")
    reduced_active_water_connectivity = _context_value(context, "reduced_active_water_connectivity")
    reduced_water_geometry_report = _context_value(context, "reduced_water_geometry_report")
    refinement_region_summary = _context_value(context, "refinement_region_summary")
    region14_aperture_carve_enabled = _context_value(context, "region14_aperture_carve_enabled")
    region14_aperture_carve_source = _context_value(context, "region14_aperture_carve_source")
    region14_aperture_geometry = _context_value(context, "region14_aperture_geometry")
    reuse_accepted_fsi_trial_state = _context_value(context, "reuse_accepted_fsi_trial_state")
    rows = _context_value(context, "rows")
    run_checkpoint_path = _context_value(context, "run_checkpoint_path")
    run_process_completion_status = _context_value(context, "run_process_completion_status")
    runtime_budget_report = _context_value(context, "runtime_budget_report")
    secondary_shell_region_id = _context_value(context, "secondary_shell_region_id")
    signed_positive_source_flux_ratio = _context_value(context, "signed_positive_source_flux_ratio")
    simulator = _context_value(context, "simulator")
    solid_density_scale = _context_value(context, "solid_density_scale")
    solid_mpm_bounds_max_m = _context_value(context, "solid_mpm_bounds_max_m")
    solid_mpm_bounds_min_m = _context_value(context, "solid_mpm_bounds_min_m")
    solid_mpm_bounds_padding_m = _context_value(context, "solid_mpm_bounds_padding_m")
    solid_mpm_flip_blend = _context_value(context, "solid_mpm_flip_blend")
    solid_mpm_force_nonzero_when_pressure_loaded = _context_value(context, "solid_mpm_force_nonzero_when_pressure_loaded")
    solid_mpm_substeps = _context_value(context, "solid_mpm_substeps")
    solid_sub_dt_s = _context_value(context, "solid_sub_dt_s")
    solid_substep_velocity_damping = _context_value(context, "solid_substep_velocity_damping")
    solid_surface_mass_report = _context_value(context, "solid_surface_mass_report")
    source_config_fluid_active_mask_requested = _context_value(context, "source_config_fluid_active_mask_requested")
    source_config_fluid_topology_report = _context_value(context, "source_config_fluid_topology_report")
    source_config_path = _context_value(context, "source_config_path")
    source_config_reduced_water_intersection_requested = _context_value(context, "source_config_reduced_water_intersection_requested")
    source_config_region14_aperture_requested = _context_value(context, "source_config_region14_aperture_requested")
    spec = _context_value(context, "spec")
    stable_solid_dt_s = _context_value(context, "stable_solid_dt_s")
    step_count = _context_value(context, "step_count")
    tail_refinement_geometry = _context_value(context, "tail_refinement_geometry")
    tail_refinement_region = _context_value(context, "tail_refinement_region")
    time = _context_value(context, "time")
    total_fsi_face_area_m2 = _context_value(context, "total_fsi_face_area_m2")
    tri_metadata = _context_value(context, "tri_metadata")
    validation_scope_report = _context_value(context, "validation_scope_report")
    vector_norm = _context_value(context, "vector_norm")

    last = rows[-1] if rows else {}
    max_abs_main = max(abs(float(row["main_displacement_z_m"])) for row in rows) if rows else 0.0
    max_abs_tail = max(abs(float(row["tail_displacement_z_m"])) for row in rows) if rows else 0.0
    max_cfl = max(float(row["cfl"]) for row in rows) if rows else 0.0
    max_div_l2 = max(float(row["divergence_l2"]) for row in rows) if rows else 0.0
    max_interior_div_l2 = (
        max(float(row["interior_divergence_l2"]) for row in rows) if rows else 0.0
    )
    max_outlet_negative_z = max(float(row["outlet_flow_negative_z_m3s"]) for row in rows) if rows else 0.0
    max_pressure_traction = max(float(row["pressure_traction_abs_force_n"]) for row in rows) if rows else 0.0
    max_viscous_traction = (
        max(
            vector_norm(
                (
                    float(row["viscous_traction_force_x_n"]),
                    float(row["viscous_traction_force_y_n"]),
                    float(row["viscous_traction_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_fluid_stress_traction = (
        max(
            vector_norm(
                (
                    float(row["fluid_stress_traction_force_x_n"]),
                    float(row["fluid_stress_traction_force_y_n"]),
                    float(row["fluid_stress_traction_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_projected_ibm = max(float(row["projected_ibm_residual_mps"]) for row in rows) if rows else 0.0
    max_pressure_projection_cg_iterations_total = (
        max(int(row["pressure_projection_cg_iterations_total"]) for row in rows)
        if rows
        else 0
    )
    max_pressure_projection_cg_project_calls = (
        max(int(row.get("pressure_projection_cg_project_calls", 0) or 0) for row in rows)
        if rows
        else 0
    )
    mean_pressure_projection_cg_project_calls = (
        sum(int(row.get("pressure_projection_cg_project_calls", 0) or 0) for row in rows)
        / len(rows)
        if rows
        else 0.0
    )
    mean_pressure_projection_cg_iterations_total = (
        sum(int(row["pressure_projection_cg_iterations_total"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    max_pressure_projection_cg_iterations_max = (
        max(int(row["pressure_projection_cg_iterations_max"]) for row in rows)
        if rows
        else 0
    )
    max_pressure_projection_cg_relative_residual = (
        max(float(row["pressure_projection_cg_max_relative_residual"]) for row in rows)
        if rows
        else 0.0
    )
    max_pressure_projection_cg_initial_relative_residual = (
        max(float(row["pressure_projection_cg_max_initial_relative_residual"]) for row in rows)
        if rows
        else 0.0
    )
    pressure_projection_cg_breakdown_count = (
        sum(int(row["pressure_projection_cg_breakdown_count"]) for row in rows)
        if rows
        else 0
    )
    pressure_projection_cg_converged_all = (
        all(_row_bool(row["pressure_projection_cg_converged_all"]) for row in rows)
        if rows
        else True
    )
    pressure_projection_physical_failure_count = (
        sum(
            1
            for row in rows
            if _row_bool(row.get("pressure_projection_physical_failure", False))
        )
        if rows
        else 0
    )
    max_hibm_unreached_incompatible_component_count = (
        max(
            int(row.get("hibm_unreached_incompatible_component_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_hibm_unreached_component_raw_count = _rows_max_int(
        rows,
        "hibm_unreached_component_raw_count",
    )
    max_hibm_unreached_largest_component_cell_count = _rows_max_int(
        rows,
        "hibm_unreached_largest_component_cell_count",
    )
    max_hibm_unreached_singleton_component_count = _rows_max_int(
        rows,
        "hibm_unreached_singleton_component_count",
    )
    max_hibm_unreached_small_component_count = _rows_max_int(
        rows,
        "hibm_unreached_small_component_count",
    )
    max_hibm_unreached_small_component_cell_count = _rows_max_int(
        rows,
        "hibm_unreached_small_component_cell_count",
    )
    max_hibm_projection_overflow_singleton_cleanup_cell_count = _rows_max_int(
        rows,
        "hibm_projection_overflow_singleton_cleanup_cell_count",
    )
    max_hibm_projection_overflow_singleton_cleanup_component_count = _rows_max_int(
        rows,
        "hibm_projection_overflow_singleton_cleanup_component_count",
    )
    max_hibm_projection_tiny_unreached_cleanup_cell_count = _rows_max_int(
        rows,
        "hibm_projection_tiny_unreached_cleanup_cell_count",
    )
    max_hibm_projection_tiny_unreached_cleanup_component_count = _rows_max_int(
        rows,
        "hibm_projection_tiny_unreached_cleanup_component_count",
    )
    max_hibm_unreached_component_rhs_mean_max_abs = (
        max(
            float(row.get("hibm_unreached_component_rhs_mean_max_abs", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_hibm_unreached_component_rhs_integral_max_abs = (
        max(
            float(
                row.get("hibm_unreached_component_rhs_integral_max_abs", 0.0)
                or 0.0
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_pressure_interface_matrix_diagonal_integral = (
        max(abs(float(row.get("pressure_interface_matrix_diagonal_integral", 0.0) or 0.0)) for row in rows)
        if rows
        else 0.0
    )
    max_pressure_interface_matrix_rhs_integral = (
        max(abs(float(row.get("pressure_interface_matrix_rhs_integral", 0.0) or 0.0)) for row in rows)
        if rows
        else 0.0
    )
    max_pressure_interface_matrix_max_abs_diagonal = (
        max(float(row.get("pressure_interface_matrix_max_abs_diagonal", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_hibm_pressure_neumann_raw_transmissibility_m = (
        max(float(row.get("hibm_pressure_neumann_max_raw_transmissibility_m", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_hibm_pressure_neumann_transmissibility_limit_m = (
        max(float(row.get("hibm_pressure_neumann_max_transmissibility_limit_m", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_hibm_pressure_neumann_transmissibility_capped_row_count = (
        max(int(row.get("hibm_pressure_neumann_transmissibility_capped_row_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_pressure_interface_matrix_active_cells = (
        max(int(row.get("pressure_interface_matrix_active_cells", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_trial_pressure_projection_cg_project_calls = (
        max(int(row.get("fsi_trial_pressure_projection_cg_project_calls", 0) or 0) for row in rows)
        if rows
        else 0
    )
    mean_fsi_trial_pressure_projection_cg_project_calls = (
        sum(
            int(row.get("fsi_trial_pressure_projection_cg_project_calls", 0) or 0)
            for row in rows
        )
        / len(rows)
        if rows
        else 0.0
    )
    max_fsi_trial_pressure_projection_cg_iterations_total = (
        max(
            int(row.get("fsi_trial_pressure_projection_cg_iterations_total", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    mean_fsi_trial_pressure_projection_cg_iterations_total = (
        sum(
            int(row.get("fsi_trial_pressure_projection_cg_iterations_total", 0) or 0)
            for row in rows
        )
        / len(rows)
        if rows
        else 0.0
    )
    max_fsi_trial_pressure_projection_cg_iterations_max = (
        max(
            int(row.get("fsi_trial_pressure_projection_cg_iterations_max", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_trial_pressure_projection_cg_relative_residual = (
        max(
            float(row.get("fsi_trial_pressure_projection_cg_max_relative_residual", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_fsi_trial_pressure_projection_cg_initial_relative_residual = (
        max(
            float(
                row.get(
                    "fsi_trial_pressure_projection_cg_max_initial_relative_residual",
                    0.0,
                )
                or 0.0
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    fsi_trial_pressure_projection_cg_breakdown_count = (
        sum(int(row.get("fsi_trial_pressure_projection_cg_breakdown_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    fsi_trial_pressure_projection_cg_converged_all = (
        all(_row_bool(row.get("fsi_trial_pressure_projection_cg_converged_all", True)) for row in rows)
        if rows
        else True
    )
    max_total_pressure_projection_cg_project_calls = (
        max(int(row.get("total_pressure_projection_cg_project_calls", 0) or 0) for row in rows)
        if rows
        else 0
    )
    mean_total_pressure_projection_cg_project_calls = (
        sum(int(row.get("total_pressure_projection_cg_project_calls", 0) or 0) for row in rows)
        / len(rows)
        if rows
        else 0.0
    )
    max_total_pressure_projection_cg_iterations_total = (
        max(int(row.get("total_pressure_projection_cg_iterations_total", 0) or 0) for row in rows)
        if rows
        else 0
    )
    mean_total_pressure_projection_cg_iterations_total = (
        sum(int(row.get("total_pressure_projection_cg_iterations_total", 0) or 0) for row in rows)
        / len(rows)
        if rows
        else 0.0
    )
    max_total_pressure_projection_cg_iterations_max = (
        max(int(row.get("total_pressure_projection_cg_iterations_max", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_total_pressure_projection_cg_relative_residual = (
        max(
            float(row.get("total_pressure_projection_cg_max_relative_residual", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_total_pressure_projection_cg_initial_relative_residual = (
        max(
            float(
                row.get("total_pressure_projection_cg_max_initial_relative_residual", 0.0)
                or 0.0
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    total_pressure_projection_cg_breakdown_count = (
        sum(int(row.get("total_pressure_projection_cg_breakdown_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    total_pressure_projection_cg_converged_all = (
        all(_row_bool(row.get("total_pressure_projection_cg_converged_all", True)) for row in rows)
        if rows
        else True
    )
    accepted_fsi_trial_state_reuse_count = (
        sum(1 for row in rows if _row_bool(row.get("accepted_fsi_trial_state_reused", False)))
        if rows
        else 0
    )
    accepted_fsi_trial_state_readvance_count = (
        sum(
            1
            for row in rows
            if _row_bool(row.get("accepted_fsi_trial_state_readvanced", False))
        )
        if rows
        else 0
    )
    fsi_all_trials_rejected_count = (
        sum(1 for row in rows if _row_bool(row.get("fsi_all_trials_rejected", False)))
        if rows
        else 0
    )
    fsi_zero_force_commit_blocked_count = (
        sum(
            1
            for row in rows
            if _row_bool(row.get("fsi_zero_force_commit_blocked", False))
        )
        if rows
        else 0
    )
    max_abs_pressure_load_pa = (
        max(abs(float(row["pressure_load_pa"])) for row in rows) if rows else 0.0
    )
    max_abs_volume_flux_m3s = max(abs(float(row["volume_flux_m3s"])) for row in rows) if rows else 0.0
    max_abs_fsi_volume_source_m3s = (
        max(abs(float(row["fsi_volume_source_m3s"])) for row in rows) if rows else 0.0
    )
    timing_fields = (
        "step_wall_time_s",
        "fsi_coupling_wall_time_s",
        "solid_advance_wall_time_s",
        "fluid_advance_wall_time_s",
        "sample_wall_time_s",
        "surface_diagnostics_wall_time_s",
        "checkpoint_wall_time_s",
    )
    timing_summary = {}
    for field in timing_fields:
        values = [
            float(row.get(field, 0.0) or 0.0)
            for row in rows
            if math.isfinite(float(row.get(field, 0.0) or 0.0))
        ]
        timing_summary[f"max_{field}"] = max(values) if values else 0.0
        timing_summary[f"mean_{field}"] = (
            sum(values) / float(len(values)) if values else 0.0
        )
        if field == "step_wall_time_s" and len(values) >= 3:
            steady_state_values = values[1:]
            timing_summary["steady_state_warmup_excluded_steps"] = 1
            timing_summary["steady_state_step_wall_time_sample_count"] = len(
                steady_state_values
            )
            timing_summary["steady_state_mean_step_wall_time_s"] = sum(
                steady_state_values
            ) / float(len(steady_state_values))
    final_outlet_flux_ratio = _final_row_number(last, "main_volume_flux_to_outlet_ratio")
    final_downstream_flux_ratio = _final_row_number(last, "main_volume_flux_to_downstream_ratio")
    final_fsi_volume_source_m3s = _final_row_number(last, "fsi_volume_source_m3s")
    final_pressure_outlet_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_source_volume_flux_m3s",
    )
    final_pressure_outlet_positive_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_positive_source_volume_flux_m3s",
    )
    final_pressure_outlet_abs_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_abs_source_volume_flux_m3s",
    )
    final_pressure_outlet_reachable_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_reachable_source_volume_flux_m3s",
    )
    final_pressure_outlet_unreached_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_unreached_source_volume_flux_m3s",
    )
    final_pressure_outlet_reachability_valid = (
        _row_bool(last.get("pressure_outlet_reachability_valid", False))
        if last is not None
        else False
    )
    final_pressure_outlet_reachability_revision = _final_row_number(
        last,
        "pressure_outlet_reachability_revision",
    )
    final_pressure_outlet_reachable_source_cell_count = _final_row_number(
        last,
        "pressure_outlet_reachable_source_cell_count",
    )
    final_pressure_outlet_unreached_source_cell_count = _final_row_number(
        last,
        "pressure_outlet_unreached_source_cell_count",
    )
    final_pressure_outlet_unreached_source_abs_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_unreached_source_abs_flux_m3s",
    )
    final_pressure_outlet_unreached_source_centroid_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_x_m",
    )
    final_pressure_outlet_unreached_source_centroid_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_y_m",
    )
    final_pressure_outlet_unreached_source_centroid_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_z_m",
    )
    final_pressure_outlet_unreached_source_min_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_x_m",
    )
    final_pressure_outlet_unreached_source_min_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_y_m",
    )
    final_pressure_outlet_unreached_source_min_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_z_m",
    )
    final_pressure_outlet_unreached_source_max_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_x_m",
    )
    final_pressure_outlet_unreached_source_max_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_y_m",
    )
    final_pressure_outlet_unreached_source_max_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_z_m",
    )
    final_pressure_outlet_velocity_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_velocity_flux_m3s",
    )
    final_pressure_outlet_velocity_to_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_source_ratio",
    )
    final_pressure_outlet_velocity_to_net_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_net_source_ratio",
    )
    final_pressure_outlet_velocity_to_positive_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_positive_source_ratio",
    )
    final_pressure_outlet_velocity_to_abs_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_abs_source_ratio",
    )
    final_pressure_outlet_pressure_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_pressure_flux_m3s",
    )
    final_pressure_outlet_pressure_to_source_ratio = _final_row_number(
        last,
        "pressure_outlet_pressure_to_source_ratio",
    )
    final_pressure_outlet_pressure_to_net_source_ratio = _final_row_number(
        last,
        "pressure_outlet_pressure_to_net_source_ratio",
    )
    final_pressure_outlet_pressure_to_positive_source_ratio = _final_row_number(
        last,
        "pressure_outlet_pressure_to_positive_source_ratio",
    )
    final_pressure_outlet_pressure_to_abs_source_ratio = _final_row_number(
        last,
        "pressure_outlet_pressure_to_abs_source_ratio",
    )
    final_pressure_outlet_projection_pre_velocity_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_projection_pre_velocity_flux_m3s",
    )
    final_pressure_outlet_projection_post_pressure_velocity_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_projection_post_pressure_velocity_flux_m3s",
    )
    final_pressure_outlet_projection_post_boundary_velocity_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_projection_post_boundary_velocity_flux_m3s",
    )
    final_outlet_to_fsi_volume_source_ratio = signed_positive_source_flux_ratio(
        outlet_negative_z_flux_m3s=_final_row_number(last, "outlet_flow_negative_z_m3s"),
        source_flux_m3s=final_fsi_volume_source_m3s,
    )
    max_outlet_flux_ratio = (
        max(float(row["main_volume_flux_to_outlet_ratio"]) for row in rows) if rows else 0.0
    )
    min_outlet_flux_ratio = (
        min(float(row["main_volume_flux_to_outlet_ratio"]) for row in rows) if rows else 0.0
    )
    max_velocity_constraint_cells = (
        max(int(row["fsi_velocity_constraint_active_cells"]) for row in rows) if rows else 0
    )
    max_velocity_constraint_delta_mps = (
        max(float(row["fsi_velocity_constraint_max_delta_mps"]) for row in rows) if rows else 0.0
    )
    max_velocity_constraint_momentum_delta_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_momentum_delta_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_momentum_delta_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_momentum_delta_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_velocity_constraint_primary_momentum_delta_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_primary_momentum_delta_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_primary_momentum_delta_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_primary_momentum_delta_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_secondary_momentum_delta_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_secondary_momentum_delta_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_secondary_momentum_delta_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_secondary_momentum_delta_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_primary_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_primary_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_velocity_constraint_secondary_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_secondary_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_velocity_constraint_step_impulse_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_step_impulse_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_step_impulse_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_step_impulse_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_primary_step_impulse_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_primary_step_impulse_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_primary_step_impulse_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_primary_step_impulse_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_secondary_step_impulse_n_s = (
        max(
            vector_norm(
                (
                    float(row.get("fsi_velocity_constraint_secondary_step_impulse_x_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_secondary_step_impulse_y_n_s", 0.0) or 0.0),
                    float(row.get("fsi_velocity_constraint_secondary_step_impulse_z_n_s", 0.0) or 0.0),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_velocity_constraint_step_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_step_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_velocity_constraint_primary_step_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_primary_step_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_velocity_constraint_secondary_step_equivalent_force_norm_n = (
        max(float(row.get("fsi_velocity_constraint_secondary_step_equivalent_force_norm_n", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_positive_main_interface_reaction_power_w = (
        max(float(row["relaxed_main_interface_reaction_power_w_next"]) for row in rows)
        if rows
        else 0.0
    )
    max_interface_reaction_relaxation_effective = (
        max(float(row["interface_reaction_relaxation_effective"]) for row in rows) if rows else 0.0
    )
    min_interface_reaction_relaxation_effective = (
        min(float(row["interface_reaction_relaxation_effective"]) for row in rows) if rows else 0.0
    )
    max_fsi_coupling_iterations_used = (
        max(int(row["fsi_coupling_iterations_used"]) for row in rows) if rows else 0
    )
    max_fsi_coupling_iterations_requested = (
        max(int(row.get("fsi_coupling_iterations_requested", 0) or 0) for row in rows)
        if rows
        else 0
    )
    total_fsi_coupling_adaptive_iterations_triggered = (
        sum(
            1
            for row in rows
            if bool(row.get("fsi_coupling_adaptive_iterations_triggered", False))
        )
        if rows
        else 0
    )
    total_fsi_coupling_adaptive_iterations_residual_triggered = (
        sum(
            1
            for row in rows
            if bool(
                row.get(
                    "fsi_coupling_adaptive_iterations_residual_triggered",
                    False,
                )
            )
        )
        if rows
        else 0
    )
    total_fsi_coupling_adaptive_iterations_cfl_triggered = (
        sum(
            1
            for row in rows
            if bool(row.get("fsi_coupling_adaptive_iterations_cfl_triggered", False))
        )
        if rows
        else 0
    )
    total_fsi_coupling_same_step_rerun_triggered = (
        sum(
            1
            for row in rows
            if bool(row.get("fsi_coupling_same_step_rerun_triggered", False))
        )
        if rows
        else 0
    )
    total_fsi_coupling_same_step_rerun_count = (
        sum(
            int(row.get("fsi_coupling_same_step_rerun_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    same_step_rerun_initial_residuals = [
        float(row.get("fsi_coupling_same_step_rerun_initial_residual_norm_n", 0.0))
        for row in rows
        if math.isfinite(
            float(
                row.get(
                    "fsi_coupling_same_step_rerun_initial_residual_norm_n",
                    math.nan,
                )
            )
        )
    ]
    max_fsi_coupling_same_step_rerun_initial_residual_norm_n = (
        max(same_step_rerun_initial_residuals)
        if same_step_rerun_initial_residuals
        else 0.0
    )
    max_fsi_coupling_residual_continuation_iteration_count = (
        max(
            int(row.get("fsi_coupling_residual_continuation_iteration_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_residual_continuation_iteration_count = (
        sum(
            int(row.get("fsi_coupling_residual_continuation_iteration_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_residual_continuation_rebound_secant_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_residual_continuation_rebound_secant_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_residual_continuation_rebound_secant_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_residual_continuation_rebound_secant_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fluid_substeps = (
        max(int(float(row.get("fluid_substeps", effective_fluid_substeps))) for row in rows)
        if rows
        else effective_fluid_substeps
    )
    max_fsi_coupling_rejected_trial_count = (
        max(int(row.get("fsi_coupling_rejected_trial_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_rejected_trial_backtrack_count = (
        max(
            int(row.get("fsi_coupling_rejected_trial_backtrack_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_residual_growth_rejected_trial_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_residual_growth_rejected_trial_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_max_residual_rejected_trial_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_max_residual_rejected_trial_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trial_cfl_rejected_count = (
        max(
            int(row.get("fsi_coupling_trial_cfl_rejected_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trial_interior_divergence_rejected_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_trial_interior_divergence_rejected_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_limited_update_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_trust_region_limited_update_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_shrink_count = (
        max(
            int(row.get("fsi_coupling_trust_region_shrink_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_growth_count = (
        max(
            int(row.get("fsi_coupling_trust_region_growth_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_rebound_backtrack_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_backtrack_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_rebound_stop_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_stop_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_trust_region_rebound_stop_suppressed_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_stop_suppressed_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_rejected_trial_count = (
        sum(int(row.get("fsi_coupling_rejected_trial_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    total_fsi_coupling_rejected_trial_backtrack_count = (
        sum(
            int(row.get("fsi_coupling_rejected_trial_backtrack_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_max_residual_rejected_trial_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_max_residual_rejected_trial_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trial_cfl_rejected_count = (
        sum(
            int(row.get("fsi_coupling_trial_cfl_rejected_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trial_interior_divergence_rejected_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_trial_interior_divergence_rejected_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_limited_update_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_trust_region_limited_update_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_shrink_count = (
        sum(
            int(row.get("fsi_coupling_trust_region_shrink_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_growth_count = (
        sum(
            int(row.get("fsi_coupling_trust_region_growth_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_rebound_backtrack_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_backtrack_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_rebound_stop_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_stop_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_trust_region_rebound_stop_suppressed_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_trust_region_rebound_stop_suppressed_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    total_fsi_coupling_residual_growth_rejected_trial_count = (
        sum(
            int(
                row.get(
                    "fsi_coupling_residual_growth_rejected_trial_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    fsi_coupling_accepted_trial_cfl_values = [
        float(row.get("fsi_coupling_accepted_trial_cfl", math.nan))
        for row in rows
        if math.isfinite(float(row.get("fsi_coupling_accepted_trial_cfl", math.nan)))
    ]
    max_fsi_coupling_accepted_trial_cfl = (
        max(fsi_coupling_accepted_trial_cfl_values)
        if fsi_coupling_accepted_trial_cfl_values
        else math.nan
    )
    fsi_coupling_trial_cfl_values = [
        float(row.get("fsi_coupling_trial_cfl_max", math.nan))
        for row in rows
        if math.isfinite(float(row.get("fsi_coupling_trial_cfl_max", math.nan)))
    ]
    max_fsi_coupling_trial_cfl = (
        max(fsi_coupling_trial_cfl_values)
        if fsi_coupling_trial_cfl_values
        else math.nan
    )
    fsi_coupling_accepted_trial_interior_divergence_l2_values = [
        float(row.get("fsi_coupling_accepted_trial_interior_divergence_l2", math.nan))
        for row in rows
        if math.isfinite(
            float(
                row.get(
                    "fsi_coupling_accepted_trial_interior_divergence_l2",
                    math.nan,
                )
            )
        )
    ]
    max_fsi_coupling_accepted_trial_interior_divergence_l2 = (
        max(fsi_coupling_accepted_trial_interior_divergence_l2_values)
        if fsi_coupling_accepted_trial_interior_divergence_l2_values
        else math.nan
    )
    fsi_coupling_trial_interior_divergence_l2_values = [
        float(row.get("fsi_coupling_trial_interior_divergence_l2_max", math.nan))
        for row in rows
        if math.isfinite(
            float(row.get("fsi_coupling_trial_interior_divergence_l2_max", math.nan))
        )
    ]
    max_fsi_coupling_trial_interior_divergence_l2 = (
        max(fsi_coupling_trial_interior_divergence_l2_values)
        if fsi_coupling_trial_interior_divergence_l2_values
        else math.nan
    )
    fsi_coupling_residual_norm_n_values = [
        float(row["fsi_coupling_residual_norm_n"])
        for row in rows
        if math.isfinite(float(row["fsi_coupling_residual_norm_n"]))
    ]
    max_fsi_coupling_residual_norm_n = (
        max(fsi_coupling_residual_norm_n_values)
        if fsi_coupling_residual_norm_n_values
        else math.nan
    )
    max_fsi_coupling_residual_norm_mps = (
        max(float(row.get("fsi_coupling_residual_norm_mps", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_residual_max_mps = (
        max(float(row.get("fsi_coupling_residual_max_mps", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    fsi_coupling_not_converged_count = (
        count_enabled_unconverged_fsi_rows(rows)
        if rows
        else 0
    )
    max_fsi_coupling_iqn_ils_least_squares_update_count = (
        max(int(row.get("fsi_coupling_iqn_ils_least_squares_update_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_interface_map_amplification = (
        max(float(row.get("fsi_coupling_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_physical_interface_map_amplification = (
        max(float(row.get("fsi_coupling_physical_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_physical_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_physical_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_raw_interface_map_amplification = (
        max(float(row.get("fsi_coupling_raw_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_raw_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_raw_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_interface_map_amplification_sample_count = (
        max(int(row.get("fsi_coupling_interface_map_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_residual_jacobian_amplification_sample_count = (
        max(int(row.get("fsi_coupling_residual_jacobian_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_physical_interface_map_amplification_sample_count = (
        max(int(row.get("fsi_coupling_physical_interface_map_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_physical_residual_jacobian_amplification_sample_count = (
        max(int(row.get("fsi_coupling_physical_residual_jacobian_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_raw_interface_map_amplification_sample_count = (
        max(int(row.get("fsi_coupling_raw_interface_map_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_raw_residual_jacobian_amplification_sample_count = (
        max(int(row.get("fsi_coupling_raw_residual_jacobian_amplification_sample_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_primary_response_constraint_force_solid_mobility_ratio = (
        max(float(row.get("fsi_primary_response_constraint_force_solid_mobility_ratio", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_secondary_response_constraint_force_solid_mobility_ratio = (
        max(float(row.get("fsi_secondary_response_constraint_force_solid_mobility_ratio", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_primary_velocity_target_solid_mobility_ratio = (
        max(float(row.get("fsi_primary_velocity_target_solid_mobility_ratio", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_secondary_velocity_target_solid_mobility_ratio = (
        max(float(row.get("fsi_secondary_velocity_target_solid_mobility_ratio", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    final_main_interface_reaction_power_w = _final_row_number(
        last,
        "relaxed_main_interface_reaction_power_w_next",
    )
    main_interface_reaction_passivity_limit_count = (
        sum(1 for row in rows if bool(row.get("main_interface_reaction_passivity_limited", False)))
        if rows
        else 0
    )
    tail_interface_reaction_passivity_limit_count = (
        sum(1 for row in rows if bool(row.get("tail_interface_reaction_passivity_limited", False)))
        if rows
        else 0
    )
    max_abs_main_interface_reaction_robin_impedance_force_n = (
        max(
            abs(float(row.get("main_interface_reaction_robin_impedance_force_z_n", 0.0) or 0.0))
            for row in rows
        )
        if rows
        else 0.0
    )
    max_abs_tail_interface_reaction_robin_impedance_force_n = (
        max(
            abs(float(row.get("tail_interface_reaction_robin_impedance_force_z_n", 0.0) or 0.0))
            for row in rows
        )
        if rows
        else 0.0
    )
    max_active_force_cells = max(int(row["fsi_active_force_cells"]) for row in rows) if rows else 0
    max_fsi_grid_decomposition_relative_error = (
        max(float(row["fsi_last_correction_grid_decomposition_relative_error"]) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_grid_decomposition_residual_n = (
        max(float(row["fsi_last_correction_grid_decomposition_residual_abs_n"]) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_action_reaction_residual_n = (
        max(float(row["fsi_action_reaction_residual_abs_n"]) for row in rows) if rows else 0.0
    )
    max_fsi_action_reaction_relative_error = (
        max(float(row["fsi_action_reaction_relative_error"]) for row in rows) if rows else 0.0
    )
    max_fluid_reaction_action_reaction_relative_error = (
        max(float(row["fsi_fluid_reaction_action_reaction_relative_error"]) for row in rows)
        if rows
        else 0.0
    )
    max_fluid_reaction_action_reaction_residual_z_n = (
        max(abs(float(row["fsi_fluid_reaction_action_reaction_residual_z_n"])) for row in rows)
        if rows
        else 0.0
    )
    max_fluid_reaction_full_relative_error = (
        max(
            max(
                float(row["main_fsi_fluid_reaction_full_relative_error"]),
                float(row["tail_fsi_fluid_reaction_full_relative_error"]),
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_fluid_reaction_full_residual_n = (
        max(
            max(
                float(row["main_fsi_fluid_reaction_full_residual_n"]),
                float(row["tail_fsi_fluid_reaction_full_residual_n"]),
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_solid_mpm_transfer_error = (
        max(float(row["solid_mpm_transfer_relative_error"]) for row in rows) if rows else 0.0
    )
    max_solid_mpm_total_force_n = (
        max(
            vector_norm(
                (
                    float(row["solid_mpm_total_force_x_n"]),
                    float(row["solid_mpm_total_force_y_n"]),
                    float(row["solid_mpm_total_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_solid_mpm_active_grid_nodes = (
        max(int(row["solid_mpm_active_grid_nodes"]) for row in rows) if rows else 0
    )
    final_outlet_negative_z = _final_row_number(last, "outlet_flow_negative_z_m3s")
    final_all_sections_negative_z = bool(
        last
        and _final_row_number(last, "lip_flow_negative_z_m3s") > 0.0
        and _final_row_number(last, "outlet_flow_negative_z_m3s") > 0.0
        and _final_row_number(last, "downstream_flow_negative_z_m3s") > 0.0
    )
    min_probe_valid_fraction = (
        min(float(row["fsi_probe_valid_fraction"]) for row in rows) if rows else 0.0
    )
    min_force_probe_valid_fraction = (
        min(float(row["fsi_force_probe_valid_fraction"]) for row in rows) if rows else 0.0
    )
    obstacle_cell_count = 0
    if not args.disable_reduced_obstacles:
        obstacle_cell_count = simulator.fluid.obstacle_cell_count()
    obstacle_mask = None
    fluid_obstacle = getattr(simulator.fluid, "obstacle", None)
    if fluid_obstacle is not None and hasattr(fluid_obstacle, "to_numpy"):
        obstacle_mask = fluid_obstacle.to_numpy()
    active_water = reduced_active_water_connectivity(
        spec,
        obstacle_cell_count=obstacle_cell_count,
        obstacle_mask=obstacle_mask,
    )
    fluid_grid_spacing_m = (
        None
        if fluid_grid_uniform_spacing_m is None
        else [float(value) for value in fluid_grid_uniform_spacing_m]
    )
    fluid_grid_min_spacing_m = [float(value) for value in fluid_grid_axis_min_spacing_m]
    fluid_grid_max_spacing_m = [float(value) for value in fluid_grid_axis_max_spacing_m]
    fluid_grid_nodes = [
        int(simulator.fluid.nx),
        int(simulator.fluid.ny),
        int(simulator.fluid.nz),
    ]
    solid_mpm_grid_spacing_m = [
        _final_row_number(last, "solid_mpm_grid_dx_m"),
        _final_row_number(last, "solid_mpm_grid_dy_m"),
        _final_row_number(last, "solid_mpm_grid_dz_m"),
    ]
    required_row_fields = finite_required_row_fields_for_solid_model(args.solid_model)
    nonfinite_diagnostics = finite_field_diagnostics(rows, required_row_fields)
    solid_model_should_report_force = args.solid_model in (
        "tri_mooney_shell_mpm",
        "neo_hookean_mpm",
    )
    solid_mpm_force_components_n: tuple[float, float, float] | None = None
    if last and solid_model_should_report_force:
        solid_mpm_force_components_n = _required_finite_row_vector(
            last,
            (
                "solid_mpm_total_force_x_n",
                "solid_mpm_total_force_y_n",
                "solid_mpm_total_force_z_n",
            ),
            context=f"{args.solid_model} final row",
        )
    solid_mpm_force_required = solid_model_should_report_force and bool(rows)
    if solid_mpm_force_required and solid_mpm_force_components_n is None:
        raise RuntimeError(
            f"{args.solid_model} final row did not produce a solid force vector"
        )
    constraint_force_mobility_scale_delta = (
        fsi_constraint_force_solid_mobility_ratio
        / (1.0 + fsi_constraint_force_solid_mobility_ratio)
    )
    response_constraint_force_mobility_scale_delta = max(
        (
            max_fsi_primary_response_constraint_force_solid_mobility_ratio
            / (1.0 + max_fsi_primary_response_constraint_force_solid_mobility_ratio)
        ),
        (
            max_fsi_secondary_response_constraint_force_solid_mobility_ratio
            / (1.0 + max_fsi_secondary_response_constraint_force_solid_mobility_ratio)
        ),
    )
    velocity_target_mobility_scale_delta = max(
        (
            max_fsi_primary_velocity_target_solid_mobility_ratio
            / (1.0 + max_fsi_primary_velocity_target_solid_mobility_ratio)
        ),
        (
            max_fsi_secondary_velocity_target_solid_mobility_ratio
            / (1.0 + max_fsi_secondary_velocity_target_solid_mobility_ratio)
        ),
    )
    fsi_coupling_raw_interface_map_strict_physical = (
        abs(float(args.constraint_force_scale) - 1.0) <= 0.0
        and fsi_velocity_constraint_blend <= 0.0
        and fsi_constraint_force_solid_mobility_ratio <= 0.0
        and not fsi_solid_response_mobility_coupling
        and fsi_velocity_target_solid_mobility_ratio <= 0.0
        and not fsi_solid_response_velocity_mobility_coupling
        and interface_reaction_robin_impedance_ns_m <= 0.0
        and interface_reaction_robin_matrix_impedance_ns_m <= 0.0
        and not interface_reaction_passivity_limit
    )
    boundary_drive_compliance = boundary_drive_compliance_report(
        prescribed_velocity_boundary=fsi_velocity_constraint_blend > 0.0,
        prescribed_pressure_or_flow_boundary=max_abs_pressure_load_pa > 0.0,
        nonzero_fluid_traction_scale=max(
            abs(float(args.constraint_force_scale) - 1.0),
            constraint_force_mobility_scale_delta,
            response_constraint_force_mobility_scale_delta,
            velocity_target_mobility_scale_delta,
        ),
    )
    validation_scope = validation_scope_report(
        requested_steps=step_count,
        completed_steps=len(rows),
        full_pressure_waveform_steps=full_pressure_waveform_steps,
        partial_run_stopped=partial_run_stopped,
        partial_run_reason=partial_run_reason,
    )
    pressure_flux_trend = pressure_flux_trend_report(
        rows,
        requested_steps=step_count,
        min_trend_steps=200,
    )
    runtime_budget = runtime_budget_report(
        timing_summary=timing_summary,
        requested_steps=step_count,
        completed_steps=len(rows),
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    outlet_to_fsi_gate_scope = outlet_to_fsi_volume_source_gate_scope(
        fluid_grid_resolution=fluid_grid_resolution,
        validation_scope_complete=bool(validation_scope["validation_scope_complete"]),
    )
    final_outlet_to_fsi_volume_source_ratio_physical = physical_outlet_to_fsi_volume_source_passes(
        outlet_negative_z_flux_m3s=final_outlet_negative_z,
        fsi_volume_source_m3s=final_fsi_volume_source_m3s,
        min_ratio=float(args.min_outlet_to_main_volume_flux_ratio),
    )
    fsi_physical_interface_map_stability = fsi_physical_interface_map_stability_report(
        fsi_coupling_enabled=legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=fsi_coupling_iterations,
        ),
        fsi_coupling_iterations=fsi_coupling_iterations,
        max_physical_interface_map_amplification=(
            max_fsi_coupling_raw_interface_map_amplification
        ),
        measurement_sample_count=(
            max_fsi_coupling_raw_interface_map_amplification_sample_count
        ),
        raw_interface_map_strict_physical=(
            fsi_coupling_raw_interface_map_strict_physical
        ),
    )
    checks = {
        "pressure_schedule_applied": pressure_schedule_applied_in_history(rows),
        "surface_fsi_force_spread_active": max_active_force_cells > 0,
        "fsi_velocity_constraint_active": (
            max_velocity_constraint_cells > 0
            if fsi_velocity_constraint_blend > 0.0
            else True
        ),
        "projected_ibm_samples_present": bool(rows and int(last["projected_ibm_sample_count"]) > 0),
        "solid_mpm_particles_present": bool(rows and int(last["solid_mpm_particle_count"]) > 0),
        "solid_mpm_grid_transfer_active": max_solid_mpm_active_grid_nodes > 0,
        "fsi_coupling_converged": fsi_coupling_not_converged_count == 0,
        "solid_mpm_force_nonzero_when_pressure_loaded": solid_mpm_force_nonzero_when_pressure_loaded(
            rows,
            force_required=solid_mpm_force_required,
        ),
        "fsi_physical_interface_map_stable": bool(
            fsi_physical_interface_map_stability["passes"]
        ),
        "finite_primary_diagnostics": len(nonfinite_diagnostics) == 0,
        "fsi_probe_valid_fraction_positive": min_probe_valid_fraction > 0.0,
        "fsi_force_probe_valid_fraction_all_valid": min_force_probe_valid_fraction >= 1.0,
        "active_water_connectivity_passed": bool(active_water["connectivity_passed"]),
        "negative_z_outlet_flow_present": max_outlet_negative_z > 0.0,
        "final_negative_z_outlet_flow": final_outlet_negative_z > 0.0,
        "pressure_outlet_velocity_to_source_ratio_near_one": pressure_outlet_source_ratio_passes(
            source_volume_flux_m3s=(
                final_pressure_outlet_positive_source_volume_flux_m3s
            ),
            velocity_outlet_flux_m3s=final_pressure_outlet_velocity_flux_m3s,
            pressure_outlet_flux_m3s=final_pressure_outlet_pressure_flux_m3s,
            ratio_tolerance=pressure_outlet_source_ratio_tolerance,
        ),
        "final_negative_z_all_sections": final_all_sections_negative_z,
        "section_samples_present": bool(
            rows
            and int(last["lip_sample_count"]) > 0
            and int(last["outlet_sample_count"]) > 0
            and int(last["downstream_sample_count"]) > 0
        ),
        "cfl_below_0p5": max_cfl < 0.5,
        "projection_divergence_finite": math.isfinite(max_div_l2),
        "projection_divergence_below_tolerance": (
            math.isfinite(max_div_l2)
            and math.isfinite(max_interior_div_l2)
            and max_interior_div_l2 <= float(args.projection_divergence_tolerance)
        ),
        "pressure_projection_cg_converged_all": pressure_projection_cg_converged_all,
        "pressure_projection_cg_no_breakdown": (
            pressure_projection_cg_breakdown_count == 0
        ),
        "pressure_projection_no_physical_failure": (
            pressure_projection_physical_failure_count == 0
        ),
        "total_pressure_projection_cg_converged_all": (
            total_pressure_projection_cg_converged_all
        ),
        "total_pressure_projection_cg_no_breakdown": (
            total_pressure_projection_cg_breakdown_count == 0
        ),
    }
    if bool(outlet_to_fsi_gate_scope["hard_gate"]):
        checks["final_outlet_to_fsi_volume_source_ratio_physical"] = (
            final_outlet_to_fsi_volume_source_ratio_physical
        )
    diagnostic_checks = {
        "projection_pressure_traction_diagnostic_nonzero": max_pressure_traction > 0.0,
        "solid_model_choice_supported": args.solid_model in ("neo_hookean_mpm", "tri_mooney_shell_mpm"),
        "boundary_drive_has_no_prescribed_driver": bool(boundary_drive_compliance["compliant"]),
    }
    if not bool(outlet_to_fsi_gate_scope["hard_gate"]):
        diagnostic_checks["final_outlet_to_fsi_volume_source_ratio_physical"] = (
            final_outlet_to_fsi_volume_source_ratio_physical
        )
    completed_step_checks_passed = checks_passed(checks)
    validation_scope_complete = bool(validation_scope["validation_scope_complete"])
    validation_passed = completed_step_checks_passed if validation_scope_complete else None
    interface_constraint_note = (
        " A direct FSI interface velocity constraint is enabled on water-side "
        "probe cells before projection; boundary_drive_compliance reports it as "
        "a prescribed interface velocity constraint, not as a nozzle pressure or "
        "flow boundary."
        if fsi_velocity_constraint_blend > 0.0
        else ""
    )
    summary = {
        "case": "Squid soft robot",
        "model_class": "latest-core reduced FSI validation with true STL region diagnostics",
        "full_259k_mesh_reproduction": False,
        "full_mesh_blocker": (
            "Current simulation_core has reusable GPU fluid, Ecoflex material, "
            "surface stress, membrane, arbitrary-triangle diagnostics, and "
            "a true Neo-Hookean layered MPM path for FSI faces, but this runner is still "
            "a reduced assembly rather than a complete volumetric 259k-surface CAD "
            "reproduction of all four parts."
        ),
        "uses_generic_simulation_core": True,
        "solid_model": {
            "type": args.solid_model,
            "solid_particle_size_m": _final_row_number(last, "solid_mpm_particle_spacing_m"),
            "solid_particle_count": _final_row_int(last, "solid_mpm_particle_count"),
            "solid_mpm_grid_spacing_m": solid_mpm_grid_spacing_m,
            "solid_mpm_layers": int(args.solid_mpm_layers),
            "solid_mpm_substeps": int(solid_mpm_substeps),
            "solid_mpm_sub_dt_s": float(solid_sub_dt_s),
            "solid_mpm_stable_dt_s": float(stable_solid_dt_s),
            "solid_mpm_bounds_min_m": [
                float(value) for value in solid_mpm_bounds_min_m
            ],
            "solid_mpm_bounds_max_m": [
                float(value) for value in solid_mpm_bounds_max_m
            ],
            "solid_mpm_bounds_padding_m": float(solid_mpm_bounds_padding_m),
            "solid_mpm_cfl": float(args.solid_mpm_cfl),
            "solid_mpm_velocity_damping": float(args.solid_mpm_velocity_damping),
            "solid_mpm_substep_velocity_damping": float(solid_substep_velocity_damping),
            "solid_mpm_flip_blend": solid_mpm_flip_blend,
            "membrane_thickness_scale": membrane_thickness_scale,
            "solid_density_scale": solid_density_scale,
            "solid_density_kgm3": float(material.density_kgm3),
            "main_membrane_thickness_m": float(spec.main_membrane_thickness_m),
            "tail_membrane_thickness_m": float(spec.tail_membrane_thickness_m),
            "main_surface_mass_kg_m2": solid_surface_mass_report[
                "main_surface_mass_kg_m2"
            ],
            "tail_surface_mass_kg_m2": solid_surface_mass_report[
                "tail_surface_mass_kg_m2"
            ],
            "main_surface_mass_scale": solid_surface_mass_report[
                "main_surface_mass_scale"
            ],
            "tail_surface_mass_scale": solid_surface_mass_report[
                "tail_surface_mass_scale"
            ],
            "estimated_solid_particle_spacing_m": float(estimated_solid_particle_spacing_m),
            "total_fsi_face_area_m2": float(total_fsi_face_area_m2),
            "is_physical_mpm": args.solid_model in ("neo_hookean_mpm", "tri_mooney_shell_mpm"),
            "mooney_c1_pa": 0.5 * material.shear_modulus_pa
            if args.solid_model == "tri_mooney_shell_mpm"
            else None,
            "mooney_c2_pa": 0.0 if args.solid_model == "tri_mooney_shell_mpm" else None,
            "mooney_membrane_force_scale": args.mooney_membrane_force_scale
            if args.solid_model == "tri_mooney_shell_mpm"
            else None,
            "note": (
                "neo_hookean_mpm is the volumetric layered constitutive branch. "
                "tri_mooney_shell_mpm is the paper-calibrated arbitrary-triangle "
                "Mooney shell branch with Ecoflex shear modulus mapped to c1=mu/2. "
                "The present runner remains a reduced squid assembly until the full "
                "259k-face CAD solid model is coupled."
            ),
        },
        "fluid_grid_spacing_m": fluid_grid_spacing_m,
        "fluid_grid_min_spacing_m": fluid_grid_min_spacing_m,
        "fluid_grid_max_spacing_m": fluid_grid_max_spacing_m,
        "fluid_grid_nodes": fluid_grid_nodes,
        "fluid_grid_graded_enabled": spec.graded_grid is not None,
        "fluid_grid_refinement_region_count": (
            0 if spec.graded_grid is None else len(spec.graded_grid.refinement_regions)
        ),
        "fluid_grid_resolution": fluid_grid_resolution,
        "tail_refinement_enabled": tail_refinement_region is not None,
        "tail_refinement_geometry": tail_refinement_geometry,
        "tail_refinement_region": refinement_region_summary(tail_refinement_region),
        "time_step_scale": float(args.time_step_scale),
        "membrane_thickness_scale": membrane_thickness_scale,
        "solid_density_scale": solid_density_scale,
        "solid_surface_mass_budget": solid_surface_mass_report,
        "surface_fsi_force_spreading_enabled": True,
        "constraint_force_scale": args.constraint_force_scale,
        "fsi_constraint_force_mobility_scale_delta": constraint_force_mobility_scale_delta,
        "fsi_solid_response_mobility_coupling": fsi_solid_response_mobility_coupling,
        "fsi_response_constraint_force_mobility_scale_delta": (
            response_constraint_force_mobility_scale_delta
        ),
        "fluid_stress_action_on_fluid_enabled": True,
        "fluid_stress_action_on_fluid_note": (
            "Surface force spreading adds the opposite of sampled -pI + viscous "
            "stress traction to the fluid grid. boundary_drive_compliance only "
            "reports whether an artificial velocity/interface, pressure, flow, "
            "or traction-scale drive was prescribed. The accepted-step "
            "pressure_traction_* columns are sampled from the projection pressure "
            "field and are diagnostic evidence only, not an independent physical "
            "validation gate."
        ),
        "fsi_velocity_constraint_blend": fsi_velocity_constraint_blend,
        "fsi_constraint_force_solid_mobility_ratio": fsi_constraint_force_solid_mobility_ratio,
        "fsi_solid_response_dt_s": fsi_solid_response_dt_s,
        "max_fsi_primary_response_constraint_force_solid_mobility_ratio": (
            max_fsi_primary_response_constraint_force_solid_mobility_ratio
        ),
        "max_fsi_secondary_response_constraint_force_solid_mobility_ratio": (
            max_fsi_secondary_response_constraint_force_solid_mobility_ratio
        ),
        "fsi_velocity_target_solid_mobility_ratio": fsi_velocity_target_solid_mobility_ratio,
        "fsi_solid_response_velocity_mobility_coupling": (
            fsi_solid_response_velocity_mobility_coupling
        ),
        "fsi_velocity_target_mobility_scale_delta": velocity_target_mobility_scale_delta,
        "max_fsi_primary_velocity_target_solid_mobility_ratio": (
            max_fsi_primary_velocity_target_solid_mobility_ratio
        ),
        "max_fsi_secondary_velocity_target_solid_mobility_ratio": (
            max_fsi_secondary_velocity_target_solid_mobility_ratio
        ),
        "fsi_velocity_constraint_solid_mobility_ratio": fsi_velocity_constraint_solid_mobility_ratio,
        "fluid_substeps": effective_fluid_substeps,
        "fluid_substep_dt_s": effective_fluid_substep_dt_s,
        "adaptive_fluid_substeps_enabled": adaptive_fluid_substeps_enabled,
        "adaptive_fluid_substeps_target_cfl": float(
            args.adaptive_fluid_substeps_target_cfl
        ),
        "adaptive_fluid_substeps_max": int(args.adaptive_fluid_substeps_max),
        "adaptive_fluid_substeps_safety": float(args.adaptive_fluid_substeps_safety),
        "max_fluid_substeps": max_fluid_substeps,
        "ibm_correction_iterations": max(1, int(args.ibm_correction_iterations)),
        "ibm_correction_dt_s": float(spec.dt_s)
        / float(effective_fluid_substeps)
        / float(max(1, int(args.ibm_correction_iterations))),
        "pressure_projection_budget": pressure_projection_budget,
        "pressure_solver_requested": str(args.pressure_solver),
        "pressure_solver": pressure_solver_name,
        "pressure_solve_failure_policy": str(args.pressure_solve_failure_policy),
        "fluid_advection_scheme": str(args.fluid_advection_scheme),
        "cg_preconditioner": cg_preconditioner,
        "multigrid_cycles": multigrid_cycles,
        "effective_multigrid_cycles": effective_multigrid_cycles,
        "divergence_cleanup_iterations": projection_divergence_cleanup_iterations,
        "divergence_cleanup_relaxation": float(args.divergence_cleanup_relaxation),
        "hibm_post_dirichlet_consistency_projections": int(
            args.hibm_post_dirichlet_consistency_projections
        ),
        "fsi_coupling_mode": fsi_coupling_mode,
        "fsi_coupling_mode_report": fsi_coupling_mode_report,
        "fsi_stabilization_preset": fsi_stabilization_preset,
        "fsi_stabilization_preset_conflict_policy": (
            FSI_STABILIZATION_PRESET_CONFLICT_POLICY
        ),
        "fsi_stabilization_effective_parameters": (
            fsi_stabilization_effective_parameters
        ),
        "fsi_coupling_iterations_requested": fsi_coupling_iterations,
        "max_fsi_coupling_iterations_requested": (
            max_fsi_coupling_iterations_requested
        ),
        "fsi_coupling_adaptive_iterations_max": (
            fsi_coupling_adaptive_iterations_max
        ),
        "fsi_coupling_adaptive_iterations_residual_threshold_n": (
            fsi_coupling_adaptive_iterations_residual_threshold_n
        ),
        "fsi_coupling_adaptive_iterations_cfl_threshold": (
            fsi_coupling_adaptive_iterations_cfl_threshold
        ),
        "total_fsi_coupling_adaptive_iterations_triggered": (
            total_fsi_coupling_adaptive_iterations_triggered
        ),
        "total_fsi_coupling_adaptive_iterations_residual_triggered": (
            total_fsi_coupling_adaptive_iterations_residual_triggered
        ),
        "total_fsi_coupling_adaptive_iterations_cfl_triggered": (
            total_fsi_coupling_adaptive_iterations_cfl_triggered
        ),
        "fsi_coupling_same_step_rerun_iterations_max": (
            fsi_coupling_same_step_rerun_iterations_max
        ),
        "fsi_coupling_same_step_rerun_residual_threshold_n": (
            fsi_coupling_same_step_rerun_residual_threshold_n
        ),
        "fsi_coupling_same_step_rerun_fluid_substep_factor": (
            fsi_coupling_same_step_rerun_fluid_substep_factor
        ),
        "total_fsi_coupling_same_step_rerun_triggered": (
            total_fsi_coupling_same_step_rerun_triggered
        ),
        "total_fsi_coupling_same_step_rerun_count": (
            total_fsi_coupling_same_step_rerun_count
        ),
        "max_fsi_coupling_same_step_rerun_initial_residual_norm_n": (
            max_fsi_coupling_same_step_rerun_initial_residual_norm_n
        ),
        "fsi_coupling_residual_continuation_iterations_max": (
            fsi_coupling_residual_continuation_iterations_max
        ),
        "fsi_coupling_residual_continuation_threshold_n": (
            fsi_coupling_residual_continuation_threshold_n
        ),
        "fsi_coupling_residual_continuation_rebound_secant_from_best": (
            fsi_coupling_residual_continuation_rebound_secant_from_best
        ),
        "fsi_coupling_residual_continuation_rebound_secant_factor": (
            fsi_coupling_residual_continuation_rebound_secant_factor
        ),
        "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max": (
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
        ),
        "max_fsi_coupling_residual_continuation_iteration_count": (
            max_fsi_coupling_residual_continuation_iteration_count
        ),
        "total_fsi_coupling_residual_continuation_iteration_count": (
            total_fsi_coupling_residual_continuation_iteration_count
        ),
        "max_fsi_coupling_residual_continuation_rebound_secant_count": (
            max_fsi_coupling_residual_continuation_rebound_secant_count
        ),
        "total_fsi_coupling_residual_continuation_rebound_secant_count": (
            total_fsi_coupling_residual_continuation_rebound_secant_count
        ),
        "max_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count": (
            max_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count
        ),
        "total_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count": (
            total_fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count
        ),
        "fsi_coupling_solver": fsi_coupling_solver,
        "max_fsi_coupling_iterations_used": max_fsi_coupling_iterations_used,
        "max_fsi_coupling_rejected_trial_count": (
            max_fsi_coupling_rejected_trial_count
        ),
        "max_fsi_coupling_rejected_trial_backtrack_count": (
            max_fsi_coupling_rejected_trial_backtrack_count
        ),
        "max_fsi_coupling_residual_growth_rejected_trial_count": (
            max_fsi_coupling_residual_growth_rejected_trial_count
        ),
        "max_fsi_coupling_max_residual_rejected_trial_count": (
            max_fsi_coupling_max_residual_rejected_trial_count
        ),
        "max_fsi_coupling_trial_cfl_rejected_count": (
            max_fsi_coupling_trial_cfl_rejected_count
        ),
        "max_fsi_coupling_trial_interior_divergence_rejected_count": (
            max_fsi_coupling_trial_interior_divergence_rejected_count
        ),
        "max_fsi_coupling_trust_region_limited_update_count": (
            max_fsi_coupling_trust_region_limited_update_count
        ),
        "max_fsi_coupling_trust_region_shrink_count": (
            max_fsi_coupling_trust_region_shrink_count
        ),
        "max_fsi_coupling_trust_region_growth_count": (
            max_fsi_coupling_trust_region_growth_count
        ),
        "max_fsi_coupling_trust_region_rebound_backtrack_count": (
            max_fsi_coupling_trust_region_rebound_backtrack_count
        ),
        "max_fsi_coupling_trust_region_rebound_stop_count": (
            max_fsi_coupling_trust_region_rebound_stop_count
        ),
        "max_fsi_coupling_trust_region_rebound_stop_suppressed_count": (
            max_fsi_coupling_trust_region_rebound_stop_suppressed_count
        ),
        "total_fsi_coupling_rejected_trial_count": (
            total_fsi_coupling_rejected_trial_count
        ),
        "total_fsi_coupling_rejected_trial_backtrack_count": (
            total_fsi_coupling_rejected_trial_backtrack_count
        ),
        "total_fsi_coupling_residual_growth_rejected_trial_count": (
            total_fsi_coupling_residual_growth_rejected_trial_count
        ),
        "total_fsi_coupling_max_residual_rejected_trial_count": (
            total_fsi_coupling_max_residual_rejected_trial_count
        ),
        "total_fsi_coupling_trial_cfl_rejected_count": (
            total_fsi_coupling_trial_cfl_rejected_count
        ),
        "total_fsi_coupling_trial_interior_divergence_rejected_count": (
            total_fsi_coupling_trial_interior_divergence_rejected_count
        ),
        "total_fsi_coupling_trust_region_limited_update_count": (
            total_fsi_coupling_trust_region_limited_update_count
        ),
        "total_fsi_coupling_trust_region_shrink_count": (
            total_fsi_coupling_trust_region_shrink_count
        ),
        "total_fsi_coupling_trust_region_growth_count": (
            total_fsi_coupling_trust_region_growth_count
        ),
        "total_fsi_coupling_trust_region_rebound_backtrack_count": (
            total_fsi_coupling_trust_region_rebound_backtrack_count
        ),
        "total_fsi_coupling_trust_region_rebound_stop_count": (
            total_fsi_coupling_trust_region_rebound_stop_count
        ),
        "total_fsi_coupling_trust_region_rebound_stop_suppressed_count": (
            total_fsi_coupling_trust_region_rebound_stop_suppressed_count
        ),
        "max_fsi_coupling_accepted_trial_cfl": (
            max_fsi_coupling_accepted_trial_cfl
        ),
        "max_fsi_coupling_trial_cfl": max_fsi_coupling_trial_cfl,
        "fsi_coupling_trial_interior_divergence_tolerance": (
            fsi_coupling_trial_interior_divergence_tolerance
        ),
        "max_fsi_coupling_accepted_trial_interior_divergence_l2": (
            max_fsi_coupling_accepted_trial_interior_divergence_l2
        ),
        "max_fsi_coupling_trial_interior_divergence_l2": (
            max_fsi_coupling_trial_interior_divergence_l2
        ),
        "max_fsi_coupling_iqn_ils_least_squares_update_count": (
            max_fsi_coupling_iqn_ils_least_squares_update_count
        ),
        "fsi_coupling_tolerance_n": fsi_coupling_tolerance_n,
        "fsi_marker_coupling_tolerance_mps": fsi_marker_coupling_tolerance_mps,
        "fsi_coupling_target_map_relaxation": fsi_coupling_target_map_relaxation,
        "fsi_coupling_rejected_trial_backtrack": (
            fsi_coupling_rejected_trial_backtrack
        ),
        "fsi_coupling_residual_growth_rejection_factor": (
            fsi_coupling_residual_growth_rejection_factor
        ),
        "fsi_coupling_max_accepted_residual_n": (
            fsi_coupling_max_accepted_residual_n
        ),
        "fsi_coupling_trust_region_force_increment_n": (
            fsi_coupling_trust_region_force_increment_n
        ),
        "fsi_coupling_trust_region_adaptive": (
            fsi_coupling_trust_region_adaptive
        ),
        "fsi_coupling_trust_region_shrink_factor": (
            fsi_coupling_trust_region_shrink_factor
        ),
        "fsi_coupling_trust_region_growth_factor": (
            fsi_coupling_trust_region_growth_factor
        ),
        "fsi_coupling_trust_region_rebound_factor": (
            fsi_coupling_trust_region_rebound_factor
        ),
        "fsi_coupling_trust_region_rebound_backtrack": (
            fsi_coupling_trust_region_rebound_backtrack
        ),
        "fsi_coupling_trust_region_rebound_stop_factor": (
            fsi_coupling_trust_region_rebound_stop_factor
        ),
        "fsi_coupling_trust_region_rebound_stop_max_residual_n": (
            fsi_coupling_trust_region_rebound_stop_max_residual_n
        ),
        "max_fsi_coupling_residual_norm_n": max_fsi_coupling_residual_norm_n,
        "max_fsi_coupling_residual_norm_mps": (
            max_fsi_coupling_residual_norm_mps
        ),
        "max_fsi_coupling_residual_max_mps": max_fsi_coupling_residual_max_mps,
        "max_fsi_coupling_interface_map_amplification": (
            max_fsi_coupling_interface_map_amplification
        ),
        "max_fsi_coupling_residual_jacobian_amplification": (
            max_fsi_coupling_residual_jacobian_amplification
        ),
        "max_fsi_coupling_physical_interface_map_amplification": (
            max_fsi_coupling_physical_interface_map_amplification
        ),
        "max_fsi_coupling_physical_residual_jacobian_amplification": (
            max_fsi_coupling_physical_residual_jacobian_amplification
        ),
        "max_fsi_coupling_raw_interface_map_amplification": (
            max_fsi_coupling_raw_interface_map_amplification
        ),
        "max_fsi_coupling_raw_residual_jacobian_amplification": (
            max_fsi_coupling_raw_residual_jacobian_amplification
        ),
        "max_fsi_coupling_interface_map_amplification_sample_count": (
            max_fsi_coupling_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_residual_jacobian_amplification_sample_count
        ),
        "max_fsi_coupling_physical_interface_map_amplification_sample_count": (
            max_fsi_coupling_physical_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_physical_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_physical_residual_jacobian_amplification_sample_count
        ),
        "max_fsi_coupling_raw_interface_map_amplification_sample_count": (
            max_fsi_coupling_raw_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_raw_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_raw_residual_jacobian_amplification_sample_count
        ),
        "fsi_coupling_raw_interface_map_strict_physical": (
            fsi_coupling_raw_interface_map_strict_physical
        ),
        "fsi_physical_interface_map_stability": fsi_physical_interface_map_stability,
        "fsi_coupling_not_converged_count": fsi_coupling_not_converged_count,
        "reuse_accepted_fsi_trial_state": reuse_accepted_fsi_trial_state,
        "accepted_fsi_trial_state_reuse_count": accepted_fsi_trial_state_reuse_count,
        "accepted_fsi_trial_state_readvance_count": (
            accepted_fsi_trial_state_readvance_count
        ),
        "fsi_all_trials_rejected_count": fsi_all_trials_rejected_count,
        "fsi_zero_force_commit_blocked_count": fsi_zero_force_commit_blocked_count,
        "fsi_coupling_note": (
            "Physical MPM solid models use Taichi device-side reduced-state, solid, and "
            "fluid snapshots for step-internal interface-reaction fixed-point iterations when "
            "fsi_coupling_iterations_requested > 1. By default the final step is "
            "re-advanced from the saved state with the accepted interface-reaction guess "
            "from the selected fixed-point solver. If explicit accepted-trial reuse is "
            "enabled, the runner reuses only a core-confirmed final accepted trial state "
            "with full reports; otherwise it falls back to the re-advance path. "
            "max_fsi_coupling_interface_map_amplification is the actual solver-map "
            "amplification after fsi_coupling_target_map_relaxation; "
            "max_fsi_coupling_physical_interface_map_amplification reports the "
            "selected unrelaxed target map before solver relaxation, which may "
            "include Robin stabilization depending on target mode; "
            "max_fsi_coupling_raw_interface_map_amplification reports the raw "
            "fluid-reaction target with Robin subtracted back out and is the "
            "completed-step stability gate for physical MPM coupling when more "
            "than one FSI iteration is requested."
        ),
        "interface_reaction_relaxation": interface_reaction_relaxation,
        "interface_reaction_aitken": interface_reaction_aitken,
        "interface_reaction_aitken_lower_bound": (
            interface_reaction_aitken_lower_bound
        ),
        "interface_reaction_aitken_upper_bound": (
            interface_reaction_aitken_upper_bound
        ),
        "interface_reaction_aitken_note": (
            "When enabled, Aitken Delta^2 adapts both step-internal interface-reaction fixed-point "
            "updates and the accepted-step next interface-reaction residual; relaxation is clipped "
            "to [interface_reaction_aitken_lower_bound, interface_reaction_aitken_upper_bound]."
        ),
        "interface_reaction_passivity_limit": interface_reaction_passivity_limit,
        "interface_reaction_passivity_limiter_note": (
            "Disabled by default for physically honest squid validation because it "
            "can change the committed projected-IBM interface reaction. If explicitly "
            "enabled, the limiter projects only the committed interface reaction to "
            "the zero-power boundary when it would inject energy after the fixed-point "
            "solve. It does not alter trial guesses and does not prescribe nozzle "
            "velocity, pressure, or flow."
        ),
        "interface_reaction_robin_impedance_ns_m": (
            interface_reaction_robin_impedance_ns_m
        ),
        "interface_reaction_robin_matrix_impedance_ns_m": (
            interface_reaction_robin_matrix_impedance_ns_m
        ),
        "interface_reaction_robin_target_mode": interface_reaction_robin_target_mode,
        "interface_reaction_robin_impedance_note": (
            "Explicit opt-in Phase-C coupling stabilizer. The Robin-Neumann "
            "impedance force -Z*(v_n-v_{n-1}) is applied on the fluid-side "
            "interface pass. The target mode controls whether the committed "
            "solid reaction keeps that stabilized target or subtracts the "
            "Robin term back out as a physical force target."
        ),
        "interface_reaction_robin_matrix_impedance_note": (
            "Explicit opt-in Phase-1B pressure-matrix impedance. Positive values "
            "scatter per-marker interface terms into the FV-CG pressure operator; "
            "the default 0 preserves the previous partitioned solve."
        ),
        "fsi_grid_force_decomposition_source": "last_correction_grid_force_vs_primary_plus_secondary_fluid_force",
        "fsi_fluid_reaction_action_reaction_note": (
            "fsi_action_reaction_* and fsi_fluid_reaction_action_reaction_* are retained "
            "as diagnostics only because their targets are constructed as equal-and-opposite "
            "reaction forces. fsi_grid_force_decomposition_* is also diagnostic only because "
            "the grid force and component forces are accumulated from the same scattered "
            "node forces. Active validation relies on independent physical quantities such "
            "as outlet/source flux, finite fields, CFL, divergence, sample coverage, and "
            "pressure-loaded solid response."
        ),
        "fsi_surface_positions_follow_solid_z": True,
        "fsi_surface_position_update_note": (
            "The reduced FSI triangle centroids are translated by the current main/tail "
            "solid z displacement before IBM force spreading and pressure traction sampling. "
            "Normals remain rest normals in this reduced shell validation."
        ),
        "pressure_outlet_boundary": pressure_outlet_boundary_report,
        "pressure_outlet_zmin_enabled": pressure_outlet_zmin_enabled,
        "pressure_outlet_zmin_no_backflow_enabled": pressure_outlet_zmin_enabled,
        "boundary_drive_compliance": boundary_drive_compliance,
        "boundary_drive_compliance_gate": "diagnostic_only",
        "fluid_surface_traction_source": "sampled_projected_fluid_stress_field",
        "fluid_to_solid_interface_reaction_enabled": True,
        "tail_hydraulic_scalar_drive_enabled": False,
        "reduced_chamber_nozzle_obstacles_enabled": not args.disable_reduced_obstacles,
        "pressure_boundary_shell_mapping": asdict(pressure_boundary_mapping),
        "pressure_load_source_region_id": int(pressure_load_source_region_id),
        "pressure_load_region_id": int(pressure_load_region_id),
        "pressure_load_direction": tuple(float(v) for v in pressure_load_direction),
        "pressure_closure_normal": tuple(float(v) for v in pressure_closure_normal),
        "pressure_far_side_normal_sign": float(pressure_far_side_normal_sign),
        "shell_primary_region_id": int(primary_shell_region_id),
        "shell_secondary_region_id": int(secondary_shell_region_id),
        "source_config_fluid_active_mask_requested": (
            source_config_fluid_active_mask_requested
        ),
        "source_config_reduced_water_intersection_requested": (
            source_config_reduced_water_intersection_requested
        ),
        "initial_fluid_obstacle_mode": initial_fluid_obstacle_mode,
        "source_config_fluid_topology": source_config_fluid_topology_report,
        "source_config_region14_aperture_requested": (
            source_config_region14_aperture_requested
        ),
        "region14_aperture_carve_enabled": region14_aperture_carve_enabled,
        "region14_aperture_carve_source": region14_aperture_carve_source,
        "open_downstream_farfield_enabled": bool(spec.downstream_farfield_open_enabled),
        "region14_aperture_geometry": region14_aperture_geometry,
        "reduced_water_geometry": reduced_water_geometry_report(spec),
        "source_config_used_as_input_only": str(source_config_path),
        "cad_provenance": cad_provenance,
        "real_cad_step_path": cad_provenance.get("cad_step_path"),
        "real_cad_step_direct_binding": bool(
            cad_provenance.get("direct_cad_step_binding", False)
        ),
        "real_cad_step_derived_surface_mesh_binding": bool(
            cad_provenance.get("step_derived_surface_mesh_binding", False)
        ),
        "real_cad_step_binding": real_cad_step_binding,
        "pressure_schedule_input": pressure_schedule_input,
        "spec": asdict(spec),
        "tri_surface_diagnostics": tri_metadata,
        "active_water_connectivity": active_water,
        "material": {
            "name": material.name,
            "density_kgm3": material.density_kgm3,
            "youngs_modulus_pa": material.youngs_modulus_pa,
            "poissons_ratio": material.poissons_ratio,
            "shear_modulus_pa": material.shear_modulus_pa,
            "bulk_modulus_pa": material.bulk_modulus_pa,
            "modulus_100_pa": material.modulus_100_pa,
            "source": material.source,
            "calibration_note": material.calibration_note,
        },
        "steps": step_count,
        "requested_steps": step_count,
        "completed_steps": len(rows),
        "full_pressure_waveform_steps": full_pressure_waveform_steps,
        "steps_explicit": bool(getattr(args, "steps_explicit", True)),
        "partial_run": partial_run_stopped,
        "partial_run_reason": partial_run_reason if partial_run_stopped else None,
        "validation_scope": validation_scope["validation_scope"],
        "validation_scope_complete": validation_scope["validation_scope_complete"],
        "validation_scope_reason": validation_scope["validation_scope_reason"],
        "completed_step_checks_passed": completed_step_checks_passed,
        "pressure_flux_trend": pressure_flux_trend,
        "timing": timing_summary,
        "runtime_budget": runtime_budget,
        "max_wall_time_s": max_wall_time_s,
        "resume_from_checkpoint": bool(args.resume_from_checkpoint),
        "checkpoint_every_step": bool(args.checkpoint_every_step),
        "checkpoint_path": str(run_checkpoint_path),
        "first_step_executed": int(first_step) if rows else None,
        "projection_iterations": args.projection_iterations,
        "cg_tolerance": cg_tolerance,
        "projection_divergence_tolerance": args.projection_divergence_tolerance,
        "history_csv": str(history_path),
        "final": last,
        "nonfinite_diagnostics": nonfinite_diagnostics,
        "checks": checks,
        "diagnostic_checks": diagnostic_checks,
        "validation_passed": validation_passed,
        "reproduction_status": (
            "reduced_validation_partial"
            if not validation_scope_complete
            else (
                "reduced_validation_passed"
                if validation_passed
                else "reduced_validation_failed"
            )
        ),
        "max_abs_main_displacement_m": max_abs_main,
        "max_abs_tail_displacement_m": max_abs_tail,
        "max_cfl": max_cfl,
        "max_divergence_l2": max_div_l2,
        "max_interior_divergence_l2": max_interior_div_l2,
        "max_outlet_negative_z_flow_m3s": max_outlet_negative_z,
        "final_outlet_negative_z_flow_m3s": final_outlet_negative_z,
        "max_pressure_traction_abs_force_n": max_pressure_traction,
        "max_viscous_traction_force_n": max_viscous_traction,
        "max_fluid_stress_traction_force_n": max_fluid_stress_traction,
        "max_projected_ibm_residual_mps": max_projected_ibm,
        "max_pressure_projection_cg_project_calls": max_pressure_projection_cg_project_calls,
        "mean_pressure_projection_cg_project_calls": mean_pressure_projection_cg_project_calls,
        "max_pressure_projection_cg_iterations_total": (
            max_pressure_projection_cg_iterations_total
        ),
        "mean_pressure_projection_cg_iterations_total": (
            mean_pressure_projection_cg_iterations_total
        ),
        "max_pressure_projection_cg_iterations_max": max_pressure_projection_cg_iterations_max,
        "max_pressure_projection_cg_relative_residual": (
            max_pressure_projection_cg_relative_residual
        ),
        "max_pressure_projection_cg_initial_relative_residual": (
            max_pressure_projection_cg_initial_relative_residual
        ),
        "pressure_projection_cg_converged_all": pressure_projection_cg_converged_all,
        "pressure_projection_cg_breakdown_count": pressure_projection_cg_breakdown_count,
        "pressure_projection_physical_failure": (
            pressure_projection_physical_failure_count > 0
        ),
        "pressure_projection_physical_failure_count": (
            pressure_projection_physical_failure_count
        ),
        "max_hibm_unreached_incompatible_component_count": (
            max_hibm_unreached_incompatible_component_count
        ),
        "max_hibm_unreached_component_raw_count": (
            max_hibm_unreached_component_raw_count
        ),
        "max_hibm_unreached_largest_component_cell_count": (
            max_hibm_unreached_largest_component_cell_count
        ),
        "max_hibm_unreached_singleton_component_count": (
            max_hibm_unreached_singleton_component_count
        ),
        "max_hibm_unreached_small_component_count": (
            max_hibm_unreached_small_component_count
        ),
        "max_hibm_unreached_small_component_cell_count": (
            max_hibm_unreached_small_component_cell_count
        ),
        "max_hibm_projection_overflow_singleton_cleanup_cell_count": (
            max_hibm_projection_overflow_singleton_cleanup_cell_count
        ),
        "max_hibm_projection_overflow_singleton_cleanup_component_count": (
            max_hibm_projection_overflow_singleton_cleanup_component_count
        ),
        "max_hibm_projection_tiny_unreached_cleanup_cell_count": (
            max_hibm_projection_tiny_unreached_cleanup_cell_count
        ),
        "max_hibm_projection_tiny_unreached_cleanup_component_count": (
            max_hibm_projection_tiny_unreached_cleanup_component_count
        ),
        "max_hibm_unreached_component_rhs_mean_max_abs": (
            max_hibm_unreached_component_rhs_mean_max_abs
        ),
        "max_hibm_unreached_component_rhs_integral_max_abs": (
            max_hibm_unreached_component_rhs_integral_max_abs
        ),
        "max_pressure_interface_matrix_diagonal_integral": (
            max_pressure_interface_matrix_diagonal_integral
        ),
        "max_pressure_interface_matrix_rhs_integral": (
            max_pressure_interface_matrix_rhs_integral
        ),
        "max_pressure_interface_matrix_max_abs_diagonal": (
            max_pressure_interface_matrix_max_abs_diagonal
        ),
        "max_hibm_pressure_neumann_raw_transmissibility_m": (
            max_hibm_pressure_neumann_raw_transmissibility_m
        ),
        "max_hibm_pressure_neumann_transmissibility_limit_m": (
            max_hibm_pressure_neumann_transmissibility_limit_m
        ),
        "max_hibm_pressure_neumann_transmissibility_capped_row_count": (
            max_hibm_pressure_neumann_transmissibility_capped_row_count
        ),
        "max_pressure_interface_matrix_active_cells": (
            max_pressure_interface_matrix_active_cells
        ),
        "max_fsi_trial_pressure_projection_cg_project_calls": (
            max_fsi_trial_pressure_projection_cg_project_calls
        ),
        "mean_fsi_trial_pressure_projection_cg_project_calls": (
            mean_fsi_trial_pressure_projection_cg_project_calls
        ),
        "max_fsi_trial_pressure_projection_cg_iterations_total": (
            max_fsi_trial_pressure_projection_cg_iterations_total
        ),
        "mean_fsi_trial_pressure_projection_cg_iterations_total": (
            mean_fsi_trial_pressure_projection_cg_iterations_total
        ),
        "max_fsi_trial_pressure_projection_cg_iterations_max": (
            max_fsi_trial_pressure_projection_cg_iterations_max
        ),
        "max_fsi_trial_pressure_projection_cg_relative_residual": (
            max_fsi_trial_pressure_projection_cg_relative_residual
        ),
        "max_fsi_trial_pressure_projection_cg_initial_relative_residual": (
            max_fsi_trial_pressure_projection_cg_initial_relative_residual
        ),
        "fsi_trial_pressure_projection_cg_converged_all": (
            fsi_trial_pressure_projection_cg_converged_all
        ),
        "fsi_trial_pressure_projection_cg_breakdown_count": (
            fsi_trial_pressure_projection_cg_breakdown_count
        ),
        "max_total_pressure_projection_cg_project_calls": (
            max_total_pressure_projection_cg_project_calls
        ),
        "mean_total_pressure_projection_cg_project_calls": (
            mean_total_pressure_projection_cg_project_calls
        ),
        "max_total_pressure_projection_cg_iterations_total": (
            max_total_pressure_projection_cg_iterations_total
        ),
        "mean_total_pressure_projection_cg_iterations_total": (
            mean_total_pressure_projection_cg_iterations_total
        ),
        "max_total_pressure_projection_cg_iterations_max": (
            max_total_pressure_projection_cg_iterations_max
        ),
        "max_total_pressure_projection_cg_relative_residual": (
            max_total_pressure_projection_cg_relative_residual
        ),
        "max_total_pressure_projection_cg_initial_relative_residual": (
            max_total_pressure_projection_cg_initial_relative_residual
        ),
        "total_pressure_projection_cg_converged_all": (
            total_pressure_projection_cg_converged_all
        ),
        "total_pressure_projection_cg_breakdown_count": (
            total_pressure_projection_cg_breakdown_count
        ),
        "max_abs_pressure_load_pa": max_abs_pressure_load_pa,
        "max_abs_main_volume_flux_m3s": max_abs_volume_flux_m3s,
        "max_abs_fsi_volume_source_m3s": max_abs_fsi_volume_source_m3s,
        "final_fsi_volume_source_m3s": final_fsi_volume_source_m3s,
        "final_pressure_outlet_source_volume_flux_m3s": final_pressure_outlet_source_volume_flux_m3s,
        "final_pressure_outlet_positive_source_volume_flux_m3s": (
            final_pressure_outlet_positive_source_volume_flux_m3s
        ),
        "final_pressure_outlet_abs_source_volume_flux_m3s": (
            final_pressure_outlet_abs_source_volume_flux_m3s
        ),
        "final_pressure_outlet_reachable_source_volume_flux_m3s": (
            final_pressure_outlet_reachable_source_volume_flux_m3s
        ),
        "final_pressure_outlet_unreached_source_volume_flux_m3s": (
            final_pressure_outlet_unreached_source_volume_flux_m3s
        ),
        "final_pressure_outlet_reachability_valid": (
            final_pressure_outlet_reachability_valid
        ),
        "final_pressure_outlet_reachability_revision": (
            final_pressure_outlet_reachability_revision
        ),
        "final_pressure_outlet_reachable_source_cell_count": (
            final_pressure_outlet_reachable_source_cell_count
        ),
        "final_pressure_outlet_unreached_source_cell_count": (
            final_pressure_outlet_unreached_source_cell_count
        ),
        "final_pressure_outlet_unreached_source_abs_flux_m3s": (
            final_pressure_outlet_unreached_source_abs_flux_m3s
        ),
        "final_pressure_outlet_unreached_source_centroid_x_m": (
            final_pressure_outlet_unreached_source_centroid_x_m
        ),
        "final_pressure_outlet_unreached_source_centroid_y_m": (
            final_pressure_outlet_unreached_source_centroid_y_m
        ),
        "final_pressure_outlet_unreached_source_centroid_z_m": (
            final_pressure_outlet_unreached_source_centroid_z_m
        ),
        "final_pressure_outlet_unreached_source_min_x_m": (
            final_pressure_outlet_unreached_source_min_x_m
        ),
        "final_pressure_outlet_unreached_source_min_y_m": (
            final_pressure_outlet_unreached_source_min_y_m
        ),
        "final_pressure_outlet_unreached_source_min_z_m": (
            final_pressure_outlet_unreached_source_min_z_m
        ),
        "final_pressure_outlet_unreached_source_max_x_m": (
            final_pressure_outlet_unreached_source_max_x_m
        ),
        "final_pressure_outlet_unreached_source_max_y_m": (
            final_pressure_outlet_unreached_source_max_y_m
        ),
        "final_pressure_outlet_unreached_source_max_z_m": (
            final_pressure_outlet_unreached_source_max_z_m
        ),
        "final_pressure_outlet_velocity_flux_m3s": final_pressure_outlet_velocity_flux_m3s,
        "final_pressure_outlet_velocity_to_source_ratio": (
            final_pressure_outlet_velocity_to_source_ratio
        ),
        "final_pressure_outlet_velocity_to_net_source_ratio": (
            final_pressure_outlet_velocity_to_net_source_ratio
        ),
        "final_pressure_outlet_velocity_to_positive_source_ratio": (
            final_pressure_outlet_velocity_to_positive_source_ratio
        ),
        "final_pressure_outlet_velocity_to_abs_source_ratio": (
            final_pressure_outlet_velocity_to_abs_source_ratio
        ),
        "final_pressure_outlet_pressure_flux_m3s": final_pressure_outlet_pressure_flux_m3s,
        "final_pressure_outlet_pressure_to_source_ratio": (
            final_pressure_outlet_pressure_to_source_ratio
        ),
        "final_pressure_outlet_pressure_to_net_source_ratio": (
            final_pressure_outlet_pressure_to_net_source_ratio
        ),
        "final_pressure_outlet_pressure_to_positive_source_ratio": (
            final_pressure_outlet_pressure_to_positive_source_ratio
        ),
        "final_pressure_outlet_pressure_to_abs_source_ratio": (
            final_pressure_outlet_pressure_to_abs_source_ratio
        ),
        "final_pressure_outlet_projection_pre_velocity_flux_m3s": (
            final_pressure_outlet_projection_pre_velocity_flux_m3s
        ),
        "final_pressure_outlet_projection_post_pressure_velocity_flux_m3s": (
            final_pressure_outlet_projection_post_pressure_velocity_flux_m3s
        ),
        "final_pressure_outlet_projection_post_boundary_velocity_flux_m3s": (
            final_pressure_outlet_projection_post_boundary_velocity_flux_m3s
        ),
        "pressure_outlet_source_ratio_tolerance": pressure_outlet_source_ratio_tolerance,
        "final_outlet_to_fsi_volume_source_ratio": final_outlet_to_fsi_volume_source_ratio,
        "outlet_to_fsi_volume_source_gate_scope": outlet_to_fsi_gate_scope,
        "final_outlet_to_main_volume_flux_ratio": final_outlet_flux_ratio,
        "required_min_outlet_to_main_volume_flux_ratio": args.min_outlet_to_main_volume_flux_ratio,
        "final_downstream_to_main_volume_flux_ratio": final_downstream_flux_ratio,
        "max_outlet_to_main_volume_flux_ratio": max_outlet_flux_ratio,
        "min_outlet_to_main_volume_flux_ratio": min_outlet_flux_ratio,
        "max_fsi_velocity_constraint_active_cells": max_velocity_constraint_cells,
        "max_fsi_velocity_constraint_delta_mps": max_velocity_constraint_delta_mps,
        "max_fsi_velocity_constraint_momentum_delta_n_s": (
            max_velocity_constraint_momentum_delta_n_s
        ),
        "max_fsi_velocity_constraint_equivalent_force_norm_n": (
            max_velocity_constraint_equivalent_force_norm_n
        ),
        "max_fsi_velocity_constraint_primary_momentum_delta_n_s": (
            max_velocity_constraint_primary_momentum_delta_n_s
        ),
        "max_fsi_velocity_constraint_secondary_momentum_delta_n_s": (
            max_velocity_constraint_secondary_momentum_delta_n_s
        ),
        "max_fsi_velocity_constraint_primary_equivalent_force_norm_n": (
            max_velocity_constraint_primary_equivalent_force_norm_n
        ),
        "max_fsi_velocity_constraint_secondary_equivalent_force_norm_n": (
            max_velocity_constraint_secondary_equivalent_force_norm_n
        ),
        "max_fsi_velocity_constraint_step_impulse_n_s": (
            max_velocity_constraint_step_impulse_n_s
        ),
        "max_fsi_velocity_constraint_primary_step_impulse_n_s": (
            max_velocity_constraint_primary_step_impulse_n_s
        ),
        "max_fsi_velocity_constraint_secondary_step_impulse_n_s": (
            max_velocity_constraint_secondary_step_impulse_n_s
        ),
        "max_fsi_velocity_constraint_step_equivalent_force_norm_n": (
            max_velocity_constraint_step_equivalent_force_norm_n
        ),
        "max_fsi_velocity_constraint_primary_step_equivalent_force_norm_n": (
            max_velocity_constraint_primary_step_equivalent_force_norm_n
        ),
        "max_fsi_velocity_constraint_secondary_step_equivalent_force_norm_n": (
            max_velocity_constraint_secondary_step_equivalent_force_norm_n
        ),
        "max_fsi_grid_force_decomposition_residual_n": max_fsi_grid_decomposition_residual_n,
        "max_fsi_grid_force_decomposition_relative_error": max_fsi_grid_decomposition_relative_error,
        "max_fsi_action_reaction_residual_n": max_fsi_action_reaction_residual_n,
        "max_fsi_action_reaction_relative_error": max_fsi_action_reaction_relative_error,
        "max_fsi_fluid_reaction_action_reaction_residual_z_n": (
            max_fluid_reaction_action_reaction_residual_z_n
        ),
        "max_fsi_fluid_reaction_action_reaction_relative_error": (
            max_fluid_reaction_action_reaction_relative_error
        ),
        "max_fsi_fluid_reaction_full_3d_residual_n": max_fluid_reaction_full_residual_n,
        "max_fsi_fluid_reaction_full_3d_relative_error": max_fluid_reaction_full_relative_error,
        "interface_reaction_aitken_lower_bound": (
            interface_reaction_aitken_lower_bound
        ),
        "interface_reaction_aitken_upper_bound": (
            interface_reaction_aitken_upper_bound
        ),
        "max_interface_reaction_relaxation_effective": max_interface_reaction_relaxation_effective,
        "min_interface_reaction_relaxation_effective": min_interface_reaction_relaxation_effective,
        "max_positive_main_interface_reaction_power_w": max_positive_main_interface_reaction_power_w,
        "final_main_interface_reaction_power_w": final_main_interface_reaction_power_w,
        "main_interface_reaction_passivity_limit_count": main_interface_reaction_passivity_limit_count,
        "tail_interface_reaction_passivity_limit_count": tail_interface_reaction_passivity_limit_count,
        "max_abs_main_interface_reaction_robin_impedance_force_n": (
            max_abs_main_interface_reaction_robin_impedance_force_n
        ),
        "max_abs_tail_interface_reaction_robin_impedance_force_n": (
            max_abs_tail_interface_reaction_robin_impedance_force_n
        ),
        "max_solid_mpm_transfer_relative_error": max_solid_mpm_transfer_error,
        "max_solid_mpm_total_force_n": max_solid_mpm_total_force_n,
        "max_solid_mpm_active_grid_nodes": max_solid_mpm_active_grid_nodes,
        "min_fsi_probe_valid_fraction": min_probe_valid_fraction,
        "min_fsi_force_probe_valid_fraction": min_force_probe_valid_fraction,
        "max_fsi_active_force_cells": max_active_force_cells,
        "interpretation_note": (
            "Default mode does not prescribe a nozzle velocity boundary. The structure step "
            "computes membrane motion, the true STL FSI faces spread IBM no-slip constraint "
            "forces plus the opposite of sampled fluid stress traction to the fluid grid, "
            "and lip/outlet/downstream flow is sampled from the projected fluid field. The "
            "derived nozzle_velocity_z_mps column is retained only as a kinematic diagnostic; "
            "validation uses the sampled outlet-to-FSI-volume-source flux ratio."
            + interface_constraint_note
        ),
    }
    summary_path = output_dir / "summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    process_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": run_process_completion_status(
                    validation_scope_complete=validation_scope_complete,
                    validation_passed=validation_passed,
                    partial_run_stopped=partial_run_stopped,
                    requested_steps=step_count,
                    completed_steps=len(rows),
                ),
                "validation_passed": validation_passed,
                "reproduction_status": summary["reproduction_status"],
                "validation_scope": summary["validation_scope"],
                "validation_scope_complete": summary["validation_scope_complete"],
                "validation_scope_reason": summary["validation_scope_reason"],
                "completed_step_checks_passed": completed_step_checks_passed,
                "finished_at_unix": time.time(),
                "summary_json": str(summary_path),
                "history_csv": str(history_path),
                "requested_steps": step_count,
                "completed_steps": len(rows),
                "full_pressure_waveform_steps": full_pressure_waveform_steps,
                "partial_run_reason": partial_run_reason if partial_run_stopped else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_sharp_case_run_report(context: Mapping[str, object]) -> dict[str, object]:
    FSI_STABILIZATION_PRESET_CONFLICT_POLICY = _context_value(context, "FSI_STABILIZATION_PRESET_CONFLICT_POLICY")
    _final_row_int = _context_value(context, "_final_row_int")
    _final_row_number = _context_value(context, "_final_row_number")
    _final_row_number_or_none = _context_value(context, "_final_row_number_or_none")
    _required_finite_row_number = _context_value(context, "_required_finite_row_number")
    _row_bool = _context_value(context, "_row_bool")
    _rows_any_bool = _context_value(context, "_rows_any_bool")
    _rows_max_int = _context_value(context, "_rows_max_int")
    adaptive_fluid_substeps_enabled = _context_value(context, "adaptive_fluid_substeps_enabled")
    args = _context_value(context, "args")
    asdict = _context_value(context, "asdict")
    cad_provenance = _context_value(context, "cad_provenance")
    checks_passed = _context_value(context, "checks_passed")
    effective_fluid_substep_dt_s = _context_value(context, "effective_fluid_substep_dt_s")
    effective_fluid_substeps = _context_value(context, "effective_fluid_substeps")
    finite_field_diagnostics = _context_value(context, "finite_field_diagnostics")
    finite_required_row_fields_for_mode = _context_value(context, "finite_required_row_fields_for_mode")
    fluid_grid_axis_max_spacing_m = _context_value(context, "fluid_grid_axis_max_spacing_m")
    fluid_grid_axis_min_spacing_m = _context_value(context, "fluid_grid_axis_min_spacing_m")
    fluid_grid_resolution = _context_value(context, "fluid_grid_resolution")
    fluid_grid_uniform_spacing_m = _context_value(context, "fluid_grid_uniform_spacing_m")
    fsi_coupling_adaptive_iterations_cfl_threshold = _context_value(context, "fsi_coupling_adaptive_iterations_cfl_threshold")
    fsi_coupling_adaptive_iterations_max = _context_value(context, "fsi_coupling_adaptive_iterations_max")
    fsi_coupling_adaptive_iterations_residual_threshold_n = _context_value(context, "fsi_coupling_adaptive_iterations_residual_threshold_n")
    fsi_coupling_iterations = _context_value(context, "fsi_coupling_iterations")
    fsi_coupling_mode = _context_value(context, "fsi_coupling_mode")
    fsi_coupling_mode_report = _context_value(context, "fsi_coupling_mode_report")
    fsi_coupling_residual_continuation_iterations_max = _context_value(context, "fsi_coupling_residual_continuation_iterations_max")
    fsi_coupling_residual_continuation_threshold_n = _context_value(context, "fsi_coupling_residual_continuation_threshold_n")
    fsi_coupling_same_step_rerun_iterations_max = _context_value(context, "fsi_coupling_same_step_rerun_iterations_max")
    fsi_coupling_same_step_rerun_residual_threshold_n = _context_value(context, "fsi_coupling_same_step_rerun_residual_threshold_n")
    fsi_coupling_solver = _context_value(context, "fsi_coupling_solver")
    fsi_coupling_target_map_relaxation = _context_value(context, "fsi_coupling_target_map_relaxation")
    fsi_coupling_tolerance_n = _context_value(context, "fsi_coupling_tolerance_n")
    fsi_marker_coupling_tolerance_mps = _context_value(context, "fsi_marker_coupling_tolerance_mps")
    fsi_stabilization_effective_parameters = _context_value(context, "fsi_stabilization_effective_parameters")
    fsi_stabilization_preset = _context_value(context, "fsi_stabilization_preset")
    full_pressure_waveform_steps = _context_value(context, "full_pressure_waveform_steps")
    history_path = _context_value(context, "history_path")
    initial_fluid_obstacle_mode = _context_value(context, "initial_fluid_obstacle_mode")
    json = _context_value(context, "json")
    material = _context_value(context, "material")
    math = _context_value(context, "math")
    membrane_thickness_scale = _context_value(context, "membrane_thickness_scale")
    os = _context_value(context, "os")
    outlet_to_fsi_volume_source_gate_scope = _context_value(context, "outlet_to_fsi_volume_source_gate_scope")
    output_dir = _context_value(context, "output_dir")
    partial_run_reason = _context_value(context, "partial_run_reason")
    partial_run_stopped = _context_value(context, "partial_run_stopped")
    physical_outlet_to_fsi_volume_source_passes = _context_value(context, "physical_outlet_to_fsi_volume_source_passes")
    pressure_boundary_mapping = _context_value(context, "pressure_boundary_mapping")
    pressure_closure_normal = _context_value(context, "pressure_closure_normal")
    pressure_far_side_normal_sign = _context_value(context, "pressure_far_side_normal_sign")
    pressure_load_direction = _context_value(context, "pressure_load_direction")
    pressure_load_region_id = _context_value(context, "pressure_load_region_id")
    pressure_load_source_region_id = _context_value(context, "pressure_load_source_region_id")
    pressure_outlet_boundary_report = _context_value(context, "pressure_outlet_boundary_report")
    pressure_outlet_zmin_enabled = _context_value(context, "pressure_outlet_zmin_enabled")
    pressure_schedule_applied_in_history = _context_value(context, "pressure_schedule_applied_in_history")
    pressure_schedule_input = _context_value(context, "pressure_schedule_input")
    pressure_solver_name = _context_value(context, "pressure_solver_name")
    primary_shell_region_id = _context_value(context, "primary_shell_region_id")
    process_path = _context_value(context, "process_path")
    real_cad_step_binding = _context_value(context, "real_cad_step_binding")
    reduced_water_geometry_report = _context_value(context, "reduced_water_geometry_report")
    region14_aperture_carve_enabled = _context_value(context, "region14_aperture_carve_enabled")
    region14_aperture_carve_source = _context_value(context, "region14_aperture_carve_source")
    region14_aperture_geometry = _context_value(context, "region14_aperture_geometry")
    rows = _context_value(context, "rows")
    run_process_completion_status = _context_value(context, "run_process_completion_status")
    secondary_shell_region_id = _context_value(context, "secondary_shell_region_id")
    signed_positive_source_flux_ratio = _context_value(context, "signed_positive_source_flux_ratio")
    simulator = _context_value(context, "simulator")
    solid_density_scale = _context_value(context, "solid_density_scale")
    solid_mpm_bounds_max_m = _context_value(context, "solid_mpm_bounds_max_m")
    solid_mpm_bounds_min_m = _context_value(context, "solid_mpm_bounds_min_m")
    solid_mpm_bounds_padding_m = _context_value(context, "solid_mpm_bounds_padding_m")
    solid_mpm_force_nonzero_when_pressure_loaded = _context_value(context, "solid_mpm_force_nonzero_when_pressure_loaded")
    solid_mpm_substeps = _context_value(context, "solid_mpm_substeps")
    solid_sub_dt_s = _context_value(context, "solid_sub_dt_s")
    solid_substep_velocity_damping = _context_value(context, "solid_substep_velocity_damping")
    source_config_fluid_active_mask_requested = _context_value(context, "source_config_fluid_active_mask_requested")
    source_config_fluid_topology_report = _context_value(context, "source_config_fluid_topology_report")
    source_config_path = _context_value(context, "source_config_path")
    source_config_reduced_water_intersection_requested = _context_value(context, "source_config_reduced_water_intersection_requested")
    source_config_region14_aperture_requested = _context_value(context, "source_config_region14_aperture_requested")
    spec = _context_value(context, "spec")
    stable_solid_dt_s = _context_value(context, "stable_solid_dt_s")
    step_count = _context_value(context, "step_count")
    time = _context_value(context, "time")
    tri_metadata = _context_value(context, "tri_metadata")
    validation_scope_report = _context_value(context, "validation_scope_report")
    vector_norm = _context_value(context, "vector_norm")

    last = rows[-1] if rows else {}
    max_cfl = max(float(row["cfl"]) for row in rows) if rows else 0.0
    max_fluid_substeps = (
        max(int(float(row.get("fluid_substeps", effective_fluid_substeps))) for row in rows)
        if rows
        else effective_fluid_substeps
    )
    max_div_l2 = max(float(row["divergence_l2"]) for row in rows) if rows else 0.0
    max_interior_div_l2 = (
        max(float(row["interior_divergence_l2"]) for row in rows) if rows else 0.0
    )
    max_no_slip_l2 = (
        max(float(row["hibm_no_slip_residual_l2_mps"]) for row in rows)
        if rows
        else 0.0
    )
    max_no_slip_max = (
        max(float(row["hibm_no_slip_residual_max_mps"]) for row in rows)
        if rows
        else 0.0
    )
    max_no_slip_valid_marker_count = (
        max(
            int(row.get("hibm_no_slip_residual_valid_marker_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_no_slip_invalid_marker_count = (
        max(
            int(row.get("hibm_no_slip_residual_invalid_marker_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    post_solid_no_slip_residual_required = (
        any(
            _row_bool(
                row.get("hibm_post_solid_kinematic_projection_applied", False)
            )
            for row in rows
        )
        if rows
        else False
    )
    max_post_solid_no_slip_valid_marker_count = (
        max(
            int(
                row.get(
                    "hibm_post_solid_no_slip_residual_valid_marker_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_post_solid_no_slip_invalid_marker_count = (
        max(
            int(
                row.get(
                    "hibm_post_solid_no_slip_residual_invalid_marker_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_residual_norm_mps = (
        max(
            float(row.get("fsi_coupling_residual_norm_mps", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_fsi_coupling_residual_max_mps = (
        max(
            float(row.get("fsi_coupling_residual_max_mps", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_ib_node_count = (
        max(int(row["hibm_ib_node_count"]) for row in rows) if rows else 0
    )
    max_ib_invalid_count = (
        max(int(row["hibm_ib_invalid_projection_count"]) for row in rows)
        if rows
        else 0
    )
    max_hibm_internal_obstacle_cell_count = (
        max(int(row["hibm_internal_obstacle_cell_count"]) for row in rows)
        if rows
        else 0
    )
    max_hibm_solid_band_nonprojectable_cell_count = (
        max(int(row["hibm_solid_band_nonprojectable_cell_count"]) for row in rows)
        if rows
        else 0
    )
    max_hibm_row_cloud_orphan_cell_count = _rows_max_int(
        rows,
        "hibm_row_cloud_orphan_cell_count",
    )
    max_hibm_row_cloud_orphan_component_count = _rows_max_int(
        rows,
        "hibm_row_cloud_orphan_component_count",
    )
    max_hibm_overflow_singleton_cleanup_cell_count = _rows_max_int(
        rows,
        "hibm_overflow_singleton_cleanup_cell_count",
    )
    max_hibm_overflow_singleton_cleanup_component_count = _rows_max_int(
        rows,
        "hibm_overflow_singleton_cleanup_component_count",
    )
    max_hibm_pressure_disconnected_nonprojectable_cell_count = (
        max(
            int(row["hibm_pressure_disconnected_nonprojectable_cell_count"])
            for row in rows
        )
        if rows
        else 0
    )
    max_hibm_pressure_disconnected_component_count = _rows_max_int(
        rows,
        "hibm_pressure_disconnected_component_count",
    )
    max_hibm_pressure_disconnected_component_raw_count = _rows_max_int(
        rows,
        "hibm_pressure_disconnected_component_raw_count",
    )
    max_hibm_pressure_disconnected_largest_component_cell_count = (
        _rows_max_int(
            rows,
            "hibm_pressure_disconnected_largest_component_cell_count",
        )
    )
    max_hibm_pressure_disconnected_singleton_component_count = _rows_max_int(
        rows,
        "hibm_pressure_disconnected_singleton_component_count",
    )
    max_hibm_pressure_disconnected_small_component_count = _rows_max_int(
        rows,
        "hibm_pressure_disconnected_small_component_count",
    )
    max_hibm_pressure_disconnected_small_component_cell_count = _rows_max_int(
        rows,
        "hibm_pressure_disconnected_small_component_cell_count",
    )
    hibm_pressure_disconnected_component_overflow_seen = _rows_any_bool(
        rows,
        "hibm_pressure_disconnected_component_overflow",
    )
    max_hibm_next_row_cloud_orphan_cell_count = _rows_max_int(
        rows,
        "hibm_next_row_cloud_orphan_cell_count",
    )
    max_hibm_next_row_cloud_orphan_component_count = _rows_max_int(
        rows,
        "hibm_next_row_cloud_orphan_component_count",
    )
    max_hibm_next_overflow_singleton_cleanup_cell_count = _rows_max_int(
        rows,
        "hibm_next_overflow_singleton_cleanup_cell_count",
    )
    max_hibm_next_overflow_singleton_cleanup_component_count = _rows_max_int(
        rows,
        "hibm_next_overflow_singleton_cleanup_component_count",
    )
    max_hibm_next_pressure_disconnected_nonprojectable_cell_count = (
        _rows_max_int(
            rows,
            "hibm_next_pressure_disconnected_nonprojectable_cell_count",
        )
    )
    max_hibm_next_pressure_disconnected_component_count = _rows_max_int(
        rows,
        "hibm_next_pressure_disconnected_component_count",
    )
    max_hibm_next_pressure_disconnected_component_raw_count = _rows_max_int(
        rows,
        "hibm_next_pressure_disconnected_component_raw_count",
    )
    max_hibm_next_pressure_disconnected_largest_component_cell_count = (
        _rows_max_int(
            rows,
            "hibm_next_pressure_disconnected_largest_component_cell_count",
        )
    )
    max_hibm_next_pressure_disconnected_singleton_component_count = (
        _rows_max_int(
            rows,
            "hibm_next_pressure_disconnected_singleton_component_count",
        )
    )
    max_hibm_next_pressure_disconnected_small_component_count = _rows_max_int(
        rows,
        "hibm_next_pressure_disconnected_small_component_count",
    )
    max_hibm_next_pressure_disconnected_small_component_cell_count = (
        _rows_max_int(
            rows,
            "hibm_next_pressure_disconnected_small_component_cell_count",
        )
    )
    hibm_next_pressure_disconnected_component_overflow_seen = _rows_any_bool(
        rows,
        "hibm_next_pressure_disconnected_component_overflow",
    )
    max_hibm_air_backed_reachability_barrier_cell_count = (
        max(
            int(row["hibm_air_backed_reachability_barrier_cell_count"])
            for row in rows
        )
        if rows
        else 0
    )
    max_hibm_air_backed_seed_fallback_cell_count = (
        max(
            int(row["hibm_air_backed_seed_fallback_cell_count"])
            for row in rows
        )
        if rows
        else 0
    )
    max_full_stress_invalid_count = (
        max(int(row["hibm_full_stress_invalid_marker_count"]) for row in rows)
        if rows
        else 0
    )
    max_velocity_dirichlet_invalid_count = (
        max(
            int(row["hibm_velocity_dirichlet_invalid_reconstruction_count"])
            for row in rows
        )
        if rows
        else 0
    )
    max_pressure_neumann_invalid_count = (
        max(int(row["hibm_pressure_neumann_invalid_reconstruction_count"]) for row in rows)
        if rows
        else 0
    )
    max_pressure_neumann_invalid_unreconstructable_count = (
        max(
            int(
                row.get(
                    "hibm_pressure_neumann_invalid_unreconstructable_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_pressure_neumann_skipped_velocity_dirichlet_count = (
        max(
            int(
                row.get(
                    "hibm_pressure_neumann_skipped_velocity_dirichlet_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_pressure_neumann_skipped_pressure_boundary_adjacent_count = (
        max(
            int(
                row.get(
                    "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_marker_force_n = (
        max(
            vector_norm(
                (
                    float(row["hibm_marker_total_force_x_n"]),
                    float(row["hibm_marker_total_force_y_n"]),
                    float(row["hibm_marker_total_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_marker_count = (
        max(int(row["hibm_marker_total_count"]) for row in rows) if rows else 0
    )
    max_main_marker_count = (
        max(int(row["hibm_marker_primary_count"]) for row in rows) if rows else 0
    )
    max_tail_marker_count = (
        max(int(row["hibm_marker_secondary_count"]) for row in rows) if rows else 0
    )
    max_primary_stress_valid_count = (
        max(
            int(row.get("hibm_marker_primary_stress_valid_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_primary_stress_invalid_count = (
        max(
            int(row.get("hibm_marker_primary_stress_invalid_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_secondary_stress_valid_count = (
        max(
            int(row.get("hibm_marker_secondary_stress_valid_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_secondary_stress_invalid_count = (
        max(
            int(row.get("hibm_marker_secondary_stress_invalid_count", 0) or 0)
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_stress_invalid_count = max(
        max_primary_stress_invalid_count,
        max_secondary_stress_invalid_count,
    )
    max_tail_marker_force_n = (
        max(
            vector_norm(
                (
                    float(row["tail_fsi_fluid_force_x_n"]),
                    float(row["tail_fsi_fluid_force_y_n"]),
                    float(row["tail_fsi_fluid_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_fsi_action_reaction_residual_n = (
        max(float(row["fsi_action_reaction_residual_abs_n"]) for row in rows)
        if rows
        else 0.0
    )
    max_scatter_action_reaction_residual_n = (
        max(
            float(row["hibm_mpm_scatter_action_reaction_residual_n"])
            for row in rows
        )
        if rows
        else 0.0
    )
    max_solid_mpm_total_force_n = (
        max(
            vector_norm(
                (
                    float(row["solid_mpm_total_force_x_n"]),
                    float(row["solid_mpm_total_force_y_n"]),
                    float(row["solid_mpm_total_force_z_n"]),
                )
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_solid_mpm_grid_out_of_bounds_particle_count = (
        max(int(row["solid_mpm_grid_out_of_bounds_particle_count"]) for row in rows)
        if rows
        else 0
    )
    max_interface_reaction_relaxation_effective = (
        max(float(row.get("interface_reaction_relaxation_effective", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    min_interface_reaction_relaxation_effective = (
        min(float(row.get("interface_reaction_relaxation_effective", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_iterations_used = (
        max(int(row.get("fsi_coupling_iterations_used", 0) or 0) for row in rows)
        if rows
        else 0
    )
    max_fsi_coupling_iqn_ils_least_squares_update_count = (
        max(
            int(
                row.get("fsi_coupling_iqn_ils_least_squares_update_count", 0)
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_interface_map_amplification = (
        max(float(row.get("fsi_coupling_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_physical_interface_map_amplification = (
        max(float(row.get("fsi_coupling_physical_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_physical_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_physical_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_raw_interface_map_amplification = (
        max(float(row.get("fsi_coupling_raw_interface_map_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_raw_residual_jacobian_amplification = (
        max(float(row.get("fsi_coupling_raw_residual_jacobian_amplification", 0.0) or 0.0) for row in rows)
        if rows
        else 0.0
    )
    max_fsi_coupling_interface_map_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_interface_map_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_residual_jacobian_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_residual_jacobian_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_physical_interface_map_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_physical_interface_map_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_physical_residual_jacobian_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_physical_residual_jacobian_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_raw_interface_map_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_raw_interface_map_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_fsi_coupling_raw_residual_jacobian_amplification_sample_count = (
        max(
            int(
                row.get(
                    "fsi_coupling_raw_residual_jacobian_amplification_sample_count",
                    0,
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_outlet_negative_z = (
        max(float(row["outlet_flow_negative_z_m3s"]) for row in rows)
        if rows
        else 0.0
    )
    final_outlet_negative_z = _final_row_number(
        last,
        "outlet_flow_negative_z_m3s",
    )
    final_all_sections_negative_z = bool(
        last
        and _final_row_number(last, "lip_flow_negative_z_m3s") > 0.0
        and _final_row_number(last, "outlet_flow_negative_z_m3s") > 0.0
        and _final_row_number(last, "downstream_flow_negative_z_m3s") > 0.0
    )
    final_jet_sections_negative_z = bool(
        last
        and _final_row_number(last, "outlet_flow_negative_z_m3s") > 0.0
        and _final_row_number(last, "downstream_flow_negative_z_m3s") > 0.0
    )
    pressure_projection_cg_converged_all = (
        all(_row_bool(row.get("pressure_projection_cg_converged_all", True)) for row in rows)
        if rows
        else False
    )
    pressure_projection_cg_breakdown_count = (
        sum(int(row.get("pressure_projection_cg_breakdown_count", 0) or 0) for row in rows)
        if rows
        else 0
    )
    pressure_projection_physical_failure_count = (
        sum(
            1
            for row in rows
            if _row_bool(
                row.get("pressure_projection_physical_failure", False)
            )
        )
        if rows
        else 0
    )
    max_hibm_unreached_incompatible_component_count = (
        max(
            int(
                row.get("hibm_unreached_incompatible_component_count", 0)
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    max_hibm_unreached_component_raw_count = _rows_max_int(
        rows,
        "hibm_unreached_component_raw_count",
    )
    max_hibm_unreached_largest_component_cell_count = _rows_max_int(
        rows,
        "hibm_unreached_largest_component_cell_count",
    )
    max_hibm_unreached_singleton_component_count = _rows_max_int(
        rows,
        "hibm_unreached_singleton_component_count",
    )
    max_hibm_unreached_small_component_count = _rows_max_int(
        rows,
        "hibm_unreached_small_component_count",
    )
    max_hibm_unreached_small_component_cell_count = _rows_max_int(
        rows,
        "hibm_unreached_small_component_cell_count",
    )
    max_hibm_projection_overflow_singleton_cleanup_cell_count = (
        _rows_max_int(
            rows,
            "hibm_projection_overflow_singleton_cleanup_cell_count",
        )
    )
    max_hibm_projection_overflow_singleton_cleanup_component_count = (
        _rows_max_int(
            rows,
            "hibm_projection_overflow_singleton_cleanup_component_count",
        )
    )
    max_hibm_projection_tiny_unreached_cleanup_cell_count = _rows_max_int(
        rows,
        "hibm_projection_tiny_unreached_cleanup_cell_count",
    )
    max_hibm_projection_tiny_unreached_cleanup_component_count = (
        _rows_max_int(
            rows,
            "hibm_projection_tiny_unreached_cleanup_component_count",
        )
    )
    max_hibm_unreached_component_rhs_mean_max_abs = (
        max(
            float(row.get("hibm_unreached_component_rhs_mean_max_abs", 0.0) or 0.0)
            for row in rows
        )
        if rows
        else 0.0
    )
    max_hibm_unreached_component_rhs_integral_max_abs = (
        max(
            float(
                row.get(
                    "hibm_unreached_component_rhs_integral_max_abs",
                    0.0,
                )
                or 0.0
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    total_pressure_projection_cg_converged_all = (
        all(
            _row_bool(
                row.get(
                    "total_pressure_projection_cg_converged_all",
                    row.get("pressure_projection_cg_converged_all", True),
                )
            )
            for row in rows
        )
        if rows
        else False
    )
    total_pressure_projection_cg_breakdown_count = (
        sum(
            int(
                row.get(
                    "total_pressure_projection_cg_breakdown_count",
                    row.get("pressure_projection_cg_breakdown_count", 0),
                )
                or 0
            )
            for row in rows
        )
        if rows
        else 0
    )
    pre_projection_divergence_measured_all = (
        all(
            _row_bool(row.get("pre_projection_divergence_measured", False))
            for row in rows
        )
        if rows
        else False
    )
    pre_projection_divergence_sources = sorted(
        {
            str(row.get("pre_projection_divergence_source", ""))
            for row in rows
            if str(row.get("pre_projection_divergence_source", ""))
        }
    )
    max_projection_to_pre_divergence_l2_ratio = (
        max(
            _required_finite_row_number(
                row,
                "projection_to_pre_divergence_l2_ratio",
                context="summary row",
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_post_boundary_to_pre_divergence_l2_ratio = (
        max(
            _required_finite_row_number(
                row,
                "post_boundary_to_pre_divergence_l2_ratio",
                context="summary row",
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    max_post_constraint_to_pre_divergence_l2_ratio = (
        max(
            _required_finite_row_number(
                row,
                "post_constraint_to_pre_divergence_l2_ratio",
                context="summary row",
            )
            for row in rows
        )
        if rows
        else 0.0
    )
    required_row_fields = finite_required_row_fields_for_mode(
        fsi_coupling_mode,
        solid_model=args.solid_model,
    )
    nonfinite_diagnostics = finite_field_diagnostics(rows, required_row_fields)
    validation_scope = validation_scope_report(
        requested_steps=step_count,
        completed_steps=len(rows),
        full_pressure_waveform_steps=full_pressure_waveform_steps,
        partial_run_stopped=partial_run_stopped,
        partial_run_reason=partial_run_reason,
    )
    validation_scope_complete = bool(validation_scope["validation_scope_complete"])
    final_fsi_volume_source_m3s = _final_row_number(
        last,
        "fsi_volume_source_m3s",
    )
    final_pressure_outlet_positive_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_positive_source_volume_flux_m3s",
    )
    final_pressure_outlet_abs_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_abs_source_volume_flux_m3s",
    )
    final_pressure_outlet_reachable_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_reachable_source_volume_flux_m3s",
    )
    final_pressure_outlet_unreached_source_volume_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_unreached_source_volume_flux_m3s",
    )
    final_pressure_outlet_reachable_source_cell_count = _final_row_number(
        last,
        "pressure_outlet_reachable_source_cell_count",
    )
    final_pressure_outlet_unreached_source_cell_count = _final_row_number(
        last,
        "pressure_outlet_unreached_source_cell_count",
    )
    final_pressure_outlet_unreached_source_abs_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_unreached_source_abs_flux_m3s",
    )
    final_pressure_outlet_unreached_source_centroid_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_x_m",
    )
    final_pressure_outlet_unreached_source_centroid_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_y_m",
    )
    final_pressure_outlet_unreached_source_centroid_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_centroid_z_m",
    )
    final_pressure_outlet_unreached_source_min_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_x_m",
    )
    final_pressure_outlet_unreached_source_min_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_y_m",
    )
    final_pressure_outlet_unreached_source_min_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_min_z_m",
    )
    final_pressure_outlet_unreached_source_max_x_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_x_m",
    )
    final_pressure_outlet_unreached_source_max_y_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_y_m",
    )
    final_pressure_outlet_unreached_source_max_z_m = _final_row_number_or_none(
        last,
        "pressure_outlet_unreached_source_max_z_m",
    )
    final_pressure_outlet_velocity_to_positive_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_positive_source_ratio",
    )
    final_pressure_outlet_velocity_to_abs_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_abs_source_ratio",
    )
    final_outlet_to_fsi_volume_source_ratio = signed_positive_source_flux_ratio(
        outlet_negative_z_flux_m3s=final_outlet_negative_z,
        source_flux_m3s=final_fsi_volume_source_m3s,
    )
    max_outlet_to_fsi_volume_source_ratio = (
        max(float(row["main_volume_flux_to_outlet_ratio"]) for row in rows)
        if rows
        else 0.0
    )
    outlet_to_fsi_gate_scope = outlet_to_fsi_volume_source_gate_scope(
        fluid_grid_resolution=fluid_grid_resolution,
        validation_scope_complete=validation_scope_complete,
    )
    final_outlet_to_fsi_volume_source_ratio_physical = (
        physical_outlet_to_fsi_volume_source_passes(
            outlet_negative_z_flux_m3s=final_outlet_negative_z,
            fsi_volume_source_m3s=final_fsi_volume_source_m3s,
            min_ratio=float(args.min_outlet_to_main_volume_flux_ratio),
        )
    )
    timing_fields = (
        "step_wall_time_s",
        "fsi_coupling_wall_time_s",
        "solid_advance_wall_time_s",
        "fluid_advance_wall_time_s",
        "sample_wall_time_s",
        "surface_diagnostics_wall_time_s",
        "checkpoint_wall_time_s",
    )
    timing_summary = {}
    for field in timing_fields:
        values = [
            float(row.get(field, 0.0) or 0.0)
            for row in rows
            if math.isfinite(float(row.get(field, 0.0) or 0.0))
        ]
        timing_summary[f"max_{field}"] = max(values) if values else 0.0
        timing_summary[f"mean_{field}"] = (
            sum(values) / float(len(values)) if values else 0.0
        )
    solid_mpm_force_required = bool(rows)
    checks = {
        "pressure_schedule_applied": pressure_schedule_applied_in_history(rows),
        "hibm_ib_nodes_present": max_ib_node_count > 0,
        "hibm_internal_obstacle_cells_present": (
            max_hibm_internal_obstacle_cell_count > 0
            or max_hibm_solid_band_nonprojectable_cell_count > 0
            or max_hibm_pressure_disconnected_nonprojectable_cell_count > 0
        ),
        "hibm_velocity_dirichlet_rows_present": bool(
            last and int(last["hibm_velocity_dirichlet_active_rows"]) > 0
        ),
        "hibm_pressure_neumann_rows_present": bool(
            last
            and (
                int(last["hibm_pressure_neumann_active_rows"]) > 0
                or int(
                    last.get(
                        "hibm_pressure_neumann_skipped_velocity_dirichlet_count",
                        0,
                    )
                    or 0
                )
                > 0
            )
        ),
        "hibm_velocity_dirichlet_reconstruction_valid": (
            max_velocity_dirichlet_invalid_count == 0
        ),
        "hibm_pressure_neumann_reconstruction_valid": (
            max_pressure_neumann_invalid_count == 0
        ),
        "hibm_no_slip_residual_samples_present": (
            max_no_slip_valid_marker_count > 0
        ),
        "hibm_no_slip_residual_all_markers_measured": (
            post_solid_no_slip_residual_required
            or max_no_slip_invalid_marker_count == 0
        ),
        "hibm_post_solid_no_slip_residual_samples_present": (
            (not post_solid_no_slip_residual_required)
            or max_post_solid_no_slip_valid_marker_count > 0
        ),
        "hibm_post_solid_no_slip_residual_all_markers_measured": (
            (not post_solid_no_slip_residual_required)
            or max_post_solid_no_slip_invalid_marker_count == 0
        ),
        "hibm_full_stress_marker_samples_present": bool(
            last and int(last["hibm_full_stress_valid_marker_count"]) > 0
        ),
        "hibm_marker_force_scattered_to_mpm": max_marker_force_n > 0.0,
        "tail_markers_present": max_tail_marker_count > 0,
        "tail_marker_participates": (
            max_tail_marker_count > 0 and max_tail_marker_force_n > 0.0
        ),
        "solid_mpm_force_nonzero_when_pressure_loaded": solid_mpm_force_nonzero_when_pressure_loaded(
            rows,
            force_required=solid_mpm_force_required,
        ),
        "fsi_coupling_explicit_steps_completed": all(
            bool(row.get("fsi_coupling_step_completed", False)) for row in rows
        )
        if rows
        else False,
        "fsi_coupling_convergence_not_claimed": all(
            (
                bool(row.get("fsi_coupling_explicit_single_pass", False))
                and not bool(row.get("fsi_coupling_convergence_measured", False))
                and not bool(row.get("fsi_coupling_converged", False))
            )
            or (
                not bool(row.get("fsi_coupling_explicit_single_pass", False))
                and bool(row.get("fsi_coupling_convergence_measured", False))
                and bool(row.get("fsi_coupling_converged", False))
                and str(row.get("fsi_coupling_residual_units", "")) == "m/s"
                and str(row.get("fsi_coupling_residual_source", ""))
                == "marker_surface_fixed_point_velocity_residual_l2_mps"
            )
            for row in rows
        )
        if rows
        else False,
        "finite_primary_diagnostics": len(nonfinite_diagnostics) == 0,
        "negative_z_outlet_flow_present": max_outlet_negative_z > 0.0,
        "final_negative_z_outlet_flow": final_outlet_negative_z > 0.0,
        "final_negative_z_jet_sections": final_jet_sections_negative_z,
        "section_samples_present": bool(
            rows
            and int(last["lip_sample_count"]) > 0
            and int(last["outlet_sample_count"]) > 0
            and int(last["downstream_sample_count"]) > 0
        ),
        "cfl_below_0p5": max_cfl < 0.5,
        "projection_divergence_finite": math.isfinite(max_div_l2),
        "projection_divergence_below_tolerance": (
            math.isfinite(max_interior_div_l2)
            and max_interior_div_l2 <= float(args.projection_divergence_tolerance)
        ),
        "pre_projection_divergence_measured": pre_projection_divergence_measured_all,
        "pressure_projection_cg_converged_all": pressure_projection_cg_converged_all,
        "pressure_projection_cg_no_breakdown": (
            pressure_projection_cg_breakdown_count == 0
        ),
        "pressure_projection_no_physical_failure": (
            pressure_projection_physical_failure_count == 0
        ),
        "total_pressure_projection_cg_converged_all": (
            total_pressure_projection_cg_converged_all
        ),
        "total_pressure_projection_cg_no_breakdown": (
            total_pressure_projection_cg_breakdown_count == 0
        ),
    }
    diagnostic_checks = {
        "hibm_invalid_projection_count_zero": max_ib_invalid_count == 0,
        "hibm_full_stress_invalid_marker_count_zero": (
            max_full_stress_invalid_count == 0
        ),
        "hibm_fsi_full_stress_invalid_marker_count_zero": (
            max_fsi_stress_invalid_count == 0
        ),
        "hibm_action_reaction_residual_bounded": math.isfinite(
            max_fsi_action_reaction_residual_n
        ),
        "hibm_scatter_action_reaction_residual_bounded": math.isfinite(
            max_scatter_action_reaction_residual_n
        ),
        "solid_mpm_grid_out_of_bounds_particle_count_zero": (
            max_solid_mpm_grid_out_of_bounds_particle_count == 0
        ),
        "projection_divergence_not_increased": (
            max_projection_to_pre_divergence_l2_ratio <= 1.0 + 1.0e-12
        ),
        "post_boundary_divergence_not_increased": (
            max_post_boundary_to_pre_divergence_l2_ratio <= 1.0 + 1.0e-12
        ),
        "post_constraint_divergence_not_increased": (
            max_post_constraint_to_pre_divergence_l2_ratio <= 1.0 + 1.0e-12
        ),
        "sharp_runner_skips_legacy_projected_force_report": True,
    }
    if bool(outlet_to_fsi_gate_scope["hard_gate"]):
        checks["final_outlet_to_fsi_volume_source_ratio_physical"] = (
            final_outlet_to_fsi_volume_source_ratio_physical
        )
    else:
        diagnostic_checks["final_outlet_to_fsi_volume_source_ratio_physical"] = (
            final_outlet_to_fsi_volume_source_ratio_physical
        )
    completed_step_checks_passed = checks_passed(checks)
    validation_passed = completed_step_checks_passed if validation_scope_complete else None
    summary = {
        "case": "Squid soft robot",
        "model_class": "sharp-interface HIBM-MPM case runner",
        "uses_generic_simulation_core": True,
        "source_config_used_as_input_only": str(source_config_path),
        "cad_provenance": cad_provenance,
        "real_cad_step_path": cad_provenance.get("cad_step_path"),
        "real_cad_step_direct_binding": bool(
            cad_provenance.get("direct_cad_step_binding", False)
        ),
        "real_cad_step_derived_surface_mesh_binding": bool(
            cad_provenance.get("step_derived_surface_mesh_binding", False)
        ),
        "real_cad_step_binding": real_cad_step_binding,
        "pressure_schedule_input": pressure_schedule_input,
        "pressure_boundary_shell_mapping": asdict(pressure_boundary_mapping),
        "pressure_load_source_region_id": int(pressure_load_source_region_id),
        "pressure_load_region_id": int(pressure_load_region_id),
        "pressure_load_direction": tuple(float(v) for v in pressure_load_direction),
        "pressure_closure_normal": tuple(float(v) for v in pressure_closure_normal),
        "pressure_far_side_normal_sign": float(pressure_far_side_normal_sign),
        "pressure_outlet_boundary": pressure_outlet_boundary_report,
        "pressure_outlet_zmin_enabled": pressure_outlet_zmin_enabled,
        "shell_primary_region_id": int(primary_shell_region_id),
        "shell_secondary_region_id": int(secondary_shell_region_id),
        "source_config_fluid_active_mask_requested": (
            source_config_fluid_active_mask_requested
        ),
        "source_config_reduced_water_intersection_requested": (
            source_config_reduced_water_intersection_requested
        ),
        "initial_fluid_obstacle_mode": initial_fluid_obstacle_mode,
        "source_config_fluid_topology": source_config_fluid_topology_report,
        "source_config_region14_aperture_requested": (
            source_config_region14_aperture_requested
        ),
        "region14_aperture_carve_enabled": region14_aperture_carve_enabled,
        "region14_aperture_carve_source": region14_aperture_carve_source,
        "open_downstream_farfield_enabled": bool(
            spec.downstream_farfield_open_enabled
        ),
        "region14_aperture_geometry": region14_aperture_geometry,
        "reduced_water_geometry": reduced_water_geometry_report(spec),
        "tri_surface_diagnostics": tri_metadata,
        "solid_model": {
            "type": args.solid_model,
            "solid_particle_size_m": _final_row_number(
                last,
                "solid_mpm_particle_spacing_m",
            ),
            "solid_particle_count": _final_row_int(
                last,
                "solid_mpm_particle_count",
            ),
            "solid_mpm_layers": int(args.solid_mpm_layers),
            "solid_mpm_substeps": int(solid_mpm_substeps),
            "solid_mpm_sub_dt_s": float(solid_sub_dt_s),
            "solid_mpm_stable_dt_s": float(stable_solid_dt_s),
            "solid_mpm_bounds_min_m": [
                float(value) for value in solid_mpm_bounds_min_m
            ],
            "solid_mpm_bounds_max_m": [
                float(value) for value in solid_mpm_bounds_max_m
            ],
            "solid_mpm_bounds_padding_m": float(solid_mpm_bounds_padding_m),
            "solid_mpm_cfl": float(args.solid_mpm_cfl),
            "solid_mpm_velocity_damping": float(args.solid_mpm_velocity_damping),
            "solid_mpm_substep_velocity_damping": float(
                solid_substep_velocity_damping
            ),
            "membrane_thickness_scale": membrane_thickness_scale,
            "solid_density_scale": solid_density_scale,
            "solid_density_kgm3": float(material.density_kgm3),
            "main_membrane_thickness_m": float(spec.main_membrane_thickness_m),
            "tail_membrane_thickness_m": float(spec.tail_membrane_thickness_m),
            "is_physical_mpm": True,
        },
        "fluid_grid_spacing_m": (
            None
            if fluid_grid_uniform_spacing_m is None
            else [float(value) for value in fluid_grid_uniform_spacing_m]
        ),
        "fluid_grid_min_spacing_m": [
            float(value) for value in fluid_grid_axis_min_spacing_m
        ],
        "fluid_grid_max_spacing_m": [
            float(value) for value in fluid_grid_axis_max_spacing_m
        ],
        "fluid_grid_nodes": [
            int(simulator.fluid.nx),
            int(simulator.fluid.ny),
            int(simulator.fluid.nz),
        ],
        "fluid_grid_graded_enabled": spec.graded_grid is not None,
        "fluid_grid_resolution": fluid_grid_resolution,
        "history_csv": str(history_path),
        "completed_steps": len(rows),
        "requested_steps": step_count,
        "full_pressure_waveform_steps": full_pressure_waveform_steps,
        "validation_scope": validation_scope,
        "completed_step_checks": checks,
        "diagnostic_checks": diagnostic_checks,
        "completed_step_checks_passed": completed_step_checks_passed,
        "validation_passed": validation_passed,
        "nonfinite_diagnostics": nonfinite_diagnostics,
        "pre_projection_divergence_measured_all": (
            pre_projection_divergence_measured_all
        ),
        "pre_projection_divergence_sources": pre_projection_divergence_sources,
        "timing_summary": timing_summary,
        "fsi_coupling_mode": fsi_coupling_mode,
        "fsi_coupling_mode_report": fsi_coupling_mode_report,
        "fsi_stabilization_preset": fsi_stabilization_preset,
        "fsi_stabilization_preset_conflict_policy": (
            FSI_STABILIZATION_PRESET_CONFLICT_POLICY
        ),
        "fsi_stabilization_effective_parameters": (
            fsi_stabilization_effective_parameters
        ),
        "fsi_coupling_iterations_requested": fsi_coupling_iterations,
        "fsi_coupling_adaptive_iterations_max": (
            fsi_coupling_adaptive_iterations_max
        ),
        "fsi_coupling_adaptive_iterations_residual_threshold_n": (
            fsi_coupling_adaptive_iterations_residual_threshold_n
        ),
        "fsi_coupling_adaptive_iterations_cfl_threshold": (
            fsi_coupling_adaptive_iterations_cfl_threshold
        ),
        "fsi_coupling_same_step_rerun_iterations_max": (
            fsi_coupling_same_step_rerun_iterations_max
        ),
        "fsi_coupling_same_step_rerun_residual_threshold_n": (
            fsi_coupling_same_step_rerun_residual_threshold_n
        ),
        "fsi_coupling_residual_continuation_iterations_max": (
            fsi_coupling_residual_continuation_iterations_max
        ),
        "fsi_coupling_residual_continuation_threshold_n": (
            fsi_coupling_residual_continuation_threshold_n
        ),
        "fsi_coupling_solver": fsi_coupling_solver,
        "max_fsi_coupling_iterations_used": max_fsi_coupling_iterations_used,
        "max_fsi_coupling_iqn_ils_least_squares_update_count": (
            max_fsi_coupling_iqn_ils_least_squares_update_count
        ),
        "fsi_coupling_tolerance_n": fsi_coupling_tolerance_n,
        "fsi_marker_coupling_tolerance_mps": (
            fsi_marker_coupling_tolerance_mps
        ),
        "fsi_coupling_target_map_relaxation": fsi_coupling_target_map_relaxation,
        "fsi_coupling_explicit_single_pass": bool(
            rows
            and all(
                bool(row.get("fsi_coupling_explicit_single_pass", False))
                for row in rows
            )
        ),
        "fsi_coupling_step_completed": bool(
            rows
            and all(
                bool(row.get("fsi_coupling_step_completed", False))
                for row in rows
            )
        ),
        "fsi_coupling_convergence_measured": bool(
            rows
            and any(
                bool(row.get("fsi_coupling_convergence_measured", False))
                for row in rows
            )
        ),
        "fsi_coupling_converged": bool(
            rows
            and any(
                bool(row.get("fsi_coupling_convergence_measured", False))
                for row in rows
            )
            and all(bool(row.get("fsi_coupling_converged", False)) for row in rows)
        ),
        "legacy_projected_reduced_coupling_used": False,
        "surface_fsi_force_spreading_enabled": True,
        "fluid_stress_action_on_fluid_enabled": True,
        "pressure_solver_requested": str(args.pressure_solver),
        "pressure_solver_resolved": pressure_solver_name,
        "pressure_solver_actual": str(last.get("pressure_solver_actual", "")),
        "pressure_solve_failure_policy": str(args.pressure_solve_failure_policy),
        "pressure_solver_forced_to_fv_cg_count": sum(
            1 for row in rows if bool(row.get("pressure_solver_forced_to_fv_cg", False))
        ),
        "pressure_solver_force_reason": str(
            last.get("pressure_solver_force_reason", "")
        ),
        "max_cfl": max_cfl,
        "fluid_substeps": effective_fluid_substeps,
        "fluid_substep_dt_s": effective_fluid_substep_dt_s,
        "adaptive_fluid_substeps_enabled": adaptive_fluid_substeps_enabled,
        "adaptive_fluid_substeps_target_cfl": float(
            args.adaptive_fluid_substeps_target_cfl
        ),
        "adaptive_fluid_substeps_max": int(args.adaptive_fluid_substeps_max),
        "adaptive_fluid_substeps_safety": float(
            args.adaptive_fluid_substeps_safety
        ),
        "max_fluid_substeps": max_fluid_substeps,
        "max_divergence_l2": max_div_l2,
        "max_interior_divergence_l2": max_interior_div_l2,
        "max_projection_to_pre_divergence_l2_ratio": (
            max_projection_to_pre_divergence_l2_ratio
        ),
        "max_post_boundary_to_pre_divergence_l2_ratio": (
            max_post_boundary_to_pre_divergence_l2_ratio
        ),
        "max_post_constraint_to_pre_divergence_l2_ratio": (
            max_post_constraint_to_pre_divergence_l2_ratio
        ),
        "pressure_projection_cg_converged_all": pressure_projection_cg_converged_all,
        "pressure_projection_cg_breakdown_count": pressure_projection_cg_breakdown_count,
        "pressure_projection_physical_failure": (
            pressure_projection_physical_failure_count > 0
        ),
        "pressure_projection_physical_failure_count": (
            pressure_projection_physical_failure_count
        ),
        "max_hibm_unreached_incompatible_component_count": (
            max_hibm_unreached_incompatible_component_count
        ),
        "max_hibm_unreached_component_raw_count": (
            max_hibm_unreached_component_raw_count
        ),
        "max_hibm_unreached_largest_component_cell_count": (
            max_hibm_unreached_largest_component_cell_count
        ),
        "max_hibm_unreached_singleton_component_count": (
            max_hibm_unreached_singleton_component_count
        ),
        "max_hibm_unreached_small_component_count": (
            max_hibm_unreached_small_component_count
        ),
        "max_hibm_unreached_small_component_cell_count": (
            max_hibm_unreached_small_component_cell_count
        ),
        "max_hibm_projection_overflow_singleton_cleanup_cell_count": (
            max_hibm_projection_overflow_singleton_cleanup_cell_count
        ),
        "max_hibm_projection_overflow_singleton_cleanup_component_count": (
            max_hibm_projection_overflow_singleton_cleanup_component_count
        ),
        "max_hibm_projection_tiny_unreached_cleanup_cell_count": (
            max_hibm_projection_tiny_unreached_cleanup_cell_count
        ),
        "max_hibm_projection_tiny_unreached_cleanup_component_count": (
            max_hibm_projection_tiny_unreached_cleanup_component_count
        ),
        "max_hibm_unreached_component_rhs_mean_max_abs": (
            max_hibm_unreached_component_rhs_mean_max_abs
        ),
        "max_hibm_unreached_component_rhs_integral_max_abs": (
            max_hibm_unreached_component_rhs_integral_max_abs
        ),
        "total_pressure_projection_cg_converged_all": total_pressure_projection_cg_converged_all,
        "total_pressure_projection_cg_breakdown_count": total_pressure_projection_cg_breakdown_count,
        "max_hibm_no_slip_residual_l2_mps": max_no_slip_l2,
        "max_hibm_no_slip_residual_mps": max_no_slip_max,
        "max_hibm_no_slip_residual_valid_marker_count": (
            max_no_slip_valid_marker_count
        ),
        "max_hibm_no_slip_residual_invalid_marker_count": (
            max_no_slip_invalid_marker_count
        ),
        "max_hibm_post_solid_no_slip_residual_valid_marker_count": (
            max_post_solid_no_slip_valid_marker_count
        ),
        "max_hibm_post_solid_no_slip_residual_invalid_marker_count": (
            max_post_solid_no_slip_invalid_marker_count
        ),
        "max_fsi_coupling_residual_norm_mps": (
            max_fsi_coupling_residual_norm_mps
        ),
        "max_fsi_coupling_residual_max_mps": (
            max_fsi_coupling_residual_max_mps
        ),
        "max_fsi_coupling_interface_map_amplification": (
            max_fsi_coupling_interface_map_amplification
        ),
        "max_fsi_coupling_residual_jacobian_amplification": (
            max_fsi_coupling_residual_jacobian_amplification
        ),
        "max_fsi_coupling_physical_interface_map_amplification": (
            max_fsi_coupling_physical_interface_map_amplification
        ),
        "max_fsi_coupling_physical_residual_jacobian_amplification": (
            max_fsi_coupling_physical_residual_jacobian_amplification
        ),
        "max_fsi_coupling_raw_interface_map_amplification": (
            max_fsi_coupling_raw_interface_map_amplification
        ),
        "max_fsi_coupling_raw_residual_jacobian_amplification": (
            max_fsi_coupling_raw_residual_jacobian_amplification
        ),
        "max_fsi_coupling_interface_map_amplification_sample_count": (
            max_fsi_coupling_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_residual_jacobian_amplification_sample_count
        ),
        "max_fsi_coupling_physical_interface_map_amplification_sample_count": (
            max_fsi_coupling_physical_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_physical_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_physical_residual_jacobian_amplification_sample_count
        ),
        "max_fsi_coupling_raw_interface_map_amplification_sample_count": (
            max_fsi_coupling_raw_interface_map_amplification_sample_count
        ),
        "max_fsi_coupling_raw_residual_jacobian_amplification_sample_count": (
            max_fsi_coupling_raw_residual_jacobian_amplification_sample_count
        ),
        "max_interface_reaction_relaxation_effective": (
            max_interface_reaction_relaxation_effective
        ),
        "min_interface_reaction_relaxation_effective": (
            min_interface_reaction_relaxation_effective
        ),
        "max_hibm_ib_node_count": max_ib_node_count,
        "max_hibm_ib_invalid_projection_count": max_ib_invalid_count,
        "max_hibm_internal_obstacle_cell_count": (
            max_hibm_internal_obstacle_cell_count
        ),
        "max_hibm_solid_band_nonprojectable_cell_count": (
            max_hibm_solid_band_nonprojectable_cell_count
        ),
        "max_hibm_row_cloud_orphan_cell_count": (
            max_hibm_row_cloud_orphan_cell_count
        ),
        "max_hibm_row_cloud_orphan_component_count": (
            max_hibm_row_cloud_orphan_component_count
        ),
        "max_hibm_overflow_singleton_cleanup_cell_count": (
            max_hibm_overflow_singleton_cleanup_cell_count
        ),
        "max_hibm_overflow_singleton_cleanup_component_count": (
            max_hibm_overflow_singleton_cleanup_component_count
        ),
        "max_hibm_pressure_disconnected_nonprojectable_cell_count": (
            max_hibm_pressure_disconnected_nonprojectable_cell_count
        ),
        "max_hibm_pressure_disconnected_component_count": (
            max_hibm_pressure_disconnected_component_count
        ),
        "max_hibm_pressure_disconnected_component_raw_count": (
            max_hibm_pressure_disconnected_component_raw_count
        ),
        "max_hibm_pressure_disconnected_largest_component_cell_count": (
            max_hibm_pressure_disconnected_largest_component_cell_count
        ),
        "max_hibm_pressure_disconnected_singleton_component_count": (
            max_hibm_pressure_disconnected_singleton_component_count
        ),
        "max_hibm_pressure_disconnected_small_component_count": (
            max_hibm_pressure_disconnected_small_component_count
        ),
        "max_hibm_pressure_disconnected_small_component_cell_count": (
            max_hibm_pressure_disconnected_small_component_cell_count
        ),
        "hibm_pressure_disconnected_component_overflow_seen": (
            hibm_pressure_disconnected_component_overflow_seen
        ),
        "max_hibm_next_row_cloud_orphan_cell_count": (
            max_hibm_next_row_cloud_orphan_cell_count
        ),
        "max_hibm_next_row_cloud_orphan_component_count": (
            max_hibm_next_row_cloud_orphan_component_count
        ),
        "max_hibm_next_overflow_singleton_cleanup_cell_count": (
            max_hibm_next_overflow_singleton_cleanup_cell_count
        ),
        "max_hibm_next_overflow_singleton_cleanup_component_count": (
            max_hibm_next_overflow_singleton_cleanup_component_count
        ),
        "max_hibm_next_pressure_disconnected_nonprojectable_cell_count": (
            max_hibm_next_pressure_disconnected_nonprojectable_cell_count
        ),
        "max_hibm_next_pressure_disconnected_component_count": (
            max_hibm_next_pressure_disconnected_component_count
        ),
        "max_hibm_next_pressure_disconnected_component_raw_count": (
            max_hibm_next_pressure_disconnected_component_raw_count
        ),
        "max_hibm_next_pressure_disconnected_largest_component_cell_count": (
            max_hibm_next_pressure_disconnected_largest_component_cell_count
        ),
        "max_hibm_next_pressure_disconnected_singleton_component_count": (
            max_hibm_next_pressure_disconnected_singleton_component_count
        ),
        "max_hibm_next_pressure_disconnected_small_component_count": (
            max_hibm_next_pressure_disconnected_small_component_count
        ),
        "max_hibm_next_pressure_disconnected_small_component_cell_count": (
            max_hibm_next_pressure_disconnected_small_component_cell_count
        ),
        "hibm_next_pressure_disconnected_component_overflow_seen": (
            hibm_next_pressure_disconnected_component_overflow_seen
        ),
        "max_hibm_velocity_dirichlet_invalid_reconstruction_count": (
            max_velocity_dirichlet_invalid_count
        ),
        "max_hibm_pressure_neumann_invalid_reconstruction_count": (
            max_pressure_neumann_invalid_count
        ),
        "max_hibm_pressure_neumann_invalid_unreconstructable_count": (
            max_pressure_neumann_invalid_unreconstructable_count
        ),
        "max_hibm_air_backed_reachability_barrier_cell_count": (
            max_hibm_air_backed_reachability_barrier_cell_count
        ),
        "max_hibm_air_backed_seed_fallback_cell_count": (
            max_hibm_air_backed_seed_fallback_cell_count
        ),
        "max_hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
            max_pressure_neumann_skipped_velocity_dirichlet_count
        ),
        "max_hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count": (
            max_pressure_neumann_skipped_pressure_boundary_adjacent_count
        ),
        "max_hibm_full_stress_invalid_marker_count": (
            max_full_stress_invalid_count
        ),
        "max_hibm_primary_stress_valid_marker_count": (
            max_primary_stress_valid_count
        ),
        "max_hibm_primary_stress_invalid_marker_count": (
            max_primary_stress_invalid_count
        ),
        "max_hibm_secondary_stress_valid_marker_count": (
            max_secondary_stress_valid_count
        ),
        "max_hibm_secondary_stress_invalid_marker_count": (
            max_secondary_stress_invalid_count
        ),
        "max_hibm_fsi_stress_invalid_marker_count": (
            max_fsi_stress_invalid_count
        ),
        "max_hibm_marker_count": max_marker_count,
        "max_main_marker_count": max_main_marker_count,
        "max_tail_marker_count": max_tail_marker_count,
        "max_hibm_marker_force_n": max_marker_force_n,
        "max_tail_marker_force_n": max_tail_marker_force_n,
        "max_fsi_action_reaction_residual_n": (
            max_fsi_action_reaction_residual_n
        ),
        "max_hibm_mpm_scatter_action_reaction_residual_n": (
            max_scatter_action_reaction_residual_n
        ),
        "max_solid_mpm_total_force_n": max_solid_mpm_total_force_n,
        "max_solid_mpm_grid_out_of_bounds_particle_count": (
            max_solid_mpm_grid_out_of_bounds_particle_count
        ),
        "max_outlet_negative_z_flow_m3s": max_outlet_negative_z,
        "final_outlet_negative_z_flow_m3s": final_outlet_negative_z,
        "final_fsi_volume_source_m3s": final_fsi_volume_source_m3s,
        "final_pressure_outlet_positive_source_volume_flux_m3s": (
            final_pressure_outlet_positive_source_volume_flux_m3s
        ),
        "final_pressure_outlet_abs_source_volume_flux_m3s": (
            final_pressure_outlet_abs_source_volume_flux_m3s
        ),
        "final_pressure_outlet_reachable_source_volume_flux_m3s": (
            final_pressure_outlet_reachable_source_volume_flux_m3s
        ),
        "final_pressure_outlet_unreached_source_volume_flux_m3s": (
            final_pressure_outlet_unreached_source_volume_flux_m3s
        ),
        "final_pressure_outlet_reachable_source_cell_count": (
            final_pressure_outlet_reachable_source_cell_count
        ),
        "final_pressure_outlet_unreached_source_cell_count": (
            final_pressure_outlet_unreached_source_cell_count
        ),
        "final_pressure_outlet_unreached_source_abs_flux_m3s": (
            final_pressure_outlet_unreached_source_abs_flux_m3s
        ),
        "final_pressure_outlet_unreached_source_centroid_x_m": (
            final_pressure_outlet_unreached_source_centroid_x_m
        ),
        "final_pressure_outlet_unreached_source_centroid_y_m": (
            final_pressure_outlet_unreached_source_centroid_y_m
        ),
        "final_pressure_outlet_unreached_source_centroid_z_m": (
            final_pressure_outlet_unreached_source_centroid_z_m
        ),
        "final_pressure_outlet_unreached_source_min_x_m": (
            final_pressure_outlet_unreached_source_min_x_m
        ),
        "final_pressure_outlet_unreached_source_min_y_m": (
            final_pressure_outlet_unreached_source_min_y_m
        ),
        "final_pressure_outlet_unreached_source_min_z_m": (
            final_pressure_outlet_unreached_source_min_z_m
        ),
        "final_pressure_outlet_unreached_source_max_x_m": (
            final_pressure_outlet_unreached_source_max_x_m
        ),
        "final_pressure_outlet_unreached_source_max_y_m": (
            final_pressure_outlet_unreached_source_max_y_m
        ),
        "final_pressure_outlet_unreached_source_max_z_m": (
            final_pressure_outlet_unreached_source_max_z_m
        ),
        "final_pressure_outlet_velocity_to_positive_source_ratio": (
            final_pressure_outlet_velocity_to_positive_source_ratio
        ),
        "final_pressure_outlet_velocity_to_abs_source_ratio": (
            final_pressure_outlet_velocity_to_abs_source_ratio
        ),
        "final_outlet_to_fsi_volume_source_ratio": (
            final_outlet_to_fsi_volume_source_ratio
        ),
        "final_outlet_to_fsi_volume_source_ratio_physical": (
            final_outlet_to_fsi_volume_source_ratio_physical
        ),
        "max_outlet_to_fsi_volume_source_ratio": (
            max_outlet_to_fsi_volume_source_ratio
        ),
        "outlet_to_fsi_volume_source_gate_scope": outlet_to_fsi_gate_scope,
        "required_min_outlet_to_main_volume_flux_ratio": (
            args.min_outlet_to_main_volume_flux_ratio
        ),
        "final_negative_z_all_sections": final_all_sections_negative_z,
        "final_negative_z_jet_sections": final_jet_sections_negative_z,
        "phase5_validation_complete": False,
        "interpretation_note": (
            "hibm_mpm_sharp uses simulation_core marker fields, IB node "
            "classification, no-slip Dirichlet rows, pressure Neumann rows, "
            "full-stress marker traction, marker-to-MPM scatter, and surface "
            "state update. Legacy projected/reduced force reports are not the "
            "primary coupling variable in this mode."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    process_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": run_process_completion_status(
                    validation_scope_complete=validation_scope_complete,
                    validation_passed=validation_passed,
                    partial_run_stopped=partial_run_stopped,
                    requested_steps=step_count,
                    completed_steps=len(rows),
                ),
                "validation_passed": validation_passed,
                "finished_at_unix": time.time(),
                "summary_json": str(summary_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
