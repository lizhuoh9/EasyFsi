from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict

from simulation_core import (
    checks_passed,
    finite_field_diagnostics,
    vector_norm,
)

from .coupling_common import (
    outlet_to_fsi_volume_source_gate_scope,
    physical_outlet_to_fsi_volume_source_passes,
)
from .history import (
    _final_row_int,
    _final_row_number,
    _final_row_number_or_none,
    _required_finite_row_number,
    _row_bool,
    _rows_any_bool,
    _rows_max_int,
    finite_required_row_fields_for_solid_model,
    solid_mpm_force_nonzero_when_pressure_loaded,
)
from .outputs import run_process_completion_status
from .rows import signed_positive_source_flux_ratio
from .schedules import pressure_schedule_applied_in_history
from .setup import (
    reduced_water_geometry_report,
)


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


def build_sharp_case_run_report(context: Mapping[str, object]) -> dict[str, object]:
    adaptive_fluid_substeps_enabled = _context_value(context, "adaptive_fluid_substeps_enabled")
    args = _context_value(context, "args")
    cad_provenance = _context_value(context, "cad_provenance")
    effective_fluid_substep_dt_s = _context_value(context, "effective_fluid_substep_dt_s")
    effective_fluid_substeps = _context_value(context, "effective_fluid_substeps")
    fluid_grid_axis_max_spacing_m = _context_value(context, "fluid_grid_axis_max_spacing_m")
    fluid_grid_axis_min_spacing_m = _context_value(context, "fluid_grid_axis_min_spacing_m")
    fluid_grid_resolution = _context_value(context, "fluid_grid_resolution")
    fluid_grid_uniform_spacing_m = _context_value(context, "fluid_grid_uniform_spacing_m")
    fsi_coupling_iterations = _context_value(context, "fsi_coupling_iterations")
    fsi_marker_coupling_tolerance_mps = _context_value(context, "fsi_marker_coupling_tolerance_mps")
    full_pressure_waveform_steps = _context_value(context, "full_pressure_waveform_steps")
    history_path = _context_value(context, "history_path")
    initial_fluid_obstacle_mode = _context_value(context, "initial_fluid_obstacle_mode")
    material = _context_value(context, "material")
    membrane_thickness_scale = _context_value(context, "membrane_thickness_scale")
    output_dir = _context_value(context, "output_dir")
    partial_run_reason = _context_value(context, "partial_run_reason")
    partial_run_stopped = _context_value(context, "partial_run_stopped")
    pressure_boundary_mapping = _context_value(context, "pressure_boundary_mapping")
    pressure_closure_normal = _context_value(context, "pressure_closure_normal")
    pressure_far_side_normal_sign = _context_value(context, "pressure_far_side_normal_sign")
    pressure_load_direction = _context_value(context, "pressure_load_direction")
    pressure_load_region_id = _context_value(context, "pressure_load_region_id")
    pressure_load_source_region_id = _context_value(context, "pressure_load_source_region_id")
    pressure_outlet_boundary_report = _context_value(context, "pressure_outlet_boundary_report")
    pressure_outlet_zmin_enabled = _context_value(context, "pressure_outlet_zmin_enabled")
    pressure_schedule_input = _context_value(context, "pressure_schedule_input")
    pressure_solver_name = _context_value(context, "pressure_solver_name")
    primary_shell_region_id = _context_value(context, "primary_shell_region_id")
    process_path = _context_value(context, "process_path")
    real_cad_step_binding = _context_value(context, "real_cad_step_binding")
    region14_aperture_carve_enabled = _context_value(context, "region14_aperture_carve_enabled")
    region14_aperture_carve_source = _context_value(context, "region14_aperture_carve_source")
    region14_aperture_geometry = _context_value(context, "region14_aperture_geometry")
    rows = _context_value(context, "rows")
    secondary_shell_region_id = _context_value(context, "secondary_shell_region_id")
    simulator = _context_value(context, "simulator")
    solid_density_scale = _context_value(context, "solid_density_scale")
    solid_mpm_bounds_max_m = _context_value(context, "solid_mpm_bounds_max_m")
    solid_mpm_bounds_min_m = _context_value(context, "solid_mpm_bounds_min_m")
    solid_mpm_bounds_padding_m = _context_value(context, "solid_mpm_bounds_padding_m")
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
    tri_metadata = _context_value(context, "tri_metadata")

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
    max_velocity_dirichlet_invariant_violation_count = (
        max(
            int(row["hibm_velocity_dirichlet_invariant_violation_count"])
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
    required_row_fields = finite_required_row_fields_for_solid_model(
        args.solid_model
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
        "hibm_velocity_dirichlet_components_present": bool(
            last
            and int(
                last["hibm_velocity_dirichlet_final_active_component_count"]
            )
            > 0
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
        "hibm_velocity_dirichlet_invariants_valid": (
            max_velocity_dirichlet_invariant_violation_count == 0
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
        "fsi_coupling_iterations_requested": fsi_coupling_iterations,
        "max_fsi_coupling_iterations_used": max_fsi_coupling_iterations_used,
        "max_fsi_coupling_iqn_ils_least_squares_update_count": (
            max_fsi_coupling_iqn_ils_least_squares_update_count
        ),
        "fsi_marker_coupling_tolerance_mps": (
            fsi_marker_coupling_tolerance_mps
        ),
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
        "max_hibm_velocity_dirichlet_invariant_violation_count": (
            max_velocity_dirichlet_invariant_violation_count
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
            "state update. Marker velocity is the canonical coupling unknown."
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
