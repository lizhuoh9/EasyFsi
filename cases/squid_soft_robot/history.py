from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

from simulation_core import (
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    vector_norm,
)

FINITE_REQUIRED_ROW_FIELDS = (
    "main_displacement_z_m",
    "tail_displacement_z_m",
    "main_velocity_z_mps",
    "tail_velocity_z_mps",
    "max_fluid_speed_mps",
    "cfl",
    "outlet_flow_negative_z_m3s",
    "divergence_l2",
    "divergence_max_abs",
    "interior_divergence_l2",
    "interior_divergence_max_abs",
    "pressure_correctable_divergence_l2",
    "pressure_correctable_divergence_max_abs",
    "pressure_correctable_divergence_cell_count",
    "pressure_fixed_divergence_l2",
    "pressure_fixed_divergence_max_abs",
    "pressure_fixed_divergence_cell_count",
    "interior_pressure_correctable_divergence_l2",
    "interior_pressure_correctable_divergence_max_abs",
    "interior_pressure_correctable_divergence_cell_count",
    "interior_pressure_fixed_divergence_l2",
    "interior_pressure_fixed_divergence_max_abs",
    "interior_pressure_fixed_divergence_cell_count",
    "pre_projection_divergence_l2",
    "pre_projection_divergence_max_abs",
    "projection_divergence_l2",
    "projection_divergence_max_abs",
    "projection_to_pre_divergence_l2_ratio",
    "post_boundary_divergence_l2",
    "post_boundary_divergence_max_abs",
    "post_boundary_to_pre_divergence_l2_ratio",
    "post_constraint_divergence_l2",
    "post_constraint_divergence_max_abs",
    "post_constraint_to_pre_divergence_l2_ratio",
    "pressure_traction_force_x_n",
    "pressure_traction_force_y_n",
    "pressure_traction_force_z_n",
    "pressure_traction_abs_force_n",
    "viscous_traction_force_x_n",
    "viscous_traction_force_y_n",
    "viscous_traction_force_z_n",
    "fluid_stress_traction_force_x_n",
    "fluid_stress_traction_force_y_n",
    "fluid_stress_traction_force_z_n",
    "projected_ibm_residual_mps",
    "projected_ibm_residual_l2_mps",
    "fsi_probe_valid_fraction",
    "fsi_force_probe_valid_fraction",
    "fsi_probe_invalid_area_m2",
    "fsi_probe_invalid_volume_source_m3s",
    "fsi_force_probe_invalid_area_m2",
    "fsi_force_probe_invalid_volume_source_m3s",
    "fsi_volume_source_m3s",
    "main_fsi_volume_source_m3s",
    "tail_fsi_volume_source_m3s",
    "pressure_outlet_source_volume_flux_m3s",
    "pressure_outlet_positive_source_volume_flux_m3s",
    "pressure_outlet_abs_source_volume_flux_m3s",
    "pressure_outlet_reachable_source_volume_flux_m3s",
    "pressure_outlet_unreached_source_volume_flux_m3s",
    "pressure_outlet_reachability_valid",
    "pressure_outlet_reachability_revision",
    "pressure_outlet_velocity_flux_m3s",
    "pressure_outlet_velocity_to_source_ratio",
    "pressure_outlet_velocity_to_net_source_ratio",
    "pressure_outlet_velocity_to_positive_source_ratio",
    "pressure_outlet_velocity_to_abs_source_ratio",
    "pressure_outlet_pressure_flux_m3s",
    "pressure_outlet_pressure_to_source_ratio",
    "pressure_outlet_pressure_to_net_source_ratio",
    "pressure_outlet_pressure_to_positive_source_ratio",
    "pressure_outlet_pressure_to_abs_source_ratio",
    "pressure_outlet_projection_pre_velocity_flux_m3s",
    "pressure_outlet_projection_post_pressure_velocity_flux_m3s",
    "pressure_outlet_projection_post_boundary_velocity_flux_m3s",
    "pressure_projection_cg_project_calls",
    "pressure_projection_cg_iterations_total",
    "pressure_projection_cg_iterations_max",
    "pressure_projection_cg_host_residual_checks",
    "pressure_projection_cg_mean_projection_count",
    "pressure_projection_cg_restart_count",
    "pressure_projection_cg_restart_count_measured",
    "pressure_projection_cg_restart_policy",
    "pressure_projection_cg_max_relative_residual",
    "pressure_projection_cg_max_initial_relative_residual",
    "pressure_projection_cg_breakdown_count",
    "pressure_projection_cg_breakdown_code",
    "pressure_projection_cg_breakdown_dAd",
    "fsi_trial_pressure_projection_cg_project_calls",
    "fsi_trial_pressure_projection_cg_iterations_total",
    "fsi_trial_pressure_projection_cg_iterations_max",
    "fsi_trial_pressure_projection_cg_host_residual_checks",
    "fsi_trial_pressure_projection_cg_mean_projection_count",
    "fsi_trial_pressure_projection_cg_max_relative_residual",
    "fsi_trial_pressure_projection_cg_max_initial_relative_residual",
    "fsi_trial_pressure_projection_cg_breakdown_count",
    "total_pressure_projection_cg_project_calls",
    "total_pressure_projection_cg_iterations_total",
    "total_pressure_projection_cg_iterations_max",
    "total_pressure_projection_cg_host_residual_checks",
    "total_pressure_projection_cg_mean_projection_count",
    "total_pressure_projection_cg_max_relative_residual",
    "total_pressure_projection_cg_max_initial_relative_residual",
    "total_pressure_projection_cg_breakdown_count",
    "fsi_action_reaction_residual_abs_n",
    "fsi_action_reaction_relative_error",
    "fsi_fluid_reaction_action_reaction_relative_error",
    "main_fsi_fluid_force_x_n",
    "main_fsi_fluid_force_y_n",
    "main_fsi_fluid_force_z_n",
    "tail_fsi_fluid_force_x_n",
    "tail_fsi_fluid_force_y_n",
    "tail_fsi_fluid_force_z_n",
    "main_fsi_fluid_reaction_x_n",
    "main_fsi_fluid_reaction_y_n",
    "main_fsi_fluid_reaction_z_n",
    "tail_fsi_fluid_reaction_x_n",
    "tail_fsi_fluid_reaction_y_n",
    "tail_fsi_fluid_reaction_z_n",
    "fsi_last_correction_grid_force_x_n",
    "fsi_last_correction_grid_force_y_n",
    "fsi_last_correction_grid_force_z_n",
    "fsi_last_correction_grid_decomposition_residual_abs_n",
    "fsi_last_correction_grid_decomposition_relative_error",
    "main_fsi_fluid_reaction_full_residual_n",
    "main_fsi_fluid_reaction_full_relative_error",
    "tail_fsi_fluid_reaction_full_residual_n",
    "tail_fsi_fluid_reaction_full_relative_error",
    "solid_mpm_transfer_relative_error",
    "solid_mpm_max_speed_mps",
    "solid_mpm_total_force_x_n",
    "solid_mpm_total_force_y_n",
    "solid_mpm_total_force_z_n",
)

NEO_HOOKEAN_REQUIRED_ROW_FIELDS = (
    "solid_mpm_max_abs_j",
)

HIBM_MPM_SHARP_REQUIRED_ROW_FIELDS = (
    "main_displacement_z_m",
    "tail_displacement_z_m",
    "main_velocity_z_mps",
    "tail_velocity_z_mps",
    "max_fluid_speed_mps",
    "cfl",
    "outlet_flow_negative_z_m3s",
    "divergence_l2",
    "divergence_max_abs",
    "interior_divergence_l2",
    "interior_divergence_max_abs",
    "pressure_correctable_divergence_l2",
    "pressure_correctable_divergence_max_abs",
    "pressure_correctable_divergence_cell_count",
    "pressure_fixed_divergence_l2",
    "pressure_fixed_divergence_max_abs",
    "pressure_fixed_divergence_cell_count",
    "interior_pressure_correctable_divergence_l2",
    "interior_pressure_correctable_divergence_max_abs",
    "interior_pressure_correctable_divergence_cell_count",
    "interior_pressure_fixed_divergence_l2",
    "interior_pressure_fixed_divergence_max_abs",
    "interior_pressure_fixed_divergence_cell_count",
    "pre_projection_divergence_l2",
    "pre_projection_divergence_max_abs",
    "projection_divergence_l2",
    "projection_divergence_max_abs",
    "projection_to_pre_divergence_l2_ratio",
    "post_boundary_divergence_l2",
    "post_boundary_divergence_max_abs",
    "post_boundary_to_pre_divergence_l2_ratio",
    "post_constraint_divergence_l2",
    "post_constraint_divergence_max_abs",
    "post_constraint_to_pre_divergence_l2_ratio",
    "hibm_ib_node_count",
    "hibm_ib_external_node_count",
    "hibm_ib_internal_node_count",
    "hibm_internal_obstacle_cell_count",
    "hibm_solid_band_nonprojectable_cell_count",
    "hibm_solid_band_interior_cell_count",
    "hibm_solid_band_enclosed_water_cell_count",
    "hibm_solid_band_velocity_dirichlet_protected_cell_count",
    "hibm_solid_band_mask_protected_cell_count",
    "hibm_row_cloud_orphan_cell_count",
    "hibm_row_cloud_orphan_component_count",
    "hibm_overflow_singleton_cleanup_cell_count",
    "hibm_overflow_singleton_cleanup_component_count",
    "hibm_pressure_disconnected_nonprojectable_cell_count",
    "hibm_pressure_disconnected_component_count",
    "hibm_pressure_disconnected_component_raw_count",
    "hibm_pressure_disconnected_largest_component_cell_count",
    "hibm_pressure_disconnected_singleton_component_count",
    "hibm_pressure_disconnected_small_component_threshold_cells",
    "hibm_pressure_disconnected_small_component_count",
    "hibm_pressure_disconnected_small_component_cell_count",
    "hibm_pressure_disconnected_component_overflow",
    "hibm_next_solid_band_interior_cell_count",
    "hibm_next_solid_band_enclosed_water_cell_count",
    "hibm_next_solid_band_velocity_dirichlet_protected_cell_count",
    "hibm_next_solid_band_mask_protected_cell_count",
    "hibm_next_row_cloud_orphan_cell_count",
    "hibm_next_row_cloud_orphan_component_count",
    "hibm_next_overflow_singleton_cleanup_cell_count",
    "hibm_next_overflow_singleton_cleanup_component_count",
    "hibm_next_pressure_disconnected_nonprojectable_cell_count",
    "hibm_next_pressure_disconnected_component_count",
    "hibm_next_pressure_disconnected_component_raw_count",
    "hibm_next_pressure_disconnected_largest_component_cell_count",
    "hibm_next_pressure_disconnected_singleton_component_count",
    "hibm_next_pressure_disconnected_small_component_threshold_cells",
    "hibm_next_pressure_disconnected_small_component_count",
    "hibm_next_pressure_disconnected_small_component_cell_count",
    "hibm_next_pressure_disconnected_component_overflow",
    "hibm_air_backed_cell_count",
    "hibm_air_backed_component_count",
    "hibm_air_backed_cell_volume_m3",
    "hibm_air_backed_seed_marker_count",
    "hibm_air_backed_seed_missed_marker_count",
    "hibm_air_backed_seed_fallback_cell_count",
    "hibm_air_backed_reachability_barrier_cell_count",
    "hibm_ib_invalid_projection_count",
    "hibm_boundary_no_slip_count",
    "hibm_boundary_pressure_neumann_count",
    "hibm_velocity_dirichlet_active_rows",
    "hibm_velocity_dirichlet_invalid_reconstruction_count",
    "hibm_velocity_dirichlet_invalid_no_fluid_sample_count",
    "hibm_velocity_dirichlet_invalid_nonpositive_gap_count",
    "hibm_velocity_dirichlet_invalid_node_behind_boundary_count",
    "hibm_velocity_dirichlet_invalid_node_beyond_interior_count",
    "hibm_velocity_dirichlet_narrow_gap_count",
    "hibm_velocity_dirichlet_relocated_rows",
    "hibm_velocity_dirichlet_relocation_merged_rows",
    "hibm_velocity_dirichlet_relocation_blocked_rows",
    "hibm_velocity_dirichlet_min_projection_weight",
    "hibm_velocity_dirichlet_max_projection_weight",
    "hibm_pressure_neumann_active_rows",
    "hibm_pressure_neumann_skipped_velocity_dirichlet_count",
    "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count",
    "hibm_pressure_neumann_skipped_obstacle_owner_count",
    "hibm_pressure_neumann_rhs_integral",
    "hibm_pressure_neumann_max_abs_rhs",
    "hibm_pressure_neumann_invalid_reconstruction_count",
    "hibm_pressure_neumann_min_reconstruction_gap_m",
    "hibm_pressure_neumann_max_reconstruction_gap_m",
    "hibm_pressure_neumann_max_transmissibility_m",
    "hibm_pressure_neumann_max_raw_transmissibility_m",
    "hibm_pressure_neumann_max_transmissibility_limit_m",
    "hibm_pressure_neumann_transmissibility_capped_row_count",
    "hibm_pressure_neumann_max_diagonal_per_m2",
    "hibm_pressure_neumann_active_marker_count",
    "hibm_pressure_neumann_max_rows_per_marker",
    "hibm_pressure_neumann_gradient_available",
    "hibm_pressure_neumann_gradient_active_marker_count",
    "hibm_pressure_neumann_gradient_max_abs_pa_per_m",
    "hibm_pressure_neumann_gradient_raw_max_abs_pa_per_m",
    "hibm_pressure_neumann_gradient_limited_count",
    "hibm_added_mass_stability_measured",
    "hibm_semi_implicit_coupling_enabled",
    "hibm_semi_implicit_coupling_matrix_active",
    "hibm_pressure_correctable_divergence_l2",
    "hibm_pressure_correctable_divergence_max_abs",
    "hibm_pressure_correctable_divergence_cell_count",
    "hibm_pressure_fixed_divergence_l2",
    "hibm_pressure_fixed_divergence_max_abs",
    "hibm_pressure_fixed_divergence_cell_count",
    "hibm_interior_pressure_correctable_divergence_l2",
    "hibm_interior_pressure_correctable_divergence_max_abs",
    "hibm_interior_pressure_correctable_divergence_cell_count",
    "hibm_interior_pressure_fixed_divergence_l2",
    "hibm_interior_pressure_fixed_divergence_max_abs",
    "hibm_interior_pressure_fixed_divergence_cell_count",
    "hibm_no_slip_residual_valid_marker_count",
    "hibm_no_slip_residual_invalid_marker_count",
    "hibm_no_slip_residual_max_mps",
    "hibm_no_slip_residual_l2_mps",
    "hibm_no_slip_residual_direct_sample_marker_count",
    "hibm_no_slip_residual_normal_walk_sample_marker_count",
    "hibm_no_slip_residual_nearest_fluid_sample_marker_count",
    "hibm_no_slip_residual_zero_normal_marker_count",
    "hibm_no_slip_residual_no_fluid_sample_marker_count",
    "hibm_no_slip_residual_primary_region_valid_marker_count",
    "hibm_no_slip_residual_primary_region_invalid_marker_count",
    "hibm_no_slip_residual_secondary_region_valid_marker_count",
    "hibm_no_slip_residual_secondary_region_invalid_marker_count",
    "hibm_no_slip_residual_other_region_valid_marker_count",
    "hibm_no_slip_residual_other_region_invalid_marker_count",
    "hibm_post_solid_kinematic_projection_applied",
    "hibm_post_solid_divergence_l2",
    "hibm_post_solid_divergence_max_abs",
    "hibm_post_solid_interior_divergence_l2",
    "hibm_post_solid_interior_divergence_max_abs",
    "hibm_post_solid_projection_divergence_l2",
    "hibm_post_solid_projection_divergence_max_abs",
    "hibm_post_solid_post_boundary_divergence_l2",
    "hibm_post_solid_post_boundary_divergence_max_abs",
    "hibm_post_solid_post_constraint_divergence_l2",
    "hibm_post_solid_post_constraint_divergence_max_abs",
    "hibm_post_solid_no_slip_residual_valid_marker_count",
    "hibm_post_solid_no_slip_residual_invalid_marker_count",
    "hibm_post_solid_no_slip_residual_max_mps",
    "hibm_post_solid_no_slip_residual_l2_mps",
    "hibm_post_solid_no_slip_residual_direct_sample_marker_count",
    "hibm_post_solid_no_slip_residual_normal_walk_sample_marker_count",
    "hibm_post_solid_no_slip_residual_nearest_fluid_sample_marker_count",
    "hibm_post_solid_no_slip_residual_zero_normal_marker_count",
    "hibm_post_solid_no_slip_residual_no_fluid_sample_marker_count",
    "hibm_post_solid_no_slip_residual_primary_region_valid_marker_count",
    "hibm_post_solid_no_slip_residual_primary_region_invalid_marker_count",
    "hibm_post_solid_no_slip_residual_secondary_region_valid_marker_count",
    "hibm_post_solid_no_slip_residual_secondary_region_invalid_marker_count",
    "hibm_post_solid_no_slip_residual_other_region_valid_marker_count",
    "hibm_post_solid_no_slip_residual_other_region_invalid_marker_count",
    "hibm_full_stress_valid_marker_count",
    "hibm_full_stress_invalid_marker_count",
    "hibm_full_stress_max_abs_traction_pa",
    "hibm_full_stress_two_sided_pressure_marker_count",
    "hibm_full_stress_one_sided_pressure_marker_count",
    "hibm_full_stress_one_sided_extended_marker_count",
    "hibm_full_stress_one_sided_gradient_missing_marker_count",
    "hibm_marker_primary_count",
    "hibm_marker_secondary_count",
    "hibm_marker_total_count",
    "hibm_marker_primary_stress_valid_count",
    "hibm_marker_primary_stress_invalid_count",
    "hibm_marker_secondary_stress_valid_count",
    "hibm_marker_secondary_stress_invalid_count",
    "hibm_marker_total_force_x_n",
    "hibm_marker_total_force_y_n",
    "hibm_marker_total_force_z_n",
    "hibm_marker_primary_force_norm_sum_n",
    "hibm_marker_secondary_force_norm_sum_n",
    "hibm_marker_total_force_norm_sum_n",
    "hibm_marker_primary_force_norm_max_n",
    "hibm_marker_secondary_force_norm_max_n",
    "hibm_marker_total_force_norm_max_n",
    "hibm_marker_action_reaction_residual_n",
    "hibm_mpm_external_force_clear_particle_count",
    "hibm_mpm_external_force_clear_max_abs_before_n",
    "hibm_mpm_external_force_fresh_for_solid_step",
    "hibm_mpm_scatter_action_reaction_residual_n",
    "hibm_surface_updated_marker_count",
    "hibm_surface_invalid_marker_count",
    "hibm_surface_max_displacement_m",
    "hibm_surface_max_speed_mps",
    "hibm_post_dirichlet_consistency_projection_applied",
    "hibm_post_dirichlet_consistency_projection_count",
    "pressure_projection_cg_project_calls",
    "pressure_projection_cg_iterations_total",
    "pressure_projection_cg_iterations_max",
    "pressure_projection_cg_host_residual_checks",
    "pressure_projection_cg_mean_projection_count",
    "pressure_projection_cg_restart_count",
    "pressure_projection_cg_restart_count_measured",
    "pressure_projection_cg_max_relative_residual",
    "pressure_projection_cg_max_initial_relative_residual",
    "pressure_projection_cg_breakdown_count",
    "pressure_projection_cg_breakdown_code",
    "pressure_projection_cg_breakdown_dAd",
    "pressure_solve_failed",
    "pressure_projection_physical_failure",
    "hibm_unreached_incompatible_component_count",
    "hibm_unreached_component_raw_count",
    "hibm_unreached_largest_component_cell_count",
    "hibm_unreached_singleton_component_count",
    "hibm_unreached_small_component_threshold_cells",
    "hibm_unreached_small_component_count",
    "hibm_unreached_small_component_cell_count",
    "hibm_projection_overflow_singleton_cleanup_cell_count",
    "hibm_projection_overflow_singleton_cleanup_component_count",
    "hibm_projection_tiny_unreached_cleanup_cell_count",
    "hibm_projection_tiny_unreached_cleanup_component_count",
    "hibm_unreached_component_rhs_mean_max_abs",
    "hibm_unreached_component_rhs_integral_max_abs",
    "fsi_added_mass_stability_measured",
    "fsi_semi_implicit_coupling_enabled",
    "fsi_semi_implicit_coupling_matrix_active",
    "fsi_action_reaction_residual_abs_n",
    "fsi_coupling_residual_norm_mps",
    "fsi_coupling_residual_max_mps",
    "main_fsi_fluid_force_x_n",
    "main_fsi_fluid_force_y_n",
    "main_fsi_fluid_force_z_n",
    "tail_fsi_fluid_force_x_n",
    "tail_fsi_fluid_force_y_n",
    "tail_fsi_fluid_force_z_n",
    "main_fsi_fluid_reaction_x_n",
    "main_fsi_fluid_reaction_y_n",
    "main_fsi_fluid_reaction_z_n",
    "tail_fsi_fluid_reaction_x_n",
    "tail_fsi_fluid_reaction_y_n",
    "tail_fsi_fluid_reaction_z_n",
    "fsi_grid_force_x_n",
    "fsi_grid_force_y_n",
    "fsi_grid_force_z_n",
    "fsi_volume_source_m3s",
    "pressure_outlet_source_volume_flux_m3s",
    "pressure_outlet_positive_source_volume_flux_m3s",
    "pressure_outlet_abs_source_volume_flux_m3s",
    "pressure_outlet_reachability_valid",
    "pressure_outlet_reachability_revision",
    "pressure_outlet_velocity_flux_m3s",
    "pressure_outlet_velocity_to_source_ratio",
    "pressure_outlet_velocity_to_net_source_ratio",
    "pressure_outlet_velocity_to_positive_source_ratio",
    "pressure_outlet_velocity_to_abs_source_ratio",
    "pressure_outlet_pressure_flux_m3s",
    "pressure_outlet_pressure_to_source_ratio",
    "pressure_outlet_pressure_to_net_source_ratio",
    "pressure_outlet_pressure_to_positive_source_ratio",
    "pressure_outlet_pressure_to_abs_source_ratio",
    "solid_mpm_transfer_relative_error",
    "solid_mpm_max_speed_mps",
    "solid_mpm_grid_out_of_bounds_particle_count",
    "solid_mpm_total_force_x_n",
    "solid_mpm_total_force_y_n",
    "solid_mpm_total_force_z_n",
)

def finite_required_row_fields_for_solid_model(solid_model: str) -> tuple[str, ...]:
    if solid_model == "neo_hookean_mpm":
        return FINITE_REQUIRED_ROW_FIELDS + NEO_HOOKEAN_REQUIRED_ROW_FIELDS
    return FINITE_REQUIRED_ROW_FIELDS

def finite_required_row_fields_for_mode(
    fsi_coupling_mode: str,
    *,
    solid_model: str,
) -> tuple[str, ...]:
    if str(fsi_coupling_mode) == FSI_COUPLING_MODE_HIBM_MPM_SHARP:
        fields = HIBM_MPM_SHARP_REQUIRED_ROW_FIELDS
    else:
        fields = finite_required_row_fields_for_solid_model(solid_model)
    if solid_model == "neo_hookean_mpm":
        return fields + tuple(
            field for field in NEO_HOOKEAN_REQUIRED_ROW_FIELDS if field not in fields
        )
    return fields

def divergence_sample_report_fields(
    divergence: Mapping[str, object],
) -> dict[str, object]:
    pre_projection_measured = (
        "pre_projection_l2" in divergence
        and "pre_projection_max_abs" in divergence
    )
    pre_projection_l2 = float(divergence.get("pre_projection_l2", divergence["l2"]))
    projection_l2 = float(divergence.get("projection_l2", divergence["l2"]))
    post_boundary_l2 = float(
        divergence.get(
            "post_boundary_l2",
            divergence.get("projection_l2", divergence["l2"]),
        )
    )
    post_constraint_l2 = float(divergence.get("post_constraint_l2", divergence["l2"]))
    projection_ratio_measured = pre_projection_measured and "projection_l2" in divergence
    post_boundary_ratio_measured = (
        pre_projection_measured and "post_boundary_l2" in divergence
    )
    post_constraint_ratio_measured = (
        pre_projection_measured and "post_constraint_l2" in divergence
    )
    pressure_divergence_split_measured = (
        "pressure_correctable_l2" in divergence
        and "pressure_fixed_l2" in divergence
        and "interior_pressure_correctable_l2" in divergence
        and "interior_pressure_fixed_l2" in divergence
    )
    pressure_correctable_l2 = float(
        divergence.get("pressure_correctable_l2", divergence["l2"])
    )
    pressure_correctable_max_abs = float(
        divergence.get("pressure_correctable_max_abs", divergence["max_abs"])
    )
    pressure_correctable_cell_count = int(
        divergence.get("pressure_correctable_cell_count", 0) or 0
    )
    pressure_fixed_l2 = float(divergence.get("pressure_fixed_l2", 0.0))
    pressure_fixed_max_abs = float(divergence.get("pressure_fixed_max_abs", 0.0))
    pressure_fixed_cell_count = int(
        divergence.get("pressure_fixed_cell_count", 0) or 0
    )
    interior_pressure_correctable_l2 = float(
        divergence.get(
            "interior_pressure_correctable_l2",
            divergence.get("interior_l2", divergence["l2"]),
        )
    )
    interior_pressure_correctable_max_abs = float(
        divergence.get(
            "interior_pressure_correctable_max_abs",
            divergence.get("interior_max_abs", divergence["max_abs"]),
        )
    )
    interior_pressure_correctable_cell_count = int(
        divergence.get("interior_pressure_correctable_cell_count", 0) or 0
    )
    interior_pressure_fixed_l2 = float(
        divergence.get("interior_pressure_fixed_l2", 0.0)
    )
    interior_pressure_fixed_max_abs = float(
        divergence.get("interior_pressure_fixed_max_abs", 0.0)
    )
    interior_pressure_fixed_cell_count = int(
        divergence.get("interior_pressure_fixed_cell_count", 0) or 0
    )

    def l2_ratio(numerator: float, denominator: float, *, measured: bool) -> float:
        if not measured:
            return 1.0
        numerator_abs = abs(float(numerator))
        denominator_abs = abs(float(denominator))
        if denominator_abs <= 1.0e-30:
            if numerator_abs <= 1.0e-30:
                return 0.0
            denominator_abs = 1.0e-30
        return numerator_abs / denominator_abs

    return {
        "divergence_l2": float(divergence["l2"]),
        "divergence_max_abs": float(divergence["max_abs"]),
        "interior_divergence_l2": float(divergence.get("interior_l2", divergence["l2"])),
        "interior_divergence_max_abs": float(
            divergence.get("interior_max_abs", divergence["max_abs"])
        ),
        "unreached_divergence_l2": float(divergence.get("unreached_l2", 0.0)),
        "unreached_divergence_max_abs": float(
            divergence.get("unreached_max_abs", 0.0)
        ),
        "unreached_divergence_cell_count": int(
            float(divergence.get("unreached_cell_count", 0))
        ),
        "pressure_correctable_divergence_l2": pressure_correctable_l2,
        "pressure_correctable_divergence_max_abs": pressure_correctable_max_abs,
        "pressure_correctable_divergence_cell_count": (
            pressure_correctable_cell_count
        ),
        "pressure_fixed_divergence_l2": pressure_fixed_l2,
        "pressure_fixed_divergence_max_abs": pressure_fixed_max_abs,
        "pressure_fixed_divergence_cell_count": pressure_fixed_cell_count,
        "interior_pressure_correctable_divergence_l2": (
            interior_pressure_correctable_l2
        ),
        "interior_pressure_correctable_divergence_max_abs": (
            interior_pressure_correctable_max_abs
        ),
        "interior_pressure_correctable_divergence_cell_count": (
            interior_pressure_correctable_cell_count
        ),
        "interior_pressure_fixed_divergence_l2": interior_pressure_fixed_l2,
        "interior_pressure_fixed_divergence_max_abs": (
            interior_pressure_fixed_max_abs
        ),
        "interior_pressure_fixed_divergence_cell_count": (
            interior_pressure_fixed_cell_count
        ),
        "pressure_divergence_split_measured": pressure_divergence_split_measured,
        "pressure_divergence_split_source": (
            "fluid_projection_report"
            if pressure_divergence_split_measured
            else "fallback_unsplit_final_divergence"
        ),
        "pre_projection_divergence_l2": pre_projection_l2,
        "pre_projection_divergence_max_abs": float(
            divergence.get("pre_projection_max_abs", divergence["max_abs"])
        ),
        "pre_projection_divergence_measured": pre_projection_measured,
        "pre_projection_divergence_source": (
            "fluid_projection_report"
            if pre_projection_measured
            else "fallback_final_divergence"
        ),
        "projection_divergence_l2": projection_l2,
        "projection_divergence_max_abs": float(
            divergence.get("projection_max_abs", divergence["max_abs"])
        ),
        "projection_to_pre_divergence_l2_ratio": l2_ratio(
            projection_l2,
            pre_projection_l2,
            measured=projection_ratio_measured,
        ),
        "projection_divergence_ratio_measured": projection_ratio_measured,
        "post_boundary_divergence_l2": post_boundary_l2,
        "post_boundary_divergence_max_abs": float(
            divergence.get(
                "post_boundary_max_abs",
                divergence.get("projection_max_abs", divergence["max_abs"]),
            )
        ),
        "post_boundary_to_pre_divergence_l2_ratio": l2_ratio(
            post_boundary_l2,
            pre_projection_l2,
            measured=post_boundary_ratio_measured,
        ),
        "post_boundary_divergence_ratio_measured": post_boundary_ratio_measured,
        "post_constraint_divergence_l2": post_constraint_l2,
        "post_constraint_divergence_max_abs": float(
            divergence.get("post_constraint_max_abs", divergence["max_abs"])
        ),
        "post_constraint_to_pre_divergence_l2_ratio": l2_ratio(
            post_constraint_l2,
            pre_projection_l2,
            measured=post_constraint_ratio_measured,
        ),
        "post_constraint_divergence_ratio_measured": post_constraint_ratio_measured,
    }

def _required_finite_row_number(
    row: dict[str, object],
    field: str,
    *,
    context: str,
) -> float:
    if field not in row:
        raise KeyError(f"{context} missing required numeric field {field!r}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} field {field!r} must be numeric, got {row[field]!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} field {field!r} is non-finite: {value!r}")
    return value

def _required_finite_row_vector(
    row: dict[str, object],
    fields: tuple[str, str, str],
    *,
    context: str,
) -> tuple[float, float, float]:
    return tuple(
        _required_finite_row_number(row, field, context=context)
        for field in fields
    )

def _final_row_number(
    final_row: dict[str, object] | None,
    field: str,
    *,
    context: str = "final row",
) -> float:
    if final_row is None:
        return 0.0
    return _required_finite_row_number(final_row, field, context=context)

def _final_row_number_or_none(
    final_row: dict[str, object] | None,
    field: str,
) -> float | None:
    if final_row is None:
        return None
    try:
        value = float(final_row.get(field, math.nan))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value

def _final_row_int(
    final_row: dict[str, object] | None,
    field: str,
    *,
    context: str = "final row",
) -> int:
    if final_row is None:
        return 0
    value = _required_finite_row_number(final_row, field, context=context)
    integer_value = int(value)
    if float(integer_value) != value:
        raise ValueError(f"{context} field {field!r} must be an integer, got {value!r}")
    return integer_value

def _row_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)

def _rows_max_int(rows: Sequence[Mapping[str, object]], key: str) -> int:
    if not rows:
        return 0
    return max(int(row.get(key, 0) or 0) for row in rows)

def _rows_any_bool(rows: Sequence[Mapping[str, object]], key: str) -> bool:
    return any(_row_bool(row.get(key, False)) for row in rows)

def count_enabled_unconverged_fsi_rows(
    rows: Sequence[Mapping[str, object]],
) -> int:
    return sum(
        1
        for row in rows
        if _row_bool(row.get("fsi_coupling_enabled", False))
        and not _row_bool(row.get("fsi_coupling_converged", False))
    )

def solid_mpm_force_nonzero_when_pressure_loaded(
    rows: Sequence[dict[str, object]],
    *,
    force_required: bool,
    tolerance_n: float = 0.0,
) -> bool:
    if not force_required:
        return True

    loaded_row_count = 0
    for index, row in enumerate(rows):
        pressure_pa = _required_finite_row_number(
            row,
            "pressure_load_pa",
            context=f"history row {index}",
        )
        if abs(pressure_pa) <= 0.0:
            continue
        loaded_row_count += 1
        force_components_n = _required_finite_row_vector(
            row,
            (
                "solid_mpm_total_force_x_n",
                "solid_mpm_total_force_y_n",
                "solid_mpm_total_force_z_n",
            ),
            context=f"pressure-loaded history row {index}",
        )
        if vector_norm(force_components_n) > tolerance_n:
            return True

    return loaded_row_count == 0

def _required_finite_triplet(
    value: object,
    *,
    field: str,
    context: str,
) -> tuple[float, float, float]:
    try:
        components = tuple(float(component) for component in value)  # type: ignore[operator]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} field {field!r} must be a finite 3D vector") from exc
    if len(components) != 3:
        raise ValueError(
            f"{context} field {field!r} must have exactly 3 components, got {len(components)}"
        )
    for component_index, component in enumerate(components):
        if not math.isfinite(component):
            raise ValueError(
                f"{context} field {field!r}[{component_index}] is not finite: {component!r}"
            )
    return components

def _required_finite_report_number(
    report: object,
    *,
    field: str,
    context: str,
) -> float:
    if not hasattr(report, field):
        raise AttributeError(f"{context} missing required numeric report field {field!r}")
    try:
        value = float(getattr(report, field))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} field {field!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} field {field!r} is non-finite: {value!r}")
    return value

def solid_force_vector_from_report(
    report: object,
    *,
    solid_model: str,
) -> tuple[float, float, float]:
    if hasattr(report, "total_force_n"):
        return _required_finite_triplet(
            getattr(report, "total_force_n"),
            field="total_force_n",
            context=f"solid model {solid_model!r} report",
        )
    if solid_model == "neo_hookean_mpm" and hasattr(report, "external_force_n"):
        return _required_finite_triplet(
            getattr(report, "external_force_n"),
            field="external_force_n",
            context=f"solid model {solid_model!r} report",
        )
    raise AttributeError(
        f"solid model {solid_model!r} report does not expose a finite 3D force vector"
    )

def required_projected_ibm_force_report(report: object | None) -> object:
    if report is None:
        raise RuntimeError("projected IBM step did not produce a force report")
    context = "projected IBM force report"
    for field in (
        "grid_force_n",
        "primary_fluid_force_n",
        "secondary_fluid_force_n",
        "constraint_force_n",
        "primary_constraint_force_n",
        "secondary_constraint_force_n",
    ):
        _required_finite_triplet(getattr(report, field), field=field, context=context)
    for field in (
        "volume_source_m3s",
        "primary_volume_source_m3s",
        "secondary_volume_source_m3s",
        "active_force_cells",
        "force_sample_count",
        "force_invalid_probe_count",
        "force_valid_probe_count",
        "force_valid_probe_fraction",
        "invalid_probe_area_m2",
        "invalid_probe_volume_source_m3s",
    ):
        _required_finite_report_number(report, field=field, context=context)
    return report

def required_fluid_impulse_report(report: object | None) -> object:
    if report is None:
        raise RuntimeError("projected IBM step did not produce a fluid impulse report")
    context = "fluid impulse report"
    for field in ("grid_impulse_n_s", "momentum_delta_n_s"):
        _required_finite_triplet(getattr(report, field), field=field, context=context)
    _required_finite_report_number(report, field="impulse_relative_error", context=context)
    _required_finite_report_number(report, field="active_velocity_cells", context=context)
    return report

WRITE_CSV_REPLACE_ATTEMPTS = 20

WRITE_CSV_REPLACE_BACKOFF_S = 0.25

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    last_error: PermissionError | None = None
    for _ in range(WRITE_CSV_REPLACE_ATTEMPTS):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(WRITE_CSV_REPLACE_BACKOFF_S)
    raise last_error

def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
