from __future__ import annotations

import math
from collections.abc import Mapping

from .history import solid_force_vector_from_report
from .source_config import _vector3


def _mapping_vector3(
    mapping: Mapping[str, object],
    key: str,
) -> tuple[float, float, float]:
    return _vector3(mapping[key], name=key)


def _mapping_float(
    mapping: Mapping[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    return float(mapping.get(key, default) or 0.0)


def _mapping_int(
    mapping: Mapping[str, object],
    key: str,
    default: int = 0,
) -> int:
    return int(mapping.get(key, default) or 0)


def build_hibm_mpm_sharp_case_row(
    *,
    step: int,
    sample_report: Mapping[str, object],
    sharp_summary: Mapping[str, object],
    fluid_projection_report: Mapping[str, object],
    pressure_outlet_report: Mapping[str, object],
    fluid_dt_s: float,
    solid_mpm_report,
    solid_model: str,
    fsi_coupling_mode_report: Mapping[str, object],
    fsi_coupling_iterations_requested: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "step": int(step),
        **dict(sample_report),
        **dict(sharp_summary),
    }
    primary_force_n = _mapping_vector3(sharp_summary, "hibm_marker_primary_force_n")
    secondary_force_n = _mapping_vector3(sharp_summary, "hibm_marker_secondary_force_n")
    total_force_n = _mapping_vector3(sharp_summary, "hibm_marker_total_force_n")
    primary_marker_count = int(sharp_summary["hibm_marker_primary_count"])
    secondary_marker_count = int(sharp_summary["hibm_marker_secondary_count"])
    total_marker_count = int(sharp_summary["hibm_marker_total_count"])
    actual_fluid_substeps = max(
        1,
        _mapping_int(fluid_projection_report, "fluid_substeps", 1),
    )
    primary_reaction_n = tuple(-value for value in primary_force_n)
    secondary_reaction_n = tuple(-value for value in secondary_force_n)
    solid_mpm_total_force_n = solid_force_vector_from_report(
        solid_mpm_report,
        solid_model=solid_model,
    )
    sharp_fsi_volume_source_m3s = _mapping_float(
        pressure_outlet_report,
        "source_volume_flux_m3s",
    )
    scatter_force_residual_n = _mapping_float(
        sharp_summary,
        "hibm_mpm_scatter_action_reaction_residual_n",
    )
    sharp_fsi_convergence_measured = (
        "hibm_fsi_coupling_residual_l2_mps" in sharp_summary
    )
    sharp_explicit_single_pass = bool(
        sharp_summary.get("hibm_fsi_coupling_explicit_single_pass", True)
    )
    sharp_added_mass_status = str(
        sharp_summary.get("hibm_added_mass_stability_status", "unmeasured")
    )
    if sharp_explicit_single_pass and not sharp_fsi_convergence_measured:
        sharp_added_mass_status = "unmeasured_single_pass"
    if "hibm_fsi_coupling_residual_l2_mps" in sharp_summary:
        sharp_fsi_residual_source = str(
            sharp_summary.get(
                "hibm_fsi_coupling_residual_source",
                "marker_surface_fixed_point_velocity_residual_l2_mps",
            )
        )
        sharp_fsi_residual_l2_mps = _mapping_float(
            sharp_summary,
            "hibm_fsi_coupling_residual_l2_mps",
        )
        sharp_fsi_residual_max_mps = _mapping_float(
            sharp_summary,
            "hibm_fsi_coupling_residual_max_mps",
        )
    elif bool(sharp_summary.get("hibm_post_solid_kinematic_projection_applied", False)):
        post_solid_no_slip_valid_marker_count = _mapping_int(
            sharp_summary,
            "hibm_post_solid_no_slip_residual_valid_marker_count",
            0,
        )
        if total_marker_count > 0 and post_solid_no_slip_valid_marker_count <= 0:
            sharp_fsi_residual_source = (
                "unmeasured_no_valid_post_solid_no_slip_markers"
            )
            sharp_fsi_residual_l2_mps = math.nan
            sharp_fsi_residual_max_mps = math.nan
        else:
            sharp_fsi_residual_source = (
                "hibm_post_solid_no_slip_velocity_residual_l2_mps"
            )
            sharp_fsi_residual_l2_mps = _mapping_float(
                sharp_summary,
                "hibm_post_solid_no_slip_residual_l2_mps",
            )
            sharp_fsi_residual_max_mps = _mapping_float(
                sharp_summary,
                "hibm_post_solid_no_slip_residual_max_mps",
            )
    else:
        no_slip_valid_marker_count = _mapping_int(
            sharp_summary,
            "hibm_no_slip_residual_valid_marker_count",
            0,
        )
        if total_marker_count > 0 and no_slip_valid_marker_count <= 0:
            sharp_fsi_residual_source = "unmeasured_no_valid_no_slip_markers"
            sharp_fsi_residual_l2_mps = math.nan
            sharp_fsi_residual_max_mps = math.nan
        else:
            sharp_fsi_residual_source = "hibm_no_slip_velocity_residual_l2_mps"
            sharp_fsi_residual_l2_mps = _mapping_float(
                sharp_summary,
                "hibm_no_slip_residual_l2_mps",
            )
            sharp_fsi_residual_max_mps = _mapping_float(
                sharp_summary,
                "hibm_no_slip_residual_max_mps",
            )

    row.update(
        {
            "fsi_coupling_iterations_requested": int(
                fsi_coupling_iterations_requested
            ),
            "fsi_coupling_mode": str(fsi_coupling_mode_report["mode"]),
            "fsi_coupling_mode_paper_hibm_mpm": bool(
                fsi_coupling_mode_report["paper_hibm_mpm"]
            ),
            "region_pair_reaction_diagnostic_only": bool(
                fsi_coupling_mode_report["region_pair_reaction_diagnostic_only"]
            ),
            "fsi_coupling_solver": "hibm_mpm_sharp",
            "fsi_coupling_scheme": str(
                sharp_summary.get("hibm_coupling_scheme", "explicit_loose")
            ),
            "fsi_coupling_iterations_used": _mapping_int(
                sharp_summary,
                "hibm_fsi_coupling_iterations_used",
                1,
            ),
            "fsi_coupling_enabled": True,
            "fsi_coupling_explicit_single_pass": sharp_explicit_single_pass,
            "fsi_added_mass_stability_status": sharp_added_mass_status,
            "fsi_added_mass_stability_measured": bool(
                sharp_summary.get("hibm_added_mass_stability_measured", False)
            ),
            "fsi_added_mass_stabilization": str(
                sharp_summary.get("hibm_added_mass_stabilization", "none")
            ),
            "fsi_semi_implicit_coupling_enabled": bool(
                sharp_summary.get("hibm_semi_implicit_coupling_enabled", False)
            ),
            "fsi_semi_implicit_coupling_matrix_active": bool(
                sharp_summary.get(
                    "hibm_semi_implicit_coupling_matrix_active",
                    False,
                )
            ),
            "fsi_coupling_step_completed": True,
            "fsi_coupling_convergence_measured": bool(
                sharp_fsi_convergence_measured
            ),
            "fsi_coupling_converged": bool(
                sharp_summary.get("hibm_fsi_coupling_converged", False)
            ),
            "fluid_substeps": actual_fluid_substeps,
            "fluid_substep_dt_s": float(fluid_dt_s) / float(actual_fluid_substeps),
            "fluid_advection_scheme": str(
                fluid_projection_report.get("fluid_advection_scheme", "euler")
            ),
            "fsi_coupling_residual_norm_n": math.nan,
            "fsi_coupling_residual_norm_mps": sharp_fsi_residual_l2_mps,
            "fsi_coupling_residual_max_mps": sharp_fsi_residual_max_mps,
            "fsi_coupling_residual_units": "m/s",
            "fsi_coupling_residual_source": sharp_fsi_residual_source,
            "hibm_marker_total_force_x_n": total_force_n[0],
            "hibm_marker_total_force_y_n": total_force_n[1],
            "hibm_marker_total_force_z_n": total_force_n[2],
            "hibm_marker_primary_count": primary_marker_count,
            "hibm_marker_secondary_count": secondary_marker_count,
            "hibm_marker_total_count": total_marker_count,
            "main_fsi_fluid_force_x_n": primary_force_n[0],
            "main_fsi_fluid_force_y_n": primary_force_n[1],
            "main_fsi_fluid_force_z_n": primary_force_n[2],
            "tail_fsi_fluid_force_x_n": secondary_force_n[0],
            "tail_fsi_fluid_force_y_n": secondary_force_n[1],
            "tail_fsi_fluid_force_z_n": secondary_force_n[2],
            "main_fsi_fluid_reaction_x_n": primary_reaction_n[0],
            "main_fsi_fluid_reaction_y_n": primary_reaction_n[1],
            "main_fsi_fluid_reaction_z_n": primary_reaction_n[2],
            "tail_fsi_fluid_reaction_x_n": secondary_reaction_n[0],
            "tail_fsi_fluid_reaction_y_n": secondary_reaction_n[1],
            "tail_fsi_fluid_reaction_z_n": secondary_reaction_n[2],
            "main_interface_reaction_z_n": primary_reaction_n[2],
            "tail_interface_reaction_z_n": secondary_reaction_n[2],
            "fsi_action_reaction_balance_measured": False,
            "fsi_action_reaction_residual_abs_n": scatter_force_residual_n,
            "fsi_action_reaction_residual_source": (
                "marker_to_mpm_scatter_force_conservation"
            ),
            "fsi_fluid_reaction_action_reaction_relative_error": math.nan,
            "fsi_fluid_reaction_action_reaction_measured": False,
            "fsi_grid_force_x_n": total_force_n[0],
            "fsi_grid_force_y_n": total_force_n[1],
            "fsi_grid_force_z_n": total_force_n[2],
            "fsi_volume_source_m3s": sharp_fsi_volume_source_m3s,
            "fsi_volume_source_semantics": (
                "computed_pressure_outlet_source_field_not_region_decomposed"
            ),
            "pressure_outlet_source_volume_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "source_volume_flux_m3s",
            ),
            "pressure_outlet_positive_source_volume_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "positive_source_volume_flux_m3s",
            ),
            "pressure_outlet_abs_source_volume_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "abs_source_volume_flux_m3s",
            ),
            "pressure_outlet_reachable_source_volume_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_reachable_source_volume_flux_m3s",
            ),
            "pressure_outlet_unreached_source_volume_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_volume_flux_m3s",
            ),
            "pressure_outlet_reachability_valid": bool(
                pressure_outlet_report.get("zmin_reachability_valid", False)
            ),
            "pressure_outlet_reachability_revision": _mapping_int(
                pressure_outlet_report,
                "zmin_reachability_revision",
            ),
            "pressure_outlet_reachable_source_cell_count": _mapping_int(
                pressure_outlet_report,
                "zmin_reachable_source_cell_count",
            ),
            "pressure_outlet_unreached_source_cell_count": _mapping_int(
                pressure_outlet_report,
                "zmin_unreached_source_cell_count",
            ),
            "pressure_outlet_unreached_source_abs_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_abs_flux_m3s",
            ),
            "pressure_outlet_unreached_source_centroid_x_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_centroid_x_m",
            ),
            "pressure_outlet_unreached_source_centroid_y_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_centroid_y_m",
            ),
            "pressure_outlet_unreached_source_centroid_z_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_centroid_z_m",
            ),
            "pressure_outlet_unreached_source_min_x_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_min_x_m",
            ),
            "pressure_outlet_unreached_source_min_y_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_min_y_m",
            ),
            "pressure_outlet_unreached_source_min_z_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_min_z_m",
            ),
            "pressure_outlet_unreached_source_max_x_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_max_x_m",
            ),
            "pressure_outlet_unreached_source_max_y_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_max_y_m",
            ),
            "pressure_outlet_unreached_source_max_z_m": _mapping_float(
                pressure_outlet_report,
                "zmin_unreached_source_max_z_m",
            ),
            "pressure_outlet_velocity_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_velocity_outlet_flux_m3s",
            ),
            "pressure_outlet_velocity_to_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_velocity_outlet_to_source_ratio",
            ),
            "pressure_outlet_velocity_to_net_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_velocity_outlet_to_net_source_ratio",
            ),
            "pressure_outlet_velocity_to_positive_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_velocity_outlet_to_positive_source_ratio",
            ),
            "pressure_outlet_velocity_to_abs_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_velocity_outlet_to_abs_source_ratio",
            ),
            "pressure_outlet_pressure_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_pressure_outlet_flux_m3s",
            ),
            "pressure_outlet_pressure_to_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_pressure_outlet_to_source_ratio",
            ),
            "pressure_outlet_pressure_to_net_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_pressure_outlet_to_net_source_ratio",
            ),
            "pressure_outlet_pressure_to_positive_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_pressure_outlet_to_positive_source_ratio",
            ),
            "pressure_outlet_pressure_to_abs_source_ratio": _mapping_float(
                pressure_outlet_report,
                "zmin_pressure_outlet_to_abs_source_ratio",
            ),
            "pressure_outlet_projection_pre_velocity_flux_m3s": _mapping_float(
                pressure_outlet_report,
                "zmin_projection_pre_velocity_outlet_flux_m3s",
            ),
            "pressure_outlet_projection_post_pressure_velocity_flux_m3s": (
                _mapping_float(
                    pressure_outlet_report,
                    "zmin_projection_post_pressure_velocity_outlet_flux_m3s",
                )
            ),
            "pressure_outlet_projection_post_boundary_velocity_flux_m3s": (
                _mapping_float(
                    pressure_outlet_report,
                    "zmin_projection_post_boundary_velocity_outlet_flux_m3s",
                )
            ),
            "pressure_projection_cg_project_calls": _mapping_int(
                fluid_projection_report,
                "cg_project_calls",
            ),
            "pressure_solver_requested": str(
                fluid_projection_report.get("pressure_solver_requested", "")
            ),
            "pressure_solver_actual": str(
                fluid_projection_report.get("pressure_solver", "")
            ),
            "pressure_solver_forced_to_fv_cg": bool(
                fluid_projection_report.get("pressure_solver_forced_to_fv_cg", False)
            ),
            "pressure_solver_force_reason": str(
                fluid_projection_report.get("pressure_solver_force_reason", "")
            ),
            "pressure_nullspace_policy": str(
                fluid_projection_report.get("pressure_nullspace_policy", "")
            ),
            "pressure_nullspace_compatibility_measured": bool(
                fluid_projection_report.get(
                    "pressure_nullspace_compatibility_measured",
                    False,
                )
            ),
            "pressure_nullspace_zero_mean_projection_applied": bool(
                fluid_projection_report.get(
                    "pressure_nullspace_zero_mean_projection_applied",
                    False,
                )
            ),
            "pressure_system_anchored_by_interface_matrix": bool(
                fluid_projection_report.get(
                    "pressure_system_anchored_by_interface_matrix",
                    False,
                )
            ),
            "pressure_interface_neumann_active_rows": _mapping_int(
                fluid_projection_report,
                "pressure_interface_neumann_active_rows",
            ),
            "hibm_post_dirichlet_consistency_projection_applied": bool(
                fluid_projection_report.get(
                    "hibm_post_dirichlet_consistency_projection_applied",
                    False,
                )
            ),
            "hibm_post_dirichlet_consistency_projection_count": _mapping_int(
                fluid_projection_report,
                "hibm_post_dirichlet_consistency_projection_count",
            ),
            "pressure_solve_failure_policy": str(
                fluid_projection_report.get("pressure_solve_failure_policy", "")
            ),
            "pressure_solve_failed": bool(
                fluid_projection_report.get("pressure_solve_failed", False)
            ),
            "pressure_solve_failure_action": str(
                fluid_projection_report.get("pressure_solve_failure_action", "")
            ),
            "pressure_projection_physical_failure": bool(
                fluid_projection_report.get(
                    "pressure_projection_physical_failure",
                    False,
                )
            ),
            "pressure_projection_physical_failure_reason": str(
                fluid_projection_report.get(
                    "pressure_projection_physical_failure_reason",
                    "",
                )
            ),
            "pressure_projection_physical_failure_action": str(
                fluid_projection_report.get(
                    "pressure_projection_physical_failure_action",
                    "",
                )
            ),
            "hibm_unreached_incompatible_component_count": _mapping_int(
                fluid_projection_report,
                "hibm_unreached_incompatible_component_count",
            ),
            "hibm_unreached_component_raw_count": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_raw_count",
            ),
            "hibm_unreached_largest_component_cell_count": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_largest_cell_count",
            ),
            "hibm_unreached_singleton_component_count": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_singleton_count",
            ),
            "hibm_unreached_small_component_threshold_cells": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_small_threshold_cells",
            ),
            "hibm_unreached_small_component_count": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_small_count",
            ),
            "hibm_unreached_small_component_cell_count": _mapping_int(
                fluid_projection_report,
                "cg_unreached_component_small_cell_count",
            ),
            "hibm_projection_overflow_singleton_cleanup_cell_count": (
                _mapping_int(
                    fluid_projection_report,
                    "hibm_projection_overflow_singleton_cleanup_cell_count",
                )
            ),
            "hibm_projection_overflow_singleton_cleanup_component_count": (
                _mapping_int(
                    fluid_projection_report,
                    "hibm_projection_overflow_singleton_cleanup_component_count",
                )
            ),
            "hibm_projection_tiny_unreached_cleanup_cell_count": (
                _mapping_int(
                    fluid_projection_report,
                    "hibm_projection_tiny_unreached_cleanup_cell_count",
                )
            ),
            "hibm_projection_tiny_unreached_cleanup_component_count": (
                _mapping_int(
                    fluid_projection_report,
                    "hibm_projection_tiny_unreached_cleanup_component_count",
                )
            ),
            "hibm_unreached_component_rhs_mean_max_abs": _mapping_float(
                fluid_projection_report,
                "hibm_unreached_component_rhs_mean_max_abs",
            ),
            "hibm_unreached_component_rhs_integral_max_abs": _mapping_float(
                fluid_projection_report,
                "hibm_unreached_component_rhs_integral_max_abs",
            ),
            "pressure_projection_cg_iterations_total": _mapping_int(
                fluid_projection_report,
                "cg_iterations_total",
            ),
            "pressure_projection_cg_iterations_max": _mapping_int(
                fluid_projection_report,
                "cg_iterations_max",
            ),
            "pressure_projection_cg_host_residual_checks": _mapping_int(
                fluid_projection_report,
                "cg_host_residual_checks",
            ),
            "pressure_projection_cg_mean_projection_count": _mapping_int(
                fluid_projection_report,
                "cg_mean_projection_count",
            ),
            "pressure_projection_cg_restart_count": _mapping_int(
                fluid_projection_report,
                "cg_restart_count",
            ),
            "pressure_projection_cg_restart_count_measured": bool(
                fluid_projection_report.get("cg_restart_count_measured", False)
            ),
            "pressure_projection_cg_restart_policy": str(
                fluid_projection_report.get("cg_restart_policy", "")
            ),
            "pressure_projection_cg_converged_all": bool(
                fluid_projection_report.get("cg_converged_all", True)
            ),
            "pressure_projection_cg_max_relative_residual": _mapping_float(
                fluid_projection_report,
                "cg_relative_residual_max",
            ),
            "pressure_projection_cg_max_initial_relative_residual": _mapping_float(
                fluid_projection_report,
                "cg_initial_relative_residual_max",
            ),
            "pressure_projection_cg_breakdown_count": _mapping_int(
                fluid_projection_report,
                "cg_breakdown_count",
            ),
            "pressure_projection_cg_breakdown_code": _mapping_int(
                fluid_projection_report,
                "cg_breakdown_code",
            ),
            "pressure_projection_cg_breakdown_dAd": _mapping_float(
                fluid_projection_report,
                "cg_breakdown_dAd",
            ),
            "pressure_interface_matrix_rhs_integral": _mapping_float(
                sharp_summary,
                "hibm_pressure_neumann_rhs_integral",
            ),
            "pressure_interface_matrix_active_cells": _mapping_int(
                sharp_summary,
                "hibm_pressure_neumann_active_rows",
            ),
            "pressure_projection_interface_matrix_diagonal_integral": _mapping_float(
                fluid_projection_report,
                "pressure_interface_matrix_diagonal_integral",
            ),
            "pressure_projection_interface_matrix_rhs_integral": _mapping_float(
                fluid_projection_report,
                "pressure_interface_matrix_rhs_integral",
            ),
            "pressure_projection_interface_matrix_max_abs_diagonal": _mapping_float(
                fluid_projection_report,
                "pressure_interface_matrix_max_abs_diagonal",
            ),
            "pressure_projection_interface_matrix_active_cells": _mapping_int(
                fluid_projection_report,
                "pressure_interface_matrix_active_cells",
            ),
            "pressure_projection_interface_matrix_active": bool(
                fluid_projection_report.get("pressure_interface_matrix_active", False)
            ),
            "pressure_projection_interface_matrix_row_count": _mapping_int(
                fluid_projection_report,
                "pressure_interface_matrix_row_count",
            ),
            "pressure_projection_interface_matrix_row_active_count": _mapping_int(
                fluid_projection_report,
                "pressure_interface_matrix_row_active_count",
            ),
            "pressure_projection_interface_matrix_row_invalid_count": _mapping_int(
                fluid_projection_report,
                "pressure_interface_matrix_row_invalid_count",
            ),
            "pressure_projection_interface_matrix_row_overflow_count": _mapping_int(
                fluid_projection_report,
                "pressure_interface_matrix_row_overflow_count",
            ),
            "pressure_projection_interface_matrix_row_diagonal_integral": (
                _mapping_float(
                    fluid_projection_report,
                    "pressure_interface_matrix_row_diagonal_integral",
                )
            ),
            "pressure_projection_interface_matrix_row_diagonal_abs_mismatch": (
                _mapping_float(
                    fluid_projection_report,
                    "pressure_interface_matrix_row_diagonal_integral_abs_mismatch",
                )
            ),
            "pressure_projection_interface_matrix_row_diagonal_max_abs_density_mismatch": (
                _mapping_float(
                    fluid_projection_report,
                    "pressure_interface_matrix_row_diagonal_max_abs_density_mismatch",
                )
            ),
            "pressure_projection_interface_matrix_row_max_transmissibility": (
                _mapping_float(
                    fluid_projection_report,
                    "pressure_interface_matrix_row_max_transmissibility",
                )
            ),
            "solid_mpm_particle_count": int(solid_mpm_report.particle_count),
            "solid_mpm_active_grid_nodes": int(solid_mpm_report.active_grid_nodes),
            "solid_mpm_grid_out_of_bounds_particle_count": int(
                solid_mpm_report.grid_out_of_bounds_particle_count
            ),
            "solid_mpm_particle_spacing_m": float(
                solid_mpm_report.particle_spacing_m
            ),
            "solid_mpm_grid_dx_m": float(solid_mpm_report.grid_spacing_m[0]),
            "solid_mpm_grid_dy_m": float(solid_mpm_report.grid_spacing_m[1]),
            "solid_mpm_grid_dz_m": float(solid_mpm_report.grid_spacing_m[2]),
            "solid_mpm_total_mass_kg": float(solid_mpm_report.total_mass_kg),
            "solid_mpm_particle_momentum_x_kg_mps": float(
                solid_mpm_report.particle_momentum_kg_mps[0]
            ),
            "solid_mpm_particle_momentum_y_kg_mps": float(
                solid_mpm_report.particle_momentum_kg_mps[1]
            ),
            "solid_mpm_particle_momentum_z_kg_mps": float(
                solid_mpm_report.particle_momentum_kg_mps[2]
            ),
            "solid_mpm_grid_momentum_x_kg_mps": float(
                solid_mpm_report.grid_momentum_kg_mps[0]
            ),
            "solid_mpm_grid_momentum_y_kg_mps": float(
                solid_mpm_report.grid_momentum_kg_mps[1]
            ),
            "solid_mpm_grid_momentum_z_kg_mps": float(
                solid_mpm_report.grid_momentum_kg_mps[2]
            ),
            "solid_mpm_transfer_relative_error": float(
                solid_mpm_report.transfer_relative_error
            ),
            "solid_mpm_max_speed_mps": float(solid_mpm_report.max_speed_mps),
            "solid_mpm_total_force_x_n": solid_mpm_total_force_n[0],
            "solid_mpm_total_force_y_n": solid_mpm_total_force_n[1],
            "solid_mpm_total_force_z_n": solid_mpm_total_force_n[2],
        }
    )
    if solid_model == "neo_hookean_mpm":
        row["solid_mpm_max_abs_j"] = float(solid_mpm_report.max_abs_j)
    return row


def signed_positive_source_flux_ratio(
    *,
    outlet_negative_z_flux_m3s: float,
    source_flux_m3s: float,
    min_source_flux_m3s: float = 1.0e-18,
) -> float:
    outlet_flux = float(outlet_negative_z_flux_m3s)
    source_flux = float(source_flux_m3s)
    if (
        not math.isfinite(outlet_flux)
        or not math.isfinite(source_flux)
        or source_flux <= float(min_source_flux_m3s)
    ):
        return 0.0
    return outlet_flux / source_flux
