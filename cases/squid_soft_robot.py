import argparse
import csv
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path

import numpy as np
import taichi as ti


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core import (
    CG_PRECONDITIONER_CHOICES,
    CartesianGrid,
    CartesianFluidSolver,
    FSI_COUPLING_MODE_CHOICES,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FluidDomainSpec,
    GradedGridSpec,
    HibmMpmSharpCouplingState,
    INTERFACE_REACTION_SOLVER_CHOICES,
    InterfaceReactionRelaxationState,
    InterfaceReactionTargetEvaluation,
    NeoHookeanMpmState,
    ProjectedIbmRegionPairStepConfig,
    RefinementRegion,
    SurfaceMesh,
    TaichiRuntimeConfig,
    TriMooneyShellMpmState,
    TriSurfaceRegionDiagnostics,
    action_reaction_balance,
    advance_projected_ibm_region_pair_fluid_step,
    boundary_drive_compliance_report,
    build_graded_grid,
    checks_passed,
    finite_field_diagnostics,
    hibm_mpm_sharp_step_summary,
    require_implemented_fsi_coupling_mode,
    robin_neumann_impedance_force,
    solve_and_apply_interface_reaction_step,
    update_interface_reaction_for_next_step,
    vector_norm,
)
from simulation_core.hyperelastic import ecoflex_0010_material
from simulation_core.runtime import init_taichi


DEFAULT_SOURCE_CONFIG = str(
    Path("_diagnostic_runs")
    / "supportdiag_region4_008step_outflowguard_finalsamples_debugdl2_20260602"
    / "simulation_config.json"
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
    "pressure_outlet_velocity_flux_m3s",
    "pressure_outlet_velocity_to_source_ratio",
    "pressure_outlet_pressure_flux_m3s",
    "pressure_outlet_pressure_to_source_ratio",
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

PRESSURE_SOLVER_CHOICES = ("auto", "jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg")
PRESSURE_SOLVE_FAILURE_POLICY_CHOICES = ("raise", "report")
FLUID_ADVECTION_SCHEME_CHOICES = ("euler", "rk2")
INTERFACE_REACTION_ROBIN_TARGET_CHOICES = ("stabilized", "physical")

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
    "hibm_pressure_disconnected_nonprojectable_cell_count",
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
    "hibm_no_slip_residual_max_mps",
    "hibm_no_slip_residual_l2_mps",
    "hibm_full_stress_valid_marker_count",
    "hibm_full_stress_invalid_marker_count",
    "hibm_full_stress_max_abs_traction_pa",
    "hibm_marker_primary_count",
    "hibm_marker_secondary_count",
    "hibm_marker_total_count",
    "hibm_marker_total_force_x_n",
    "hibm_marker_total_force_y_n",
    "hibm_marker_total_force_z_n",
    "hibm_marker_action_reaction_residual_n",
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
    "fsi_added_mass_stability_measured",
    "fsi_semi_implicit_coupling_enabled",
    "fsi_semi_implicit_coupling_matrix_active",
    "fsi_action_reaction_residual_abs_n",
    "fsi_coupling_residual_norm_n",
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
    "solid_mpm_transfer_relative_error",
    "solid_mpm_max_speed_mps",
    "solid_mpm_grid_out_of_bounds_particle_count",
    "solid_mpm_total_force_x_n",
    "solid_mpm_total_force_y_n",
    "solid_mpm_total_force_z_n",
)

RUN_CHECKPOINT_VERSION = 2
RUN_CHECKPOINT_FILENAME = "run_checkpoint.npz"
CHECKPOINT_ARG_FINGERPRINT_FIELDS = (
    "source_config",
    "steps_explicit",
    "projection_iterations",
    "fluid_advection_scheme",
    "pressure_solver",
    "pressure_solve_failure_policy",
    "cg_preconditioner",
    "cg_tolerance",
    "multigrid_cycles",
    "divergence_cleanup_iterations",
    "divergence_cleanup_relaxation",
    "projection_divergence_tolerance",
    "grid_scale",
    "use_graded_grid",
    "graded_grid_target_spacing_m",
    "graded_grid_farfield_spacing_m",
    "graded_grid_growth_ratio",
    "graded_grid_max_cells",
    "use_tail_refinement",
    "tail_refinement_target_spacing_m",
    "tail_refinement_padding_m",
    "time_step_scale",
    "solid_model",
    "solid_mpm_layers",
    "solid_mpm_substeps",
    "membrane_thickness_scale",
    "solid_density_scale",
    "solid_mpm_cfl",
    "solid_mpm_velocity_damping",
    "solid_mpm_flip_blend",
    "mooney_membrane_force_scale",
    "poissons_ratio",
    "constraint_force_scale",
    "fsi_constraint_force_solid_mobility_ratio",
    "fsi_solid_response_mobility_coupling",
    "fsi_velocity_target_solid_mobility_ratio",
    "fsi_solid_response_velocity_mobility_coupling",
    "fsi_velocity_constraint_blend",
    "fsi_velocity_constraint_solid_mobility_ratio",
    "interface_reaction_relaxation",
    "interface_reaction_aitken",
    "interface_reaction_passivity_limit",
    "interface_reaction_robin_impedance_ns_m",
    "interface_reaction_robin_matrix_impedance_ns_m",
    "interface_reaction_robin_target_mode",
    "fsi_coupling_mode",
    "fsi_coupling_solver",
    "fsi_coupling_target_map_relaxation",
    "reuse_accepted_fsi_trial_state",
    "min_outlet_to_main_volume_flux_ratio",
    "pressure_outlet_source_ratio_tolerance",
    "fluid_substeps",
    "ibm_correction_iterations",
    "fsi_coupling_iterations",
    "fsi_coupling_tolerance_n",
    "disable_pressure_outlet_zmin",
    "disable_reduced_obstacles",
    "use_region14_aperture_carve",
    "open_downstream_farfield",
    "use_nozzle_taper",
    "nozzle_taper_length_m",
    "nozzle_taper_inlet_radius_m",
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


@dataclass(frozen=True)
class SquidReducedSpec:
    source_config_path: str
    fluid_bounds_min_m: tuple[float, float, float]
    fluid_bounds_max_m: tuple[float, float, float]
    grid_nodes: tuple[int, int, int]
    dt_s: float
    water_density_kgm3: float
    water_viscosity_pa_s: float
    base_dt_s: float | None = None
    main_membrane_side_m: float = 64.0e-3
    main_membrane_thickness_m: float = 3.0e-3
    tail_membrane_side_m: float = 60.0e-3
    tail_membrane_thickness_m: float = 2.5e-3
    nozzle_radius_m: float = 3.0e-3
    nozzle_length_m: float = 26.254e-3
    main_added_mass_length_m: float = 84.5e-3
    tail_added_mass_length_m: float = 41.87e-3
    damping_multiplier: float = 2.5
    chamber_radius_m: float = 39.0e-3
    chamber_z_min_m: float = 1.0
    chamber_z_max_m: float = 1.04
    nozzle_z_max_m: float = 1.009626
    outlet_plume_radius_m: float = 6.0e-3
    monitor_center_x_m: float = -0.031311
    monitor_center_y_m: float = 0.015907
    monitor_radius_m: float = 3.0e-3
    lip_z_m: float = 0.967754
    outlet_z_m: float = 0.9565
    downstream_z_m: float = 0.9415
    pressure_t0_s: float = 0.0
    pressure_t1_s: float = 1.0
    pressure_t2_s: float = 2.0
    pressure_p0_pa: float = 0.0
    pressure_p1_pa: float = 8000.0
    pressure_p2_pa: float = -8000.0
    downstream_farfield_open_enabled: bool = False
    downstream_farfield_open_z_max_m: float = 0.967754
    nozzle_taper_enabled: bool = False
    nozzle_taper_length_m: float = 0.0
    nozzle_taper_inlet_radius_m: float | None = None
    cartesian_grid: CartesianGrid | None = None
    graded_grid: GradedGridSpec | None = None

    @property
    def main_area_m2(self) -> float:
        return self.main_membrane_side_m * self.main_membrane_side_m

    @property
    def tail_area_m2(self) -> float:
        return self.tail_membrane_side_m * self.tail_membrane_side_m

    @property
    def nozzle_area_m2(self) -> float:
        return math.pi * self.nozzle_radius_m * self.nozzle_radius_m


@ti.data_oriented
class ReducedSquidFSI:
    def __init__(
        self,
        spec: SquidReducedSpec,
        runtime: TaichiRuntimeConfig,
    ):
        init_taichi(runtime)
        self.spec = spec
        self.fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=spec.fluid_bounds_min_m,
                bounds_max_m=spec.fluid_bounds_max_m,
                grid_nodes=spec.grid_nodes,
                density_kgm3=spec.water_density_kgm3,
                viscosity_pa_s=spec.water_viscosity_pa_s,
                dt_s=spec.dt_s,
                cartesian_grid=spec.cartesian_grid,
                graded_grid=spec.graded_grid,
            ),
            runtime=runtime,
        )

        self.time_s = ti.field(dtype=ti.f32, shape=())
        self.pressure_load_pa = ti.field(dtype=ti.f32, shape=())
        self.hydraulic_pressure_pa = ti.field(dtype=ti.f32, shape=())
        self.main_w_m = ti.field(dtype=ti.f32, shape=())
        self.main_v_mps = ti.field(dtype=ti.f32, shape=())
        self.tail_w_m = ti.field(dtype=ti.f32, shape=())
        self.tail_v_mps = ti.field(dtype=ti.f32, shape=())
        self.primary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.secondary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.volume_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.nozzle_velocity_z_mps = ti.field(dtype=ti.f32, shape=())
        self.max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.lip_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.outlet_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.downstream_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.lip_sample_count = ti.field(dtype=ti.i32, shape=())
        self.outlet_sample_count = ti.field(dtype=ti.i32, shape=())
        self.downstream_sample_count = ti.field(dtype=ti.i32, shape=())
        self.sample_report_float_snapshot = ti.Vector.field(15, dtype=ti.f32, shape=())
        self.sample_report_count_snapshot = ti.Vector.field(3, dtype=ti.i32, shape=())
        self.sample_report_host_snapshot = ti.field(dtype=ti.f64, shape=18)
        self.saved_time_s = ti.field(dtype=ti.f32, shape=())
        self.saved_pressure_load_pa = ti.field(dtype=ti.f32, shape=())
        self.saved_hydraulic_pressure_pa = ti.field(dtype=ti.f32, shape=())
        self.saved_main_w_m = ti.field(dtype=ti.f32, shape=())
        self.saved_main_v_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_tail_w_m = ti.field(dtype=ti.f32, shape=())
        self.saved_tail_v_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_primary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.saved_secondary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.saved_volume_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_nozzle_velocity_z_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_lip_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_outlet_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_downstream_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_lip_sample_count = ti.field(dtype=ti.i32, shape=())
        self.saved_outlet_sample_count = ti.field(dtype=ti.i32, shape=())
        self.saved_downstream_sample_count = ti.field(dtype=ti.i32, shape=())
        self.last_sample_report_host_reads = 0

        self._reset_kernel()

    @ti.kernel
    def _reset_kernel(self):
        self.time_s[None] = 0.0
        self.pressure_load_pa[None] = 0.0
        self.hydraulic_pressure_pa[None] = 0.0
        self.main_w_m[None] = 0.0
        self.main_v_mps[None] = 0.0
        self.tail_w_m[None] = 0.0
        self.tail_v_mps[None] = 0.0
        self.primary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.secondary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.volume_flux_m3s[None] = 0.0
        self.nozzle_velocity_z_mps[None] = 0.0
        self.max_speed_mps[None] = 0.0
        self.lip_flow_z_m3s[None] = 0.0
        self.outlet_flow_z_m3s[None] = 0.0
        self.downstream_flow_z_m3s[None] = 0.0
        self.lip_sample_count[None] = 0
        self.outlet_sample_count[None] = 0
        self.downstream_sample_count[None] = 0
        self.sample_report_float_snapshot[None] = ti.Vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.sample_report_count_snapshot[None] = ti.Vector([0, 0, 0])
        for index in ti.static(range(18)):
            self.sample_report_host_snapshot[index] = 0.0
        self.saved_time_s[None] = 0.0
        self.saved_pressure_load_pa[None] = 0.0
        self.saved_hydraulic_pressure_pa[None] = 0.0
        self.saved_main_w_m[None] = 0.0
        self.saved_main_v_mps[None] = 0.0
        self.saved_tail_w_m[None] = 0.0
        self.saved_tail_v_mps[None] = 0.0
        self.saved_primary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.saved_secondary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.saved_volume_flux_m3s[None] = 0.0
        self.saved_nozzle_velocity_z_mps[None] = 0.0
        self.saved_max_speed_mps[None] = 0.0
        self.saved_lip_flow_z_m3s[None] = 0.0
        self.saved_outlet_flow_z_m3s[None] = 0.0
        self.saved_downstream_flow_z_m3s[None] = 0.0
        self.saved_lip_sample_count[None] = 0
        self.saved_outlet_sample_count[None] = 0
        self.saved_downstream_sample_count[None] = 0

    @ti.kernel
    def save_reduced_state_kernel(self):
        self.saved_time_s[None] = self.time_s[None]
        self.saved_pressure_load_pa[None] = self.pressure_load_pa[None]
        self.saved_hydraulic_pressure_pa[None] = self.hydraulic_pressure_pa[None]
        self.saved_main_w_m[None] = self.main_w_m[None]
        self.saved_main_v_mps[None] = self.main_v_mps[None]
        self.saved_tail_w_m[None] = self.tail_w_m[None]
        self.saved_tail_v_mps[None] = self.tail_v_mps[None]
        self.saved_primary_interface_reaction_force_n[None] = self.primary_interface_reaction_force_n[None]
        self.saved_secondary_interface_reaction_force_n[None] = self.secondary_interface_reaction_force_n[None]
        self.saved_volume_flux_m3s[None] = self.volume_flux_m3s[None]
        self.saved_nozzle_velocity_z_mps[None] = self.nozzle_velocity_z_mps[None]
        self.saved_max_speed_mps[None] = self.max_speed_mps[None]
        self.saved_lip_flow_z_m3s[None] = self.lip_flow_z_m3s[None]
        self.saved_outlet_flow_z_m3s[None] = self.outlet_flow_z_m3s[None]
        self.saved_downstream_flow_z_m3s[None] = self.downstream_flow_z_m3s[None]
        self.saved_lip_sample_count[None] = self.lip_sample_count[None]
        self.saved_outlet_sample_count[None] = self.outlet_sample_count[None]
        self.saved_downstream_sample_count[None] = self.downstream_sample_count[None]

    @ti.kernel
    def restore_reduced_state_kernel(self):
        self.time_s[None] = self.saved_time_s[None]
        self.pressure_load_pa[None] = self.saved_pressure_load_pa[None]
        self.hydraulic_pressure_pa[None] = self.saved_hydraulic_pressure_pa[None]
        self.main_w_m[None] = self.saved_main_w_m[None]
        self.main_v_mps[None] = self.saved_main_v_mps[None]
        self.tail_w_m[None] = self.saved_tail_w_m[None]
        self.tail_v_mps[None] = self.saved_tail_v_mps[None]
        self.primary_interface_reaction_force_n[None] = self.saved_primary_interface_reaction_force_n[None]
        self.secondary_interface_reaction_force_n[None] = self.saved_secondary_interface_reaction_force_n[None]
        self.volume_flux_m3s[None] = self.saved_volume_flux_m3s[None]
        self.nozzle_velocity_z_mps[None] = self.saved_nozzle_velocity_z_mps[None]
        self.max_speed_mps[None] = self.saved_max_speed_mps[None]
        self.lip_flow_z_m3s[None] = self.saved_lip_flow_z_m3s[None]
        self.outlet_flow_z_m3s[None] = self.saved_outlet_flow_z_m3s[None]
        self.downstream_flow_z_m3s[None] = self.saved_downstream_flow_z_m3s[None]
        self.lip_sample_count[None] = self.saved_lip_sample_count[None]
        self.outlet_sample_count[None] = self.saved_outlet_sample_count[None]
        self.downstream_sample_count[None] = self.saved_downstream_sample_count[None]

    def save_reduced_state(self) -> None:
        self.save_reduced_state_kernel()

    def restore_reduced_state(self) -> None:
        self.restore_reduced_state_kernel()

    @ti.kernel
    def set_interface_reaction_kernel(
        self,
        primary_force_n: ti.types.vector(3, ti.f32),
        secondary_force_n: ti.types.vector(3, ti.f32),
    ):
        self.primary_interface_reaction_force_n[None] = primary_force_n
        self.secondary_interface_reaction_force_n[None] = secondary_force_n

    def set_interface_reaction(
        self,
        *,
        primary_force_n: Sequence[float],
        secondary_force_n: Sequence[float],
    ) -> None:
        primary = _vector3(primary_force_n, name="primary_force_n")
        secondary = _vector3(secondary_force_n, name="secondary_force_n")
        self.set_interface_reaction_kernel(
            ti.Vector(primary),
            ti.Vector(secondary),
        )

    @ti.kernel
    def set_structure_state_kernel(
        self,
        time_s: ti.f32,
        pressure_pa: ti.f32,
        hydraulic_pressure_pa: ti.f32,
        main_displacement_z_m: ti.f32,
        main_velocity_z_mps: ti.f32,
        tail_displacement_z_m: ti.f32,
        tail_velocity_z_mps: ti.f32,
        volume_flux_m3s: ti.f32,
        nozzle_velocity_z_mps: ti.f32,
    ):
        self.time_s[None] = time_s
        self.pressure_load_pa[None] = pressure_pa
        self.hydraulic_pressure_pa[None] = hydraulic_pressure_pa
        self.main_w_m[None] = main_displacement_z_m
        self.main_v_mps[None] = main_velocity_z_mps
        self.tail_w_m[None] = tail_displacement_z_m
        self.tail_v_mps[None] = tail_velocity_z_mps
        self.volume_flux_m3s[None] = volume_flux_m3s
        self.nozzle_velocity_z_mps[None] = nozzle_velocity_z_mps

    def set_structure_state(
        self,
        *,
        time_s: float,
        pressure_pa: float,
        hydraulic_pressure_pa: float,
        main_displacement_z_m: float,
        main_velocity_z_mps: float,
        tail_displacement_z_m: float,
        tail_velocity_z_mps: float,
        volume_flux_m3s: float,
        nozzle_velocity_z_mps: float,
    ) -> None:
        self.set_structure_state_kernel(
            float(time_s),
            float(pressure_pa),
            float(hydraulic_pressure_pa),
            float(main_displacement_z_m),
            float(main_velocity_z_mps),
            float(tail_displacement_z_m),
            float(tail_velocity_z_mps),
            float(volume_flux_m3s),
            float(nozzle_velocity_z_mps),
        )

    @ti.func
    def _cell_disk_intersects_axisymmetric_region(
        self,
        rx: ti.f32,
        ry: ti.f32,
        half_width_x_m: ti.f32,
        half_width_y_m: ti.f32,
        radius_m: ti.f32,
    ) -> ti.i32:
        closest_x = ti.max(ti.abs(rx) - half_width_x_m, 0.0)
        closest_y = ti.max(ti.abs(ry) - half_width_y_m, 0.0)
        intersects = closest_x * closest_x + closest_y * closest_y <= radius_m * radius_m
        return ti.cast(intersects, ti.i32)

    @ti.func
    def _cell_z_interval_intersects(
        self,
        cell_min_z_m: ti.f32,
        cell_max_z_m: ti.f32,
        lower_z_m: ti.f32,
        upper_z_m: ti.f32,
    ) -> ti.i32:
        return ti.cast(cell_max_z_m >= lower_z_m and cell_min_z_m <= upper_z_m, ti.i32)

    @ti.func
    def _conservative_taper_radius_m(
        self,
        cell_min_z_m: ti.f32,
        cell_max_z_m: ti.f32,
        nozzle_radius_m: ti.f32,
        nozzle_taper_enabled: ti.i32,
        nozzle_taper_start_z_m: ti.f32,
        nozzle_taper_end_z_m: ti.f32,
        nozzle_taper_inlet_radius_m: ti.f32,
    ) -> ti.f32:
        radius_m = nozzle_radius_m
        if (
            nozzle_taper_enabled == 1
            and cell_max_z_m >= nozzle_taper_start_z_m
            and cell_min_z_m <= nozzle_taper_end_z_m
        ):
            overlap_hi_z_m = ti.min(cell_max_z_m, nozzle_taper_end_z_m)
            fraction = (overlap_hi_z_m - nozzle_taper_start_z_m) / ti.max(
                nozzle_taper_end_z_m - nozzle_taper_start_z_m,
                1.0e-12,
            )
            radius_m = nozzle_radius_m + (
                nozzle_taper_inlet_radius_m - nozzle_radius_m
            ) * ti.min(ti.max(fraction, 0.0), 1.0)
        return radius_m

    @ti.kernel
    def mark_reduced_squid_water_domain_kernel(
        self,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        bounds_min_z: ti.f32,
        center_x_m: ti.f32,
        center_y_m: ti.f32,
        chamber_radius_m: ti.f32,
        chamber_z_min_m: ti.f32,
        chamber_z_max_m: ti.f32,
        nozzle_radius_m: ti.f32,
        nozzle_z_max_m: ti.f32,
        downstream_z_m: ti.f32,
        outlet_plume_radius_m: ti.f32,
        nozzle_taper_enabled: ti.i32,
        nozzle_taper_start_z_m: ti.f32,
        nozzle_taper_end_z_m: ti.f32,
        nozzle_taper_inlet_radius_m: ti.f32,
        downstream_farfield_open_enabled: ti.i32,
        downstream_farfield_open_z_max_m: ti.f32,
    ):
        for i, j, k in self.fluid.obstacle:
            x = cell_center_x_m[i]
            y = cell_center_y_m[j]
            z = cell_center_z_m[k]
            rx = x - center_x_m
            ry = y - center_y_m
            half_width_x_m = 0.5 * cell_width_x_m[i]
            half_width_y_m = 0.5 * cell_width_y_m[j]
            half_width_z_m = 0.5 * cell_width_z_m[k]
            cell_min_z_m = z - half_width_z_m
            cell_max_z_m = z + half_width_z_m
            chamber = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    chamber_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    chamber_z_min_m,
                    chamber_z_max_m,
                ) == 1
            )
            local_nozzle_radius_m = nozzle_radius_m
            local_nozzle_radius_m = self._conservative_taper_radius_m(
                cell_min_z_m,
                cell_max_z_m,
                nozzle_radius_m,
                nozzle_taper_enabled,
                nozzle_taper_start_z_m,
                nozzle_taper_end_z_m,
                nozzle_taper_inlet_radius_m,
            )
            nozzle = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    local_nozzle_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    downstream_z_m,
                    nozzle_z_max_m,
                ) == 1
            )
            outlet_plume = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    outlet_plume_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    bounds_min_z,
                    downstream_z_m,
                ) == 1
            )
            downstream_farfield = (
                downstream_farfield_open_enabled == 1
                and cell_min_z_m <= downstream_farfield_open_z_max_m
            )
            self.fluid.obstacle[i, j, k] = 0 if chamber or nozzle or outlet_plume or downstream_farfield else 1

    def mark_reduced_squid_water_domain(self) -> None:
        spec = self.spec
        taper_start_z_m, taper_end_z_m, taper_inlet_radius_m = nozzle_taper_geometry(spec)
        self.mark_reduced_squid_water_domain_kernel(
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.cell_width_x_m,
            self.fluid.cell_width_y_m,
            self.fluid.cell_width_z_m,
            float(spec.fluid_bounds_min_m[2]),
            float(spec.monitor_center_x_m),
            float(spec.monitor_center_y_m),
            float(spec.chamber_radius_m),
            float(spec.chamber_z_min_m),
            float(spec.chamber_z_max_m),
            float(spec.nozzle_radius_m),
            float(spec.nozzle_z_max_m),
            float(spec.downstream_z_m),
            float(spec.outlet_plume_radius_m),
            1 if spec.nozzle_taper_enabled else 0,
            float(taper_start_z_m),
            float(taper_end_z_m),
            float(taper_inlet_radius_m),
            1 if spec.downstream_farfield_open_enabled else 0,
            float(spec.downstream_farfield_open_z_max_m),
        )

    @ti.func
    def _section_area_fraction(
        self,
        rx,
        ry,
        half_width_x_m,
        half_width_y_m,
        radius_m,
    ):
        hits = 0
        for sx, sy in ti.ndrange(8, 8):
            sample_x = rx + (-half_width_x_m + (ti.cast(sx, ti.f32) + 0.5) * half_width_x_m / 4.0)
            sample_y = ry + (-half_width_y_m + (ti.cast(sy, ti.f32) + 0.5) * half_width_y_m / 4.0)
            if sample_x * sample_x + sample_y * sample_y <= radius_m * radius_m:
                hits += 1
        return ti.cast(hits, ti.f32) / 64.0

    @ti.func
    def _accumulate_section(
        self,
        velocity_z,
        z,
        target_z,
        dz,
        rx,
        ry,
        radius_m,
        cell_area_m2,
        cell_width_x_m,
        cell_width_y_m,
        section_id,
    ):
        if ti.abs(z - target_z) <= 0.5 * dz:
            area_fraction = self._section_area_fraction(
                rx,
                ry,
                0.5 * cell_width_x_m,
                0.5 * cell_width_y_m,
                radius_m,
            )
            section_area_m2 = cell_area_m2 * area_fraction
            section_flux_m3s = velocity_z * section_area_m2
            if section_id == 0:
                ti.atomic_add(self.lip_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.lip_sample_count[None], 1)
            elif section_id == 1:
                ti.atomic_add(self.outlet_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.outlet_sample_count[None], 1)
            else:
                ti.atomic_add(self.downstream_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.downstream_sample_count[None], 1)

    @ti.kernel
    def sample_sections_kernel(
        self,
        velocity: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_x_m: ti.f32,
        center_y_m: ti.f32,
        lip_radius_m: ti.f32,
        outlet_radius_m: ti.f32,
        downstream_radius_m: ti.f32,
        lip_z_m: ti.f32,
        outlet_z_m: ti.f32,
        downstream_z_m: ti.f32,
    ):
        self.lip_flow_z_m3s[None] = 0.0
        self.outlet_flow_z_m3s[None] = 0.0
        self.downstream_flow_z_m3s[None] = 0.0
        self.lip_sample_count[None] = 0
        self.outlet_sample_count[None] = 0
        self.downstream_sample_count[None] = 0
        self.max_speed_mps[None] = 0.0
        for i, j, k in velocity:
            if self.fluid.obstacle[i, j, k] == 0:
                x = cell_center_x_m[i]
                y = cell_center_y_m[j]
                z = cell_center_z_m[k]
                rx = x - center_x_m
                ry = y - center_y_m
                vz = velocity[i, j, k].z
                dz = cell_width_z_m[k]
                cell_width_x = cell_width_x_m[i]
                cell_width_y = cell_width_y_m[j]
                cell_area_m2 = cell_width_x_m[i] * cell_width_y_m[j]
                self._accumulate_section(
                    vz,
                    z,
                    lip_z_m,
                    dz,
                    rx,
                    ry,
                    lip_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    0,
                )
                self._accumulate_section(
                    vz,
                    z,
                    outlet_z_m,
                    dz,
                    rx,
                    ry,
                    outlet_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    1,
                )
                self._accumulate_section(
                    vz,
                    z,
                    downstream_z_m,
                    dz,
                    rx,
                    ry,
                    downstream_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    2,
                )
                ti.atomic_max(self.max_speed_mps[None], velocity[i, j, k].norm())
        self.sample_report_float_snapshot[None] = ti.Vector(
            [
                self.time_s[None],
                self.pressure_load_pa[None],
                self.hydraulic_pressure_pa[None],
                self.main_w_m[None],
                self.main_v_mps[None],
                self.tail_w_m[None],
                self.tail_v_mps[None],
                self.primary_interface_reaction_force_n[None].z,
                self.secondary_interface_reaction_force_n[None].z,
                self.volume_flux_m3s[None],
                self.nozzle_velocity_z_mps[None],
                self.lip_flow_z_m3s[None],
                self.outlet_flow_z_m3s[None],
                self.downstream_flow_z_m3s[None],
                self.max_speed_mps[None],
            ]
        )
        self.sample_report_count_snapshot[None] = ti.Vector(
            [
                self.lip_sample_count[None],
                self.outlet_sample_count[None],
                self.downstream_sample_count[None],
            ]
        )
        self.sample_report_host_snapshot[0] = ti.cast(self.time_s[None], ti.f64)
        self.sample_report_host_snapshot[1] = ti.cast(self.pressure_load_pa[None], ti.f64)
        self.sample_report_host_snapshot[2] = ti.cast(self.hydraulic_pressure_pa[None], ti.f64)
        self.sample_report_host_snapshot[3] = ti.cast(self.main_w_m[None], ti.f64)
        self.sample_report_host_snapshot[4] = ti.cast(self.main_v_mps[None], ti.f64)
        self.sample_report_host_snapshot[5] = ti.cast(self.tail_w_m[None], ti.f64)
        self.sample_report_host_snapshot[6] = ti.cast(self.tail_v_mps[None], ti.f64)
        self.sample_report_host_snapshot[7] = ti.cast(
            self.primary_interface_reaction_force_n[None].z,
            ti.f64,
        )
        self.sample_report_host_snapshot[8] = ti.cast(
            self.secondary_interface_reaction_force_n[None].z,
            ti.f64,
        )
        self.sample_report_host_snapshot[9] = ti.cast(self.volume_flux_m3s[None], ti.f64)
        self.sample_report_host_snapshot[10] = ti.cast(self.nozzle_velocity_z_mps[None], ti.f64)
        self.sample_report_host_snapshot[11] = ti.cast(self.lip_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[12] = ti.cast(self.outlet_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[13] = ti.cast(self.downstream_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[14] = ti.cast(self.max_speed_mps[None], ti.f64)
        self.sample_report_host_snapshot[15] = ti.cast(self.lip_sample_count[None], ti.f64)
        self.sample_report_host_snapshot[16] = ti.cast(self.outlet_sample_count[None], ti.f64)
        self.sample_report_host_snapshot[17] = ti.cast(self.downstream_sample_count[None], ti.f64)

    def project_and_sample(
        self,
        projection_iterations: int,
        pressure_outlet_zmin: bool,
    ) -> dict[str, object]:
        divergence = self.fluid.project(
            iterations=projection_iterations,
            pressure_outlet_zmin=pressure_outlet_zmin,
        )
        return self.sample_after_projection(divergence)

    def sample_after_projection(
        self,
        divergence: dict[str, float],
        *,
        dt_s: float | None = None,
    ) -> dict[str, object]:
        spec = self.spec
        self.sample_sections_kernel(
            self.fluid.velocity,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.cell_width_x_m,
            self.fluid.cell_width_y_m,
            self.fluid.cell_width_z_m,
            float(spec.monitor_center_x_m),
            float(spec.monitor_center_y_m),
            float(spec.monitor_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.lip_z_m),
            float(spec.outlet_z_m),
            float(spec.downstream_z_m),
        )
        h = min(cartesian_grid_axis_min_spacing_m(self.fluid.grid))
        sample_values = self.sample_report_float_snapshot[None]
        sample_counts = self.sample_report_count_snapshot[None]
        self.last_sample_report_host_reads = 1
        max_speed = float(sample_values[14])
        cfl_dt_s = float(self.spec.dt_s) if dt_s is None else float(dt_s)
        return {
            "time_s": float(sample_values[0]),
            "pressure_load_pa": float(sample_values[1]),
            "hydraulic_pressure_pa": float(sample_values[2]),
            "main_displacement_z_m": float(sample_values[3]),
            "main_velocity_z_mps": float(sample_values[4]),
            "tail_displacement_z_m": float(sample_values[5]),
            "tail_velocity_z_mps": float(sample_values[6]),
            "main_interface_reaction_z_n": float(sample_values[7]),
            "tail_interface_reaction_z_n": float(sample_values[8]),
            "volume_flux_m3s": float(sample_values[9]),
            "nozzle_velocity_z_mps": float(sample_values[10]),
            "lip_flow_z_m3s": float(sample_values[11]),
            "outlet_flow_z_m3s": float(sample_values[12]),
            "downstream_flow_z_m3s": float(sample_values[13]),
            "lip_flow_negative_z_m3s": -float(sample_values[11]),
            "outlet_flow_negative_z_m3s": -float(sample_values[12]),
            "downstream_flow_negative_z_m3s": -float(sample_values[13]),
            "lip_sample_count": int(sample_counts[0]),
            "outlet_sample_count": int(sample_counts[1]),
            "downstream_sample_count": int(sample_counts[2]),
            "max_fluid_speed_mps": max_speed,
            "cfl": max_speed * cfl_dt_s / max(h, 1.0e-12),
            **divergence_sample_report_fields(divergence),
        }


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


def load_source_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"source config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _mapping_config_float(
    mapping: Mapping[str, object],
    keys: Sequence[str],
    default: float,
    *,
    field: str,
) -> float:
    for key in keys:
        if key in mapping:
            value = float(mapping[key])
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            return value
    return float(default)


def pressure_schedule_from_config(
    config: Mapping[str, object],
    analysis: Mapping[str, object],
) -> dict[str, float]:
    defaults = {
        "pressure_t0_s": 0.0,
        "pressure_t1_s": 1.0,
        "pressure_t2_s": 2.0,
        "pressure_p0_pa": 0.0,
        "pressure_p1_pa": 8000.0,
        "pressure_p2_pa": -8000.0,
    }
    sources: list[Mapping[str, object]] = []
    top_schedule = config.get("pressure_schedule", {})
    if isinstance(top_schedule, Mapping):
        sources.append(top_schedule)
    sources.append(config)
    sources.append(analysis)
    analysis_schedule = analysis.get("pressure_schedule", {})
    if isinstance(analysis_schedule, Mapping):
        sources.append(analysis_schedule)

    schedule = dict(defaults)
    aliases = {
        "pressure_t0_s": ("pressure_t0_s", "t0_s"),
        "pressure_t1_s": ("pressure_t1_s", "t1_s"),
        "pressure_t2_s": ("pressure_t2_s", "t2_s"),
        "pressure_p0_pa": ("pressure_p0_pa", "p0_pa"),
        "pressure_p1_pa": ("pressure_p1_pa", "p1_pa"),
        "pressure_p2_pa": ("pressure_p2_pa", "p2_pa"),
    }
    for source in sources:
        schedule = {
            field: _mapping_config_float(
                source,
                aliases[field],
                value,
                field=field,
            )
            for field, value in schedule.items()
        }
    if not (
        schedule["pressure_t0_s"]
        < schedule["pressure_t1_s"]
        < schedule["pressure_t2_s"]
    ):
        raise ValueError("pressure schedule times must satisfy t0 < t1 < t2")
    return schedule


def _face_ids_for_region(config: dict[str, object], region_id: int) -> list[int]:
    selections = config.get("named_selections", [])
    if not isinstance(selections, list):
        return []
    for selection in selections:
        if isinstance(selection, dict) and int(selection.get("id", -1)) == int(region_id):
            values = selection.get("face_ids", [])
            if isinstance(values, list):
                return [int(value) for value in values]
    return []


def _vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (vector[0], vector[1], vector[2])


def _combine_region_pair_vectors(
    primary_vector: Sequence[float],
    secondary_vector: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    primary = _vector3(primary_vector, name="primary_vector")
    secondary = _vector3(secondary_vector, name="secondary_vector")
    return primary + secondary


def robin_previous_velocity_for_step(
    state: InterfaceReactionRelaxationState,
    step_start_velocity_mps: Sequence[float],
) -> tuple[float, ...]:
    step_start_velocity = tuple(float(value) for value in step_start_velocity_mps)
    if not step_start_velocity:
        raise ValueError("step_start_velocity_mps must contain at least one value")
    if any(not math.isfinite(value) for value in step_start_velocity):
        raise ValueError("step_start_velocity_mps must contain only finite values")
    if state.previous_velocity_mps is None:
        return step_start_velocity
    previous_velocity = tuple(float(value) for value in state.previous_velocity_mps)
    if len(previous_velocity) != len(step_start_velocity):
        raise ValueError("previous_velocity_mps and step_start_velocity_mps must match")
    if any(not math.isfinite(value) for value in previous_velocity):
        raise ValueError("previous_velocity_mps must contain only finite values")
    return previous_velocity


def interface_reaction_target_for_mode(
    mode: str,
    *,
    raw_target_force_n: Sequence[float],
    stabilized_target_force_n: Sequence[float],
) -> tuple[float, ...]:
    raw_target = tuple(float(value) for value in raw_target_force_n)
    stabilized_target = tuple(float(value) for value in stabilized_target_force_n)
    if not raw_target:
        raise ValueError("raw_target_force_n must contain at least one value")
    if len(raw_target) != len(stabilized_target):
        raise ValueError("raw_target_force_n and stabilized_target_force_n must match")
    if any(not math.isfinite(value) for value in raw_target + stabilized_target):
        raise ValueError("interface reaction targets must contain only finite values")
    target_mode = str(mode)
    if target_mode == "stabilized":
        return stabilized_target
    if target_mode == "physical":
        return raw_target
    choices = ", ".join(INTERFACE_REACTION_ROBIN_TARGET_CHOICES)
    raise ValueError(f"--interface-reaction-robin-target-mode must be one of: {choices}")


def legacy_projected_reduced_fsi_coupling_enabled(
    *,
    fsi_coupling_mode: str,
    solid_model: str,
    fsi_coupling_iterations: int,
) -> bool:
    if str(fsi_coupling_mode) != FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
        return False
    return (
        str(solid_model) in ("tri_mooney_shell_mpm", "neo_hookean_mpm")
        and int(fsi_coupling_iterations) > 1
    )


def raise_for_unsupported_hibm_mpm_sharp_robin_options(
    *,
    fsi_coupling_mode: str,
    interface_reaction_robin_impedance_ns_m: float,
    interface_reaction_robin_matrix_impedance_ns_m: float,
) -> None:
    if str(fsi_coupling_mode) != FSI_COUPLING_MODE_HIBM_MPM_SHARP:
        return
    enabled_options: list[str] = []
    if float(interface_reaction_robin_impedance_ns_m) > 0.0:
        enabled_options.append("--interface-reaction-robin-impedance-ns-m")
    if float(interface_reaction_robin_matrix_impedance_ns_m) > 0.0:
        enabled_options.append("--interface-reaction-robin-matrix-impedance-ns-m")
    if enabled_options:
        joined_options = ", ".join(enabled_options)
        raise ValueError(
            "hibm_mpm_sharp currently reports explicit_loose coupling and has "
            "no marker-level Robin semi-implicit pressure/interface solve; "
            f"do not pass {joined_options} with --fsi-coupling-mode "
            "hibm_mpm_sharp until that marker-level Robin path is implemented."
        )


def build_hibm_mpm_sharp_coupling_state(
    *,
    fluid,
    solid_mpm,
    runtime: TaichiRuntimeConfig | None,
) -> HibmMpmSharpCouplingState:
    marker_count = int(getattr(solid_mpm, "particle_count"))
    if marker_count <= 0:
        raise ValueError("initialize solid_mpm particles before HIBM-MPM coupling")
    surface_region_id = getattr(solid_mpm, "region_id", None)
    if surface_region_id is None:
        surface_region_id = getattr(solid_mpm, "vertex_region_id", None)
    if surface_region_id is None:
        raise ValueError("solid_mpm must expose a Taichi surface region field")
    projection_triangle_indices = getattr(solid_mpm, "face_indices", None)
    projection_triangle_count = int(getattr(solid_mpm, "face_count", 0) or 0)
    projection_triangle_capacity = (
        projection_triangle_count
        if projection_triangle_indices is not None and projection_triangle_count > 0
        else None
    )
    coupling = HibmMpmSharpCouplingState(
        grid_nodes=fluid.grid.grid_nodes,
        bounds_min_m=fluid.grid.bounds_min_m,
        bounds_max_m=fluid.grid.bounds_max_m,
        marker_capacity=marker_count,
        projection_triangle_capacity=projection_triangle_capacity,
        runtime=runtime,
    )
    projection_kwargs = {}
    if projection_triangle_indices is not None and projection_triangle_count > 0:
        projection_kwargs = {
            "projection_triangle_indices": projection_triangle_indices,
            "projection_triangle_count": projection_triangle_count,
        }
    coupling.load_markers_from_surface_fields(
        solid_mpm.x,
        solid_mpm.surface_normal,
        solid_mpm.area_weight_m2,
        surface_region_id,
        marker_count=marker_count,
        surface_velocity_mps=solid_mpm.v,
        **projection_kwargs,
    )
    return coupling


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
    scatter_force_residual_n = _mapping_float(
        sharp_summary,
        "hibm_mpm_scatter_action_reaction_residual_n",
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
            "main_tail_region_reaction_diagnostic_only": bool(
                fsi_coupling_mode_report[
                    "main_tail_region_reaction_diagnostic_only"
                ]
            ),
            "fsi_coupling_solver": "hibm_mpm_sharp",
            "fsi_coupling_scheme": str(
                sharp_summary.get("hibm_coupling_scheme", "explicit_loose")
            ),
            "fsi_coupling_iterations_used": 0,
            "fsi_coupling_enabled": True,
            "fsi_coupling_explicit_single_pass": True,
            "fsi_added_mass_stability_status": str(
                sharp_summary.get("hibm_added_mass_stability_status", "unmeasured")
            ),
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
            "fsi_coupling_convergence_measured": False,
            "fsi_coupling_converged": False,
            "fluid_substeps": actual_fluid_substeps,
            "fluid_substep_dt_s": float(fluid_dt_s) / float(actual_fluid_substeps),
            "fluid_advection_scheme": str(
                fluid_projection_report.get("fluid_advection_scheme", "euler")
            ),
            "fsi_coupling_residual_norm_n": scatter_force_residual_n,
            "fsi_coupling_residual_source": (
                "marker_to_mpm_scatter_force_conservation"
            ),
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


def _split_region_pair_vector(
    values: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 6:
        raise ValueError("region-pair force vector must contain 6 values")
    return (
        (vector[0], vector[1], vector[2]),
        (vector[3], vector[4], vector[5]),
    )


def _taichi_vector3_to_tuple(value: object) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _surface_mesh_path(config: dict[str, object]) -> Path:
    mesh_path = config.get("surface_mesh_cache_path") or config.get("mesh_path")
    if not isinstance(mesh_path, str) or not mesh_path:
        raise ValueError("source config does not contain a mesh path")
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def nozzle_taper_geometry(spec: SquidReducedSpec) -> tuple[float, float, float]:
    taper_end_z_m = float(spec.chamber_z_min_m)
    taper_length_m = max(0.0, float(spec.nozzle_taper_length_m))
    taper_start_z_m = max(float(spec.downstream_z_m), taper_end_z_m - taper_length_m)
    inlet_radius_m = (
        float(spec.nozzle_taper_inlet_radius_m)
        if spec.nozzle_taper_inlet_radius_m is not None
        else float(spec.chamber_radius_m)
    )
    return (taper_start_z_m, taper_end_z_m, inlet_radius_m)


def nozzle_radius_at_z_m(spec: SquidReducedSpec, z_m: float) -> float:
    base_radius_m = float(spec.nozzle_radius_m)
    if not bool(spec.nozzle_taper_enabled):
        return base_radius_m
    taper_start_z_m, taper_end_z_m, inlet_radius_m = nozzle_taper_geometry(spec)
    if taper_end_z_m <= taper_start_z_m or z_m < taper_start_z_m or z_m >= taper_end_z_m:
        return base_radius_m
    fraction = (float(z_m) - taper_start_z_m) / max(taper_end_z_m - taper_start_z_m, 1.0e-12)
    return base_radius_m + (inlet_radius_m - base_radius_m) * min(max(fraction, 0.0), 1.0)


def reduced_water_geometry_report(spec: SquidReducedSpec) -> dict[str, object]:
    taper_start_z_m, taper_end_z_m, inlet_radius_m = nozzle_taper_geometry(spec)
    mid_z_m = 0.5 * (taper_start_z_m + taper_end_z_m)
    return {
        "nozzle_taper_enabled": bool(spec.nozzle_taper_enabled),
        "nozzle_taper_start_z_m": float(taper_start_z_m),
        "nozzle_taper_end_z_m": float(taper_end_z_m),
        "nozzle_taper_length_m": float(max(0.0, taper_end_z_m - taper_start_z_m)),
        "nozzle_taper_inlet_radius_m": float(inlet_radius_m),
        "nozzle_throat_radius_m": float(spec.nozzle_radius_m),
        "nozzle_radius_at_taper_start_m": float(nozzle_radius_at_z_m(spec, taper_start_z_m)),
        "nozzle_radius_at_taper_mid_m": float(nozzle_radius_at_z_m(spec, mid_z_m)),
        "nozzle_radius_at_taper_end_m": float(inlet_radius_m)
        if bool(spec.nozzle_taper_enabled) and taper_end_z_m > taper_start_z_m
        else float(spec.nozzle_radius_m),
        "outlet_plume_radius_m": float(spec.outlet_plume_radius_m),
        "downstream_farfield_open_enabled": bool(spec.downstream_farfield_open_enabled),
    }


def _reduced_water_mask(points_m: np.ndarray, spec: SquidReducedSpec) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    rx = points[:, 0] - float(spec.monitor_center_x_m)
    ry = points[:, 1] - float(spec.monitor_center_y_m)
    radius = np.sqrt(rx * rx + ry * ry)
    z = points[:, 2]
    nozzle_radius = np.asarray(
        [nozzle_radius_at_z_m(spec, float(value)) for value in z],
        dtype=np.float64,
    )
    chamber = (
        (radius <= float(spec.chamber_radius_m))
        & (z >= float(spec.chamber_z_min_m))
        & (z <= float(spec.chamber_z_max_m))
    )
    nozzle = (
        (radius <= nozzle_radius)
        & (z >= float(spec.downstream_z_m))
        & (z <= float(spec.nozzle_z_max_m))
    )
    outlet_plume = (
        (radius <= float(spec.outlet_plume_radius_m))
        & (z >= float(spec.fluid_bounds_min_m[2]))
        & (z < float(spec.downstream_z_m))
    )
    downstream_farfield = np.zeros_like(z, dtype=bool)
    if bool(spec.downstream_farfield_open_enabled):
        downstream_farfield = z <= float(spec.downstream_farfield_open_z_max_m)
    return chamber | nozzle | outlet_plume | downstream_farfield


def _orient_normals_to_reduced_water(
    centroids_m: np.ndarray,
    normals: np.ndarray,
    region_ids: np.ndarray,
    spec: SquidReducedSpec,
    probe_distance_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    plus_active = _reduced_water_mask(centroids_m + normals * probe_distance_m, spec)
    minus_active = _reduced_water_mask(centroids_m - normals * probe_distance_m, spec)
    flip = (~plus_active) & minus_active
    oriented = np.array(normals, copy=True)
    oriented[flip] *= -1.0
    final_active = plus_active | flip
    both_active = plus_active & minus_active
    neither_active = (~plus_active) & (~minus_active)
    by_region: dict[str, dict[str, int]] = {}
    for region in sorted({int(value) for value in region_ids.tolist()}):
        mask = region_ids == region
        by_region[str(region)] = {
            "face_count": int(np.count_nonzero(mask)),
            "plus_active_count": int(np.count_nonzero(plus_active & mask)),
            "minus_active_count": int(np.count_nonzero(minus_active & mask)),
            "flipped_count": int(np.count_nonzero(flip & mask)),
            "both_active_count": int(np.count_nonzero(both_active & mask)),
            "neither_active_count": int(np.count_nonzero(neither_active & mask)),
            "final_active_count": int(np.count_nonzero(final_active & mask)),
        }
    return oriented, {
        "method": "reduced_water_side_probe_orientation",
        "probe_distance_m": float(probe_distance_m),
        "flipped_count": int(np.count_nonzero(flip)),
        "plus_active_count": int(np.count_nonzero(plus_active)),
        "minus_active_count": int(np.count_nonzero(minus_active)),
        "both_active_count": int(np.count_nonzero(both_active)),
        "neither_active_count": int(np.count_nonzero(neither_active)),
        "final_active_count": int(np.count_nonzero(final_active)),
        "face_count": int(len(region_ids)),
        "by_region": by_region,
    }


def build_tri_surface_diagnostics(
    config: dict[str, object],
    runtime: TaichiRuntimeConfig,
    *,
    spec: SquidReducedSpec | None = None,
    probe_distance_m: float | None = None,
    region_ids: tuple[int, ...] = (7, 8),
    solid_region_ids: tuple[int, ...] = (7, 8, 5),
) -> tuple[TriSurfaceRegionDiagnostics, dict[str, object], SurfaceMesh, np.ndarray]:
    import trimesh

    mesh_path = _surface_mesh_path(config)
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh must contain triangular faces")

    def build_region_subset(active_region_ids: tuple[int, ...], label: str):
        selected_face_ids: list[int] = []
        selected_region_ids: list[int] = []
        region_face_counts: dict[str, int] = {}
        for region_id in active_region_ids:
            ids = _face_ids_for_region(config, region_id)
            region_face_counts[str(region_id)] = len(ids)
            selected_face_ids.extend(ids)
            selected_region_ids.extend([region_id] * len(ids))
        if not selected_face_ids:
            raise ValueError(f"no selected {label} region faces found")
        if max(selected_face_ids) >= len(faces) or min(selected_face_ids) < 0:
            raise ValueError(f"selected {label} face IDs are outside the surface mesh")

        selected_faces = faces[np.asarray(selected_face_ids, dtype=np.int64)]
        tri = vertices[selected_faces]
        area_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        doubled_area = np.linalg.norm(area_normals, axis=1)
        valid = doubled_area > 1.0e-20
        if not np.all(valid):
            selected_faces = selected_faces[valid]
            tri = tri[valid]
            area_normals = area_normals[valid]
            doubled_area = doubled_area[valid]
            selected_region_ids = [
                region for region, keep in zip(selected_region_ids, valid.tolist(), strict=True) if keep
            ]
        centroids = np.mean(tri, axis=1)
        areas = 0.5 * doubled_area
        normals = area_normals / doubled_area[:, None]
        region_array = np.asarray(selected_region_ids, dtype=np.int32)
        return selected_faces, centroids, areas, normals, region_array, region_face_counts

    selected_faces, centroids, areas, normals, region_array, region_face_counts = (
        build_region_subset(region_ids, "FSI diagnostic")
    )
    normal_orientation: dict[str, object] = {
        "method": "mesh_face_winding",
        "probe_distance_m": None,
        "flipped_count": 0,
        "face_count": int(len(region_array)),
    }
    if spec is not None and probe_distance_m is not None:
        normals, normal_orientation = _orient_normals_to_reduced_water(
            centroids,
            normals,
            region_array,
            spec,
            float(probe_distance_m),
        )
    solid_faces, solid_centroids, solid_areas, _, solid_region_array, solid_region_face_counts = (
        build_region_subset(solid_region_ids, "solid MPM")
    )
    unique_vertex_ids, inverse_vertex_ids = np.unique(
        solid_faces.reshape(-1),
        return_inverse=True,
    )
    tri_surface_mesh = SurfaceMesh(
        vertices=vertices[unique_vertex_ids],
        faces=inverse_vertex_ids.reshape((-1, 3)).astype(np.int32),
    )

    diagnostics = TriSurfaceRegionDiagnostics(face_capacity=int(len(areas)), runtime=runtime)
    diagnostics.load_faces(
        centroid_m=centroids.astype(np.float32),
        normal=normals.astype(np.float32),
        area_m2=areas.astype(np.float32),
        region_id=region_array,
    )
    metadata = {
        "mesh_path": str(mesh_path),
        "mesh_scale_to_m": mesh_scale_to_m,
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "diagnostic_face_count": int(len(areas)),
        "region_face_counts": region_face_counts,
        "diagnostic_area_m2_by_region": {
            str(region): float(np.sum(areas[region_array == region])) for region in region_ids
        },
        "solid_region_face_counts": solid_region_face_counts,
        "solid_area_m2_by_region": {
            str(region): float(np.sum(solid_areas[solid_region_array == region]))
            for region in solid_region_ids
        },
        "solid_surface_vertex_count": int(tri_surface_mesh.vertex_count),
        "solid_surface_face_count": int(tri_surface_mesh.face_count),
        "solid_surface_edge_note": "deduplicated from FSI triangles plus fixed rim triangles for TriMooneyShellMpmState",
        "centroid_bounds_min_m": [float(value) for value in np.min(centroids, axis=0)],
        "centroid_bounds_max_m": [float(value) for value in np.max(centroids, axis=0)],
        "solid_centroid_bounds_min_m": [float(value) for value in np.min(solid_centroids, axis=0)],
        "solid_centroid_bounds_max_m": [float(value) for value in np.max(solid_centroids, axis=0)],
        "normal_orientation": normal_orientation,
    }
    return diagnostics, metadata, tri_surface_mesh, solid_region_array


def compute_region_geometry_stats(
    config: dict[str, object],
    region_id: int,
) -> dict[str, object]:
    import trimesh

    face_ids = _face_ids_for_region(config, region_id)
    if not face_ids:
        return {
            "region_id": int(region_id),
            "available": False,
            "face_count": 0,
        }
    mesh_path = _surface_mesh_path(config)
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max(face_ids) >= len(faces) or min(face_ids) < 0:
        raise ValueError(f"region {region_id} face IDs are outside the surface mesh")
    selected_faces = faces[np.asarray(face_ids, dtype=np.int64)]
    tri = vertices[selected_faces]
    area_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    doubled_area = np.linalg.norm(area_normals, axis=1)
    valid = doubled_area > 1.0e-20
    tri = tri[valid]
    area_normals = area_normals[valid]
    doubled_area = doubled_area[valid]
    if tri.size == 0:
        return {
            "region_id": int(region_id),
            "available": False,
            "face_count": int(len(face_ids)),
            "valid_face_count": 0,
        }
    areas = 0.5 * doubled_area
    centroids = np.mean(tri, axis=1)
    vertices_flat = tri.reshape((-1, 3))
    area_total = float(np.sum(areas))
    if area_total > 0.0:
        area_weighted_centroid = np.sum(centroids * areas[:, None], axis=0) / area_total
    else:
        area_weighted_centroid = np.mean(centroids, axis=0)
    xy_center = area_weighted_centroid[:2]
    vertex_radius = np.linalg.norm(vertices_flat[:, :2] - xy_center[None, :], axis=1)
    centroid_radius = np.linalg.norm(centroids[:, :2] - xy_center[None, :], axis=1)
    normals = area_normals / doubled_area[:, None]
    area_weighted_normal = np.sum(normals * areas[:, None], axis=0)
    normal_norm = float(np.linalg.norm(area_weighted_normal))
    if normal_norm > 0.0:
        area_weighted_normal = area_weighted_normal / normal_norm
    return {
        "region_id": int(region_id),
        "available": True,
        "mesh_path": str(mesh_path),
        "mesh_scale_to_m": mesh_scale_to_m,
        "face_count": int(len(face_ids)),
        "valid_face_count": int(len(areas)),
        "area_m2": area_total,
        "area_weighted_centroid_m": [float(value) for value in area_weighted_centroid],
        "vertex_bounds_min_m": [float(value) for value in np.min(vertices_flat, axis=0)],
        "vertex_bounds_max_m": [float(value) for value in np.max(vertices_flat, axis=0)],
        "centroid_bounds_min_m": [float(value) for value in np.min(centroids, axis=0)],
        "centroid_bounds_max_m": [float(value) for value in np.max(centroids, axis=0)],
        "vertex_radius_min_m": float(np.min(vertex_radius)),
        "vertex_radius_mean_m": float(np.mean(vertex_radius)),
        "vertex_radius_p95_m": float(np.percentile(vertex_radius, 95.0)),
        "vertex_radius_max_m": float(np.max(vertex_radius)),
        "centroid_radius_max_m": float(np.max(centroid_radius)),
        "area_weighted_normal": [float(value) for value in area_weighted_normal],
    }


def spec_with_region14_aperture(
    spec: SquidReducedSpec,
    aperture_stats: dict[str, object],
    *,
    open_downstream_farfield: bool = False,
) -> SquidReducedSpec:
    if not bool(aperture_stats.get("available", False)):
        return spec
    center = aperture_stats.get("area_weighted_centroid_m", [])
    radius = float(aperture_stats.get("vertex_radius_p95_m", spec.nozzle_radius_m))
    if not isinstance(center, list | tuple) or len(center) < 2 or radius <= 0.0:
        return spec
    aperture_z = float(center[2]) if len(center) >= 3 else float(spec.lip_z_m)
    return replace(
        spec,
        monitor_center_x_m=float(center[0]),
        monitor_center_y_m=float(center[1]),
        nozzle_radius_m=radius,
        outlet_plume_radius_m=radius,
        monitor_radius_m=radius,
        downstream_farfield_open_enabled=bool(open_downstream_farfield),
        downstream_farfield_open_z_max_m=aperture_z,
    )


def spec_with_nozzle_taper(
    spec: SquidReducedSpec,
    *,
    taper_length_m: float | None = None,
    inlet_radius_m: float | None = None,
) -> SquidReducedSpec:
    length_m = (
        min(
            float(spec.nozzle_length_m),
            max(float(spec.chamber_z_min_m) - float(spec.downstream_z_m), 0.0),
        )
        if taper_length_m is None
        else float(taper_length_m)
    )
    if length_m <= 0.0:
        raise ValueError("nozzle taper length must be positive")
    inlet_radius = (
        float(spec.chamber_radius_m)
        if inlet_radius_m is None
        else float(inlet_radius_m)
    )
    if inlet_radius <= float(spec.nozzle_radius_m):
        raise ValueError("nozzle taper inlet radius must exceed the throat radius")
    return replace(
        spec,
        nozzle_taper_enabled=True,
        nozzle_taper_length_m=length_m,
        nozzle_taper_inlet_radius_m=inlet_radius,
    )


def spec_with_nozzle_graded_grid(
    spec: SquidReducedSpec,
    *,
    target_spacing_m: float | None = None,
    farfield_spacing_m: float = 3.0e-3,
    max_growth_ratio: float = 1.2,
    max_cells: int | None = None,
    extra_refinement_regions: Sequence[RefinementRegion] = (),
) -> SquidReducedSpec:
    target_spacing = (
        float(target_spacing_m)
        if target_spacing_m is not None
        else float(spec.nozzle_radius_m) / 5.0
    )
    if target_spacing <= 0.0:
        raise ValueError("graded grid target spacing must be positive")
    if farfield_spacing_m <= 0.0:
        raise ValueError("graded grid farfield spacing must be positive")
    if max_growth_ratio <= 1.0:
        raise ValueError("graded grid max growth ratio must be greater than 1")

    taper_start_z_m, taper_end_z_m, taper_inlet_radius_m = nozzle_taper_geometry(spec)
    radius_m = (
        max(float(spec.nozzle_radius_m), float(taper_inlet_radius_m))
        if bool(spec.nozzle_taper_enabled) and taper_end_z_m > taper_start_z_m
        else float(spec.nozzle_radius_m)
    )
    bounds_min = spec.fluid_bounds_min_m
    bounds_max = spec.fluid_bounds_max_m
    region_bounds_min = (
        max(float(bounds_min[0]), float(spec.monitor_center_x_m) - radius_m),
        max(float(bounds_min[1]), float(spec.monitor_center_y_m) - radius_m),
        max(float(bounds_min[2]), min(float(spec.downstream_z_m), float(spec.nozzle_z_max_m))),
    )
    region_bounds_max = (
        min(float(bounds_max[0]), float(spec.monitor_center_x_m) + radius_m),
        min(float(bounds_max[1]), float(spec.monitor_center_y_m) + radius_m),
        min(float(bounds_max[2]), max(float(spec.downstream_z_m), float(spec.nozzle_z_max_m))),
    )
    if any(hi <= lo for lo, hi in zip(region_bounds_min, region_bounds_max, strict=True)):
        raise ValueError("graded nozzle refinement region does not overlap the fluid domain")
    refinement_regions = (
        RefinementRegion(
            bounds_min_m=region_bounds_min,
            bounds_max_m=region_bounds_max,
            target_spacing_m=target_spacing,
        ),
    ) + tuple(extra_refinement_regions)

    graded_grid = GradedGridSpec(
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        farfield_spacing_m=float(farfield_spacing_m),
        max_growth_ratio=float(max_growth_ratio),
        max_cells=max_cells,
        refinement_regions=refinement_regions,
    )
    grid = build_graded_grid(graded_grid)
    return replace(
        spec,
        grid_nodes=grid.grid_nodes,
        cartesian_grid=None,
        graded_grid=graded_grid,
    )


def tail_refinement_region_from_geometry(
    spec: SquidReducedSpec,
    tail_geometry: dict[str, object],
    *,
    target_spacing_m: float,
    padding_m: float,
) -> RefinementRegion | None:
    if not bool(tail_geometry.get("available", False)):
        return None
    target_spacing = float(target_spacing_m)
    if target_spacing <= 0.0:
        raise ValueError("tail refinement target spacing must be positive")
    padding = float(padding_m)
    if padding < 0.0:
        raise ValueError("tail refinement padding must be non-negative")
    raw_min = required_tuple3(
        tail_geometry.get("vertex_bounds_min_m"),
        field="tail refinement vertex_bounds_min_m",
    )
    raw_max = required_tuple3(
        tail_geometry.get("vertex_bounds_max_m"),
        field="tail refinement vertex_bounds_max_m",
    )
    bounds_min = tuple(
        max(float(domain_min), float(raw_value) - padding)
        for domain_min, raw_value in zip(spec.fluid_bounds_min_m, raw_min, strict=True)
    )
    bounds_max = tuple(
        min(float(domain_max), float(raw_value) + padding)
        for domain_max, raw_value in zip(spec.fluid_bounds_max_m, raw_max, strict=True)
    )
    if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("tail refinement region does not overlap the fluid domain")
    return RefinementRegion(
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        target_spacing_m=target_spacing,
    )


def refinement_region_summary(region: RefinementRegion | None) -> dict[str, object] | None:
    if region is None:
        return None
    return {
        "bounds_min_m": [float(value) for value in region.bounds_min_m],
        "bounds_max_m": [float(value) for value in region.bounds_max_m],
        "target_spacing_m": [float(value) for value in region.target_spacing_m],
    }


def cartesian_grid_for_spec(spec: SquidReducedSpec) -> CartesianGrid:
    if spec.graded_grid is not None:
        return build_graded_grid(spec.graded_grid)
    if spec.cartesian_grid is not None:
        return spec.cartesian_grid
    return CartesianGrid.uniform(
        bounds_min_m=spec.fluid_bounds_min_m,
        bounds_max_m=spec.fluid_bounds_max_m,
        grid_nodes=spec.grid_nodes,
    )


def cartesian_grid_axis_min_spacing_m(grid: CartesianGrid) -> tuple[float, float, float]:
    return (
        float(min(grid.cell_widths_x_m)),
        float(min(grid.cell_widths_y_m)),
        float(min(grid.cell_widths_z_m)),
    )


def cartesian_grid_axis_max_spacing_m(grid: CartesianGrid) -> tuple[float, float, float]:
    return (
        float(max(grid.cell_widths_x_m)),
        float(max(grid.cell_widths_y_m)),
        float(max(grid.cell_widths_z_m)),
    )


def cartesian_grid_uniform_spacing_m(grid: CartesianGrid) -> tuple[float, float, float] | None:
    if not grid.is_uniform:
        return None
    return tuple(float(value) for value in grid.uniform_spacing_m)


def _count_axis_centers_in_bounds(
    centers: Sequence[float],
    lower: float,
    upper: float,
) -> int:
    return sum(1 for value in centers if lower <= float(value) <= upper)


def _axis_width_range_in_bounds(
    centers: Sequence[float],
    widths: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    selected = [
        float(width)
        for center, width in zip(centers, widths, strict=True)
        if lower <= float(center) <= upper
    ]
    if not selected:
        return None
    return (min(selected), max(selected))


def _max_adjacent_spacing_ratio(widths: Sequence[float]) -> float:
    ratios = []
    for left, right in zip(widths, widths[1:], strict=False):
        left_value = float(left)
        right_value = float(right)
        if left_value > 0.0 and right_value > 0.0:
            ratios.append(max(left_value / right_value, right_value / left_value))
    return max(ratios, default=1.0)


def fluid_grid_resolution_report(spec: SquidReducedSpec) -> dict[str, object]:
    grid = cartesian_grid_for_spec(spec)
    radius_m = float(spec.nozzle_radius_m)
    nozzle_bounds_min = (
        float(spec.monitor_center_x_m) - radius_m,
        float(spec.monitor_center_y_m) - radius_m,
        min(float(spec.downstream_z_m), float(spec.nozzle_z_max_m)),
    )
    nozzle_bounds_max = (
        float(spec.monitor_center_x_m) + radius_m,
        float(spec.monitor_center_y_m) + radius_m,
        max(float(spec.downstream_z_m), float(spec.nozzle_z_max_m)),
    )
    axes = (
        (grid.cell_centers_x_m, grid.cell_widths_x_m, nozzle_bounds_min[0], nozzle_bounds_max[0]),
        (grid.cell_centers_y_m, grid.cell_widths_y_m, nozzle_bounds_min[1], nozzle_bounds_max[1]),
        (grid.cell_centers_z_m, grid.cell_widths_z_m, nozzle_bounds_min[2], nozzle_bounds_max[2]),
    )
    nozzle_cells = tuple(
        _count_axis_centers_in_bounds(centers, lower, upper)
        for centers, _widths, lower, upper in axes
    )
    nozzle_width_ranges = tuple(
        _axis_width_range_in_bounds(centers, widths, lower, upper)
        for centers, widths, lower, upper in axes
    )
    min_widths = tuple(
        None if width_range is None else width_range[0]
        for width_range in nozzle_width_ranges
    )
    max_widths = tuple(
        None if width_range is None else width_range[1]
        for width_range in nozzle_width_ranges
    )
    target_spacing = None
    if spec.graded_grid is not None and spec.graded_grid.refinement_regions:
        target_spacing = spec.graded_grid.refinement_regions[0].target_spacing_m
    return {
        "graded_enabled": spec.graded_grid is not None,
        "grid_nodes": [int(value) for value in grid.grid_nodes],
        "bounds_min_m": [float(value) for value in grid.bounds_min_m],
        "bounds_max_m": [float(value) for value in grid.bounds_max_m],
        "nozzle_bounds_min_m": [float(value) for value in nozzle_bounds_min],
        "nozzle_bounds_max_m": [float(value) for value in nozzle_bounds_max],
        "nozzle_cells_x": int(nozzle_cells[0]),
        "nozzle_cells_y": int(nozzle_cells[1]),
        "nozzle_cells_z": int(nozzle_cells[2]),
        "nozzle_diameter_cells_min": int(min(nozzle_cells[0], nozzle_cells[1])),
        "nozzle_resolves_diameter_10_cells": min(nozzle_cells[0], nozzle_cells[1]) >= 10,
        "nozzle_min_cell_width_m": [
            None if value is None else float(value) for value in min_widths
        ],
        "nozzle_max_cell_width_m": [
            None if value is None else float(value) for value in max_widths
        ],
        "global_min_cell_width_m": [
            float(value) for value in cartesian_grid_axis_min_spacing_m(grid)
        ],
        "global_max_cell_width_m": [
            float(value) for value in cartesian_grid_axis_max_spacing_m(grid)
        ],
        "max_adjacent_spacing_ratio": [
            float(_max_adjacent_spacing_ratio(grid.cell_widths_x_m)),
            float(_max_adjacent_spacing_ratio(grid.cell_widths_y_m)),
            float(_max_adjacent_spacing_ratio(grid.cell_widths_z_m)),
        ],
        "graded_grid_target_spacing_m": (
            None if target_spacing is None else [float(value) for value in target_spacing]
        ),
    }


def required_tuple3(values: object, *, field: str) -> tuple[float, float, float]:
    if isinstance(values, list | tuple) and len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    raise ValueError(f"{field} must contain exactly 3 numeric components")


def solid_mpm_bounds_from_surface_metadata(
    metadata: Mapping[str, object],
    *,
    fallback_bounds_min_m: Sequence[float],
    fallback_bounds_max_m: Sequence[float],
    padding_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fallback_min = required_tuple3(
        fallback_bounds_min_m,
        field="fallback_bounds_min_m",
    )
    fallback_max = required_tuple3(
        fallback_bounds_max_m,
        field="fallback_bounds_max_m",
    )
    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_m must be a finite non-negative number")
    surface_min = required_tuple3(
        metadata.get("solid_centroid_bounds_min_m", fallback_min),
        field="metadata.solid_centroid_bounds_min_m",
    )
    surface_max = required_tuple3(
        metadata.get("solid_centroid_bounds_max_m", fallback_max),
        field="metadata.solid_centroid_bounds_max_m",
    )
    bounds_min = tuple(
        min(domain_min, solid_min - padding)
        for domain_min, solid_min in zip(fallback_min, surface_min, strict=True)
    )
    bounds_max = tuple(
        max(domain_max, solid_max + padding)
        for domain_max, solid_max in zip(fallback_max, surface_max, strict=True)
    )
    if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("solid MPM bounds must have positive extent")
    return bounds_min, bounds_max


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


def pressure_schedule_applied_in_history(rows: Sequence[dict[str, object]]) -> bool:
    for index, row in enumerate(rows):
        pressure_pa = _required_finite_row_number(
            row,
            "pressure_load_pa",
            context=f"history row {index}",
        )
        if abs(pressure_pa) > 0.0:
            return True
    return False


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


def pressure_schedule_pa(time_s: float, spec: SquidReducedSpec | None = None) -> float:
    if spec is None:
        t0_s, t1_s, t2_s = 0.0, 1.0, 2.0
        p0_pa, p1_pa, p2_pa = 0.0, 8000.0, -8000.0
    else:
        t0_s = float(spec.pressure_t0_s)
        t1_s = float(spec.pressure_t1_s)
        t2_s = float(spec.pressure_t2_s)
        p0_pa = float(spec.pressure_p0_pa)
        p1_pa = float(spec.pressure_p1_pa)
        p2_pa = float(spec.pressure_p2_pa)
    if not (t0_s < t1_s < t2_s):
        raise ValueError("pressure schedule times must satisfy t0 < t1 < t2")
    time = float(time_s)
    if time <= t0_s:
        return p0_pa
    if time <= t1_s:
        alpha = (time - t0_s) / (t1_s - t0_s)
        return p0_pa + (p1_pa - p0_pa) * alpha
    if time <= t2_s:
        alpha = (time - t1_s) / (t2_s - t1_s)
        return p1_pa + (p2_pa - p1_pa) * alpha
    return p2_pa


def pressure_schedule_step_end_pa(
    current_time_s: float,
    dt_s: float,
    spec: SquidReducedSpec | None = None,
) -> float:
    return pressure_schedule_pa(float(current_time_s) + float(dt_s), spec)


def hydraulic_diagnostics(
    spec: SquidReducedSpec,
    main_velocity_z_mps: float,
) -> tuple[float, float, float]:
    q_m3s = -float(spec.main_area_m2) * float(main_velocity_z_mps)
    nozzle_speed_mps = q_m3s / max(float(spec.nozzle_area_m2), 1.0e-12)
    viscous_dp_pa = (
        8.0
        * float(spec.water_viscosity_pa_s)
        * float(spec.nozzle_length_m)
        * q_m3s
        / max(math.pi * float(spec.nozzle_radius_m) ** 4, 1.0e-18)
    )
    inertial_dp_pa = 0.5 * float(spec.water_density_kgm3) * nozzle_speed_mps * abs(nozzle_speed_mps)
    return viscous_dp_pa + inertial_dp_pa, q_m3s, -nozzle_speed_mps


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


def physical_positive_source_flux_ratio_passes(
    *,
    outlet_negative_z_flux_m3s: float,
    source_flux_m3s: float,
    min_ratio: float,
    min_source_flux_m3s: float = 1.0e-18,
) -> bool:
    outlet_flux = float(outlet_negative_z_flux_m3s)
    source_flux = float(source_flux_m3s)
    ratio = signed_positive_source_flux_ratio(
        outlet_negative_z_flux_m3s=outlet_flux,
        source_flux_m3s=source_flux,
        min_source_flux_m3s=min_source_flux_m3s,
    )
    return (
        math.isfinite(outlet_flux)
        and math.isfinite(source_flux)
        and source_flux > float(min_source_flux_m3s)
        and outlet_flux > 0.0
        and ratio >= float(min_ratio)
    )


def physical_outlet_to_fsi_volume_source_passes(
    *,
    outlet_negative_z_flux_m3s: float,
    fsi_volume_source_m3s: float,
    min_ratio: float,
) -> bool:
    return physical_positive_source_flux_ratio_passes(
        outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
        source_flux_m3s=fsi_volume_source_m3s,
        min_ratio=min_ratio,
    )


def fsi_physical_interface_map_stability_report(
    *,
    fsi_coupling_enabled: bool,
    fsi_coupling_iterations: int,
    max_physical_interface_map_amplification: float,
    measurement_sample_count: int,
    raw_interface_map_strict_physical: bool,
    limit: float = 1.0,
) -> dict[str, object]:
    enabled = bool(fsi_coupling_enabled)
    iterations = int(fsi_coupling_iterations)
    samples = int(measurement_sample_count)
    strict_physical = bool(raw_interface_map_strict_physical)
    if not enabled:
        return {
            "applicable": False,
            "measured": False,
            "passes": True,
            "status": "not_applicable_coupling_disabled",
            "reason": "fsi_coupling_disabled",
            "sample_count": samples,
        }
    if iterations <= 1 or samples <= 0:
        return {
            "applicable": True,
            "measured": False,
            "passes": False,
            "status": "unmeasured",
            "reason": "insufficient_distinct_fsi_trials",
            "sample_count": samples,
        }
    if not strict_physical:
        return {
            "applicable": True,
            "measured": True,
            "passes": False,
            "status": "masked_by_stabilizer",
            "reason": "raw_interface_map_not_strict_physical",
            "sample_count": samples,
        }
    amplification = float(max_physical_interface_map_amplification)
    stable_limit = float(limit)
    passes = (
        math.isfinite(amplification)
        and math.isfinite(stable_limit)
        and stable_limit >= 0.0
        and amplification <= stable_limit
    )
    return {
        "applicable": True,
        "measured": True,
        "passes": passes,
        "status": "stable" if passes else "unstable",
        "reason": "measured_raw_interface_map",
        "sample_count": samples,
        "amplification": amplification,
        "limit": stable_limit,
    }


def fsi_physical_interface_map_stability_passes(
    *,
    fsi_coupling_enabled: bool,
    fsi_coupling_iterations: int,
    max_physical_interface_map_amplification: float,
    measurement_sample_count: int,
    raw_interface_map_strict_physical: bool = True,
    limit: float = 1.0,
) -> bool:
    return bool(
        fsi_physical_interface_map_stability_report(
            fsi_coupling_enabled=fsi_coupling_enabled,
            fsi_coupling_iterations=fsi_coupling_iterations,
            max_physical_interface_map_amplification=max_physical_interface_map_amplification,
            measurement_sample_count=measurement_sample_count,
            raw_interface_map_strict_physical=raw_interface_map_strict_physical,
            limit=limit,
        )["passes"]
    )


def solid_response_constraint_force_mobility_ratio(
    *,
    previous_velocity_mps: Sequence[float],
    current_velocity_mps: Sequence[float],
    reaction_force_n: Sequence[float],
    interface_area_m2: float,
    probe_distance_m: float,
    density_kgm3: float,
    dt_s: float,
    axis: int = 2,
    min_abs_reaction_force_n: float = 1.0e-12,
) -> float:
    previous_velocity = _vector3(previous_velocity_mps, name="previous_velocity_mps")
    current_velocity = _vector3(current_velocity_mps, name="current_velocity_mps")
    reaction_force = _vector3(reaction_force_n, name="reaction_force_n")
    if any(not math.isfinite(value) for value in previous_velocity):
        raise ValueError("previous_velocity_mps must contain only finite values")
    if any(not math.isfinite(value) for value in current_velocity):
        raise ValueError("current_velocity_mps must contain only finite values")
    if any(not math.isfinite(value) for value in reaction_force):
        raise ValueError("reaction_force_n must contain only finite values")
    axis_index = int(axis)
    if axis_index not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    area = float(interface_area_m2)
    probe_distance = float(probe_distance_m)
    density = float(density_kgm3)
    dt = float(dt_s)
    min_force = float(min_abs_reaction_force_n)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("interface_area_m2 must be a finite non-negative number")
    if not math.isfinite(probe_distance) or probe_distance <= 0.0:
        raise ValueError("probe_distance_m must be a finite positive number")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density_kgm3 must be a finite positive number")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a finite positive number")
    if not math.isfinite(min_force) or min_force < 0.0:
        raise ValueError("min_abs_reaction_force_n must be a finite non-negative number")
    force_component = reaction_force[axis_index]
    if abs(force_component) <= min_force or area <= 0.0:
        return 0.0
    velocity_delta = current_velocity[axis_index] - previous_velocity[axis_index]
    solid_mobility_mps_per_n = abs(velocity_delta / force_component)
    fluid_interface_stiffness_n_per_mps = density * area * probe_distance / dt
    return solid_mobility_mps_per_n * fluid_interface_stiffness_n_per_mps


def outlet_to_fsi_volume_source_gate_scope(
    *,
    fluid_grid_resolution: dict[str, object],
    validation_scope_complete: bool,
) -> dict[str, object]:
    nozzle_resolved = bool(fluid_grid_resolution.get("nozzle_resolves_diameter_10_cells", False))
    reasons: list[str] = []
    if not nozzle_resolved:
        reasons.append("nozzle_grid_not_resolved")
    if not bool(validation_scope_complete):
        reasons.append("jet_development_scope_incomplete")
    hard_gate = not reasons
    return {
        "gate": "completed_step_check" if hard_gate else "diagnostic_only",
        "hard_gate": hard_gate,
        "nozzle_resolved": nozzle_resolved,
        "jet_development_evaluable": bool(validation_scope_complete),
        "nozzle_diameter_cells_min": int(
            fluid_grid_resolution.get("nozzle_diameter_cells_min", 0) or 0
        ),
        "reasons": reasons,
    }


def pressure_outlet_source_ratio_passes(
    *,
    source_volume_flux_m3s: float,
    velocity_outlet_flux_m3s: float,
    pressure_outlet_flux_m3s: float,
    ratio_tolerance: float,
    min_source_flux_m3s: float = 1.0e-18,
) -> bool:
    source_flux = float(source_volume_flux_m3s)
    velocity_flux = float(velocity_outlet_flux_m3s)
    pressure_flux = float(pressure_outlet_flux_m3s)
    tolerance = float(ratio_tolerance)
    # The physical conservation gate uses the final outlet-face velocity flux.
    # The pressure-correction flux is a diagnostic and may be small for the
    # open z-min projection; require it to be finite without treating it as an
    # independent mass-conservation flux.
    if (
        not math.isfinite(source_flux)
        or not math.isfinite(velocity_flux)
        or not math.isfinite(pressure_flux)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or source_flux <= float(min_source_flux_m3s)
        or velocity_flux <= 0.0
    ):
        return False
    velocity_ratio = velocity_flux / source_flux
    return abs(velocity_ratio - 1.0) <= tolerance


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


def run_process_completion_status(
    *,
    validation_scope_complete: bool,
    validation_passed: bool | None,
    partial_run_stopped: bool,
    requested_steps: int,
    completed_steps: int,
) -> str:
    if bool(partial_run_stopped) or int(completed_steps) < int(requested_steps):
        return "partial"
    if bool(validation_scope_complete):
        return "finished" if bool(validation_passed) else "validation_failed"
    return "finished"


def pressure_flux_trend_report(
    rows: Sequence[dict[str, object]],
    *,
    requested_steps: int,
    min_trend_steps: int = 200,
    near_zero_pressure_ratio: float = 1.0e-3,
    rising_pressure_ratio: float = 1.0e-2,
    growth_factor: float = 5.0,
) -> dict[str, object]:
    required_steps = max(1, int(min_trend_steps))
    requested = int(requested_steps)
    completed = len(rows)
    report: dict[str, object] = {
        "required_steps": required_steps,
        "requested_steps": requested,
        "completed_steps": completed,
        "complete": completed >= required_steps and requested >= required_steps,
    }
    if not report["complete"]:
        report.update(
            {
                "conclusion": "incomplete",
                "reason": "insufficient_completed_steps_for_pressure_flux_trend",
            }
        )
        return report

    pressure_ratio_abs = [
        abs(
            _required_finite_row_number(
                row,
                "pressure_outlet_pressure_to_source_ratio",
                context=f"pressure-flux trend row {index}",
            )
        )
        for index, row in enumerate(rows)
    ]
    velocity_ratio = [
        _required_finite_row_number(
            row,
            "pressure_outlet_velocity_to_source_ratio",
            context=f"pressure-flux trend row {index}",
        )
        for index, row in enumerate(rows)
    ]
    pressure_load = [
        _required_finite_row_number(
            row,
            "pressure_load_pa",
            context=f"pressure-flux trend row {index}",
        )
        for index, row in enumerate(rows)
    ]
    window = max(1, min(20, completed // 10))
    early_pressure_ratio_mean = sum(pressure_ratio_abs[:window]) / float(window)
    late_pressure_ratio_mean = sum(pressure_ratio_abs[-window:]) / float(window)
    max_pressure_ratio = max(pressure_ratio_abs)
    min_pressure_load = min(pressure_load)
    max_pressure_load = max(pressure_load)
    growth_denominator = max(early_pressure_ratio_mean, 1.0e-12)
    pressure_ratio_growth = late_pressure_ratio_mean / growth_denominator
    pressure_ratio_rise = late_pressure_ratio_mean - early_pressure_ratio_mean
    final_velocity_ratio = velocity_ratio[-1]
    report.update(
        {
            "window_steps": window,
            "early_pressure_ratio_mean_abs": early_pressure_ratio_mean,
            "late_pressure_ratio_mean_abs": late_pressure_ratio_mean,
            "max_pressure_ratio_abs": max_pressure_ratio,
            "pressure_ratio_rise_abs": pressure_ratio_rise,
            "pressure_ratio_growth_factor": pressure_ratio_growth,
            "min_pressure_load_pa": min_pressure_load,
            "max_pressure_load_pa": max_pressure_load,
            "pressure_load_range_pa": max_pressure_load - min_pressure_load,
            "final_velocity_to_source_ratio": final_velocity_ratio,
            "final_pressure_to_source_ratio": pressure_ratio_abs[-1],
        }
    )
    if max_pressure_ratio <= float(near_zero_pressure_ratio):
        report.update(
            {
                "conclusion": "pressure_implied_flux_remained_near_zero_kinematic_ibm_dominated",
                "reason": None,
            }
        )
    elif (
        late_pressure_ratio_mean >= float(rising_pressure_ratio)
        and pressure_ratio_growth >= float(growth_factor)
        and pressure_ratio_rise > 0.0
    ):
        report.update(
            {
                "conclusion": "pressure_implied_flux_rose_pressure_driven_component_present",
                "reason": None,
            }
        )
    else:
        report.update(
            {
                "conclusion": "pressure_implied_flux_trend_inconclusive",
                "reason": "pressure_ratio_not_near_zero_but_not_a_clear_late_rise",
            }
        )
    return report


def force_decomposition_report(
    *,
    grid_force_n: Sequence[float],
    component_forces_n: Sequence[Sequence[float]],
    tolerance: float = 1.0e-6,
) -> dict[str, object]:
    grid_force = _vector3(grid_force_n, name="grid_force_n")
    component_vectors = tuple(
        _vector3(component, name="component_force_n") for component in component_forces_n
    )
    component_sum = tuple(
        sum(component[index] for component in component_vectors) for index in range(3)
    )
    residual = tuple(grid_force[index] - component_sum[index] for index in range(3))
    residual_norm = vector_norm(residual)
    scale = max(vector_norm(grid_force) + vector_norm(component_sum), 1.0e-30)
    relative_error = residual_norm / scale
    return {
        "grid_force_n": grid_force,
        "component_sum_n": component_sum,
        "residual_components_n": residual,
        "residual_norm_n": residual_norm,
        "relative_error": relative_error,
        "passed": relative_error <= float(tolerance),
    }


def resolve_pressure_solver(
    pressure_solver: str,
    *,
    graded_grid_enabled: bool,
) -> str:
    solver_name = str(pressure_solver)
    if solver_name == "auto":
        return "fv_cg" if graded_grid_enabled else "fv_multigrid"
    if solver_name not in PRESSURE_SOLVER_CHOICES:
        raise ValueError(f"unsupported pressure solver: {pressure_solver!r}")
    if graded_grid_enabled and solver_name not in {"fv_jacobi", "fv_multigrid", "fv_cg"}:
        raise ValueError("--use-graded-grid requires an FV pressure solver")
    return solver_name


def effective_fluid_substeps_for_grid(
    spec: SquidReducedSpec,
    requested_substeps: int,
    *,
    grid: CartesianGrid | None = None,
) -> int:
    requested = max(1, int(requested_substeps))
    if spec.graded_grid is None:
        return requested
    grid_for_spacing = cartesian_grid_for_spec(spec) if grid is None else grid
    min_spacing_m = min(cartesian_grid_axis_min_spacing_m(grid_for_spacing))
    farfield_spacing_m = max(float(value) for value in spec.graded_grid.farfield_spacing_m)
    # Resolve the finest graded cells at a half-farfield CFL; the full-step
    # ratio was not enough for the projected-IBM divergence guard.
    fine_cell_spacing_ratio = int(math.ceil(farfield_spacing_m / max(min_spacing_m, 1.0e-12)))
    reference_dt_s = float(spec.base_dt_s) if spec.base_dt_s is not None else float(spec.dt_s)
    dt_scale = float(spec.dt_s) / max(reference_dt_s, 1.0e-12)
    graded_substeps = max(1, int(math.ceil(2 * fine_cell_spacing_ratio * dt_scale)))
    return max(requested, graded_substeps)


def pressure_projection_budget_report(
    *,
    fluid_substeps: int,
    ibm_correction_iterations: int,
    fsi_coupling_iterations: int,
    projection_iterations: int,
    fsi_coupling_enabled: bool,
) -> dict[str, object]:
    substeps = max(1, int(fluid_substeps))
    correction_iterations = max(1, int(ibm_correction_iterations))
    coupling_iterations = max(1, int(fsi_coupling_iterations))
    pressure_iterations = max(1, int(projection_iterations))
    trial_evaluations = coupling_iterations if bool(fsi_coupling_enabled) else 0
    accepted_evaluations = 1
    project_calls_per_fluid_evaluation = substeps * correction_iterations
    trial_project_calls = trial_evaluations * project_calls_per_fluid_evaluation
    accepted_project_calls = accepted_evaluations * project_calls_per_fluid_evaluation
    total_project_calls = trial_project_calls + accepted_project_calls
    return {
        "fluid_substeps": substeps,
        "ibm_correction_iterations": correction_iterations,
        "fsi_coupling_enabled": bool(fsi_coupling_enabled),
        "fsi_coupling_trial_evaluations_per_physical_step_max": trial_evaluations,
        "accepted_fluid_step_evaluations_per_physical_step": accepted_evaluations,
        "fluid_step_evaluations_per_physical_step_max": (
            trial_evaluations + accepted_evaluations
        ),
        "pressure_project_calls_per_fluid_evaluation": project_calls_per_fluid_evaluation,
        "trial_pressure_project_calls_per_step_max": trial_project_calls,
        "full_report_pressure_project_calls_per_step": accepted_project_calls,
        "pressure_project_calls_per_physical_step_max": total_project_calls,
        "projection_iterations_per_project_call_budget": pressure_iterations,
        "cg_iteration_budget_per_physical_step_max": (
            total_project_calls * pressure_iterations
        ),
        "note": (
            "Budget only: this reports the current algorithmic projection-count upper "
            "bound and does not change pressure, velocity, flow, IBM force, or FSI "
            "coupling physics."
        ),
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


def resolve_divergence_cleanup_iterations(
    value: int,
    *,
    graded_grid_enabled: bool,
    value_was_explicit: bool = True,
) -> int:
    iterations = max(0, int(value))
    if graded_grid_enabled and iterations > 0:
        if not bool(value_was_explicit):
            return 0
        raise ValueError(
            "--use-graded-grid requires --divergence-cleanup-iterations 0 until "
            "non-uniform cleanup operators are implemented"
        )
    return iterations


def resolve_step_count(requested_steps: int | None, spec: SquidReducedSpec) -> int:
    if requested_steps is not None:
        steps = int(requested_steps)
        if steps <= 0:
            raise ValueError("--steps must be positive")
        return steps
    target_time_s = max(float(spec.pressure_t2_s), float(spec.dt_s))
    return max(1, int(math.ceil(target_time_s / max(float(spec.dt_s), 1.0e-12))))


def infer_spec(
    source_config_path: Path,
    grid_scale: float,
    time_step_scale: float = 1.0,
) -> SquidReducedSpec:
    if grid_scale <= 0.0:
        raise ValueError("--grid-scale must be positive")
    if time_step_scale <= 0.0:
        raise ValueError("--time-step-scale must be positive")
    config = load_source_config(source_config_path)
    analysis = config.get("analysis_settings", {}) if isinstance(config, dict) else {}
    domains = config.get("domains", {}) if isinstance(config, dict) else {}
    fluid_domain = domains.get("fluid", {}) if isinstance(domains, dict) else {}

    bounds_min_value = (
        analysis.get("fluid_bounds_min_m", (-0.09, -0.044, 0.9))
        if isinstance(analysis, dict)
        else (-0.09, -0.044, 0.9)
    )
    bounds_max_value = (
        analysis.get("fluid_bounds_max_m", (0.029, 0.076, 1.04))
        if isinstance(analysis, dict)
        else (0.029, 0.076, 1.04)
    )
    bounds_min = required_tuple3(
        bounds_min_value,
        field="analysis_settings.fluid_bounds_min_m",
    )
    bounds_max = required_tuple3(
        bounds_max_value,
        field="analysis_settings.fluid_bounds_max_m",
    )
    base_dt_s = float(analysis.get("time_step_s", 5.0e-4)) if isinstance(analysis, dict) else 5.0e-4
    dt_s = base_dt_s * float(time_step_scale)
    grid_size_m = 2.5e-3
    water_density_kgm3 = 1025.0
    water_viscosity_pa_s = 0.00105
    if isinstance(fluid_domain, dict):
        grid_size_m = float(fluid_domain.get("grid_size_m", grid_size_m))
        water_density_kgm3 = float(
            fluid_domain.get("density_kgm3", water_density_kgm3)
        )
        water_viscosity_pa_s = float(
            fluid_domain.get("viscosity_pa_s", water_viscosity_pa_s)
        )
    if isinstance(analysis, dict):
        grid_size_m = float(analysis.get("fluid_grid_size_m", grid_size_m) or grid_size_m)
        water_density_kgm3 = float(
            analysis.get("water_density_kgm3", water_density_kgm3)
            or water_density_kgm3
        )
        water_viscosity_pa_s = float(
            analysis.get("water_viscosity_pa_s", water_viscosity_pa_s)
            or water_viscosity_pa_s
        )
    if not math.isfinite(water_density_kgm3) or water_density_kgm3 <= 0.0:
        raise ValueError("water density must be finite and positive")
    if not math.isfinite(water_viscosity_pa_s) or water_viscosity_pa_s < 0.0:
        raise ValueError("water viscosity must be finite and non-negative")
    pressure_schedule = pressure_schedule_from_config(config, analysis)
    grid_size_m *= float(grid_scale)
    grid_nodes = tuple(
        max(8, int(math.ceil((hi - lo) / grid_size_m)))
        for lo, hi in zip(bounds_min, bounds_max, strict=True)
    )

    return SquidReducedSpec(
        source_config_path=str(source_config_path),
        fluid_bounds_min_m=bounds_min,
        fluid_bounds_max_m=bounds_max,
        grid_nodes=grid_nodes,
        dt_s=dt_s,
        water_density_kgm3=water_density_kgm3,
        water_viscosity_pa_s=water_viscosity_pa_s,
        base_dt_s=base_dt_s,
        **pressure_schedule,
    )


def _finite_positive_scale(value: float, *, option_name: str) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{option_name} must be a finite positive number")
    return scale


def spec_with_membrane_thickness_scale(
    spec: SquidReducedSpec,
    scale: float,
) -> SquidReducedSpec:
    thickness_scale = _finite_positive_scale(
        scale,
        option_name="--membrane-thickness-scale",
    )
    return replace(
        spec,
        main_membrane_thickness_m=(
            float(spec.main_membrane_thickness_m) * thickness_scale
        ),
        tail_membrane_thickness_m=(
            float(spec.tail_membrane_thickness_m) * thickness_scale
        ),
    )


def shell_surface_mass_budget(
    *,
    spec: SquidReducedSpec,
    density_kgm3: float,
    baseline_spec: SquidReducedSpec,
    baseline_density_kgm3: float,
) -> dict[str, float]:
    density = _finite_positive_scale(
        density_kgm3,
        option_name="density_kgm3",
    )
    baseline_density = _finite_positive_scale(
        baseline_density_kgm3,
        option_name="baseline_density_kgm3",
    )
    main_surface_mass = density * float(spec.main_membrane_thickness_m)
    tail_surface_mass = density * float(spec.tail_membrane_thickness_m)
    baseline_main_surface_mass = (
        baseline_density * float(baseline_spec.main_membrane_thickness_m)
    )
    baseline_tail_surface_mass = (
        baseline_density * float(baseline_spec.tail_membrane_thickness_m)
    )
    return {
        "density_kgm3": density,
        "baseline_density_kgm3": baseline_density,
        "main_membrane_thickness_m": float(spec.main_membrane_thickness_m),
        "tail_membrane_thickness_m": float(spec.tail_membrane_thickness_m),
        "baseline_main_membrane_thickness_m": float(
            baseline_spec.main_membrane_thickness_m
        ),
        "baseline_tail_membrane_thickness_m": float(
            baseline_spec.tail_membrane_thickness_m
        ),
        "main_surface_mass_kg_m2": main_surface_mass,
        "tail_surface_mass_kg_m2": tail_surface_mass,
        "baseline_main_surface_mass_kg_m2": baseline_main_surface_mass,
        "baseline_tail_surface_mass_kg_m2": baseline_tail_surface_mass,
        "main_surface_mass_scale": main_surface_mass
        / max(baseline_main_surface_mass, 1.0e-30),
        "tail_surface_mass_scale": tail_surface_mass
        / max(baseline_tail_surface_mass, 1.0e-30),
    }


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resume_history_rows_for_checkpoint(
    rows: list[dict[str, object]],
    *,
    completed_step: int,
) -> list[dict[str, object]]:
    checkpoint_step = int(completed_step)
    if checkpoint_step < 0:
        raise ValueError("completed_step must be non-negative")
    if len(rows) < checkpoint_step:
        raise ValueError(
            "resume requires history.csv to contain at least the checkpointed "
            f"steps: len(history)={len(rows)} checkpoint={checkpoint_step}"
        )
    return list(rows[:checkpoint_step])


def validate_resume_history_checkpoint_alignment(
    rows: list[dict[str, object]],
    *,
    completed_step: int,
    checkpoint_time_s: float,
    dt_s: float,
) -> None:
    checkpoint_step = int(completed_step)
    if checkpoint_step == 0:
        if rows:
            raise ValueError("resume history must be empty for a zero-step checkpoint")
        return
    if len(rows) != checkpoint_step:
        raise ValueError(
            "resume history row count must equal the checkpointed step count after truncation: "
            f"len(history)={len(rows)} checkpoint={checkpoint_step}"
        )
    try:
        history_step = int(rows[-1]["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume history final row must contain an integer step") from exc
    if history_step != checkpoint_step:
        raise ValueError(
            "resume history final row step does not match checkpoint: "
            f"history={history_step} checkpoint={checkpoint_step}"
        )
    try:
        history_time_s = float(rows[-1]["time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume history final row must contain finite time_s") from exc
    if not math.isfinite(history_time_s):
        raise ValueError("resume history final row must contain finite time_s")
    tolerance_s = max(abs(float(dt_s)) * 1.0e-4, 1.0e-7)
    if abs(history_time_s - float(checkpoint_time_s)) > tolerance_s:
        raise ValueError(
            "resume history final row time_s does not match checkpoint: "
            f"history={history_time_s:.9g} checkpoint={float(checkpoint_time_s):.9g}"
        )


def checkpoint_path_for_args(args: argparse.Namespace, output_dir: Path) -> Path:
    raw_path = getattr(args, "checkpoint_path", None)
    if raw_path:
        return Path(raw_path).resolve()
    return output_dir / RUN_CHECKPOINT_FILENAME


def _checkpoint_normalized_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_checkpoint_normalized_value(item) for item in value]
    if isinstance(value, list):
        return [_checkpoint_normalized_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_normalized_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    return value


def checkpoint_run_fingerprint(
    *,
    args: argparse.Namespace,
    spec: SquidReducedSpec,
    step_count: int,
    full_pressure_waveform_steps: int,
) -> dict[str, object]:
    spec_payload = asdict(spec)
    spec_payload["source_config_path"] = str(Path(spec.source_config_path).resolve())
    arg_payload = {
        name: getattr(args, name, None)
        for name in CHECKPOINT_ARG_FINGERPRINT_FIELDS
    }
    if arg_payload.get("source_config") is not None:
        arg_payload["source_config"] = str(Path(str(arg_payload["source_config"])).resolve())
    payload = {
        "requested_steps": int(step_count),
        "full_pressure_waveform_steps": int(full_pressure_waveform_steps),
        "spec": spec_payload,
        "args": arg_payload,
    }
    return _checkpoint_normalized_value(payload)  # type: ignore[return-value]


def validate_checkpoint_run_fingerprint(
    metadata: dict[str, object],
    *,
    args: argparse.Namespace,
    spec: SquidReducedSpec,
    step_count: int,
    full_pressure_waveform_steps: int,
) -> None:
    actual = metadata.get("run_fingerprint")
    expected = checkpoint_run_fingerprint(
        args=args,
        spec=spec,
        step_count=step_count,
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    if actual != expected:
        raise ValueError(
            "checkpoint run fingerprint does not match current configuration; "
            "restart with the same source config, pressure schedule, grid, solver, "
            "solid, and FSI options"
        )


def _array_to_payload(payload: dict[str, np.ndarray], name: str, value: object) -> None:
    payload[name] = np.asarray(value).copy()


def _read_scalar_field(field: ti.template()) -> float:
    return float(field[None])


def _write_scalar_field(field: ti.template(), value: object) -> None:
    field[None] = float(np.asarray(value))


def _read_vector_field(field: ti.template()) -> np.ndarray:
    value = field[None]
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)


def _write_vector_field(field: ti.template(), value: object) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    field[None] = ti.Vector([float(array[0]), float(array[1]), float(array[2])])


def _checkpoint_interface_state_dict(
    state: InterfaceReactionRelaxationState,
) -> dict[str, object]:
    return {
        "relaxation": float(state.relaxation),
        "previous_residual_n": (
            None
            if state.previous_residual_n is None
            else [float(value) for value in state.previous_residual_n]
        ),
        "previous_velocity_mps": (
            None
            if state.previous_velocity_mps is None
            else [float(value) for value in state.previous_velocity_mps]
        ),
    }


def _checkpoint_interface_vector(
    data: object,
    *,
    name: str,
) -> tuple[float, ...] | None:
    if data is None:
        return None
    try:
        vector = tuple(float(value) for value in data)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"checkpoint {name} must be a vector or null") from exc
    except ValueError as exc:
        raise ValueError(f"checkpoint {name} must contain numeric values") from exc
    if not vector:
        raise ValueError(f"checkpoint {name} must not be empty")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"checkpoint {name} must contain only finite values")
    return vector


def _interface_state_from_checkpoint(data: object) -> InterfaceReactionRelaxationState:
    if not isinstance(data, dict):
        raise ValueError("checkpoint interface_reaction_state must be an object")
    residual = _checkpoint_interface_vector(
        data.get("previous_residual_n"),
        name="previous_residual_n",
    )
    velocity = _checkpoint_interface_vector(
        data.get("previous_velocity_mps"),
        name="previous_velocity_mps",
    )
    if residual is not None and velocity is not None and len(residual) != len(velocity):
        raise ValueError(
            "checkpoint previous_residual_n and previous_velocity_mps must have the same length"
        )
    try:
        relaxation = float(data.get("relaxation", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint relaxation must be finite") from exc
    if not math.isfinite(relaxation):
        raise ValueError("checkpoint relaxation must be finite")
    return InterfaceReactionRelaxationState(
        previous_residual_n=residual,
        previous_velocity_mps=velocity,
        relaxation=relaxation,
    )


def write_run_checkpoint(
    path: Path,
    *,
    completed_step: int,
    step_count: int,
    full_pressure_waveform_steps: int,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    interface_reaction_state: InterfaceReactionRelaxationState,
) -> None:
    payload: dict[str, np.ndarray] = {}
    metadata = {
        "version": RUN_CHECKPOINT_VERSION,
        "completed_step": int(completed_step),
        "requested_steps": int(step_count),
        "full_pressure_waveform_steps": int(full_pressure_waveform_steps),
        "solid_model": str(args.solid_model),
        "grid_nodes": [int(value) for value in simulator.spec.grid_nodes],
        "particle_count": int(getattr(solid_mpm, "particle_count", 0)),
        "run_fingerprint": checkpoint_run_fingerprint(
            args=args,
            spec=simulator.spec,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
        ),
        "interface_reaction_state": _checkpoint_interface_state_dict(
            interface_reaction_state
        ),
    }
    _array_to_payload(payload, "__metadata__", np.asarray(json.dumps(metadata)))

    for name in (
        "time_s",
        "pressure_load_pa",
        "hydraulic_pressure_pa",
        "main_w_m",
        "main_v_mps",
        "tail_w_m",
        "tail_v_mps",
        "volume_flux_m3s",
        "nozzle_velocity_z_mps",
        "max_speed_mps",
        "lip_flow_z_m3s",
        "outlet_flow_z_m3s",
        "downstream_flow_z_m3s",
    ):
        _array_to_payload(payload, f"sim_{name}", _read_scalar_field(getattr(simulator, name)))
    for name in (
        "lip_sample_count",
        "outlet_sample_count",
        "downstream_sample_count",
    ):
        _array_to_payload(payload, f"sim_{name}", int(getattr(simulator, name)[None]))
    for name in (
        "primary_interface_reaction_force_n",
        "secondary_interface_reaction_force_n",
    ):
        _array_to_payload(payload, f"sim_{name}", _read_vector_field(getattr(simulator, name)))

    fluid = simulator.fluid
    for name in ("velocity", "velocity_prev", "pressure"):
        _array_to_payload(payload, f"fluid_{name}", getattr(fluid, name).to_numpy())

    if args.solid_model == "tri_mooney_shell_mpm":
        for name in ("x", "u", "v"):
            _array_to_payload(payload, f"solid_{name}", getattr(solid_mpm, name).to_numpy())
    elif args.solid_model == "neo_hookean_mpm":
        for name in ("x", "v", "C", "F"):
            _array_to_payload(payload, f"solid_{name}", getattr(solid_mpm, name).to_numpy())
    else:
        raise ValueError(f"unsupported solid model for checkpoint: {args.solid_model!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)


def load_run_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    simulator: ReducedSquidFSI,
    solid_mpm: object,
    step_count: int | None = None,
    full_pressure_waveform_steps: int | None = None,
) -> tuple[int, InterfaceReactionRelaxationState]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["__metadata__"]))
        if int(metadata.get("version", -1)) != RUN_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {metadata.get('version')!r}"
            )
        if str(metadata.get("solid_model")) != str(args.solid_model):
            raise ValueError(
                "checkpoint solid model does not match --solid-model: "
                f"{metadata.get('solid_model')!r} != {args.solid_model!r}"
            )
        if tuple(int(value) for value in metadata.get("grid_nodes", ())) != tuple(
            int(value) for value in simulator.spec.grid_nodes
        ):
            raise ValueError("checkpoint grid shape does not match current configuration")
        if int(metadata.get("particle_count", -1)) != int(getattr(solid_mpm, "particle_count", 0)):
            raise ValueError("checkpoint solid particle count does not match current configuration")
        validate_checkpoint_run_fingerprint(
            metadata,
            args=args,
            spec=simulator.spec,
            step_count=(
                int(metadata["requested_steps"])
                if step_count is None
                else int(step_count)
            ),
            full_pressure_waveform_steps=(
                int(metadata["full_pressure_waveform_steps"])
                if full_pressure_waveform_steps is None
                else int(full_pressure_waveform_steps)
            ),
        )

        for name in (
            "time_s",
            "pressure_load_pa",
            "hydraulic_pressure_pa",
            "main_w_m",
            "main_v_mps",
            "tail_w_m",
            "tail_v_mps",
            "volume_flux_m3s",
            "nozzle_velocity_z_mps",
            "max_speed_mps",
            "lip_flow_z_m3s",
            "outlet_flow_z_m3s",
            "downstream_flow_z_m3s",
        ):
            _write_scalar_field(getattr(simulator, name), checkpoint[f"sim_{name}"])
        for name in (
            "lip_sample_count",
            "outlet_sample_count",
            "downstream_sample_count",
        ):
            getattr(simulator, name)[None] = int(np.asarray(checkpoint[f"sim_{name}"]))
        for name in (
            "primary_interface_reaction_force_n",
            "secondary_interface_reaction_force_n",
        ):
            _write_vector_field(getattr(simulator, name), checkpoint[f"sim_{name}"])

        fluid = simulator.fluid
        for name in ("velocity", "velocity_prev", "pressure"):
            getattr(fluid, name).from_numpy(checkpoint[f"fluid_{name}"])
        fluid.pressure_tmp.from_numpy(checkpoint["fluid_pressure"])
        fluid.pressure_accum.from_numpy(checkpoint["fluid_pressure"])

        if args.solid_model == "tri_mooney_shell_mpm":
            for name in ("x", "u", "v"):
                getattr(solid_mpm, name).from_numpy(checkpoint[f"solid_{name}"])
        elif args.solid_model == "neo_hookean_mpm":
            for name in ("x", "v", "C", "F"):
                getattr(solid_mpm, name).from_numpy(checkpoint[f"solid_{name}"])
        else:
            raise ValueError(f"unsupported solid model for checkpoint: {args.solid_model!r}")

        return (
            int(metadata["completed_step"]),
            _interface_state_from_checkpoint(metadata.get("interface_reaction_state")),
        )


def _raise_for_step_numerical_guard(
    row: dict[str, object],
    *,
    cfl_limit: float,
    divergence_l2_limit: float,
) -> None:
    step = row.get("step")
    finite_fields = (
        "max_fluid_speed_mps",
        "cfl",
        "divergence_l2",
        "divergence_max_abs",
        "interior_divergence_l2",
        "interior_divergence_max_abs",
        "pressure_correctable_divergence_l2",
        "pressure_correctable_divergence_max_abs",
        "pressure_fixed_divergence_l2",
        "pressure_fixed_divergence_max_abs",
        "interior_pressure_correctable_divergence_l2",
        "interior_pressure_correctable_divergence_max_abs",
        "interior_pressure_fixed_divergence_l2",
        "interior_pressure_fixed_divergence_max_abs",
        "projection_divergence_l2",
        "projection_divergence_max_abs",
        "post_boundary_divergence_l2",
        "post_boundary_divergence_max_abs",
        "post_constraint_divergence_l2",
        "post_constraint_divergence_max_abs",
    )
    values: dict[str, float] = {}
    for field in finite_fields:
        values[field] = _required_finite_row_number(
            row,
            field,
            context=f"step {step} numerical guard",
        )
    if values["cfl"] >= float(cfl_limit):
        raise RuntimeError(
            f"step {step} numerical guard failed: cfl={values['cfl']:.6e} "
            f">= {float(cfl_limit):.6e}"
        )
    if values["interior_divergence_l2"] > float(divergence_l2_limit):
        raise RuntimeError(
            f"step {step} numerical guard failed: interior_divergence_l2="
            f"{values['interior_divergence_l2']:.6e} > {float(divergence_l2_limit):.6e}"
        )
    for converged_field, breakdown_field in (
        (
            "total_pressure_projection_cg_converged_all",
            "total_pressure_projection_cg_breakdown_count",
        ),
        (
            "pressure_projection_cg_converged_all",
            "pressure_projection_cg_breakdown_count",
        ),
    ):
        if converged_field in row and not _row_bool(row[converged_field]):
            raise RuntimeError(
                f"step {step} numerical guard failed: {converged_field}=false"
            )
        if breakdown_field in row:
            breakdown_count = _required_finite_row_number(
                row,
                breakdown_field,
                context=f"step {step} numerical guard",
            )
            if breakdown_count > 0.0:
                raise RuntimeError(
                    f"step {step} numerical guard failed: {breakdown_field}="
                    f"{breakdown_count:.0f}"
                )


def _ascii_vtk_numbers(values: np.ndarray, *, precision: int = 9) -> str:
    flat = np.asarray(values).reshape(-1)
    return " ".join(f"{float(value):.{precision}g}" for value in flat)


def _write_fluid_snapshot_npz(
    *,
    snapshot_dir: Path,
    step: int,
    fluid,
    markers,
    marker_count: int,
    time_s: float,
    pressure_pa: float,
) -> Path | None:
    """Dump a compact per-step visualization snapshot (case-level host read)."""
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        velocity = np.asarray(fluid.velocity.to_numpy(), dtype=np.float32)
        obstacle = np.asarray(fluid.obstacle.to_numpy(), dtype=np.int8)
        speed = np.linalg.norm(velocity, axis=-1).astype(np.float32)
        positions = np.asarray(
            markers.x_gamma_m.to_numpy()[: int(marker_count)],
            dtype=np.float32,
        )
        nx, ny, nz = speed.shape
        path = snapshot_dir / f"snapshot_{int(step):06d}.npz"
        np.savez_compressed(
            path,
            step=np.int64(step),
            time_s=np.float64(time_s),
            pressure_pa=np.float64(pressure_pa),
            speed_xz=speed[:, ny // 2, :],
            speed_yz=speed[nx // 2, :, :],
            obstacle_xz=obstacle[:, ny // 2, :],
            marker_positions_m=positions,
            cell_center_x_m=np.asarray(
                fluid.cell_center_x_m.to_numpy(), dtype=np.float32
            ),
            cell_center_y_m=np.asarray(
                fluid.cell_center_y_m.to_numpy(), dtype=np.float32
            ),
            cell_center_z_m=np.asarray(
                fluid.cell_center_z_m.to_numpy(), dtype=np.float32
            ),
        )
        return path
    except Exception as exc:  # noqa: BLE001 - snapshot must not kill a long run
        print(f"[snapshot] step {step} failed: {exc}", flush=True)
        return None


def _write_minimal_fluid_vti(
    *,
    output_dir: Path,
    step: int,
    fluid,
) -> Path | None:
    try:
        velocity = np.asarray(fluid.velocity.to_numpy(), dtype=np.float32)
        obstacle = np.asarray(fluid.obstacle.to_numpy(), dtype=np.int32)
        divergence = np.asarray(fluid.divergence.to_numpy(), dtype=np.float32)
        if velocity.ndim != 4 or velocity.shape[-1] != 3:
            return None
        if obstacle.shape != velocity.shape[:3] or divergence.shape != velocity.shape[:3]:
            return None
        nx, ny, nz = (int(value) for value in velocity.shape[:3])
        if nx <= 0 or ny <= 0 or nz <= 0:
            return None
        x_centers = np.asarray(fluid.cell_center_x_m.to_numpy(), dtype=np.float64)
        y_centers = np.asarray(fluid.cell_center_y_m.to_numpy(), dtype=np.float64)
        z_centers = np.asarray(fluid.cell_center_z_m.to_numpy(), dtype=np.float64)
        width_x = np.asarray(fluid.cell_width_x_m.to_numpy(), dtype=np.float64)
        width_y = np.asarray(fluid.cell_width_y_m.to_numpy(), dtype=np.float64)
        width_z = np.asarray(fluid.cell_width_z_m.to_numpy(), dtype=np.float64)
        spacing = (
            float(np.mean(width_x)) if width_x.size else 1.0,
            float(np.mean(width_y)) if width_y.size else 1.0,
            float(np.mean(width_z)) if width_z.size else 1.0,
        )
        origin = (
            float(x_centers[0]) if x_centers.size else 0.0,
            float(y_centers[0]) if y_centers.size else 0.0,
            float(z_centers[0]) if z_centers.size else 0.0,
        )
        speed = np.linalg.norm(velocity, axis=3).astype(np.float32)
        active_fluid = (obstacle == 0).astype(np.int32)
        path = output_dir / f"sharp_failure_step_{int(step):06d}_fluid.vti"
        extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
        text = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
            f'  <ImageData WholeExtent="{extent}" '
            f'Origin="{origin[0]:.9g} {origin[1]:.9g} {origin[2]:.9g}" '
            f'Spacing="{spacing[0]:.9g} {spacing[1]:.9g} {spacing[2]:.9g}">\n'
            f'    <Piece Extent="{extent}">\n'
            '      <PointData Scalars="speed_mps" Vectors="velocity_mps">\n'
            '        <DataArray type="Float32" Name="velocity_mps" '
            'NumberOfComponents="3" format="ascii">\n'
            f'          {_ascii_vtk_numbers(velocity)}\n'
            '        </DataArray>\n'
            '        <DataArray type="Float32" Name="speed_mps" format="ascii">\n'
            f'          {_ascii_vtk_numbers(speed)}\n'
            '        </DataArray>\n'
            '        <DataArray type="Int32" Name="obstacle" format="ascii">\n'
            f'          {" ".join(str(int(value)) for value in obstacle.reshape(-1))}\n'
            '        </DataArray>\n'
            '        <DataArray type="Int32" Name="active_fluid" format="ascii">\n'
            f'          {" ".join(str(int(value)) for value in active_fluid.reshape(-1))}\n'
            '        </DataArray>\n'
            '        <DataArray type="Float32" Name="divergence" format="ascii">\n'
            f'          {_ascii_vtk_numbers(divergence)}\n'
            '        </DataArray>\n'
            '      </PointData>\n'
            '      <CellData/>\n'
            '    </Piece>\n'
            '  </ImageData>\n'
            '</VTKFile>\n'
        )
        path.write_text(text, encoding="utf-8")
        return path
    except (AttributeError, OSError, ValueError, TypeError):
        return None


def _write_step_failure_artifacts(
    *,
    process_path: Path,
    output_dir: Path,
    rows: list[dict[str, object]],
    step: int,
    exc: Exception,
    fluid=None,
) -> Path:
    partial_history_path = output_dir / "history.csv"
    write_csv(partial_history_path, rows)
    failure_fluid_vti = (
        _write_minimal_fluid_vti(output_dir=output_dir, step=step, fluid=fluid)
        if fluid is not None
        else None
    )
    process_payload: dict[str, object] = {}
    if process_path.exists():
        try:
            parsed = json.loads(process_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                process_payload.update(parsed)
        except (OSError, json.JSONDecodeError):
            pass
    process_payload.update(
        {
            "pid": os.getpid(),
            "status": "failed",
            "failed_at_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_step": int(step),
            "history_csv": str(partial_history_path),
        }
    )
    if failure_fluid_vti is not None:
        process_payload["failure_fluid_vti"] = str(failure_fluid_vti)
    process_path.write_text(
        json.dumps(process_payload, indent=2),
        encoding="utf-8",
    )
    return partial_history_path


def _pressure_correctable_mask_from_host_fields(
    *,
    obstacle: np.ndarray,
    velocity_dirichlet_active: np.ndarray,
    pressure_outlet_zmin: bool,
) -> np.ndarray:
    obstacle_mask = obstacle != 0
    fixed = velocity_dirichlet_active != 0
    nx, ny, nz = obstacle_mask.shape
    correctable = np.zeros(obstacle_mask.shape, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if obstacle_mask[i, j, k]:
                    continue
                if i > 0 and not obstacle_mask[i - 1, j, k] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    i < nx - 1
                    and not obstacle_mask[i + 1, j, k]
                    and not fixed[i + 1, j, k]
                ):
                    correctable[i, j, k] = True
                if j > 0 and not obstacle_mask[i, j - 1, k] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    j < ny - 1
                    and not obstacle_mask[i, j + 1, k]
                    and not fixed[i, j + 1, k]
                ):
                    correctable[i, j, k] = True
                if k > 0 and not obstacle_mask[i, j, k - 1] and not fixed[i, j, k]:
                    correctable[i, j, k] = True
                if (
                    k < nz - 1
                    and not obstacle_mask[i, j, k + 1]
                    and not fixed[i, j, k + 1]
                ):
                    correctable[i, j, k] = True
                if pressure_outlet_zmin and k == 0 and not fixed[i, j, k]:
                    correctable[i, j, k] = True
    return correctable


def _write_hibm_zero_correctable_cell_dump(
    *,
    output_dir: Path,
    step: int,
    fluid,
    markers,
    pressure_outlet_zmin: bool,
) -> dict[str, object]:
    obstacle = fluid.obstacle.to_numpy()
    velocity_dirichlet_active = fluid.velocity_dirichlet_boundary_active.to_numpy()
    correctable = _pressure_correctable_mask_from_host_fields(
        obstacle=obstacle,
        velocity_dirichlet_active=velocity_dirichlet_active,
        pressure_outlet_zmin=pressure_outlet_zmin,
    )
    active_fluid = obstacle == 0
    interior = np.zeros(active_fluid.shape, dtype=bool)
    if all(axis_size > 2 for axis_size in active_fluid.shape):
        interior[1:-1, 1:-1, 1:-1] = True
    zero_correctable = active_fluid & interior & ~correctable
    indices = np.argwhere(zero_correctable)

    x_centers = fluid.cell_center_x_m.to_numpy()
    y_centers = fluid.cell_center_y_m.to_numpy()
    z_centers = fluid.cell_center_z_m.to_numpy()
    width_x = fluid.cell_width_x_m.to_numpy()
    width_y = fluid.cell_width_y_m.to_numpy()
    width_z = fluid.cell_width_z_m.to_numpy()
    divergence = fluid.divergence.to_numpy()
    volume_source = fluid.volume_source_s.to_numpy()

    marker_count = int(markers.marker_count)
    marker_positions = markers.x_gamma_m.to_numpy()[:marker_count]
    marker_normals = markers.n_gamma.to_numpy()[:marker_count]
    marker_regions = markers.region_id.to_numpy()[:marker_count]
    nearest_index = np.full(indices.shape[0], -1, dtype=np.int64)
    nearest_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_signed_distance = np.full(indices.shape[0], math.nan, dtype=np.float64)
    nearest_region = np.full(indices.shape[0], -1, dtype=np.int64)
    if marker_count > 0 and len(indices) > 0:
        positions = np.column_stack(
            (
                x_centers[indices[:, 0]],
                y_centers[indices[:, 1]],
                z_centers[indices[:, 2]],
            )
        )
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            delta = positions[start:end, None, :] - marker_positions[None, :, :]
            distance2 = np.einsum("cmq,cmq->cm", delta, delta)
            local_index = np.argmin(distance2, axis=1)
            global_index = local_index.astype(np.int64)
            local_delta = delta[np.arange(end - start), local_index, :]
            local_normals = marker_normals[global_index]
            nearest_index[start:end] = global_index
            nearest_distance[start:end] = np.sqrt(distance2[np.arange(end - start), local_index])
            nearest_signed_distance[start:end] = np.einsum(
                "cq,cq->c",
                local_delta,
                local_normals,
            )
            nearest_region[start:end] = marker_regions[global_index]

    dump_dir = output_dir / "hibm_zero_correctable_cells"
    dump_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dump_dir / f"step_{int(step):06d}_interior_zero_correctable_cells.csv"
    fieldnames = (
        "i",
        "j",
        "k",
        "x_m",
        "y_m",
        "z_m",
        "divergence_s",
        "volume_source_s",
        "residual_s",
        "nearest_marker_index",
        "nearest_marker_region_id",
        "nearest_marker_distance_m",
        "nearest_marker_signed_distance_m",
        "local_cell_diagonal_m",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (i, j, k) in enumerate(indices):
            div = float(divergence[i, j, k])
            src = float(volume_source[i, j, k])
            writer.writerow(
                {
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "x_m": float(x_centers[i]),
                    "y_m": float(y_centers[j]),
                    "z_m": float(z_centers[k]),
                    "divergence_s": div,
                    "volume_source_s": src,
                    "residual_s": div - src,
                    "nearest_marker_index": int(nearest_index[row_index]),
                    "nearest_marker_region_id": int(nearest_region[row_index]),
                    "nearest_marker_distance_m": float(nearest_distance[row_index]),
                    "nearest_marker_signed_distance_m": float(
                        nearest_signed_distance[row_index]
                    ),
                    "local_cell_diagonal_m": math.sqrt(
                        float(width_x[i]) ** 2
                        + float(width_y[j]) ** 2
                        + float(width_z[k]) ** 2
                    ),
                }
            )

    local_diagonal = np.sqrt(
        width_x[indices[:, 0]] ** 2
        + width_y[indices[:, 1]] ** 2
        + width_z[indices[:, 2]] ** 2
    ) if len(indices) > 0 else np.array([], dtype=np.float64)
    shell_band_candidate = (
        np.isfinite(nearest_distance)
        & (nearest_distance <= local_diagonal)
        if len(indices) > 0
        else np.array([], dtype=bool)
    )
    region_counts: dict[str, int] = {}
    for region in nearest_region:
        key = str(int(region))
        region_counts[key] = region_counts.get(key, 0) + 1
    summary = {
        "step": int(step),
        "pressure_outlet_zmin": bool(pressure_outlet_zmin),
        "zero_correctable_interior_cell_count": int(len(indices)),
        "active_fluid_cell_count": int(np.count_nonzero(active_fluid)),
        "pressure_correctable_cell_count": int(np.count_nonzero(active_fluid & correctable)),
        "nearest_marker_count": int(marker_count),
        "nearest_marker_region_counts": region_counts,
        "shell_band_candidate_cell_count": int(np.count_nonzero(shell_band_candidate)),
        "shell_band_candidate_rule": (
            "nearest marker distance <= local cell diagonal"
        ),
        "nearest_marker_distance_min_m": (
            float(np.nanmin(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_distance_mean_m": (
            float(np.nanmean(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_distance_max_m": (
            float(np.nanmax(nearest_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_min_m": (
            float(np.nanmin(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_mean_m": (
            float(np.nanmean(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "nearest_marker_signed_distance_max_m": (
            float(np.nanmax(nearest_signed_distance)) if len(indices) > 0 else math.nan
        ),
        "negative_signed_distance_count": int(
            np.count_nonzero(nearest_signed_distance < 0.0)
        ),
        "positive_signed_distance_count": int(
            np.count_nonzero(nearest_signed_distance > 0.0)
        ),
        "csv_path": str(csv_path),
    }
    if len(indices) > 0:
        summary["i_min"] = int(np.min(indices[:, 0]))
        summary["i_max"] = int(np.max(indices[:, 0]))
        summary["j_min"] = int(np.min(indices[:, 1]))
        summary["j_max"] = int(np.max(indices[:, 1]))
        summary["k_min"] = int(np.min(indices[:, 2]))
        summary["k_max"] = int(np.max(indices[:, 2]))
    summary_path = dump_dir / f"step_{int(step):06d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def reduced_active_water_connectivity(
    spec: SquidReducedSpec,
    obstacle_cell_count: int,
    obstacle_mask: np.ndarray | None = None,
) -> dict[str, object]:
    total_cells = int(spec.grid_nodes[0] * spec.grid_nodes[1] * spec.grid_nodes[2])
    active_cells = total_cells - int(obstacle_cell_count)
    if obstacle_mask is None:
        return {
            "method": "latest_core_reduced_chamber_nozzle_obstacle_seeded_from_z_min_analytic_fallback",
            "component_count": 1,
            "seed_boundary": "z_min",
            "active_cell_count": active_cells,
            "inactive_cell_count": int(obstacle_cell_count),
            "z_min_connected_active_cell_count": active_cells,
            "trapped_active_cell_count": 0,
            "connectivity_passed": active_cells > 0,
            "limitation": "No obstacle mask was supplied, so connectivity fell back to the legacy analytic count.",
        }
    mask = np.asarray(obstacle_mask, dtype=np.int32)
    if mask.shape != tuple(spec.grid_nodes):
        raise ValueError(
            f"obstacle_mask shape {mask.shape!r} does not match grid_nodes {tuple(spec.grid_nodes)!r}"
        )
    active = mask == 0
    active_cells = int(np.count_nonzero(active))
    inactive_cells = int(mask.size - active_cells)
    visited = np.zeros(active.shape, dtype=bool)
    component_count = 0
    z_min_connected_active_cells = 0
    trapped_active_cells = 0
    nx, ny, nz = active.shape
    for seed_index in zip(*np.nonzero(active), strict=False):
        if visited[seed_index]:
            continue
        component_count += 1
        stack = [tuple(int(value) for value in seed_index)]
        visited[seed_index] = True
        component_size = 0
        touches_z_min = False
        while stack:
            i, j, k = stack.pop()
            component_size += 1
            touches_z_min = touches_z_min or k == 0
            for ni, nj, nk in (
                (i - 1, j, k),
                (i + 1, j, k),
                (i, j - 1, k),
                (i, j + 1, k),
                (i, j, k - 1),
                (i, j, k + 1),
            ):
                if (
                    0 <= ni < nx
                    and 0 <= nj < ny
                    and 0 <= nk < nz
                    and active[ni, nj, nk]
                    and not visited[ni, nj, nk]
                ):
                    visited[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
        if touches_z_min:
            z_min_connected_active_cells += component_size
        else:
            trapped_active_cells += component_size
    return {
        "method": "latest_core_reduced_chamber_nozzle_obstacle_flood_fill_from_z_min",
        "component_count": component_count,
        "seed_boundary": "z_min",
        "active_cell_count": active_cells,
        "inactive_cell_count": inactive_cells,
        "z_min_connected_active_cell_count": z_min_connected_active_cells,
        "trapped_active_cell_count": trapped_active_cells,
        "connectivity_passed": active_cells > 0 and trapped_active_cells == 0,
    }


def _mark_existing_run_process_failed(args: argparse.Namespace, exc: Exception) -> None:
    try:
        process_path = Path(args.output_dir).resolve() / "run_process.json"
        process_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {}
        if process_path.exists():
            try:
                parsed = json.loads(process_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except (OSError, json.JSONDecodeError):
                pass
        payload.update(
            {
                "pid": os.getpid(),
                "status": "failed",
                "failed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command": payload.get("command", " ".join(sys.argv)),
                "uses_generic_simulation_core": True,
            }
        )
        process_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _run_process_failure_guard(func):
    @wraps(func)
    def wrapper(args: argparse.Namespace) -> dict[str, object]:
        try:
            return func(args)
        except Exception as exc:
            _mark_existing_run_process_failed(args, exc)
            raise

    return wrapper


@_run_process_failure_guard
def run(args: argparse.Namespace) -> dict[str, object]:
    membrane_thickness_scale = _finite_positive_scale(
        args.membrane_thickness_scale,
        option_name="--membrane-thickness-scale",
    )
    solid_density_scale = _finite_positive_scale(
        args.solid_density_scale,
        option_name="--solid-density-scale",
    )
    interface_reaction_relaxation = float(args.interface_reaction_relaxation)
    if not math.isfinite(interface_reaction_relaxation) or not 0.0 <= interface_reaction_relaxation <= 1.0:
        raise ValueError("--interface-reaction-relaxation must be a finite number in [0, 1]")
    fsi_constraint_force_solid_mobility_ratio = float(
        args.fsi_constraint_force_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_constraint_force_solid_mobility_ratio)
        or fsi_constraint_force_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-constraint-force-solid-mobility-ratio must be a finite non-negative number"
        )
    fsi_solid_response_mobility_coupling = bool(
        args.fsi_solid_response_mobility_coupling
    )
    fsi_velocity_target_solid_mobility_ratio = float(
        args.fsi_velocity_target_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_velocity_target_solid_mobility_ratio)
        or fsi_velocity_target_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-velocity-target-solid-mobility-ratio must be a finite "
            "non-negative number"
        )
    fsi_solid_response_velocity_mobility_coupling = bool(
        args.fsi_solid_response_velocity_mobility_coupling
    )
    fsi_velocity_constraint_blend = float(args.fsi_velocity_constraint_blend)
    if not math.isfinite(fsi_velocity_constraint_blend) or not 0.0 <= fsi_velocity_constraint_blend <= 1.0:
        raise ValueError("--fsi-velocity-constraint-blend must be a finite number in [0, 1]")
    fsi_velocity_constraint_solid_mobility_ratio = float(
        args.fsi_velocity_constraint_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_velocity_constraint_solid_mobility_ratio)
        or fsi_velocity_constraint_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-velocity-constraint-solid-mobility-ratio must be a finite non-negative number"
        )
    fsi_coupling_iterations = max(1, int(args.fsi_coupling_iterations))
    fsi_coupling_tolerance_n = float(args.fsi_coupling_tolerance_n)
    if not math.isfinite(fsi_coupling_tolerance_n) or fsi_coupling_tolerance_n < 0.0:
        raise ValueError("--fsi-coupling-tolerance-n must be a finite non-negative number")
    fsi_coupling_target_map_relaxation = float(args.fsi_coupling_target_map_relaxation)
    if (
        not math.isfinite(fsi_coupling_target_map_relaxation)
        or not 0.0 < fsi_coupling_target_map_relaxation <= 1.0
    ):
        raise ValueError("--fsi-coupling-target-map-relaxation must be a finite number in (0, 1]")
    fsi_coupling_solver = str(args.fsi_coupling_solver)
    if fsi_coupling_solver not in INTERFACE_REACTION_SOLVER_CHOICES:
        choices = ", ".join(INTERFACE_REACTION_SOLVER_CHOICES)
        raise ValueError(f"--fsi-coupling-solver must be one of: {choices}")
    fsi_coupling_mode = str(args.fsi_coupling_mode)
    fsi_coupling_mode_report = require_implemented_fsi_coupling_mode(fsi_coupling_mode)
    sharp_case_runner_enabled = fsi_coupling_mode == FSI_COUPLING_MODE_HIBM_MPM_SHARP
    reuse_accepted_fsi_trial_state = bool(args.reuse_accepted_fsi_trial_state)
    pressure_outlet_source_ratio_tolerance = float(args.pressure_outlet_source_ratio_tolerance)
    if not math.isfinite(pressure_outlet_source_ratio_tolerance) or pressure_outlet_source_ratio_tolerance < 0.0:
        raise ValueError("--pressure-outlet-source-ratio-tolerance must be a finite non-negative number")
    cg_tolerance = float(args.cg_tolerance)
    if not math.isfinite(cg_tolerance) or cg_tolerance < 0.0:
        raise ValueError("--cg-tolerance must be a finite non-negative number")
    cg_preconditioner = str(args.cg_preconditioner)
    if cg_preconditioner not in CG_PRECONDITIONER_CHOICES:
        choices = ", ".join(CG_PRECONDITIONER_CHOICES)
        raise ValueError(f"--cg-preconditioner must be one of: {choices}")
    interface_reaction_passivity_limit = bool(args.interface_reaction_passivity_limit)
    interface_reaction_robin_impedance_ns_m = float(
        args.interface_reaction_robin_impedance_ns_m
    )
    if (
        not math.isfinite(interface_reaction_robin_impedance_ns_m)
        or interface_reaction_robin_impedance_ns_m < 0.0
    ):
        raise ValueError(
            "--interface-reaction-robin-impedance-ns-m must be a finite "
            "non-negative number"
        )
    interface_reaction_robin_matrix_impedance_ns_m = float(
        args.interface_reaction_robin_matrix_impedance_ns_m
    )
    if (
        not math.isfinite(interface_reaction_robin_matrix_impedance_ns_m)
        or interface_reaction_robin_matrix_impedance_ns_m < 0.0
    ):
        raise ValueError(
            "--interface-reaction-robin-matrix-impedance-ns-m must be a "
            "finite non-negative number"
        )
    interface_reaction_robin_target_mode = str(args.interface_reaction_robin_target_mode)
    if interface_reaction_robin_target_mode not in INTERFACE_REACTION_ROBIN_TARGET_CHOICES:
        choices = ", ".join(INTERFACE_REACTION_ROBIN_TARGET_CHOICES)
        raise ValueError(f"--interface-reaction-robin-target-mode must be one of: {choices}")
    raise_for_unsupported_hibm_mpm_sharp_robin_options(
        fsi_coupling_mode=fsi_coupling_mode,
        interface_reaction_robin_impedance_ns_m=(
            interface_reaction_robin_impedance_ns_m
        ),
        interface_reaction_robin_matrix_impedance_ns_m=(
            interface_reaction_robin_matrix_impedance_ns_m
        ),
    )
    interface_reaction_aitken = bool(args.interface_reaction_aitken)
    max_wall_time_s = float(args.max_wall_time_s)
    if not math.isfinite(max_wall_time_s) or max_wall_time_s < 0.0:
        raise ValueError("--max-wall-time-s must be a finite non-negative number")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoint_path = checkpoint_path_for_args(args, output_dir)
    process_path = output_dir / "run_process.json"
    run_started_at_unix = time.time()
    run_started_at_perf = time.perf_counter()
    process_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "running",
                "started_at_unix": run_started_at_unix,
                "command": " ".join(sys.argv),
                "uses_generic_simulation_core": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    source_config_path = Path(args.source_config).resolve()
    spec = infer_spec(
        source_config_path,
        grid_scale=args.grid_scale,
        time_step_scale=args.time_step_scale,
    )
    baseline_spec = spec
    spec = spec_with_membrane_thickness_scale(spec, membrane_thickness_scale)
    baseline_material = ecoflex_0010_material(poissons_ratio=args.poissons_ratio)
    solid_density_kgm3 = (
        float(baseline_material.density_kgm3) * solid_density_scale
    )
    material = replace(baseline_material, density_kgm3=solid_density_kgm3)
    material.validate()
    solid_surface_mass_report = shell_surface_mass_budget(
        spec=spec,
        density_kgm3=material.density_kgm3,
        baseline_spec=baseline_spec,
        baseline_density_kgm3=baseline_material.density_kgm3,
    )
    source_config = load_source_config(source_config_path)
    region14_aperture_geometry = compute_region_geometry_stats(source_config, 14)
    tail_refinement_geometry: dict[str, object] = {
        "available": False,
        "region_id": 8,
        "reason": "not_requested",
    }
    tail_refinement_region: RefinementRegion | None = None
    if args.use_region14_aperture_carve:
        spec = spec_with_region14_aperture(
            spec,
            region14_aperture_geometry,
            open_downstream_farfield=args.open_downstream_farfield,
        )
    if args.use_nozzle_taper:
        spec = spec_with_nozzle_taper(
            spec,
            taper_length_m=args.nozzle_taper_length_m,
            inlet_radius_m=args.nozzle_taper_inlet_radius_m,
        )
    solid_mpm_grid_nodes = spec.grid_nodes
    if args.use_tail_refinement:
        if not args.use_graded_grid:
            raise ValueError("--use-tail-refinement requires --use-graded-grid")
        tail_refinement_geometry = compute_region_geometry_stats(source_config, 8)
        tail_target_spacing_m = (
            float(args.tail_refinement_target_spacing_m)
            if args.tail_refinement_target_spacing_m is not None
            else min(float(spec.tail_membrane_thickness_m), float(args.graded_grid_farfield_spacing_m))
        )
        tail_padding_m = (
            float(args.tail_refinement_padding_m)
            if args.tail_refinement_padding_m is not None
            else 2.0 * tail_target_spacing_m
        )
        tail_refinement_region = tail_refinement_region_from_geometry(
            spec,
            tail_refinement_geometry,
            target_spacing_m=tail_target_spacing_m,
            padding_m=tail_padding_m,
        )
        if tail_refinement_region is None:
            raise ValueError(
                "--use-tail-refinement requires available source-config region 8 tail FSI geometry"
            )
    if args.use_graded_grid:
        spec = spec_with_nozzle_graded_grid(
            spec,
            target_spacing_m=args.graded_grid_target_spacing_m,
            farfield_spacing_m=float(args.graded_grid_farfield_spacing_m),
            max_growth_ratio=float(args.graded_grid_growth_ratio),
            max_cells=args.graded_grid_max_cells,
            extra_refinement_regions=(
                () if tail_refinement_region is None else (tail_refinement_region,)
            ),
        )
    graded_grid_enabled = spec.graded_grid is not None
    full_pressure_waveform_steps = resolve_step_count(None, spec)
    step_count = resolve_step_count(args.steps, spec)
    pressure_solver_name = resolve_pressure_solver(
        args.pressure_solver,
        graded_grid_enabled=graded_grid_enabled,
    )
    if (
        interface_reaction_robin_matrix_impedance_ns_m > 0.0
        and pressure_solver_name != "fv_cg"
    ):
        raise ValueError(
            "--interface-reaction-robin-matrix-impedance-ns-m requires "
            "--pressure-solver fv_cg so the interface impedance enters the "
            "pressure matrix"
        )
    projection_divergence_cleanup_iterations = resolve_divergence_cleanup_iterations(
        args.divergence_cleanup_iterations,
        graded_grid_enabled=graded_grid_enabled,
        value_was_explicit=bool(
            getattr(args, "divergence_cleanup_iterations_explicit", True)
        ),
    )
    multigrid_cycles = None if args.multigrid_cycles is None else int(args.multigrid_cycles)
    if multigrid_cycles is not None and multigrid_cycles <= 0:
        raise ValueError("--multigrid-cycles must be positive")
    grid_for_effective_cycles = cartesian_grid_for_spec(spec)
    effective_multigrid_cycles = (
        (
            CartesianFluidSolver.DEFAULT_MULTIGRID_CYCLES
            if grid_for_effective_cycles.is_uniform
            else CartesianFluidSolver.DEFAULT_NONUNIFORM_MULTIGRID_CYCLES
        )
        if pressure_solver_name == "fv_multigrid" and multigrid_cycles is None
        else multigrid_cycles
    )
    effective_fluid_substeps = effective_fluid_substeps_for_grid(
        spec,
        args.fluid_substeps,
        grid=grid_for_effective_cycles,
    )
    effective_fluid_substep_dt_s = float(spec.dt_s) / float(effective_fluid_substeps)
    fluid_grid_resolution = fluid_grid_resolution_report(spec)
    pressure_projection_budget = pressure_projection_budget_report(
        fluid_substeps=effective_fluid_substeps,
        ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
        fsi_coupling_iterations=fsi_coupling_iterations,
        projection_iterations=int(args.projection_iterations),
        fsi_coupling_enabled=legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=fsi_coupling_iterations,
        ),
    )
    if args.preflight_only:
        grid = cartesian_grid_for_spec(spec)
        uniform_spacing_m = cartesian_grid_uniform_spacing_m(grid)
        summary_path = output_dir / "preflight_summary.json"
        summary = {
            "case": "Squid soft robot",
            "preflight_only": True,
            "uses_generic_simulation_core": True,
            "summary_json": str(summary_path),
            "source_config_used_as_input_only": str(source_config_path),
            "pressure_solver_requested": str(args.pressure_solver),
            "pressure_solver": pressure_solver_name,
            "pressure_solve_failure_policy": str(args.pressure_solve_failure_policy),
            "fluid_advection_scheme": str(args.fluid_advection_scheme),
            "cg_preconditioner": cg_preconditioner,
            "multigrid_cycles": multigrid_cycles,
            "effective_multigrid_cycles": effective_multigrid_cycles,
            "divergence_cleanup_iterations": projection_divergence_cleanup_iterations,
            "fsi_coupling_mode": fsi_coupling_mode,
            "fsi_coupling_mode_report": fsi_coupling_mode_report,
            "steps": step_count,
            "full_pressure_waveform_steps": full_pressure_waveform_steps,
            "steps_explicit": bool(getattr(args, "steps_explicit", True)),
            "membrane_thickness_scale": membrane_thickness_scale,
            "solid_density_scale": solid_density_scale,
            "solid_density_kgm3": float(material.density_kgm3),
            "solid_surface_mass_budget": solid_surface_mass_report,
            "fluid_substeps": effective_fluid_substeps,
            "fluid_substep_dt_s": effective_fluid_substep_dt_s,
            "pressure_projection_budget": pressure_projection_budget,
            "interface_reaction_passivity_limit": interface_reaction_passivity_limit,
            "interface_reaction_robin_impedance_ns_m": (
                interface_reaction_robin_impedance_ns_m
            ),
            "interface_reaction_robin_matrix_impedance_ns_m": (
                interface_reaction_robin_matrix_impedance_ns_m
            ),
            "interface_reaction_robin_target_mode": (
                interface_reaction_robin_target_mode
            ),
            "fsi_coupling_target_map_relaxation": (
                fsi_coupling_target_map_relaxation
            ),
            "interface_reaction_aitken": interface_reaction_aitken,
            "interface_reaction_relaxation": interface_reaction_relaxation,
            "fluid_grid_spacing_m": (
                None if uniform_spacing_m is None else [float(value) for value in uniform_spacing_m]
            ),
            "fluid_grid_min_spacing_m": [
                float(value) for value in cartesian_grid_axis_min_spacing_m(grid)
            ],
            "fluid_grid_max_spacing_m": [
                float(value) for value in cartesian_grid_axis_max_spacing_m(grid)
            ],
            "fluid_grid_nodes": spec.grid_nodes,
            "fluid_grid_graded_enabled": graded_grid_enabled,
            "fluid_grid_refinement_region_count": (
                0 if spec.graded_grid is None else len(spec.graded_grid.refinement_regions)
            ),
            "fluid_grid_resolution": fluid_grid_resolution,
            "tail_refinement_enabled": tail_refinement_region is not None,
            "tail_refinement_geometry": tail_refinement_geometry,
            "tail_refinement_region": refinement_region_summary(tail_refinement_region),
            "region14_aperture_carve_enabled": bool(args.use_region14_aperture_carve),
            "open_downstream_farfield_enabled": bool(spec.downstream_farfield_open_enabled),
            "region14_aperture_geometry": region14_aperture_geometry,
            "reduced_water_geometry": reduced_water_geometry_report(spec),
            "spec": asdict(spec),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        process_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "status": "preflight_complete",
                    "finished_at_unix": time.time(),
                    "summary_json": str(summary_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary
    runtime = TaichiRuntimeConfig(arch=args.arch)
    simulator = ReducedSquidFSI(
        spec,
        runtime=runtime,
    )
    fluid_grid = simulator.fluid.grid
    fluid_grid_axis_min_spacing_m = cartesian_grid_axis_min_spacing_m(fluid_grid)
    fluid_grid_axis_max_spacing_m = cartesian_grid_axis_max_spacing_m(fluid_grid)
    fluid_grid_uniform_spacing_m = cartesian_grid_uniform_spacing_m(fluid_grid)
    fluid_probe_distance_m = min(fluid_grid_axis_min_spacing_m)
    if not args.disable_reduced_obstacles:
        simulator.mark_reduced_squid_water_domain()
    tri_diagnostics, tri_metadata, tri_surface_mesh, tri_surface_region_ids = (
        build_tri_surface_diagnostics(
            source_config,
            runtime,
            spec=spec,
            probe_distance_m=fluid_probe_distance_m,
        )
    )
    total_fsi_face_area_m2 = (
        float(tri_metadata["diagnostic_area_m2_by_region"].get("7", 0.0))
        + float(tri_metadata["diagnostic_area_m2_by_region"].get("8", 0.0))
    )
    primary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get("7", 0.0)
    )
    secondary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get("8", 0.0)
    )
    total_solid_volume_m3 = (
        float(tri_metadata["diagnostic_area_m2_by_region"].get("7", 0.0))
        * spec.main_membrane_thickness_m
        + float(tri_metadata["diagnostic_area_m2_by_region"].get("8", 0.0))
        * spec.tail_membrane_thickness_m
        + float(tri_metadata["solid_area_m2_by_region"].get("5", 0.0))
        * spec.main_membrane_thickness_m
    )
    estimated_solid_particle_count = max(
        1,
        int(tri_metadata["solid_surface_face_count"]) * max(1, int(args.solid_mpm_layers)),
    )
    estimated_solid_particle_spacing_m = (
        total_solid_volume_m3 / float(estimated_solid_particle_count)
    ) ** (1.0 / 3.0)
    solid_mpm_bounds_padding_m = 3.0 * max(
        float(fluid_probe_distance_m),
        float(estimated_solid_particle_spacing_m),
    )
    solid_mpm_bounds_min_m, solid_mpm_bounds_max_m = (
        solid_mpm_bounds_from_surface_metadata(
            tri_metadata,
            fallback_bounds_min_m=spec.fluid_bounds_min_m,
            fallback_bounds_max_m=spec.fluid_bounds_max_m,
            padding_m=solid_mpm_bounds_padding_m,
        )
    )
    stable_solid_dt_s = material.stable_explicit_dt_s(
        estimated_solid_particle_spacing_m,
        cfl=args.solid_mpm_cfl,
    )
    solid_mpm_substeps = int(args.solid_mpm_substeps)
    if solid_mpm_substeps <= 0:
        solid_mpm_substeps = max(1, math.ceil(spec.dt_s / max(stable_solid_dt_s, 1.0e-12)))
    solid_sub_dt_s = spec.dt_s / float(solid_mpm_substeps)
    solid_mpm_flip_blend = float(args.solid_mpm_flip_blend)
    if not 0.0 <= solid_mpm_flip_blend <= 1.0:
        raise ValueError("--solid-mpm-flip-blend must be in [0, 1]")
    solid_substep_velocity_damping = float(args.solid_mpm_velocity_damping) ** (
        solid_sub_dt_s / max(float(spec.dt_s), 1.0e-12)
    )
    if args.solid_model == "tri_mooney_shell_mpm":
        solid_mpm = TriMooneyShellMpmState(
            tri_surface_mesh,
            thickness_m=spec.main_membrane_thickness_m,
            density_kgm3=material.density_kgm3,
            c1_pa=0.5 * material.shear_modulus_pa,
            c2_pa=0.0,
            membrane_force_scale=args.mooney_membrane_force_scale,
            grid_nodes=solid_mpm_grid_nodes,
            bounds_padding_fraction=0.05,
            face_region_id=tri_surface_region_ids,
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
            runtime=runtime,
        )
    elif args.solid_model == "neo_hookean_mpm":
        solid_mpm = NeoHookeanMpmState(
            particle_capacity=tri_diagnostics.face_count * args.solid_mpm_layers,
            bounds_min_m=solid_mpm_bounds_min_m,
            bounds_max_m=solid_mpm_bounds_max_m,
            grid_nodes=solid_mpm_grid_nodes,
            runtime=runtime,
        )
        solid_mpm.initialize_layered_tri_surface(
            tri_diagnostics,
            layer_count=args.solid_mpm_layers,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
        )
    else:
        raise ValueError(f"Unsupported solid model: {args.solid_model}")

    sharp_coupling_state = (
        build_hibm_mpm_sharp_coupling_state(
            fluid=simulator.fluid,
            solid_mpm=solid_mpm,
            runtime=runtime,
        )
        if sharp_case_runner_enabled
        else None
    )

    def publish_solid_report_to_reduced_state(current_time_s: float, report) -> None:
        hydraulic_pressure_pa, volume_flux_m3s, nozzle_velocity_z_mps = hydraulic_diagnostics(
            spec,
            report.primary_mean_velocity_mps[2],
        )
        simulator.set_structure_state(
            time_s=current_time_s + spec.dt_s,
            pressure_pa=pressure_schedule_step_end_pa(current_time_s, spec.dt_s, spec),
            hydraulic_pressure_pa=hydraulic_pressure_pa,
            main_displacement_z_m=report.primary_mean_displacement_m[2],
            main_velocity_z_mps=report.primary_mean_velocity_mps[2],
            tail_displacement_z_m=report.secondary_mean_displacement_m[2],
            tail_velocity_z_mps=report.secondary_mean_velocity_mps[2],
            volume_flux_m3s=volume_flux_m3s,
            nozzle_velocity_z_mps=nozzle_velocity_z_mps,
        )

    def advance_physical_solid_step(
        current_time_s: float,
        primary_reaction_n: Sequence[float],
        secondary_reaction_n: Sequence[float],
    ):
        primary_reaction = _vector3(primary_reaction_n, name="primary_reaction_n")
        secondary_reaction = _vector3(secondary_reaction_n, name="secondary_reaction_n")
        simulator.set_interface_reaction(
            primary_force_n=primary_reaction,
            secondary_force_n=secondary_reaction,
        )
        if args.solid_model == "tri_mooney_shell_mpm":
            report = None
            for substep in range(solid_mpm_substeps):
                sub_time_s = current_time_s + float(substep) * solid_sub_dt_s
                pressure_pa = pressure_schedule_pa(sub_time_s, spec)
                report = solid_mpm.advance_region_loads(
                    dt_s=solid_sub_dt_s,
                    primary_region_id=7,
                    secondary_region_id=8,
                    primary_area_load_npm2=(0.0, 0.0, -pressure_pa),
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                    velocity_damping=solid_substep_velocity_damping,
                    flip_blend=solid_mpm_flip_blend,
                    read_report=False,
                )
            report = solid_mpm.report()
        elif args.solid_model == "neo_hookean_mpm":
            report = None
            for substep in range(solid_mpm_substeps):
                sub_time_s = current_time_s + float(substep) * solid_sub_dt_s
                pressure_pa = pressure_schedule_pa(sub_time_s, spec)
                solid_mpm.set_layered_region_loads(
                    primary_region_id=7,
                    secondary_region_id=8,
                    primary_area_load_npm2=(0.0, 0.0, -pressure_pa),
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                )
                report = solid_mpm.step(
                    dt_s=solid_sub_dt_s,
                    mu_pa=material.shear_modulus_pa,
                    lambda_pa=material.lame_lambda_pa,
                    velocity_damping=solid_substep_velocity_damping,
                    primary_region_id=7,
                    secondary_region_id=8,
                    read_report=False,
                )
            report = solid_mpm.report()
        else:
            raise ValueError(f"Unsupported solid model: {args.solid_model}")

        publish_solid_report_to_reduced_state(current_time_s, report)
        return report

    def z_displacement_vector(displacement_z_m: float) -> tuple[float, float, float]:
        return (0.0, 0.0, float(displacement_z_m))

    def z_velocity_vector(velocity_z_mps: float) -> tuple[float, float, float]:
        return (0.0, 0.0, float(velocity_z_mps))

    def advance_fluid_step(
        *,
        primary_velocity_mps: tuple[float, float, float] | None = None,
        secondary_velocity_mps: tuple[float, float, float] | None = None,
        primary_constraint_force_solid_mobility_ratio: float | None = None,
        secondary_constraint_force_solid_mobility_ratio: float | None = None,
        primary_velocity_target_solid_mobility_ratio: float | None = None,
        secondary_velocity_target_solid_mobility_ratio: float | None = None,
        primary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        secondary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_full_report: bool = True,
    ):
        primary_velocity = (
            z_velocity_vector(float(simulator.main_v_mps[None]))
            if primary_velocity_mps is None
            else _vector3(primary_velocity_mps, name="primary_velocity_mps")
        )
        secondary_velocity = (
            z_velocity_vector(float(simulator.tail_v_mps[None]))
            if secondary_velocity_mps is None
            else _vector3(secondary_velocity_mps, name="secondary_velocity_mps")
        )
        return advance_projected_ibm_region_pair_fluid_step(
            simulator.fluid,
            tri_diagnostics,
            ProjectedIbmRegionPairStepConfig(
                primary_region_id=7,
                secondary_region_id=8,
                primary_velocity_mps=primary_velocity,
                secondary_velocity_mps=secondary_velocity,
                dt_s=spec.dt_s,
                fluid_substeps=effective_fluid_substeps,
                ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
                projection_iterations=int(args.projection_iterations),
                divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                pressure_outlet_zmin=not args.disable_pressure_outlet_zmin,
                pressure_solver=pressure_solver_name,
                fluid_advection_scheme=str(args.fluid_advection_scheme),
                multigrid_cycles=effective_multigrid_cycles,
                cg_tolerance=cg_tolerance,
                cg_preconditioner=cg_preconditioner,
                velocity_constraint_blend=fsi_velocity_constraint_blend,
                velocity_constraint_solid_mobility_ratio=fsi_velocity_constraint_solid_mobility_ratio,
                constraint_force_scale=float(args.constraint_force_scale),
                constraint_force_solid_mobility_ratio=fsi_constraint_force_solid_mobility_ratio,
                primary_constraint_force_solid_mobility_ratio=(
                    primary_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=(
                    secondary_constraint_force_solid_mobility_ratio
                ),
                velocity_target_solid_mobility_ratio=(
                    fsi_velocity_target_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=(
                    primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=(
                    secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=primary_interface_impedance_force_n,
                secondary_interface_impedance_force_n=secondary_interface_impedance_force_n,
                primary_pressure_robin_impedance_ns_m=(
                    interface_reaction_robin_matrix_impedance_ns_m
                ),
                secondary_pressure_robin_impedance_ns_m=(
                    interface_reaction_robin_matrix_impedance_ns_m
                ),
                primary_interface_area_m2=primary_fsi_face_area_m2,
                secondary_interface_area_m2=secondary_fsi_face_area_m2,
                density_kgm3=spec.water_density_kgm3,
                viscosity_pa_s=spec.water_viscosity_pa_s,
                bounds_min_m=spec.fluid_bounds_min_m,
                bounds_max_m=spec.fluid_bounds_max_m,
                grid_nodes=spec.grid_nodes,
                read_full_report=read_full_report,
            ),
        )

    def diagnose_interface_reaction_target(row: dict[str, object], fluid_report):
        tri_report = tri_diagnostics.diagnose_from_fields(
            simulator.fluid.velocity,
            simulator.fluid.pressure,
            grid_fields=simulator.fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=z_velocity_vector(float(row["main_velocity_z_mps"])),
            secondary_velocity_mps=z_velocity_vector(float(row["tail_velocity_z_mps"])),
            probe_distance_m=fluid_probe_distance_m,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            spacing_m=fluid_grid_axis_min_spacing_m,
            grid_nodes=spec.grid_nodes,
            viscosity_pa_s=spec.water_viscosity_pa_s,
        )
        return tri_report

    history_path = output_dir / "history.csv"
    rows: list[dict[str, object]] = []
    partial_run_stopped = False
    partial_run_reason = ""
    interface_reaction_state = InterfaceReactionRelaxationState(
        relaxation=float(interface_reaction_relaxation),
    )
    first_step = 1
    if args.resume_from_checkpoint:
        completed_step, interface_reaction_state = load_run_checkpoint(
            run_checkpoint_path,
            args=args,
            simulator=simulator,
            solid_mpm=solid_mpm,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
        )
        if completed_step >= step_count:
            raise ValueError(
                f"checkpoint already completed {completed_step} steps, "
                f"which is not less than requested --steps={step_count}"
        )
        rows = read_csv_rows(history_path)
        rows = resume_history_rows_for_checkpoint(
            rows,
            completed_step=completed_step,
        )
        validate_resume_history_checkpoint_alignment(
            rows,
            completed_step=completed_step,
            checkpoint_time_s=float(simulator.time_s[None]),
            dt_s=spec.dt_s,
        )
        first_step = completed_step + 1

    for step in range(first_step, step_count + 1):
        step_wall_started_at = time.perf_counter()
        fsi_coupling_wall_time_s = 0.0
        solid_advance_wall_time_s = 0.0
        fluid_advance_wall_time_s = 0.0
        sample_wall_time_s = 0.0
        surface_diagnostics_wall_time_s = 0.0
        checkpoint_wall_time_s = 0.0
        solid_mpm_report = None
        velocity_constraint_report = None
        velocity_constraint_spread_report = None
        fsi_coupling_iterations_used = 1
        fsi_coupling_converged = fsi_coupling_iterations <= 1
        fsi_coupling_residual_norm_n = 0.0
        fsi_coupling_relaxation_effective = interface_reaction_relaxation
        fsi_coupling_iqn_ils_least_squares_update_count = 0
        fsi_coupling_interface_map_amplification = 0.0
        fsi_coupling_residual_jacobian_amplification = 0.0
        fsi_coupling_physical_interface_map_amplification = 0.0
        fsi_coupling_physical_residual_jacobian_amplification = 0.0
        fsi_coupling_raw_interface_map_amplification = 0.0
        fsi_coupling_raw_residual_jacobian_amplification = 0.0
        fsi_coupling_interface_map_amplification_sample_count = 0
        fsi_coupling_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_physical_interface_map_amplification_sample_count = 0
        fsi_coupling_physical_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_raw_interface_map_amplification_sample_count = 0
        fsi_coupling_raw_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_trial_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_residual_history_n: tuple[tuple[float, ...], ...] = ()
        accepted_fsi_trial_payload: dict[str, object] | None = None
        accepted_fsi_trial_state_reused = False
        fsi_trial_pressure_projection_cg_project_calls = 0
        fsi_trial_pressure_projection_cg_iterations_total = 0
        fsi_trial_pressure_projection_cg_iterations_max = 0
        fsi_trial_pressure_projection_cg_host_residual_checks = 0
        fsi_trial_pressure_projection_cg_mean_projection_count = 0
        fsi_trial_pressure_projection_cg_converged_all = True
        fsi_trial_pressure_projection_cg_max_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_max_initial_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_breakdown_count = 0
        fsi_primary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_secondary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_primary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_secondary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_coupling_enabled = legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=fsi_coupling_iterations,
        )
        step_start_main_velocity_z_mps = float(simulator.main_v_mps[None])
        step_start_tail_velocity_z_mps = float(simulator.tail_v_mps[None])
        step_start_interface_velocity_mps = _combine_region_pair_vectors(
            z_velocity_vector(step_start_main_velocity_z_mps),
            z_velocity_vector(step_start_tail_velocity_z_mps),
        )
        robin_previous_velocity_mps = robin_previous_velocity_for_step(
            interface_reaction_state,
            step_start_interface_velocity_mps,
        )
        step_start_main_displacement_z_m = float(simulator.main_w_m[None])
        step_start_tail_displacement_z_m = float(simulator.tail_w_m[None])

        def response_constraint_force_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            if not fsi_solid_response_mobility_coupling:
                return 0.0, 0.0
            correction_dt_s = (
                float(spec.dt_s)
                / float(effective_fluid_substeps)
                / float(max(1, int(args.ibm_correction_iterations)))
            )
            primary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=correction_dt_s,
            )
            secondary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=correction_dt_s,
            )
            return primary_ratio, secondary_ratio

        def velocity_target_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            base_ratio = fsi_velocity_target_solid_mobility_ratio
            if not fsi_solid_response_velocity_mobility_coupling:
                return base_ratio, base_ratio
            correction_dt_s = (
                float(spec.dt_s)
                / float(effective_fluid_substeps)
                / float(max(1, int(args.ibm_correction_iterations)))
            )
            primary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=correction_dt_s,
            )
            secondary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=correction_dt_s,
            )
            return primary_ratio, secondary_ratio

        if fsi_coupling_enabled:
            fsi_coupling_wall_started_at = time.perf_counter()
            current_step_time_s = float(simulator.time_s[None])

            def save_fsi_step_state() -> None:
                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()

            def restore_fsi_trial_state() -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()

            def accumulate_fsi_trial_pressure_projection_stats(fluid_report) -> None:
                nonlocal fsi_trial_pressure_projection_cg_project_calls
                nonlocal fsi_trial_pressure_projection_cg_iterations_total
                nonlocal fsi_trial_pressure_projection_cg_iterations_max
                nonlocal fsi_trial_pressure_projection_cg_host_residual_checks
                nonlocal fsi_trial_pressure_projection_cg_mean_projection_count
                nonlocal fsi_trial_pressure_projection_cg_converged_all
                nonlocal fsi_trial_pressure_projection_cg_max_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_max_initial_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_breakdown_count

                project_calls = int(
                    getattr(fluid_report, "pressure_projection_cg_project_calls", 0) or 0
                )
                if project_calls <= 0:
                    return
                fsi_trial_pressure_projection_cg_project_calls += project_calls
                fsi_trial_pressure_projection_cg_iterations_total += int(
                    getattr(fluid_report, "pressure_projection_cg_iterations_total", 0) or 0
                )
                fsi_trial_pressure_projection_cg_iterations_max = max(
                    fsi_trial_pressure_projection_cg_iterations_max,
                    int(getattr(fluid_report, "pressure_projection_cg_iterations_max", 0) or 0),
                )
                fsi_trial_pressure_projection_cg_host_residual_checks += int(
                    getattr(fluid_report, "pressure_projection_cg_host_residual_checks", 0)
                    or 0
                )
                fsi_trial_pressure_projection_cg_mean_projection_count += int(
                    getattr(
                        fluid_report,
                        "pressure_projection_cg_mean_projection_count",
                        0,
                    )
                    or 0
                )
                fsi_trial_pressure_projection_cg_converged_all = (
                    fsi_trial_pressure_projection_cg_converged_all
                    and bool(
                        getattr(fluid_report, "pressure_projection_cg_converged_all", True)
                    )
                )
                fsi_trial_pressure_projection_cg_max_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_max_initial_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_initial_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_initial_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_breakdown_count += int(
                    getattr(fluid_report, "pressure_projection_cg_breakdown_count", 0) or 0
                )

            def evaluate_fsi_interface_reaction_target(reaction_force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
                nonlocal fsi_primary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_secondary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_primary_velocity_target_solid_mobility_ratio
                nonlocal fsi_secondary_velocity_target_solid_mobility_ratio
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(reaction_force_n)
                trial_solid_report = advance_physical_solid_step(
                    current_step_time_s,
                    primary_reaction_n,
                    secondary_reaction_n,
                )
                trial_primary_velocity_mps = z_velocity_vector(
                    0.5 * (step_start_main_velocity_z_mps + float(simulator.main_v_mps[None]))
                )
                trial_secondary_velocity_mps = z_velocity_vector(
                    0.5 * (step_start_tail_velocity_z_mps + float(simulator.tail_v_mps[None]))
                )
                trial_solid_interface_velocity_mps = _combine_region_pair_vectors(
                    trial_solid_report.primary_mean_velocity_mps,
                    trial_solid_report.secondary_mean_velocity_mps,
                )
                trial_robin_impedance_force_n = robin_neumann_impedance_force(
                    velocity_mps=trial_solid_interface_velocity_mps,
                    previous_velocity_mps=robin_previous_velocity_mps,
                    impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
                )
                (
                    trial_primary_robin_impedance_force_n,
                    trial_secondary_robin_impedance_force_n,
                ) = _split_region_pair_vector(trial_robin_impedance_force_n)
                (
                    primary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                ) = response_constraint_force_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_primary_response_constraint_force_solid_mobility_ratio,
                    primary_response_constraint_force_solid_mobility_ratio,
                )
                fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_secondary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                )
                (
                    primary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                ) = velocity_target_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_velocity_target_solid_mobility_ratio = max(
                    fsi_primary_velocity_target_solid_mobility_ratio,
                    primary_velocity_target_solid_mobility_ratio,
                )
                fsi_secondary_velocity_target_solid_mobility_ratio = max(
                    fsi_secondary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                )
                tri_diagnostics.update_region_offsets(
                    primary_region_id=7,
                    secondary_region_id=8,
                    primary_offset_m=z_displacement_vector(
                        0.5 * (step_start_main_displacement_z_m + float(simulator.main_w_m[None]))
                    ),
                    secondary_offset_m=z_displacement_vector(
                        0.5 * (step_start_tail_displacement_z_m + float(simulator.tail_w_m[None]))
                    ),
                )
                trial_fluid_report = advance_fluid_step(
                    primary_velocity_mps=trial_primary_velocity_mps,
                    secondary_velocity_mps=trial_secondary_velocity_mps,
                    primary_constraint_force_solid_mobility_ratio=(
                        primary_response_constraint_force_solid_mobility_ratio
                    ),
                    secondary_constraint_force_solid_mobility_ratio=(
                        secondary_response_constraint_force_solid_mobility_ratio
                    ),
                    primary_velocity_target_solid_mobility_ratio=(
                        primary_velocity_target_solid_mobility_ratio
                    ),
                    secondary_velocity_target_solid_mobility_ratio=(
                        secondary_velocity_target_solid_mobility_ratio
                    ),
                    primary_interface_impedance_force_n=trial_primary_robin_impedance_force_n,
                    secondary_interface_impedance_force_n=trial_secondary_robin_impedance_force_n,
                    read_full_report=reuse_accepted_fsi_trial_state,
                )
                accumulate_fsi_trial_pressure_projection_stats(trial_fluid_report)
                primary_target_n = trial_fluid_report.interface_reaction_target.primary_force_n
                secondary_target_n = trial_fluid_report.interface_reaction_target.secondary_force_n
                stabilized_target_force_n = _combine_region_pair_vectors(
                    primary_target_n,
                    secondary_target_n,
                )
                raw_target_force_n = _combine_region_pair_vectors(
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            primary_target_n,
                            trial_primary_robin_impedance_force_n,
                        )
                    ),
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            secondary_target_n,
                            trial_secondary_robin_impedance_force_n,
                        )
                    )
                )
                selected_target_force_n = interface_reaction_target_for_mode(
                    interface_reaction_robin_target_mode,
                    raw_target_force_n=raw_target_force_n,
                    stabilized_target_force_n=stabilized_target_force_n,
                )
                return InterfaceReactionTargetEvaluation(
                    target_force_n=selected_target_force_n,
                    diagnostic_target_force_n=raw_target_force_n,
                    velocity_mps=trial_solid_interface_velocity_mps,
                    payload={
                        "solid_report": trial_solid_report,
                        "fluid_report": trial_fluid_report,
                        "raw_target_force_n": raw_target_force_n,
                        "selected_target_force_n": selected_target_force_n,
                        "robin_impedance_force_n": trial_robin_impedance_force_n,
                        "primary_response_constraint_force_solid_mobility_ratio": (
                            primary_response_constraint_force_solid_mobility_ratio
                        ),
                        "secondary_response_constraint_force_solid_mobility_ratio": (
                            secondary_response_constraint_force_solid_mobility_ratio
                        ),
                        "primary_velocity_target_solid_mobility_ratio": (
                            primary_velocity_target_solid_mobility_ratio
                        ),
                        "secondary_velocity_target_solid_mobility_ratio": (
                            secondary_velocity_target_solid_mobility_ratio
                        ),
                    },
                )

            def apply_accepted_fsi_interface_reaction(reaction_force_n: tuple[float, ...]) -> None:
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(reaction_force_n)
                simulator.set_interface_reaction(
                    primary_force_n=primary_reaction_n,
                    secondary_force_n=secondary_reaction_n,
                )

            def commit_accepted_fsi_trial_state(payload: object | None) -> None:
                nonlocal accepted_fsi_trial_payload
                accepted_fsi_trial_payload = payload if isinstance(payload, dict) else None

            fixed_point_result = solve_and_apply_interface_reaction_step(
                initial_force_n=_combine_region_pair_vectors(
                    simulator.primary_interface_reaction_force_n[None],
                    simulator.secondary_interface_reaction_force_n[None],
                ),
                save_state=save_fsi_step_state,
                evaluate_target=evaluate_fsi_interface_reaction_target,
                restore_state=restore_fsi_trial_state,
                apply_force=apply_accepted_fsi_interface_reaction,
                commit_accepted_state=(
                    commit_accepted_fsi_trial_state
                    if reuse_accepted_fsi_trial_state
                    else None
                ),
                max_iterations=fsi_coupling_iterations,
                tolerance_n=fsi_coupling_tolerance_n,
                initial_relaxation=interface_reaction_relaxation,
                use_aitken=interface_reaction_aitken,
                passivity_limit=interface_reaction_passivity_limit,
                solver=fsi_coupling_solver,
                target_map_relaxation=fsi_coupling_target_map_relaxation,
            )
            fsi_coupling_iterations_used = fixed_point_result.iterations_used
            fsi_coupling_converged = fixed_point_result.converged
            fsi_coupling_residual_norm_n = fixed_point_result.residual_norm_n
            fsi_coupling_relaxation_effective = fixed_point_result.relaxation
            fsi_coupling_iqn_ils_least_squares_update_count = (
                fixed_point_result.iqn_ils_least_squares_update_count
            )
            fsi_coupling_interface_map_amplification = (
                fixed_point_result.interface_map_amplification_max
            )
            fsi_coupling_residual_jacobian_amplification = (
                fixed_point_result.residual_jacobian_amplification_max
            )
            fsi_coupling_physical_interface_map_amplification = (
                fixed_point_result.physical_interface_map_amplification_max
            )
            fsi_coupling_physical_residual_jacobian_amplification = (
                fixed_point_result.physical_residual_jacobian_amplification_max
            )
            fsi_coupling_raw_interface_map_amplification = (
                fixed_point_result.diagnostic_interface_map_amplification_max
            )
            fsi_coupling_raw_residual_jacobian_amplification = (
                fixed_point_result.diagnostic_residual_jacobian_amplification_max
            )
            fsi_coupling_interface_map_amplification_sample_count = (
                fixed_point_result.interface_map_amplification_sample_count
            )
            fsi_coupling_residual_jacobian_amplification_sample_count = (
                fixed_point_result.residual_jacobian_amplification_sample_count
            )
            fsi_coupling_physical_interface_map_amplification_sample_count = (
                fixed_point_result.physical_interface_map_amplification_sample_count
            )
            fsi_coupling_physical_residual_jacobian_amplification_sample_count = (
                fixed_point_result.physical_residual_jacobian_amplification_sample_count
            )
            fsi_coupling_raw_interface_map_amplification_sample_count = (
                fixed_point_result.diagnostic_interface_map_amplification_sample_count
            )
            fsi_coupling_raw_residual_jacobian_amplification_sample_count = (
                fixed_point_result.diagnostic_residual_jacobian_amplification_sample_count
            )
            fsi_coupling_trial_force_history_n = fixed_point_result.trial_force_history_n
            fsi_coupling_target_force_history_n = fixed_point_result.target_force_history_n
            fsi_coupling_residual_history_n = fixed_point_result.residual_history_n
            fsi_coupling_physical_target_force_history_n = (
                fixed_point_result.physical_target_force_history_n
            )
            fsi_coupling_physical_residual_history_n = (
                fixed_point_result.physical_residual_history_n
            )
            fsi_coupling_raw_target_force_history_n = (
                fixed_point_result.diagnostic_target_force_history_n
            )
            fsi_coupling_raw_residual_history_n = (
                fixed_point_result.diagnostic_residual_history_n
            )
            fsi_coupling_wall_time_s = time.perf_counter() - fsi_coupling_wall_started_at

        if sharp_case_runner_enabled:
            if sharp_coupling_state is None:
                raise RuntimeError("sharp HIBM-MPM coupling state was not initialized")
            current_time_s = float(simulator.time_s[None])
            pressure_pa = pressure_schedule_step_end_pa(
                current_time_s,
                spec.dt_s,
                spec,
            )
            solid_mpm_report = None

            def advance_sharp_solid_substeps():
                nonlocal solid_advance_wall_time_s
                solid_wall_started_at = time.perf_counter()
                # The waveform drive enters through the far-pressure closure in
                # the marker traction sampling (region 7 below), so no direct
                # solid area load is added here: the membrane feels
                # (p_water - p_air) through scattered marker forces, which
                # restores the added-mass back-pressure that a direct area
                # load bypassed.
                report = None
                for _ in range(solid_mpm_substeps):
                    if args.solid_model == "tri_mooney_shell_mpm":
                        report = solid_mpm.advance_with_external_forces(
                            dt_s=solid_sub_dt_s,
                            primary_region_id=7,
                            secondary_region_id=8,
                            velocity_damping=solid_substep_velocity_damping,
                            flip_blend=solid_mpm_flip_blend,
                            read_report=False,
                        )
                    elif args.solid_model == "neo_hookean_mpm":
                        report = solid_mpm.step(
                            dt_s=solid_sub_dt_s,
                            mu_pa=material.shear_modulus_pa,
                            lambda_pa=material.lame_lambda_pa,
                            velocity_damping=solid_substep_velocity_damping,
                            primary_region_id=7,
                            secondary_region_id=8,
                            read_report=False,
                        )
                    else:
                        raise ValueError(f"Unsupported solid model: {args.solid_model}")
                solid_advance_wall_time_s = (
                    time.perf_counter() - solid_wall_started_at
                )
                return solid_mpm.report() if report is None else report

            fluid_wall_started_at = time.perf_counter()
            try:
                sharp_report = sharp_coupling_state.advance_mpm_step(
                    fluid=simulator.fluid,
                    mpm_external_force_n=solid_mpm.external_force_n,
                    mpm_particle_position_m=solid_mpm.x,
                    mpm_particle_velocity_mps=solid_mpm.v,
                    mpm_particle_normal=solid_mpm.surface_normal,
                    mpm_particle_area_m2=solid_mpm.area_weight_m2,
                    mpm_particle_count=solid_mpm.particle_count,
                    solid_step=advance_sharp_solid_substeps,
                    search_radius_m=max(
                        2.0 * fluid_probe_distance_m,
                        estimated_solid_particle_spacing_m,
                    ),
                    interior_probe_distance_m=fluid_probe_distance_m,
                    mpm_support_radius_m=max(
                        2.0 * estimated_solid_particle_spacing_m,
                        fluid_probe_distance_m,
                    ),
                    primary_region_id=7,
                    secondary_region_id=8,
                    far_pressure_region_id=7,
                    far_pressure_pa=pressure_pa,
                    far_pressure_inside_probe_max_multiplier=6.0,
                    fluid_dt_s=spec.dt_s,
                    fluid_substeps=effective_fluid_substeps,
                    projection_iterations=int(args.projection_iterations),
                    run_fluid_predictor=True,
                    pressure_neumann_density_kgm3=spec.water_density_kgm3,
                    pressure_neumann_dt_s=spec.dt_s,
                    pressure_outlet_zmin=not args.disable_pressure_outlet_zmin,
                    pressure_solver=pressure_solver_name,
                    pressure_solve_failure_policy=str(args.pressure_solve_failure_policy),
                    fluid_advection_scheme=str(args.fluid_advection_scheme),
                    multigrid_cycles=effective_multigrid_cycles,
                    cg_tolerance=cg_tolerance,
                    cg_preconditioner=cg_preconditioner,
                    divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                    divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                    diagnostic_disable_pressure_neumann_matrix_rows=bool(
                        args.diagnostic_disable_pressure_neumann_matrix_rows
                    ),
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                )
                raise
            fluid_advance_wall_time_s = max(
                0.0,
                time.perf_counter() - fluid_wall_started_at - solid_advance_wall_time_s,
            )
            solid_mpm_report = sharp_report.mpm
            if solid_mpm_report is None:
                solid_mpm_report = solid_mpm.report()
            publish_solid_report_to_reduced_state(current_time_s, solid_mpm_report)
            sample_wall_started_at = time.perf_counter()
            fluid_substep_dt_s = effective_fluid_substep_dt_s
            sample_report = simulator.sample_after_projection(
                sharp_report.fluid_to_mpm_loads.fluid_projection,
                dt_s=fluid_substep_dt_s,
            )
            sample_wall_time_s = time.perf_counter() - sample_wall_started_at
            sharp_summary = hibm_mpm_sharp_step_summary(sharp_report)
            row = build_hibm_mpm_sharp_case_row(
                step=step,
                sample_report=sample_report,
                sharp_summary=sharp_summary,
                fluid_projection_report=sharp_report.fluid_to_mpm_loads.fluid_projection,
                fluid_dt_s=spec.dt_s,
                solid_mpm_report=solid_mpm_report,
                solid_model=args.solid_model,
                fsi_coupling_mode_report=fsi_coupling_mode_report,
                fsi_coupling_iterations_requested=fsi_coupling_iterations,
            )
            expected_flux_m3s = float(row["volume_flux_m3s"])
            lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
            outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
            downstream_negative_z_flux_m3s = float(row["downstream_flow_negative_z_m3s"])
            row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_downstream_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["outlet_flux_deficit_m3s"] = (
                expected_flux_m3s - outlet_negative_z_flux_m3s
            )
            row["downstream_flux_deficit_m3s"] = (
                expected_flux_m3s - downstream_negative_z_flux_m3s
            )
            row["accepted_fsi_trial_state_reused"] = False
            row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
            row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
            row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
            row["sample_wall_time_s"] = sample_wall_time_s
            row["surface_diagnostics_wall_time_s"] = 0.0
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            rows.append(row)
            if args.diagnostic_dump_zero_correctable_cells:
                zero_correctable_summary = _write_hibm_zero_correctable_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=not args.disable_pressure_outlet_zmin,
                )
                row["diagnostic_zero_correctable_interior_cell_count"] = int(
                    zero_correctable_summary["zero_correctable_interior_cell_count"]
                )
                row["diagnostic_zero_correctable_shell_band_candidate_count"] = int(
                    zero_correctable_summary["shell_band_candidate_cell_count"]
                )
            snapshot_interval = int(args.fluid_snapshot_interval)
            if snapshot_interval > 0 and (
                step % snapshot_interval == 0 or step == step_count
            ):
                _write_fluid_snapshot_npz(
                    snapshot_dir=output_dir / "snapshots",
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    marker_count=int(sharp_coupling_state.markers.marker_count),
                    time_s=float(row["time_s"]),
                    pressure_pa=float(row["pressure_load_pa"]),
                )
            try:
                _raise_for_step_numerical_guard(
                    row,
                    cfl_limit=0.5,
                    divergence_l2_limit=float(args.projection_divergence_tolerance),
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                )
                raise
            if args.checkpoint_every_step:
                write_csv(history_path, rows)
                checkpoint_wall_started_at = time.perf_counter()
                write_run_checkpoint(
                    run_checkpoint_path,
                    completed_step=step,
                    step_count=step_count,
                    full_pressure_waveform_steps=full_pressure_waveform_steps,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid_mpm,
                    interface_reaction_state=interface_reaction_state,
                )
                checkpoint_wall_time_s = time.perf_counter() - checkpoint_wall_started_at
                row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
                row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
                write_csv(history_path, rows)
            if args.progress and (
                step == 1 or step == step_count or step % args.progress_interval == 0
            ):
                print(
                    "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                    "main_z={main_displacement_z_m:.6e}m "
                    "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                    "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                    "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                    "interior_div_l2={interior_divergence_l2:.3e}".format(
                        **row
                    ),
                    flush=True,
                )
            if (
                max_wall_time_s > 0.0
                and step < step_count
                and time.perf_counter() - run_started_at_perf >= max_wall_time_s
            ):
                partial_run_stopped = True
                partial_run_reason = "max_wall_time_s"
                break
            continue

        reused_fluid_step_report = None
        if accepted_fsi_trial_payload is not None:
            solid_mpm_report = accepted_fsi_trial_payload.get("solid_report")
            reused_fluid_step_report = accepted_fsi_trial_payload.get("fluid_report")
            if solid_mpm_report is None or reused_fluid_step_report is None:
                raise RuntimeError("accepted FSI trial payload is missing reusable reports")
            accepted_fsi_trial_state_reused = True
        else:
            solid_wall_started_at = time.perf_counter()
            current_time_s = float(simulator.time_s[None])
            primary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.primary_interface_reaction_force_n[None]
            )
            secondary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.secondary_interface_reaction_force_n[None]
            )
            solid_mpm_report = advance_physical_solid_step(
                current_time_s,
                primary_interface_reaction_n,
                secondary_interface_reaction_n,
            )
            solid_advance_wall_time_s = time.perf_counter() - solid_wall_started_at
        tri_diagnostics.update_region_offsets(
            primary_region_id=7,
            secondary_region_id=8,
            primary_offset_m=z_displacement_vector(
                0.5 * (step_start_main_displacement_z_m + float(simulator.main_w_m[None]))
            ),
            secondary_offset_m=z_displacement_vector(
                0.5 * (step_start_tail_displacement_z_m + float(simulator.tail_w_m[None]))
            ),
        )
        if reused_fluid_step_report is None:
            fluid_wall_started_at = time.perf_counter()
            accepted_fluid_step_robin_impedance_force_n = robin_neumann_impedance_force(
                velocity_mps=_combine_region_pair_vectors(
                    solid_mpm_report.primary_mean_velocity_mps,
                    solid_mpm_report.secondary_mean_velocity_mps,
                ),
                previous_velocity_mps=robin_previous_velocity_mps,
                impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
            )
            (
                accepted_primary_fluid_step_robin_impedance_force_n,
                accepted_secondary_fluid_step_robin_impedance_force_n,
            ) = _split_region_pair_vector(accepted_fluid_step_robin_impedance_force_n)
            (
                accepted_primary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            ) = response_constraint_force_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                fsi_primary_response_constraint_force_solid_mobility_ratio,
                accepted_primary_response_constraint_force_solid_mobility_ratio,
            )
            fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                fsi_secondary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            )
            (
                accepted_primary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            ) = velocity_target_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_velocity_target_solid_mobility_ratio = max(
                fsi_primary_velocity_target_solid_mobility_ratio,
                accepted_primary_velocity_target_solid_mobility_ratio,
            )
            fsi_secondary_velocity_target_solid_mobility_ratio = max(
                fsi_secondary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            )
            fluid_step_report = advance_fluid_step(
                primary_velocity_mps=z_velocity_vector(
                    0.5 * (step_start_main_velocity_z_mps + float(simulator.main_v_mps[None]))
                ),
                secondary_velocity_mps=z_velocity_vector(
                    0.5 * (step_start_tail_velocity_z_mps + float(simulator.tail_v_mps[None]))
                ),
                primary_constraint_force_solid_mobility_ratio=(
                    accepted_primary_response_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=(
                    accepted_secondary_response_constraint_force_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=(
                    accepted_primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=(
                    accepted_secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=(
                    accepted_primary_fluid_step_robin_impedance_force_n
                ),
                secondary_interface_impedance_force_n=(
                    accepted_secondary_fluid_step_robin_impedance_force_n
                ),
            )
            fluid_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
        else:
            fluid_step_report = reused_fluid_step_report
        divergence = fluid_step_report.divergence
        pressure_outlet_report = fluid_step_report.pressure_outlet_report
        force_report = required_projected_ibm_force_report(fluid_step_report.force_report)
        impulse_report = required_fluid_impulse_report(fluid_step_report.impulse_report)
        velocity_constraint_report = fluid_step_report.velocity_constraint_report
        velocity_constraint_spread_report = fluid_step_report.velocity_constraint_spread_report
        ibm_correction_iterations = fluid_step_report.ibm_correction_iterations
        ibm_correction_dt_s = fluid_step_report.ibm_correction_dt_s
        fluid_substeps = fluid_step_report.fluid_substeps
        fluid_substep_dt_s = fluid_step_report.fluid_substep_dt_s
        sample_wall_started_at = time.perf_counter()
        sample_report = simulator.sample_after_projection(
            divergence,
            dt_s=fluid_substep_dt_s,
        )
        sample_wall_time_s = time.perf_counter() - sample_wall_started_at
        row = {
            "step": step,
            **sample_report,
        }
        primary_fluid_force_n = fluid_step_report.primary_equivalent_fluid_force_n
        secondary_fluid_force_n = fluid_step_report.secondary_equivalent_fluid_force_n
        primary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_impulse_n_s",
        )
        secondary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_impulse_n_s",
        )
        primary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_equivalent_fluid_force_n",
        )
        secondary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_equivalent_fluid_force_n",
        )
        primary_interface_reaction_n = fluid_step_report.interface_reaction_target.primary_force_n
        secondary_interface_reaction_n = fluid_step_report.interface_reaction_target.secondary_force_n
        row["ibm_correction_iterations"] = ibm_correction_iterations
        row["ibm_correction_dt_s"] = ibm_correction_dt_s
        row["fluid_substeps"] = fluid_substeps
        row["fluid_substep_dt_s"] = fluid_substep_dt_s
        row["pressure_outlet_source_volume_flux_m3s"] = pressure_outlet_report[
            "source_volume_flux_m3s"
        ]
        row["pressure_outlet_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_velocity_to_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_source_ratio"
        ]
        row["pressure_outlet_pressure_flux_m3s"] = pressure_outlet_report[
            "zmin_pressure_outlet_flux_m3s"
        ]
        row["pressure_outlet_pressure_to_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_source_ratio"
        ]
        row["pressure_outlet_projection_pre_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_pre_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_projection_post_pressure_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_post_pressure_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_projection_post_boundary_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_post_boundary_velocity_outlet_flux_m3s"
        ]
        if not sharp_case_runner_enabled:
            row["pressure_projection_cg_project_calls"] = (
                getattr(fluid_step_report, "pressure_projection_cg_project_calls", 0)
            )
            row["pressure_projection_cg_iterations_total"] = (
                getattr(fluid_step_report, "pressure_projection_cg_iterations_total", 0)
            )
            row["pressure_projection_cg_iterations_max"] = (
                getattr(fluid_step_report, "pressure_projection_cg_iterations_max", 0)
            )
            row["pressure_projection_cg_host_residual_checks"] = (
                getattr(fluid_step_report, "pressure_projection_cg_host_residual_checks", 0)
            )
            row["pressure_projection_cg_mean_projection_count"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_mean_projection_count",
                    0,
                )
            )
            row["pressure_projection_cg_restart_count"] = (
                getattr(fluid_step_report, "pressure_projection_cg_restart_count", 0)
            )
            row["pressure_projection_cg_restart_count_measured"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_restart_count_measured",
                    False,
                )
            )
            row["pressure_projection_cg_restart_policy"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_restart_policy",
                    "",
                )
            )
            row["pressure_projection_cg_converged_all"] = (
                getattr(fluid_step_report, "pressure_projection_cg_converged_all", True)
            )
            row["pressure_projection_cg_max_relative_residual"] = (
                getattr(fluid_step_report, "pressure_projection_cg_max_relative_residual", 0.0)
            )
            row["pressure_projection_cg_max_initial_relative_residual"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_max_initial_relative_residual",
                    0.0,
                )
            )
            row["pressure_projection_cg_breakdown_count"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_count", 0)
            )
            row["pressure_projection_cg_breakdown_code"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_code", 0)
            )
            row["pressure_projection_cg_breakdown_dAd"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_dAd", 0.0)
            )
            row["pressure_interface_matrix_diagonal_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_diagonal_integral",
                0.0,
            )
            row["pressure_interface_matrix_rhs_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_rhs_integral",
                0.0,
            )
            row["pressure_interface_matrix_max_abs_diagonal"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_max_abs_diagonal",
                0.0,
            )
            row["pressure_interface_matrix_active_cells"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_active_cells",
                0,
            )
        row["fsi_trial_pressure_projection_cg_project_calls"] = (
            fsi_trial_pressure_projection_cg_project_calls
        )
        row["fsi_trial_pressure_projection_cg_iterations_total"] = (
            fsi_trial_pressure_projection_cg_iterations_total
        )
        row["fsi_trial_pressure_projection_cg_iterations_max"] = (
            fsi_trial_pressure_projection_cg_iterations_max
        )
        row["fsi_trial_pressure_projection_cg_host_residual_checks"] = (
            fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["fsi_trial_pressure_projection_cg_mean_projection_count"] = (
            fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["fsi_trial_pressure_projection_cg_converged_all"] = (
            fsi_trial_pressure_projection_cg_converged_all
        )
        row["fsi_trial_pressure_projection_cg_max_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_max_initial_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_initial_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_breakdown_count"] = (
            fsi_trial_pressure_projection_cg_breakdown_count
        )
        accepted_pressure_projection_cg_project_calls_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_project_calls"])
        )
        accepted_pressure_projection_cg_iterations_total_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_iterations_total"])
        )
        accepted_pressure_projection_cg_host_residual_checks_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_host_residual_checks"])
        )
        accepted_pressure_projection_cg_mean_projection_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_mean_projection_count"])
        )
        accepted_pressure_projection_cg_breakdown_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_breakdown_count"])
        )
        row["total_pressure_projection_cg_project_calls"] = (
            accepted_pressure_projection_cg_project_calls_for_cost
            + fsi_trial_pressure_projection_cg_project_calls
        )
        row["total_pressure_projection_cg_iterations_total"] = (
            accepted_pressure_projection_cg_iterations_total_for_cost
            + fsi_trial_pressure_projection_cg_iterations_total
        )
        row["total_pressure_projection_cg_iterations_max"] = max(
            int(row["pressure_projection_cg_iterations_max"]),
            fsi_trial_pressure_projection_cg_iterations_max,
        )
        row["total_pressure_projection_cg_host_residual_checks"] = (
            accepted_pressure_projection_cg_host_residual_checks_for_cost
            + fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["total_pressure_projection_cg_mean_projection_count"] = (
            accepted_pressure_projection_cg_mean_projection_count_for_cost
            + fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["total_pressure_projection_cg_converged_all"] = (
            bool(row["pressure_projection_cg_converged_all"])
            and fsi_trial_pressure_projection_cg_converged_all
        )
        row["total_pressure_projection_cg_max_relative_residual"] = max(
            float(row["pressure_projection_cg_max_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_relative_residual,
        )
        row["total_pressure_projection_cg_max_initial_relative_residual"] = max(
            float(row["pressure_projection_cg_max_initial_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_initial_relative_residual,
        )
        row["total_pressure_projection_cg_breakdown_count"] = (
            accepted_pressure_projection_cg_breakdown_count_for_cost
            + fsi_trial_pressure_projection_cg_breakdown_count
        )
        row["fsi_coupling_iterations_requested"] = fsi_coupling_iterations
        row["fsi_coupling_mode"] = fsi_coupling_mode
        row["fsi_coupling_mode_paper_hibm_mpm"] = bool(
            fsi_coupling_mode_report["paper_hibm_mpm"]
        )
        row["main_tail_region_reaction_diagnostic_only"] = bool(
            fsi_coupling_mode_report["main_tail_region_reaction_diagnostic_only"]
        )
        row["fsi_coupling_solver"] = fsi_coupling_solver
        row["fsi_coupling_iterations_used"] = fsi_coupling_iterations_used
        row["fsi_coupling_enabled"] = fsi_coupling_enabled
        row["fsi_coupling_converged"] = fsi_coupling_converged
        row["fsi_coupling_residual_norm_n"] = fsi_coupling_residual_norm_n
        row["fsi_coupling_relaxation_effective"] = fsi_coupling_relaxation_effective
        row["fsi_coupling_target_map_relaxation"] = (
            fsi_coupling_target_map_relaxation
        )
        row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
            fsi_coupling_iqn_ils_least_squares_update_count
        )
        row["fsi_coupling_interface_map_amplification"] = (
            fsi_coupling_interface_map_amplification
        )
        row["fsi_coupling_residual_jacobian_amplification"] = (
            fsi_coupling_residual_jacobian_amplification
        )
        row["fsi_coupling_physical_interface_map_amplification"] = (
            fsi_coupling_physical_interface_map_amplification
        )
        row["fsi_coupling_physical_residual_jacobian_amplification"] = (
            fsi_coupling_physical_residual_jacobian_amplification
        )
        row["fsi_coupling_raw_interface_map_amplification"] = (
            fsi_coupling_raw_interface_map_amplification
        )
        row["fsi_coupling_raw_residual_jacobian_amplification"] = (
            fsi_coupling_raw_residual_jacobian_amplification
        )
        row["fsi_coupling_interface_map_amplification_sample_count"] = (
            fsi_coupling_interface_map_amplification_sample_count
        )
        row["fsi_coupling_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
            fsi_coupling_physical_interface_map_amplification_sample_count
        )
        row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_physical_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
            fsi_coupling_raw_interface_map_amplification_sample_count
        )
        row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_raw_residual_jacobian_amplification_sample_count
        )
        row["accepted_fsi_trial_state_reused"] = accepted_fsi_trial_state_reused
        row["fsi_coupling_trial_force_history_n"] = fsi_coupling_trial_force_history_n
        row["fsi_coupling_target_force_history_n"] = fsi_coupling_target_force_history_n
        row["fsi_coupling_residual_history_n"] = fsi_coupling_residual_history_n
        row["fsi_coupling_physical_target_force_history_n"] = (
            fsi_coupling_physical_target_force_history_n
        )
        row["fsi_coupling_physical_residual_history_n"] = (
            fsi_coupling_physical_residual_history_n
        )
        row["fsi_coupling_raw_target_force_history_n"] = (
            fsi_coupling_raw_target_force_history_n
        )
        row["fsi_coupling_raw_residual_history_n"] = (
            fsi_coupling_raw_residual_history_n
        )
        expected_flux_m3s = float(row["volume_flux_m3s"])
        lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
        outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
        downstream_negative_z_flux_m3s = float(row["downstream_flow_negative_z_m3s"])
        row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_downstream_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["outlet_flux_deficit_m3s"] = expected_flux_m3s - outlet_negative_z_flux_m3s
        row["downstream_flux_deficit_m3s"] = (
            expected_flux_m3s - downstream_negative_z_flux_m3s
        )
        row["fsi_velocity_constraint_blend"] = fsi_velocity_constraint_blend
        row["fsi_constraint_force_solid_mobility_ratio"] = (
            fsi_constraint_force_solid_mobility_ratio
        )
        row["fsi_solid_response_mobility_coupling"] = (
            fsi_solid_response_mobility_coupling
        )
        row["fsi_velocity_target_solid_mobility_ratio"] = (
            fsi_velocity_target_solid_mobility_ratio
        )
        row["fsi_solid_response_velocity_mobility_coupling"] = (
            fsi_solid_response_velocity_mobility_coupling
        )
        row["fsi_primary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_primary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_secondary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_secondary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_primary_velocity_target_solid_mobility_ratio"] = (
            fsi_primary_velocity_target_solid_mobility_ratio
        )
        row["fsi_secondary_velocity_target_solid_mobility_ratio"] = (
            fsi_secondary_velocity_target_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_solid_mobility_ratio"] = (
            fsi_velocity_constraint_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_active_cells"] = (
            0 if velocity_constraint_report is None else velocity_constraint_report.active_cells
        )
        row["fsi_velocity_constraint_max_delta_mps"] = (
            0.0 if velocity_constraint_report is None else velocity_constraint_report.max_delta_mps
        )
        row["fsi_velocity_constraint_mean_delta_mps"] = (
            0.0 if velocity_constraint_report is None else velocity_constraint_report.mean_delta_mps
        )
        velocity_constraint_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_primary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "primary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_secondary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "secondary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        row["fsi_velocity_constraint_momentum_delta_x_n_s"] = (
            velocity_constraint_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_momentum_delta_y_n_s"] = (
            velocity_constraint_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_momentum_delta_z_n_s"] = (
            velocity_constraint_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        row["fsi_velocity_constraint_primary_momentum_delta_x_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_y_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_z_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_x_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_y_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_z_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_primary_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_primary_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        row["fsi_velocity_constraint_secondary_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_secondary_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        velocity_constraint_step_impulse_n_s = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_impulse_n_s,
                secondary_velocity_constraint_step_impulse_n_s,
            )
        )
        velocity_constraint_step_equivalent_fluid_force_n = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_equivalent_fluid_force_n,
                secondary_velocity_constraint_step_equivalent_fluid_force_n,
            )
        )
        row["fsi_velocity_constraint_step_impulse_x_n_s"] = (
            velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_step_impulse_y_n_s"] = (
            velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_step_impulse_z_n_s"] = (
            velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_primary_step_impulse_x_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_primary_step_impulse_y_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_primary_step_impulse_z_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_x_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_y_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_z_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_step_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_step_equivalent_fluid_force_n
        )
        row["fsi_velocity_constraint_primary_step_equivalent_force_norm_n"] = (
            vector_norm(primary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_secondary_step_equivalent_force_norm_n"] = (
            vector_norm(secondary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_sample_count"] = (
            0
            if velocity_constraint_spread_report is None
            else velocity_constraint_spread_report.projected_ibm_sample_count
        )
        surface_diagnostics_wall_started_at = time.perf_counter()
        tri_report = tri_diagnostics.diagnose_from_fields(
            simulator.fluid.velocity,
            simulator.fluid.pressure,
            grid_fields=simulator.fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=z_velocity_vector(float(row["main_velocity_z_mps"])),
            secondary_velocity_mps=z_velocity_vector(float(row["tail_velocity_z_mps"])),
            probe_distance_m=fluid_probe_distance_m,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            spacing_m=fluid_grid_axis_min_spacing_m,
            grid_nodes=spec.grid_nodes,
            viscosity_pa_s=spec.water_viscosity_pa_s,
        )
        surface_diagnostics_wall_time_s = (
            time.perf_counter() - surface_diagnostics_wall_started_at
        )
        row.update(
            {
                "pressure_traction_force_x_n": tri_report.pressure_traction_force_n[0],
                "pressure_traction_force_y_n": tri_report.pressure_traction_force_n[1],
                "pressure_traction_force_z_n": tri_report.pressure_traction_force_n[2],
                "main_pressure_traction_force_z_n": tri_report.primary_pressure_traction_force_n[2],
                "tail_pressure_traction_force_z_n": tri_report.secondary_pressure_traction_force_n[2],
                "viscous_traction_force_x_n": tri_report.viscous_traction_force_n[0],
                "viscous_traction_force_y_n": tri_report.viscous_traction_force_n[1],
                "viscous_traction_force_z_n": tri_report.viscous_traction_force_n[2],
                "main_viscous_traction_force_z_n": tri_report.primary_viscous_traction_force_n[2],
                "tail_viscous_traction_force_z_n": tri_report.secondary_viscous_traction_force_n[2],
                "fluid_stress_traction_force_x_n": tri_report.fluid_stress_traction_force_n[0],
                "fluid_stress_traction_force_y_n": tri_report.fluid_stress_traction_force_n[1],
                "fluid_stress_traction_force_z_n": tri_report.fluid_stress_traction_force_n[2],
                "main_fluid_stress_traction_force_z_n": tri_report.primary_fluid_stress_traction_force_n[2],
                "tail_fluid_stress_traction_force_z_n": tri_report.secondary_fluid_stress_traction_force_n[2],
                "pressure_traction_abs_force_n": tri_report.pressure_traction_abs_force_n,
                "pressure_traction_area_m2": tri_report.pressure_traction_area_m2,
                "pressure_traction_face_count": tri_report.pressure_traction_face_count,
                "projected_ibm_residual_mps": tri_report.projected_ibm_residual_mps,
                "projected_ibm_residual_l2_mps": tri_report.projected_ibm_residual_l2_mps,
                "projected_ibm_sample_count": tri_report.projected_ibm_sample_count,
                "fsi_probe_invalid_count": tri_report.invalid_probe_count,
                "fsi_probe_valid_fraction": tri_report.valid_probe_fraction,
                "fsi_probe_invalid_area_m2": tri_report.invalid_probe_area_m2,
                "fsi_probe_invalid_volume_source_m3s": tri_report.invalid_probe_volume_source_m3s,
                "fsi_force_probe_sample_count": force_report.force_sample_count,
                "fsi_force_probe_invalid_count": force_report.force_invalid_probe_count,
                "fsi_force_probe_valid_fraction": force_report.force_valid_probe_fraction,
                "fsi_force_probe_invalid_area_m2": force_report.invalid_probe_area_m2,
                "fsi_force_probe_invalid_volume_source_m3s": force_report.invalid_probe_volume_source_m3s,
                "fsi_grid_force_x_n": primary_fluid_force_n[0] + secondary_fluid_force_n[0],
                "fsi_grid_force_y_n": primary_fluid_force_n[1] + secondary_fluid_force_n[1],
                "fsi_grid_force_z_n": primary_fluid_force_n[2] + secondary_fluid_force_n[2],
                "fsi_last_correction_grid_force_x_n": force_report.grid_force_n[0],
                "fsi_last_correction_grid_force_y_n": force_report.grid_force_n[1],
                "fsi_last_correction_grid_force_z_n": force_report.grid_force_n[2],
                "main_fsi_fluid_force_x_n": primary_fluid_force_n[0],
                "main_fsi_fluid_force_y_n": primary_fluid_force_n[1],
                "main_fsi_fluid_force_z_n": primary_fluid_force_n[2],
                "tail_fsi_fluid_force_x_n": secondary_fluid_force_n[0],
                "tail_fsi_fluid_force_y_n": secondary_fluid_force_n[1],
                "tail_fsi_fluid_force_z_n": secondary_fluid_force_n[2],
                "main_fsi_fluid_reaction_x_n": primary_interface_reaction_n[0],
                "main_fsi_fluid_reaction_y_n": primary_interface_reaction_n[1],
                "main_fsi_fluid_reaction_z_n": primary_interface_reaction_n[2],
                "tail_fsi_fluid_reaction_x_n": secondary_interface_reaction_n[0],
                "tail_fsi_fluid_reaction_y_n": secondary_interface_reaction_n[1],
                "tail_fsi_fluid_reaction_z_n": secondary_interface_reaction_n[2],
                "fsi_constraint_force_x_n": force_report.constraint_force_n[0],
                "fsi_constraint_force_y_n": force_report.constraint_force_n[1],
                "fsi_constraint_force_z_n": force_report.constraint_force_n[2],
                "main_fsi_constraint_force_z_n": force_report.primary_constraint_force_n[2],
                "tail_fsi_constraint_force_z_n": force_report.secondary_constraint_force_n[2],
                "main_fsi_constraint_reaction_z_n": -force_report.primary_constraint_force_n[2],
                "tail_fsi_constraint_reaction_z_n": -force_report.secondary_constraint_force_n[2],
                "fsi_volume_source_m3s": force_report.volume_source_m3s,
                "main_fsi_volume_source_m3s": force_report.primary_volume_source_m3s,
                "tail_fsi_volume_source_m3s": force_report.secondary_volume_source_m3s,
                "fsi_active_force_cells": force_report.active_force_cells,
                "fluid_impulse_x_ns": impulse_report.grid_impulse_n_s[0],
                "fluid_impulse_y_ns": impulse_report.grid_impulse_n_s[1],
                "fluid_impulse_z_ns": impulse_report.grid_impulse_n_s[2],
                "fluid_impulse_relative_error": impulse_report.impulse_relative_error,
            }
        )
        force_decomposition = force_decomposition_report(
            grid_force_n=force_report.grid_force_n,
            component_forces_n=(
                force_report.primary_fluid_force_n,
                force_report.secondary_fluid_force_n,
            ),
        )
        row.update(
            {
                "fsi_last_correction_grid_decomposition_residual_x_n": force_decomposition[
                    "residual_components_n"
                ][0],
                "fsi_last_correction_grid_decomposition_residual_y_n": force_decomposition[
                    "residual_components_n"
                ][1],
                "fsi_last_correction_grid_decomposition_residual_z_n": force_decomposition[
                    "residual_components_n"
                ][2],
                "fsi_last_correction_grid_decomposition_residual_abs_n": force_decomposition[
                    "residual_norm_n"
                ],
                "fsi_last_correction_grid_decomposition_relative_error": force_decomposition[
                    "relative_error"
                ],
            }
        )
        fsi_interface_reaction_n = (
            primary_interface_reaction_n[0] + secondary_interface_reaction_n[0],
            primary_interface_reaction_n[1] + secondary_interface_reaction_n[1],
            primary_interface_reaction_n[2] + secondary_interface_reaction_n[2],
        )
        fsi_interface_balance = action_reaction_balance(
            (
                row["fsi_grid_force_x_n"],
                row["fsi_grid_force_y_n"],
                row["fsi_grid_force_z_n"],
            ),
            fsi_interface_reaction_n,
        )
        row.update(
            {
                "fsi_action_reaction_residual_x_n": fsi_interface_balance.residual_components_n[0],
                "fsi_action_reaction_residual_y_n": fsi_interface_balance.residual_components_n[1],
                "fsi_action_reaction_residual_z_n": fsi_interface_balance.residual_components_n[2],
                "fsi_action_reaction_residual_abs_n": fsi_interface_balance.residual_norm_n,
                "fsi_action_reaction_relative_error": fsi_interface_balance.relative_error,
            }
        )
        fluid_reaction_balance = action_reaction_balance(
            (
                row["main_fsi_fluid_force_x_n"],
                row["main_fsi_fluid_force_y_n"],
                row["main_fsi_fluid_force_z_n"],
                row["tail_fsi_fluid_force_x_n"],
                row["tail_fsi_fluid_force_y_n"],
                row["tail_fsi_fluid_force_z_n"],
            ),
            (
                row["main_fsi_fluid_reaction_x_n"],
                row["main_fsi_fluid_reaction_y_n"],
                row["main_fsi_fluid_reaction_z_n"],
                row["tail_fsi_fluid_reaction_x_n"],
                row["tail_fsi_fluid_reaction_y_n"],
                row["tail_fsi_fluid_reaction_z_n"],
            ),
        )
        row["fsi_fluid_reaction_action_reaction_residual_x_n"] = (
            fluid_reaction_balance.residual_components_n[0]
            + fluid_reaction_balance.residual_components_n[3]
        )
        row["fsi_fluid_reaction_action_reaction_residual_y_n"] = (
            fluid_reaction_balance.residual_components_n[1]
            + fluid_reaction_balance.residual_components_n[4]
        )
        row["fsi_fluid_reaction_action_reaction_residual_z_n"] = (
            fluid_reaction_balance.residual_components_n[2]
            + fluid_reaction_balance.residual_components_n[5]
        )
        row["fsi_fluid_reaction_action_reaction_residual_abs_n"] = (
            fluid_reaction_balance.residual_norm_n
        )
        row["fsi_fluid_reaction_action_reaction_relative_error"] = (
            fluid_reaction_balance.relative_error
        )
        main_full_reaction_balance = fluid_step_report.primary_interface_reaction_balance
        tail_full_reaction_balance = fluid_step_report.secondary_interface_reaction_balance
        row["main_fsi_fluid_reaction_full_residual_n"] = main_full_reaction_balance.residual_norm_n
        row["main_fsi_fluid_reaction_full_relative_error"] = main_full_reaction_balance.relative_error
        row["tail_fsi_fluid_reaction_full_residual_n"] = tail_full_reaction_balance.residual_norm_n
        row["tail_fsi_fluid_reaction_full_relative_error"] = tail_full_reaction_balance.relative_error
        if solid_mpm_report is None:
            solid_mpm_report = solid_mpm.report()
        solid_report_context = f"solid model {args.solid_model!r} report"
        solid_mpm_total_force_n = solid_force_vector_from_report(
            solid_mpm_report,
            solid_model=args.solid_model,
        )
        solid_mpm_row = {
            "solid_mpm_particle_count": solid_mpm_report.particle_count,
            "solid_mpm_active_grid_nodes": solid_mpm_report.active_grid_nodes,
            "solid_mpm_particle_spacing_m": solid_mpm_report.particle_spacing_m,
            "solid_mpm_grid_dx_m": solid_mpm_report.grid_spacing_m[0],
            "solid_mpm_grid_dy_m": solid_mpm_report.grid_spacing_m[1],
            "solid_mpm_grid_dz_m": solid_mpm_report.grid_spacing_m[2],
            "solid_mpm_total_mass_kg": solid_mpm_report.total_mass_kg,
            "solid_mpm_particle_momentum_x_kg_mps": solid_mpm_report.particle_momentum_kg_mps[0],
            "solid_mpm_particle_momentum_y_kg_mps": solid_mpm_report.particle_momentum_kg_mps[1],
            "solid_mpm_particle_momentum_z_kg_mps": solid_mpm_report.particle_momentum_kg_mps[2],
            "solid_mpm_grid_momentum_x_kg_mps": solid_mpm_report.grid_momentum_kg_mps[0],
            "solid_mpm_grid_momentum_y_kg_mps": solid_mpm_report.grid_momentum_kg_mps[1],
            "solid_mpm_grid_momentum_z_kg_mps": solid_mpm_report.grid_momentum_kg_mps[2],
            "solid_mpm_transfer_relative_error": solid_mpm_report.transfer_relative_error,
            "solid_mpm_max_speed_mps": _required_finite_report_number(
                solid_mpm_report,
                field="max_speed_mps",
                context=solid_report_context,
            ),
            "solid_mpm_total_force_x_n": solid_mpm_total_force_n[0],
            "solid_mpm_total_force_y_n": solid_mpm_total_force_n[1],
            "solid_mpm_total_force_z_n": solid_mpm_total_force_n[2],
        }
        if args.solid_model == "neo_hookean_mpm":
            solid_mpm_row["solid_mpm_max_abs_j"] = _required_finite_report_number(
                solid_mpm_report,
                field="max_abs_j",
                context=solid_report_context,
            )
        row.update(solid_mpm_row)
        previous_primary_reaction_n = _taichi_vector3_to_tuple(
            simulator.primary_interface_reaction_force_n[None]
        )
        previous_secondary_reaction_n = _taichi_vector3_to_tuple(
            simulator.secondary_interface_reaction_force_n[None]
        )
        stabilized_primary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.primary_force_n,
            name="stabilized_primary_reaction_target_n",
        )
        stabilized_secondary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.secondary_force_n,
            name="stabilized_secondary_reaction_target_n",
        )
        accepted_interface_velocity_mps = _combine_region_pair_vectors(
            solid_mpm_report.primary_mean_velocity_mps,
            solid_mpm_report.secondary_mean_velocity_mps,
        )
        accepted_robin_impedance_force_n = robin_neumann_impedance_force(
            velocity_mps=accepted_interface_velocity_mps,
            previous_velocity_mps=robin_previous_velocity_mps,
            impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
        )
        (
            accepted_primary_robin_impedance_force_n,
            accepted_secondary_robin_impedance_force_n,
        ) = _split_region_pair_vector(accepted_robin_impedance_force_n)
        raw_primary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_primary_reaction_target_n,
                accepted_primary_robin_impedance_force_n,
            )
        )
        raw_secondary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_secondary_reaction_target_n,
                accepted_secondary_robin_impedance_force_n,
            )
        )
        selected_reaction_target_n = interface_reaction_target_for_mode(
            interface_reaction_robin_target_mode,
            raw_target_force_n=_combine_region_pair_vectors(
                raw_primary_reaction_target_n,
                raw_secondary_reaction_target_n,
            ),
            stabilized_target_force_n=_combine_region_pair_vectors(
                stabilized_primary_reaction_target_n,
                stabilized_secondary_reaction_target_n,
            ),
        )
        (
            selected_primary_reaction_target_n,
            selected_secondary_reaction_target_n,
        ) = _split_region_pair_vector(selected_reaction_target_n)
        raw_main_reaction_target_z_n = raw_primary_reaction_target_n[2]
        raw_tail_reaction_target_z_n = raw_secondary_reaction_target_n[2]
        main_velocity_z_mps = float(row["main_velocity_z_mps"])
        tail_velocity_z_mps = float(row["tail_velocity_z_mps"])
        reaction_step_update = update_interface_reaction_for_next_step(
            previous_force_n=_combine_region_pair_vectors(
                previous_primary_reaction_n,
                previous_secondary_reaction_n,
            ),
            target_force_n=selected_reaction_target_n,
            velocity_mps=accepted_interface_velocity_mps,
            state=interface_reaction_state,
            initial_relaxation=interface_reaction_relaxation,
            use_aitken=interface_reaction_aitken,
            passivity_limit=interface_reaction_passivity_limit,
            robin_impedance_ns_per_m=0.0,
        )
        interface_reaction_state = reaction_step_update.next_state
        reaction_update = reaction_step_update.update
        interface_reaction_relaxation_used = reaction_step_update.relaxation
        relaxed_primary_reaction_n, relaxed_secondary_reaction_n = _split_region_pair_vector(
            reaction_update.force_n
        )
        relaxed_main_reaction_z_n = relaxed_primary_reaction_n[2]
        relaxed_tail_reaction_z_n = relaxed_secondary_reaction_n[2]
        row["interface_reaction_relaxation"] = interface_reaction_relaxation
        row["interface_reaction_aitken"] = interface_reaction_aitken
        row["interface_reaction_relaxation_effective"] = interface_reaction_relaxation_used
        row["interface_reaction_passivity_limit"] = interface_reaction_passivity_limit
        row["interface_reaction_robin_impedance_ns_m"] = (
            interface_reaction_robin_impedance_ns_m
        )
        row["interface_reaction_robin_matrix_impedance_ns_m"] = (
            interface_reaction_robin_matrix_impedance_ns_m
        )
        row["interface_reaction_robin_target_mode"] = interface_reaction_robin_target_mode
        row["raw_main_pressure_traction_z_n"] = tri_report.primary_pressure_traction_force_n[2]
        row["raw_tail_pressure_traction_z_n"] = tri_report.secondary_pressure_traction_force_n[2]
        row["raw_main_interface_reaction_z_n"] = raw_main_reaction_target_z_n
        row["raw_tail_interface_reaction_z_n"] = raw_tail_reaction_target_z_n
        row["main_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[2]
        )
        row["tail_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[5]
        )
        row["main_interface_reaction_stabilized_target_z_n"] = (
            stabilized_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_stabilized_target_z_n"] = (
            stabilized_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_selected_target_z_n"] = (
            selected_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_selected_target_z_n"] = (
            selected_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_residual_z_n"] = reaction_update.residual_n[2]
        row["tail_interface_reaction_residual_z_n"] = reaction_update.residual_n[5]
        row["relaxed_main_interface_reaction_z_n_next"] = relaxed_main_reaction_z_n
        row["relaxed_tail_interface_reaction_z_n_next"] = relaxed_tail_reaction_z_n
        row["raw_main_pressure_traction_power_w"] = (
            tri_report.primary_pressure_traction_force_n[2] * main_velocity_z_mps
        )
        row["raw_main_interface_reaction_power_w"] = raw_main_reaction_target_z_n * main_velocity_z_mps
        row["relaxed_main_interface_reaction_power_w_next"] = reaction_update.power_w[2]
        row["raw_tail_pressure_traction_power_w"] = (
            tri_report.secondary_pressure_traction_force_n[2] * tail_velocity_z_mps
        )
        row["raw_tail_interface_reaction_power_w"] = raw_tail_reaction_target_z_n * tail_velocity_z_mps
        row["relaxed_tail_interface_reaction_power_w_next"] = reaction_update.power_w[5]
        row["main_interface_reaction_passivity_limited"] = reaction_update.passivity_limited[2]
        row["tail_interface_reaction_passivity_limited"] = reaction_update.passivity_limited[5]
        simulator.set_interface_reaction(
            primary_force_n=relaxed_primary_reaction_n,
            secondary_force_n=relaxed_secondary_reaction_n,
        )
        row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
        row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
        row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
        row["sample_wall_time_s"] = sample_wall_time_s
        row["surface_diagnostics_wall_time_s"] = surface_diagnostics_wall_time_s
        row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
        row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
        rows.append(row)
        try:
            _raise_for_step_numerical_guard(
                row,
                cfl_limit=0.5,
                divergence_l2_limit=float(args.projection_divergence_tolerance),
            )
        except Exception as exc:
            _write_step_failure_artifacts(
                process_path=process_path,
                output_dir=output_dir,
                rows=rows,
                step=step,
                exc=exc,
                fluid=simulator.fluid,
            )
            raise
        if args.checkpoint_every_step:
            write_csv(history_path, rows)
            checkpoint_wall_started_at = time.perf_counter()
            write_run_checkpoint(
                run_checkpoint_path,
                completed_step=step,
                step_count=step_count,
                full_pressure_waveform_steps=full_pressure_waveform_steps,
                args=args,
                simulator=simulator,
                solid_mpm=solid_mpm,
                interface_reaction_state=interface_reaction_state,
            )
            checkpoint_wall_time_s = time.perf_counter() - checkpoint_wall_started_at
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            write_csv(history_path, rows)
        if args.progress and (step == 1 or step == step_count or step % args.progress_interval == 0):
            print(
                "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                "main_z={main_displacement_z_m:.6e}m "
                "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                "interior_div_l2={interior_divergence_l2:.3e}".format(
                    **row
                ),
                flush=True,
            )
        if (
            max_wall_time_s > 0.0
            and step < step_count
            and time.perf_counter() - run_started_at_perf >= max_wall_time_s
        ):
            partial_run_stopped = True
            partial_run_reason = "max_wall_time_s"
            break

    write_csv(history_path, rows)

    if sharp_case_runner_enabled:
        last = rows[-1] if rows else {}
        max_cfl = max(float(row["cfl"]) for row in rows) if rows else 0.0
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
        max_hibm_pressure_disconnected_nonprojectable_cell_count = (
            max(
                int(row["hibm_pressure_disconnected_nonprojectable_cell_count"])
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
                not bool(row.get("fsi_coupling_convergence_measured", False))
                and not bool(row.get("fsi_coupling_converged", False))
                for row in rows
            )
            if rows
            else False,
            "finite_primary_diagnostics": len(nonfinite_diagnostics) == 0,
            "negative_z_outlet_flow_present": max_outlet_negative_z > 0.0,
            "final_negative_z_outlet_flow": final_outlet_negative_z > 0.0,
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
                math.isfinite(max_interior_div_l2)
                and max_interior_div_l2 <= float(args.projection_divergence_tolerance)
            ),
            "pre_projection_divergence_measured": pre_projection_divergence_measured_all,
            "pressure_projection_cg_converged_all": pressure_projection_cg_converged_all,
            "pressure_projection_cg_no_breakdown": (
                pressure_projection_cg_breakdown_count == 0
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
        completed_step_checks_passed = checks_passed(checks)
        validation_scope_complete = bool(validation_scope["validation_scope_complete"])
        validation_passed = completed_step_checks_passed if validation_scope_complete else None
        summary = {
            "case": "Squid soft robot",
            "model_class": "sharp-interface HIBM-MPM case runner",
            "uses_generic_simulation_core": True,
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
            "total_pressure_projection_cg_converged_all": total_pressure_projection_cg_converged_all,
            "total_pressure_projection_cg_breakdown_count": total_pressure_projection_cg_breakdown_count,
            "max_hibm_no_slip_residual_l2_mps": max_no_slip_l2,
            "max_hibm_no_slip_residual_mps": max_no_slip_max,
            "max_hibm_ib_node_count": max_ib_node_count,
            "max_hibm_ib_invalid_projection_count": max_ib_invalid_count,
            "max_hibm_internal_obstacle_cell_count": (
                max_hibm_internal_obstacle_cell_count
            ),
            "max_hibm_solid_band_nonprojectable_cell_count": (
                max_hibm_solid_band_nonprojectable_cell_count
            ),
            "max_hibm_pressure_disconnected_nonprojectable_cell_count": (
                max_hibm_pressure_disconnected_nonprojectable_cell_count
            ),
            "max_hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
                max_pressure_neumann_skipped_velocity_dirichlet_count
            ),
            "max_hibm_full_stress_invalid_marker_count": (
                max_full_stress_invalid_count
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
            "final_negative_z_all_sections": final_all_sections_negative_z,
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
    final_pressure_outlet_velocity_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_velocity_flux_m3s",
    )
    final_pressure_outlet_velocity_to_source_ratio = _final_row_number(
        last,
        "pressure_outlet_velocity_to_source_ratio",
    )
    final_pressure_outlet_pressure_flux_m3s = _final_row_number(
        last,
        "pressure_outlet_pressure_flux_m3s",
    )
    final_pressure_outlet_pressure_to_source_ratio = _final_row_number(
        last,
        "pressure_outlet_pressure_to_source_ratio",
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
    max_fsi_coupling_residual_norm_n = (
        max(float(row["fsi_coupling_residual_norm_n"]) for row in rows) if rows else 0.0
    )
    fsi_coupling_not_converged_count = (
        sum(1 for row in rows if bool(row.get("fsi_coupling_enabled", False)) and not bool(row.get("fsi_coupling_converged", False)))
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
        prescribed_pressure_or_flow_boundary=False,
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
            source_volume_flux_m3s=final_pressure_outlet_source_volume_flux_m3s,
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
        "fsi_coupling_mode": fsi_coupling_mode,
        "fsi_coupling_mode_report": fsi_coupling_mode_report,
        "fsi_coupling_iterations_requested": fsi_coupling_iterations,
        "fsi_coupling_solver": fsi_coupling_solver,
        "max_fsi_coupling_iterations_used": max_fsi_coupling_iterations_used,
        "max_fsi_coupling_iqn_ils_least_squares_update_count": (
            max_fsi_coupling_iqn_ils_least_squares_update_count
        ),
        "fsi_coupling_tolerance_n": fsi_coupling_tolerance_n,
        "fsi_coupling_target_map_relaxation": fsi_coupling_target_map_relaxation,
        "max_fsi_coupling_residual_norm_n": max_fsi_coupling_residual_norm_n,
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
        "interface_reaction_aitken_note": (
            "When enabled, Aitken Delta^2 adapts both step-internal interface-reaction fixed-point "
            "updates and the accepted-step next interface-reaction residual; relaxation is clipped "
            "to [0.01, 1.5]."
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
        "pressure_outlet_zmin_enabled": not args.disable_pressure_outlet_zmin,
        "pressure_outlet_zmin_no_backflow_enabled": not args.disable_pressure_outlet_zmin,
        "boundary_drive_compliance": boundary_drive_compliance,
        "boundary_drive_compliance_gate": "diagnostic_only",
        "fluid_surface_traction_source": "sampled_projected_fluid_stress_field",
        "fluid_to_solid_interface_reaction_enabled": True,
        "tail_hydraulic_scalar_drive_enabled": False,
        "reduced_chamber_nozzle_obstacles_enabled": not args.disable_reduced_obstacles,
        "region14_aperture_carve_enabled": bool(args.use_region14_aperture_carve),
        "open_downstream_farfield_enabled": bool(spec.downstream_farfield_open_enabled),
        "region14_aperture_geometry": region14_aperture_geometry,
        "reduced_water_geometry": reduced_water_geometry_report(spec),
        "source_config_used_as_input_only": str(source_config_path),
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
        "final_pressure_outlet_velocity_flux_m3s": final_pressure_outlet_velocity_flux_m3s,
        "final_pressure_outlet_velocity_to_source_ratio": (
            final_pressure_outlet_velocity_to_source_ratio
        ),
        "final_pressure_outlet_pressure_flux_m3s": final_pressure_outlet_pressure_flux_m3s,
        "final_pressure_outlet_pressure_to_source_ratio": (
            final_pressure_outlet_pressure_to_source_ratio
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output_008step"))
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of physical time steps. Default runs through the full configured "
            "pressure waveform; pass an explicit small value for smoke tests."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Build the reduced case spec and grid diagnostics, write preflight_summary.json, "
            "and exit before Taichi/MPM/FSI initialization."
        ),
    )
    parser.add_argument("--projection-iterations", type=int, default=3000)
    parser.add_argument(
        "--pressure-solver",
        choices=PRESSURE_SOLVER_CHOICES,
        default="auto",
        help=(
            "Pressure projection solver. auto uses fv_multigrid on uniform FV grids "
            "and fv_cg on graded FV grids."
        ),
    )
    parser.add_argument(
        "--pressure-solve-failure-policy",
        choices=PRESSURE_SOLVE_FAILURE_POLICY_CHOICES,
        default="raise",
        help=(
            "Policy when the pressure solve reports nonconvergence: raise aborts "
            "the step; report returns the failure state in diagnostics."
        ),
    )
    parser.add_argument(
        "--fluid-advection-scheme",
        choices=FLUID_ADVECTION_SCHEME_CHOICES,
        default="euler",
        help=(
            "Fluid predictor semi-Lagrangian backtrace scheme. euler preserves the "
            "legacy single-backtrace predictor; rk2 uses a midpoint backtrace."
        ),
    )
    parser.add_argument(
        "--cg-tolerance",
        type=float,
        default=1.0e-6,
        help="Relative residual tolerance for --pressure-solver fv_cg.",
    )
    parser.add_argument(
        "--cg-preconditioner",
        choices=CG_PRECONDITIONER_CHOICES,
        default="auto",
        help=(
            "Preconditioner for --pressure-solver fv_cg. auto preserves the current "
            "behavior: multigrid on graded FV grids and Jacobi on uniform grids."
        ),
    )
    parser.add_argument(
        "--multigrid-cycles",
        type=int,
        default=None,
        help="Optional V-cycle count when --pressure-solver resolves to fv_multigrid.",
    )
    parser.add_argument(
        "--divergence-cleanup-iterations",
        type=int,
        default=8,
        help=(
            "Optional local post-projection divergence cleanup iterations. This enforces "
            "the fluid incompressibility constraint and does not prescribe nozzle velocity, "
            "pressure, or flow."
        ),
    )
    parser.add_argument(
        "--divergence-cleanup-relaxation",
        type=float,
        default=0.7,
        help="Relaxation for local post-projection divergence cleanup; must be in [0, 1].",
    )
    parser.add_argument(
        "--diagnostic-disable-pressure-neumann-matrix-rows",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: keep no-slip velocity "
            "Dirichlet rows and wall BCs but suppress pressure-Neumann "
            "interface matrix/RHS rows."
        ),
    )
    parser.add_argument(
        "--diagnostic-dump-zero-correctable-cells",
        action="store_true",
        help=(
            "Diagnostic-only HIBM-MPM sharp switch: dump interior fluid cells whose "
            "divergence stencil has no pressure-correctable faces."
        ),
    )
    parser.add_argument(
        "--projection-divergence-tolerance",
        type=float,
        default=1.0e-2,
        help="Validation gate for post-projection divergence L2.",
    )
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-graded-grid",
        action="store_true",
        help=(
            "Use a tensor-product graded Cartesian fluid grid with a nozzle refinement "
            "column. This changes only mesh resolution, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--graded-grid-target-spacing-m",
        type=float,
        default=None,
        help="Target cell spacing inside the nozzle refinement column. Default is nozzle_radius/5.",
    )
    parser.add_argument(
        "--graded-grid-farfield-spacing-m",
        type=float,
        default=3.0e-3,
        help="Far-field fluid cell spacing for --use-graded-grid.",
    )
    parser.add_argument(
        "--graded-grid-growth-ratio",
        type=float,
        default=1.2,
        help="Maximum adjacent-cell spacing ratio for --use-graded-grid; must be greater than 1.",
    )
    parser.add_argument(
        "--graded-grid-max-cells",
        type=int,
        default=5_000_000,
        help="Maximum generated fluid cells for --use-graded-grid. Use 0 to disable this guard.",
    )
    parser.add_argument(
        "--use-tail-refinement",
        action="store_true",
        help=(
            "Add an optional region 8 tail FSI bounding-box refinement region to the "
            "graded Cartesian fluid grid. This changes only mesh resolution, not "
            "velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--tail-refinement-target-spacing-m",
        type=float,
        default=None,
        help=(
            "Target cell spacing inside the optional region 8 tail refinement box. "
            "Default is min(tail membrane thickness, graded-grid far-field spacing)."
        ),
    )
    parser.add_argument(
        "--tail-refinement-padding-m",
        type=float,
        default=None,
        help=(
            "Padding around source-config region 8 vertex bounds for optional tail "
            "mesh refinement. Default is two tail target cells."
        ),
    )
    parser.add_argument(
        "--time-step-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the source configuration time step for time-refinement studies. "
            "Use more steps to keep the same physical duration when this is below 1."
        ),
    )
    parser.add_argument(
        "--solid-model",
        choices=("tri_mooney_shell_mpm", "neo_hookean_mpm"),
        default="tri_mooney_shell_mpm",
        help=(
            "Solid model. tri_mooney_shell_mpm is the paper-calibrated arbitrary-triangle "
            "shell MPM; neo_hookean_mpm is the volumetric layered branch."
        ),
    )
    parser.add_argument("--solid-mpm-layers", type=int, default=2)
    parser.add_argument(
        "--solid-mpm-substeps",
        type=int,
        default=0,
        help="Neo-Hookean MPM substeps per fluid step. Use 0 for Ecoflex CFL-based auto substepping.",
    )
    parser.add_argument(
        "--membrane-thickness-scale",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier for main/tail shell thickness. This changes the "
            "physical shell surface mass and membrane thickness; default 1 preserves "
            "the baseline Ecoflex geometry."
        ),
    )
    parser.add_argument(
        "--solid-density-scale",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier for the Ecoflex solid density. This isolates "
            "rho_s*h_s surface-mass scaling without changing the membrane modulus; "
            "default 1 preserves the baseline material card."
        ),
    )
    parser.add_argument("--solid-mpm-cfl", type=float, default=0.35)
    parser.add_argument("--solid-mpm-velocity-damping", type=float, default=1.0)
    parser.add_argument(
        "--solid-mpm-flip-blend",
        type=float,
        default=0.95,
        help="Tri-Mooney shell MPM G2P blend: 0 is PIC, 1 is FLIP.",
    )
    parser.add_argument("--mooney-membrane-force-scale", type=float, default=1.0)
    parser.add_argument("--poissons-ratio", type=float, default=0.49)
    parser.add_argument("--arch", default="cuda")
    parser.add_argument("--constraint-force-scale", type=float, default=1.0)
    parser.add_argument(
        "--fsi-constraint-force-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the projected-IBM "
            "constraint force. Zero preserves the explicit fluid-mass force; "
            "positive values scale the constraint force by 1/(1+ratio)."
        ),
    )
    parser.add_argument(
        "--fsi-solid-response-mobility-coupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the measured trial solid velocity response to compute per-region "
            "projected-IBM constraint-force mobility ratios. Disabled by default; "
            "when enabled, this changes the raw physical interface operator rather "
            "than only relaxing the fixed-point solver map."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-target-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the projected-IBM "
            "target boundary velocity. Zero preserves the explicit target "
            "velocity. Positive values use sampled fluid velocity plus "
            "(solid-target minus sampled-fluid)/(1+ratio), changing both the "
            "constraint residual and FSI volume source."
        ),
    )
    parser.add_argument(
        "--fsi-solid-response-velocity-mobility-coupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the measured trial solid velocity response to add per-region "
            "projected-IBM target-velocity mobility ratios. Disabled by default; "
            "when enabled, this changes the local interface boundary-velocity "
            "operator, not just the fixed-point solver map."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-constraint-blend",
        type=float,
        default=0.0,
        help=(
            "Blend factor for enforcing no-slip velocity on the FSI water-side "
            "probe cells before projection. Nonzero values are reported as a "
            "prescribed interface velocity constraint in validation."
        ),
    )
    parser.add_argument(
        "--fsi-velocity-constraint-solid-mobility-ratio",
        type=float,
        default=0.0,
        help=(
            "Dimensionless solid/fluid mobility ratio for the FSI velocity "
            "constraint. Zero preserves the hard overwrite operator; positive "
            "values apply blend/(1+ratio) to emulate a coupled mobility denominator."
        ),
    )
    parser.add_argument(
        "--interface-reaction-relaxation",
        type=float,
        default=0.5,
        help=(
            "Under-relaxation for the interface reaction fed back to the solid. "
            "This is a partitioned FSI coupling relaxation, not a nozzle boundary condition."
        ),
    )
    parser.add_argument(
        "--interface-reaction-aitken",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Aitken Delta^2 adaptation for both step-internal interface-reaction "
            "fixed-point updates and accepted-step interface-reaction residual updates. "
            "Enabled by default for added-mass stability; use --no-interface-reaction-aitken "
            "only for diagnostics."
        ),
    )
    parser.add_argument(
        "--interface-reaction-passivity-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional diagnostic limiter for committed interface reactions. Disabled by "
            "default because it projects positive-power committed reactions to the "
            "zero-power boundary and can otherwise change the projected-IBM reaction "
            "used by the solid. It never prescribes nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-impedance-ns-m",
        type=float,
        default=0.0,
        help=(
            "Explicit Phase-C Robin-Neumann interface impedance in N*s/m. The "
            "default 0 preserves the existing partitioned Aitken/IQN path. "
            "Positive values add -Z*(v_n-v_{n-1}) to the accepted step-to-step "
            "interface reaction target; this changes only the interface coupling "
            "law, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-matrix-impedance-ns-m",
        type=float,
        default=0.0,
        help=(
            "Opt-in Phase-1B interface impedance in N*s/m that is scattered as "
            "per-marker terms into the FV-CG pressure matrix. The default 0 "
            "preserves the existing explicit partitioned path; positive values "
            "require --pressure-solver fv_cg."
        ),
    )
    parser.add_argument(
        "--interface-reaction-robin-target-mode",
        choices=INTERFACE_REACTION_ROBIN_TARGET_CHOICES,
        default="stabilized",
        help=(
            "How the fluid-side Robin impedance contribution enters the committed "
            "solid interface reaction. stabilized preserves the current Phase-C "
            "path; physical subtracts the Robin term from the returned target so "
            "the impedance acts only as a fluid-side boundary stabilizer."
        ),
    )
    parser.add_argument(
        "--min-outlet-to-main-volume-flux-ratio",
        type=float,
        default=0.1,
        help=(
            "Validation gate for real sampled outlet flux relative to the kinematic "
            "main-membrane volume-flux estimate. Values far below this mean the "
            "reported jet is not present in the fluid field."
        ),
    )
    parser.add_argument(
        "--pressure-outlet-source-ratio-tolerance",
        type=float,
        default=0.1,
        help=(
            "Validation tolerance for the pressure-outlet boundary-face velocity "
            "flux ratio relative to the FSI volume source. The pressure-implied "
            "flux is reported as a finite diagnostic, not as this conservation gate."
        ),
    )
    parser.add_argument(
        "--fluid-substeps",
        type=int,
        default=1,
        help=(
            "Number of fluid predictor/IBM/projection substeps per physical solid step. "
            "This is a time-integration refinement for CFL stability, not a nozzle "
            "velocity, pressure, or flow boundary."
        ),
    )
    parser.add_argument(
        "--ibm-correction-iterations",
        type=int,
        default=2,
        help=(
            "Number of force-spread/body-force/projection correction passes per fluid step. "
            "This repeats the projected IBM no-slip correction; it does not prescribe nozzle "
            "velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-iterations",
        type=int,
        default=6,
        help=(
            "Solid-fluid fixed-point iterations per physical MPM time step. Values above "
            "1 use Taichi device-side solid/fluid snapshots to re-advance the current "
            "step with updated pressure/constraint interface reaction; this is not a nozzle "
            "velocity, pressure, or flow boundary."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-mode",
        choices=FSI_COUPLING_MODE_CHOICES,
        default=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
        help=(
            "Solver-level FSI coupling mode. legacy_projected_reduced keeps the "
            "existing projected-IBM plus reduced region-reaction path as legacy "
            "diagnostic coupling. hibm_mpm_sharp selects the generic "
            "sharp-interface HIBM-MPM solver path; the current squid case "
            "runner supports it with --solid-model neo_hookean_mpm while "
            "Phase 5 long-run validation remains incomplete."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-solver",
        choices=INTERFACE_REACTION_SOLVER_CHOICES,
        default="aitken",
        help=(
            "Step-internal interface-reaction fixed-point solver. aitken preserves the "
            "existing scalar-relaxed path; iqn_ils uses inverse least-squares secant "
            "updates of the same interface residual equation."
        ),
    )
    parser.add_argument(
        "--fsi-coupling-tolerance-n",
        type=float,
        default=1.0e-3,
        help="Convergence tolerance for the two-component step-internal interface-reaction residual in Newtons.",
    )
    parser.add_argument(
        "--fsi-coupling-target-map-relaxation",
        type=float,
        default=1.0,
        help=(
            "Semi-implicit fixed-point target-map relaxation beta in (0, 1]. "
            "beta=1 preserves the raw physical target map. beta<1 solves the "
            "same physical fixed point through F + beta*(T(F)-F) and reports "
            "both solver-map and raw physical-map amplification."
        ),
    )
    parser.add_argument(
        "--reuse-accepted-fsi-trial-state",
        action="store_true",
        help=(
            "Experimental performance path: when the fixed-point solver proves the "
            "last trial is the accepted FSI state, reuse that full-report trial state "
            "instead of re-advancing the accepted solid/fluid step. Disabled by default."
        ),
    )
    parser.add_argument("--disable-pressure-outlet-zmin", action="store_true")
    parser.add_argument("--disable-reduced-obstacles", action="store_true")
    parser.add_argument(
        "--use-region14-aperture-carve",
        action="store_true",
        help=(
            "Use source-config region 14 open-edge aperture geometry to set the reduced "
            "nozzle/outlet carve center and radius. This changes only the obstacle/opening "
            "geometry, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--open-downstream-farfield",
        action="store_true",
        help=(
            "With --use-region14-aperture-carve, keep the external domain below the "
            "region 14 aperture plane as active water instead of a narrow outlet plume. "
            "This is an obstacle/topology correction, not a flow boundary condition."
        ),
    )
    parser.add_argument(
        "--use-nozzle-taper",
        action="store_true",
        help=(
            "Use an analytic converging inlet taper upstream of the reduced nozzle throat. "
            "This changes only obstacle geometry, not nozzle velocity, pressure, or flow."
        ),
    )
    parser.add_argument(
        "--nozzle-taper-length-m",
        type=float,
        default=None,
        help=(
            "Length of the analytic nozzle taper. Default with --use-nozzle-taper is "
            "min(nozzle_length, chamber_z_min - downstream_z)."
        ),
    )
    parser.add_argument(
        "--nozzle-taper-inlet-radius-m",
        type=float,
        default=None,
        help="Inlet radius of the analytic taper. Default is the reduced chamber radius.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--max-wall-time-s",
        type=float,
        default=0.0,
        help=(
            "Stop gracefully after the current completed step once this wall-time "
            "budget is exceeded. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-step",
        action="store_true",
        help=(
            "Write a restart checkpoint after every completed physical step. This is "
            "intended for long validation trends that must be resumed across runs."
        ),
    )
    parser.add_argument(
        "--fluid-snapshot-interval",
        type=int,
        default=0,
        help=(
            "Write a compact visualization snapshot (fluid speed slices + marker "
            "positions, .npz) every N completed steps in the sharp runner. 0 disables."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help=(
            "Resume from the checkpoint path and append to the existing history.csv. "
            "The checkpoint and history must agree on the completed step count."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Path for --checkpoint-every-step/--resume-from-checkpoint. Defaults to "
            "run_checkpoint.npz inside --output-dir."
        ),
    )
    args = parser.parse_args(argv)
    raw_args = sys.argv[1:] if argv is None else list(argv)
    args.divergence_cleanup_iterations_explicit = any(
        token == "--divergence-cleanup-iterations"
        or token.startswith("--divergence-cleanup-iterations=")
        for token in raw_args
    )
    args.steps_explicit = any(token == "--steps" or token.startswith("--steps=") for token in raw_args)
    if args.graded_grid_max_cells is not None and args.graded_grid_max_cells < 0:
        parser.error("--graded-grid-max-cells must be non-negative")
    if args.graded_grid_max_cells == 0:
        args.graded_grid_max_cells = None
    if args.use_tail_refinement and not args.use_graded_grid:
        parser.error("--use-tail-refinement requires --use-graded-grid")
    if (
        args.tail_refinement_target_spacing_m is not None
        and args.tail_refinement_target_spacing_m <= 0.0
    ):
        parser.error("--tail-refinement-target-spacing-m must be positive")
    if (
        args.tail_refinement_padding_m is not None
        and args.tail_refinement_padding_m < 0.0
    ):
        parser.error("--tail-refinement-padding-m must be non-negative")
    return args


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    result = main()
    summary_json = result.get("summary_json")
    if summary_json is None:
        summary_json = str(Path(result["history_csv"]).with_name("summary.json"))
    print(json.dumps({"summary_json": str(summary_json)}, indent=2))
