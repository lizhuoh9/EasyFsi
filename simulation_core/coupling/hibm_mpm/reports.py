from dataclasses import dataclass
from typing import Any

from .constants import HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS

@dataclass(frozen=True)
class HibmMpmSurfaceMarkerForceReport:
    primary_marker_force_n: tuple[float, float, float]
    secondary_marker_force_n: tuple[float, float, float]
    total_marker_force_n: tuple[float, float, float]
    primary_marker_count: int
    secondary_marker_count: int
    total_marker_count: int
    fluid_reaction_force_n: tuple[float, float, float]
    action_reaction_residual_n: float
    primary_stress_valid_marker_count: int = 0
    primary_stress_invalid_marker_count: int = 0
    secondary_stress_valid_marker_count: int = 0
    secondary_stress_invalid_marker_count: int = 0
    primary_marker_force_norm_sum_n: float = 0.0
    secondary_marker_force_norm_sum_n: float = 0.0
    total_marker_force_norm_sum_n: float = 0.0
    primary_marker_force_norm_max_n: float = 0.0
    secondary_marker_force_norm_max_n: float = 0.0
    total_marker_force_norm_max_n: float = 0.0


@dataclass(frozen=True)
class HibmMpmFluidStressSampleReport:
    valid_marker_count: int
    invalid_marker_count: int
    max_abs_traction_pa: float
    two_sided_pressure_marker_count: int = 0
    viscous_gradient_invalid_marker_count: int = 0
    far_pressure_closed_marker_count: int = 0
    far_pressure_closed_extended_marker_count: int = 0
    far_pressure_anchor_closed_marker_count: int = 0
    far_pressure_node_anchor_closed_marker_count: int = 0
    closure_gradient_missing_marker_count: int = 0
    far_pressure_outside_suppressed_marker_count: int = 0
    two_sided_extended_marker_count: int = 0
    one_sided_pressure_marker_count: int = 0
    one_sided_extended_marker_count: int = 0
    one_sided_gradient_missing_marker_count: int = 0
    marker_diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class HibmMpmNoSlipResidualReport:
    valid_marker_count: int
    invalid_marker_count: int
    max_no_slip_residual_mps: float
    l2_no_slip_residual_mps: float
    direct_sample_marker_count: int = 0
    normal_walk_sample_marker_count: int = 0
    nearest_fluid_sample_marker_count: int = 0
    zero_normal_marker_count: int = 0
    no_fluid_sample_marker_count: int = 0
    primary_region_valid_marker_count: int = 0
    primary_region_invalid_marker_count: int = 0
    secondary_region_valid_marker_count: int = 0
    secondary_region_invalid_marker_count: int = 0
    other_region_valid_marker_count: int = 0
    other_region_invalid_marker_count: int = 0


@dataclass(frozen=True)
class HibmMpmMpmForceScatterReport:
    active_marker_count: int
    invalid_marker_count: int
    active_particle_count: int
    total_marker_force_n: tuple[float, float, float]
    total_mpm_external_force_n: tuple[float, float, float]
    action_reaction_residual_n: float


@dataclass(frozen=True)
class HibmMpmExternalForceClearReport:
    cleared_particle_count: int
    max_abs_external_force_before_n: float


@dataclass(frozen=True)
class HibmMpmSurfaceUpdateReport:
    updated_marker_count: int
    invalid_marker_count: int
    max_marker_displacement_m: float
    max_marker_speed_mps: float
    geometry_updated_marker_count: int = 0
    geometry_invalid_marker_count: int = 0
    max_marker_normal_change: float = 0.0
    max_marker_area_change_m2: float = 0.0


@dataclass(frozen=True)
class HibmMpmIbNodeSearchReport:
    near_boundary_node_count: int
    external_ib_node_count: int
    internal_node_count: int
    invalid_projection_count: int


@dataclass(frozen=True)
class HibmMpmIbBoundaryConditionReport:
    no_slip_dirichlet_count: int
    pressure_neumann_count: int
    inactive_internal_node_count: int
    max_abs_velocity_mps: float = 0.0


@dataclass(frozen=True)
class HibmMpmVelocityDirichletBoundaryReport:
    active_velocity_dirichlet_rows: int
    inactive_obstacle_rows: int
    max_abs_velocity_mps: float
    raw_reconstructed_max_abs_velocity_mps: float = 0.0
    boundary_velocity_only_row_count: int = 0
    primary_region_active_rows: int = 0
    secondary_region_active_rows: int = 0
    other_region_active_rows: int = 0
    unassigned_region_active_rows: int = 0
    invalid_reconstruction_row_count: int = 0
    invalid_no_fluid_sample_row_count: int = 0
    invalid_nonpositive_gap_row_count: int = 0
    invalid_node_behind_boundary_row_count: int = 0
    invalid_node_beyond_interior_row_count: int = 0
    narrow_gap_boundary_velocity_row_count: int = 0
    relocated_row_count: int = 0
    relocation_merged_row_count: int = 0
    relocation_blocked_row_count: int = 0
    min_projection_weight: float = 0.0
    max_projection_weight: float = 0.0


@dataclass(frozen=True)
class HibmMpmPressureDisconnectedRegionReport:
    cell_count: int = 0
    component_count: int = 0
    component_raw_count: int = 0
    largest_component_cell_count: int = 0
    singleton_component_count: int = 0
    small_component_threshold_cells: int = (
        HIBM_PRESSURE_DISCONNECTED_SMALL_COMPONENT_THRESHOLD_CELLS
    )
    small_component_count: int = 0
    small_component_cell_count: int = 0
    component_overflow: bool = False
    component_labels_converged: bool = True
    min_i: int = -1
    min_j: int = -1
    min_k: int = -1
    max_i: int = -1
    max_j: int = -1
    max_k: int = -1
    primary_region_stencil_cell_count: int = 0
    secondary_region_stencil_cell_count: int = 0
    other_region_stencil_cell_count: int = 0
    unassigned_region_stencil_cell_count: int = 0


@dataclass(frozen=True)
class HibmMpmPressureNeumannMatrixReport:
    active_pressure_neumann_rows: int
    rhs_integral: float
    max_abs_rhs: float
    skipped_velocity_dirichlet_row_count: int = 0
    skipped_pressure_boundary_adjacent_row_count: int = 0
    skipped_obstacle_owner_row_count: int = 0
    relocated_obstacle_owner_row_count: int = 0
    duplicate_owner_row_count: int = 0
    overflow_owner_row_count: int = 0
    max_owner_slot_count: int = 0
    pressure_interface_row_list_enabled: bool = False
    pressure_interface_row_list_count: int = 0
    active_pressure_neumann_marker_count: int = 0
    max_pressure_neumann_rows_per_marker: int = 0
    invalid_reconstruction_row_count: int = 0
    invalid_unreconstructable_row_count: int = 0
    invalid_bad_marker_row_count: int = 0
    invalid_nonpositive_volume_row_count: int = 0
    min_reconstruction_gap_m: float = 0.0
    max_reconstruction_gap_m: float = 0.0
    max_transmissibility_m: float = 0.0
    max_raw_transmissibility_m: float = 0.0
    max_transmissibility_limit_m: float = 0.0
    transmissibility_capped_row_count: int = 0
    max_diagonal_per_m2: float = 0.0


@dataclass(frozen=True)
class HibmMpmPressureNeumannGradientReport:
    active_marker_count: int
    max_abs_gradient_pa_per_m: float
    max_raw_abs_gradient_pa_per_m: float = 0.0
    limited_gradient_count: int = 0


@dataclass(frozen=True)
class HibmMpmSharpFluidToMpmLoadReport:
    ib_node_search: HibmMpmIbNodeSearchReport
    internal_obstacle_cell_count: int
    solid_band_nonprojectable_cell_count: int
    pressure_disconnected_nonprojectable_cell_count: int
    pressure_disconnected_region: HibmMpmPressureDisconnectedRegionReport
    boundary_conditions: HibmMpmIbBoundaryConditionReport
    pressure_neumann_gradient: HibmMpmPressureNeumannGradientReport | None
    velocity_dirichlet: HibmMpmVelocityDirichletBoundaryReport
    pressure_neumann: HibmMpmPressureNeumannMatrixReport
    fluid_predictor_applied: bool
    fluid_projection: dict[str, Any]
    no_slip_residual: HibmMpmNoSlipResidualReport
    fluid_stress: HibmMpmFluidStressSampleReport
    marker_forces: HibmMpmSurfaceMarkerForceReport
    mpm_external_force_clear: HibmMpmExternalForceClearReport
    mpm_force_scatter: HibmMpmMpmForceScatterReport
    pressure_neumann_invalid_diagnostic_rows: tuple[dict[str, Any], ...] = ()
    # S2-A8' band population split, sampled from the final band sweep:
    # interior slivers (classified candidates) vs enclosed water
    # (unclassified candidates). -1 means the band ran without a split
    # (default bitwise-unchanged mode).
    solid_band_interior_cell_count: int = -1
    solid_band_enclosed_water_cell_count: int = -1
    solid_band_velocity_dirichlet_protected_cell_count: int = -1
    solid_band_mask_protected_cell_count: int = -1
    row_cloud_orphan_cell_count: int = 0
    row_cloud_orphan_component_count: int = 0
    overflow_singleton_cleanup_cell_count: int = 0
    overflow_singleton_cleanup_component_count: int = 0
    # S2-A12 air-backed closure region (default off => -1 sentinels):
    # selected-component conversion census + far-side seeding health.
    air_backed_cell_count: int = -1
    air_backed_component_count: int = -1
    air_backed_cell_volume_m3: float = -1.0
    air_backed_seed_marker_count: int = -1
    air_backed_seed_missed_marker_count: int = -1
    air_backed_seed_fallback_cell_count: int = -1
    air_backed_reachability_barrier_cell_count: int = -1


@dataclass(frozen=True)
class HibmMpmSharpMpmStepReport:
    fluid_to_mpm_loads: HibmMpmSharpFluidToMpmLoadReport
    mpm: Any
    surface_feedback: HibmMpmSurfaceUpdateReport
    next_ib_node_search: HibmMpmIbNodeSearchReport
    next_internal_obstacle_cell_count: int
    next_solid_band_nonprojectable_cell_count: int
    next_pressure_disconnected_nonprojectable_cell_count: int
    next_pressure_disconnected_region: HibmMpmPressureDisconnectedRegionReport
    next_boundary_conditions: HibmMpmIbBoundaryConditionReport
    next_velocity_dirichlet: HibmMpmVelocityDirichletBoundaryReport
    next_pressure_neumann: HibmMpmPressureNeumannMatrixReport
    next_pressure_neumann_gradient: HibmMpmPressureNeumannGradientReport | None = None
    next_pressure_neumann_invalid_diagnostic_rows: tuple[dict[str, Any], ...] = ()
    next_solid_band_interior_cell_count: int = -1
    next_solid_band_enclosed_water_cell_count: int = -1
    next_solid_band_velocity_dirichlet_protected_cell_count: int = -1
    next_solid_band_mask_protected_cell_count: int = -1
    next_row_cloud_orphan_cell_count: int = 0
    next_row_cloud_orphan_component_count: int = 0
    next_overflow_singleton_cleanup_cell_count: int = 0
    next_overflow_singleton_cleanup_component_count: int = 0
    post_solid_kinematic_projection_applied: bool = False
    post_solid_fluid_projection: dict[str, Any] | None = None
    post_solid_no_slip_residual: HibmMpmNoSlipResidualReport | None = None


@dataclass(frozen=True)
class HibmMpmSharpNeoHookeanStepReport(HibmMpmSharpMpmStepReport):
    pass