import argparse
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

import numpy as np
import taichi as ti


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core import (
    AxisAlignedBoundary,
    CG_PRECONDITIONER_CHOICES,
    CartesianGrid,
    CartesianFluidSolver,
    CflSubstepController,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FluidDomainSpec,
    GradedGridSpec,
    HibmMpmSharpCouplingState,
    INTERFACE_REACTION_SOLVER_CHOICES,
    InterfaceReactionFixedPointResult,
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
from simulation_core.pressure_interface import (
    far_pressure_side_normal_sign_from_direction,
)
from simulation_core.runtime import init_taichi

from .cli import (
    FLUID_ADVECTION_SCHEME_CHOICES,
    FSI_STABILIZATION_PRESET_CHOICES,
    FSI_STABILIZATION_PRESET_CONFLICT_POLICY,
    FSI_STABILIZATION_PRESET_MANAGED_FIELDS,
    INTERFACE_REACTION_ROBIN_TARGET_CHOICES,
    PRESSURE_SOLVE_FAILURE_POLICY_CHOICES,
    PRESSURE_SOLVER_CHOICES,
    fsi_stabilization_effective_parameters_from_args,
    parse_args,
    raise_for_unsupported_hibm_mpm_sharp_iteration_options,
    resolve_fsi_stabilization_preset_parameters,
)
from .checkpointing import (
    CHECKPOINT_ARG_FINGERPRINT_FIELDS,
    CHECKPOINT_MARKER_STATE_FIELD_NAMES,
    RUN_CHECKPOINT_FILENAME,
    RUN_CHECKPOINT_VERSION,
    checkpoint_path_for_args,
    checkpoint_run_fingerprint,
    load_run_checkpoint,
    relaxed_sharp_marker_state_arrays,
    relaxed_sharp_pressure_neumann_gradient_state_array,
    restore_sharp_marker_state_arrays,
    restore_sharp_pressure_neumann_gradient_state_array,
    resume_history_rows_for_checkpoint,
    sharp_marker_fixed_point_residual_diagnostics_mps,
    sharp_marker_fixed_point_residual_mps,
    sharp_marker_state_arrays,
    sharp_pressure_neumann_gradient_state_array,
    validate_checkpoint_run_fingerprint,
    validate_resume_history_checkpoint_alignment,
    write_run_checkpoint,
)
from .diagnostics import (
    _raise_for_closure_coverage_floor,
    _raise_for_step_numerical_guard,
    _raise_for_step_solid_out_of_bounds_guard,
    force_decomposition_report,
    fsi_trial_acceptance_passes,
    fsi_trial_acceptance_rejection_reason,
    sharp_report_fluid_projection_failure_reason,
)
from .history import (
    FINITE_REQUIRED_ROW_FIELDS,
    HIBM_MPM_SHARP_REQUIRED_ROW_FIELDS,
    NEO_HOOKEAN_REQUIRED_ROW_FIELDS,
    _final_row_int,
    _final_row_number,
    _final_row_number_or_none,
    _required_finite_report_number,
    _required_finite_row_number,
    _required_finite_row_vector,
    _required_finite_triplet,
    _row_bool,
    _rows_any_bool,
    _rows_max_int,
    count_enabled_unconverged_fsi_rows,
    divergence_sample_report_fields,
    finite_required_row_fields_for_mode,
    finite_required_row_fields_for_solid_model,
    read_csv_rows,
    required_fluid_impulse_report,
    required_projected_ibm_force_report,
    solid_force_vector_from_report,
    solid_mpm_force_nonzero_when_pressure_loaded,
    write_csv,
)
from .outputs import run_process_completion_status
from .schedules import (
    PRESSURE_SCHEDULE_FIELDS,
    pressure_schedule_applied_in_history,
    pressure_schedule_dict,
    pressure_schedule_from_config,
    pressure_schedule_pa,
    pressure_schedule_step_end_pa,
    spec_with_pressure_schedule_overrides,
)
from .snapshots import (
    _write_fluid_snapshot_npz,
    _write_hibm_high_residual_cell_dump,
    _write_hibm_pressure_neumann_invalid_row_dump,
    _write_hibm_zero_correctable_cell_dump,
    _write_step_failure_artifacts,
)
from .source_config import (
    DEFAULT_SOURCE_CONFIG,
    PressureBoundaryShellMapping,
    _face_ids_for_region,
    _selection_ids_as_int_tuple,
    _source_config_pressure_load_direction,
    _vector3,
    load_source_config,
    source_config_cad_provenance_report,
    source_config_pressure_boundary_shell_mapping,
    source_config_pressure_load_region_id,
    source_config_requests_fluid_active_mask,
    source_config_requests_reduced_water_intersection,
    source_config_requests_region14_aperture_carve,
    source_config_shell_region_pair,
    source_config_solid_obstacle_particle_region_ids,
    source_config_volume_particle_cache_path,
)
from .spec import (
    SquidReducedSpec,
    _finite_positive_scale,
    infer_spec,
    required_tuple3,
    resolve_step_count,
    shell_surface_mass_budget,
    spec_with_membrane_thickness_scale,
)


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
        preserve_existing_obstacles: ti.i32,
    ):
        for i, j, k in self.fluid.obstacle:
            existing_obstacle = self.fluid.obstacle[i, j, k] != 0
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
            reduced_water = chamber or nozzle or outlet_plume or downstream_farfield
            if preserve_existing_obstacles == 1:
                self.fluid.obstacle[i, j, k] = 0 if reduced_water and not existing_obstacle else 1
            else:
                self.fluid.obstacle[i, j, k] = 0 if reduced_water else 1

    def _apply_reduced_squid_water_domain(
        self,
        *,
        preserve_existing_obstacles: bool,
    ) -> None:
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
            1 if preserve_existing_obstacles else 0,
        )

    def mark_reduced_squid_water_domain(self) -> None:
        self._apply_reduced_squid_water_domain(preserve_existing_obstacles=False)

    def intersect_current_obstacles_with_reduced_squid_water_domain(self) -> None:
        self._apply_reduced_squid_water_domain(preserve_existing_obstacles=True)

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

    def sample_cfl_report(
        self,
        *,
        dt_s: float | None = None,
    ) -> dict[str, float]:
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
        self.last_sample_report_host_reads = 1
        max_speed = float(sample_values[14])
        cfl_dt_s = float(self.spec.dt_s) if dt_s is None else float(dt_s)
        return {
            "max_fluid_speed_mps": max_speed,
            "cfl": max_speed * cfl_dt_s / max(h, 1.0e-12),
        }


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


def fsi_same_step_rerun_triggered(
    *,
    current_iterations_requested: int,
    rerun_iterations_max: int,
    residual_norm_n: float,
    residual_threshold_n: float,
    converged: bool,
    safety_rejected: bool = False,
) -> bool:
    """Return whether a projected/reduced FSI step should be rerun in-place."""
    if rerun_iterations_max <= current_iterations_requested:
        return False
    if safety_rejected:
        return True
    if not math.isfinite(residual_threshold_n):
        return False
    residual = float(residual_norm_n)
    if math.isnan(residual):
        return False
    return (not bool(converged)) and residual > residual_threshold_n


def fsi_same_step_rerun_fluid_substeps(
    *,
    current_substeps: int,
    max_substeps: int,
    substep_factor: float,
    safety_rejected: bool,
) -> int:
    """Return same-step fluid substeps after an all-rejected safety attempt."""
    current = int(current_substeps)
    maximum = int(max_substeps)
    factor = float(substep_factor)
    if current < 1 or maximum <= current:
        return current
    if not safety_rejected:
        return current
    if not math.isfinite(factor) or factor <= 1.0:
        return current
    requested = max(current + 1, int(math.ceil(float(current) * factor)))
    return min(maximum, requested)


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


def _cell_indices_for_points(
    points_m: np.ndarray,
    grid: CartesianGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must be an Nx3 array")
    faces = (
        np.asarray(grid.cell_faces_x_m, dtype=np.float64),
        np.asarray(grid.cell_faces_y_m, dtype=np.float64),
        np.asarray(grid.cell_faces_z_m, dtype=np.float64),
    )
    nodes = tuple(int(value) for value in grid.grid_nodes)
    i = np.searchsorted(faces[0], points[:, 0], side="right") - 1
    j = np.searchsorted(faces[1], points[:, 1], side="right") - 1
    k = np.searchsorted(faces[2], points[:, 2], side="right") - 1
    tolerance = 1.0e-12
    on_x_max = np.isclose(points[:, 0], faces[0][-1], rtol=0.0, atol=tolerance)
    on_y_max = np.isclose(points[:, 1], faces[1][-1], rtol=0.0, atol=tolerance)
    on_z_max = np.isclose(points[:, 2], faces[2][-1], rtol=0.0, atol=tolerance)
    i[on_x_max] = nodes[0] - 1
    j[on_y_max] = nodes[1] - 1
    k[on_z_max] = nodes[2] - 1
    valid = (
        (i >= 0)
        & (i < nodes[0])
        & (j >= 0)
        & (j < nodes[1])
        & (k >= 0)
        & (k < nodes[2])
    )
    return i.astype(np.int64), j.astype(np.int64), k.astype(np.int64), valid


def _surface_region_seed_mask(
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    region_ids: Sequence[int],
    radius_cells: int = 1,
    normal_probe_distance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    import trimesh

    nodes = tuple(int(value) for value in grid.grid_nodes)
    seed = np.zeros(nodes, dtype=bool)
    unique_region_ids = tuple(sorted({int(value) for value in region_ids}))
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in unique_region_ids:
        face_ids = _face_ids_for_region(dict(config), region_id)
        region_face_counts[str(region_id)] = len(face_ids)
        selected_face_ids.extend(face_ids)
    radius = max(0, int(radius_cells))
    if not selected_face_ids:
        return seed, {
            "fluid_active_mask_surface_seed_region_ids": unique_region_ids,
            "fluid_active_mask_surface_seed_face_count": 0,
            "fluid_active_mask_surface_seed_point_count": 0,
            "fluid_active_mask_surface_seed_point_in_grid_count": 0,
            "fluid_active_mask_surface_seed_cell_count": 0,
            "fluid_active_mask_surface_seed_radius_cells": radius,
            "fluid_active_mask_surface_seed_normal_probe_distance_m": max(
                0.0,
                float(normal_probe_distance_m),
            ),
            "fluid_active_mask_surface_seed_normal_probe_point_count": 0,
            "fluid_active_mask_surface_seed_region_face_counts": region_face_counts,
        }
    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    points = np.concatenate(
        (
            centroids,
            tri[:, 0, :],
            tri[:, 1, :],
            tri[:, 2, :],
        ),
        axis=0,
    )
    normal_probe_point_count = 0
    normal_probe_distance = max(0.0, float(normal_probe_distance_m))
    if normal_probe_distance > 0.0:
        raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
        normal_norms = np.linalg.norm(raw_normals, axis=1)
        valid_normals = normal_norms > 1.0e-30
        if np.any(valid_normals):
            unit_normals = np.zeros_like(raw_normals)
            unit_normals[valid_normals] = (
                raw_normals[valid_normals]
                / normal_norms[valid_normals, None]
            )
            normal_probe_points = (
                centroids[valid_normals]
                + unit_normals[valid_normals] * normal_probe_distance
            )
            normal_probe_point_count = int(normal_probe_points.shape[0])
            points = np.concatenate((points, normal_probe_points), axis=0)
    i, j, k, valid = _cell_indices_for_points(points, grid)
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                seed[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return seed, {
        "fluid_active_mask_surface_seed_region_ids": unique_region_ids,
        "fluid_active_mask_surface_seed_face_count": int(len(selected_face_ids)),
        "fluid_active_mask_surface_seed_point_count": int(points.shape[0]),
        "fluid_active_mask_surface_seed_point_in_grid_count": int(np.count_nonzero(valid)),
        "fluid_active_mask_surface_seed_cell_count": int(np.count_nonzero(seed)),
        "fluid_active_mask_surface_seed_radius_cells": radius,
        "fluid_active_mask_surface_seed_normal_probe_distance_m": normal_probe_distance,
        "fluid_active_mask_surface_seed_normal_probe_point_count": normal_probe_point_count,
        "fluid_active_mask_surface_seed_region_face_counts": region_face_counts,
    }


def _clear_surface_region_normal_probe_obstacle_cells(
    obstacle: np.ndarray,
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    region_ids: Sequence[int],
    normal_probe_distance_m: float,
    radius_cells: int = 0,
) -> dict[str, object]:
    import trimesh

    if obstacle.shape != tuple(int(value) for value in grid.grid_nodes):
        raise ValueError("obstacle shape must match grid.grid_nodes")
    unique_region_ids = tuple(sorted({int(value) for value in region_ids}))
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in unique_region_ids:
        face_ids = _face_ids_for_region(dict(config), region_id)
        region_face_counts[str(region_id)] = len(face_ids)
        selected_face_ids.extend(face_ids)
    probe_distance = max(0.0, float(normal_probe_distance_m))
    radius = max(0, int(radius_cells))
    if not selected_face_ids or probe_distance <= 0.0:
        return {
            "fluid_active_mask_surface_probe_clear_region_ids": unique_region_ids,
            "fluid_active_mask_surface_probe_clear_face_count": int(
                len(selected_face_ids)
            ),
            "fluid_active_mask_surface_probe_clear_point_count": 0,
            "fluid_active_mask_surface_probe_clear_cell_count": 0,
            "fluid_active_mask_surface_probe_clear_cells_ijk": (),
            "fluid_active_mask_surface_probe_clear_radius_cells": radius,
            "fluid_active_mask_surface_probe_clear_distance_m": probe_distance,
            "fluid_active_mask_surface_probe_clear_region_face_counts": (
                region_face_counts
            ),
        }

    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    normal_norms = np.linalg.norm(raw_normals, axis=1)
    valid_normals = normal_norms > 1.0e-30
    points = np.empty((0, 3), dtype=np.float64)
    if np.any(valid_normals):
        unit_normals = (
            raw_normals[valid_normals]
            / normal_norms[valid_normals, None]
        )
        points = centroids[valid_normals] + unit_normals * probe_distance

    nodes = tuple(int(value) for value in grid.grid_nodes)
    i, j, k, valid = _cell_indices_for_points(points, grid)
    clear_mask = np.zeros(obstacle.shape, dtype=bool)
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                clear_mask[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    cleared_mask = obstacle & clear_mask
    cleared = int(np.count_nonzero(cleared_mask))
    cleared_cells_ijk = tuple(
        tuple(int(value) for value in cell)
        for cell in np.argwhere(cleared_mask)
    )
    obstacle[clear_mask] = False
    return {
        "fluid_active_mask_surface_probe_clear_region_ids": unique_region_ids,
        "fluid_active_mask_surface_probe_clear_face_count": int(len(selected_face_ids)),
        "fluid_active_mask_surface_probe_clear_point_count": int(points.shape[0]),
        "fluid_active_mask_surface_probe_clear_cell_count": cleared,
        "fluid_active_mask_surface_probe_clear_cells_ijk": cleared_cells_ijk,
        "fluid_active_mask_surface_probe_clear_radius_cells": radius,
        "fluid_active_mask_surface_probe_clear_distance_m": probe_distance,
        "fluid_active_mask_surface_probe_clear_region_face_counts": region_face_counts,
    }


def _solid_band_protection_mask_from_cells(
    shape: Sequence[int],
    cells_ijk: Sequence[Sequence[int]],
    *,
    radius_cells: int = 0,
) -> np.ndarray:
    mask = np.zeros(tuple(int(value) for value in shape), dtype=np.int32)
    radius = max(0, int(radius_cells))
    offsets = tuple(range(-radius, radius + 1))
    nx, ny, nz = mask.shape
    for cell in cells_ijk:
        if len(cell) != 3:
            continue
        ci, cj, ck = (int(value) for value in cell)
        for di in offsets:
            i = ci + di
            if i < 0 or i >= nx:
                continue
            for dj in offsets:
                j = cj + dj
                if j < 0 or j >= ny:
                    continue
                for dk in offsets:
                    k = ck + dk
                    if 0 <= k < nz:
                        mask[i, j, k] = 1
    return mask


def _mark_particle_obstacle_cells(
    *,
    grid: CartesianGrid,
    particle_positions_m: np.ndarray,
    particle_region_ids: np.ndarray,
    obstacle_region_ids: Sequence[int],
    dilation_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    nodes = tuple(int(value) for value in grid.grid_nodes)
    obstacle = np.zeros(nodes, dtype=bool)
    obstacle_regions = {int(value) for value in obstacle_region_ids}
    selected = np.isin(
        np.asarray(particle_region_ids, dtype=np.int32),
        np.asarray(sorted(obstacle_regions), dtype=np.int32),
    )
    i, j, k, valid = _cell_indices_for_points(
        np.asarray(particle_positions_m, dtype=np.float64)[selected],
        grid,
    )
    selected_valid_count = int(np.count_nonzero(valid))
    radius = max(0, int(dilation_cells))
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                obstacle[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return obstacle, {
        "particle_obstacle_region_ids": tuple(sorted(obstacle_regions)),
        "selected_particle_count": int(np.count_nonzero(selected)),
        "selected_particle_in_grid_count": selected_valid_count,
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "particle_stamp_dilation_cells": radius,
        "particle_stamp_method": "volume_particle_cell_stamp",
    }


def _mark_surface_obstacle_cells(
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    surface_region_ids: Sequence[int],
    dilation_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    import trimesh

    nodes = tuple(int(value) for value in grid.grid_nodes)
    obstacle = np.zeros(nodes, dtype=bool)
    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in surface_region_ids:
        face_ids = _face_ids_for_region(dict(config), int(region_id))
        region_face_counts[str(int(region_id))] = len(face_ids)
        selected_face_ids.extend(face_ids)
    if not selected_face_ids:
        return obstacle, {
            "surface_obstacle_region_ids": tuple(int(value) for value in surface_region_ids),
            "selected_surface_face_count": 0,
            "selected_surface_point_count": 0,
            "selected_surface_point_in_grid_count": 0,
            "raw_solid_obstacle_cell_count": 0,
            "surface_stamp_dilation_cells": max(0, int(dilation_cells)),
            "surface_stamp_method": "surface_triangle_centroid_and_vertex_cell_stamp",
            "surface_region_face_counts": region_face_counts,
        }
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    points = np.concatenate(
        (
            centroids,
            tri[:, 0, :],
            tri[:, 1, :],
            tri[:, 2, :],
        ),
        axis=0,
    )
    i, j, k, valid = _cell_indices_for_points(points, grid)
    radius = max(0, int(dilation_cells))
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                obstacle[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return obstacle, {
        "surface_obstacle_region_ids": tuple(int(value) for value in surface_region_ids),
        "surface_mesh_path": str(mesh_path),
        "selected_surface_face_count": int(len(selected_face_ids)),
        "selected_surface_point_count": int(points.shape[0]),
        "selected_surface_point_in_grid_count": int(np.count_nonzero(valid)),
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "surface_stamp_dilation_cells": radius,
        "surface_stamp_method": "surface_triangle_centroid_and_vertex_cell_stamp",
        "surface_region_face_counts": region_face_counts,
    }


def _apply_region14_opening_carve_to_obstacle(
    obstacle: np.ndarray,
    grid: CartesianGrid,
    *,
    aperture_geometry: Mapping[str, object],
    radius_cells: int,
    depth_cells: int,
) -> int:
    if not bool(aperture_geometry.get("available", False)):
        return 0
    center = aperture_geometry.get("area_weighted_centroid_m", ())
    if not isinstance(center, Sequence) or len(center) < 3:
        return 0
    radius_m = float(aperture_geometry.get("vertex_radius_p95_m", 0.0))
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        return 0
    center_x = float(center[0])
    center_y = float(center[1])
    center_z = float(center[2])
    max_xy_spacing_m = max(
        max(float(value) for value in grid.cell_widths_x_m),
        max(float(value) for value in grid.cell_widths_y_m),
    )
    max_z_spacing_m = max(float(value) for value in grid.cell_widths_z_m)
    carve_radius_m = radius_m + max(0, int(radius_cells)) * max_xy_spacing_m
    carve_half_depth_m = max(1, int(depth_cells)) * max_z_spacing_m
    x = np.asarray(grid.cell_centers_x_m, dtype=np.float64)
    y = np.asarray(grid.cell_centers_y_m, dtype=np.float64)
    z = np.asarray(grid.cell_centers_z_m, dtype=np.float64)
    radial = (
        (x[:, None, None] - center_x) ** 2
        + (y[None, :, None] - center_y) ** 2
    ) <= carve_radius_m * carve_radius_m
    axial = np.abs(z[None, None, :] - center_z) <= carve_half_depth_m
    carve = radial & axial
    carved = int(np.count_nonzero(obstacle & carve))
    obstacle[carve] = False
    return carved


def _z_min_connected_active_mask(
    candidate_fluid_mask: np.ndarray,
    *,
    seed_radius_cells: int,
) -> tuple[np.ndarray, int]:
    active = np.asarray(candidate_fluid_mask, dtype=bool)
    seed = np.zeros(active.shape, dtype=bool)
    nz = active.shape[2]
    seed_depth = max(1, int(seed_radius_cells))
    seed_depth = min(seed_depth, nz)
    seed[:, :, :seed_depth] = True
    return _connected_active_mask(active, seed)


def _connected_active_mask(
    candidate_fluid_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    active = np.asarray(candidate_fluid_mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    if seed.shape != active.shape:
        raise ValueError("seed_mask shape must match candidate_fluid_mask")
    visited = np.zeros(active.shape, dtype=bool)
    nx, ny, nz = active.shape
    stack: list[tuple[int, int, int]] = []
    seed_indices = np.argwhere(active & seed)
    for i, j, k in seed_indices:
        index = (int(i), int(j), int(k))
        visited[index] = True
        stack.append(index)
    seed_count = len(stack)
    while stack:
        i, j, k = stack.pop()
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
    return visited, seed_count


def _component_count(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.shape, dtype=bool)
    nx, ny, nz = active.shape
    count = 0
    for seed_index in zip(*np.nonzero(active), strict=False):
        if visited[seed_index]:
            continue
        count += 1
        stack = [tuple(int(value) for value in seed_index)]
        visited[seed_index] = True
        while stack:
            i, j, k = stack.pop()
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
    return count


def _mask_bbox_ijk(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return None
    mins = tuple(int(value) for value in np.min(indices, axis=0))
    maxs = tuple(int(value) for value in np.max(indices, axis=0))
    return mins, maxs


def _minimum_obstacle_carve_path(
    *,
    obstacle: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int] | None:
    obstacle_bool = np.asarray(obstacle, dtype=bool)
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if obstacle_bool.shape != source.shape or source.shape != target.shape:
        raise ValueError("obstacle, source_mask, and target_mask shapes must match")
    nx, ny, nz = obstacle_bool.shape
    total = nx * ny * nz
    source_indices = np.flatnonzero(source.reshape(-1))
    if source_indices.size == 0 or not np.any(target):
        return None
    target_flat = target.reshape(-1)
    obstacle_flat = obstacle_bool.reshape(-1)
    max_distance = np.iinfo(np.int32).max
    distance = np.full(total, max_distance, dtype=np.int32)
    previous = np.full(total, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for flat_index in source_indices:
        index = int(flat_index)
        distance[index] = 0
        queue.append(index)

    def unravel(index: int) -> tuple[int, int, int]:
        i = index // (ny * nz)
        rem = index - i * ny * nz
        j = rem // nz
        k = rem - j * nz
        return int(i), int(j), int(k)

    def ravel(i: int, j: int, k: int) -> int:
        return int((i * ny + j) * nz + k)

    end_index = -1
    while queue:
        index = queue.popleft()
        if target_flat[index]:
            end_index = int(index)
            break
        i, j, k = unravel(index)
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            neighbor = ravel(ni, nj, nk)
            step_cost = 1 if obstacle_flat[neighbor] else 0
            candidate_distance = int(distance[index]) + step_cost
            if candidate_distance >= int(distance[neighbor]):
                continue
            distance[neighbor] = candidate_distance
            previous[neighbor] = index
            if step_cost == 0:
                queue.appendleft(neighbor)
            else:
                queue.append(neighbor)
    if end_index < 0:
        return None
    path_indices: list[int] = []
    cursor = end_index
    while cursor >= 0:
        path_indices.append(int(cursor))
        cursor = int(previous[cursor])
    path_indices.reverse()
    return [unravel(index) for index in path_indices], int(distance[end_index])


def _connect_surface_seed_components_to_zmin(
    obstacle: np.ndarray,
    *,
    boundary_seed: np.ndarray,
    surface_seed: np.ndarray,
    max_carve_cells: int,
) -> dict[str, object]:
    if obstacle.shape != boundary_seed.shape or boundary_seed.shape != surface_seed.shape:
        raise ValueError("obstacle, boundary_seed, and surface_seed shapes must match")
    max_carve = max(0, int(max_carve_cells))
    carved_mask = np.zeros(obstacle.shape, dtype=bool)
    total_carved = 0
    connected_paths = 0
    skipped_max_carve = False
    path_reports: list[dict[str, object]] = []

    candidate_fluid = ~np.asarray(obstacle, dtype=bool)
    zmin_active, _ = _connected_active_mask(candidate_fluid, boundary_seed)
    surface_active, _ = _connected_active_mask(candidate_fluid, surface_seed)
    target = surface_active & ~zmin_active
    initial_unreachable_cells = int(np.count_nonzero(target))
    initial_unreachable_components = _component_count(target)

    while np.any(target):
        path_result = _minimum_obstacle_carve_path(
            obstacle=obstacle,
            source_mask=zmin_active,
            target_mask=target,
        )
        if path_result is None:
            break
        path, obstacle_cost = path_result
        if obstacle_cost <= 0:
            break
        if total_carved + obstacle_cost > max_carve:
            skipped_max_carve = True
            break
        path_carved = 0
        for i, j, k in path:
            if bool(obstacle[i, j, k]):
                obstacle[i, j, k] = False
                carved_mask[i, j, k] = True
                path_carved += 1
        if path_carved <= 0:
            break
        total_carved += path_carved
        connected_paths += 1
        path_reports.append(
            {
                "path_cell_count": int(len(path)),
                "carved_cell_count": int(path_carved),
                "obstacle_cost": int(obstacle_cost),
                "start_ijk": tuple(int(value) for value in path[0]),
                "end_ijk": tuple(int(value) for value in path[-1]),
            }
        )
        candidate_fluid = ~np.asarray(obstacle, dtype=bool)
        zmin_active, _ = _connected_active_mask(candidate_fluid, boundary_seed)
        surface_active, _ = _connected_active_mask(candidate_fluid, surface_seed)
        target = surface_active & ~zmin_active

    final_unreachable_cells = int(np.count_nonzero(target))
    final_unreachable_components = _component_count(target)
    bbox = _mask_bbox_ijk(carved_mask)
    return {
        "enabled": True,
        "max_carve_cells": int(max_carve),
        "initial_unreachable_surface_seed_cell_count": initial_unreachable_cells,
        "initial_unreachable_surface_seed_component_count": initial_unreachable_components,
        "final_unreachable_surface_seed_cell_count": final_unreachable_cells,
        "final_unreachable_surface_seed_component_count": final_unreachable_components,
        "connected_path_count": int(connected_paths),
        "carved_cell_count": int(total_carved),
        "skipped_by_max_carve_limit": bool(skipped_max_carve),
        "carved_bbox_ijk": None if bbox is None else bbox,
        "paths": tuple(path_reports),
    }


def build_source_config_fluid_obstacle_mask(
    *,
    config: Mapping[str, object],
    source_config_path: Path,
    grid: CartesianGrid,
    aperture_geometry: Mapping[str, object],
    connect_surface_seeds_to_zmin: bool = False,
    surface_seed_zmin_connection_max_carve_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        raise ValueError("source_config analysis_settings must be a mapping")
    mode = str(analysis.get("fluid_active_mask_mode", ""))
    if mode not in {"ibamr_like_connected_component", "fsi_connected_component"}:
        raise ValueError(
            "source_config fluid active mask mode must be "
            "ibamr_like_connected_component or fsi_connected_component"
        )
    seed_sides = tuple(
        str(value)
        for value in analysis.get("fluid_active_mask_seed_boundary_sides", ("z_min",))
    )
    if "z_min" not in seed_sides:
        raise ValueError("refactored squid source-config active mask currently requires z_min seeding")
    surface_only_region_ids = tuple(
        sorted(
            set(
                _selection_ids_as_int_tuple(
                    analysis.get("solid_obstacle_surface_only_region_ids", ()),
                )
            )
        )
    )
    cache_path: Path | None = None
    available_region_ids: tuple[int, ...] = ()
    obstacle_region_ids: tuple[int, ...] = ()
    obstacle_report: dict[str, object]
    if surface_only_region_ids:
        obstacle, obstacle_report = _mark_surface_obstacle_cells(
            config=config,
            grid=grid,
            surface_region_ids=surface_only_region_ids,
            dilation_cells=int(analysis.get("solid_obstacle_mask_dilation_cells", 0) or 0),
        )
        obstacle_region_ids = surface_only_region_ids
    else:
        cache_path = source_config_volume_particle_cache_path(source_config_path)
        particle_cache = np.load(cache_path)
        positions = np.asarray(particle_cache["particle_rest_positions_m"], dtype=np.float64)
        region_ids = np.asarray(particle_cache["particle_region_ids"], dtype=np.int32)
        available_region_ids = tuple(
            int(value) for value in np.unique(region_ids).astype(np.int32).tolist()
        )
        obstacle_region_ids = source_config_solid_obstacle_particle_region_ids(
            config,
            available_region_ids,
        )
        obstacle, obstacle_report = _mark_particle_obstacle_cells(
            grid=grid,
            particle_positions_m=positions,
            particle_region_ids=region_ids,
            obstacle_region_ids=obstacle_region_ids,
            dilation_cells=int(analysis.get("solid_obstacle_mask_dilation_cells", 0) or 0),
        )
    carved_count = 0
    if source_config_requests_region14_aperture_carve(config):
        carved_count = _apply_region14_opening_carve_to_obstacle(
            obstacle,
            grid,
            aperture_geometry=aperture_geometry,
            radius_cells=int(
                analysis.get("solid_obstacle_opening_carve_radius_cells", 1) or 1
            ),
            depth_cells=int(
                analysis.get("solid_obstacle_opening_carve_depth_cells", 2) or 2
            ),
        )
    seed_radius_cells = int(analysis.get("fluid_active_mask_seed_radius_cells", 1) or 1)
    surface_seed_normal_probe_cells = max(
        0,
        int(analysis.get("fluid_active_mask_surface_seed_normal_probe_cells", 1) or 0),
    )
    surface_seed_normal_probe_distance_m = 0.0
    if surface_seed_normal_probe_cells > 0:
        min_cell_spacing_m = min(
            min(float(value) for value in grid.cell_widths_x_m),
            min(float(value) for value in grid.cell_widths_y_m),
            min(float(value) for value in grid.cell_widths_z_m),
        )
        surface_seed_normal_probe_distance_m = (
            float(surface_seed_normal_probe_cells) * min_cell_spacing_m
        )
    clear_region_ids = _selection_ids_as_int_tuple(
        analysis.get("fluid_active_mask_surface_probe_clear_region_ids", ()),
    )
    if (
        not clear_region_ids
        and bool(
            analysis.get(
                "fluid_active_mask_clear_primary_fsi_surface_probe_obstacles",
                True,
            )
        )
    ):
        moving_surface_ids = _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
        )
        if moving_surface_ids:
            clear_region_ids = (int(moving_surface_ids[0]),)
    if clear_region_ids:
        surface_probe_clear_report = _clear_surface_region_normal_probe_obstacle_cells(
            obstacle,
            config=config,
            grid=grid,
            region_ids=clear_region_ids,
            normal_probe_distance_m=surface_seed_normal_probe_distance_m,
            radius_cells=int(
                analysis.get(
                    "fluid_active_mask_surface_probe_clear_radius_cells",
                    0,
                )
                or 0
            ),
        )
    else:
        surface_probe_clear_report = {
            "fluid_active_mask_surface_probe_clear_region_ids": (),
            "fluid_active_mask_surface_probe_clear_face_count": 0,
            "fluid_active_mask_surface_probe_clear_point_count": 0,
            "fluid_active_mask_surface_probe_clear_cell_count": 0,
            "fluid_active_mask_surface_probe_clear_cells_ijk": (),
            "fluid_active_mask_surface_probe_clear_radius_cells": 0,
            "fluid_active_mask_surface_probe_clear_distance_m": (
                surface_seed_normal_probe_distance_m
            ),
            "fluid_active_mask_surface_probe_clear_region_face_counts": {},
        }
    candidate_fluid = ~obstacle
    boundary_seed = np.zeros(candidate_fluid.shape, dtype=bool)
    if "z_min" in seed_sides:
        seed_depth = min(max(1, seed_radius_cells), candidate_fluid.shape[2])
        boundary_seed[:, :, :seed_depth] = True
    seed_region_ids = set(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_region_ids", ()),
        )
    )
    seed_region_ids.update(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_region_id", ()),
        )
    )
    seed_region_ids.update(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_surface_region_ids", ()),
        )
    )
    if bool(analysis.get("fluid_active_mask_seed_fsi_contact_surfaces", True)):
        seed_region_ids.update(
            _selection_ids_as_int_tuple(
                analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
            )
        )
    surface_seed_report: dict[str, object]
    if seed_region_ids:
        surface_seed, surface_seed_report = _surface_region_seed_mask(
            config=config,
            grid=grid,
            region_ids=tuple(sorted(seed_region_ids)),
            radius_cells=seed_radius_cells,
            normal_probe_distance_m=surface_seed_normal_probe_distance_m,
        )
    else:
        surface_seed = np.zeros(candidate_fluid.shape, dtype=bool)
        surface_seed_report = {
            "fluid_active_mask_surface_seed_region_ids": (),
            "fluid_active_mask_surface_seed_face_count": 0,
            "fluid_active_mask_surface_seed_point_count": 0,
            "fluid_active_mask_surface_seed_point_in_grid_count": 0,
            "fluid_active_mask_surface_seed_cell_count": 0,
            "fluid_active_mask_surface_seed_radius_cells": seed_radius_cells,
            "fluid_active_mask_surface_seed_normal_probe_distance_m": (
                surface_seed_normal_probe_distance_m
            ),
            "fluid_active_mask_surface_seed_normal_probe_point_count": 0,
            "fluid_active_mask_surface_seed_region_face_counts": {},
        }
    surface_seed_zmin_connection_report: dict[str, object] = {
        "enabled": False,
        "reason": "not_requested",
        "max_carve_cells": int(max(0, surface_seed_zmin_connection_max_carve_cells)),
    }
    if connect_surface_seeds_to_zmin:
        surface_seed_zmin_connection_report = _connect_surface_seed_components_to_zmin(
            obstacle,
            boundary_seed=boundary_seed,
            surface_seed=surface_seed,
            max_carve_cells=surface_seed_zmin_connection_max_carve_cells,
        )
    candidate_fluid = ~obstacle
    active_water, seed_cell_count = _connected_active_mask(
        candidate_fluid,
        boundary_seed | surface_seed,
    )
    final_obstacle = ~active_water
    report = {
        "enabled": True,
        "method": "source_config_cad_obstacle_z_min_connected_component",
        "mode": mode,
        "source_config_path": str(source_config_path),
        "volume_particle_cache_path": None if cache_path is None else str(cache_path),
        "grid_nodes": tuple(int(value) for value in grid.grid_nodes),
        "available_particle_region_ids": available_region_ids,
        "obstacle_region_ids": obstacle_region_ids,
        "solid_obstacle_opening_carved_cell_count": int(carved_count),
        "fluid_active_mask_seed_boundary_sides": seed_sides,
        "fluid_active_mask_seed_radius_cells": seed_radius_cells,
        "fluid_active_mask_boundary_seed_cell_count": int(
            np.count_nonzero(boundary_seed & candidate_fluid)
        ),
        "fluid_active_mask_surface_seed_candidate_cell_count": int(
            np.count_nonzero(surface_seed & candidate_fluid)
        ),
        "fluid_active_mask_seed_cell_count": int(seed_cell_count),
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "candidate_fluid_cell_count": int(np.count_nonzero(candidate_fluid)),
        "fluid_active_cell_count": int(np.count_nonzero(active_water)),
        "fluid_inactive_cell_count": int(final_obstacle.size - np.count_nonzero(active_water)),
        "final_obstacle_cell_count": int(np.count_nonzero(final_obstacle)),
        "host_device_transfer_policy": "one_time_initial_obstacle_from_numpy_before_steps",
        **obstacle_report,
        **surface_probe_clear_report,
        **surface_seed_report,
        "fluid_active_mask_surface_seed_zmin_connection": (
            surface_seed_zmin_connection_report
        ),
    }
    return final_obstacle.astype(np.int32), report


def _active_water_mask_for_points(
    points_m: np.ndarray,
    *,
    grid: CartesianGrid,
    obstacle_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(obstacle_mask, dtype=np.int32)
    if mask.shape != tuple(grid.grid_nodes):
        raise ValueError(
            f"obstacle_mask shape {mask.shape!r} does not match grid_nodes {tuple(grid.grid_nodes)!r}"
        )
    i, j, k, valid = _cell_indices_for_points(points_m, grid)
    active = np.zeros(valid.shape, dtype=bool)
    active[valid] = mask[i[valid], j[valid], k[valid]] == 0
    return active


def _orient_normals_to_active_water_mask(
    centroids_m: np.ndarray,
    normals: np.ndarray,
    region_ids: np.ndarray,
    *,
    grid: CartesianGrid,
    obstacle_mask: np.ndarray,
    probe_distance_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    plus_active = _active_water_mask_for_points(
        centroids_m + normals * probe_distance_m,
        grid=grid,
        obstacle_mask=obstacle_mask,
    )
    minus_active = _active_water_mask_for_points(
        centroids_m - normals * probe_distance_m,
        grid=grid,
        obstacle_mask=obstacle_mask,
    )
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
        "method": "source_config_active_water_mask_probe_orientation",
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


def _area_weighted_normal_by_region(
    normals: np.ndarray,
    areas_m2: np.ndarray,
    region_ids: np.ndarray,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for region in sorted({int(value) for value in region_ids.tolist()}):
        mask = region_ids == region
        if not np.any(mask):
            continue
        weighted_normal = np.sum(normals[mask] * areas_m2[mask, None], axis=0)
        norm = float(np.linalg.norm(weighted_normal))
        if norm <= 1.0e-30:
            continue
        result[str(region)] = [
            float(value) for value in (weighted_normal / norm).tolist()
        ]
    return result


def build_tri_surface_diagnostics(
    config: dict[str, object],
    runtime: TaichiRuntimeConfig,
    *,
    spec: SquidReducedSpec | None = None,
    probe_distance_m: float | None = None,
    water_obstacle_mask: np.ndarray | None = None,
    water_grid: CartesianGrid | None = None,
    region_ids: tuple[int, ...] = (7, 8),
    solid_region_ids: tuple[int, ...] = (7, 8, 5),
) -> tuple[
    TriSurfaceRegionDiagnostics,
    dict[str, object],
    SurfaceMesh,
    np.ndarray,
    TriSurfaceRegionDiagnostics,
]:
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
    if (
        water_obstacle_mask is not None
        and water_grid is not None
        and probe_distance_m is not None
    ):
        normals, normal_orientation = _orient_normals_to_active_water_mask(
            centroids,
            normals,
            region_array,
            grid=water_grid,
            obstacle_mask=water_obstacle_mask,
            probe_distance_m=float(probe_distance_m),
        )
    elif spec is not None and probe_distance_m is not None:
        normals, normal_orientation = _orient_normals_to_reduced_water(
            centroids,
            normals,
            region_array,
            spec,
            float(probe_distance_m),
        )
    solid_faces, solid_centroids, solid_areas, solid_normals, solid_region_array, solid_region_face_counts = (
        build_region_subset(solid_region_ids, "solid MPM")
    )
    solid_normal_orientation: dict[str, object] = {
        "method": "mesh_face_winding",
        "probe_distance_m": None,
        "flipped_count": 0,
        "face_count": int(len(solid_region_array)),
    }
    if (
        water_obstacle_mask is not None
        and water_grid is not None
        and probe_distance_m is not None
    ):
        solid_normals, solid_normal_orientation = _orient_normals_to_active_water_mask(
            solid_centroids,
            solid_normals,
            solid_region_array,
            grid=water_grid,
            obstacle_mask=water_obstacle_mask,
            probe_distance_m=float(probe_distance_m),
        )
    elif spec is not None and probe_distance_m is not None:
        solid_normals, solid_normal_orientation = _orient_normals_to_reduced_water(
            solid_centroids,
            solid_normals,
            solid_region_array,
            spec,
            float(probe_distance_m),
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
    # S2-A11c: the solid subset (FSI regions + the fixed rim) gets its own
    # diagnostics object so the layered solid path can bind the rim
    # constraint (fixed_region_id) AND represent the rim as markers whose
    # velocity-Dirichlet rows seal the membrane-edge annulus.
    solid_diagnostics = TriSurfaceRegionDiagnostics(
        face_capacity=int(len(solid_areas)),
        runtime=runtime,
    )
    solid_diagnostics.load_faces(
        centroid_m=solid_centroids.astype(np.float32),
        normal=solid_normals.astype(np.float32),
        area_m2=solid_areas.astype(np.float32),
        region_id=solid_region_array,
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
        "diagnostic_area_weighted_normal_by_region": _area_weighted_normal_by_region(
            normals,
            areas,
            region_array,
        ),
        "solid_region_face_counts": solid_region_face_counts,
        "solid_area_m2_by_region": {
            str(region): float(np.sum(solid_areas[solid_region_array == region]))
            for region in solid_region_ids
        },
        "solid_area_weighted_normal_by_region": _area_weighted_normal_by_region(
            solid_normals,
            solid_areas,
            solid_region_array,
        ),
        "solid_surface_vertex_count": int(tri_surface_mesh.vertex_count),
        "solid_surface_face_count": int(tri_surface_mesh.face_count),
        "solid_surface_edge_note": "deduplicated from FSI triangles plus fixed rim triangles for TriMooneyShellMpmState",
        "centroid_bounds_min_m": [float(value) for value in np.min(centroids, axis=0)],
        "centroid_bounds_max_m": [float(value) for value in np.max(centroids, axis=0)],
        "solid_centroid_bounds_min_m": [float(value) for value in np.min(solid_centroids, axis=0)],
        "solid_centroid_bounds_max_m": [float(value) for value in np.max(solid_centroids, axis=0)],
        "normal_orientation": normal_orientation,
        "solid_normal_orientation": solid_normal_orientation,
    }
    return diagnostics, metadata, tri_surface_mesh, solid_region_array, solid_diagnostics


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


def solid_mpm_bounds_padding_distance_m(
    *,
    fluid_grid_axis_max_spacing_m: Sequence[float],
    estimated_solid_particle_spacing_m: float,
) -> float:
    fluid_axis_spacing = tuple(float(value) for value in fluid_grid_axis_max_spacing_m)
    if len(fluid_axis_spacing) != 3:
        raise ValueError("fluid_grid_axis_max_spacing_m must contain exactly 3 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in fluid_axis_spacing):
        raise ValueError("fluid_grid_axis_max_spacing_m entries must be finite and positive")
    solid_spacing = float(estimated_solid_particle_spacing_m)
    if not math.isfinite(solid_spacing) or solid_spacing <= 0.0:
        raise ValueError("estimated_solid_particle_spacing_m must be finite and positive")
    return 3.0 * max(max(fluid_axis_spacing), solid_spacing)


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


def resolve_pressure_solver(
    pressure_solver: str,
    *,
    graded_grid_enabled: bool,
    fsi_coupling_mode: str | None = None,
) -> str:
    solver_name = str(pressure_solver)
    if solver_name == "auto":
        if str(fsi_coupling_mode) == FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
            return "fv_cg"
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


# Windows MoveFileEx(REPLACE_EXISTING) fails with EACCES while ANY external
# process (a monitor, Excel, antivirus, an indexer) holds the destination
# open without FILE_SHARE_DELETE. 2026-06-13 incident: a monitoring reader
# killed the 4000-step production run at step 506 through exactly this
# window. Transient holders are absorbed by retrying; a persistent holder
# still raises after the budget (5 s) - never hang, never silently drop.


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
        "method": "latest_core_obstacle_flood_fill_from_z_min",
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
    fsi_coupling_adaptive_iterations_max_arg = int(
        args.fsi_coupling_adaptive_iterations_max
    )
    if fsi_coupling_adaptive_iterations_max_arg < 0:
        raise ValueError("--fsi-coupling-adaptive-iterations-max must be non-negative")
    if (
        fsi_coupling_adaptive_iterations_max_arg > 0
        and fsi_coupling_adaptive_iterations_max_arg < fsi_coupling_iterations
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-max must be 0 or at least "
            "--fsi-coupling-iterations"
        )
    fsi_coupling_adaptive_iterations_max = (
        fsi_coupling_adaptive_iterations_max_arg
        if fsi_coupling_adaptive_iterations_max_arg > 0
        else fsi_coupling_iterations
    )
    fsi_coupling_adaptive_iterations_residual_threshold_n = float(
        args.fsi_coupling_adaptive_iterations_residual_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_adaptive_iterations_residual_threshold_n)
        or (
            math.isfinite(fsi_coupling_adaptive_iterations_residual_threshold_n)
            and fsi_coupling_adaptive_iterations_residual_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-residual-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_adaptive_iterations_cfl_threshold = float(
        args.fsi_coupling_adaptive_iterations_cfl_threshold
    )
    if not (
        math.isinf(fsi_coupling_adaptive_iterations_cfl_threshold)
        or (
            math.isfinite(fsi_coupling_adaptive_iterations_cfl_threshold)
            and fsi_coupling_adaptive_iterations_cfl_threshold >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-cfl-threshold must be "
            "non-negative or infinity"
        )
    fsi_coupling_same_step_rerun_iterations_max_arg = int(
        args.fsi_coupling_same_step_rerun_iterations_max
    )
    if fsi_coupling_same_step_rerun_iterations_max_arg < 0:
        raise ValueError(
            "--fsi-coupling-same-step-rerun-iterations-max must be non-negative"
        )
    if (
        fsi_coupling_same_step_rerun_iterations_max_arg > 0
        and fsi_coupling_same_step_rerun_iterations_max_arg < fsi_coupling_iterations
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-iterations-max must be 0 or at least "
            "--fsi-coupling-iterations"
        )
    fsi_coupling_same_step_rerun_iterations_max = (
        fsi_coupling_same_step_rerun_iterations_max_arg
        if fsi_coupling_same_step_rerun_iterations_max_arg > 0
        else fsi_coupling_iterations
    )
    fsi_coupling_same_step_rerun_residual_threshold_n = float(
        args.fsi_coupling_same_step_rerun_residual_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_same_step_rerun_residual_threshold_n)
        or (
            math.isfinite(fsi_coupling_same_step_rerun_residual_threshold_n)
            and fsi_coupling_same_step_rerun_residual_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-residual-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_same_step_rerun_fluid_substep_factor = float(
        args.fsi_coupling_same_step_rerun_fluid_substep_factor
    )
    if (
        not math.isfinite(fsi_coupling_same_step_rerun_fluid_substep_factor)
        or fsi_coupling_same_step_rerun_fluid_substep_factor < 1.0
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-fluid-substep-factor must be "
            "finite and at least 1"
        )
    fsi_coupling_residual_continuation_iterations_max = int(
        args.fsi_coupling_residual_continuation_iterations_max
    )
    if fsi_coupling_residual_continuation_iterations_max < 0:
        raise ValueError(
            "--fsi-coupling-residual-continuation-iterations-max must be non-negative"
        )
    fsi_coupling_residual_continuation_threshold_n = float(
        args.fsi_coupling_residual_continuation_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_residual_continuation_threshold_n)
        or (
            math.isfinite(fsi_coupling_residual_continuation_threshold_n)
            and fsi_coupling_residual_continuation_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-continuation-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_residual_continuation_rebound_secant_from_best = bool(
        args.fsi_coupling_residual_continuation_rebound_secant_from_best
    )
    fsi_coupling_residual_continuation_rebound_secant_factor = float(
        args.fsi_coupling_residual_continuation_rebound_secant_factor
    )
    if not (
        math.isinf(fsi_coupling_residual_continuation_rebound_secant_factor)
        or (
            math.isfinite(fsi_coupling_residual_continuation_rebound_secant_factor)
            and fsi_coupling_residual_continuation_rebound_secant_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-continuation-rebound-secant-factor must "
            "be >= 1 or infinity"
        )
    fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max = int(
        args.fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
    )
    if fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max < 0:
        raise ValueError(
            "--fsi-coupling-residual-continuation-rebound-secant-evaluation-"
            "extensions-max must be non-negative"
        )
    fsi_coupling_trial_interior_divergence_tolerance = float(
        args.fsi_coupling_trial_interior_divergence_tolerance
    )
    if not (
        math.isinf(fsi_coupling_trial_interior_divergence_tolerance)
        or (
            math.isfinite(fsi_coupling_trial_interior_divergence_tolerance)
            and fsi_coupling_trial_interior_divergence_tolerance >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trial-interior-divergence-tolerance must be "
            "non-negative or infinity"
        )
    fsi_coupling_tolerance_n = float(args.fsi_coupling_tolerance_n)
    if not math.isfinite(fsi_coupling_tolerance_n) or fsi_coupling_tolerance_n < 0.0:
        raise ValueError("--fsi-coupling-tolerance-n must be a finite non-negative number")
    fsi_marker_coupling_tolerance_mps = float(args.fsi_marker_coupling_tolerance_mps)
    if (
        not math.isfinite(fsi_marker_coupling_tolerance_mps)
        or fsi_marker_coupling_tolerance_mps < 0.0
    ):
        raise ValueError(
            "--fsi-marker-coupling-tolerance-mps must be a finite non-negative number"
        )
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
    fsi_coupling_rejected_trial_backtrack = float(
        args.fsi_coupling_rejected_trial_backtrack
    )
    if (
        not math.isfinite(fsi_coupling_rejected_trial_backtrack)
        or not 0.0 < fsi_coupling_rejected_trial_backtrack <= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-rejected-trial-backtrack must be a finite number in (0, 1]"
        )
    fsi_coupling_residual_growth_rejection_factor = float(
        args.fsi_coupling_residual_growth_rejection_factor
    )
    if not (
        math.isinf(fsi_coupling_residual_growth_rejection_factor)
        or (
            math.isfinite(fsi_coupling_residual_growth_rejection_factor)
            and fsi_coupling_residual_growth_rejection_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-growth-rejection-factor must be >= 1 or infinity"
        )
    fsi_coupling_max_accepted_residual_n = float(
        args.fsi_coupling_max_accepted_residual_n
    )
    if not (
        math.isinf(fsi_coupling_max_accepted_residual_n)
        or (
            math.isfinite(fsi_coupling_max_accepted_residual_n)
            and fsi_coupling_max_accepted_residual_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-max-accepted-residual-n must be non-negative or infinity"
        )
    fsi_coupling_trust_region_force_increment_n = float(
        args.fsi_coupling_trust_region_force_increment_n
    )
    if not (
        math.isinf(fsi_coupling_trust_region_force_increment_n)
        or (
            math.isfinite(fsi_coupling_trust_region_force_increment_n)
            and fsi_coupling_trust_region_force_increment_n > 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-force-increment-n must be positive or infinity"
        )
    fsi_coupling_trust_region_adaptive = bool(
        args.fsi_coupling_trust_region_adaptive
    )
    fsi_coupling_trust_region_shrink_factor = float(
        args.fsi_coupling_trust_region_shrink_factor
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_shrink_factor)
        and 0.0 < fsi_coupling_trust_region_shrink_factor <= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-shrink-factor must be finite and in (0, 1]"
        )
    fsi_coupling_trust_region_growth_factor = float(
        args.fsi_coupling_trust_region_growth_factor
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_growth_factor)
        and fsi_coupling_trust_region_growth_factor >= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-growth-factor must be finite and >= 1"
        )
    fsi_coupling_trust_region_rebound_factor = float(
        args.fsi_coupling_trust_region_rebound_factor
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_factor)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_factor)
            and fsi_coupling_trust_region_rebound_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-factor must be >= 1 or infinity"
        )
    fsi_coupling_trust_region_rebound_backtrack = float(
        args.fsi_coupling_trust_region_rebound_backtrack
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_rebound_backtrack)
        and 0.0 < fsi_coupling_trust_region_rebound_backtrack < 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-backtrack must be finite and in (0, 1)"
        )
    fsi_coupling_trust_region_rebound_stop_factor = float(
        args.fsi_coupling_trust_region_rebound_stop_factor
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_stop_factor)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_stop_factor)
            and fsi_coupling_trust_region_rebound_stop_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-stop-factor must be >= 1 or infinity"
        )
    fsi_coupling_trust_region_rebound_stop_max_residual_n = float(
        args.fsi_coupling_trust_region_rebound_stop_max_residual_n
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_stop_max_residual_n)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_stop_max_residual_n)
            and fsi_coupling_trust_region_rebound_stop_max_residual_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-stop-max-residual-n must be "
            "non-negative or infinity"
        )
    if (
        fsi_coupling_trust_region_adaptive
        and math.isinf(fsi_coupling_trust_region_force_increment_n)
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-adaptive requires a finite "
            "--fsi-coupling-trust-region-force-increment-n"
        )
    fsi_stabilization_preset = str(args.fsi_stabilization_preset)
    fsi_stabilization_effective_parameters = (
        fsi_stabilization_effective_parameters_from_args(args)
    )
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
    raise_for_unsupported_hibm_mpm_sharp_iteration_options(
        fsi_coupling_mode=fsi_coupling_mode,
        fsi_coupling_iterations=fsi_coupling_iterations,
    )
    interface_reaction_aitken = bool(args.interface_reaction_aitken)
    interface_reaction_aitken_lower_bound = float(
        args.interface_reaction_aitken_lower_bound
    )
    if (
        not math.isfinite(interface_reaction_aitken_lower_bound)
        or not 0.0 <= interface_reaction_aitken_lower_bound <= 1.5
    ):
        raise ValueError(
            "--interface-reaction-aitken-lower-bound must be a finite number in [0, 1.5]"
        )
    interface_reaction_aitken_upper_bound = float(
        args.interface_reaction_aitken_upper_bound
    )
    if (
        not math.isfinite(interface_reaction_aitken_upper_bound)
        or not interface_reaction_aitken_lower_bound
        <= interface_reaction_aitken_upper_bound
        <= 1.5
    ):
        raise ValueError(
            "--interface-reaction-aitken-upper-bound must be finite and satisfy "
            "interface_reaction_aitken_lower_bound <= upper <= 1.5"
        )
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
    spec, pressure_schedule_input = spec_with_pressure_schedule_overrides(
        spec,
        {
            field: getattr(args, field, None)
            for field in PRESSURE_SCHEDULE_FIELDS
        },
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
    cad_step_arg = getattr(args, "cad_step_path", None)
    cad_step_path = None if cad_step_arg in (None, "") else Path(cad_step_arg).resolve()
    cad_provenance = source_config_cad_provenance_report(
        source_config,
        source_config_path=source_config_path,
        cad_step_path=cad_step_path,
    )
    real_cad_step_binding = bool(
        cad_provenance.get(
            "real_cad_step_binding",
            cad_provenance.get("direct_cad_step_binding", False),
        )
    )
    if bool(getattr(args, "require_real_cad_step", False)) and not real_cad_step_binding:
        raise ValueError(
            "source config must provide a verified real STEP CAD binding when "
            "--require-real-cad-step is set; cached STL files require matching "
            "source STEP and surface-cache hashes, and unrelated mesh paths are "
            "not accepted as the real CAD input"
        )
    pressure_boundary_mapping = source_config_pressure_boundary_shell_mapping(
        source_config,
    )
    pressure_load_source_region_id = pressure_boundary_mapping.source_region_id
    primary_shell_region_id = pressure_boundary_mapping.primary_shell_region_id
    secondary_shell_region_id = pressure_boundary_mapping.secondary_shell_region_id
    pressure_load_region_id = pressure_boundary_mapping.target_shell_region_id
    pressure_load_direction = _source_config_pressure_load_direction(source_config)
    region14_aperture_geometry = compute_region_geometry_stats(source_config, 14)
    source_config_fluid_active_mask_requested = (
        source_config_requests_fluid_active_mask(source_config)
    )
    source_config_reduced_water_intersection_requested = (
        source_config_requests_reduced_water_intersection(source_config)
        or bool(getattr(args, "source_config_intersect_reduced_water_domain", False))
    )
    source_config_region14_aperture_requested = (
        source_config_requests_region14_aperture_carve(source_config)
    )
    region14_aperture_carve_requested = (
        bool(args.use_region14_aperture_carve)
        or source_config_region14_aperture_requested
    )
    region14_aperture_geometry_available = bool(
        region14_aperture_geometry.get("available", False)
    )
    region14_aperture_carve_enabled = (
        region14_aperture_carve_requested
        and not bool(args.disable_region14_aperture_carve)
        and region14_aperture_geometry_available
    )
    if bool(args.disable_region14_aperture_carve):
        region14_aperture_carve_source = "disabled_by_cli"
    elif not region14_aperture_carve_requested:
        region14_aperture_carve_source = "not_requested"
    elif not region14_aperture_geometry_available:
        region14_aperture_carve_source = "requested_but_unavailable"
    elif bool(args.use_region14_aperture_carve) and source_config_region14_aperture_requested:
        region14_aperture_carve_source = "source_config_and_cli"
    elif source_config_region14_aperture_requested:
        region14_aperture_carve_source = "source_config"
    else:
        region14_aperture_carve_source = "cli"
    tail_refinement_geometry: dict[str, object] = {
        "available": False,
        "region_id": 8,
        "reason": "not_requested",
    }
    tail_refinement_region: RefinementRegion | None = None
    if region14_aperture_carve_enabled:
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
        fsi_coupling_mode=fsi_coupling_mode,
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
    solid_response_dt_s = float(spec.dt_s)
    fsi_solid_response_dt_s = solid_response_dt_s
    adaptive_fluid_substeps_enabled = bool(args.adaptive_fluid_substeps)
    fluid_substep_controller = (
        CflSubstepController(
            base_substeps=effective_fluid_substeps,
            target_cfl=float(args.adaptive_fluid_substeps_target_cfl),
            max_substeps=int(args.adaptive_fluid_substeps_max),
            growth_safety=float(args.adaptive_fluid_substeps_safety),
        )
        if adaptive_fluid_substeps_enabled
        else None
    )
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
            "shell_primary_region_id": int(primary_shell_region_id),
            "shell_secondary_region_id": int(secondary_shell_region_id),
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
            "fsi_stabilization_preset": fsi_stabilization_preset,
            "fsi_stabilization_preset_conflict_policy": (
                FSI_STABILIZATION_PRESET_CONFLICT_POLICY
            ),
            "fsi_stabilization_effective_parameters": (
                fsi_stabilization_effective_parameters
            ),
            "steps": step_count,
            "full_pressure_waveform_steps": full_pressure_waveform_steps,
            "steps_explicit": bool(getattr(args, "steps_explicit", True)),
            "membrane_thickness_scale": membrane_thickness_scale,
            "solid_density_scale": solid_density_scale,
            "solid_density_kgm3": float(material.density_kgm3),
            "solid_surface_mass_budget": solid_surface_mass_report,
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
            "fsi_coupling_iterations_base": fsi_coupling_iterations,
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
            "fsi_coupling_same_step_rerun_fluid_substep_factor": (
                fsi_coupling_same_step_rerun_fluid_substep_factor
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
            "fsi_coupling_trial_interior_divergence_tolerance": (
                fsi_coupling_trial_interior_divergence_tolerance
            ),
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
            "interface_reaction_aitken": interface_reaction_aitken,
            "interface_reaction_aitken_lower_bound": (
                interface_reaction_aitken_lower_bound
            ),
            "interface_reaction_aitken_upper_bound": (
                interface_reaction_aitken_upper_bound
            ),
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
            "source_config_fluid_active_mask_requested": (
                source_config_fluid_active_mask_requested
            ),
            "source_config_reduced_water_intersection_requested": (
                source_config_reduced_water_intersection_requested
            ),
            "source_config_region14_aperture_requested": (
                source_config_region14_aperture_requested
            ),
            "region14_aperture_carve_enabled": region14_aperture_carve_enabled,
            "region14_aperture_carve_source": region14_aperture_carve_source,
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
    initial_fluid_obstacle_mode = "disabled"
    source_config_fluid_topology_report: dict[str, object] = {
        "enabled": False,
        "reason": "not_requested",
    }
    source_config_water_obstacle_mask: np.ndarray | None = None
    if not args.disable_reduced_obstacles:
        if source_config_fluid_active_mask_requested:
            source_config_water_obstacle_mask, source_config_fluid_topology_report = (
                build_source_config_fluid_obstacle_mask(
                    config=source_config,
                    source_config_path=source_config_path,
                    grid=fluid_grid,
                    aperture_geometry=region14_aperture_geometry,
                    connect_surface_seeds_to_zmin=bool(
                        args.source_config_connect_surface_seeds_to_zmin
                    ),
                    surface_seed_zmin_connection_max_carve_cells=int(
                        args.source_config_surface_seed_zmin_connection_max_carve_cells
                    ),
                )
            )
            simulator.fluid.obstacle.from_numpy(source_config_water_obstacle_mask)
            surface_probe_clear_cells = tuple(
                source_config_fluid_topology_report.get(
                    "fluid_active_mask_surface_probe_clear_cells_ijk",
                    (),
                )
                or ()
            )
            if surface_probe_clear_cells:
                analysis_settings = source_config.get("analysis_settings", {})
                if not isinstance(analysis_settings, Mapping):
                    analysis_settings = {}
                protection_radius_cells = int(
                    analysis_settings.get(
                        "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells",
                        0,
                    )
                    or 0
                )
                solid_band_protection_mask = _solid_band_protection_mask_from_cells(
                    source_config_water_obstacle_mask.shape,
                    surface_probe_clear_cells,
                    radius_cells=protection_radius_cells,
                )
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells"
                ] = int(max(0, protection_radius_cells))
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_cell_count"
                ] = int(np.count_nonzero(solid_band_protection_mask))
                simulator.fluid.set_hibm_solid_band_protection_mask_from_numpy(
                    solid_band_protection_mask,
                )
            else:
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells"
                ] = 0
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_cell_count"
                ] = 0
            pre_intersection_obstacle_cell_count = int(
                source_config_fluid_topology_report.get("final_obstacle_cell_count", 0)
                or 0
            )
            total_fluid_cell_count = int(np.prod(tuple(int(value) for value in fluid_grid.grid_nodes)))
            if source_config_reduced_water_intersection_requested:
                simulator.intersect_current_obstacles_with_reduced_squid_water_domain()
                combined_obstacle_cell_count = simulator.fluid.obstacle_cell_count()
                source_config_water_obstacle_mask = simulator.fluid.obstacle.to_numpy()
                source_config_fluid_topology_report = {
                    **source_config_fluid_topology_report,
                    "source_config_active_mask_intersected_with_reduced_water_domain": True,
                    "pre_reduced_intersection_final_obstacle_cell_count": (
                        pre_intersection_obstacle_cell_count
                    ),
                    "pre_reduced_intersection_fluid_active_cell_count": (
                        total_fluid_cell_count - pre_intersection_obstacle_cell_count
                    ),
                    "reduced_water_intersection_added_obstacle_cell_count": max(
                        combined_obstacle_cell_count - pre_intersection_obstacle_cell_count,
                        0,
                    ),
                    "fluid_active_cell_count": total_fluid_cell_count
                    - combined_obstacle_cell_count,
                    "fluid_inactive_cell_count": combined_obstacle_cell_count,
                    "final_obstacle_cell_count": combined_obstacle_cell_count,
                    "host_device_transfer_policy": (
                        "one_time_initial_obstacle_upload_plus_combined_mask_snapshot"
                    ),
                }
                initial_fluid_obstacle_mode = (
                    "source_config_active_mask_intersected_reduced_analytic"
                )
            else:
                source_config_fluid_topology_report = {
                    **source_config_fluid_topology_report,
                    "source_config_active_mask_intersected_with_reduced_water_domain": False,
                    "pre_reduced_intersection_final_obstacle_cell_count": (
                        pre_intersection_obstacle_cell_count
                    ),
                    "pre_reduced_intersection_fluid_active_cell_count": (
                        total_fluid_cell_count - pre_intersection_obstacle_cell_count
                    ),
                    "reduced_water_intersection_added_obstacle_cell_count": 0,
                    "source_config_active_mask_intersection_policy": (
                        "cad_active_mask_authoritative"
                    ),
                }
                initial_fluid_obstacle_mode = "source_config_active_mask"
            simulator.fluid.snapshot_hibm_base_obstacle()
        else:
            simulator.mark_reduced_squid_water_domain()
            initial_fluid_obstacle_mode = "reduced_analytic"
            source_config_fluid_topology_report = {
                "enabled": False,
                "reason": "source_config_fluid_active_mask_not_requested",
            }
    elif source_config_fluid_active_mask_requested:
        source_config_fluid_topology_report = {
            "enabled": False,
            "reason": "disabled_by_disable_reduced_obstacles",
        }
    tri_surface_result = build_tri_surface_diagnostics(
        source_config,
        runtime,
        spec=spec,
        probe_distance_m=fluid_probe_distance_m,
        water_obstacle_mask=source_config_water_obstacle_mask,
        water_grid=fluid_grid if source_config_water_obstacle_mask is not None else None,
        region_ids=(primary_shell_region_id, secondary_shell_region_id),
        solid_region_ids=tuple(
            dict.fromkeys((primary_shell_region_id, secondary_shell_region_id, 5))
        ),
    )
    if len(tri_surface_result) == 5:
        (
            tri_diagnostics,
            tri_metadata,
            tri_surface_mesh,
            tri_surface_region_ids,
            solid_diagnostics,
        ) = tri_surface_result
    elif len(tri_surface_result) == 4:
        (
            tri_diagnostics,
            tri_metadata,
            tri_surface_mesh,
            tri_surface_region_ids,
        ) = tri_surface_result
        solid_diagnostics = tri_diagnostics
    else:
        raise ValueError(
            "build_tri_surface_diagnostics must return 4 or 5 result entries"
        )
    diagnostic_region_normals = tri_metadata.get(
        "diagnostic_area_weighted_normal_by_region",
        {},
    )
    if not isinstance(diagnostic_region_normals, Mapping):
        raise ValueError("tri surface diagnostics did not report region normals")
    pressure_closure_normal = diagnostic_region_normals.get(str(primary_shell_region_id))
    if pressure_closure_normal is None:
        raise ValueError(
            "tri surface diagnostics did not report a pressure closure normal "
            f"for region {primary_shell_region_id}"
        )
    pressure_closure_normal = _vector3(
        pressure_closure_normal,
        name="pressure_closure_normal",
    )
    pressure_far_side_normal_sign = far_pressure_side_normal_sign_from_direction(
        pressure_direction=pressure_load_direction,
        interface_normal=pressure_closure_normal,
    )
    pressure_outlet_boundary = (
        None
        if args.disable_pressure_outlet_zmin
        else AxisAlignedBoundary.pressure_outlet(axis="z", side="min")
    )
    pressure_outlet_zmin_enabled = (
        bool(pressure_outlet_boundary.legacy_zmin_outlet)
        if pressure_outlet_boundary is not None
        else False
    )
    pressure_outlet_boundary_report = (
        None
        if pressure_outlet_boundary is None
        else {
            **asdict(pressure_outlet_boundary),
            "selector": pressure_outlet_boundary.selector,
        }
    )
    total_fsi_face_area_m2 = (
        float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(primary_shell_region_id),
                0.0,
            )
        )
        + float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(secondary_shell_region_id),
                0.0,
            )
        )
    )
    primary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(primary_shell_region_id),
            0.0,
        )
    )
    secondary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(secondary_shell_region_id),
            0.0,
        )
    )
    total_solid_volume_m3 = (
        float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(primary_shell_region_id),
                0.0,
            )
        )
        * spec.main_membrane_thickness_m
        + float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(secondary_shell_region_id),
                0.0,
            )
        )
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
    solid_mpm_bounds_padding_m = solid_mpm_bounds_padding_distance_m(
        fluid_grid_axis_max_spacing_m=fluid_grid_axis_max_spacing_m,
        estimated_solid_particle_spacing_m=estimated_solid_particle_spacing_m,
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
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            fixed_region_id=5,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
            runtime=runtime,
        )
    elif args.solid_model == "neo_hookean_mpm":
        solid_mpm = NeoHookeanMpmState(
            particle_capacity=solid_diagnostics.face_count * args.solid_mpm_layers,
            bounds_min_m=solid_mpm_bounds_min_m,
            bounds_max_m=solid_mpm_bounds_max_m,
            grid_nodes=solid_mpm_grid_nodes,
            runtime=runtime,
        )
        solid_mpm.initialize_layered_tri_surface(
            solid_diagnostics,
            layer_count=args.solid_mpm_layers,
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            fixed_region_id=5,
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
                pressure_area_load_npm2 = tuple(
                    float(pressure_pa) * float(component)
                    for component in pressure_load_direction
                )
                report = solid_mpm.advance_region_loads(
                    dt_s=solid_sub_dt_s,
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_area_load_npm2=pressure_area_load_npm2,
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                    primary_area_load_region_id=pressure_load_region_id,
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
                pressure_area_load_npm2 = tuple(
                    float(pressure_pa) * float(component)
                    for component in pressure_load_direction
                )
                solid_mpm.set_layered_region_loads(
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_area_load_npm2=pressure_area_load_npm2,
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                )
                report = solid_mpm.step(
                    dt_s=solid_sub_dt_s,
                    mu_pa=material.shear_modulus_pa,
                    lambda_pa=material.lame_lambda_pa,
                    velocity_damping=solid_substep_velocity_damping,
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
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
        fluid_substeps: int | None = None,
        read_full_report: bool = True,
    ):
        step_substeps = (
            effective_fluid_substeps if fluid_substeps is None else int(fluid_substeps)
        )
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
                primary_region_id=primary_shell_region_id,
                secondary_region_id=secondary_shell_region_id,
                primary_velocity_mps=primary_velocity,
                secondary_velocity_mps=secondary_velocity,
                dt_s=spec.dt_s,
                fluid_substeps=step_substeps,
                ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
                projection_iterations=int(args.projection_iterations),
                divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                pressure_outlet_zmin=pressure_outlet_zmin_enabled,
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
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
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
            sharp_coupling_state=sharp_coupling_state,
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

    previous_step_cfl = None
    previous_step_fsi_coupling_residual_norm_n = None
    previous_step_fluid_substeps = effective_fluid_substeps
    if rows:
        try:
            previous_step_cfl = float(rows[-1]["cfl"])
        except (KeyError, TypeError, ValueError):
            previous_step_cfl = None
        try:
            previous_step_fsi_coupling_residual_norm_n = float(
                rows[-1]["fsi_coupling_residual_norm_n"]
            )
        except (KeyError, TypeError, ValueError):
            previous_step_fsi_coupling_residual_norm_n = None
        try:
            previous_step_fluid_substeps = max(
                effective_fluid_substeps,
                int(float(rows[-1].get("fluid_substeps", effective_fluid_substeps))),
            )
        except (TypeError, ValueError):
            previous_step_fluid_substeps = effective_fluid_substeps

    for step in range(first_step, step_count + 1):
        step_wall_started_at = time.perf_counter()
        step_fluid_substeps = effective_fluid_substeps
        if fluid_substep_controller is not None:
            step_fluid_substeps = fluid_substep_controller.substeps_for_next_step(
                previous_cfl=previous_step_cfl,
                previous_substeps=previous_step_fluid_substeps,
            )
        step_fluid_substep_dt_s = float(spec.dt_s) / float(step_fluid_substeps)
        step_fsi_coupling_iterations = fsi_coupling_iterations
        fsi_coupling_adaptive_iterations_residual_triggered = (
            previous_step_fsi_coupling_residual_norm_n is not None
            and math.isfinite(previous_step_fsi_coupling_residual_norm_n)
            and math.isfinite(
                fsi_coupling_adaptive_iterations_residual_threshold_n
            )
            and previous_step_fsi_coupling_residual_norm_n
            > fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        fsi_coupling_adaptive_iterations_cfl_triggered = (
            previous_step_cfl is not None
            and math.isfinite(previous_step_cfl)
            and math.isfinite(fsi_coupling_adaptive_iterations_cfl_threshold)
            and previous_step_cfl
            > fsi_coupling_adaptive_iterations_cfl_threshold
        )
        fsi_coupling_adaptive_iterations_triggered = (
            fsi_coupling_adaptive_iterations_max > fsi_coupling_iterations
            and (
                fsi_coupling_adaptive_iterations_residual_triggered
                or fsi_coupling_adaptive_iterations_cfl_triggered
            )
        )
        if fsi_coupling_adaptive_iterations_triggered:
            step_fsi_coupling_iterations = fsi_coupling_adaptive_iterations_max
        fsi_coupling_same_step_rerun_triggered = False
        fsi_coupling_same_step_rerun_count = 0
        fsi_coupling_same_step_rerun_initial_iterations_requested = (
            step_fsi_coupling_iterations
        )
        fsi_coupling_same_step_rerun_initial_iterations_used = 0
        fsi_coupling_same_step_rerun_initial_residual_norm_n = math.nan
        fsi_coupling_same_step_rerun_initial_converged = False
        fsi_coupling_same_step_rerun_safety_rejected = False
        fsi_coupling_same_step_rerun_initial_fluid_substeps = step_fluid_substeps
        fsi_coupling_same_step_rerun_final_fluid_substeps = step_fluid_substeps
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
        fsi_coupling_converged = step_fsi_coupling_iterations <= 1
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
        fsi_coupling_rejected_trial_count = 0
        fsi_coupling_rejected_trial_backtrack_count = 0
        fsi_coupling_residual_growth_rejected_trial_count = 0
        fsi_coupling_max_residual_rejected_trial_count = 0
        fsi_coupling_trial_cfl_rejected_count = 0
        fsi_coupling_trial_interior_divergence_rejected_count = 0
        fsi_coupling_trust_region_limited_update_count = 0
        fsi_coupling_trust_region_shrink_count = 0
        fsi_coupling_trust_region_growth_count = 0
        fsi_coupling_trust_region_rebound_backtrack_count = 0
        fsi_coupling_trust_region_rebound_stop_count = 0
        fsi_coupling_trust_region_rebound_stop_suppressed_count = 0
        fsi_coupling_residual_continuation_iteration_count = 0
        fsi_coupling_residual_continuation_rebound_secant_count = 0
        fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = 0
        fsi_coupling_trust_region_effective_force_increment_n = (
            fsi_coupling_trust_region_force_increment_n
        )
        fsi_coupling_accepted_trial_cfl = math.nan
        fsi_coupling_accepted_trial_max_fluid_speed_mps = math.nan
        fsi_coupling_accepted_trial_interior_divergence_l2 = math.nan
        fsi_coupling_trial_cfl_max = math.nan
        fsi_coupling_trial_interior_divergence_l2_max = math.nan
        fsi_coupling_trial_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fixed_point_result = None
        accepted_fsi_trial_payload: dict[str, object] | None = None
        accepted_fsi_trial_state_reused = False
        accepted_fsi_trial_state_readvanced = False
        fsi_all_trials_rejected = False
        fsi_zero_force_commit_blocked = False
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
            fsi_coupling_iterations=step_fsi_coupling_iterations,
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
            primary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
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
            primary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
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
                nonlocal fsi_coupling_trial_cfl_max
                nonlocal fsi_coupling_trial_interior_divergence_l2_max
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
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
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
                    fluid_substeps=step_fluid_substeps,
                    read_full_report=reuse_accepted_fsi_trial_state,
                )
                accumulate_fsi_trial_pressure_projection_stats(trial_fluid_report)
                trial_fluid_substep_dt_s = float(
                    getattr(
                        trial_fluid_report,
                        "fluid_substep_dt_s",
                        step_fluid_substep_dt_s,
                    )
                )
                trial_sample_report = simulator.sample_cfl_report(
                    dt_s=trial_fluid_substep_dt_s,
                )
                trial_cfl = float(trial_sample_report["cfl"])
                if math.isfinite(trial_cfl):
                    fsi_coupling_trial_cfl_max = (
                        trial_cfl
                        if not math.isfinite(fsi_coupling_trial_cfl_max)
                        else max(fsi_coupling_trial_cfl_max, trial_cfl)
                    )
                trial_interior_divergence_l2 = math.nan
                if math.isfinite(fsi_coupling_trial_interior_divergence_tolerance):
                    trial_projection_sample_report = simulator.sample_after_projection(
                        trial_fluid_report.divergence,
                        dt_s=trial_fluid_substep_dt_s,
                    )
                    trial_interior_divergence_l2 = float(
                        trial_projection_sample_report["interior_divergence_l2"]
                    )
                    if math.isfinite(trial_interior_divergence_l2):
                        fsi_coupling_trial_interior_divergence_l2_max = (
                            trial_interior_divergence_l2
                            if not math.isfinite(
                                fsi_coupling_trial_interior_divergence_l2_max
                            )
                            else max(
                                fsi_coupling_trial_interior_divergence_l2_max,
                                trial_interior_divergence_l2,
                            )
                        )
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
                        "trial_cfl": trial_cfl,
                        "trial_interior_divergence_l2": trial_interior_divergence_l2,
                        "trial_max_fluid_speed_mps": float(
                            trial_sample_report["max_fluid_speed_mps"]
                        ),
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

            def accept_fsi_interface_reaction_evaluation(
                evaluation: InterfaceReactionTargetEvaluation,
            ) -> bool:
                nonlocal fsi_coupling_trial_cfl_rejected_count
                nonlocal fsi_coupling_trial_interior_divergence_rejected_count
                payload = evaluation.payload
                if not isinstance(payload, Mapping):
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                rejection_reason = fsi_trial_acceptance_rejection_reason(
                    payload,
                    cfl_limit=0.5,
                    interior_divergence_l2_limit=(
                        fsi_coupling_trial_interior_divergence_tolerance
                    ),
                )
                if rejection_reason == "cfl":
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                if rejection_reason == "interior_divergence_l2":
                    fsi_coupling_trial_interior_divergence_rejected_count += 1
                    return False
                return True

            def apply_accepted_fsi_interface_reaction(reaction_force_n: tuple[float, ...]) -> None:
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(reaction_force_n)
                simulator.set_interface_reaction(
                    primary_force_n=primary_reaction_n,
                    secondary_force_n=secondary_reaction_n,
                )

            def commit_accepted_fsi_trial_state(payload: object | None) -> None:
                nonlocal accepted_fsi_trial_payload
                accepted_fsi_trial_payload = payload if isinstance(payload, dict) else None

            initial_fsi_reaction_force_n = _combine_region_pair_vectors(
                simulator.primary_interface_reaction_force_n[None],
                simulator.secondary_interface_reaction_force_n[None],
            )

            def solve_fsi_interface_reaction_attempt(
                iterations_requested: int,
            ) -> InterfaceReactionFixedPointResult:
                return solve_and_apply_interface_reaction_step(
                    initial_force_n=initial_fsi_reaction_force_n,
                    save_state=save_fsi_step_state,
                    evaluate_target=evaluate_fsi_interface_reaction_target,
                    restore_state=restore_fsi_trial_state,
                    apply_force=apply_accepted_fsi_interface_reaction,
                    commit_accepted_state=(
                        commit_accepted_fsi_trial_state
                        if reuse_accepted_fsi_trial_state
                        else None
                    ),
                    max_iterations=iterations_requested,
                    tolerance_n=fsi_coupling_tolerance_n,
                    initial_relaxation=interface_reaction_relaxation,
                    use_aitken=interface_reaction_aitken,
                    passivity_limit=interface_reaction_passivity_limit,
                    solver=fsi_coupling_solver,
                    target_map_relaxation=fsi_coupling_target_map_relaxation,
                    accept_evaluation=accept_fsi_interface_reaction_evaluation,
                    aitken_lower_bound=interface_reaction_aitken_lower_bound,
                    aitken_upper_bound=interface_reaction_aitken_upper_bound,
                    rejected_trial_backtrack=fsi_coupling_rejected_trial_backtrack,
                    residual_growth_rejection_factor=(
                        fsi_coupling_residual_growth_rejection_factor
                    ),
                    max_accepted_residual_n=fsi_coupling_max_accepted_residual_n,
                    trust_region_force_increment_n=(
                        fsi_coupling_trust_region_force_increment_n
                    ),
                    trust_region_adaptive=fsi_coupling_trust_region_adaptive,
                    trust_region_shrink_factor=(
                        fsi_coupling_trust_region_shrink_factor
                    ),
                    trust_region_growth_factor=(
                        fsi_coupling_trust_region_growth_factor
                    ),
                    trust_region_rebound_factor=(
                        fsi_coupling_trust_region_rebound_factor
                    ),
                    trust_region_rebound_backtrack=(
                        fsi_coupling_trust_region_rebound_backtrack
                    ),
                    trust_region_rebound_stop_factor=(
                        fsi_coupling_trust_region_rebound_stop_factor
                    ),
                    trust_region_rebound_stop_max_residual_n=(
                        fsi_coupling_trust_region_rebound_stop_max_residual_n
                    ),
                    residual_continuation_iterations_max=(
                        fsi_coupling_residual_continuation_iterations_max
                    ),
                    residual_continuation_threshold_n=(
                        fsi_coupling_residual_continuation_threshold_n
                    ),
                    residual_continuation_rebound_secant_from_best=(
                        fsi_coupling_residual_continuation_rebound_secant_from_best
                    ),
                    residual_continuation_rebound_secant_factor=(
                        fsi_coupling_residual_continuation_rebound_secant_factor
                    ),
                    residual_continuation_rebound_secant_evaluation_extensions_max=(
                        fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
                    ),
                )

            fixed_point_result = solve_fsi_interface_reaction_attempt(
                step_fsi_coupling_iterations
            )
            fsi_coupling_first_attempt_safety_rejected = (
                fixed_point_result.accepted_trial_index is None
                and fixed_point_result.rejected_trial_count > 0
            )
            fsi_coupling_same_step_rerun_next_fluid_substeps = (
                fsi_same_step_rerun_fluid_substeps(
                    current_substeps=step_fluid_substeps,
                    max_substeps=int(args.adaptive_fluid_substeps_max),
                    substep_factor=(
                        fsi_coupling_same_step_rerun_fluid_substep_factor
                    ),
                    safety_rejected=fsi_coupling_first_attempt_safety_rejected,
                )
            )
            fsi_coupling_same_step_iteration_rerun = fsi_same_step_rerun_triggered(
                current_iterations_requested=step_fsi_coupling_iterations,
                rerun_iterations_max=fsi_coupling_same_step_rerun_iterations_max,
                residual_norm_n=fixed_point_result.residual_norm_n,
                residual_threshold_n=(
                    fsi_coupling_same_step_rerun_residual_threshold_n
                ),
                converged=fixed_point_result.converged,
                safety_rejected=fsi_coupling_first_attempt_safety_rejected,
            )
            fsi_coupling_same_step_fluid_substep_rerun = (
                fsi_coupling_same_step_rerun_next_fluid_substeps
                > step_fluid_substeps
            )
            if (
                fsi_coupling_same_step_iteration_rerun
                or fsi_coupling_same_step_fluid_substep_rerun
            ):
                fsi_coupling_same_step_rerun_triggered = True
                fsi_coupling_same_step_rerun_count = 1
                fsi_coupling_same_step_rerun_initial_iterations_requested = (
                    step_fsi_coupling_iterations
                )
                fsi_coupling_same_step_rerun_initial_iterations_used = (
                    fixed_point_result.iterations_used
                )
                fsi_coupling_same_step_rerun_initial_residual_norm_n = (
                    fixed_point_result.residual_norm_n
                )
                fsi_coupling_same_step_rerun_initial_converged = (
                    fixed_point_result.converged
                )
                fsi_coupling_same_step_rerun_safety_rejected = (
                    fsi_coupling_first_attempt_safety_rejected
                )
                fsi_coupling_same_step_rerun_initial_fluid_substeps = (
                    step_fluid_substeps
                )
                restore_fsi_trial_state()
                accepted_fsi_trial_payload = None
                if fsi_coupling_same_step_iteration_rerun:
                    step_fsi_coupling_iterations = (
                        fsi_coupling_same_step_rerun_iterations_max
                    )
                if fsi_coupling_same_step_fluid_substep_rerun:
                    step_fluid_substeps = (
                        fsi_coupling_same_step_rerun_next_fluid_substeps
                    )
                    step_fluid_substep_dt_s = float(spec.dt_s) / float(
                        step_fluid_substeps
                    )
                fsi_coupling_same_step_rerun_final_fluid_substeps = (
                    step_fluid_substeps
                )
                fixed_point_result = solve_fsi_interface_reaction_attempt(
                    step_fsi_coupling_iterations
                )
            fsi_coupling_iterations_used = fixed_point_result.iterations_used
            fsi_coupling_converged = fixed_point_result.converged
            fsi_coupling_residual_norm_n = fixed_point_result.residual_norm_n
            fsi_coupling_relaxation_effective = fixed_point_result.relaxation
            fsi_all_trials_rejected = fixed_point_result.all_trials_rejected
            fsi_zero_force_commit_blocked = fixed_point_result.zero_force_commit_blocked
            fsi_coupling_rejected_trial_count = fixed_point_result.rejected_trial_count
            fsi_coupling_rejected_trial_backtrack_count = (
                fixed_point_result.rejected_trial_backtrack_count
            )
            fsi_coupling_residual_growth_rejected_trial_count = (
                fixed_point_result.residual_growth_rejected_trial_count
            )
            fsi_coupling_max_residual_rejected_trial_count = (
                fixed_point_result.max_residual_rejected_trial_count
            )
            fsi_coupling_trust_region_limited_update_count = (
                fixed_point_result.trust_region_limited_update_count
            )
            fsi_coupling_trust_region_shrink_count = (
                fixed_point_result.trust_region_shrink_count
            )
            fsi_coupling_trust_region_growth_count = (
                fixed_point_result.trust_region_growth_count
            )
            fsi_coupling_trust_region_rebound_backtrack_count = (
                fixed_point_result.trust_region_rebound_backtrack_count
            )
            fsi_coupling_trust_region_rebound_stop_count = (
                fixed_point_result.trust_region_rebound_stop_count
            )
            fsi_coupling_trust_region_rebound_stop_suppressed_count = (
                fixed_point_result.trust_region_rebound_stop_suppressed_count
            )
            fsi_coupling_residual_continuation_iteration_count = (
                fixed_point_result.residual_continuation_iteration_count
            )
            fsi_coupling_residual_continuation_rebound_secant_count = (
                fixed_point_result.residual_continuation_rebound_secant_count
            )
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = (
                fixed_point_result.residual_continuation_rebound_secant_evaluation_extension_count
            )
            fsi_coupling_trust_region_effective_force_increment_n = (
                fixed_point_result.trust_region_effective_force_increment_n
            )
            if isinstance(fixed_point_result.accepted_payload, Mapping):
                fsi_coupling_accepted_trial_cfl = float(
                    fixed_point_result.accepted_payload.get("trial_cfl", math.nan)
                )
                fsi_coupling_accepted_trial_max_fluid_speed_mps = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_max_fluid_speed_mps",
                        math.nan,
                    )
                )
                fsi_coupling_accepted_trial_interior_divergence_l2 = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_interior_divergence_l2",
                        math.nan,
                    )
                )
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
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
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
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                            read_report=False,
                        )
                    else:
                        raise ValueError(f"Unsupported solid model: {args.solid_model}")
                solid_advance_wall_time_s += (
                    time.perf_counter() - solid_wall_started_at
                )
                return solid_mpm.report() if report is None else report

            def advance_sharp_trial_once():
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
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    far_pressure_region_id=pressure_load_region_id,
                    far_pressure_barrier_region_id=5,
                    far_pressure_pa=pressure_pa,
                    far_pressure_side_normal_sign=pressure_far_side_normal_sign,
                    far_pressure_inside_probe_max_multiplier=12.0,
                    two_sided_probe_max_multiplier=12.0,
                    one_sided_pressure_region_id=secondary_shell_region_id,
                    one_sided_reference_pressure_pa=0.0,
                    one_sided_probe_max_multiplier=12.0,
                    far_pressure_air_backed=True,
                    far_pressure_air_backed_probe_normal_sign=pressure_far_side_normal_sign,
                    fluid_dt_s=spec.dt_s,
                    fluid_substeps=step_fluid_substeps,
                    projection_iterations=int(args.projection_iterations),
                    run_fluid_predictor=True,
                    pressure_neumann_density_kgm3=spec.water_density_kgm3,
                    pressure_neumann_dt_s=spec.dt_s,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                    pressure_solver=pressure_solver_name,
                    pressure_solve_failure_policy=str(args.pressure_solve_failure_policy),
                    fluid_advection_scheme=str(args.fluid_advection_scheme),
                    multigrid_cycles=effective_multigrid_cycles,
                    cg_tolerance=cg_tolerance,
                    cg_preconditioner=cg_preconditioner,
                    divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                    divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                    convert_internal_nodes_to_obstacles=False,
                    post_dirichlet_consistency_projection_iterations=int(
                        args.hibm_post_dirichlet_consistency_projections
                    ),
                    diagnostic_disable_pressure_neumann_matrix_rows=bool(
                        args.diagnostic_disable_pressure_neumann_matrix_rows
                    ),
                )
                return sharp_report

            def restore_sharp_trial_state(
                marker_state: Mapping[str, object],
                pressure_gradient_state: object,
            ) -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()
                restore_sharp_marker_state_arrays(
                    sharp_coupling_state.markers,
                    marker_state,
                )
                restore_sharp_pressure_neumann_gradient_state_array(
                    sharp_coupling_state,
                    pressure_gradient_state,
                )

            def advance_sharp_marker_fixed_point_step():
                nonlocal fsi_coupling_iterations_used
                nonlocal fsi_coupling_converged
                nonlocal fsi_coupling_residual_norm_n
                nonlocal fsi_coupling_relaxation_effective
                nonlocal fsi_coupling_iqn_ils_least_squares_update_count
                nonlocal fsi_coupling_physical_interface_map_amplification
                nonlocal fsi_coupling_physical_interface_map_amplification_sample_count

                requested_iterations = max(1, int(fsi_coupling_iterations))
                if requested_iterations <= 1:
                    report = advance_sharp_trial_once()
                    fsi_coupling_iterations_used = 1
                    fsi_coupling_converged = False
                    fsi_coupling_residual_norm_n = math.nan
                    return report, {
                        "hibm_coupling_scheme": "explicit_loose",
                        "hibm_added_mass_stability_status": (
                            "unmeasured_single_pass"
                        ),
                        "hibm_added_mass_stability_measured": False,
                        "hibm_added_mass_stabilization": "none",
                        "hibm_semi_implicit_coupling_enabled": False,
                        "hibm_semi_implicit_coupling_matrix_active": False,
                        "hibm_fsi_coupling_iterations_used": 1,
                        "hibm_fsi_coupling_converged": False,
                        "hibm_fsi_coupling_explicit_single_pass": True,
                        "hibm_fsi_coupling_residual_source": (
                            "unmeasured_single_pass"
                        ),
                    }

                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()
                marker_guess = sharp_marker_state_arrays(sharp_coupling_state.markers)
                pressure_gradient_state = (
                    sharp_pressure_neumann_gradient_state_array(sharp_coupling_state)
                )
                previous_velocity_residual_vector: np.ndarray | None = None
                residual_history: list[float] = []
                residual_max_history: list[float] = []
                combined_residual_history: list[float] = []
                combined_residual_max_history: list[float] = []
                residual_position_history: list[float] = []
                residual_velocity_history: list[float] = []
                residual_primary_region_history: list[float] = []
                residual_secondary_region_history: list[float] = []
                residual_other_region_history: list[float] = []
                residual_max_marker_index_history: list[int] = []
                residual_max_marker_region_history: list[int] = []
                relaxation_history: list[float] = []
                relaxation = float(interface_reaction_relaxation)
                converged = False
                iterations_used = 0
                aitken_update_count = 0
                report = None
                residual_norm_mps = math.inf
                residual_max_mps = math.inf
                combined_residual_norm_mps = math.inf
                combined_residual_max_mps = math.inf

                for iteration in range(requested_iterations):
                    restore_sharp_trial_state(marker_guess, pressure_gradient_state)
                    report = advance_sharp_trial_once()
                    marker_candidate = sharp_marker_state_arrays(
                        sharp_coupling_state.markers
                    )
                    candidate_pressure_gradient_state = (
                        sharp_pressure_neumann_gradient_state_array(
                            sharp_coupling_state
                        )
                    )
                    residual = sharp_marker_fixed_point_residual_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    marker_region_ids = (
                        sharp_coupling_state.markers.region_id.to_numpy()
                        [: int(sharp_coupling_state.markers.marker_count)]
                    )
                    residual_diagnostics = (
                        sharp_marker_fixed_point_residual_diagnostics_mps(
                            marker_guess,
                            marker_candidate,
                            dt_s=spec.dt_s,
                            marker_region_ids=marker_region_ids,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                        )
                    )
                    residual_vector = _sharp_marker_fixed_point_residual_vector_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    velocity_residual_vector = residual_vector[:, 3:].reshape(-1)
                    combined_residual_norm_mps = float(residual["l2_mps"])
                    combined_residual_max_mps = float(residual["max_mps"])
                    residual_norm_mps = float(residual_diagnostics["velocity_l2_mps"])
                    residual_max_mps = float(residual_diagnostics["velocity_max_mps"])
                    residual_history.append(residual_norm_mps)
                    residual_max_history.append(residual_max_mps)
                    combined_residual_history.append(combined_residual_norm_mps)
                    combined_residual_max_history.append(combined_residual_max_mps)
                    residual_position_history.append(
                        float(residual_diagnostics["position_l2_mps"])
                    )
                    residual_velocity_history.append(residual_norm_mps)
                    residual_primary_region_history.append(
                        float(residual_diagnostics["primary_region_l2_mps"])
                    )
                    residual_secondary_region_history.append(
                        float(residual_diagnostics["secondary_region_l2_mps"])
                    )
                    residual_other_region_history.append(
                        float(residual_diagnostics["other_region_l2_mps"])
                    )
                    residual_max_marker_index_history.append(
                        int(residual_diagnostics["max_marker_index"])
                    )
                    residual_max_marker_region_history.append(
                        int(residual_diagnostics["max_marker_region_id"])
                    )
                    relaxation_history.append(float(relaxation))
                    iterations_used = iteration + 1
                    velocity_residual_norm_mps = residual_norm_mps
                    trial_projection_failure_reason = (
                        sharp_report_fluid_projection_failure_reason(report)
                    )
                    if trial_projection_failure_reason:
                        raise RuntimeError(
                            "sharp marker fixed point trial fluid projection failed "
                            f"(iteration={int(iterations_used)}, "
                            f"reason={trial_projection_failure_reason}, "
                            f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                            f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                            f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                            f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                            f"residual_history_mps={residual_history}, "
                            f"residual_max_history_mps={residual_max_history}, "
                            f"combined_residual_history_mps={combined_residual_history}, "
                            f"combined_residual_max_history_mps={combined_residual_max_history}, "
                            f"relaxation_history={relaxation_history})"
                        )
                    if velocity_residual_norm_mps <= fsi_marker_coupling_tolerance_mps:
                        converged = True
                        break
                    if iteration == requested_iterations - 1:
                        break
                    if (
                        interface_reaction_aitken
                        and previous_velocity_residual_vector is not None
                    ):
                        relaxation = _sharp_marker_aitken_relaxation(
                            previous_relaxation=relaxation,
                            previous_residual_mps=previous_velocity_residual_vector,
                            current_residual_mps=velocity_residual_vector,
                        )
                        aitken_update_count += 1
                    previous_velocity_residual_vector = (
                        velocity_residual_vector.copy()
                    )
                    marker_guess = relaxed_sharp_marker_state_arrays(
                        marker_guess,
                        marker_candidate,
                        relaxation=relaxation,
                    )
                    pressure_gradient_state = (
                        relaxed_sharp_pressure_neumann_gradient_state_array(
                            pressure_gradient_state,
                            candidate_pressure_gradient_state,
                            relaxation=relaxation,
                        )
                    )

                if report is None:
                    raise RuntimeError("sharp marker fixed point produced no trial")
                if not converged:
                    raise RuntimeError(
                        "sharp marker fixed point did not converge "
                        f"(iterations={int(iterations_used)}, "
                        f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                        f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                        f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                        f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                        f"tolerance_mps={float(fsi_marker_coupling_tolerance_mps):.6g}, "
                        f"residual_history_mps={residual_history}, "
                        f"residual_max_history_mps={residual_max_history}, "
                        f"combined_residual_history_mps={combined_residual_history}, "
                        f"combined_residual_max_history_mps={combined_residual_max_history}, "
                        f"position_residual_history_mps={residual_position_history}, "
                        f"velocity_residual_history_mps={residual_velocity_history}, "
                        f"primary_region_residual_history_mps={residual_primary_region_history}, "
                        f"secondary_region_residual_history_mps={residual_secondary_region_history}, "
                        f"other_region_residual_history_mps={residual_other_region_history}, "
                        f"max_marker_index_history={residual_max_marker_index_history}, "
                        f"max_marker_region_history={residual_max_marker_region_history}, "
                        f"relaxation_history={relaxation_history})"
                    )
                fsi_coupling_iterations_used = iterations_used
                fsi_coupling_converged = converged
                fsi_coupling_residual_norm_n = math.nan
                fsi_coupling_relaxation_effective = relaxation
                fsi_coupling_iqn_ils_least_squares_update_count = aitken_update_count
                if len(residual_history) >= 2 and residual_history[0] > 0.0:
                    amplification = residual_history[-1] / residual_history[0]
                    fsi_coupling_physical_interface_map_amplification = amplification
                    fsi_coupling_physical_interface_map_amplification_sample_count = (
                        len(residual_history) - 1
                    )
                summary = {
                    "hibm_coupling_scheme": "marker_fixed_point",
                    "hibm_added_mass_stability_status": (
                        "converged" if converged else "not_converged"
                    ),
                    "hibm_added_mass_stability_measured": True,
                    "hibm_added_mass_stabilization": (
                        "aitken_marker_state_under_relaxation"
                        if interface_reaction_aitken
                        else "marker_state_under_relaxation"
                    ),
                    "hibm_semi_implicit_coupling_enabled": True,
                    "hibm_semi_implicit_coupling_matrix_active": False,
                    "hibm_fsi_coupling_iterations_used": iterations_used,
                    "hibm_fsi_coupling_converged": converged,
                    "hibm_fsi_coupling_explicit_single_pass": False,
                    "hibm_fsi_coupling_residual_source": (
                        "marker_surface_fixed_point_velocity_residual_l2_mps"
                    ),
                    "hibm_fsi_coupling_residual_l2_mps": residual_norm_mps,
                    "hibm_fsi_coupling_residual_max_mps": residual_max_mps,
                    "hibm_fsi_coupling_residual_history_mps": residual_history,
                    "hibm_fsi_coupling_residual_max_history_mps": residual_max_history,
                    "hibm_fsi_coupling_combined_residual_l2_mps": (
                        combined_residual_norm_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_max_mps": (
                        combined_residual_max_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_history_mps": (
                        combined_residual_history
                    ),
                    "hibm_fsi_coupling_combined_residual_max_history_mps": (
                        combined_residual_max_history
                    ),
                    "hibm_fsi_coupling_position_residual_history_mps": (
                        residual_position_history
                    ),
                    "hibm_fsi_coupling_velocity_residual_history_mps": (
                        residual_velocity_history
                    ),
                    "hibm_fsi_coupling_primary_region_residual_history_mps": (
                        residual_primary_region_history
                    ),
                    "hibm_fsi_coupling_secondary_region_residual_history_mps": (
                        residual_secondary_region_history
                    ),
                    "hibm_fsi_coupling_other_region_residual_history_mps": (
                        residual_other_region_history
                    ),
                    "hibm_fsi_coupling_max_marker_index_history": (
                        residual_max_marker_index_history
                    ),
                    "hibm_fsi_coupling_max_marker_region_history": (
                        residual_max_marker_region_history
                    ),
                    "hibm_fsi_coupling_relaxation_effective": relaxation,
                    "hibm_fsi_coupling_relaxation_history": relaxation_history,
                    "hibm_fsi_coupling_aitken_update_count": aitken_update_count,
                }
                return report, summary

            fluid_wall_started_at = time.perf_counter()
            try:
                sharp_report, sharp_fixed_point_summary = (
                    advance_sharp_marker_fixed_point_step()
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            sharp_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
            if fsi_coupling_iterations > 1:
                fsi_coupling_wall_time_s = sharp_advance_wall_time_s
            fluid_advance_wall_time_s = max(
                0.0,
                sharp_advance_wall_time_s - solid_advance_wall_time_s,
            )
            solid_mpm_report = sharp_report.mpm
            if solid_mpm_report is None:
                solid_mpm_report = solid_mpm.report()
            publish_solid_report_to_reduced_state(current_time_s, solid_mpm_report)
            sample_wall_started_at = time.perf_counter()
            fluid_substep_dt_s = step_fluid_substep_dt_s
            latest_fluid_projection_report = (
                sharp_report.post_solid_fluid_projection
                if sharp_report.post_solid_fluid_projection is not None
                else sharp_report.fluid_to_mpm_loads.fluid_projection
            )
            sample_report = simulator.sample_after_projection(
                latest_fluid_projection_report,
                dt_s=fluid_substep_dt_s,
            )
            pressure_outlet_report = simulator.fluid.pressure_outlet_fv_flux_report(
                dt_s=fluid_substep_dt_s,
            )
            sample_wall_time_s = time.perf_counter() - sample_wall_started_at
            sharp_summary = hibm_mpm_sharp_step_summary(sharp_report)
            sharp_summary.update(sharp_fixed_point_summary)
            row = build_hibm_mpm_sharp_case_row(
                step=step,
                sample_report=sample_report,
                sharp_summary=sharp_summary,
                fluid_projection_report=sharp_report.fluid_to_mpm_loads.fluid_projection,
                pressure_outlet_report=pressure_outlet_report,
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
            row["accepted_fsi_trial_state_readvanced"] = False
            row["fsi_all_trials_rejected"] = False
            row["fsi_zero_force_commit_blocked"] = False
            row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
            row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
                fsi_coupling_iqn_ils_least_squares_update_count
            )
            row["fsi_coupling_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_physical_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_physical_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_raw_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_raw_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_residual_jacobian_amplification_sample_count"] = 0
            row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = 0
            row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = 0
            row["interface_reaction_relaxation"] = interface_reaction_relaxation
            row["interface_reaction_aitken"] = interface_reaction_aitken
            row["interface_reaction_aitken_lower_bound"] = (
                interface_reaction_aitken_lower_bound
            )
            row["interface_reaction_aitken_upper_bound"] = (
                interface_reaction_aitken_upper_bound
            )
            row["interface_reaction_relaxation_effective"] = (
                fsi_coupling_relaxation_effective
            )
            row["interface_reaction_passivity_limit"] = interface_reaction_passivity_limit
            row["interface_reaction_robin_impedance_ns_m"] = (
                interface_reaction_robin_impedance_ns_m
            )
            row["interface_reaction_robin_matrix_impedance_ns_m"] = (
                interface_reaction_robin_matrix_impedance_ns_m
            )
            row["interface_reaction_robin_target_mode"] = (
                interface_reaction_robin_target_mode
            )
            row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
            row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
            row["sample_wall_time_s"] = sample_wall_time_s
            row["surface_diagnostics_wall_time_s"] = 0.0
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            row["fluid_substeps_base"] = effective_fluid_substeps
            row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
            row["adaptive_fluid_substeps_target_cfl"] = float(
                args.adaptive_fluid_substeps_target_cfl
            )
            row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
            row["adaptive_fluid_substeps_previous_substeps"] = (
                previous_step_fluid_substeps
            )
            rows.append(row)
            if args.diagnostic_dump_zero_correctable_cells:
                zero_correctable_summary = _write_hibm_zero_correctable_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_zero_correctable_interior_cell_count"] = int(
                    zero_correctable_summary["zero_correctable_interior_cell_count"]
                )
                row["diagnostic_zero_correctable_shell_band_candidate_count"] = int(
                    zero_correctable_summary["shell_band_candidate_cell_count"]
                )
            if args.diagnostic_dump_high_residual_cells:
                high_residual_summary = _write_hibm_high_residual_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_high_residual_dumped_cell_count"] = int(
                    high_residual_summary["dumped_cell_count"]
                )
                row["diagnostic_high_residual_max_abs_s"] = float(
                    high_residual_summary["max_abs_residual_s"]
                )
                row["diagnostic_high_residual_velocity_dirichlet_cell_count"] = int(
                    high_residual_summary["dumped_velocity_dirichlet_cell_count"]
                )
            if args.diagnostic_dump_pressure_neumann_invalid_rows:
                load_pressure_neumann_invalid_summary = (
                    _write_hibm_pressure_neumann_invalid_row_dump(
                        output_dir=output_dir,
                        step=step,
                        rows=(
                            sharp_report.fluid_to_mpm_loads
                            .pressure_neumann_invalid_diagnostic_rows
                        ),
                        stage="load",
                    )
                )
                next_pressure_neumann_invalid_summary = (
                    _write_hibm_pressure_neumann_invalid_row_dump(
                        output_dir=output_dir,
                        step=step,
                        rows=(
                            sharp_report
                            .next_pressure_neumann_invalid_diagnostic_rows
                        ),
                        stage="next",
                    )
                )
                row["diagnostic_pressure_neumann_invalid_load_dumped_row_count"] = int(
                    load_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_load_total_row_count"] = int(
                    load_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
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
                _raise_for_step_solid_out_of_bounds_guard(row)
                _raise_for_closure_coverage_floor(
                    rows,
                    int(args.closure_coverage_floor),
                    int(args.closure_coverage_floor_patience),
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            previous_step_cfl = float(row["cfl"])
            previous_step_fluid_substeps = int(
                float(row.get("fluid_substeps", step_fluid_substeps))
            )
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
                    sharp_coupling_state=sharp_coupling_state,
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
            accepted_fsi_trial_state_readvanced = (
                fixed_point_result is not None
                and fixed_point_result.accepted_trial_index is not None
            )
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
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
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
                fluid_substeps=step_fluid_substeps,
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
        row["pressure_outlet_positive_source_volume_flux_m3s"] = pressure_outlet_report[
            "positive_source_volume_flux_m3s"
        ]
        row["pressure_outlet_abs_source_volume_flux_m3s"] = pressure_outlet_report[
            "abs_source_volume_flux_m3s"
        ]
        row["pressure_outlet_reachable_source_volume_flux_m3s"] = pressure_outlet_report[
            "zmin_reachable_source_volume_flux_m3s"
        ]
        row["pressure_outlet_unreached_source_volume_flux_m3s"] = pressure_outlet_report[
            "zmin_unreached_source_volume_flux_m3s"
        ]
        row["pressure_outlet_reachability_valid"] = bool(
            pressure_outlet_report.get("zmin_reachability_valid", False)
        )
        row["pressure_outlet_reachability_revision"] = pressure_outlet_report[
            "zmin_reachability_revision"
        ]
        row["pressure_outlet_reachable_source_cell_count"] = pressure_outlet_report[
            "zmin_reachable_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_cell_count"] = pressure_outlet_report[
            "zmin_unreached_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_abs_flux_m3s"] = pressure_outlet_report[
            "zmin_unreached_source_abs_flux_m3s"
        ]
        row["pressure_outlet_unreached_source_centroid_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_x_m"
        ]
        row["pressure_outlet_unreached_source_centroid_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_y_m"
        ]
        row["pressure_outlet_unreached_source_centroid_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_z_m"
        ]
        row["pressure_outlet_unreached_source_min_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_x_m"
        ]
        row["pressure_outlet_unreached_source_min_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_y_m"
        ]
        row["pressure_outlet_unreached_source_min_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_z_m"
        ]
        row["pressure_outlet_unreached_source_max_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_x_m"
        ]
        row["pressure_outlet_unreached_source_max_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_y_m"
        ]
        row["pressure_outlet_unreached_source_max_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_z_m"
        ]
        row["pressure_outlet_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_velocity_to_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_source_ratio"
        ]
        row["pressure_outlet_velocity_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_velocity_to_positive_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_positive_source_ratio"
        ]
        row["pressure_outlet_velocity_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_abs_source_ratio"
        ]
        row["pressure_outlet_pressure_flux_m3s"] = pressure_outlet_report[
            "zmin_pressure_outlet_flux_m3s"
        ]
        row["pressure_outlet_pressure_to_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_source_ratio"
        ]
        row["pressure_outlet_pressure_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_pressure_to_positive_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_positive_source_ratio"
        ]
        row["pressure_outlet_pressure_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_abs_source_ratio"
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
        row["fsi_coupling_iterations_requested"] = step_fsi_coupling_iterations
        row["fsi_coupling_iterations_base"] = fsi_coupling_iterations
        row["fsi_coupling_adaptive_iterations_max"] = (
            fsi_coupling_adaptive_iterations_max
        )
        row["fsi_coupling_adaptive_iterations_residual_threshold_n"] = (
            fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        row["fsi_coupling_adaptive_iterations_cfl_threshold"] = (
            fsi_coupling_adaptive_iterations_cfl_threshold
        )
        row["fsi_coupling_adaptive_iterations_triggered"] = (
            fsi_coupling_adaptive_iterations_triggered
        )
        row["fsi_coupling_adaptive_iterations_residual_triggered"] = (
            fsi_coupling_adaptive_iterations_residual_triggered
        )
        row["fsi_coupling_adaptive_iterations_cfl_triggered"] = (
            fsi_coupling_adaptive_iterations_cfl_triggered
        )
        row["fsi_coupling_same_step_rerun_iterations_max"] = (
            fsi_coupling_same_step_rerun_iterations_max
        )
        row["fsi_coupling_same_step_rerun_residual_threshold_n"] = (
            fsi_coupling_same_step_rerun_residual_threshold_n
        )
        row["fsi_coupling_same_step_rerun_triggered"] = (
            fsi_coupling_same_step_rerun_triggered
        )
        row["fsi_coupling_same_step_rerun_count"] = (
            fsi_coupling_same_step_rerun_count
        )
        row["fsi_coupling_same_step_rerun_initial_iterations_requested"] = (
            fsi_coupling_same_step_rerun_initial_iterations_requested
        )
        row["fsi_coupling_same_step_rerun_initial_iterations_used"] = (
            fsi_coupling_same_step_rerun_initial_iterations_used
        )
        row["fsi_coupling_same_step_rerun_initial_residual_norm_n"] = (
            fsi_coupling_same_step_rerun_initial_residual_norm_n
        )
        row["fsi_coupling_same_step_rerun_initial_converged"] = (
            fsi_coupling_same_step_rerun_initial_converged
        )
        row["fsi_coupling_same_step_rerun_safety_rejected"] = (
            fsi_coupling_same_step_rerun_safety_rejected
        )
        row["fsi_coupling_same_step_rerun_fluid_substep_factor"] = (
            fsi_coupling_same_step_rerun_fluid_substep_factor
        )
        row["fsi_coupling_same_step_rerun_initial_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_initial_fluid_substeps
        )
        row["fsi_coupling_same_step_rerun_final_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_final_fluid_substeps
        )
        row["fsi_coupling_residual_continuation_iterations_max"] = (
            fsi_coupling_residual_continuation_iterations_max
        )
        row["fsi_coupling_residual_continuation_threshold_n"] = (
            fsi_coupling_residual_continuation_threshold_n
        )
        row["fsi_coupling_residual_continuation_rebound_secant_from_best"] = (
            fsi_coupling_residual_continuation_rebound_secant_from_best
        )
        row["fsi_coupling_residual_continuation_rebound_secant_factor"] = (
            fsi_coupling_residual_continuation_rebound_secant_factor
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max"
        ] = (
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
        )
        row["fsi_coupling_residual_continuation_iteration_count"] = (
            fsi_coupling_residual_continuation_iteration_count
        )
        row["fsi_coupling_residual_continuation_rebound_secant_count"] = (
            fsi_coupling_residual_continuation_rebound_secant_count
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count"
        ] = fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count
        row["fsi_coupling_previous_step_residual_norm_n"] = (
            math.nan
            if previous_step_fsi_coupling_residual_norm_n is None
            else previous_step_fsi_coupling_residual_norm_n
        )
        row["fsi_coupling_previous_step_cfl"] = (
            math.nan if previous_step_cfl is None else previous_step_cfl
        )
        row["fsi_coupling_mode"] = fsi_coupling_mode
        row["fsi_coupling_mode_paper_hibm_mpm"] = bool(
            fsi_coupling_mode_report["paper_hibm_mpm"]
        )
        row["region_pair_reaction_diagnostic_only"] = bool(
            fsi_coupling_mode_report["region_pair_reaction_diagnostic_only"]
        )
        row["fsi_coupling_solver"] = fsi_coupling_solver
        row["fsi_coupling_iterations_used"] = fsi_coupling_iterations_used
        row["fsi_coupling_enabled"] = fsi_coupling_enabled
        row["fsi_coupling_converged"] = fsi_coupling_converged
        row["fsi_all_trials_rejected"] = fsi_all_trials_rejected
        row["fsi_zero_force_commit_blocked"] = fsi_zero_force_commit_blocked
        row["fsi_coupling_residual_norm_n"] = fsi_coupling_residual_norm_n
        row["fsi_coupling_relaxation_effective"] = fsi_coupling_relaxation_effective
        row["fsi_coupling_rejected_trial_count"] = fsi_coupling_rejected_trial_count
        row["fsi_coupling_rejected_trial_backtrack_count"] = (
            fsi_coupling_rejected_trial_backtrack_count
        )
        row["fsi_coupling_residual_growth_rejected_trial_count"] = (
            fsi_coupling_residual_growth_rejected_trial_count
        )
        row["fsi_coupling_max_residual_rejected_trial_count"] = (
            fsi_coupling_max_residual_rejected_trial_count
        )
        row["fsi_coupling_trial_cfl_rejected_count"] = (
            fsi_coupling_trial_cfl_rejected_count
        )
        row["fsi_coupling_trial_interior_divergence_rejected_count"] = (
            fsi_coupling_trial_interior_divergence_rejected_count
        )
        row["fsi_coupling_trust_region_limited_update_count"] = (
            fsi_coupling_trust_region_limited_update_count
        )
        row["fsi_coupling_trust_region_shrink_count"] = (
            fsi_coupling_trust_region_shrink_count
        )
        row["fsi_coupling_trust_region_growth_count"] = (
            fsi_coupling_trust_region_growth_count
        )
        row["fsi_coupling_trust_region_rebound_backtrack_count"] = (
            fsi_coupling_trust_region_rebound_backtrack_count
        )
        row["fsi_coupling_trust_region_rebound_stop_count"] = (
            fsi_coupling_trust_region_rebound_stop_count
        )
        row["fsi_coupling_trust_region_rebound_stop_suppressed_count"] = (
            fsi_coupling_trust_region_rebound_stop_suppressed_count
        )
        row["fsi_coupling_trust_region_effective_force_increment_n"] = (
            fsi_coupling_trust_region_effective_force_increment_n
        )
        row["fsi_coupling_accepted_trial_cfl"] = fsi_coupling_accepted_trial_cfl
        row["fsi_coupling_accepted_trial_max_fluid_speed_mps"] = (
            fsi_coupling_accepted_trial_max_fluid_speed_mps
        )
        row["fsi_coupling_accepted_trial_interior_divergence_l2"] = (
            fsi_coupling_accepted_trial_interior_divergence_l2
        )
        row["fsi_coupling_trial_cfl_max"] = fsi_coupling_trial_cfl_max
        row["fsi_coupling_trial_interior_divergence_tolerance"] = (
            fsi_coupling_trial_interior_divergence_tolerance
        )
        row["fsi_coupling_trial_interior_divergence_l2_max"] = (
            fsi_coupling_trial_interior_divergence_l2_max
        )
        row["fsi_coupling_target_map_relaxation"] = (
            fsi_coupling_target_map_relaxation
        )
        row["fsi_coupling_rejected_trial_backtrack"] = (
            fsi_coupling_rejected_trial_backtrack
        )
        row["fsi_coupling_residual_growth_rejection_factor"] = (
            fsi_coupling_residual_growth_rejection_factor
        )
        row["fsi_coupling_max_accepted_residual_n"] = (
            fsi_coupling_max_accepted_residual_n
        )
        row["fsi_coupling_trust_region_force_increment_n"] = (
            fsi_coupling_trust_region_force_increment_n
        )
        row["fsi_coupling_trust_region_adaptive"] = (
            fsi_coupling_trust_region_adaptive
        )
        row["fsi_coupling_trust_region_shrink_factor"] = (
            fsi_coupling_trust_region_shrink_factor
        )
        row["fsi_coupling_trust_region_growth_factor"] = (
            fsi_coupling_trust_region_growth_factor
        )
        row["fsi_coupling_trust_region_rebound_factor"] = (
            fsi_coupling_trust_region_rebound_factor
        )
        row["fsi_coupling_trust_region_rebound_backtrack"] = (
            fsi_coupling_trust_region_rebound_backtrack
        )
        row["fsi_coupling_trust_region_rebound_stop_factor"] = (
            fsi_coupling_trust_region_rebound_stop_factor
        )
        row["fsi_coupling_trust_region_rebound_stop_max_residual_n"] = (
            fsi_coupling_trust_region_rebound_stop_max_residual_n
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
        row["accepted_fsi_trial_state_readvanced"] = (
            accepted_fsi_trial_state_readvanced
        )
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
        row["fsi_solid_response_dt_s"] = fsi_solid_response_dt_s
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
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
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
            aitken_lower_bound=interface_reaction_aitken_lower_bound,
            aitken_upper_bound=interface_reaction_aitken_upper_bound,
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
        row["interface_reaction_aitken_lower_bound"] = (
            interface_reaction_aitken_lower_bound
        )
        row["interface_reaction_aitken_upper_bound"] = (
            interface_reaction_aitken_upper_bound
        )
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
        row["fluid_substeps_base"] = effective_fluid_substeps
        row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
        row["adaptive_fluid_substeps_target_cfl"] = float(
            args.adaptive_fluid_substeps_target_cfl
        )
        row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
        row["adaptive_fluid_substeps_previous_substeps"] = previous_step_fluid_substeps
        rows.append(row)
        try:
            _raise_for_step_numerical_guard(
                row,
                cfl_limit=0.5,
                divergence_l2_limit=float(args.projection_divergence_tolerance),
            )
            _raise_for_step_solid_out_of_bounds_guard(row)
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
        previous_step_cfl = float(row["cfl"])
        previous_step_fsi_coupling_residual_norm_n = float(
            row["fsi_coupling_residual_norm_n"]
        )
        previous_step_fluid_substeps = int(
            float(row.get("fluid_substeps", step_fluid_substeps))
        )
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
                sharp_coupling_state=sharp_coupling_state,
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

    if rows and not args.checkpoint_every_step:
        # Closing checkpoint at loop exit (wall-time break or normal
        # completion) so every run can be resumed or extended. With
        # --checkpoint-every-step the final step already wrote it.
        write_run_checkpoint(
            run_checkpoint_path,
            completed_step=int(rows[-1]["step"]),
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
            args=args,
            simulator=simulator,
            solid_mpm=solid_mpm,
            interface_reaction_state=interface_reaction_state,
            sharp_coupling_state=sharp_coupling_state,
        )

    if sharp_case_runner_enabled:
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


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    result = main()
    summary_json = result.get("summary_json")
    if summary_json is None:
        summary_json = str(Path(result["history_csv"]).with_name("summary.json"))
    print(json.dumps({"summary_json": str(summary_json)}, indent=2))
