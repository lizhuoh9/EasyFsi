from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import taichi as ti

from cases.squid_soft_robot import (
    CHECKPOINT_ARG_FINGERPRINT_FIELDS,
    DEFAULT_SOURCE_CONFIG,
    FINITE_REQUIRED_ROW_FIELDS,
    RUN_CHECKPOINT_VERSION,
    ReducedSquidFSI,
    SquidReducedSpec,
    build_source_config_fluid_obstacle_mask,
    checkpoint_run_fingerprint,
    count_enabled_unconverged_fsi_rows,
    effective_fluid_substeps_for_grid,
    finite_required_row_fields_for_mode,
    finite_required_row_fields_for_solid_model,
    force_decomposition_report,
    fsi_same_step_rerun_triggered,
    fsi_trial_acceptance_passes,
    fsi_trial_acceptance_rejection_reason,
    fluid_grid_resolution_report,
    infer_spec,
    interface_reaction_target_for_mode,
    legacy_projected_reduced_fsi_coupling_enabled,
    build_hibm_mpm_sharp_coupling_state,
    build_hibm_mpm_sharp_case_row,
    sharp_marker_fixed_point_residual_mps,
    sharp_marker_fixed_point_residual_diagnostics_mps,
    sharp_report_fluid_projection_failure_reason,
    sharp_pressure_neumann_gradient_state_array,
    relaxed_sharp_pressure_neumann_gradient_state_array,
    relaxed_sharp_marker_state_arrays,
    restore_sharp_pressure_neumann_gradient_state_array,
    divergence_sample_report_fields,
    load_run_checkpoint,
    nozzle_radius_at_z_m,
    _cell_indices_for_points,
    _required_finite_row_number,
    _required_finite_row_vector,
    _raise_for_step_numerical_guard,
    _write_hibm_high_residual_cell_dump,
    _write_step_failure_artifacts,
    parse_args,
    pressure_schedule_pa,
    pressure_schedule_applied_in_history,
    pressure_schedule_step_end_pa,
    raise_for_unsupported_hibm_mpm_sharp_iteration_options,
    fsi_physical_interface_map_stability_report,
    pressure_flux_trend_report,
    fsi_physical_interface_map_stability_passes,
    physical_positive_source_flux_ratio_passes,
    physical_outlet_to_fsi_volume_source_passes,
    outlet_to_fsi_volume_source_gate_scope,
    pressure_outlet_source_ratio_passes,
    pressure_projection_budget_report,
    reduced_active_water_connectivity,
    resolve_step_count,
    robin_previous_velocity_for_step,
    solid_response_constraint_force_mobility_ratio,
    shell_surface_mass_budget,
    run_process_completion_status,
    validate_checkpoint_run_fingerprint,
    validation_scope_report,
    validate_resume_history_checkpoint_alignment,
    write_csv,
    write_run_checkpoint,
    _final_row_int,
    _final_row_number,
    _interface_state_from_checkpoint,
    _write_hibm_pressure_neumann_invalid_row_dump,
    required_fluid_impulse_report,
    required_projected_ibm_force_report,
    read_csv_rows,
    resume_history_rows_for_checkpoint,
    required_tuple3,
    reduced_water_geometry_report,
    resolve_divergence_cleanup_iterations,
    resolve_pressure_solver,
    run,
    runtime_budget_report,
    signed_positive_source_flux_ratio,
    solid_mpm_bounds_padding_distance_m,
    solid_mpm_force_nonzero_when_pressure_loaded,
    solid_mpm_bounds_from_surface_metadata,
    solid_force_vector_from_report,
    source_config_requests_fluid_active_mask,
    source_config_cad_provenance_report,
    source_config_pressure_boundary_shell_mapping,
    source_config_pressure_load_region_id,
    source_config_requests_reduced_water_intersection,
    source_config_requests_region14_aperture_carve,
    source_config_shell_region_pair,
    source_config_solid_obstacle_particle_region_ids,
    _clear_surface_region_normal_probe_obstacle_cells,
    _connect_surface_seed_components_to_zmin,
    _solid_band_protection_mask_from_cells,
    _surface_region_seed_mask,
    spec_with_membrane_thickness_scale,
    spec_with_nozzle_graded_grid,
    spec_with_nozzle_taper,
    spec_with_region14_aperture,
    tail_refinement_region_from_geometry,
)
from simulation_core import (
    CartesianFluidSolver,
    CartesianGrid,
    FluidDomainSpec,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    NeoHookeanMpmState,
    RefinementRegion,
    SurfaceMesh,
    TaichiRuntimeConfig,
    TriMooneyShellMpmState,
    build_graded_grid,
    fsi_coupling_mode_report,
)
from simulation_core.fsi_coupling import (
    InterfaceReactionRelaxationState,
    InterfaceReactionTargetEvaluation,
    aitken_relaxation_factor,
    interface_reaction_force,
    relax_interface_reaction_forces,
    solve_interface_reaction_fixed_point,
)
from simulation_core.hibm_mpm import (
    HibmMpmIbBoundaryConditions,
    HibmMpmSharpCouplingState,
    HibmMpmVelocityDirichletBoundaryReport,
    hibm_mpm_pressure_disconnected_region_report,
)


class SquidLatestCoreConfigTests(unittest.TestCase):
    @staticmethod
    def _nonuniform_reduced_grid() -> CartesianGrid:
        return CartesianGrid(
            bounds_min_m=(-0.20, -0.18, 0.0),
            cell_widths_x_m=(0.05, 0.08, 0.12, 0.15),
            cell_widths_y_m=(0.04, 0.07, 0.11, 0.14),
            cell_widths_z_m=(0.04, 0.08, 0.12, 0.20),
        )

    @staticmethod
    def _nonuniform_reduced_spec(grid: CartesianGrid) -> SquidReducedSpec:
        return SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=grid.bounds_min_m,
            fluid_bounds_max_m=grid.bounds_max_m,
            grid_nodes=grid.grid_nodes,
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.12,
            chamber_z_min_m=0.30,
            chamber_z_max_m=0.40,
            nozzle_radius_m=0.09,
            nozzle_z_max_m=0.22,
            outlet_plume_radius_m=0.11,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
            monitor_radius_m=1.0,
            lip_z_m=0.02,
            outlet_z_m=0.08,
            downstream_z_m=0.18,
            cartesian_grid=grid,
        )

    @staticmethod
    def _count_centers_between(values: tuple[float, ...], lower: float, upper: float) -> int:
        return sum(1 for value in values if lower <= value <= upper)

    @staticmethod
    def _cell_disk_intersects(
        *,
        x_m: float,
        y_m: float,
        width_x_m: float,
        width_y_m: float,
        center_x_m: float,
        center_y_m: float,
        radius_m: float,
    ) -> bool:
        closest_x = max(abs(float(x_m) - float(center_x_m)) - 0.5 * float(width_x_m), 0.0)
        closest_y = max(abs(float(y_m) - float(center_y_m)) - 0.5 * float(width_y_m), 0.0)
        return closest_x * closest_x + closest_y * closest_y <= float(radius_m) * float(radius_m)

    @staticmethod
    def _cell_z_intersects(
        *,
        z_m: float,
        width_z_m: float,
        lower_m: float,
        upper_m: float,
    ) -> bool:
        cell_min = float(z_m) - 0.5 * float(width_z_m)
        cell_max = float(z_m) + 0.5 * float(width_z_m)
        return cell_max >= float(lower_m) and cell_min <= float(upper_m)

    @staticmethod
    def _section_area_fraction(
        *,
        x_m: float,
        y_m: float,
        width_x_m: float,
        width_y_m: float,
        center_x_m: float,
        center_y_m: float,
        radius_m: float,
    ) -> float:
        hits = 0
        for sx in range(8):
            sample_x = (
                float(x_m)
                - float(center_x_m)
                - 0.5 * float(width_x_m)
                + (float(sx) + 0.5) * float(width_x_m) / 8.0
            )
            for sy in range(8):
                sample_y = (
                    float(y_m)
                    - float(center_y_m)
                    - 0.5 * float(width_y_m)
                    + (float(sy) + 0.5) * float(width_y_m) / 8.0
                )
                if sample_x * sample_x + sample_y * sample_y <= float(radius_m) * float(radius_m):
                    hits += 1
        return float(hits) / 64.0

    def test_finite_row_guard_covers_physical_credibility_fields(self) -> None:
        required_fields = set(FINITE_REQUIRED_ROW_FIELDS)

        for field in (
            "main_displacement_z_m",
            "tail_displacement_z_m",
            "main_velocity_z_mps",
            "tail_velocity_z_mps",
            "max_fluid_speed_mps",
            "cfl",
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
            "projected_ibm_residual_mps",
            "projected_ibm_residual_l2_mps",
            "pressure_traction_abs_force_n",
            "viscous_traction_force_x_n",
            "viscous_traction_force_y_n",
            "viscous_traction_force_z_n",
            "fluid_stress_traction_force_x_n",
            "fluid_stress_traction_force_y_n",
            "fluid_stress_traction_force_z_n",
            "fsi_probe_invalid_area_m2",
            "fsi_probe_invalid_volume_source_m3s",
            "fsi_force_probe_invalid_area_m2",
            "fsi_force_probe_invalid_volume_source_m3s",
            "fsi_probe_valid_fraction",
            "fsi_force_probe_valid_fraction",
            "pressure_outlet_source_volume_flux_m3s",
            "pressure_outlet_velocity_flux_m3s",
            "pressure_outlet_velocity_to_source_ratio",
            "pressure_outlet_pressure_flux_m3s",
            "pressure_outlet_pressure_to_source_ratio",
            "fsi_action_reaction_residual_abs_n",
            "fsi_action_reaction_relative_error",
            "fsi_fluid_reaction_action_reaction_relative_error",
            "fsi_last_correction_grid_decomposition_residual_abs_n",
            "fsi_last_correction_grid_decomposition_relative_error",
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
            "main_fsi_fluid_reaction_full_residual_n",
            "main_fsi_fluid_reaction_full_relative_error",
            "tail_fsi_fluid_reaction_full_residual_n",
            "tail_fsi_fluid_reaction_full_relative_error",
            "solid_mpm_transfer_relative_error",
            "solid_mpm_total_force_x_n",
            "solid_mpm_total_force_y_n",
            "solid_mpm_total_force_z_n",
            "solid_mpm_max_speed_mps",
            "outlet_flow_negative_z_m3s",
        ):
            self.assertIn(field, required_fields)
        self.assertNotIn("solid_mpm_max_abs_j", required_fields)
        self.assertNotIn("nozzle_velocity_z_mps", required_fields)

    def test_neo_hookean_finite_row_guard_adds_deformation_jacobian_field(self) -> None:
        tri_fields = set(finite_required_row_fields_for_solid_model("tri_mooney_shell_mpm"))
        neo_fields = set(finite_required_row_fields_for_solid_model("neo_hookean_mpm"))

        self.assertNotIn("solid_mpm_max_abs_j", tri_fields)
        self.assertIn("solid_mpm_max_abs_j", neo_fields)
        self.assertNotIn("nozzle_velocity_z_mps", tri_fields)

    def test_sharp_finite_row_guard_excludes_status_label_fields(self) -> None:
        sharp_fields = set(
            finite_required_row_fields_for_mode(
                FSI_COUPLING_MODE_HIBM_MPM_SHARP,
                solid_model="neo_hookean_mpm",
            )
        )

        for field in (
            "hibm_coupling_scheme",
            "hibm_added_mass_stability_status",
            "hibm_added_mass_stabilization",
            "pressure_projection_cg_restart_policy",
            "pressure_solve_failure_policy",
            "pressure_solve_failure_action",
            "pressure_projection_physical_failure_reason",
            "pressure_projection_physical_failure_action",
            "fsi_coupling_scheme",
            "fsi_added_mass_stability_status",
            "fsi_added_mass_stabilization",
        ):
            self.assertNotIn(field, sharp_fields)
        for measured_field in (
            "hibm_added_mass_stability_measured",
            "hibm_semi_implicit_coupling_enabled",
            "hibm_semi_implicit_coupling_matrix_active",
            "pressure_solve_failed",
            "pressure_projection_physical_failure",
            "hibm_unreached_incompatible_component_count",
            "hibm_unreached_component_rhs_mean_max_abs",
            "hibm_unreached_component_rhs_integral_max_abs",
            "fsi_added_mass_stability_measured",
            "fsi_semi_implicit_coupling_enabled",
            "fsi_semi_implicit_coupling_matrix_active",
        ):
            self.assertIn(measured_field, sharp_fields)

    def test_required_row_number_rejects_missing_and_nonfinite_values(self) -> None:
        with self.assertRaises(KeyError):
            _required_finite_row_number({}, "solid_mpm_total_force_z_n", context="test row")

        with self.assertRaises(ValueError):
            _required_finite_row_number(
                {"solid_mpm_total_force_z_n": float("nan")},
                "solid_mpm_total_force_z_n",
                context="test row",
            )

        with self.assertRaises(ValueError):
            _required_finite_row_number(
                {"solid_mpm_total_force_z_n": "not-a-number"},
                "solid_mpm_total_force_z_n",
                context="test row",
            )

    def test_step_numerical_guard_rejects_unstable_finite_rows(self) -> None:
        row = {
            "step": 7,
            "max_fluid_speed_mps": 0.01,
            "cfl": 0.1,
            "divergence_l2": 1.0e-3,
            "divergence_max_abs": 2.0e-3,
            "interior_divergence_l2": 1.0e-3,
            "interior_divergence_max_abs": 2.0e-3,
            "pressure_correctable_divergence_l2": 1.0e-3,
            "pressure_correctable_divergence_max_abs": 2.0e-3,
            "pressure_correctable_divergence_cell_count": 8,
            "pressure_fixed_divergence_l2": 0.0,
            "pressure_fixed_divergence_max_abs": 0.0,
            "pressure_fixed_divergence_cell_count": 0,
            "interior_pressure_correctable_divergence_l2": 1.0e-3,
            "interior_pressure_correctable_divergence_max_abs": 2.0e-3,
            "interior_pressure_correctable_divergence_cell_count": 8,
            "interior_pressure_fixed_divergence_l2": 0.0,
            "interior_pressure_fixed_divergence_max_abs": 0.0,
            "interior_pressure_fixed_divergence_cell_count": 0,
            "projection_divergence_l2": 1.0e-3,
            "projection_divergence_max_abs": 2.0e-3,
            "projection_to_pre_divergence_l2_ratio": 1.0,
            "post_boundary_divergence_l2": 1.0e-3,
            "post_boundary_divergence_max_abs": 2.0e-3,
            "post_boundary_to_pre_divergence_l2_ratio": 1.0,
            "post_constraint_divergence_l2": 1.0e-3,
            "post_constraint_divergence_max_abs": 2.0e-3,
            "post_constraint_to_pre_divergence_l2_ratio": 1.0,
        }

        _raise_for_step_numerical_guard(
            row,
            cfl_limit=0.5,
            divergence_l2_limit=1.0e-2,
        )

        high_cfl_row = dict(row, cfl=3.85)
        with self.assertRaisesRegex(RuntimeError, "cfl"):
            _raise_for_step_numerical_guard(
                high_cfl_row,
                cfl_limit=0.5,
                divergence_l2_limit=1.0e-2,
            )

        high_boundary_divergence_row = dict(row, divergence_l2=9.0e9)
        _raise_for_step_numerical_guard(
            high_boundary_divergence_row,
            cfl_limit=0.5,
            divergence_l2_limit=1.0e-2,
        )

        high_divergence_row = dict(row, divergence_l2=1.0e-3, interior_divergence_l2=9.0e9)
        with self.assertRaisesRegex(RuntimeError, "interior_divergence_l2"):
            _raise_for_step_numerical_guard(
                high_divergence_row,
                cfl_limit=0.5,
                divergence_l2_limit=1.0e-2,
            )

        cg_breakdown_row = dict(
            row,
            total_pressure_projection_cg_converged_all=False,
            total_pressure_projection_cg_breakdown_count=1,
        )
        with self.assertRaisesRegex(RuntimeError, "total_pressure_projection_cg"):
            _raise_for_step_numerical_guard(
                cg_breakdown_row,
                cfl_limit=0.5,
                divergence_l2_limit=1.0e-2,
            )

    def test_step_failure_artifacts_write_partial_history_and_failed_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            process_path = output_dir / "run_process.json"
            process_path.write_text(
                json.dumps({"status": "running", "command": "squid test"}),
                encoding="utf-8",
            )
            exc = RuntimeError("step 3 numerical guard failed: cfl=3.850000e+00")

            history_path = _write_step_failure_artifacts(
                process_path=process_path,
                output_dir=output_dir,
                rows=[{"step": 3, "cfl": 3.85, "divergence_l2": 9.0}],
                step=3,
                exc=exc,
            )

            process = json.loads(process_path.read_text(encoding="utf-8"))
            history_text = history_path.read_text(encoding="utf-8")

        self.assertTrue(history_path.name.endswith("history.csv"))
        self.assertIn("step,cfl,divergence_l2", history_text)
        self.assertIn("3,3.85,9.0", history_text)
        self.assertEqual(process["status"], "failed")
        self.assertEqual(process["command"], "squid test")
        self.assertEqual(process["failed_step"], 3)
        self.assertEqual(process["error_type"], "RuntimeError")
        self.assertIn("numerical guard failed", process["error"])
        self.assertEqual(process["history_csv"], str(history_path))

    def test_hibm_high_residual_dump_orders_largest_residual_first(self) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        divergence = np.zeros((4, 4, 4), dtype=np.float32)
        divergence[1, 1, 1] = 0.25
        divergence[2, 1, 1] = -0.75
        fluid.divergence.from_numpy(divergence)
        fluid.velocity_dirichlet_boundary_active[2, 1, 1] = 1
        fluid.velocity_dirichlet_boundary_value_mps[2, 1, 1] = (0.0, 0.0, -0.02)
        fluid.velocity_dirichlet_boundary_projection_weight[2, 1, 1] = 0.5

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = _write_hibm_high_residual_cell_dump(
                output_dir=Path(temp_dir),
                step=2,
                fluid=fluid,
                pressure_outlet_zmin=True,
                limit=2,
            )
            rows = read_csv_rows(Path(summary["csv_path"]))

        self.assertEqual(summary["dumped_cell_count"], 2)
        self.assertAlmostEqual(summary["max_abs_residual_s"], 0.75, delta=1.0e-6)
        self.assertEqual((rows[0]["i"], rows[0]["j"], rows[0]["k"]), ("2", "1", "1"))
        self.assertEqual(rows[0]["velocity_dirichlet_active"], "1")
        self.assertAlmostEqual(
            float(rows[0]["velocity_dirichlet_projection_weight"]),
            0.5,
            delta=1.0e-6,
        )

    def test_clear_velocity_dirichlet_rows_clears_marker_region_ids(self) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity_dirichlet_boundary_active[2, 1, 1] = 1
        fluid.velocity_dirichlet_boundary_value_mps[2, 1, 1] = (0.0, 0.0, -0.02)
        fluid.velocity_dirichlet_boundary_projection_weight[2, 1, 1] = 0.5
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 1, 1] = 7

        fluid.clear_velocity_dirichlet_boundary_rows()

        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[2, 1, 1]), 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_marker_region_id[2, 1, 1]), -1)

    def test_hibm_sharp_velocity_row_report_exposes_region_coverage(self) -> None:
        report = HibmMpmVelocityDirichletBoundaryReport(
            active_velocity_dirichlet_rows=7,
            inactive_obstacle_rows=1,
            max_abs_velocity_mps=0.5,
            primary_region_active_rows=2,
            secondary_region_active_rows=3,
            other_region_active_rows=1,
            unassigned_region_active_rows=1,
        )

        self.assertEqual(report.primary_region_active_rows, 2)
        self.assertEqual(report.secondary_region_active_rows, 3)
        self.assertEqual(report.other_region_active_rows, 1)
        self.assertEqual(report.unassigned_region_active_rows, 1)

        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")
        for key in (
            "hibm_velocity_dirichlet_primary_region_active_rows",
            "hibm_velocity_dirichlet_secondary_region_active_rows",
            "hibm_velocity_dirichlet_other_region_active_rows",
            "hibm_velocity_dirichlet_unassigned_region_active_rows",
            "hibm_next_velocity_dirichlet_primary_region_active_rows",
            "hibm_next_velocity_dirichlet_secondary_region_active_rows",
            "hibm_next_velocity_dirichlet_other_region_active_rows",
            "hibm_next_velocity_dirichlet_unassigned_region_active_rows",
        ):
            self.assertIn(key, source)

    def test_hibm_sharp_velocity_row_region_coverage_counts_active_rows_only(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        for i, region_id in enumerate((3, 4, 9, -1)):
            fluid.velocity_dirichlet_boundary_active[i, 0, 0] = 1
            fluid.velocity_dirichlet_boundary_marker_region_id[i, 0, 0] = region_id
        fluid.velocity_dirichlet_boundary_active[0, 1, 0] = 0
        fluid.velocity_dirichlet_boundary_marker_region_id[0, 1, 0] = 3

        counts = HibmMpmIbBoundaryConditions._velocity_dirichlet_region_row_counts(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_marker_region_id,
            primary_region_id=3,
            secondary_region_id=4,
        )

        self.assertEqual(counts, (1, 1, 1, 1))

    def test_hibm_pressure_disconnected_region_report_locates_row_stencil_touch(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        reachable = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, 1] = 0
        obstacle[1, 1, 2] = 0
        reachable[1, 1, 1] = 0
        reachable[1, 1, 2] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.hibm_pressure_outlet_reachable.from_numpy(reachable)
        fluid.hibm_pressure_reachability_barrier.fill(0)
        fluid.velocity_dirichlet_boundary_active[1, 1, 1] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[1, 1, 1] = 3
        fluid.velocity_dirichlet_boundary_active[1, 1, 3] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id[1, 1, 3] = 4
        fluid.velocity_dirichlet_boundary_active[2, 2, 2] = 0
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 2] = 3
        fluid.last_hibm_pressure_unreached_component_count = 1
        fluid.last_hibm_pressure_unreached_component_overflow = False
        fluid.last_hibm_pressure_component_labels_converged = True

        report = hibm_mpm_pressure_disconnected_region_report(
            fluid,
            primary_region_id=3,
            secondary_region_id=4,
        )

        self.assertEqual(report.cell_count, 2)
        self.assertEqual(report.component_count, 1)
        self.assertEqual((report.min_i, report.min_j, report.min_k), (1, 1, 1))
        self.assertEqual((report.max_i, report.max_j, report.max_k), (1, 1, 2))
        self.assertEqual(report.primary_region_stencil_cell_count, 1)
        self.assertEqual(report.secondary_region_stencil_cell_count, 1)

    def test_hibm_solid_band_unclassified_row_cloud_cell_is_enclosed_water(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, 1] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active[1, 1, 1] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[1, 1, 1] = 3
        node_kind_code = ti.field(dtype=ti.i32, shape=(4, 4, 4))
        node_kind_code.fill(0)

        marked = fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=False,
            node_kind_code=node_kind_code,
            unclassified_node_code=0,
        )

        self.assertEqual(marked, 0)
        self.assertEqual(fluid.last_hibm_solid_band_interior_cells, 0)
        self.assertEqual(fluid.last_hibm_solid_band_enclosed_water_cells, 1)
        self.assertEqual(
            fluid.last_hibm_solid_band_velocity_dirichlet_protected_cells,
            0,
        )
        self.assertEqual(int(fluid.obstacle[1, 1, 1]), 0)

    def test_hibm_sharp_rechecks_row_cloud_orphans_after_predictor_rows(
        self,
    ) -> None:
        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(
            source.count("convert_row_cloud_orphans_until_saturated()"),
            3,
        )

    def test_hibm_sharp_reports_next_row_cloud_orphan_cleanup(self) -> None:
        source = Path("simulation_core/hibm_mpm.py").read_text(encoding="utf-8")

        self.assertIn("next_row_cloud_orphan_cell_count", source)
        self.assertIn("hibm_next_row_cloud_orphan_cell_count", source)
        self.assertGreaterEqual(
            source.count("convert_hibm_row_cloud_orphan_components("),
            2,
        )

    def test_step_failure_artifacts_write_minimal_fluid_vti_when_available(self) -> None:
        class FakeField:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values

            def to_numpy(self) -> np.ndarray:
                return np.array(self.values)

        class FakeFluid:
            def __init__(self) -> None:
                velocity = np.zeros((2, 2, 2, 3), dtype=np.float32)
                velocity[1, 0, 0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                self.velocity = FakeField(velocity)
                self.obstacle = FakeField(np.zeros((2, 2, 2), dtype=np.int32))
                self.divergence = FakeField(np.ones((2, 2, 2), dtype=np.float32))
                self.cell_center_x_m = FakeField(np.array([0.05, 0.15], dtype=np.float32))
                self.cell_center_y_m = FakeField(np.array([0.05, 0.15], dtype=np.float32))
                self.cell_center_z_m = FakeField(np.array([0.05, 0.15], dtype=np.float32))
                self.cell_width_x_m = FakeField(np.array([0.1, 0.1], dtype=np.float32))
                self.cell_width_y_m = FakeField(np.array([0.1, 0.1], dtype=np.float32))
                self.cell_width_z_m = FakeField(np.array([0.1, 0.1], dtype=np.float32))

            def pressure_interface_matrix_terms_report(self) -> dict[str, object]:
                return {
                    "row_count": 2,
                    "row_active_count": 1,
                    "row_invalid_count": 1,
                    "row_diagonal_integral_abs_mismatch": 0.25,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            process_path = output_dir / "run_process.json"
            process_path.write_text("{}", encoding="utf-8")
            history_path = _write_step_failure_artifacts(
                process_path=process_path,
                output_dir=output_dir,
                rows=[{"step": 4, "cfl": 0.75}],
                step=4,
                exc=RuntimeError("failed"),
                fluid=FakeFluid(),
            )
            process = json.loads(process_path.read_text(encoding="utf-8"))
            vti_path = Path(process["failure_fluid_vti"])
            history_exists = history_path.exists()
            vti_text = vti_path.read_text(encoding="utf-8")

        self.assertTrue(history_exists)
        self.assertTrue(vti_path.name.endswith("_fluid.vti"))
        self.assertIn('<VTKFile type="ImageData"', vti_text)
        self.assertIn('Name="velocity_mps"', vti_text)
        self.assertIn('Name="speed_mps"', vti_text)
        self.assertEqual(process["failure_pressure_interface_matrix"]["row_count"], 2)
        self.assertEqual(
            process["failure_pressure_interface_matrix"]["row_invalid_count"],
            1,
        )
        self.assertIn('Name="obstacle"', vti_text)
        self.assertIn('Name="divergence"', vti_text)

    def test_sharp_coupling_failure_writes_partial_history_before_row_build(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_block = source.split("if sharp_case_runner_enabled:", 1)[1]
        advance_block = sharp_block.split(
            "fluid_wall_started_at = time.perf_counter()",
            1,
        )[1].split("fluid_advance_wall_time_s = max", 1)[0]

        self.assertIn("try:", advance_block)
        self.assertIn(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            advance_block,
        )
        self.assertIn("_write_step_failure_artifacts(", advance_block)
        self.assertIn("rows=rows", advance_block)
        self.assertIn("step=step", advance_block)
        self.assertIn("raise", advance_block)

    def test_sharp_sampling_uses_fluid_substep_dt_for_cfl(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_block = source.split("if sharp_case_runner_enabled:", 1)[1]
        sample_block = sharp_block.split(
            "sample_report = simulator.sample_after_projection(",
            1,
        )[1].split("sample_wall_time_s =", 1)[0]

        self.assertIn("dt_s=fluid_substep_dt_s", sample_block)
        self.assertNotIn("dt_s=spec.dt_s", sample_block)

    def test_required_tuple3_rejects_missing_or_wrong_length_values(self) -> None:
        self.assertEqual(
            required_tuple3((1.0, 2.0, 3.0), field="bounds"),
            (1.0, 2.0, 3.0),
        )

        with self.assertRaises(ValueError):
            required_tuple3(None, field="bounds")

        with self.assertRaises(ValueError):
            required_tuple3((1.0, 2.0), field="bounds")

    def test_required_row_vector_returns_valid_force_components(self) -> None:
        row = {
            "solid_mpm_total_force_x_n": 1.0,
            "solid_mpm_total_force_y_n": -2.0,
            "solid_mpm_total_force_z_n": 3.5,
        }

        force = _required_finite_row_vector(
            row,
            (
                "solid_mpm_total_force_x_n",
                "solid_mpm_total_force_y_n",
                "solid_mpm_total_force_z_n",
            ),
            context="test row",
        )

        self.assertEqual(force, (1.0, -2.0, 3.5))

    def test_infer_spec_rejects_missing_source_config(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing_squid_source_config_for_regression.json"

        with self.assertRaises(FileNotFoundError):
            infer_spec(missing_path, grid_scale=1.0)

    def test_default_source_config_is_not_machine_absolute(self) -> None:
        default_source_config = Path(DEFAULT_SOURCE_CONFIG)

        self.assertFalse(default_source_config.is_absolute())
        self.assertEqual(default_source_config.name, "simulation_config.json")

    def test_infer_spec_uses_fluid_material_properties_from_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_config = Path(temp_dir) / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {
                            "fluid": {
                                "grid_size_m": 2.5e-3,
                                "density_kgm3": 997.0,
                                "viscosity_pa_s": 8.9e-4,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = infer_spec(source_config, grid_scale=1.0)

        self.assertAlmostEqual(spec.water_density_kgm3, 997.0, delta=1.0e-12)
        self.assertAlmostEqual(spec.water_viscosity_pa_s, 8.9e-4, delta=1.0e-12)

    def test_infer_spec_uses_pressure_waveform_from_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_config = Path(temp_dir) / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "time_step_s": 0.1,
                            "pressure_schedule": {
                                "t0_s": 0.2,
                                "t1_s": 0.6,
                                "t2_s": 1.4,
                                "p0_pa": 250.0,
                                "p1_pa": 1250.0,
                                "p2_pa": -500.0,
                            },
                        },
                        "domains": {"fluid": {"grid_size_m": 0.25}},
                    }
                ),
                encoding="utf-8",
            )

            spec = infer_spec(source_config, grid_scale=1.0)

        self.assertAlmostEqual(spec.pressure_t0_s, 0.2)
        self.assertAlmostEqual(spec.pressure_t1_s, 0.6)
        self.assertAlmostEqual(spec.pressure_t2_s, 1.4)
        self.assertAlmostEqual(spec.pressure_p0_pa, 250.0)
        self.assertAlmostEqual(spec.pressure_p1_pa, 1250.0)
        self.assertAlmostEqual(spec.pressure_p2_pa, -500.0)
        self.assertEqual(resolve_step_count(None, spec), 14)
        self.assertAlmostEqual(pressure_schedule_pa(0.4, spec), 750.0)

    def test_default_step_count_reaches_full_pressure_waveform(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="config.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            pressure_t1_s=1.0,
            pressure_t2_s=2.0,
        )

        self.assertEqual(resolve_step_count(None, spec), 4000)
        self.assertEqual(resolve_step_count(2, spec), 2)
        with self.assertRaisesRegex(ValueError, "--steps must be positive"):
            resolve_step_count(0, spec)

    def test_membrane_thickness_scale_updates_surface_mass_budget(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="config.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            main_membrane_thickness_m=3.0e-3,
            tail_membrane_thickness_m=2.5e-3,
        )

        scaled_spec = spec_with_membrane_thickness_scale(spec, 3.0)
        report = shell_surface_mass_budget(
            spec=scaled_spec,
            density_kgm3=2080.0,
            baseline_spec=spec,
            baseline_density_kgm3=1040.0,
        )

        self.assertAlmostEqual(scaled_spec.main_membrane_thickness_m, 9.0e-3)
        self.assertAlmostEqual(scaled_spec.tail_membrane_thickness_m, 7.5e-3)
        self.assertAlmostEqual(report["main_surface_mass_kg_m2"], 18.72)
        self.assertAlmostEqual(report["tail_surface_mass_kg_m2"], 15.6)
        self.assertAlmostEqual(report["main_surface_mass_scale"], 6.0)
        self.assertAlmostEqual(report["tail_surface_mass_scale"], 6.0)

    def test_final_row_helpers_reject_missing_nonfinite_or_noninteger_fields(self) -> None:
        row = {
            "outlet_flow_negative_z_m3s": 2.0e-8,
            "solid_mpm_particle_count": 12,
        }

        self.assertEqual(_final_row_number(row, "outlet_flow_negative_z_m3s"), 2.0e-8)
        self.assertEqual(_final_row_int(row, "solid_mpm_particle_count"), 12)
        self.assertEqual(_final_row_number(None, "outlet_flow_negative_z_m3s"), 0.0)
        self.assertEqual(_final_row_int(None, "solid_mpm_particle_count"), 0)
        with self.assertRaises(KeyError):
            _final_row_number(row, "missing_field")
        with self.assertRaises(ValueError):
            _final_row_number({"outlet_flow_negative_z_m3s": float("nan")}, "outlet_flow_negative_z_m3s")
        with self.assertRaises(ValueError):
            _final_row_int({"solid_mpm_particle_count": 12.5}, "solid_mpm_particle_count")

    def test_pressure_schedule_applied_uses_history_not_final_row_only(self) -> None:
        self.assertTrue(
            pressure_schedule_applied_in_history(
                [
                    {"pressure_load_pa": 8000.0},
                    {"pressure_load_pa": 0.0},
                ]
            )
        )
        self.assertFalse(pressure_schedule_applied_in_history([]))
        self.assertFalse(pressure_schedule_applied_in_history([{"pressure_load_pa": 0.0}]))

    def test_divergence_sample_report_marks_missing_pre_projection_unmeasured(
        self,
    ) -> None:
        report = divergence_sample_report_fields(
            {
                "l2": 1.0e-5,
                "max_abs": 2.0e-5,
                "interior_l2": 3.0e-6,
                "interior_max_abs": 4.0e-6,
                "projection_l2": 5.0e-6,
                "projection_max_abs": 6.0e-6,
                "post_constraint_l2": 7.0e-6,
                "post_constraint_max_abs": 8.0e-6,
            }
        )

        self.assertEqual(report["pre_projection_divergence_l2"], 1.0e-5)
        self.assertEqual(report["pre_projection_divergence_max_abs"], 2.0e-5)
        self.assertFalse(report["pre_projection_divergence_measured"])
        self.assertEqual(
            report["pre_projection_divergence_source"],
            "fallback_final_divergence",
        )
        self.assertFalse(report["projection_divergence_ratio_measured"])
        self.assertEqual(report["projection_to_pre_divergence_l2_ratio"], 1.0)

    def test_divergence_sample_report_marks_pre_projection_measured(self) -> None:
        report = divergence_sample_report_fields(
            {
                "l2": 1.0e-5,
                "max_abs": 2.0e-5,
                "pre_projection_l2": 9.0e-5,
                "pre_projection_max_abs": 1.1e-4,
            }
        )

        self.assertEqual(report["pre_projection_divergence_l2"], 9.0e-5)
        self.assertEqual(report["pre_projection_divergence_max_abs"], 1.1e-4)
        self.assertTrue(report["pre_projection_divergence_measured"])
        self.assertEqual(
            report["pre_projection_divergence_source"],
            "fluid_projection_report",
        )

    def test_divergence_sample_report_exposes_stage_ratios(self) -> None:
        report = divergence_sample_report_fields(
            {
                "l2": 5.0e-5,
                "max_abs": 6.0e-5,
                "pre_projection_l2": 8.0e-5,
                "pre_projection_max_abs": 9.0e-5,
                "projection_l2": 2.0e-5,
                "projection_max_abs": 3.0e-5,
                "post_boundary_l2": 4.0e-5,
                "post_boundary_max_abs": 4.5e-5,
                "post_constraint_l2": 5.0e-5,
                "post_constraint_max_abs": 6.0e-5,
                "pressure_correctable_l2": 1.5e-5,
                "pressure_correctable_max_abs": 2.5e-5,
                "pressure_correctable_cell_count": 12,
                "pressure_fixed_l2": 3.5e-5,
                "pressure_fixed_max_abs": 4.5e-5,
                "pressure_fixed_cell_count": 2,
                "interior_pressure_correctable_l2": 1.0e-5,
                "interior_pressure_correctable_max_abs": 2.0e-5,
                "interior_pressure_correctable_cell_count": 6,
                "interior_pressure_fixed_l2": 3.0e-5,
                "interior_pressure_fixed_max_abs": 4.0e-5,
                "interior_pressure_fixed_cell_count": 1,
            }
        )

        self.assertEqual(report["projection_divergence_l2"], 2.0e-5)
        self.assertTrue(report["pressure_divergence_split_measured"])
        self.assertEqual(
            report["pressure_divergence_split_source"],
            "fluid_projection_report",
        )
        self.assertEqual(report["pressure_correctable_divergence_l2"], 1.5e-5)
        self.assertEqual(report["pressure_correctable_divergence_cell_count"], 12)
        self.assertEqual(report["pressure_fixed_divergence_l2"], 3.5e-5)
        self.assertEqual(report["pressure_fixed_divergence_cell_count"], 2)
        self.assertEqual(
            report["interior_pressure_correctable_divergence_l2"],
            1.0e-5,
        )
        self.assertEqual(
            report["interior_pressure_correctable_divergence_cell_count"],
            6,
        )
        self.assertEqual(report["interior_pressure_fixed_divergence_l2"], 3.0e-5)
        self.assertEqual(report["interior_pressure_fixed_divergence_cell_count"], 1)
        self.assertEqual(report["post_boundary_divergence_l2"], 4.0e-5)
        self.assertEqual(report["post_boundary_divergence_max_abs"], 4.5e-5)
        self.assertEqual(report["post_constraint_divergence_l2"], 5.0e-5)
        self.assertTrue(report["projection_divergence_ratio_measured"])
        self.assertEqual(report["projection_to_pre_divergence_l2_ratio"], 0.25)
        self.assertEqual(report["post_boundary_to_pre_divergence_l2_ratio"], 0.5)
        self.assertEqual(report["post_constraint_to_pre_divergence_l2_ratio"], 0.625)

    def test_solid_mpm_force_check_uses_pressure_loaded_rows(self) -> None:
        unloaded_force_row = {
            "pressure_load_pa": 0.0,
            "solid_mpm_total_force_x_n": 10.0,
            "solid_mpm_total_force_y_n": 0.0,
            "solid_mpm_total_force_z_n": 0.0,
        }
        loaded_zero_force_row = {
            "pressure_load_pa": 8000.0,
            "solid_mpm_total_force_x_n": 0.0,
            "solid_mpm_total_force_y_n": 0.0,
            "solid_mpm_total_force_z_n": 0.0,
        }
        loaded_force_row = {
            "pressure_load_pa": 8000.0,
            "solid_mpm_total_force_x_n": 0.0,
            "solid_mpm_total_force_y_n": -1.0,
            "solid_mpm_total_force_z_n": 0.0,
        }

        self.assertFalse(
            solid_mpm_force_nonzero_when_pressure_loaded(
                [unloaded_force_row, loaded_zero_force_row],
                force_required=True,
            )
        )
        self.assertTrue(
            solid_mpm_force_nonzero_when_pressure_loaded(
                [loaded_zero_force_row, loaded_force_row],
                force_required=True,
            )
        )
        self.assertTrue(
            solid_mpm_force_nonzero_when_pressure_loaded(
                [loaded_zero_force_row],
                force_required=False,
            )
        )

    def test_sharp_completed_step_checks_include_pressure_and_force_drive(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_summary_source = source.split("if sharp_case_runner_enabled:", 1)[
            1
        ].split("diagnostic_checks = {", 1)[0]

        self.assertIn(
            '"pressure_schedule_applied": pressure_schedule_applied_in_history(rows)',
            sharp_summary_source,
        )

    def test_sharp_completed_step_checks_require_pre_projection_measurement(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_summary_source = source[source.index("if sharp_case_runner_enabled:") :]

        self.assertIn(
            "pre_projection_divergence_measured_all",
            sharp_summary_source,
        )
        self.assertIn(
            '"pre_projection_divergence_measured": '
            "pre_projection_divergence_measured_all",
            sharp_summary_source,
        )
        self.assertIn(
            '"pre_projection_divergence_sources"',
            sharp_summary_source,
        )
        self.assertIn(
            '"solid_mpm_force_nonzero_when_pressure_loaded": '
            "solid_mpm_force_nonzero_when_pressure_loaded(",
            sharp_summary_source,
        )
        self.assertIn("force_required=solid_mpm_force_required", sharp_summary_source)

    def test_sharp_summary_reports_projection_stage_growth_ratios(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_summary_source = source[source.index("if sharp_case_runner_enabled:") :]

        self.assertIn("max_projection_to_pre_divergence_l2_ratio", sharp_summary_source)
        self.assertIn(
            "max_post_boundary_to_pre_divergence_l2_ratio",
            sharp_summary_source,
        )
        self.assertIn(
            "max_post_constraint_to_pre_divergence_l2_ratio",
            sharp_summary_source,
        )
        self.assertIn("projection_divergence_not_increased", sharp_summary_source)
        self.assertIn("post_constraint_divergence_not_increased", sharp_summary_source)

    def test_sharp_completed_step_checks_reject_invalid_hibm_reconstruction_rows(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_checks_source = source.split("if sharp_case_runner_enabled:", 1)[
            1
        ].split("diagnostic_checks = {", 1)[0]

        self.assertIn(
            '"hibm_velocity_dirichlet_reconstruction_valid": (',
            sharp_checks_source,
        )
        self.assertIn("max_velocity_dirichlet_invalid_count == 0", sharp_checks_source)
        self.assertIn(
            '"hibm_pressure_neumann_reconstruction_valid": (',
            sharp_checks_source,
        )
        self.assertIn("max_pressure_neumann_invalid_count == 0", sharp_checks_source)
        self.assertIn(
            '"max_hibm_velocity_dirichlet_invalid_reconstruction_count"',
            source,
        )
        self.assertIn(
            '"max_hibm_pressure_neumann_invalid_reconstruction_count"',
            source,
        )
        self.assertIn(
            '"max_hibm_pressure_neumann_invalid_unreconstructable_count"',
            source,
        )

    def test_sharp_completed_step_checks_reject_unmeasured_no_slip_residual(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_checks_source = source.split("if sharp_case_runner_enabled:", 1)[
            1
        ].split("diagnostic_checks = {", 1)[0]

        self.assertIn(
            '"hibm_no_slip_residual_samples_present": (',
            sharp_checks_source,
        )
        self.assertIn("max_no_slip_valid_marker_count > 0", sharp_checks_source)
        self.assertIn(
            '"hibm_post_solid_no_slip_residual_samples_present": (',
            sharp_checks_source,
        )
        self.assertIn(
            "max_post_solid_no_slip_valid_marker_count > 0",
            sharp_checks_source,
        )
        self.assertIn(
            '"hibm_no_slip_residual_all_markers_measured": (',
            sharp_checks_source,
        )
        self.assertIn("post_solid_no_slip_residual_required", sharp_checks_source)
        self.assertIn("max_no_slip_invalid_marker_count == 0", sharp_checks_source)
        self.assertIn(
            '"hibm_post_solid_no_slip_residual_all_markers_measured": (',
            sharp_checks_source,
        )
        self.assertIn(
            "max_post_solid_no_slip_invalid_marker_count == 0",
            sharp_checks_source,
        )
        self.assertIn('"max_hibm_no_slip_residual_valid_marker_count"', source)
        self.assertIn(
            '"max_hibm_post_solid_no_slip_residual_valid_marker_count"',
            source,
        )

    def test_pressure_neumann_invalid_row_dump_cli_and_writer(self) -> None:
        args = parse_args(["--diagnostic-dump-pressure-neumann-invalid-rows"])

        self.assertTrue(args.diagnostic_dump_pressure_neumann_invalid_rows)

        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        self.assertIn("args.diagnostic_dump_pressure_neumann_invalid_rows", source)
        self.assertIn("_write_hibm_pressure_neumann_invalid_row_dump", source)

        class FakeBoundary:
            def pressure_neumann_invalid_diagnostic_rows(self, **kwargs):
                self.kwargs = kwargs
                return [
                    {
                        "row_index": 0,
                        "reason_code": 1,
                        "reason": "unreconstructable",
                        "node_i": 2,
                        "node_j": 2,
                        "node_k": 2,
                        "owner_i": 2,
                        "owner_j": 2,
                        "owner_k": 2,
                        "neighbor_i": -1,
                        "neighbor_j": -1,
                        "neighbor_k": -1,
                        "anchor_i": -1,
                        "anchor_j": -1,
                        "anchor_k": -1,
                        "marker_index": 0,
                        "marker_region_id": 7,
                    }
                ]

        boundary = FakeBoundary()
        search = object()
        markers = object()
        fluid = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = _write_hibm_pressure_neumann_invalid_row_dump(
                output_dir=Path(temp_dir),
                step=3,
                ib_boundary=boundary,
                search=search,
                markers=markers,
                fluid=fluid,
            )
            csv_path = Path(str(summary["csv_path"]))
            rows = read_csv_rows(csv_path)

        self.assertEqual(summary["captured_invalid_row_count"], 1)
        self.assertEqual(summary["total_invalid_row_count"], 1)
        self.assertEqual(summary["reason_counts"], {"unreconstructable": 1})
        self.assertEqual(summary["marker_region_counts"], {"7": 1})
        self.assertEqual(rows[0]["reason"], "unreconstructable")
        self.assertEqual(rows[0]["node_i"], "2")
        self.assertIs(boundary.kwargs["search"], search)
        self.assertIs(boundary.kwargs["markers"], markers)
        self.assertIs(boundary.kwargs["fluid"], fluid)

    def test_physical_outlet_ratio_uses_signed_positive_source_flux(self) -> None:
        self.assertAlmostEqual(
            signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=2.0e-7,
                source_flux_m3s=1.0e-6,
            ),
            0.2,
        )
        self.assertAlmostEqual(
            signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=-2.0e-7,
                source_flux_m3s=1.0e-6,
            ),
            -0.2,
        )
        self.assertEqual(
            signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=2.0e-7,
                source_flux_m3s=0.0,
            ),
            0.0,
        )
        self.assertFalse(
            physical_positive_source_flux_ratio_passes(
                outlet_negative_z_flux_m3s=-2.0e-7,
                source_flux_m3s=1.0e-6,
                min_ratio=0.1,
            )
        )
        self.assertFalse(
            physical_positive_source_flux_ratio_passes(
                outlet_negative_z_flux_m3s=2.0e-7,
                source_flux_m3s=0.0,
                min_ratio=0.1,
            )
        )
        self.assertFalse(
            physical_positive_source_flux_ratio_passes(
                outlet_negative_z_flux_m3s=2.0e-7,
                source_flux_m3s=-1.0e-6,
                min_ratio=0.1,
            )
        )
        self.assertTrue(
            physical_positive_source_flux_ratio_passes(
                outlet_negative_z_flux_m3s=2.0e-7,
                source_flux_m3s=1.0e-6,
                min_ratio=0.1,
            )
        )

    def test_physical_outlet_gate_uses_fsi_volume_source(self) -> None:
        self.assertFalse(
            physical_outlet_to_fsi_volume_source_passes(
                outlet_negative_z_flux_m3s=1.0e-6,
                fsi_volume_source_m3s=1.0e-5,
                min_ratio=0.5,
            )
        )
        self.assertFalse(
            physical_outlet_to_fsi_volume_source_passes(
                outlet_negative_z_flux_m3s=-1.0e-6,
                fsi_volume_source_m3s=1.0e-6,
                min_ratio=0.1,
            )
        )
        self.assertTrue(
            physical_outlet_to_fsi_volume_source_passes(
                outlet_negative_z_flux_m3s=8.0e-7,
                fsi_volume_source_m3s=1.0e-6,
                min_ratio=0.5,
            )
        )

    def test_outlet_to_fsi_volume_source_gate_is_scope_aware(self) -> None:
        unresolved_short = outlet_to_fsi_volume_source_gate_scope(
            fluid_grid_resolution={
                "nozzle_resolves_diameter_10_cells": False,
                "nozzle_diameter_cells_min": 4,
            },
            validation_scope_complete=False,
        )
        resolved_developed = outlet_to_fsi_volume_source_gate_scope(
            fluid_grid_resolution={
                "nozzle_resolves_diameter_10_cells": True,
                "nozzle_diameter_cells_min": 12,
            },
            validation_scope_complete=True,
        )

        self.assertFalse(unresolved_short["hard_gate"])
        self.assertEqual(unresolved_short["gate"], "diagnostic_only")
        self.assertIn("nozzle_grid_not_resolved", unresolved_short["reasons"])
        self.assertIn("jet_development_scope_incomplete", unresolved_short["reasons"])
        self.assertTrue(resolved_developed["hard_gate"])
        self.assertEqual(resolved_developed["gate"], "completed_step_check")
        self.assertEqual(resolved_developed["reasons"], [])

    def test_pressure_outlet_gate_requires_actual_velocity_flux_near_source(self) -> None:
        self.assertFalse(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=1.0e-6,
                velocity_outlet_flux_m3s=0.0,
                pressure_outlet_flux_m3s=1.0e-6,
                ratio_tolerance=0.1,
            )
        )
        self.assertFalse(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=1.0e-6,
                velocity_outlet_flux_m3s=1.25e-6,
                pressure_outlet_flux_m3s=1.0e-6,
                ratio_tolerance=0.1,
            )
        )
        self.assertFalse(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=0.0,
                velocity_outlet_flux_m3s=1.0e-6,
                pressure_outlet_flux_m3s=1.0e-6,
                ratio_tolerance=0.1,
            )
        )
        self.assertFalse(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=-1.0e-6,
                velocity_outlet_flux_m3s=1.0e-6,
                pressure_outlet_flux_m3s=1.0e-12,
                ratio_tolerance=0.1,
            )
        )
        self.assertFalse(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=1.0e-6,
                velocity_outlet_flux_m3s=0.95e-6,
                pressure_outlet_flux_m3s=float("nan"),
                ratio_tolerance=0.1,
            )
        )
        self.assertTrue(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=1.0e-6,
                velocity_outlet_flux_m3s=0.95e-6,
                pressure_outlet_flux_m3s=1.0e-12,
                ratio_tolerance=0.1,
            )
        )
        self.assertTrue(
            pressure_outlet_source_ratio_passes(
                source_volume_flux_m3s=1.0e-6,
                velocity_outlet_flux_m3s=0.95e-6,
                pressure_outlet_flux_m3s=1.02e-6,
                ratio_tolerance=0.1,
            )
        )

    @staticmethod
    def _pressure_flux_history(
        *,
        steps: int,
        pressure_ratio_start: float,
        pressure_ratio_end: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        denominator = max(steps - 1, 1)
        for index in range(steps):
            alpha = index / float(denominator)
            pressure_ratio = (
                pressure_ratio_start
                + (pressure_ratio_end - pressure_ratio_start) * alpha
            )
            rows.append(
                {
                    "step": index + 1,
                    "pressure_load_pa": 4.0 * float(index + 1),
                    "pressure_outlet_velocity_to_source_ratio": 1.0,
                    "pressure_outlet_pressure_to_source_ratio": pressure_ratio,
                }
            )
        return rows

    def test_pressure_flux_trend_requires_requested_completed_history(self) -> None:
        report = pressure_flux_trend_report(
            self._pressure_flux_history(
                steps=20,
                pressure_ratio_start=1.0e-8,
                pressure_ratio_end=1.0e-8,
            ),
            requested_steps=200,
        )

        self.assertFalse(report["complete"])
        self.assertEqual(report["conclusion"], "incomplete")

    def test_pressure_flux_trend_records_kinematic_or_pressure_driven_conclusion(self) -> None:
        near_zero = pressure_flux_trend_report(
            self._pressure_flux_history(
                steps=200,
                pressure_ratio_start=1.0e-8,
                pressure_ratio_end=5.0e-7,
            ),
            requested_steps=200,
        )
        rising = pressure_flux_trend_report(
            self._pressure_flux_history(
                steps=200,
                pressure_ratio_start=1.0e-7,
                pressure_ratio_end=5.0e-2,
            ),
            requested_steps=200,
        )

        self.assertTrue(near_zero["complete"])
        self.assertEqual(
            near_zero["conclusion"],
            "pressure_implied_flux_remained_near_zero_kinematic_ibm_dominated",
        )
        self.assertLess(near_zero["max_pressure_ratio_abs"], 1.0e-3)
        self.assertTrue(rising["complete"])
        self.assertEqual(
            rising["conclusion"],
            "pressure_implied_flux_rose_pressure_driven_component_present",
        )
        self.assertGreater(rising["late_pressure_ratio_mean_abs"], 1.0e-2)
        self.assertGreater(rising["pressure_ratio_growth_factor"], 5.0)

    def test_validation_scope_marks_explicit_short_runs_as_partial_validation(self) -> None:
        explicit_short = validation_scope_report(
            requested_steps=200,
            completed_steps=200,
            full_pressure_waveform_steps=4000,
            partial_run_stopped=False,
        )
        self.assertEqual(explicit_short["validation_scope"], "explicit_step_count")
        self.assertFalse(explicit_short["validation_scope_complete"])
        self.assertEqual(
            explicit_short["validation_scope_reason"],
            "explicit_steps_before_full_pressure_waveform",
        )

        full_waveform = validation_scope_report(
            requested_steps=4000,
            completed_steps=4000,
            full_pressure_waveform_steps=4000,
            partial_run_stopped=False,
        )
        self.assertEqual(full_waveform["validation_scope"], "full_pressure_waveform")
        self.assertTrue(full_waveform["validation_scope_complete"])
        self.assertIsNone(full_waveform["validation_scope_reason"])

        wall_time_partial = validation_scope_report(
            requested_steps=200,
            completed_steps=1,
            full_pressure_waveform_steps=4000,
            partial_run_stopped=True,
            partial_run_reason="max_wall_time_s",
        )
        self.assertEqual(wall_time_partial["validation_scope"], "wall_time_partial")
        self.assertFalse(wall_time_partial["validation_scope_complete"])
        self.assertEqual(wall_time_partial["validation_scope_reason"], "max_wall_time_s")

    def test_run_process_status_distinguishes_short_validation_from_partial_run(self) -> None:
        self.assertEqual(
            run_process_completion_status(
                validation_scope_complete=False,
                validation_passed=None,
                partial_run_stopped=False,
                requested_steps=1,
                completed_steps=1,
            ),
            "finished",
        )
        self.assertEqual(
            run_process_completion_status(
                validation_scope_complete=False,
                validation_passed=None,
                partial_run_stopped=True,
                requested_steps=2,
                completed_steps=1,
            ),
            "partial",
        )
        self.assertEqual(
            run_process_completion_status(
                validation_scope_complete=True,
                validation_passed=False,
                partial_run_stopped=False,
                requested_steps=4000,
                completed_steps=4000,
            ),
            "validation_failed",
        )
        self.assertEqual(
            run_process_completion_status(
                validation_scope_complete=True,
                validation_passed=True,
                partial_run_stopped=False,
                requested_steps=4000,
                completed_steps=4000,
            ),
            "finished",
        )

    def test_run_rejects_nonfinite_numeric_entry_options_before_solver_start(self) -> None:
        cases = (
            ("--fsi-coupling-tolerance-n", "nan", "--fsi-coupling-tolerance-n"),
            ("--pressure-outlet-source-ratio-tolerance", "inf", "--pressure-outlet-source-ratio-tolerance"),
            ("--max-wall-time-s", "nan", "--max-wall-time-s"),
            ("--fsi-velocity-target-solid-mobility-ratio", "nan", "--fsi-velocity-target-solid-mobility-ratio"),
        )
        for flag, value, message in cases:
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    args = parse_args(
                        [
                            "--source-config",
                            str(temp_path / "source.json"),
                            "--output-dir",
                            str(temp_path / "run"),
                            "--steps",
                            "1",
                            flag,
                            value,
                        ]
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        run(args)

    def test_active_water_connectivity_detects_sealed_pocket(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(5, 5, 5),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
        )
        obstacle = np.ones(spec.grid_nodes, dtype=np.int32)
        obstacle[2, 2, 0] = 0
        obstacle[2, 2, 1] = 0
        obstacle[4, 4, 4] = 0

        report = reduced_active_water_connectivity(
            spec,
            obstacle_cell_count=int(obstacle.sum()),
            obstacle_mask=obstacle,
        )

        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["z_min_connected_active_cell_count"], 2)
        self.assertEqual(report["trapped_active_cell_count"], 1)
        self.assertFalse(report["connectivity_passed"])

    def test_force_decomposition_report_detects_grid_force_mismatch(self) -> None:
        report = force_decomposition_report(
            grid_force_n=(1.0, 2.0, -0.5),
            component_forces_n=((0.25, 1.5, -0.25), (0.75, 0.5, -0.25)),
        )

        self.assertEqual(report["residual_components_n"], (0.0, 0.0, 0.0))
        self.assertEqual(report["residual_norm_n"], 0.0)
        self.assertEqual(report["relative_error"], 0.0)
        self.assertTrue(report["passed"])

        bad_report = force_decomposition_report(
            grid_force_n=(1.0, 2.0, -0.5),
            component_forces_n=((0.25, 1.5, -0.25), (0.25, 0.5, -0.25)),
        )

        self.assertEqual(bad_report["residual_components_n"], (0.5, 0.0, 0.0))
        self.assertGreater(bad_report["relative_error"], 0.0)
        self.assertFalse(bad_report["passed"])

    def test_solid_force_vector_uses_current_tri_report_schema(self) -> None:
        force = solid_force_vector_from_report(
            SimpleNamespace(total_force_n=(1.0, -2.0, 3.0)),
            solid_model="tri_mooney_shell_mpm",
        )

        self.assertEqual(force, (1.0, -2.0, 3.0))

    def test_solid_force_vector_accepts_neo_external_force_schema(self) -> None:
        force = solid_force_vector_from_report(
            SimpleNamespace(external_force_n=(0.5, 0.25, -1.5)),
            solid_model="neo_hookean_mpm",
        )

        self.assertEqual(force, (0.5, 0.25, -1.5))

    def test_solid_force_vector_rejects_missing_force_schema(self) -> None:
        with self.assertRaises(AttributeError):
            solid_force_vector_from_report(
                SimpleNamespace(),
                solid_model="tri_mooney_shell_mpm",
            )

    def test_solid_force_vector_rejects_nonfinite_or_wrong_length_force(self) -> None:
        with self.assertRaises(ValueError):
            solid_force_vector_from_report(
                SimpleNamespace(total_force_n=(1.0, float("nan"), 3.0)),
                solid_model="tri_mooney_shell_mpm",
            )

        with self.assertRaises(ValueError):
            solid_force_vector_from_report(
                SimpleNamespace(total_force_n=(1.0, 2.0)),
                solid_model="tri_mooney_shell_mpm",
            )

    def test_required_projected_ibm_force_report_rejects_missing_or_nonfinite_report(self) -> None:
        valid_report = SimpleNamespace(
            grid_force_n=(1.0, 2.0, 3.0),
            primary_fluid_force_n=(0.5, 0.0, 0.0),
            secondary_fluid_force_n=(0.0, -0.5, 0.0),
            constraint_force_n=(0.1, 0.2, 0.3),
            primary_constraint_force_n=(0.1, 0.0, 0.0),
            secondary_constraint_force_n=(0.0, 0.2, 0.0),
            volume_source_m3s=0.01,
            primary_volume_source_m3s=0.01,
            secondary_volume_source_m3s=0.0,
            active_force_cells=7,
            force_sample_count=2,
            force_invalid_probe_count=0,
            force_valid_probe_count=2,
            force_valid_probe_fraction=1.0,
            invalid_probe_area_m2=0.0,
            invalid_probe_volume_source_m3s=0.0,
        )

        self.assertIs(required_projected_ibm_force_report(valid_report), valid_report)
        with self.assertRaises(RuntimeError):
            required_projected_ibm_force_report(None)
        with self.assertRaises(ValueError):
            required_projected_ibm_force_report(
                SimpleNamespace(
                    grid_force_n=(float("nan"), 0.0, 0.0),
                    primary_fluid_force_n=(0.0, 0.0, 0.0),
                    secondary_fluid_force_n=(0.0, 0.0, 0.0),
                    constraint_force_n=(0.0, 0.0, 0.0),
                    primary_constraint_force_n=(0.0, 0.0, 0.0),
                    secondary_constraint_force_n=(0.0, 0.0, 0.0),
                    volume_source_m3s=0.0,
                    primary_volume_source_m3s=0.0,
                    secondary_volume_source_m3s=0.0,
                    active_force_cells=1,
                    force_sample_count=1,
                    force_invalid_probe_count=0,
                    force_valid_probe_count=1,
                    force_valid_probe_fraction=1.0,
                    invalid_probe_area_m2=0.0,
                    invalid_probe_volume_source_m3s=0.0,
                )
            )

    def test_required_fluid_impulse_report_rejects_missing_or_nonfinite_report(self) -> None:
        valid_report = SimpleNamespace(
            grid_impulse_n_s=(1.0, 0.0, 0.0),
            momentum_delta_n_s=(1.0, 0.0, 0.0),
            impulse_relative_error=0.0,
            active_velocity_cells=9,
        )

        self.assertIs(required_fluid_impulse_report(valid_report), valid_report)
        with self.assertRaises(RuntimeError):
            required_fluid_impulse_report(None)
        with self.assertRaises(ValueError):
            required_fluid_impulse_report(
                SimpleNamespace(
                    grid_impulse_n_s=(0.0, 0.0, 0.0),
                    momentum_delta_n_s=(0.0, 0.0, 0.0),
                    impulse_relative_error=float("inf"),
                    active_velocity_cells=1,
                )
            )

    def test_region14_aperture_updates_monitor_radius(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            nozzle_radius_m=3.0e-3,
            monitor_radius_m=3.0e-3,
        )
        aperture_radius_m = 1.77e-3

        updated = spec_with_region14_aperture(
            spec,
            {
                "available": True,
                "area_weighted_centroid_m": [-0.031, 0.016, 0.968],
                "vertex_radius_p95_m": aperture_radius_m,
            },
        )

        self.assertAlmostEqual(updated.nozzle_radius_m, aperture_radius_m)
        self.assertAlmostEqual(updated.outlet_plume_radius_m, aperture_radius_m)
        self.assertAlmostEqual(updated.monitor_radius_m, aperture_radius_m)

    def test_source_config_requests_region14_aperture_carve(self) -> None:
        self.assertTrue(
            source_config_requests_region14_aperture_carve(
                {
                    "analysis_settings": {
                        "solid_obstacle_opening_carve_enabled": True,
                        "solid_obstacle_opening_carve_selection_ids": [14],
                    }
                }
            )
        )
        self.assertTrue(
            source_config_requests_region14_aperture_carve(
                {
                    "analysis_settings": {
                        "solid_obstacle_opening_carve_enabled": True,
                        "solid_obstacle_opening_carve_selection_ids": "12, 14",
                    }
                }
            )
        )
        self.assertFalse(
            source_config_requests_region14_aperture_carve(
                {
                    "analysis_settings": {
                        "solid_obstacle_opening_carve_enabled": True,
                        "solid_obstacle_opening_carve_selection_ids": [12],
                    }
                }
            )
        )
        self.assertFalse(
            source_config_requests_region14_aperture_carve(
                {
                    "analysis_settings": {
                        "solid_obstacle_opening_carve_enabled": False,
                        "solid_obstacle_opening_carve_selection_ids": [14],
                    }
                }
            )
        )

    def test_source_config_requests_fluid_active_mask(self) -> None:
        self.assertTrue(
            source_config_requests_fluid_active_mask(
                {"analysis_settings": {"fluid_active_mask_enabled": True}}
            )
        )
        self.assertFalse(
            source_config_requests_fluid_active_mask(
                {"analysis_settings": {"fluid_active_mask_enabled": False}}
            )
        )

    def test_source_config_solid_obstacle_regions_prefer_surface_only_bodies(self) -> None:
        config = {
            "analysis_settings": {
                "solid_obstacle_surface_only_region_ids": [3, 4],
                "solid_obstacle_exclude_fsi_contact_regions": True,
                "solid_obstacle_moving_fsi_contact_region_ids": [1, 2],
            }
        }

        self.assertEqual(
            source_config_solid_obstacle_particle_region_ids(config, [1, 2, 3, 4]),
            (3, 4),
        )

    def test_source_config_pressure_load_uses_pressure_boundary_not_stale_load_scope(
        self,
    ) -> None:
        config = {
            "analysis_settings": {
                "solid_obstacle_moving_fsi_contact_surface_region_ids": [7, 8],
            },
            "loads": [
                {
                    "type": "Pressure",
                    "scope": {
                        "type": "named_selection",
                        "named_selection_id": 5,
                        "named_selection_name": "stale pressure scope",
                    },
                },
            ],
            "named_selections": [
                {
                    "id": 5,
                    "name": "fixed rim",
                    "boundary_condition": {"type": "Fixed Support", "params": {}},
                },
                {
                    "id": 6,
                    "name": "main membrane air pressure side",
                    "boundary_condition": {
                        "type": "Pressure",
                        "params": {"Pressure": 8000.0, "Direction": "-z"},
                    },
                },
                {
                    "id": 7,
                    "name": "main membrane water FSI side",
                    "boundary_condition": {"type": "Free", "params": {}},
                },
                {
                    "id": 8,
                    "name": "tail membrane FSI side",
                    "boundary_condition": {"type": "Free", "params": {}},
                },
            ],
        }

        self.assertEqual(source_config_pressure_load_region_id(config), 6)
        self.assertEqual(source_config_shell_region_pair(config), (7, 8))
        mapping = source_config_pressure_boundary_shell_mapping(config)
        self.assertEqual(mapping.source_region_id, 6)
        self.assertEqual(mapping.target_shell_region_id, 7)
        self.assertEqual(mapping.primary_shell_region_id, 7)
        self.assertEqual(mapping.secondary_shell_region_id, 8)
        self.assertEqual(
            mapping.mapping_source,
            "inferred_dry_pressure_side_to_primary_fsi_shell_from_source_config",
        )
        self.assertTrue(mapping.boundary_condition_input_only)

    def test_source_config_pressure_mapping_requires_declared_fsi_shell_target(
        self,
    ) -> None:
        config = {
            "named_selections": [
                {
                    "id": 6,
                    "name": "main membrane air pressure side",
                    "boundary_condition": {
                        "type": "Pressure",
                        "params": {"Pressure": 8000.0},
                    },
                },
                {
                    "id": 7,
                    "name": "main membrane water FSI side",
                    "boundary_condition": {"type": "Free", "params": {}},
                },
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "does not declare ordered moving FSI contact surface regions",
        ):
            source_config_pressure_boundary_shell_mapping(config)

    def test_source_config_pressure_mapping_accepts_explicit_target(self) -> None:
        config = {
            "analysis_settings": {
                "solid_obstacle_moving_fsi_contact_surface_region_ids": [7, 8],
                "pressure_boundary_to_fsi_shell_region_ids": {"6": 7},
            },
            "named_selections": [
                {
                    "id": 6,
                    "name": "actuator pressure boundary",
                    "boundary_condition": {
                        "type": "Pressure",
                        "params": {"Pressure": 8000.0},
                    },
                },
                {
                    "id": 7,
                    "name": "primary shell",
                    "boundary_condition": {"type": "Free", "params": {}},
                },
                {
                    "id": 8,
                    "name": "secondary shell",
                    "boundary_condition": {"type": "Free", "params": {}},
                },
            ],
        }

        mapping = source_config_pressure_boundary_shell_mapping(config)

        self.assertEqual(mapping.source_region_id, 6)
        self.assertEqual(mapping.target_shell_region_id, 7)
        self.assertEqual(
            mapping.mapping_source,
            "explicit_source_config_pressure_boundary_target",
        )

    def test_source_config_fluid_obstacle_mask_uses_volume_particles_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "fluid_active_mask_enabled": True,
                            "fluid_active_mask_mode": "ibamr_like_connected_component",
                            "fluid_active_mask_seed_boundary_sides": ["z_min"],
                            "fluid_active_mask_seed_radius_cells": 1,
                            "solid_obstacle_exclude_fsi_contact_regions": True,
                            "solid_obstacle_moving_fsi_contact_region_ids": [1],
                            "solid_obstacle_mask_dilation_cells": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                temp_path / "source.test.volume_particles.npz",
                particle_rest_positions_m=np.asarray(
                    [
                        (0.15, 0.15, 0.15),  # region 1: moving FSI, not static obstacle
                        (0.35, 0.35, 0.35),  # region 3: static CAD obstacle
                        (0.65, 0.65, 0.65),  # region 4: static CAD obstacle
                    ],
                    dtype=np.float32,
                ),
                particle_region_ids=np.asarray([1, 3, 4], dtype=np.int32),
                particle_volumes_m3=np.ones(3, dtype=np.float32),
            )
            grid = CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
            )

            obstacle, report = build_source_config_fluid_obstacle_mask(
                config=json.loads(source_config.read_text(encoding="utf-8")),
                source_config_path=source_config,
                grid=grid,
                aperture_geometry={"available": False},
            )

        self.assertEqual(obstacle.shape, (4, 4, 4))
        self.assertEqual(report["obstacle_region_ids"], (3, 4))
        self.assertEqual(report["particle_obstacle_region_ids"], (3, 4))
        self.assertEqual(report["selected_particle_count"], 2)
        self.assertEqual(report["raw_solid_obstacle_cell_count"], 2)
        self.assertEqual(report["fluid_active_mask_seed_cell_count"], 16)
        self.assertEqual(report["host_device_transfer_policy"], "one_time_initial_obstacle_from_numpy_before_steps")
        self.assertEqual(int(obstacle[1, 1, 1]), 1)
        self.assertEqual(int(obstacle[2, 2, 2]), 1)
        self.assertEqual(int(obstacle[0, 0, 0]), 0)

    def test_cell_indices_include_points_on_upper_grid_boundary(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(2, 2, 2),
        )

        i, j, k, valid = _cell_indices_for_points(
            np.asarray([[1.0, 0.5, 1.0]], dtype=np.float64),
            grid,
        )

        self.assertTrue(bool(valid[0]))
        self.assertEqual((int(i[0]), int(j[0]), int(k[0])), (1, 1, 1))

    def test_source_config_active_mask_uses_surface_region_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mesh_path = temp_path / "seed_surface.stl"
            mesh_path.write_text(
                "\n".join(
                    [
                        "solid seed_surface",
                        "facet normal 0 0 1",
                        "outer loop",
                        "vertex 1.25 1.25 2.5",
                        "vertex 1.75 1.25 2.5",
                        "vertex 1.50 1.75 2.5",
                        "endloop",
                        "endfacet",
                        "endsolid seed_surface",
                    ]
                ),
                encoding="ascii",
            )
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "mesh_path": str(mesh_path),
                        "mesh_scale_to_m": 1.0,
                        "named_selections": [
                            {"id": 8, "name": "fsi seed surface", "face_ids": [0]},
                        ],
                        "analysis_settings": {
                            "fluid_active_mask_enabled": True,
                            "fluid_active_mask_mode": "ibamr_like_connected_component",
                            "fluid_active_mask_seed_boundary_sides": ["z_min"],
                            "fluid_active_mask_seed_radius_cells": 1,
                            "fluid_active_mask_seed_region_ids": [8],
                            "solid_obstacle_exclude_fsi_contact_regions": False,
                            "solid_obstacle_mask_dilation_cells": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            barrier_particles = []
            for i in range(3):
                for j in range(3):
                    barrier_particles.append((0.5 + i, 0.5 + j, 1.5))
            np.savez(
                temp_path / "source.test.volume_particles.npz",
                particle_rest_positions_m=np.asarray(barrier_particles, dtype=np.float32),
                particle_region_ids=np.full(len(barrier_particles), 3, dtype=np.int32),
                particle_volumes_m3=np.ones(len(barrier_particles), dtype=np.float32),
            )
            grid = CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(3.0, 3.0, 3.0),
                grid_nodes=(3, 3, 3),
            )

            obstacle, report = build_source_config_fluid_obstacle_mask(
                config=json.loads(source_config.read_text(encoding="utf-8")),
                source_config_path=source_config,
                grid=grid,
                aperture_geometry={"available": False},
            )

        self.assertEqual(int(obstacle[1, 1, 1]), 1)
        self.assertEqual(int(obstacle[1, 1, 0]), 0)
        self.assertEqual(int(obstacle[1, 1, 2]), 0)
        self.assertGreater(report["fluid_active_mask_surface_seed_cell_count"], 0)
        self.assertGreater(report["fluid_active_mask_seed_cell_count"], 9)

    def test_surface_region_seed_mask_adds_normal_probe_seed_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mesh_path = temp_path / "normal_probe_seed.stl"
            mesh_path.write_text(
                "\n".join(
                    [
                        "solid normal_probe_seed",
                        "facet normal 0 0 1",
                        "outer loop",
                        "vertex 0.25 0.25 1.25",
                        "vertex 0.75 0.25 1.25",
                        "vertex 0.50 0.75 1.25",
                        "endloop",
                        "endfacet",
                        "endsolid normal_probe_seed",
                    ]
                ),
                encoding="ascii",
            )
            config = {
                "mesh_path": str(mesh_path),
                "mesh_scale_to_m": 1.0,
                "named_selections": [
                    {"id": 7, "name": "fsi water side", "face_ids": [0]},
                ],
            }
            grid = CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(2.0, 2.0, 3.0),
                grid_nodes=(2, 2, 3),
            )

            seed, report = _surface_region_seed_mask(
                config=config,
                grid=grid,
                region_ids=(7,),
                radius_cells=0,
                normal_probe_distance_m=1.0,
            )

        self.assertTrue(bool(seed[0, 0, 1]))
        self.assertTrue(bool(seed[0, 0, 2]))
        self.assertEqual(
            report["fluid_active_mask_surface_seed_normal_probe_point_count"],
            1,
        )
        self.assertAlmostEqual(
            report["fluid_active_mask_surface_seed_normal_probe_distance_m"],
            1.0,
        )

    def test_surface_region_probe_clear_removes_obstacle_probe_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mesh_path = temp_path / "probe_clear.stl"
            mesh_path.write_text(
                "\n".join(
                    [
                        "solid probe_clear",
                        "facet normal 0 0 1",
                        "outer loop",
                        "vertex 0.25 0.25 1.25",
                        "vertex 0.75 0.25 1.25",
                        "vertex 0.50 0.75 1.25",
                        "endloop",
                        "endfacet",
                        "endsolid probe_clear",
                    ]
                ),
                encoding="ascii",
            )
            config = {
                "mesh_path": str(mesh_path),
                "mesh_scale_to_m": 1.0,
                "named_selections": [
                    {"id": 7, "name": "fsi water side", "face_ids": [0]},
                ],
            }
            grid = CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(2.0, 2.0, 3.0),
                grid_nodes=(2, 2, 3),
            )
            obstacle = np.zeros(tuple(grid.grid_nodes), dtype=bool)
            obstacle[0, 0, 1] = True
            obstacle[0, 0, 2] = True

            report = _clear_surface_region_normal_probe_obstacle_cells(
                obstacle,
                config=config,
                grid=grid,
                region_ids=(7,),
                normal_probe_distance_m=1.0,
                radius_cells=0,
            )

        self.assertTrue(bool(obstacle[0, 0, 1]))
        self.assertFalse(bool(obstacle[0, 0, 2]))
        self.assertEqual(
            report["fluid_active_mask_surface_probe_clear_cell_count"],
            1,
        )
        self.assertEqual(
            report["fluid_active_mask_surface_probe_clear_point_count"],
            1,
        )
        self.assertEqual(
            report["fluid_active_mask_surface_probe_clear_cells_ijk"],
            ((0, 0, 2),),
        )

    def test_connect_surface_seed_components_to_zmin_carves_minimal_barrier(self) -> None:
        obstacle = np.zeros((1, 1, 5), dtype=bool)
        obstacle[0, 0, 2] = True
        boundary_seed = np.zeros_like(obstacle, dtype=bool)
        boundary_seed[0, 0, 0] = True
        surface_seed = np.zeros_like(obstacle, dtype=bool)
        surface_seed[0, 0, 4] = True

        report = _connect_surface_seed_components_to_zmin(
            obstacle,
            boundary_seed=boundary_seed,
            surface_seed=surface_seed,
            max_carve_cells=1,
        )

        self.assertFalse(bool(obstacle[0, 0, 2]))
        self.assertEqual(report["initial_unreachable_surface_seed_component_count"], 1)
        self.assertEqual(report["final_unreachable_surface_seed_component_count"], 0)
        self.assertEqual(report["connected_path_count"], 1)
        self.assertEqual(report["carved_cell_count"], 1)
        self.assertEqual(report["carved_bbox_ijk"], ((0, 0, 2), (0, 0, 2)))

    def test_connect_surface_seed_components_to_zmin_respects_carve_limit(self) -> None:
        obstacle = np.zeros((1, 1, 5), dtype=bool)
        obstacle[0, 0, 2] = True
        boundary_seed = np.zeros_like(obstacle, dtype=bool)
        boundary_seed[0, 0, 0] = True
        surface_seed = np.zeros_like(obstacle, dtype=bool)
        surface_seed[0, 0, 4] = True

        report = _connect_surface_seed_components_to_zmin(
            obstacle,
            boundary_seed=boundary_seed,
            surface_seed=surface_seed,
            max_carve_cells=0,
        )

        self.assertTrue(bool(obstacle[0, 0, 2]))
        self.assertEqual(report["initial_unreachable_surface_seed_component_count"], 1)
        self.assertEqual(report["final_unreachable_surface_seed_component_count"], 1)
        self.assertEqual(report["connected_path_count"], 0)
        self.assertEqual(report["carved_cell_count"], 0)
        self.assertTrue(report["skipped_by_max_carve_limit"])

    def test_solid_band_protection_mask_from_cells_dilates_locally(self) -> None:
        mask = _solid_band_protection_mask_from_cells(
            (3, 3, 3),
            ((1, 1, 1),),
            radius_cells=1,
        )

        self.assertEqual(int(mask.sum()), 27)
        self.assertEqual(int(mask[1, 1, 1]), 1)
        self.assertEqual(int(mask[0, 0, 0]), 1)
        self.assertEqual(int(mask[2, 2, 2]), 1)

    def test_source_config_active_mask_does_not_intersect_reduced_water_by_default(self) -> None:
        config = {
            "analysis_settings": {
                "fluid_active_mask_enabled": True,
                "fluid_active_mask_mode": "ibamr_like_connected_component",
            },
        }

        self.assertFalse(source_config_requests_reduced_water_intersection(config))

    def test_source_config_active_mask_reduced_water_intersection_is_explicit_opt_in(self) -> None:
        config = {
            "analysis_settings": {
                "fluid_active_mask_enabled": True,
                "fluid_active_mask_intersect_reduced_water_domain": True,
            },
        }

        self.assertTrue(source_config_requests_reduced_water_intersection(config))

    def test_source_config_region14_aperture_carve_enables_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "preflight_region14"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "time_step_s": 5.0e-4,
                            "solid_obstacle_opening_carve_enabled": True,
                            "solid_obstacle_opening_carve_selection_ids": [14],
                        },
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            aperture_stats = {
                "region_id": 14,
                "available": True,
                "area_weighted_centroid_m": [0.004, -0.005, 0.967],
                "vertex_radius_p95_m": 1.8e-3,
            }
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--preflight-only",
                ]
            )
            with patch(
                "cases.squid_soft_robot.compute_region_geometry_stats",
                return_value=aperture_stats,
            ):
                summary = run(args)

            self.assertTrue(summary["source_config_region14_aperture_requested"])
            self.assertTrue(summary["region14_aperture_carve_enabled"])
            self.assertEqual(summary["region14_aperture_carve_source"], "source_config")
            self.assertAlmostEqual(
                summary["reduced_water_geometry"]["nozzle_throat_radius_m"],
                aperture_stats["vertex_radius_p95_m"],
            )

    def test_region14_aperture_carve_can_be_disabled_for_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "preflight_region14_disabled"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {
                            "time_step_s": 5.0e-4,
                            "solid_obstacle_opening_carve_enabled": True,
                            "solid_obstacle_opening_carve_selection_ids": [14],
                        },
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--disable-region14-aperture-carve",
                    "--preflight-only",
                ]
            )
            with patch(
                "cases.squid_soft_robot.compute_region_geometry_stats",
                return_value={
                    "region_id": 14,
                    "available": True,
                    "area_weighted_centroid_m": [0.004, -0.005, 0.967],
                    "vertex_radius_p95_m": 1.8e-3,
                },
            ):
                summary = run(args)

            self.assertTrue(summary["source_config_region14_aperture_requested"])
            self.assertFalse(summary["region14_aperture_carve_enabled"])
            self.assertEqual(
                summary["region14_aperture_carve_source"],
                "disabled_by_cli",
            )

    def test_nozzle_graded_grid_uses_aperture_radius_and_resolves_nozzle(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.02, -0.02, 0.90),
            fluid_bounds_max_m=(0.02, 0.02, 1.04),
            grid_nodes=(16, 16, 56),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            nozzle_radius_m=3.0e-3,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
            downstream_z_m=0.95,
            nozzle_z_max_m=0.99,
        )
        aperture_radius_m = 1.8e-3
        aperture_center = (0.004, -0.005, 0.967)
        updated = spec_with_region14_aperture(
            spec,
            {
                "available": True,
                "area_weighted_centroid_m": list(aperture_center),
                "vertex_radius_p95_m": aperture_radius_m,
            },
        )

        graded = spec_with_nozzle_graded_grid(
            updated,
            farfield_spacing_m=3.0e-3,
            max_cells=1_000_000,
        )
        self.assertIsNone(graded.cartesian_grid)
        self.assertIsNotNone(graded.graded_grid)
        self.assertEqual(graded.graded_grid.max_cells, 1_000_000)
        self.assertEqual(len(graded.graded_grid.refinement_regions), 1)
        grid = build_graded_grid(graded.graded_grid)
        self.assertEqual(graded.grid_nodes, grid.grid_nodes)
        region = graded.graded_grid.refinement_regions[0]
        target_spacing_m = aperture_radius_m / 5.0

        self.assertEqual(region.target_spacing_m, (target_spacing_m,) * 3)
        self.assertAlmostEqual(region.bounds_min_m[0], aperture_center[0] - aperture_radius_m)
        self.assertAlmostEqual(region.bounds_max_m[0], aperture_center[0] + aperture_radius_m)
        self.assertAlmostEqual(region.bounds_min_m[1], aperture_center[1] - aperture_radius_m)
        self.assertAlmostEqual(region.bounds_max_m[1], aperture_center[1] + aperture_radius_m)
        self.assertGreaterEqual(
            self._count_centers_between(
                grid.cell_centers_x_m,
                aperture_center[0] - aperture_radius_m,
                aperture_center[0] + aperture_radius_m,
            ),
            10,
        )
        self.assertGreaterEqual(
            self._count_centers_between(
                grid.cell_centers_y_m,
                aperture_center[1] - aperture_radius_m,
                aperture_center[1] + aperture_radius_m,
            ),
            10,
        )
        refined_x_widths = [
            width
            for center, width in zip(grid.cell_centers_x_m, grid.cell_widths_x_m, strict=True)
            if aperture_center[0] - aperture_radius_m <= center <= aperture_center[0] + aperture_radius_m
        ]
        self.assertLessEqual(max(refined_x_widths), target_spacing_m * 1.0_000_001)
        self.assertLessEqual(max(grid.cell_widths_x_m), 3.0e-3 * 1.0_000_001)

    def test_tail_refinement_region_uses_region8_bounds_without_physics_fields(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.02, -0.02, 0.90),
            fluid_bounds_max_m=(0.02, 0.02, 1.04),
            grid_nodes=(16, 16, 56),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
        )

        region = tail_refinement_region_from_geometry(
            spec,
            {
                "available": True,
                "region_id": 8,
                "vertex_bounds_min_m": [-0.018, -0.009, 0.930],
                "vertex_bounds_max_m": [0.010, 0.008, 0.995],
            },
            target_spacing_m=1.0e-3,
            padding_m=4.0e-3,
        )

        self.assertIsNotNone(region)
        self.assertEqual(region.target_spacing_m, (1.0e-3,) * 3)
        self.assertEqual(region.bounds_min_m, (-0.02, -0.013, 0.926))
        self.assertEqual(region.bounds_max_m, (0.014, 0.012, 0.999))
        report_text = json.dumps(
            {
                "tail_refinement_enabled": True,
                "tail_refinement_region": {
                    "bounds_min_m": region.bounds_min_m,
                    "bounds_max_m": region.bounds_max_m,
                    "target_spacing_m": region.target_spacing_m,
                },
            }
        ).lower()
        self.assertNotIn("velocity", report_text)
        self.assertNotIn("pressure", report_text)
        self.assertNotIn("flow", report_text)

    def test_nozzle_graded_grid_can_add_tail_refinement_region(self) -> None:
        base = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.05, -0.05, 0.90),
            fluid_bounds_max_m=(0.05, 0.05, 1.04),
            grid_nodes=(40, 40, 56),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.039,
            chamber_z_min_m=1.0,
            chamber_z_max_m=1.03,
            nozzle_radius_m=0.003,
            nozzle_z_max_m=1.0,
            downstream_z_m=0.94,
            nozzle_taper_enabled=True,
            nozzle_taper_length_m=0.06,
            nozzle_taper_inlet_radius_m=0.039,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
        )
        tail_region = RefinementRegion(
            bounds_min_m=(-0.041, -0.016, 0.952),
            bounds_max_m=(0.043, 0.018, 1.026),
            target_spacing_m=1.5e-3,
        )

        spec = spec_with_nozzle_graded_grid(
            base,
            target_spacing_m=6.0e-4,
            farfield_spacing_m=3.0e-3,
            max_growth_ratio=1.2,
            extra_refinement_regions=(tail_region,),
        )

        self.assertIsNotNone(spec.graded_grid)
        self.assertEqual(len(spec.graded_grid.refinement_regions), 2)
        nozzle_region, actual_tail_region = spec.graded_grid.refinement_regions
        self.assertEqual(actual_tail_region, tail_region)
        self.assertLessEqual(nozzle_region.bounds_min_m[0], -0.039)
        self.assertGreaterEqual(nozzle_region.bounds_max_m[0], 0.039)
        grid = build_graded_grid(spec.graded_grid)
        self.assertEqual(spec.grid_nodes, grid.grid_nodes)

    def test_nozzle_taper_is_geometry_only_and_reports_radius_profile(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.12, -0.12, 0.0),
            fluid_bounds_max_m=(0.12, 0.12, 0.40),
            grid_nodes=(8, 8, 8),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.12,
            chamber_z_min_m=0.30,
            chamber_z_max_m=0.38,
            nozzle_radius_m=0.03,
            nozzle_length_m=0.12,
            nozzle_z_max_m=0.34,
            downstream_z_m=0.10,
        )

        tapered = spec_with_nozzle_taper(
            spec,
            taper_length_m=0.10,
            inlet_radius_m=0.09,
        )
        report = reduced_water_geometry_report(tapered)

        self.assertTrue(report["nozzle_taper_enabled"])
        self.assertAlmostEqual(report["nozzle_taper_start_z_m"], 0.20)
        self.assertAlmostEqual(report["nozzle_taper_end_z_m"], 0.30)
        self.assertAlmostEqual(report["nozzle_throat_radius_m"], 0.03)
        self.assertAlmostEqual(nozzle_radius_at_z_m(tapered, 0.20), 0.03)
        self.assertAlmostEqual(nozzle_radius_at_z_m(tapered, 0.25), 0.06)
        self.assertAlmostEqual(report["nozzle_radius_at_taper_mid_m"], 0.06)
        self.assertNotIn("velocity", json.dumps(report).lower())
        self.assertNotIn("pressure", json.dumps(report).lower())
        self.assertNotIn("flow", json.dumps(report).lower())

    def test_reduced_obstacle_marking_applies_nozzle_taper_without_flow_boundary(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(-0.10, -0.10, 0.0),
            bounds_max_m=(0.10, 0.10, 0.40),
            grid_nodes=(5, 5, 8),
        )
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=grid.bounds_min_m,
            fluid_bounds_max_m=grid.bounds_max_m,
            grid_nodes=grid.grid_nodes,
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.09,
            chamber_z_min_m=0.30,
            chamber_z_max_m=0.36,
            nozzle_radius_m=0.01,
            nozzle_z_max_m=0.34,
            outlet_plume_radius_m=0.01,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
            downstream_z_m=0.10,
            nozzle_taper_enabled=True,
            nozzle_taper_length_m=0.10,
            nozzle_taper_inlet_radius_m=0.05,
            cartesian_grid=grid,
        )
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        simulator.mark_reduced_squid_water_domain()
        obstacle = simulator.fluid.obstacle.to_numpy()

        x_taper_open = min(
            range(len(grid.cell_centers_x_m)),
            key=lambda index: abs(grid.cell_centers_x_m[index] - 0.04),
        )
        x_taper_closed = min(
            range(len(grid.cell_centers_x_m)),
            key=lambda index: abs(grid.cell_centers_x_m[index] - 0.08),
        )
        y_center = min(
            range(len(grid.cell_centers_y_m)),
            key=lambda index: abs(grid.cell_centers_y_m[index]),
        )
        z_before_taper = min(
            range(len(grid.cell_centers_z_m)),
            key=lambda index: abs(grid.cell_centers_z_m[index] - 0.125),
        )
        z_in_taper = min(
            range(len(grid.cell_centers_z_m)),
            key=lambda index: abs(grid.cell_centers_z_m[index] - 0.225),
        )

        self.assertEqual(obstacle[x_taper_open, y_center, z_in_taper], 0)
        self.assertEqual(obstacle[x_taper_closed, y_center, z_in_taper], 1)
        self.assertEqual(obstacle[x_taper_open, y_center, z_before_taper], 1)

    def test_reduced_obstacle_intersection_preserves_cad_solid_cells(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(-0.10, -0.10, 0.0),
            bounds_max_m=(0.10, 0.10, 0.40),
            grid_nodes=(5, 5, 8),
        )
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=grid.bounds_min_m,
            fluid_bounds_max_m=grid.bounds_max_m,
            grid_nodes=grid.grid_nodes,
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.09,
            chamber_z_min_m=0.30,
            chamber_z_max_m=0.36,
            nozzle_radius_m=0.01,
            nozzle_z_max_m=0.34,
            outlet_plume_radius_m=0.01,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
            downstream_z_m=0.10,
            nozzle_taper_enabled=True,
            nozzle_taper_length_m=0.10,
            nozzle_taper_inlet_radius_m=0.05,
            cartesian_grid=grid,
        )
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        x_center = min(
            range(len(grid.cell_centers_x_m)),
            key=lambda index: abs(grid.cell_centers_x_m[index]),
        )
        y_center = min(
            range(len(grid.cell_centers_y_m)),
            key=lambda index: abs(grid.cell_centers_y_m[index]),
        )
        z_chamber = min(
            range(len(grid.cell_centers_z_m)),
            key=lambda index: abs(grid.cell_centers_z_m[index] - 0.325),
        )
        z_nozzle = min(
            range(len(grid.cell_centers_z_m)),
            key=lambda index: abs(grid.cell_centers_z_m[index] - 0.225),
        )
        x_outside = min(
            range(len(grid.cell_centers_x_m)),
            key=lambda index: abs(grid.cell_centers_x_m[index] - 0.08),
        )
        source_obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        source_obstacle[x_center, y_center, z_chamber] = 1
        simulator.fluid.obstacle.from_numpy(source_obstacle)

        simulator.intersect_current_obstacles_with_reduced_squid_water_domain()
        obstacle = simulator.fluid.obstacle.to_numpy()

        self.assertEqual(obstacle[x_center, y_center, z_chamber], 1)
        self.assertEqual(obstacle[x_center, y_center, z_nozzle], 0)
        self.assertEqual(obstacle[x_outside, y_center, z_nozzle], 1)

    def test_coarse_center_missed_nozzle_remains_connected_and_projects_source_to_outlet(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(-0.02, -0.02, 0.0),
            bounds_max_m=(0.02, 0.02, 0.08),
            grid_nodes=(4, 4, 8),
        )
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=grid.bounds_min_m,
            fluid_bounds_max_m=grid.bounds_max_m,
            grid_nodes=grid.grid_nodes,
            dt_s=1.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.018,
            chamber_z_min_m=0.065,
            chamber_z_max_m=0.080,
            nozzle_radius_m=0.002,
            nozzle_z_max_m=0.070,
            outlet_plume_radius_m=0.002,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
            downstream_z_m=0.020,
            downstream_farfield_open_enabled=True,
            downstream_farfield_open_z_max_m=0.020,
            nozzle_taper_enabled=True,
            nozzle_taper_length_m=0.045,
            nozzle_taper_inlet_radius_m=0.018,
            cartesian_grid=grid,
        )
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        simulator.mark_reduced_squid_water_domain()
        obstacle = simulator.fluid.obstacle.to_numpy()
        connectivity = reduced_active_water_connectivity(
            spec,
            obstacle_cell_count=int(np.sum(obstacle)),
            obstacle_mask=obstacle,
        )

        self.assertTrue(connectivity["connectivity_passed"])
        self.assertEqual(connectivity["trapped_active_cell_count"], 0)

        source_total_m3s = 1.0e-7
        source = np.zeros(spec.grid_nodes, dtype=np.float32)
        z_centers = np.asarray(grid.cell_centers_z_m, dtype=np.float64)
        chamber_indices = np.argwhere(
            (obstacle == 0)
            & (z_centers[np.newaxis, np.newaxis, :] > 0.065)
        )
        self.assertGreater(len(chamber_indices), 0)
        i, j, k = (int(value) for value in chamber_indices[0])
        cell_volume_m3 = (
            grid.cell_widths_x_m[i]
            * grid.cell_widths_y_m[j]
            * grid.cell_widths_z_m[k]
        )
        source[i, j, k] = source_total_m3s / cell_volume_m3
        simulator.fluid.volume_source_s.from_numpy(source)

        simulator.fluid.project(
            iterations=3000,
            pressure_outlet_zmin=True,
            dt_s=spec.dt_s,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-8,
        )
        report = simulator.fluid.pressure_outlet_fv_flux_report(dt_s=spec.dt_s)

        self.assertGreater(report["zmin_velocity_outlet_flux_m3s"], 0.0)
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=0.15,
        )

    def test_reduced_obstacle_marking_uses_nonuniform_cell_centers(self) -> None:
        grid = self._nonuniform_reduced_grid()
        spec = self._nonuniform_reduced_spec(grid)
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        simulator.mark_reduced_squid_water_domain()
        obstacle = simulator.fluid.obstacle.to_numpy()

        expected = np.ones(grid.grid_nodes, dtype=np.int32)
        for i, x in enumerate(grid.cell_centers_x_m):
            for j, y in enumerate(grid.cell_centers_y_m):
                chamber_footprint = self._cell_disk_intersects(
                    x_m=x,
                    y_m=y,
                    width_x_m=grid.cell_widths_x_m[i],
                    width_y_m=grid.cell_widths_y_m[j],
                    center_x_m=spec.monitor_center_x_m,
                    center_y_m=spec.monitor_center_y_m,
                    radius_m=spec.chamber_radius_m,
                )
                nozzle_footprint = self._cell_disk_intersects(
                    x_m=x,
                    y_m=y,
                    width_x_m=grid.cell_widths_x_m[i],
                    width_y_m=grid.cell_widths_y_m[j],
                    center_x_m=spec.monitor_center_x_m,
                    center_y_m=spec.monitor_center_y_m,
                    radius_m=spec.nozzle_radius_m,
                )
                outlet_footprint = self._cell_disk_intersects(
                    x_m=x,
                    y_m=y,
                    width_x_m=grid.cell_widths_x_m[i],
                    width_y_m=grid.cell_widths_y_m[j],
                    center_x_m=spec.monitor_center_x_m,
                    center_y_m=spec.monitor_center_y_m,
                    radius_m=spec.outlet_plume_radius_m,
                )
                for k, z in enumerate(grid.cell_centers_z_m):
                    chamber = (
                        chamber_footprint
                        and self._cell_z_intersects(
                            z_m=z,
                            width_z_m=grid.cell_widths_z_m[k],
                            lower_m=spec.chamber_z_min_m,
                            upper_m=spec.chamber_z_max_m,
                        )
                    )
                    nozzle = nozzle_footprint and self._cell_z_intersects(
                        z_m=z,
                        width_z_m=grid.cell_widths_z_m[k],
                        lower_m=spec.downstream_z_m,
                        upper_m=spec.nozzle_z_max_m,
                    )
                    outlet_plume = (
                        outlet_footprint
                        and self._cell_z_intersects(
                            z_m=z,
                            width_z_m=grid.cell_widths_z_m[k],
                            lower_m=spec.fluid_bounds_min_m[2],
                            upper_m=spec.downstream_z_m,
                        )
                    )
                    expected[i, j, k] = 0 if chamber or nozzle or outlet_plume else 1

        np.testing.assert_array_equal(obstacle, expected)
        self.assertEqual(simulator.fluid.obstacle_cell_count(), int(np.sum(expected)))

    def test_reduced_section_sampling_uses_nonuniform_cell_centers_and_areas(self) -> None:
        grid = self._nonuniform_reduced_grid()
        spec = self._nonuniform_reduced_spec(grid)
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros(grid.grid_nodes + (3,), dtype=np.float32)
        for k in range(grid.grid_nodes[2]):
            velocity[:, :, k, 2] = -float(k + 1)
        simulator.fluid.velocity.from_numpy(velocity)

        report = simulator.sample_after_projection({"l2": 0.0, "max_abs": 0.0})

        self.assertEqual(simulator.last_sample_report_host_reads, 1)

        def expected_section_area_and_count(radius_m: float) -> tuple[float, int]:
            area_m2 = 0.0
            count = 0
            for i, x_m in enumerate(grid.cell_centers_x_m):
                for j, y_m in enumerate(grid.cell_centers_y_m):
                    fraction = self._section_area_fraction(
                        x_m=x_m,
                        y_m=y_m,
                        width_x_m=grid.cell_widths_x_m[i],
                        width_y_m=grid.cell_widths_y_m[j],
                        center_x_m=spec.monitor_center_x_m,
                        center_y_m=spec.monitor_center_y_m,
                        radius_m=radius_m,
                    )
                    if fraction > 0.0:
                        area_m2 += (
                            grid.cell_widths_x_m[i]
                            * grid.cell_widths_y_m[j]
                            * fraction
                        )
                        count += 1
            return area_m2, count

        lip_area_m2, lip_count = expected_section_area_and_count(spec.monitor_radius_m)
        plume_area_m2, plume_count = expected_section_area_and_count(spec.outlet_plume_radius_m)
        self.assertEqual(report["lip_sample_count"], lip_count)
        self.assertEqual(report["outlet_sample_count"], plume_count)
        self.assertEqual(report["downstream_sample_count"], plume_count)
        self.assertAlmostEqual(report["lip_flow_z_m3s"], -lip_area_m2, delta=1.0e-7)
        self.assertAlmostEqual(report["outlet_flow_z_m3s"], -2.0 * plume_area_m2, delta=1.0e-7)
        self.assertAlmostEqual(report["downstream_flow_z_m3s"], -3.0 * plume_area_m2, delta=1.0e-7)
        self.assertAlmostEqual(report["outlet_flow_negative_z_m3s"], 2.0 * plume_area_m2, delta=1.0e-7)
        self.assertLess(report["outlet_sample_count"], report["lip_sample_count"])

    def test_reduced_section_sampling_ignores_obstacle_cells_for_cfl_and_flux(self) -> None:
        grid = self._nonuniform_reduced_grid()
        spec = self._nonuniform_reduced_spec(grid)
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros(grid.grid_nodes + (3,), dtype=np.float32)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        active_cell = (2, 2, 0)
        obstacle_cell = (1, 1, 0)
        velocity[active_cell + (2,)] = -1.0
        velocity[obstacle_cell + (2,)] = -100.0
        obstacle[obstacle_cell] = 1
        simulator.fluid.velocity.from_numpy(velocity)
        simulator.fluid.obstacle.from_numpy(obstacle)

        report = simulator.sample_after_projection({"l2": 0.0, "max_abs": 0.0})

        active_area_m2 = (
            grid.cell_widths_x_m[active_cell[0]]
            * grid.cell_widths_y_m[active_cell[1]]
            * self._section_area_fraction(
                x_m=grid.cell_centers_x_m[active_cell[0]],
                y_m=grid.cell_centers_y_m[active_cell[1]],
                width_x_m=grid.cell_widths_x_m[active_cell[0]],
                width_y_m=grid.cell_widths_y_m[active_cell[1]],
                center_x_m=spec.monitor_center_x_m,
                center_y_m=spec.monitor_center_y_m,
                radius_m=spec.monitor_radius_m,
            )
        )
        self.assertAlmostEqual(report["max_fluid_speed_mps"], 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report["lip_flow_z_m3s"], -active_area_m2, delta=1.0e-7)

    def test_pressure_solver_auto_uses_fv_cg_for_graded_grid(self) -> None:
        self.assertEqual(resolve_pressure_solver("auto", graded_grid_enabled=False), "fv_multigrid")
        self.assertEqual(resolve_pressure_solver("auto", graded_grid_enabled=True), "fv_cg")
        self.assertEqual(
            resolve_pressure_solver(
                "auto",
                graded_grid_enabled=False,
                fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
            ),
            "fv_cg",
        )
        self.assertEqual(resolve_pressure_solver("fv_jacobi", graded_grid_enabled=True), "fv_jacobi")
        self.assertEqual(resolve_pressure_solver("fv_multigrid", graded_grid_enabled=True), "fv_multigrid")
        self.assertEqual(resolve_pressure_solver("fv_cg", graded_grid_enabled=True), "fv_cg")
        with self.assertRaisesRegex(ValueError, "requires an FV pressure solver"):
            resolve_pressure_solver("jacobi", graded_grid_enabled=True)
        with self.assertRaisesRegex(ValueError, "unsupported pressure solver"):
            resolve_pressure_solver("bad", graded_grid_enabled=False)

    def test_graded_grid_rejects_uniform_divergence_cleanup(self) -> None:
        self.assertEqual(resolve_divergence_cleanup_iterations(8, graded_grid_enabled=False), 8)
        self.assertEqual(resolve_divergence_cleanup_iterations(0, graded_grid_enabled=True), 0)
        with self.assertRaisesRegex(ValueError, "requires --divergence-cleanup-iterations 0"):
            resolve_divergence_cleanup_iterations(1, graded_grid_enabled=True)

    def test_graded_nozzle_grid_report_resolves_nozzle_diameter(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="synthetic",
            fluid_bounds_min_m=(-0.09, -0.044, 0.90),
            fluid_bounds_max_m=(0.05, 0.096, 1.06),
            grid_nodes=(56, 56, 64),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=0.00105,
        )
        graded = spec_with_nozzle_graded_grid(spec)
        report = fluid_grid_resolution_report(graded)

        self.assertTrue(report["graded_enabled"])
        self.assertGreaterEqual(report["nozzle_diameter_cells_min"], 10)
        self.assertTrue(report["nozzle_resolves_diameter_10_cells"])
        self.assertLessEqual(max(report["max_adjacent_spacing_ratio"]), 1.2 + 1.0e-6)
        self.assertLessEqual(max(report["nozzle_min_cell_width_m"][:2]), spec.nozzle_radius_m / 5.0)

    def test_graded_grid_fluid_substeps_resolve_finest_cells_at_half_cfl(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="synthetic",
            fluid_bounds_min_m=(-0.09, -0.044, 0.90),
            fluid_bounds_max_m=(0.05, 0.096, 1.06),
            grid_nodes=(56, 56, 64),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=0.00105,
        )
        graded = spec_with_nozzle_graded_grid(
            spec,
            target_spacing_m=6.0e-4,
            farfield_spacing_m=3.0e-3,
            max_growth_ratio=1.2,
        )

        self.assertEqual(effective_fluid_substeps_for_grid(graded, 1), 12)
        self.assertEqual(effective_fluid_substeps_for_grid(graded, 16), 16)

        time_refined = replace(graded, dt_s=graded.dt_s * 0.25, base_dt_s=graded.dt_s)
        self.assertEqual(effective_fluid_substeps_for_grid(time_refined, 1), 3)

    def test_pressure_projection_budget_counts_trials_and_accepted_step(self) -> None:
        coupled_budget = pressure_projection_budget_report(
            fluid_substeps=1,
            ibm_correction_iterations=2,
            fsi_coupling_iterations=3,
            projection_iterations=3000,
            fsi_coupling_enabled=True,
        )

        self.assertEqual(coupled_budget["fluid_step_evaluations_per_physical_step_max"], 4)
        self.assertEqual(coupled_budget["pressure_project_calls_per_physical_step_max"], 8)
        self.assertEqual(coupled_budget["full_report_pressure_project_calls_per_step"], 2)
        self.assertEqual(coupled_budget["trial_pressure_project_calls_per_step_max"], 6)
        self.assertEqual(coupled_budget["cg_iteration_budget_per_physical_step_max"], 24000)

        uncoupled_budget = pressure_projection_budget_report(
            fluid_substeps=12,
            ibm_correction_iterations=2,
            fsi_coupling_iterations=1,
            projection_iterations=3000,
            fsi_coupling_enabled=False,
        )

        self.assertEqual(uncoupled_budget["fluid_step_evaluations_per_physical_step_max"], 1)
        self.assertEqual(uncoupled_budget["pressure_project_calls_per_physical_step_max"], 24)
        self.assertEqual(uncoupled_budget["trial_pressure_project_calls_per_step_max"], 0)
        self.assertEqual(uncoupled_budget["cg_iteration_budget_per_physical_step_max"], 72000)

    def test_runtime_budget_report_extrapolates_from_measured_step_wall_time(self) -> None:
        report = runtime_budget_report(
            timing_summary={"mean_step_wall_time_s": 2.5, "max_step_wall_time_s": 3.0},
            requested_steps=200,
            completed_steps=20,
            full_pressure_waveform_steps=4000,
        )

        self.assertEqual(report["completed_steps"], 20)
        self.assertAlmostEqual(report["measured_mean_step_wall_time_s"], 2.5)
        self.assertAlmostEqual(report["estimated_requested_run_wall_time_s"], 500.0)
        self.assertAlmostEqual(report["estimated_requested_remaining_wall_time_s"], 450.0)
        self.assertAlmostEqual(report["estimated_full_pressure_waveform_wall_time_s"], 10000.0)
        self.assertAlmostEqual(report["estimated_full_pressure_waveform_remaining_wall_time_s"], 9950.0)
        self.assertEqual(report["basis"], "measured_mean_step_wall_time_s")
        self.assertIn("does not change", report["note"])

    def test_runtime_budget_report_keeps_warmup_excluded_steady_state_estimate(self) -> None:
        report = runtime_budget_report(
            timing_summary={
                "mean_step_wall_time_s": 47.5,
                "max_step_wall_time_s": 101.0,
                "steady_state_mean_step_wall_time_s": 20.875,
                "steady_state_step_wall_time_sample_count": 2,
                "steady_state_warmup_excluded_steps": 1,
            },
            requested_steps=3,
            completed_steps=3,
            full_pressure_waveform_steps=145455,
        )

        self.assertAlmostEqual(report["measured_mean_step_wall_time_s"], 47.5)
        self.assertTrue(report["steady_state_estimate_available"])
        self.assertEqual(report["steady_state_step_wall_time_sample_count"], 2)
        self.assertEqual(report["steady_state_warmup_excluded_steps"], 1)
        self.assertAlmostEqual(report["steady_state_mean_step_wall_time_s"], 20.875)
        self.assertAlmostEqual(
            report["steady_state_estimated_full_pressure_waveform_wall_time_s"],
            20.875 * 145455,
        )
        self.assertAlmostEqual(
            report["steady_state_estimated_full_pressure_waveform_remaining_wall_time_s"],
            20.875 * (145455 - 3),
        )
        self.assertIn("warmup", report["steady_state_note"])

    def test_preflight_only_writes_graded_grid_resolution_without_fsi_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "preflight"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--use-graded-grid",
                    "--use-nozzle-taper",
                    "--divergence-cleanup-iterations",
                    "0",
                    "--preflight-only",
                ]
            )

            summary = run(args)

            self.assertTrue(summary["preflight_only"])
            self.assertEqual(summary["steps"], 4000)
            self.assertFalse(summary["steps_explicit"])
            self.assertEqual(summary["pressure_solver"], "fv_cg")
            self.assertEqual(summary["cg_preconditioner"], "auto")
            self.assertEqual(
                summary["fsi_coupling_mode"],
                FSI_COUPLING_MODE_HIBM_MPM_SHARP,
            )
            self.assertEqual(
                summary["fsi_coupling_mode_report"],
                fsi_coupling_mode_report(FSI_COUPLING_MODE_HIBM_MPM_SHARP),
            )
            self.assertTrue(summary["fsi_coupling_mode_report"]["paper_hibm_mpm"])
            self.assertIsNone(summary["effective_multigrid_cycles"])
            self.assertEqual(summary["fluid_substeps"], 12)
            self.assertAlmostEqual(summary["fluid_substep_dt_s"], 5.0e-4 / 12.0)
            self.assertEqual(
                summary["pressure_projection_budget"][
                    "pressure_project_calls_per_physical_step_max"
                ],
                24,
            )
            self.assertEqual(
                summary["pressure_projection_budget"][
                    "cg_iteration_budget_per_physical_step_max"
                ],
                72000,
            )
            self.assertEqual(summary["summary_json"], str(output_dir.resolve() / "preflight_summary.json"))
            self.assertFalse(summary["interface_reaction_passivity_limit"])
            self.assertTrue(summary["interface_reaction_aitken"])
            self.assertIsNone(summary["fluid_grid_spacing_m"])
            self.assertLessEqual(max(summary["fluid_grid_min_spacing_m"][:2]), 6.0e-4)
            self.assertGreater(min(summary["fluid_grid_max_spacing_m"]), 6.0e-4)
            self.assertTrue(summary["fluid_grid_resolution"]["nozzle_resolves_diameter_10_cells"])
            self.assertFalse(summary["region14_aperture_carve_enabled"])
            self.assertFalse(summary["region14_aperture_geometry"]["available"])
            self.assertTrue(summary["reduced_water_geometry"]["nozzle_taper_enabled"])
            self.assertGreater(
                summary["reduced_water_geometry"]["nozzle_taper_inlet_radius_m"],
                summary["reduced_water_geometry"]["nozzle_throat_radius_m"],
            )
            process = json.loads((output_dir / "run_process.json").read_text(encoding="utf-8"))
            written_summary = json.loads((output_dir / "preflight_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(process["status"], "preflight_complete")
            self.assertEqual(written_summary["summary_json"], summary["summary_json"])
            self.assertEqual(
                written_summary["fsi_coupling_mode_report"],
                summary["fsi_coupling_mode_report"],
            )
            self.assertIsNone(written_summary["fluid_grid_spacing_m"])
            self.assertIn("region14_aperture_geometry", written_summary)
            self.assertTrue(written_summary["reduced_water_geometry"]["nozzle_taper_enabled"])
            self.assertTrue(written_summary["fluid_grid_resolution"]["nozzle_resolves_diameter_10_cells"])

    def test_preflight_reports_real_step_cad_provenance_without_relabeling_cached_stl(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('flange',#20);
#11=MANIFOLD_SOLID_BREP('membrane',#30);
#12=MANIFOLD_SOLID_BREP('noozle',#40);
#13=MANIFOLD_SOLID_BREP('chamber',#50);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cached_stl = temp_path / "current_sim_step_4body_surface_mesh.stl"
            cached_stl.write_text("solid cached\nendsolid cached\n", encoding="utf-8")
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "mesh_format": "step",
                        "mesh_path": str(cached_stl),
                        "surface_mesh_cache_path": str(cached_stl),
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temp_path / "out"

            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--cad-step-path",
                    str(cad_path),
                    "--output-dir",
                    str(output_dir),
                    "--preflight-only",
                ]
            )
            summary = run(args)

            cad_report = summary["cad_provenance"]
            self.assertEqual(
                cad_report["cad_step_brep_names"],
                ["flange", "membrane", "noozle", "chamber"],
            )
            self.assertEqual(cad_report["source_config_mesh_suffix"], ".stl")
            self.assertFalse(cad_report["direct_cad_step_binding"])
            self.assertFalse(summary["real_cad_step_direct_binding"])
            self.assertEqual(summary["real_cad_step_path"], str(cad_path.resolve()))

    def test_required_real_step_cad_rejects_cached_stl_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('chamber',#20);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cached_stl = temp_path / "cached.stl"
            cached_stl.write_text("solid cached\nendsolid cached\n", encoding="utf-8")
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "mesh_format": "step",
                        "mesh_path": str(cached_stl),
                        "surface_mesh_cache_path": str(cached_stl),
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--cad-step-path",
                    str(cad_path),
                    "--require-real-cad-step",
                    "--output-dir",
                    str(temp_path / "out"),
                    "--preflight-only",
                ]
            )

            with self.assertRaisesRegex(ValueError, "verified real STEP CAD binding"):
                run(args)

    def test_required_real_step_cad_accepts_verified_step_derived_surface_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cad_path = temp_path / "sim.STEP"
            cad_path.write_text(
                """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=SI_UNIT(.MILLI.,.METRE.);
#10=MANIFOLD_SOLID_BREP('chamber',#20);
ENDSEC;
END-ISO-10303-21;
""",
                encoding="utf-8",
            )
            cache_path = temp_path / "sim.surface_mesh.stl"
            cache_path.write_text("solid derived\nendsolid derived\n", encoding="utf-8")
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "mesh_format": "step",
                        "mesh_path": str(cad_path),
                        "surface_mesh_cache_path": str(cache_path),
                        "mesh_import": {
                            "source_step_path": str(cad_path),
                            "source_step_sha256": hashlib.sha256(
                                cad_path.read_bytes()
                            ).hexdigest(),
                            "surface_mesh_cache_path": str(cache_path),
                            "surface_mesh_cache_sha256": hashlib.sha256(
                                cache_path.read_bytes()
                            ).hexdigest(),
                        },
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--cad-step-path",
                    str(cad_path),
                    "--require-real-cad-step",
                    "--output-dir",
                    str(temp_path / "out"),
                    "--preflight-only",
                ]
            )

            summary = run(args)

            self.assertTrue(summary["real_cad_step_binding"])
            self.assertFalse(summary["real_cad_step_direct_binding"])
            self.assertTrue(summary["real_cad_step_derived_surface_mesh_binding"])

    def test_preflight_pressure_schedule_cli_override_is_boundary_input_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "out"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--pressure-t1-s",
                    "0.3",
                    "--pressure-p1-pa",
                    "8000",
                    "--preflight-only",
                ]
            )

            summary = run(args)

            pressure_input = summary["pressure_schedule_input"]
            self.assertTrue(pressure_input["cli_override_applied"])
            self.assertTrue(pressure_input["boundary_condition_input_only"])
            self.assertEqual(pressure_input["overrides"], {"pressure_t1_s": 0.3, "pressure_p1_pa": 8000.0})
            self.assertAlmostEqual(summary["spec"]["pressure_t1_s"], 0.3)
            self.assertAlmostEqual(summary["spec"]["pressure_p1_pa"], 8000.0)
            self.assertIn("tail force", pressure_input["computed_response_fields"][0])

    def test_invalid_pressure_schedule_cli_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(temp_path / "out"),
                    "--pressure-t1-s",
                    "0",
                    "--preflight-only",
                ]
            )

            with self.assertRaisesRegex(ValueError, "pressure schedule times"):
                run(args)

    def test_graded_grid_preflight_defaults_divergence_cleanup_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "preflight_default_cleanup"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--use-graded-grid",
                    "--use-nozzle-taper",
                    "--preflight-only",
                ]
            )

            summary = run(args)

            self.assertTrue(summary["preflight_only"])
            self.assertEqual(summary["pressure_solver"], "fv_cg")
            self.assertEqual(summary["divergence_cleanup_iterations"], 0)

    def test_preflight_records_shell_surface_mass_scaling_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "preflight_mass_budget"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--membrane-thickness-scale",
                    "2",
                    "--solid-density-scale",
                    "3",
                    "--pressure-solver",
                    "fv_cg",
                    "--interface-reaction-robin-matrix-impedance-ns-m",
                    "125000",
                    "--preflight-only",
                ]
            )

            summary = run(args)

            self.assertTrue(summary["preflight_only"])
            self.assertAlmostEqual(summary["membrane_thickness_scale"], 2.0)
            self.assertAlmostEqual(summary["solid_density_scale"], 3.0)
            self.assertAlmostEqual(summary["solid_density_kgm3"], 3120.0)
            self.assertAlmostEqual(
                summary["interface_reaction_robin_matrix_impedance_ns_m"],
                125000.0,
            )
            self.assertAlmostEqual(
                summary["spec"]["main_membrane_thickness_m"],
                6.0e-3,
            )
            self.assertAlmostEqual(
                summary["solid_surface_mass_budget"]["main_surface_mass_scale"],
                6.0,
            )
            written_summary = json.loads(
                (output_dir / "preflight_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                written_summary["solid_surface_mass_budget"],
                summary["solid_surface_mass_budget"],
            )

    def test_run_process_marks_failed_when_run_raises_after_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "failed_run"
            missing_source_config = temp_path / "missing_source.json"
            args = parse_args(
                [
                    "--source-config",
                    str(missing_source_config),
                    "--output-dir",
                    str(output_dir),
                    "--preflight-only",
                ]
            )

            with self.assertRaises(FileNotFoundError):
                run(args)

            process = json.loads((output_dir / "run_process.json").read_text(encoding="utf-8"))
            self.assertEqual(process["status"], "failed")
            self.assertEqual(process["error_type"], "FileNotFoundError")
            self.assertIn("source config not found", process["error"])

    def test_run_process_marks_failed_for_early_argument_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "early_failed_run"
            source_config = temp_path / "source.json"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--interface-reaction-relaxation",
                    "2.0",
                    "--preflight-only",
                ]
            )

            with self.assertRaisesRegex(ValueError, "interface-reaction-relaxation"):
                run(args)

            process = json.loads((output_dir / "run_process.json").read_text(encoding="utf-8"))
            self.assertEqual(process["status"], "failed")
            self.assertEqual(process["error_type"], "ValueError")
            self.assertIn("interface-reaction-relaxation", process["error"])

    def test_nozzle_graded_grid_refines_full_taper_inlet_radius(self) -> None:
        base = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.05, -0.05, 0.90),
            fluid_bounds_max_m=(0.05, 0.05, 1.04),
            grid_nodes=(40, 40, 56),
            dt_s=5.0e-4,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            chamber_radius_m=0.039,
            chamber_z_min_m=1.0,
            chamber_z_max_m=1.03,
            nozzle_radius_m=0.003,
            nozzle_z_max_m=1.0,
            downstream_z_m=0.94,
            nozzle_taper_enabled=True,
            nozzle_taper_length_m=0.06,
            nozzle_taper_inlet_radius_m=0.039,
            monitor_center_x_m=0.0,
            monitor_center_y_m=0.0,
        )

        spec = spec_with_nozzle_graded_grid(
            base,
            target_spacing_m=6.0e-4,
            farfield_spacing_m=3.0e-3,
            max_growth_ratio=1.2,
        )

        region = spec.graded_grid.refinement_regions[0]
        self.assertLessEqual(region.bounds_min_m[0], -0.039)
        self.assertGreaterEqual(region.bounds_max_m[0], 0.039)
        self.assertLessEqual(region.bounds_min_m[1], -0.039)
        self.assertGreaterEqual(region.bounds_max_m[1], 0.039)

    def test_graded_runtime_passes_stable_fluid_config_to_projected_ibm(self) -> None:
        class FakeScalarField:
            def __init__(self, value: float = 0.0) -> None:
                self.value = float(value)

            def __getitem__(self, _key):
                return self.value

            def __setitem__(self, _key, value: float) -> None:
                self.value = float(value)

        class FakeVector3:
            def __init__(self, value) -> None:
                self._value = tuple(float(component) for component in value)
                self.x = self._value[0]
                self.y = self._value[1]
                self.z = self._value[2]

            def __getitem__(self, index: int) -> float:
                return self._value[index]

        class FakeVectorField:
            def __init__(self, value=(0.0, 0.0, 0.0)) -> None:
                self.value = tuple(float(component) for component in value)

            def __getitem__(self, _key):
                return FakeVector3(self.value)

            def __setitem__(self, _key, value) -> None:
                self.value = tuple(float(component) for component in value)

        class FakeArrayField:
            def __init__(self, values: np.ndarray) -> None:
                self.values = np.asarray(values)

            def to_numpy(self) -> np.ndarray:
                return np.array(self.values)

        class FakeFluid:
            def __init__(self, grid: CartesianGrid) -> None:
                self.grid = grid
                self.nx, self.ny, self.nz = grid.grid_nodes
                self.velocity = FakeArrayField(
                    np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                )
                self.velocity_prev = FakeArrayField(
                    np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                )
                self.pressure = FakeArrayField(
                    np.zeros(grid.grid_nodes, dtype=np.float32)
                )
                self.obstacle = FakeArrayField(
                    np.zeros(grid.grid_nodes, dtype=np.int32)
                )

            def obstacle_cell_count(self) -> int:
                return 0

            def save_state(self) -> None:
                return None

            def restore_state(self) -> None:
                return None

        sample_calls: list[None] = []
        diagnose_calls: list[None] = []
        captured_primary_offset_z_m: list[float] = []

        class FakeSimulator:
            def __init__(self, spec: SquidReducedSpec, *, runtime) -> None:
                self.spec = spec
                self.fluid = FakeFluid(build_graded_grid(spec.graded_grid) if spec.graded_grid is not None else spec.cartesian_grid)
                self.time_s = FakeScalarField(0.0)
                self.pressure_load_pa = FakeScalarField(0.0)
                self.hydraulic_pressure_pa = FakeScalarField(0.0)
                self.main_w_m = FakeScalarField(0.0)
                self.main_v_mps = FakeScalarField(0.0)
                self.tail_w_m = FakeScalarField(0.0)
                self.tail_v_mps = FakeScalarField(0.0)
                self.volume_flux_m3s = FakeScalarField(1.0e-6)
                self.nozzle_velocity_z_mps = FakeScalarField(-1.0e-3)
                self.max_speed_mps = FakeScalarField(0.0)
                self.lip_flow_z_m3s = FakeScalarField(-1.0e-6)
                self.outlet_flow_z_m3s = FakeScalarField(-1.0e-6)
                self.downstream_flow_z_m3s = FakeScalarField(-1.0e-6)
                self.lip_sample_count = FakeScalarField(1.0)
                self.outlet_sample_count = FakeScalarField(1.0)
                self.downstream_sample_count = FakeScalarField(1.0)
                self.primary_interface_reaction_force_n = FakeVectorField()
                self.secondary_interface_reaction_force_n = FakeVectorField()

            def mark_reduced_squid_water_domain(self) -> None:
                return None

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
                self.time_s[None] = time_s
                self.pressure_load_pa[None] = pressure_pa
                self.hydraulic_pressure_pa[None] = hydraulic_pressure_pa
                self.main_w_m[None] = main_displacement_z_m
                self.main_v_mps[None] = main_velocity_z_mps
                self.tail_w_m[None] = tail_displacement_z_m
                self.tail_v_mps[None] = tail_velocity_z_mps
                self.volume_flux_m3s[None] = volume_flux_m3s
                self.nozzle_velocity_z_mps[None] = nozzle_velocity_z_mps

            def set_interface_reaction(self, *, primary_force_n, secondary_force_n) -> None:
                self.primary_interface_reaction_force_n[None] = primary_force_n
                self.secondary_interface_reaction_force_n[None] = secondary_force_n

            def save_reduced_state(self) -> None:
                return None

            def restore_reduced_state(self) -> None:
                return None

            def sample_after_projection(
                self,
                divergence: dict[str, float],
                *,
                dt_s: float | None = None,
            ) -> dict[str, object]:
                sample_calls.append(None)
                return {
                    "time_s": self.time_s[None],
                    "pressure_load_pa": self.pressure_load_pa[None],
                    "hydraulic_pressure_pa": self.hydraulic_pressure_pa[None],
                    "main_displacement_z_m": self.main_w_m[None],
                    "main_velocity_z_mps": self.main_v_mps[None],
                    "tail_displacement_z_m": self.tail_w_m[None],
                    "tail_velocity_z_mps": self.tail_v_mps[None],
                    "main_interface_reaction_z_n": 0.0,
                    "tail_interface_reaction_z_n": 0.0,
                    "volume_flux_m3s": self.volume_flux_m3s[None],
                    "nozzle_velocity_z_mps": self.nozzle_velocity_z_mps[None],
                    "lip_flow_z_m3s": -1.0e-6,
                    "outlet_flow_z_m3s": -1.0e-6,
                    "downstream_flow_z_m3s": -1.0e-6,
                    "lip_flow_negative_z_m3s": 1.0e-6,
                    "outlet_flow_negative_z_m3s": 1.0e-6,
                    "downstream_flow_negative_z_m3s": 1.0e-6,
                    "lip_sample_count": 1,
                    "outlet_sample_count": 1,
                    "downstream_sample_count": 1,
                    "max_fluid_speed_mps": 0.0,
                    "cfl": 0.0,
                    **divergence_sample_report_fields(divergence),
                }

        class FakeTriDiagnostics:
            def update_region_offsets(self, **_kwargs) -> None:
                captured_primary_offset_z_m.append(float(_kwargs["primary_offset_m"][2]))
                return None

            def diagnose_from_fields(self, *_args, **_kwargs):
                diagnose_calls.append(None)
                zero = (0.0, 0.0, 0.0)
                return SimpleNamespace(
                    pressure_traction_force_n=zero,
                    primary_pressure_traction_force_n=zero,
                    secondary_pressure_traction_force_n=zero,
                    viscous_traction_force_n=zero,
                    primary_viscous_traction_force_n=zero,
                    secondary_viscous_traction_force_n=zero,
                    fluid_stress_traction_force_n=zero,
                    primary_fluid_stress_force_n=zero,
                    secondary_fluid_stress_force_n=zero,
                    primary_fluid_stress_traction_force_n=zero,
                    secondary_fluid_stress_traction_force_n=zero,
                    pressure_traction_abs_force_n=0.0,
                    pressure_traction_area_m2=1.0,
                    pressure_traction_face_count=1,
                    projected_ibm_residual_mps=0.0,
                    projected_ibm_residual_l2_mps=0.0,
                    projected_ibm_sample_count=1,
                    invalid_probe_count=0,
                    valid_probe_fraction=1.0,
                    invalid_probe_area_m2=0.0,
                    invalid_probe_volume_source_m3s=0.0,
                )

        captured_solid_grid_nodes: list[tuple[int, int, int]] = []
        captured_solid_read_report: list[bool] = []
        solid_report_calls: list[None] = []

        class FakeSolidMpm:
            def __init__(self, *_args, **_kwargs) -> None:
                captured_solid_grid_nodes.append(
                    tuple(int(value) for value in _kwargs["grid_nodes"])
                )
                self._report = SimpleNamespace(
                    particle_count=1,
                    active_grid_nodes=1,
                    particle_spacing_m=1.0e-3,
                    grid_spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
                    total_mass_kg=1.0e-6,
                    particle_momentum_kg_mps=(0.0, 0.0, 0.0),
                    grid_momentum_kg_mps=(0.0, 0.0, 0.0),
                    transfer_relative_error=0.0,
                    max_speed_mps=0.0,
                    total_force_n=(0.0, 0.0, 1.0),
                    primary_mean_velocity_mps=(0.0, 0.0, -1.0e-3),
                    secondary_mean_velocity_mps=(0.0, 0.0, 0.0),
                    primary_mean_displacement_m=(0.0, 0.0, -1.0e-6),
                    secondary_mean_displacement_m=(0.0, 0.0, 0.0),
                )
                self.x = FakeArrayField(np.zeros((1, 3), dtype=np.float32))
                self.u = FakeArrayField(np.zeros((1, 3), dtype=np.float32))
                self.v = FakeArrayField(np.zeros((1, 3), dtype=np.float32))

            def advance_region_loads(self, **_kwargs):
                captured_solid_read_report.append(bool(_kwargs.get("read_report", True)))
                return self._report

            def report(self):
                solid_report_calls.append(None)
                return self._report

            def save_state(self) -> None:
                return None

            def restore_state(self) -> None:
                return None

        captured_cycles: list[int | None] = []
        captured_cg_preconditioners: list[str] = []
        captured_fluid_substeps: list[int] = []
        captured_substep_dt_s: list[float] = []
        captured_read_full_report: list[bool] = []
        captured_primary_velocity_z_mps: list[float] = []
        pressure_outlet_pressure_flux_m3s = 1.0e-12

        def fake_fluid_step(_fluid, _surface_diagnostics, config):
            captured_cycles.append(config.multigrid_cycles)
            captured_cg_preconditioners.append(config.cg_preconditioner)
            captured_fluid_substeps.append(config.fluid_substeps)
            captured_substep_dt_s.append(config.dt_s / float(config.fluid_substeps))
            captured_read_full_report.append(bool(config.read_full_report))
            captured_primary_velocity_z_mps.append(float(config.primary_velocity_mps[2]))
            is_final_accepted_step = len(captured_cycles) > 1
            zero = (0.0, 0.0, 0.0)
            balance = SimpleNamespace(residual_norm_n=0.0, relative_error=0.0)
            trial_project_calls = 4
            accepted_project_calls = 2
            project_calls = accepted_project_calls if is_final_accepted_step else trial_project_calls
            cg_iterations_total = 20 if is_final_accepted_step else 40
            force_report = SimpleNamespace(
                grid_force_n=zero,
                primary_fluid_force_n=zero,
                secondary_fluid_force_n=zero,
                constraint_force_n=zero,
                primary_constraint_force_n=zero,
                secondary_constraint_force_n=zero,
                volume_source_m3s=1.0e-6,
                primary_volume_source_m3s=1.0e-6,
                secondary_volume_source_m3s=0.0,
                active_force_cells=1,
                force_sample_count=1,
                force_invalid_probe_count=0,
                force_valid_probe_count=1,
                force_valid_probe_fraction=1.0,
                invalid_probe_area_m2=0.0,
                invalid_probe_volume_source_m3s=0.0,
            )
            impulse_report = SimpleNamespace(
                grid_impulse_n_s=zero,
                momentum_delta_n_s=zero,
                impulse_relative_error=0.0,
                active_velocity_cells=1,
            )
            velocity_constraint_momentum_delta_n_s = (0.0, 0.0, 4.0e-10)
            velocity_constraint_primary_momentum_delta_n_s = (0.0, 0.0, 3.0e-10)
            velocity_constraint_secondary_momentum_delta_n_s = (0.0, 0.0, 1.0e-10)
            velocity_constraint_primary_step_impulse_n_s = (0.0, 0.0, 6.0e-10)
            velocity_constraint_secondary_step_impulse_n_s = (0.0, 0.0, 2.0e-10)
            velocity_constraint_primary_step_equivalent_force_n = tuple(
                component / config.dt_s
                for component in velocity_constraint_primary_step_impulse_n_s
            )
            velocity_constraint_secondary_step_equivalent_force_n = tuple(
                component / config.dt_s
                for component in velocity_constraint_secondary_step_impulse_n_s
            )
            velocity_constraint_report = SimpleNamespace(
                active_cells=2,
                max_delta_mps=2.0e-7,
                mean_delta_mps=1.0e-7,
                momentum_delta_n_s=velocity_constraint_momentum_delta_n_s,
                primary_momentum_delta_n_s=velocity_constraint_primary_momentum_delta_n_s,
                secondary_momentum_delta_n_s=velocity_constraint_secondary_momentum_delta_n_s,
            )
            divergence = {
                "l2": 0.0,
                "max_abs": 0.0,
                "interior_l2": 0.0,
                "interior_max_abs": 0.0,
                "pre_projection_l2": 0.0,
                "pre_projection_max_abs": 0.0,
                "projection_l2": 0.0,
                "projection_max_abs": 0.0,
                "post_constraint_l2": 0.0,
                "post_constraint_max_abs": 0.0,
            }
            pressure_outlet_report = {
                "source_volume_flux_m3s": 1.0e-6,
                "zmin_reachable_source_volume_flux_m3s": 1.0e-6,
                "zmin_unreached_source_volume_flux_m3s": 0.0,
                "zmin_reachable_source_cell_count": 1,
                "zmin_unreached_source_cell_count": 0,
                "zmin_unreached_source_abs_flux_m3s": 0.0,
                "zmin_unreached_source_centroid_x_m": math.nan,
                "zmin_unreached_source_centroid_y_m": math.nan,
                "zmin_unreached_source_centroid_z_m": math.nan,
                "zmin_unreached_source_min_x_m": math.nan,
                "zmin_unreached_source_min_y_m": math.nan,
                "zmin_unreached_source_min_z_m": math.nan,
                "zmin_unreached_source_max_x_m": math.nan,
                "zmin_unreached_source_max_y_m": math.nan,
                "zmin_unreached_source_max_z_m": math.nan,
                "zmin_velocity_outlet_flux_m3s": (
                    1.0e-6 if is_final_accepted_step else 2.5e-7
                ),
                "zmin_velocity_outlet_to_source_ratio": (
                    1.0 if is_final_accepted_step else 0.25
                ),
                "zmin_pressure_outlet_flux_m3s": pressure_outlet_pressure_flux_m3s,
                "zmin_pressure_outlet_to_source_ratio": pressure_outlet_pressure_flux_m3s / 1.0e-6,
                "zmin_projection_pre_velocity_outlet_flux_m3s": (
                    -9.0e-6 if not is_final_accepted_step else 0.0
                ),
                "zmin_projection_post_pressure_velocity_outlet_flux_m3s": (
                    1.0e-6 if is_final_accepted_step else 2.5e-7
                ),
                "zmin_projection_post_boundary_velocity_outlet_flux_m3s": (
                    1.0e-6 if is_final_accepted_step else 2.5e-7
                ),
            }
            return SimpleNamespace(
                divergence=divergence,
                pressure_outlet_report=pressure_outlet_report,
                force_report=force_report,
                impulse_report=impulse_report,
                velocity_constraint_report=velocity_constraint_report,
                velocity_constraint_spread_report=None,
                ibm_correction_iterations=2,
                ibm_correction_dt_s=config.dt_s / float(config.fluid_substeps) / 2.0,
                fluid_substeps=config.fluid_substeps,
                fluid_substep_dt_s=config.dt_s / float(config.fluid_substeps),
                primary_equivalent_fluid_force_n=zero,
                secondary_equivalent_fluid_force_n=zero,
                primary_velocity_constraint_impulse_n_s=(
                    velocity_constraint_primary_step_impulse_n_s
                ),
                secondary_velocity_constraint_impulse_n_s=(
                    velocity_constraint_secondary_step_impulse_n_s
                ),
                primary_velocity_constraint_equivalent_fluid_force_n=(
                    velocity_constraint_primary_step_equivalent_force_n
                ),
                secondary_velocity_constraint_equivalent_fluid_force_n=(
                    velocity_constraint_secondary_step_equivalent_force_n
                ),
                interface_reaction_target=SimpleNamespace(
                    primary_force_n=zero,
                    secondary_force_n=zero,
                ),
                primary_interface_reaction_balance=balance,
                secondary_interface_reaction_balance=balance,
                pressure_projection_cg_project_calls=project_calls,
                pressure_projection_cg_iterations_total=cg_iterations_total,
                pressure_projection_cg_iterations_max=11 if is_final_accepted_step else 17,
                pressure_projection_cg_host_residual_checks=3 if is_final_accepted_step else 5,
                pressure_projection_cg_converged_all=True,
                pressure_projection_cg_max_relative_residual=(
                    2.0e-7 if is_final_accepted_step else 3.0e-7
                ),
                pressure_projection_cg_max_initial_relative_residual=(
                    0.2 if is_final_accepted_step else 0.4
                ),
                pressure_projection_cg_breakdown_count=0,
            )

        metadata = {
            "diagnostic_area_m2_by_region": {"7": 1.0e-4, "8": 1.0e-4},
            "solid_area_m2_by_region": {"5": 1.0e-4},
            "solid_surface_face_count": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_config = temp_path / "source.json"
            output_dir = temp_path / "run"
            source_config.write_text(
                json.dumps(
                    {
                        "analysis_settings": {"time_step_s": 5.0e-4},
                        "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                    }
                ),
                encoding="utf-8",
            )
            uniform_solid_grid_nodes = infer_spec(source_config, grid_scale=1.0).grid_nodes
            args = parse_args(
                [
                    "--source-config",
                    str(source_config),
                    "--output-dir",
                    str(output_dir),
                    "--steps",
                    "2",
                    "--use-graded-grid",
                    "--use-nozzle-taper",
                    "--divergence-cleanup-iterations",
                    "0",
                    "--solid-mpm-substeps",
                    "1",
                    "--fsi-coupling-iterations",
                    "2",
                    "--constraint-force-scale",
                    "0.5",
                    "--cg-preconditioner",
                    "jacobi",
                    "--max-wall-time-s",
                    "1e-9",
                ]
            )
            with (
                patch("cases.squid_soft_robot.ReducedSquidFSI", FakeSimulator),
                patch("cases.squid_soft_robot.TriMooneyShellMpmState", FakeSolidMpm),
                patch(
                    "cases.squid_soft_robot.build_tri_surface_diagnostics",
                    return_value=(FakeTriDiagnostics(), metadata, object(), np.zeros(1, dtype=np.int32)),
                ),
                patch(
                    "cases.squid_soft_robot.advance_projected_ibm_region_pair_fluid_step",
                    side_effect=fake_fluid_step,
                ),
            ):
                summary = run(args)
            process = json.loads((output_dir / "run_process.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["pressure_solver"], "fv_cg")
        self.assertEqual(summary["cg_preconditioner"], "jacobi")
        self.assertIsNone(summary["effective_multigrid_cycles"])
        self.assertFalse(summary["reuse_accepted_fsi_trial_state"])
        self.assertEqual(summary["accepted_fsi_trial_state_reuse_count"], 0)
        self.assertEqual(len(captured_cycles), 2)
        self.assertTrue(all(cycle is None for cycle in captured_cycles))
        self.assertEqual(captured_cg_preconditioners, ["jacobi", "jacobi"])
        self.assertEqual(captured_read_full_report, [False, True])
        self.assertEqual(captured_primary_velocity_z_mps, [-5.0e-4, -5.0e-4])
        self.assertEqual(captured_primary_offset_z_m, [-5.0e-7, -5.0e-7])
        self.assertTrue(captured_solid_read_report)
        self.assertTrue(all(not value for value in captured_solid_read_report))
        self.assertEqual(len(solid_report_calls), len(captured_solid_read_report))
        self.assertEqual(len(sample_calls), 1)
        self.assertEqual(len(diagnose_calls), 1)
        self.assertEqual(captured_fluid_substeps[0], 12)
        self.assertAlmostEqual(captured_substep_dt_s[0], 5.0e-4 / 12.0)
        self.assertEqual(summary["fluid_substeps"], captured_fluid_substeps[0])
        self.assertAlmostEqual(summary["fluid_substep_dt_s"], captured_substep_dt_s[0])
        self.assertNotEqual(tuple(summary["fluid_grid_nodes"]), uniform_solid_grid_nodes)
        self.assertEqual(captured_solid_grid_nodes, [uniform_solid_grid_nodes])
        self.assertEqual(summary["constraint_force_scale"], 0.5)
        self.assertAlmostEqual(summary["final_pressure_outlet_velocity_to_source_ratio"], 1.0)
        self.assertAlmostEqual(summary["final_pressure_outlet_pressure_to_source_ratio"], 1.0e-6)
        self.assertTrue(summary["checks"]["pressure_outlet_velocity_to_source_ratio_near_one"])
        self.assertFalse(summary["outlet_to_fsi_volume_source_gate_scope"]["hard_gate"])
        self.assertNotIn("final_outlet_to_fsi_volume_source_ratio_physical", summary["checks"])
        self.assertIn(
            "final_outlet_to_fsi_volume_source_ratio_physical",
            summary["diagnostic_checks"],
        )
        self.assertNotIn("pressure_traction_nonzero", summary["checks"])
        self.assertIn(
            "projection_pressure_traction_diagnostic_nonzero",
            summary["diagnostic_checks"],
        )
        self.assertFalse(summary["boundary_drive_compliance"]["compliant"])
        self.assertEqual(
            summary["boundary_drive_compliance"]["nonzero_fluid_traction_scale"],
            0.5,
        )
        self.assertEqual(summary["boundary_drive_compliance_gate"], "diagnostic_only")
        self.assertNotIn("boundary_drive_compliant", summary["checks"])
        self.assertNotIn("solid_model_is_physical_mpm", summary["checks"])
        self.assertFalse(summary["diagnostic_checks"]["boundary_drive_has_no_prescribed_driver"])
        self.assertTrue(summary["diagnostic_checks"]["solid_model_choice_supported"])
        self.assertTrue(summary["partial_run"])
        self.assertEqual(summary["partial_run_reason"], "max_wall_time_s")
        self.assertEqual(summary["validation_scope"], "wall_time_partial")
        self.assertFalse(summary["validation_scope_complete"])
        self.assertEqual(summary["validation_scope_reason"], "max_wall_time_s")
        self.assertFalse(summary["completed_step_checks_passed"])
        self.assertFalse(summary["checks"]["fsi_physical_interface_map_stable"])
        self.assertFalse(summary["fsi_coupling_raw_interface_map_strict_physical"])
        self.assertEqual(
            summary["fsi_physical_interface_map_stability"]["status"],
            "unmeasured",
        )
        self.assertFalse(summary["fsi_physical_interface_map_stability"]["measured"])
        self.assertEqual(
            summary["max_fsi_coupling_raw_interface_map_amplification_sample_count"],
            0,
        )
        self.assertEqual(summary["requested_steps"], 2)
        self.assertEqual(summary["full_pressure_waveform_steps"], 4000)
        self.assertEqual(summary["completed_steps"], 1)
        self.assertEqual(summary["reproduction_status"], "reduced_validation_partial")
        self.assertIsNone(summary["validation_passed"])
        self.assertFalse(summary["pressure_flux_trend"]["complete"])
        self.assertEqual(summary["pressure_flux_trend"]["conclusion"], "incomplete")
        self.assertIn("timing", summary)
        self.assertGreater(summary["timing"]["max_step_wall_time_s"], 0.0)
        self.assertGreaterEqual(summary["timing"]["max_solid_advance_wall_time_s"], 0.0)
        self.assertGreaterEqual(summary["timing"]["max_fluid_advance_wall_time_s"], 0.0)
        self.assertGreaterEqual(summary["timing"]["max_surface_diagnostics_wall_time_s"], 0.0)
        self.assertIn("runtime_budget", summary)
        self.assertEqual(summary["runtime_budget"]["requested_steps"], 2)
        self.assertEqual(summary["runtime_budget"]["completed_steps"], 1)
        self.assertEqual(summary["runtime_budget"]["full_pressure_waveform_steps"], 4000)
        self.assertGreater(
            summary["runtime_budget"]["estimated_full_pressure_waveform_wall_time_s"],
            summary["runtime_budget"]["estimated_requested_run_wall_time_s"],
        )
        self.assertEqual(summary["max_pressure_projection_cg_project_calls"], 2)
        self.assertEqual(summary["max_fsi_trial_pressure_projection_cg_project_calls"], 4)
        self.assertEqual(summary["max_total_pressure_projection_cg_project_calls"], 6)
        self.assertEqual(summary["max_pressure_projection_cg_iterations_total"], 20)
        self.assertEqual(summary["max_fsi_trial_pressure_projection_cg_iterations_total"], 40)
        self.assertEqual(summary["max_total_pressure_projection_cg_iterations_total"], 60)
        self.assertIn("max_fsi_coupling_interface_map_amplification", summary)
        self.assertIn("max_fsi_coupling_residual_jacobian_amplification", summary)
        self.assertGreaterEqual(summary["max_fsi_coupling_interface_map_amplification"], 0.0)
        self.assertGreaterEqual(
            summary["max_fsi_coupling_residual_jacobian_amplification"],
            0.0,
        )
        self.assertIn("max_fsi_velocity_constraint_momentum_delta_n_s", summary)
        self.assertIn("max_fsi_velocity_constraint_equivalent_force_norm_n", summary)
        self.assertIn("max_fsi_velocity_constraint_primary_momentum_delta_n_s", summary)
        self.assertIn("max_fsi_velocity_constraint_secondary_momentum_delta_n_s", summary)
        self.assertIn("max_fsi_velocity_constraint_primary_equivalent_force_norm_n", summary)
        self.assertIn("max_fsi_velocity_constraint_secondary_equivalent_force_norm_n", summary)
        self.assertIn("max_fsi_velocity_constraint_step_impulse_n_s", summary)
        self.assertIn("max_fsi_velocity_constraint_step_equivalent_force_norm_n", summary)
        self.assertGreaterEqual(summary["max_fsi_velocity_constraint_momentum_delta_n_s"], 0.0)
        self.assertGreaterEqual(summary["max_fsi_velocity_constraint_equivalent_force_norm_n"], 0.0)
        expected_ibm_correction_dt_s = captured_substep_dt_s[0] / 2.0
        expected_step_dt_s = captured_substep_dt_s[0] * captured_fluid_substeps[0]
        self.assertAlmostEqual(summary["ibm_correction_dt_s"], expected_ibm_correction_dt_s)
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_momentum_delta_n_s"],
            4.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_primary_momentum_delta_n_s"],
            3.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_secondary_momentum_delta_n_s"],
            1.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_equivalent_force_norm_n"],
            4.0e-10 / expected_ibm_correction_dt_s,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_primary_equivalent_force_norm_n"],
            3.0e-10 / expected_ibm_correction_dt_s,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_secondary_equivalent_force_norm_n"],
            1.0e-10 / expected_ibm_correction_dt_s,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_step_impulse_n_s"],
            8.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_primary_step_impulse_n_s"],
            6.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_secondary_step_impulse_n_s"],
            2.0e-10,
        )
        self.assertAlmostEqual(
            summary["max_fsi_velocity_constraint_step_equivalent_force_norm_n"],
            8.0e-10 / expected_step_dt_s,
        )
        self.assertEqual(process["status"], "partial")
        self.assertEqual(process["validation_scope"], "wall_time_partial")
        self.assertFalse(process["validation_scope_complete"])
        self.assertEqual(process["validation_scope_reason"], "max_wall_time_s")
        self.assertIsNone(process["validation_passed"])
        self.assertEqual(process["requested_steps"], 2)
        self.assertEqual(process["full_pressure_waveform_steps"], 4000)
        self.assertEqual(process["completed_steps"], 1)
        self.assertEqual(process["partial_run_reason"], "max_wall_time_s")

    def test_fsi_not_converged_count_parses_resume_csv_booleans(self) -> None:
        rows = [
            {"fsi_coupling_enabled": "True", "fsi_coupling_converged": "True"},
            {"fsi_coupling_enabled": "True", "fsi_coupling_converged": "False"},
            {"fsi_coupling_enabled": True, "fsi_coupling_converged": False},
            {"fsi_coupling_enabled": "False", "fsi_coupling_converged": "False"},
        ]

        self.assertEqual(count_enabled_unconverged_fsi_rows(rows), 2)

    def test_partitioned_interface_reaction_defaults_are_under_relaxed_aitken_without_passivity(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py"]):
            args = parse_args()

        self.assertAlmostEqual(args.interface_reaction_relaxation, 0.5)
        self.assertEqual(args.fsi_coupling_iterations, 1)
        self.assertEqual(args.fsi_coupling_solver, "aitken")
        self.assertAlmostEqual(args.fsi_coupling_tolerance_n, 1.0e-3)
        self.assertEqual(args.fsi_coupling_adaptive_iterations_max, 0)
        self.assertTrue(
            math.isinf(args.fsi_coupling_adaptive_iterations_residual_threshold_n)
        )
        self.assertTrue(
            math.isinf(args.fsi_coupling_adaptive_iterations_cfl_threshold)
        )
        self.assertEqual(args.fsi_coupling_same_step_rerun_iterations_max, 0)
        self.assertTrue(
            math.isinf(args.fsi_coupling_same_step_rerun_residual_threshold_n)
        )
        self.assertEqual(args.fsi_coupling_residual_continuation_iterations_max, 0)
        self.assertTrue(
            math.isinf(args.fsi_coupling_residual_continuation_threshold_n)
        )
        self.assertFalse(
            args.fsi_coupling_residual_continuation_rebound_secant_from_best
        )
        self.assertTrue(
            math.isinf(args.fsi_coupling_residual_continuation_rebound_secant_factor)
        )
        self.assertEqual(
            args.fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max,
            0,
        )
        self.assertTrue(
            math.isinf(args.fsi_coupling_trial_interior_divergence_tolerance)
        )
        self.assertAlmostEqual(args.fsi_coupling_target_map_relaxation, 1.0)
        self.assertAlmostEqual(args.fsi_coupling_rejected_trial_backtrack, 1.0)
        self.assertTrue(math.isinf(args.fsi_coupling_residual_growth_rejection_factor))
        self.assertTrue(math.isinf(args.fsi_coupling_max_accepted_residual_n))
        self.assertTrue(math.isinf(args.fsi_coupling_trust_region_force_increment_n))
        self.assertFalse(args.fsi_coupling_trust_region_adaptive)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_shrink_factor, 0.5)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_growth_factor, 1.25)
        self.assertTrue(math.isinf(args.fsi_coupling_trust_region_rebound_factor))
        self.assertAlmostEqual(args.fsi_coupling_trust_region_rebound_backtrack, 0.5)
        self.assertTrue(math.isinf(args.fsi_coupling_trust_region_rebound_stop_factor))
        self.assertTrue(
            math.isinf(args.fsi_coupling_trust_region_rebound_stop_max_residual_n)
        )
        self.assertFalse(args.reuse_accepted_fsi_trial_state)
        self.assertTrue(args.interface_reaction_aitken)
        self.assertAlmostEqual(args.interface_reaction_aitken_lower_bound, 0.01)
        self.assertAlmostEqual(args.interface_reaction_aitken_upper_bound, 1.5)
        self.assertFalse(args.interface_reaction_passivity_limit)
        self.assertAlmostEqual(args.interface_reaction_robin_impedance_ns_m, 0.0)
        self.assertAlmostEqual(args.interface_reaction_robin_matrix_impedance_ns_m, 0.0)
        self.assertEqual(args.projection_iterations, 3000)
        self.assertIsNone(args.steps)
        self.assertFalse(args.steps_explicit)
        self.assertEqual(args.pressure_solver, "auto")
        self.assertEqual(args.pressure_solve_failure_policy, "raise")

    def test_fsi_coupling_max_accepted_residual_cli_is_explicit(self) -> None:
        args = parse_args(["--fsi-coupling-max-accepted-residual-n", "1.5"])

        self.assertAlmostEqual(args.fsi_coupling_max_accepted_residual_n, 1.5)

    def test_fsi_coupling_trust_region_force_increment_cli_is_explicit(self) -> None:
        args = parse_args(["--fsi-coupling-trust-region-force-increment-n", "2.5"])

        self.assertAlmostEqual(args.fsi_coupling_trust_region_force_increment_n, 2.5)

    def test_fsi_coupling_adaptive_iterations_cli_is_explicit(self) -> None:
        args = parse_args(
            [
                "--fsi-coupling-iterations",
                "6",
                "--fsi-coupling-adaptive-iterations-max",
                "12",
                "--fsi-coupling-adaptive-iterations-residual-threshold-n",
                "1.0",
                "--fsi-coupling-adaptive-iterations-cfl-threshold",
                "0.1",
            ]
        )

        self.assertEqual(args.fsi_coupling_iterations, 6)
        self.assertEqual(args.fsi_coupling_adaptive_iterations_max, 12)
        self.assertAlmostEqual(
            args.fsi_coupling_adaptive_iterations_residual_threshold_n,
            1.0,
        )
        self.assertAlmostEqual(args.fsi_coupling_adaptive_iterations_cfl_threshold, 0.1)

    def test_fsi_coupling_same_step_rerun_cli_is_explicit(self) -> None:
        args = parse_args(
            [
                "--fsi-coupling-iterations",
                "6",
                "--fsi-coupling-same-step-rerun-iterations-max",
                "12",
                "--fsi-coupling-same-step-rerun-residual-threshold-n",
                "0.5",
            ]
        )

        self.assertEqual(args.fsi_coupling_iterations, 6)
        self.assertEqual(args.fsi_coupling_same_step_rerun_iterations_max, 12)
        self.assertAlmostEqual(
            args.fsi_coupling_same_step_rerun_residual_threshold_n,
            0.5,
        )

    def test_fsi_coupling_residual_continuation_cli_is_explicit(self) -> None:
        args = parse_args(
            [
                "--fsi-coupling-residual-continuation-iterations-max",
                "6",
                "--fsi-coupling-residual-continuation-threshold-n",
                "0.25",
                "--fsi-coupling-residual-continuation-rebound-secant-from-best",
                "--fsi-coupling-residual-continuation-rebound-secant-factor",
                "1.0",
                "--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max",
                "1",
                "--fsi-coupling-trial-interior-divergence-tolerance",
                "0.1",
            ]
        )

        self.assertEqual(args.fsi_coupling_residual_continuation_iterations_max, 6)
        self.assertAlmostEqual(
            args.fsi_coupling_residual_continuation_threshold_n,
            0.25,
        )
        self.assertTrue(
            args.fsi_coupling_residual_continuation_rebound_secant_from_best
        )
        self.assertAlmostEqual(
            args.fsi_coupling_residual_continuation_rebound_secant_factor,
            1.0,
        )
        self.assertEqual(
            args.fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max,
            1,
        )
        self.assertAlmostEqual(
            args.fsi_coupling_trial_interior_divergence_tolerance,
            0.1,
        )

    def test_fsi_same_step_rerun_triggers_only_for_unconverged_high_residual(self) -> None:
        self.assertTrue(
            fsi_same_step_rerun_triggered(
                current_iterations_requested=6,
                rerun_iterations_max=12,
                residual_norm_n=0.75,
                residual_threshold_n=0.5,
                converged=False,
            )
        )
        self.assertTrue(
            fsi_same_step_rerun_triggered(
                current_iterations_requested=6,
                rerun_iterations_max=12,
                residual_norm_n=math.inf,
                residual_threshold_n=0.5,
                converged=False,
            )
        )
        self.assertFalse(
            fsi_same_step_rerun_triggered(
                current_iterations_requested=12,
                rerun_iterations_max=12,
                residual_norm_n=0.75,
                residual_threshold_n=0.5,
                converged=False,
            )
        )
        self.assertFalse(
            fsi_same_step_rerun_triggered(
                current_iterations_requested=6,
                rerun_iterations_max=12,
                residual_norm_n=0.75,
                residual_threshold_n=0.5,
                converged=True,
            )
        )
        self.assertFalse(
            fsi_same_step_rerun_triggered(
                current_iterations_requested=6,
                rerun_iterations_max=12,
                residual_norm_n=0.75,
                residual_threshold_n=math.inf,
                converged=False,
            )
        )

    def test_fsi_trial_acceptance_can_gate_interior_divergence(self) -> None:
        self.assertTrue(
            fsi_trial_acceptance_passes(
                {"trial_cfl": 0.25, "trial_interior_divergence_l2": 0.05},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            )
        )
        self.assertFalse(
            fsi_trial_acceptance_passes(
                {"trial_cfl": 0.25, "trial_interior_divergence_l2": 0.15},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            )
        )
        self.assertFalse(
            fsi_trial_acceptance_passes(
                {"trial_cfl": 0.25},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            )
        )
        self.assertTrue(
            fsi_trial_acceptance_passes(
                {"trial_cfl": 0.25},
                cfl_limit=0.5,
            )
        )

    def test_fsi_trial_acceptance_reports_rejection_reason(self) -> None:
        self.assertIsNone(
            fsi_trial_acceptance_rejection_reason(
                {"trial_cfl": 0.25, "trial_interior_divergence_l2": 0.05},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            )
        )
        self.assertEqual(
            fsi_trial_acceptance_rejection_reason(
                {"trial_cfl": 0.5, "trial_interior_divergence_l2": 0.05},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            ),
            "cfl",
        )
        self.assertEqual(
            fsi_trial_acceptance_rejection_reason(
                {"trial_cfl": 0.25, "trial_interior_divergence_l2": 0.15},
                cfl_limit=0.5,
                interior_divergence_l2_limit=0.1,
            ),
            "interior_divergence_l2",
        )

    def test_fsi_coupling_adaptive_trust_region_cli_is_explicit(self) -> None:
        args = parse_args(
            [
                "--fsi-coupling-trust-region-force-increment-n",
                "2.0",
                "--fsi-coupling-trust-region-adaptive",
                "--fsi-coupling-trust-region-shrink-factor",
                "0.25",
                "--fsi-coupling-trust-region-growth-factor",
                "1.5",
            ]
        )

        self.assertAlmostEqual(args.fsi_coupling_trust_region_force_increment_n, 2.0)
        self.assertTrue(args.fsi_coupling_trust_region_adaptive)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_shrink_factor, 0.25)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_growth_factor, 1.5)
        self.assertEqual(args.fluid_advection_scheme, "euler")
        self.assertEqual(args.cg_preconditioner, "auto")
        self.assertIsNone(args.multigrid_cycles)
        self.assertEqual(args.divergence_cleanup_iterations, 8)
        self.assertEqual(args.fluid_substeps, 1)
        self.assertFalse(args.adaptive_fluid_substeps)
        self.assertAlmostEqual(args.adaptive_fluid_substeps_target_cfl, 0.25)
        self.assertEqual(args.adaptive_fluid_substeps_max, 16)
        self.assertAlmostEqual(args.adaptive_fluid_substeps_safety, 1.25)
        self.assertEqual(args.ibm_correction_iterations, 2)
        self.assertAlmostEqual(args.constraint_force_scale, 1.0)
        self.assertAlmostEqual(args.fsi_constraint_force_solid_mobility_ratio, 0.0)
        self.assertFalse(args.fsi_solid_response_mobility_coupling)
        self.assertAlmostEqual(args.fsi_velocity_target_solid_mobility_ratio, 0.0)
        self.assertFalse(args.fsi_solid_response_velocity_mobility_coupling)
        self.assertAlmostEqual(args.projection_divergence_tolerance, 1.0e-2)
        self.assertAlmostEqual(args.min_outlet_to_main_volume_flux_ratio, 0.1)
        self.assertAlmostEqual(args.pressure_outlet_source_ratio_tolerance, 0.1)
        self.assertIsNone(args.cad_step_path)
        self.assertFalse(args.require_real_cad_step)
        self.assertIsNone(args.pressure_t0_s)
        self.assertIsNone(args.pressure_t1_s)
        self.assertIsNone(args.pressure_t2_s)
        self.assertIsNone(args.pressure_p0_pa)
        self.assertIsNone(args.pressure_p1_pa)
        self.assertIsNone(args.pressure_p2_pa)
        self.assertFalse(args.use_graded_grid)
        self.assertFalse(args.preflight_only)
        self.assertFalse(args.use_region14_aperture_carve)
        self.assertFalse(args.disable_region14_aperture_carve)
        self.assertFalse(args.use_nozzle_taper)
        self.assertFalse(args.checkpoint_every_step)
        self.assertFalse(args.resume_from_checkpoint)
        self.assertIsNone(args.checkpoint_path)
        self.assertIsNone(args.nozzle_taper_length_m)
        self.assertIsNone(args.nozzle_taper_inlet_radius_m)
        self.assertIsNone(args.graded_grid_target_spacing_m)
        self.assertAlmostEqual(args.graded_grid_farfield_spacing_m, 3.0e-3)
        self.assertAlmostEqual(args.graded_grid_growth_ratio, 1.2)
        self.assertIsNone(args.graded_grid_max_cells)
        self.assertFalse(args.use_tail_refinement)
        self.assertIsNone(args.tail_refinement_target_spacing_m)
        self.assertIsNone(args.tail_refinement_padding_m)
        self.assertAlmostEqual(args.membrane_thickness_scale, 1.0)
        self.assertAlmostEqual(args.solid_density_scale, 1.0)
        self.assertAlmostEqual(args.max_wall_time_s, 0.0)
        self.assertFalse(hasattr(args, "fluid_feedback_relaxation"))
        self.assertFalse(hasattr(args, "fluid_feedback_aitken"))
        self.assertFalse(hasattr(args, "fluid_feedback_passivity_limit"))
        self.assertFalse(hasattr(args, "solid_constraint_reaction_feedback"))
        self.assertFalse(hasattr(args, "fsi_feedback_force_mode"))
        self.assertFalse(hasattr(args, "pressure_force_scale"))
        self.assertAlmostEqual(args.solid_mpm_flip_blend, 0.95)

    def test_fsi_coupling_trust_region_rebound_cli_is_explicit(self) -> None:
        args = parse_args(
            [
                "--fsi-coupling-trust-region-rebound-factor",
                "2.0",
                "--fsi-coupling-trust-region-rebound-backtrack",
                "0.25",
                "--fsi-coupling-trust-region-rebound-stop-factor",
                "3.0",
                "--fsi-coupling-trust-region-rebound-stop-max-residual-n",
                "1.5",
            ]
        )

        self.assertAlmostEqual(args.fsi_coupling_trust_region_rebound_factor, 2.0)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_rebound_backtrack, 0.25)
        self.assertAlmostEqual(args.fsi_coupling_trust_region_rebound_stop_factor, 3.0)
        self.assertAlmostEqual(
            args.fsi_coupling_trust_region_rebound_stop_max_residual_n,
            1.5,
        )

    def test_hibm_mpm_sharp_allows_marker_fixed_point_iterations(self) -> None:
        raise_for_unsupported_hibm_mpm_sharp_iteration_options(
            fsi_coupling_mode=FSI_COUPLING_MODE_HIBM_MPM_SHARP,
            fsi_coupling_iterations=6,
        )
        raise_for_unsupported_hibm_mpm_sharp_iteration_options(
            fsi_coupling_mode=FSI_COUPLING_MODE_HIBM_MPM_SHARP,
            fsi_coupling_iterations=1,
        )
        raise_for_unsupported_hibm_mpm_sharp_iteration_options(
            fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
            fsi_coupling_iterations=6,
        )

    def test_hibm_mpm_sharp_marker_fixed_point_has_velocity_tolerance(self) -> None:
        args = parse_args(["--fsi-marker-coupling-tolerance-mps", "2.5e-4"])

        self.assertAlmostEqual(args.fsi_marker_coupling_tolerance_mps, 2.5e-4)

    def test_sharp_marker_fixed_point_residual_uses_position_and_velocity(self) -> None:
        guess = {
            "x_gamma_m": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "v_gamma_mps": np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                dtype=np.float32,
            ),
            "n_gamma": np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "A_gamma_m2": np.asarray([1.0, 1.0], dtype=np.float32),
        }
        candidate = {
            **guess,
            "x_gamma_m": np.asarray(
                [[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "v_gamma_mps": np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 2.2, 0.0]],
                dtype=np.float32,
            ),
        }

        residual = sharp_marker_fixed_point_residual_mps(
            guess,
            candidate,
            dt_s=0.5,
        )

        self.assertAlmostEqual(residual["max_mps"], 0.2, places=6)
        self.assertAlmostEqual(
            residual["l2_mps"],
            math.sqrt((0.1 * 0.1 + 0.2 * 0.2) / 2.0),
            places=6,
        )
        self.assertEqual(residual["sample_count"], 2)

    def test_sharp_marker_fixed_point_residual_diagnostics_split_region_terms(
        self,
    ) -> None:
        guess = {
            "x_gamma_m": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "v_gamma_mps": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        candidate = {
            "x_gamma_m": np.asarray(
                [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "v_gamma_mps": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]],
                dtype=np.float32,
            ),
        }

        diagnostics = sharp_marker_fixed_point_residual_diagnostics_mps(
            guess,
            candidate,
            dt_s=0.5,
            marker_region_ids=np.asarray([5, 8], dtype=np.int32),
            primary_region_id=5,
            secondary_region_id=8,
        )

        self.assertAlmostEqual(diagnostics["position_l2_mps"], math.sqrt(0.02))
        self.assertAlmostEqual(diagnostics["velocity_l2_mps"], math.sqrt(0.045))
        self.assertAlmostEqual(diagnostics["primary_region_l2_mps"], 0.2)
        self.assertAlmostEqual(diagnostics["secondary_region_l2_mps"], 0.3)
        self.assertEqual(diagnostics["max_marker_index"], 1)
        self.assertEqual(diagnostics["max_marker_region_id"], 8)

    def test_sharp_marker_fixed_point_uses_velocity_units_not_force_units(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn(
            "velocity_residual_norm_mps <= fsi_marker_coupling_tolerance_mps",
            source,
        )
        self.assertNotIn("residual_norm_mps <= fsi_coupling_tolerance_n", source)
        self.assertIn(
            "marker_surface_fixed_point_velocity_residual_l2_mps",
            source,
        )
        self.assertNotIn(
            "marker_surface_fixed_point_position_velocity_residual_l2_mps",
            source,
        )
        self.assertIn("sharp marker fixed point did not converge", source)
        self.assertIn("velocity_residual_history_mps=", source)
        self.assertIn("combined_residual_history_mps=", source)
        self.assertIn("relaxation_history=", source)

    def test_sharp_report_fluid_projection_failure_reason_reports_trial_failures(
        self,
    ) -> None:
        report = SimpleNamespace(
            fluid_to_mpm_loads=SimpleNamespace(
                fluid_projection={
                    "pressure_solve_failed": True,
                    "pressure_projection_physical_failure": True,
                    "pressure_projection_physical_failure_reason": (
                        "unreached_component_rhs_incompatible"
                    ),
                    "cg_converged_all": False,
                    "cg_breakdown_count": 4,
                }
            )
        )

        reason = sharp_report_fluid_projection_failure_reason(report)

        self.assertIn("pressure_solve_failed", reason)
        self.assertIn("unreached_component_rhs_incompatible", reason)
        self.assertIn("cg_converged_all=false", reason)
        self.assertIn("cg_breakdown_count=4", reason)

    def test_sharp_marker_fixed_point_checks_projection_failure_before_converged(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        block = source.split("def advance_sharp_marker_fixed_point_step():", 1)[1]
        loop_block = block.split("if report is None:", 1)[0]

        failure_check = loop_block.index(
            "sharp_report_fluid_projection_failure_reason(report)"
        )
        convergence_check = loop_block.index(
            "velocity_residual_norm_mps <= fsi_marker_coupling_tolerance_mps"
        )

        self.assertLess(failure_check, convergence_check)

    def test_completed_step_gate_accepts_sharp_physical_convergence(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        block = source.split('"fsi_coupling_convergence_not_claimed":', 1)[1]
        block = block.split('"finite_primary_diagnostics":', 1)[0]

        self.assertIn("fsi_coupling_explicit_single_pass", block)
        self.assertIn("fsi_coupling_convergence_measured", block)
        self.assertIn("fsi_coupling_residual_units", block)
        self.assertIn(
            "marker_surface_fixed_point_velocity_residual_l2_mps",
            block,
        )

    def test_sharp_completed_step_gate_uses_downstream_jet_sections(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        checks_block = source.split("checks = {", 1)[1].split(
            "completed_step_checks_passed",
            1,
        )[0]

        self.assertIn('"final_negative_z_jet_sections"', checks_block)
        self.assertNotIn('"final_negative_z_all_sections"', checks_block)
        self.assertIn('"final_negative_z_all_sections"', source)

    def test_relaxed_sharp_marker_state_arrays_returns_new_normalized_state(self) -> None:
        guess = {
            "x_gamma_m": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            "v_gamma_mps": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            "n_gamma": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "A_gamma_m2": np.asarray([1.0], dtype=np.float32),
        }
        candidate = {
            "x_gamma_m": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "v_gamma_mps": np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32),
            "n_gamma": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "A_gamma_m2": np.asarray([3.0], dtype=np.float32),
        }

        relaxed = relaxed_sharp_marker_state_arrays(
            guess,
            candidate,
            relaxation=0.25,
        )

        np.testing.assert_allclose(relaxed["x_gamma_m"], [[0.25, 0.0, 0.0]])
        np.testing.assert_allclose(relaxed["v_gamma_mps"], [[0.0, 0.5, 0.0]])
        np.testing.assert_allclose(
            relaxed["n_gamma"],
            [[0.94868326, 0.31622776, 0.0]],
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(relaxed["A_gamma_m2"], [1.5])
        np.testing.assert_allclose(guess["x_gamma_m"], [[0.0, 0.0, 0.0]])

    def test_sharp_pressure_neumann_gradient_state_roundtrips_marker_prefix(
        self,
    ) -> None:
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=3,
            projection_triangle_capacity=1,
        )
        coupling.markers.load_markers(
            positions_m=((0.25, 0.25, 0.25), (0.5, 0.5, 0.5)),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            normals=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            areas_m2=(0.01, 0.02),
            region_ids=(1, 2),
        )
        full = coupling.marker_pressure_neumann_gradient_pa_per_m.to_numpy()
        full[:3] = np.asarray([11.0, 22.0, 999.0], dtype=full.dtype)
        coupling.marker_pressure_neumann_gradient_pa_per_m.from_numpy(full)

        state = sharp_pressure_neumann_gradient_state_array(coupling)
        full[:3] = np.asarray([-1.0, -2.0, -3.0], dtype=full.dtype)
        coupling.marker_pressure_neumann_gradient_pa_per_m.from_numpy(full)
        restore_sharp_pressure_neumann_gradient_state_array(coupling, state)

        restored = coupling.marker_pressure_neumann_gradient_pa_per_m.to_numpy()
        np.testing.assert_allclose(restored[:2], [11.0, 22.0])
        self.assertAlmostEqual(float(restored[2]), -3.0)

    def test_relaxed_sharp_pressure_neumann_gradient_state_array_returns_new_state(
        self,
    ) -> None:
        guess = np.asarray([0.0, 10.0, -2.0], dtype=np.float32)
        candidate = np.asarray([4.0, 2.0, 6.0], dtype=np.float32)

        relaxed = relaxed_sharp_pressure_neumann_gradient_state_array(
            guess,
            candidate,
            relaxation=0.25,
        )

        np.testing.assert_allclose(relaxed, [1.0, 8.0, 0.0])
        np.testing.assert_allclose(guess, [0.0, 10.0, -2.0])

    def test_sharp_fixed_point_trial_restore_resets_pressure_neumann_gradient(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        block = source.split("def advance_sharp_marker_fixed_point_step():", 1)[1]
        self.assertIn(
            "pressure_gradient_state = (",
            block,
        )
        self.assertIn(
            "sharp_pressure_neumann_gradient_state_array(sharp_coupling_state)",
            block,
        )
        self.assertIn(
            "restore_sharp_pressure_neumann_gradient_state_array(",
            source,
        )
        self.assertIn(
            "restore_sharp_trial_state(marker_guess, pressure_gradient_state)",
            block,
        )
        self.assertIn(
            "candidate_pressure_gradient_state = (",
            block,
        )
        self.assertIn(
            "pressure_gradient_state = (",
            block,
        )
        self.assertIn(
            "relaxed_sharp_pressure_neumann_gradient_state_array(",
            block,
        )

    def test_shell_surface_mass_scales_can_be_selected_explicitly(self) -> None:
        args = parse_args(
            [
                "--membrane-thickness-scale",
                "2.5",
                "--solid-density-scale",
                "4",
            ]
        )

        self.assertAlmostEqual(args.membrane_thickness_scale, 2.5)
        self.assertAlmostEqual(args.solid_density_scale, 4.0)

    def test_iqn_ils_interface_reaction_solver_can_be_selected_explicitly(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--fsi-coupling-solver", "iqn_ils"]):
            args = parse_args()

        self.assertEqual(args.fsi_coupling_solver, "iqn_ils")
        self.assertTrue(args.interface_reaction_aitken)

    def test_fsi_coupling_target_map_relaxation_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--fsi-coupling-target-map-relaxation",
                "0.25",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.fsi_coupling_target_map_relaxation, 0.25)

    def test_fsi_coupling_rejected_trial_backtrack_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--fsi-coupling-rejected-trial-backtrack",
                "0.25",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.fsi_coupling_rejected_trial_backtrack, 0.25)

    def test_fsi_coupling_residual_growth_rejection_factor_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--fsi-coupling-residual-growth-rejection-factor",
                "2.5",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.fsi_coupling_residual_growth_rejection_factor, 2.5)

    def test_fsi_velocity_constraint_solid_mobility_ratio_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--fsi-velocity-constraint-solid-mobility-ratio",
                "2.5",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.fsi_velocity_constraint_solid_mobility_ratio, 2.5)
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        self.assertIn('"fsi_velocity_constraint_solid_mobility_ratio"', source)
        self.assertIn(
            "velocity_constraint_solid_mobility_ratio=fsi_velocity_constraint_solid_mobility_ratio",
            source,
        )
        self.assertIn(
            '"fsi_velocity_constraint_solid_mobility_ratio": fsi_velocity_constraint_solid_mobility_ratio',
            source,
        )

    def test_fsi_constraint_force_solid_mobility_ratio_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--fsi-constraint-force-solid-mobility-ratio",
                "2.5",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.fsi_constraint_force_solid_mobility_ratio, 2.5)
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        self.assertIn('"fsi_constraint_force_solid_mobility_ratio"', source)
        self.assertIn(
            "constraint_force_solid_mobility_ratio=fsi_constraint_force_solid_mobility_ratio",
            source,
        )
        self.assertIn(
            '"fsi_constraint_force_solid_mobility_ratio": fsi_constraint_force_solid_mobility_ratio',
            source,
        )
        self.assertIn("constraint_force_mobility_scale_delta", source)

    def test_interface_reaction_robin_impedance_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--interface-reaction-robin-impedance-ns-m",
                "25000",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.interface_reaction_robin_impedance_ns_m, 25000.0)

    def test_interface_reaction_robin_matrix_impedance_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--interface-reaction-robin-matrix-impedance-ns-m",
                "75000",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.interface_reaction_robin_matrix_impedance_ns_m, 75000.0)

    def test_interface_reaction_robin_target_mode_selects_target_force(self) -> None:
        raw_target = (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
        stabilized_target = (10.0, -20.0, 30.0, -40.0, 50.0, -60.0)

        self.assertEqual(
            interface_reaction_target_for_mode(
                "stabilized",
                raw_target_force_n=raw_target,
                stabilized_target_force_n=stabilized_target,
            ),
            stabilized_target,
        )
        self.assertEqual(
            interface_reaction_target_for_mode(
                "physical",
                raw_target_force_n=raw_target,
                stabilized_target_force_n=stabilized_target,
            ),
            raw_target,
        )
        with self.assertRaisesRegex(ValueError, "interface-reaction-robin-target-mode"):
            interface_reaction_target_for_mode(
                "bad",
                raw_target_force_n=raw_target,
                stabilized_target_force_n=stabilized_target,
            )

    def test_interface_reaction_robin_target_mode_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--interface-reaction-robin-target-mode",
                "physical",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.interface_reaction_robin_target_mode, "physical")

    def test_robin_previous_velocity_uses_step_start_on_cold_start(self) -> None:
        step_start_velocity = (0.0, 0.0, -2.0e-3, 0.0, 0.0, 1.0e-3)
        previous_velocity = (0.0, 0.0, -1.0e-3, 0.0, 0.0, 2.0e-3)

        cold_state = InterfaceReactionRelaxationState(previous_velocity_mps=None)
        warm_state = InterfaceReactionRelaxationState(previous_velocity_mps=previous_velocity)

        self.assertEqual(
            robin_previous_velocity_for_step(cold_state, step_start_velocity),
            step_start_velocity,
        )
        self.assertEqual(
            robin_previous_velocity_for_step(warm_state, step_start_velocity),
            previous_velocity,
        )

    def test_run_rejects_invalid_interface_reaction_robin_impedance(self) -> None:
        for value in ("nan", "inf", "-1.0"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp) / "out"
                    source_config = Path(tmp) / "simulation_config.json"
                    source_config.write_text(
                        json.dumps(
                            {
                                "analysis_settings": {"time_step_s": 5.0e-4},
                                "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                            }
                        ),
                        encoding="utf-8",
                    )
                    args = parse_args(
                        [
                            "--source-config",
                            str(source_config),
                            "--output-dir",
                            str(output_dir),
                            "--interface-reaction-robin-impedance-ns-m",
                            value,
                            "--preflight-only",
                        ]
                    )

                    with self.assertRaisesRegex(ValueError, "robin-impedance"):
                        run(args)

    def test_run_rejects_invalid_interface_reaction_robin_matrix_impedance(self) -> None:
        for value in ("nan", "inf", "-1.0"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp) / "out"
                    source_config = Path(tmp) / "simulation_config.json"
                    source_config.write_text(
                        json.dumps(
                            {
                                "analysis_settings": {"time_step_s": 5.0e-4},
                                "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                            }
                        ),
                        encoding="utf-8",
                    )
                    args = parse_args(
                        [
                            "--source-config",
                            str(source_config),
                            "--output-dir",
                            str(output_dir),
                            "--interface-reaction-robin-matrix-impedance-ns-m",
                            value,
                            "--preflight-only",
                        ]
                    )

                    with self.assertRaisesRegex(ValueError, "robin-matrix-impedance"):
                        run(args)

    def test_sharp_mode_rejects_legacy_robin_impedance_options_until_marker_robin_exists(
        self,
    ) -> None:
        for flag in (
            "--interface-reaction-robin-impedance-ns-m",
            "--interface-reaction-robin-matrix-impedance-ns-m",
        ):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp) / "out"
                    source_config = Path(tmp) / "simulation_config.json"
                    source_config.write_text(
                        json.dumps(
                            {
                                "analysis_settings": {"time_step_s": 5.0e-4},
                                "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                            }
                        ),
                        encoding="utf-8",
                    )
                    args = parse_args(
                        [
                            "--source-config",
                            str(source_config),
                            "--output-dir",
                            str(output_dir),
                            "--fsi-coupling-mode",
                            FSI_COUPLING_MODE_HIBM_MPM_SHARP,
                            "--pressure-solver",
                            "fv_cg",
                            flag,
                            "125000",
                            "--preflight-only",
                        ]
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "hibm_mpm_sharp.*marker-level Robin",
                    ):
                        run(args)

    def test_run_rejects_invalid_fsi_coupling_target_map_relaxation(self) -> None:
        for value in ("nan", "0.0", "1.1"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp) / "out"
                    source_config = Path(tmp) / "simulation_config.json"
                    source_config.write_text(
                        json.dumps(
                            {
                                "analysis_settings": {"time_step_s": 5.0e-4},
                                "domains": {"fluid": {"grid_size_m": 2.5e-3}},
                            }
                        ),
                        encoding="utf-8",
                    )
                    args = parse_args(
                        [
                            "--source-config",
                            str(source_config),
                            "--output-dir",
                            str(output_dir),
                            "--fsi-coupling-target-map-relaxation",
                            value,
                            "--preflight-only",
                        ]
                    )

                    with self.assertRaisesRegex(ValueError, "target-map-relaxation"):
                        run(args)

    def test_tail_refinement_requires_graded_grid_and_positive_spacing(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--use-tail-refinement"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--use-graded-grid",
                "--use-tail-refinement",
                "--tail-refinement-target-spacing-m",
                "0",
            ],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--use-graded-grid",
                "--use-tail-refinement",
                "--tail-refinement-padding-m",
                "-1e-3",
            ],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--use-graded-grid",
                "--use-tail-refinement",
                "--tail-refinement-target-spacing-m",
                "0.0015",
                "--tail-refinement-padding-m",
                "0.004",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.use_tail_refinement)
        self.assertAlmostEqual(args.tail_refinement_target_spacing_m, 0.0015)
        self.assertAlmostEqual(args.tail_refinement_padding_m, 0.004)

    def test_graded_grid_max_cells_zero_disables_guard_but_negative_is_invalid(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--graded-grid-max-cells", "0"]):
            args = parse_args()
        self.assertIsNone(args.graded_grid_max_cells)

        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--checkpoint-every-step",
                "--resume-from-checkpoint",
                "--checkpoint-path",
                "restart.npz",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.checkpoint_every_step)
        self.assertTrue(args.resume_from_checkpoint)
        self.assertEqual(args.checkpoint_path, "restart.npz")

        with patch("sys.argv", ["squid_soft_robot.py", "--graded-grid-max-cells", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_run_checkpoint_round_trips_dynamic_taichi_state(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.01, -0.01, -0.01),
            fluid_bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(6, 6, 6),
            dt_s=1.0e-4,
            water_density_kgm3=1000.0,
            water_viscosity_pa_s=1.0e-3,
        )
        simulator = ReducedSquidFSI(spec, runtime=runtime)
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            grid_nodes=spec.grid_nodes,
            runtime=runtime,
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        simulator.set_structure_state(
            time_s=0.02,
            pressure_pa=12.0,
            hydraulic_pressure_pa=3.0,
            main_displacement_z_m=-1.0e-4,
            main_velocity_z_mps=-2.0e-3,
            tail_displacement_z_m=5.0e-5,
            tail_velocity_z_mps=1.0e-3,
            volume_flux_m3s=4.0e-7,
            nozzle_velocity_z_mps=-1.5e-2,
        )
        simulator.set_interface_reaction(
            primary_force_n=(1.0, 2.0, 3.0),
            secondary_force_n=(-1.0, -2.0, -3.0),
        )
        velocity = np.zeros((*spec.grid_nodes, 3), dtype=np.float32)
        velocity[2, 2, 2] = (0.1, 0.2, 0.3)
        pressure = np.zeros(spec.grid_nodes, dtype=np.float32)
        pressure[1, 2, 3] = 7.5
        simulator.fluid.velocity.from_numpy(velocity)
        simulator.fluid.velocity_prev.from_numpy(velocity * 0.5)
        simulator.fluid.pressure.from_numpy(pressure)
        initial_solid_x = solid.x.to_numpy().copy()
        initial_solid_v = np.array([[0.03, -0.02, 0.01]], dtype=np.float32)
        solid.v.from_numpy(initial_solid_v)
        state = InterfaceReactionRelaxationState(
            previous_residual_n=(1.0, 0.0, -1.0, 2.0, 0.0, -2.0),
            previous_velocity_mps=(0.01, 0.0, -0.01, -0.02, 0.0, 0.02),
            relaxation=0.25,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "restart.npz"
            args = SimpleNamespace(solid_model="neo_hookean_mpm")
            write_run_checkpoint(
                checkpoint_path,
                completed_step=3,
                step_count=200,
                full_pressure_waveform_steps=4000,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                interface_reaction_state=state,
            )
            simulator.set_structure_state(
                time_s=0.0,
                pressure_pa=0.0,
                hydraulic_pressure_pa=0.0,
                main_displacement_z_m=0.0,
                main_velocity_z_mps=0.0,
                tail_displacement_z_m=0.0,
                tail_velocity_z_mps=0.0,
                volume_flux_m3s=0.0,
                nozzle_velocity_z_mps=0.0,
            )
            simulator.fluid.velocity.from_numpy(np.zeros_like(velocity))
            simulator.fluid.velocity_prev.from_numpy(np.zeros_like(velocity))
            simulator.fluid.pressure.from_numpy(np.zeros_like(pressure))
            solid.x.from_numpy(np.zeros_like(initial_solid_x))
            solid.v.from_numpy(np.zeros_like(initial_solid_v))

            completed_step, restored_state = load_run_checkpoint(
                checkpoint_path,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_run_checkpoint(
                    checkpoint_path,
                    args=SimpleNamespace(
                        solid_model="neo_hookean_mpm",
                        cg_tolerance=2.0e-6,
                    ),
                    simulator=simulator,
                    solid_mpm=solid,
                )

        self.assertEqual(completed_step, 3)
        self.assertAlmostEqual(float(simulator.time_s[None]), 0.02, delta=1.0e-7)
        np.testing.assert_allclose(simulator.fluid.velocity.to_numpy(), velocity, atol=1.0e-7)
        np.testing.assert_allclose(simulator.fluid.pressure.to_numpy(), pressure, atol=1.0e-7)
        np.testing.assert_allclose(solid.x.to_numpy(), initial_solid_x, atol=1.0e-8)
        np.testing.assert_allclose(solid.v.to_numpy(), initial_solid_v, atol=1.0e-8)
        self.assertEqual(restored_state.previous_residual_n, state.previous_residual_n)
        self.assertEqual(restored_state.previous_velocity_mps, state.previous_velocity_mps)
        self.assertAlmostEqual(restored_state.relaxation, 0.25)

    def test_checkpoint_fingerprint_includes_coupling_mode_and_solver_policy(self) -> None:
        required_fields = {
            "fsi_coupling_mode",
            "cg_preconditioner",
            "pressure_solve_failure_policy",
            "reuse_accepted_fsi_trial_state",
            "fsi_coupling_adaptive_iterations_max",
            "fsi_coupling_adaptive_iterations_residual_threshold_n",
            "fsi_coupling_adaptive_iterations_cfl_threshold",
            "fsi_coupling_same_step_rerun_iterations_max",
            "fsi_coupling_same_step_rerun_residual_threshold_n",
            "fsi_coupling_residual_continuation_iterations_max",
            "fsi_coupling_residual_continuation_threshold_n",
            "fsi_coupling_residual_continuation_rebound_secant_from_best",
            "fsi_coupling_residual_continuation_rebound_secant_factor",
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max",
            "fsi_coupling_trial_interior_divergence_tolerance",
            "fsi_coupling_max_accepted_residual_n",
            "fsi_coupling_trust_region_force_increment_n",
            "fsi_coupling_trust_region_adaptive",
            "fsi_coupling_trust_region_shrink_factor",
            "fsi_coupling_trust_region_growth_factor",
            "fsi_coupling_trust_region_rebound_factor",
            "fsi_coupling_trust_region_rebound_backtrack",
            "fsi_coupling_trust_region_rebound_stop_factor",
            "fsi_coupling_trust_region_rebound_stop_max_residual_n",
        }

        self.assertTrue(required_fields.issubset(CHECKPOINT_ARG_FINGERPRINT_FIELDS))

    def test_checkpoint_fingerprint_allows_requested_step_extension(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.01, -0.01, -0.01),
            fluid_bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(6, 6, 6),
            dt_s=1.0e-4,
            water_density_kgm3=1000.0,
            water_viscosity_pa_s=1.0e-3,
        )
        args = SimpleNamespace(
            source_config="dummy.json",
            solid_model="neo_hookean_mpm",
            pressure_t0_s=0.0,
            pressure_t1_s=0.0005,
            pressure_t2_s=0.001,
            pressure_p0_pa=0.0,
            pressure_p1_pa=8000.0,
            pressure_p2_pa=8000.0,
        )
        metadata = {
            "run_fingerprint": checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=4000,
            )
        }

        validate_checkpoint_run_fingerprint(
            metadata,
            args=args,
            spec=spec,
            step_count=2,
            full_pressure_waveform_steps=4000,
        )

    def test_checkpoint_fingerprint_rejects_pressure_schedule_change(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.01, -0.01, -0.01),
            fluid_bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(6, 6, 6),
            dt_s=1.0e-4,
            water_density_kgm3=1000.0,
            water_viscosity_pa_s=1.0e-3,
        )
        args = SimpleNamespace(
            source_config="dummy.json",
            solid_model="neo_hookean_mpm",
            pressure_t0_s=0.0,
            pressure_t1_s=0.0005,
            pressure_t2_s=0.001,
            pressure_p0_pa=0.0,
            pressure_p1_pa=8000.0,
            pressure_p2_pa=8000.0,
        )
        metadata = {
            "run_fingerprint": checkpoint_run_fingerprint(
                args=args,
                spec=spec,
                step_count=1,
                full_pressure_waveform_steps=4000,
            )
        }
        changed_args = SimpleNamespace(
            source_config="dummy.json",
            solid_model="neo_hookean_mpm",
            pressure_t0_s=0.0,
            pressure_t1_s=0.0005,
            pressure_t2_s=0.001,
            pressure_p0_pa=0.0,
            pressure_p1_pa=9000.0,
            pressure_p2_pa=9000.0,
        )

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_checkpoint_run_fingerprint(
                metadata,
                args=changed_args,
                spec=spec,
                step_count=2,
                full_pressure_waveform_steps=4000,
            )

    def test_checkpoint_interface_state_rejects_nonfinite_metadata(self) -> None:
        base_state = {
            "previous_residual_n": (1.0, 0.0, -1.0, 2.0, 0.0, -2.0),
            "previous_velocity_mps": (0.01, 0.0, -0.01, -0.02, 0.0, 0.02),
            "relaxation": 0.25,
        }

        cases = (
            ("previous_residual_n", (1.0, float("nan"), 0.0, 0.0, 0.0, 0.0)),
            ("previous_velocity_mps", (1.0, 0.0, float("inf"), 0.0, 0.0, 0.0)),
            ("relaxation", float("nan")),
            ("relaxation", None),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name):
                metadata = dict(base_state)
                metadata[field_name] = value
                with self.assertRaisesRegex(ValueError, field_name):
                    _interface_state_from_checkpoint(metadata)

    def test_resume_history_rows_for_checkpoint_truncates_ahead_history(self) -> None:
        rows = [{"step": "1"}, {"step": "2"}, {"step": "3"}]

        self.assertEqual(
            resume_history_rows_for_checkpoint(rows, completed_step=2),
            rows[:2],
        )
        with self.assertRaisesRegex(ValueError, "at least the checkpointed steps"):
            resume_history_rows_for_checkpoint(rows[:1], completed_step=2)

    def test_write_csv_preserves_union_columns_for_schema_evolving_history(
        self,
    ) -> None:
        rows = [
            {
                "step": 1,
                "time_s": 0.1,
                "fsi_coupling_mode": FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
            },
            {
                "step": 2,
                "time_s": 0.2,
                "fsi_coupling_mode": FSI_COUPLING_MODE_HIBM_MPM_SHARP,
                "pre_projection_divergence_measured": True,
                "fsi_coupling_mode_paper_hibm_mpm": True,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.csv"
            write_csv(history_path, rows)
            header = history_path.read_text(encoding="utf-8").splitlines()[0]
            read_back = read_csv_rows(history_path)

        self.assertEqual(
            header.split(","),
            [
                "step",
                "time_s",
                "fsi_coupling_mode",
                "pre_projection_divergence_measured",
                "fsi_coupling_mode_paper_hibm_mpm",
            ],
        )
        self.assertEqual(read_back[0]["pre_projection_divergence_measured"], "")
        self.assertEqual(read_back[1]["pre_projection_divergence_measured"], "True")

    def test_resume_history_checkpoint_alignment_requires_step_and_time_match(self) -> None:
        rows = [{"step": "1", "time_s": "0.1"}, {"step": "2", "time_s": "0.2"}]

        validate_resume_history_checkpoint_alignment(
            rows,
            completed_step=2,
            checkpoint_time_s=0.2,
            dt_s=0.1,
        )
        with self.assertRaisesRegex(ValueError, "step does not match"):
            validate_resume_history_checkpoint_alignment(
                [{"step": "9", "time_s": "0.2"}],
                completed_step=1,
                checkpoint_time_s=0.2,
                dt_s=0.1,
            )
        with self.assertRaisesRegex(ValueError, "time_s does not match"):
            validate_resume_history_checkpoint_alignment(
                rows,
                completed_step=2,
                checkpoint_time_s=0.25,
                dt_s=0.1,
            )

    def test_default_output_directory_is_gitignored(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py"]):
            args = parse_args()
        repo_root = Path(__file__).resolve().parents[1]
        output_path = Path(args.output_dir).resolve()
        candidate = output_path / "run_process.json"

        result = subprocess.run(
            ["git", "check-ignore", str(candidate.relative_to(repo_root))],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runner_has_no_old_feedback_api_names(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        forbidden_tokens = (
            "Feedback",
            "feedback",
            "fluid_pressure_force",
            "fluid_pressure_feedback",
            "set_fluid_pressure_feedback",
            "pressure_feedback_z_n",
            "pressure_feedback_power_w",
            "fluid_to_solid_pressure_feedback",
            "fluid_feedback_",
            "fluid-feedback",
            "pressure_force_scale",
            "pressure-force-scale",
            "main_interface_reaction_force_z_n",
            "tail_interface_reaction_force_z_n",
            "main_force_z_n",
            "tail_force_z_n",
            "reaction_force_z_n",
            "primary_force_x_n",
            "primary_force_y_n",
            "primary_force_z_n",
            "secondary_force_x_n",
            "secondary_force_y_n",
            "secondary_force_z_n",
            "fsi_coupling_trial_force_history_z_n",
            "fsi_coupling_target_force_history_z_n",
            "fsi_coupling_residual_history_z_n",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source, msg=token)

    def test_validation_gates_real_fluid_flux_and_projection_divergence(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn('"final_outlet_to_fsi_volume_source_ratio_physical"', source)
        self.assertIn("physical_outlet_to_fsi_volume_source_passes", source)
        self.assertIn("args.min_outlet_to_main_volume_flux_ratio", source)
        self.assertIn('"required_min_outlet_to_main_volume_flux_ratio"', source)
        self.assertIn('"projection_divergence_below_tolerance"', source)
        self.assertIn('"pressure_outlet_velocity_to_source_ratio_near_one"', source)
        self.assertIn("pressure_outlet_source_ratio_passes", source)
        self.assertIn("args.pressure_outlet_source_ratio_tolerance", source)
        self.assertIn("args.projection_divergence_tolerance", source)
        self.assertNotIn("max_div_l2 <= float(args.projection_divergence_tolerance)", source)
        self.assertIn("max_interior_div_l2 <= float(args.projection_divergence_tolerance)", source)
        self.assertIn('"max_interior_divergence_l2"', source)
        self.assertIn("validation uses the sampled outlet-to-FSI-volume-source flux ratio", source)

    def test_active_squid_case_is_not_legacy_wrapper(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertNotIn("squid_soft_robot_latest_core_20260603", source)
        self.assertNotIn("run_squid_latest_core", source)

    def test_interface_reaction_summary_key_is_not_double_renamed(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn('"fluid_to_solid_interface_reaction_enabled"', source)
        self.assertNotIn("interface_reaction_interface_reaction", source)

    def test_runner_reports_fixed_point_interface_map_amplification(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("fixed_point_result.interface_map_amplification_max", source)
        self.assertIn("fixed_point_result.residual_jacobian_amplification_max", source)
        self.assertIn("fixed_point_result.physical_interface_map_amplification_max", source)
        self.assertIn("fixed_point_result.physical_residual_jacobian_amplification_max", source)
        self.assertIn("fixed_point_result.diagnostic_interface_map_amplification_max", source)
        self.assertIn("fixed_point_result.diagnostic_residual_jacobian_amplification_max", source)
        self.assertIn("target_map_relaxation=fsi_coupling_target_map_relaxation", source)
        self.assertIn("diagnostic_target_force_n=raw_target_force_n", source)
        self.assertIn('"fsi_coupling_interface_map_amplification"', source)
        self.assertIn('"fsi_coupling_residual_jacobian_amplification"', source)
        self.assertIn('"fsi_coupling_physical_interface_map_amplification"', source)
        self.assertIn('"fsi_coupling_physical_residual_jacobian_amplification"', source)
        self.assertIn('"fsi_coupling_raw_interface_map_amplification"', source)
        self.assertIn('"fsi_coupling_raw_residual_jacobian_amplification"', source)
        self.assertIn('"max_fsi_coupling_interface_map_amplification"', source)
        self.assertIn('"max_fsi_coupling_residual_jacobian_amplification"', source)
        self.assertIn('"max_fsi_coupling_physical_interface_map_amplification"', source)
        self.assertIn('"max_fsi_coupling_physical_residual_jacobian_amplification"', source)
        self.assertIn('"max_fsi_coupling_raw_interface_map_amplification"', source)
        self.assertIn('"max_fsi_coupling_raw_residual_jacobian_amplification"', source)

    def test_fsi_physical_interface_map_stability_gate_rejects_unstable_raw_map(self) -> None:
        self.assertTrue(
            fsi_physical_interface_map_stability_passes(
                fsi_coupling_enabled=True,
                fsi_coupling_iterations=3,
                max_physical_interface_map_amplification=1.0,
                measurement_sample_count=1,
            )
        )
        self.assertFalse(
            fsi_physical_interface_map_stability_passes(
                fsi_coupling_enabled=True,
                fsi_coupling_iterations=3,
                max_physical_interface_map_amplification=1.0001,
                measurement_sample_count=1,
            )
        )
        self.assertFalse(
            fsi_physical_interface_map_stability_passes(
                fsi_coupling_enabled=True,
                fsi_coupling_iterations=3,
                max_physical_interface_map_amplification=float("nan"),
                measurement_sample_count=1,
            )
        )
        self.assertTrue(
            fsi_physical_interface_map_stability_passes(
                fsi_coupling_enabled=False,
                fsi_coupling_iterations=3,
                max_physical_interface_map_amplification=float("nan"),
                measurement_sample_count=0,
            )
        )
        self.assertFalse(
            fsi_physical_interface_map_stability_passes(
                fsi_coupling_enabled=True,
                fsi_coupling_iterations=1,
                max_physical_interface_map_amplification=float("nan"),
                measurement_sample_count=0,
            )
        )

    def test_fsi_physical_interface_map_stability_report_marks_unmeasured_and_masked(
        self,
    ) -> None:
        unmeasured = fsi_physical_interface_map_stability_report(
            fsi_coupling_enabled=True,
            fsi_coupling_iterations=1,
            max_physical_interface_map_amplification=0.0,
            measurement_sample_count=0,
            raw_interface_map_strict_physical=True,
        )
        self.assertFalse(unmeasured["passes"])
        self.assertFalse(unmeasured["measured"])
        self.assertEqual(unmeasured["status"], "unmeasured")

        masked = fsi_physical_interface_map_stability_report(
            fsi_coupling_enabled=True,
            fsi_coupling_iterations=3,
            max_physical_interface_map_amplification=0.25,
            measurement_sample_count=1,
            raw_interface_map_strict_physical=False,
        )
        self.assertFalse(masked["passes"])
        self.assertTrue(masked["measured"])
        self.assertEqual(masked["status"], "masked_by_stabilizer")

    def test_completed_checks_include_raw_physical_interface_map_stability_gate(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn('"fsi_physical_interface_map_stable"', source)
        self.assertIn("fsi_physical_interface_map_stability_report", source)
        self.assertIn(
            "max_fsi_coupling_raw_interface_map_amplification",
            source,
        )
        self.assertIn(
            "max_fsi_coupling_raw_interface_map_amplification_sample_count",
            source,
        )
        self.assertIn('"fsi_physical_interface_map_stability"', source)
        self.assertIn('"fsi_coupling_raw_interface_map_strict_physical"', source)

    def test_fsi_coupling_mode_default_is_generic_hibm_mpm_sharp(self) -> None:
        args = parse_args([])

        self.assertEqual(
            args.fsi_coupling_mode,
            FSI_COUPLING_MODE_HIBM_MPM_SHARP,
        )
        report = fsi_coupling_mode_report(args.fsi_coupling_mode)
        self.assertFalse(report["legacy"])
        self.assertTrue(report["paper_hibm_mpm"])
        self.assertFalse(report["region_pair_reaction_diagnostic_only"])
        self.assertNotIn("main_tail", json.dumps(report))
        self.assertNotIn("main/tail", json.dumps(report))

        explicit_legacy = parse_args(
            ["--fsi-coupling-mode", FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED]
        )
        legacy_report = fsi_coupling_mode_report(explicit_legacy.fsi_coupling_mode)
        self.assertTrue(legacy_report["legacy"])

    def test_squid_history_uses_generic_region_pair_reaction_key(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("region_pair_reaction_diagnostic_only", source)
        self.assertNotIn("main_tail_region_reaction_diagnostic_only", source)

    def test_squid_case_wires_core_fsi_coupling_mode_without_owning_solver(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        args = parse_args(["--fsi-coupling-mode", FSI_COUPLING_MODE_HIBM_MPM_SHARP])
        self.assertEqual(args.fsi_coupling_mode, FSI_COUPLING_MODE_HIBM_MPM_SHARP)
        self.assertIn('"fsi_coupling_mode"', source)
        self.assertIn('"fsi_coupling_mode_report"', source)
        self.assertIn("require_implemented_fsi_coupling_mode", source)

    def test_sharp_mode_does_not_enable_legacy_reduced_fixed_point(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertTrue(
            legacy_projected_reduced_fsi_coupling_enabled(
                fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
                solid_model="neo_hookean_mpm",
                fsi_coupling_iterations=2,
            )
        )
        self.assertFalse(
            legacy_projected_reduced_fsi_coupling_enabled(
                fsi_coupling_mode=FSI_COUPLING_MODE_HIBM_MPM_SHARP,
                solid_model="neo_hookean_mpm",
                fsi_coupling_iterations=8,
            )
        )
        self.assertFalse(
            legacy_projected_reduced_fsi_coupling_enabled(
                fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
                solid_model="neo_hookean_mpm",
                fsi_coupling_iterations=1,
            )
        )
        self.assertIn("legacy_projected_reduced_fsi_coupling_enabled", source)
        self.assertIn("fsi_coupling_mode=fsi_coupling_mode", source)

    def test_legacy_reduced_strong_coupling_is_not_ibm_correction_iterations(self) -> None:
        self.assertFalse(
            legacy_projected_reduced_fsi_coupling_enabled(
                fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
                solid_model="tri_mooney_shell_mpm",
                fsi_coupling_iterations=1,
            )
        )
        self.assertTrue(
            legacy_projected_reduced_fsi_coupling_enabled(
                fsi_coupling_mode=FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
                solid_model="tri_mooney_shell_mpm",
                fsi_coupling_iterations=6,
            )
        )

    def test_squid_case_builds_sharp_coupling_from_core_taichi_fields(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 8
        solid.v[0] = (0.0, 0.0, -0.125)

        coupling = build_hibm_mpm_sharp_coupling_state(
            fluid=fluid,
            solid_mpm=solid,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        self.assertEqual(coupling.markers.marker_count, 1)
        self.assertEqual(coupling.markers.marker_region_id(0), 8)
        self.assertEqual(coupling.markers.marker_velocity_mps(0), (0.0, 0.0, -0.125))
        self.assertIn("HibmMpmSharpCouplingState", source)
        self.assertIn("surface_velocity_mps=solid_mpm.v", source)
        self.assertNotIn("build_hibm_mpm_sharp_coupling_state_from_numpy", source)

    def test_squid_case_preserves_tri_mooney_triangle_projection_topology(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.25, 0.25, 0.5],
                    [0.75, 0.25, 0.5],
                    [0.25, 0.75, 0.5],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        solid = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.01,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([202], dtype=np.int32),
            primary_region_id=101,
            secondary_region_id=202,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=2.0,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        coupling = build_hibm_mpm_sharp_coupling_state(
            fluid=fluid,
            solid_mpm=solid,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        report = coupling.ib_search.search_and_classify(
            coupling.markers,
            search_radius_m=0.08,
            interior_probe_distance_m=0.05,
        )

        self.assertEqual(coupling.markers.projection_triangle_count, 1)
        self.assertGreaterEqual(report.near_boundary_node_count, 1)
        np.testing.assert_allclose(
            coupling.ib_search.boundary_point_m((2, 2, 4)),
            (0.3125, 0.3125, 0.5),
            atol=1.0e-6,
        )

    def test_sharp_case_drives_main_membrane_via_far_pressure_closure(
        self,
    ) -> None:
        # Contract updated 2026-06-11 (S2-A wiring): the old contract pinned a
        # direct solid area load `(0, 0, -pressure_pa)` as the waveform drive.
        # The 2-second run forensics proved that drive path is structurally
        # one-way (the air side of the main membrane is outside the water
        # domain, so two-sided marker sampling never validates and the solid
        # free-falls without added-mass back-pressure). The waveform now
        # enters as the known far-side pressure of the marker traction
        # closure, and the direct area load is forbidden to prevent double
        # counting the air pressure.
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_solid_step = source.split("def advance_sharp_solid_substeps():", 1)[
            1
        ].split("fluid_wall_started_at", 1)[0]

        self.assertNotIn("solid_mpm.add_region_area_load(", sharp_solid_step)
        self.assertIn("solid_mpm.advance_with_external_forces(", sharp_solid_step)
        self.assertNotIn("solid_mpm.add_region_normal_pressure(", sharp_solid_step)
        self.assertNotIn("hibm_mpm_sharp case runner currently requires", source)
        sharp_advance_call = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("sharp_summary = hibm_mpm_sharp_step_summary", 1)[0]
        self.assertIn("far_pressure_region_id=pressure_load_region_id", sharp_advance_call)
        self.assertIn("far_pressure_pa=pressure_pa", sharp_advance_call)
        self.assertIn(
            "far_pressure_side_normal_sign=pressure_far_side_normal_sign",
            sharp_advance_call,
        )
        self.assertIn(
            "pressure_outlet_zmin=pressure_outlet_zmin_enabled",
            sharp_advance_call,
        )
        self.assertNotIn(
            "pressure_outlet_zmin=not args.disable_pressure_outlet_zmin",
            sharp_advance_call,
        )
        self.assertIn(
            "far_pressure_inside_probe_max_multiplier=12.0", sharp_advance_call
        )
        self.assertIn(
            "one_sided_pressure_region_id=secondary_shell_region_id",
            sharp_advance_call,
        )
        self.assertIn("one_sided_reference_pressure_pa=0.0", sharp_advance_call)
        self.assertIn("one_sided_probe_max_multiplier=12.0", sharp_advance_call)
        sharp_pressure_setup = source.split("if sharp_case_runner_enabled:", 1)[
            1
        ].split("def advance_sharp_solid_substeps():", 1)[0]
        self.assertIn("pressure_schedule_step_end_pa(", sharp_pressure_setup)
        self.assertIn("current_time_s", sharp_pressure_setup)
        self.assertIn("spec.dt_s", sharp_pressure_setup)
        self.assertIn("AxisAlignedBoundary.pressure_outlet", source)
        self.assertIn("pressure_outlet_boundary_report", source)
        self.assertIn('"pressure_outlet_boundary"', source)
        self.assertNotIn(
            "pressure_schedule_pa(current_time_s, spec)",
            sharp_pressure_setup,
        )

    def test_sharp_case_forwards_divergence_cleanup_to_core_projection(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_call = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("sharp_summary = hibm_mpm_sharp_step_summary", 1)[0]

        self.assertIn(
            "divergence_cleanup_iterations=projection_divergence_cleanup_iterations",
            sharp_call,
        )
        self.assertIn(
            "divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation)",
            sharp_call,
        )
        self.assertIn("fluid_substeps=step_fluid_substeps", sharp_call)
        self.assertIn(
            "fluid_advection_scheme=str(args.fluid_advection_scheme)",
            sharp_call,
        )
        self.assertIn(
            "pressure_solve_failure_policy=str(args.pressure_solve_failure_policy)",
            sharp_call,
        )
        self.assertNotIn("fluid_substeps=1", sharp_call)

    def test_pressure_schedule_is_reported_as_prescribed_boundary_drive(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        self.assertIn(
            "prescribed_pressure_or_flow_boundary=max_abs_pressure_load_pa > 0.0",
            source,
        )

    def test_pressure_solve_failure_policy_is_explicit_cli_state(self) -> None:
        args = parse_args(["--pressure-solve-failure-policy", "report"])

        self.assertEqual(args.pressure_solve_failure_policy, "report")
        with patch(
            "sys.argv",
            ["squid_soft_robot.py", "--pressure-solve-failure-policy", "bad"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_neo_hookean_solid_mpm_bounds_expand_to_surface_metadata(self) -> None:
        metadata = {
            "solid_centroid_bounds_min_m": (-0.05, -0.02, 1.01),
            "solid_centroid_bounds_max_m": (0.01, 0.05, 1.043),
        }

        bounds_min, bounds_max = solid_mpm_bounds_from_surface_metadata(
            metadata,
            fallback_bounds_min_m=(-0.09, -0.044, 0.9),
            fallback_bounds_max_m=(0.029, 0.076, 1.04),
            padding_m=0.015,
        )

        self.assertEqual(bounds_min, (-0.09, -0.044, 0.9))
        for actual, expected in zip(bounds_max, (0.029, 0.076, 1.058), strict=True):
            self.assertAlmostEqual(actual, expected)

        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        self.assertIn("solid_mpm_bounds_from_surface_metadata", source)
        self.assertIn("bounds_min_m=solid_mpm_bounds_min_m", source)
        self.assertIn("bounds_max_m=solid_mpm_bounds_max_m", source)

    def test_solid_mpm_bounds_padding_uses_farfield_spacing_on_graded_grid(self) -> None:
        padding = solid_mpm_bounds_padding_distance_m(
            fluid_grid_axis_max_spacing_m=(0.0029, 0.0030, 0.0028),
            estimated_solid_particle_spacing_m=0.00103,
        )

        self.assertAlmostEqual(padding, 0.009)
        self.assertGreater(padding, 3.0 * 0.00103)

        with self.assertRaises(ValueError):
            solid_mpm_bounds_padding_distance_m(
                fluid_grid_axis_max_spacing_m=(0.0030, 0.0, 0.0028),
                estimated_solid_particle_spacing_m=0.00103,
            )

    def test_sharp_case_row_uses_hibm_marker_fields_not_projected_ibm(self) -> None:
        sample_report = {
            "time_s": 1.0e-3,
            "pressure_load_pa": 2000.0,
            "hydraulic_pressure_pa": 0.0,
            "main_displacement_z_m": -1.0e-5,
            "main_velocity_z_mps": -2.0e-3,
            "tail_displacement_z_m": -2.0e-6,
            "tail_velocity_z_mps": -4.0e-4,
            "volume_flux_m3s": 1.0e-7,
            "nozzle_velocity_z_mps": -1.0e-2,
            "lip_flow_negative_z_m3s": 1.0e-7,
            "outlet_flow_negative_z_m3s": 1.0e-7,
            "downstream_flow_negative_z_m3s": 1.0e-7,
            "lip_sample_count": 4,
            "outlet_sample_count": 4,
            "downstream_sample_count": 4,
            "max_fluid_speed_mps": 0.01,
            "cfl": 0.1,
            "divergence_l2": 1.0e-4,
            "divergence_max_abs": 2.0e-4,
            "interior_divergence_l2": 1.0e-4,
            "interior_divergence_max_abs": 2.0e-4,
            "pre_projection_divergence_l2": 3.0e-4,
            "pre_projection_divergence_max_abs": 4.0e-4,
            "projection_divergence_l2": 1.0e-4,
            "projection_divergence_max_abs": 2.0e-4,
            "projection_to_pre_divergence_l2_ratio": 1.0 / 3.0,
            "post_boundary_divergence_l2": 1.5e-4,
            "post_boundary_divergence_max_abs": 2.5e-4,
            "post_boundary_to_pre_divergence_l2_ratio": 0.5,
            "post_constraint_divergence_l2": 1.0e-4,
            "post_constraint_divergence_max_abs": 2.0e-4,
            "post_constraint_to_pre_divergence_l2_ratio": 1.0 / 3.0,
            "pressure_correctable_divergence_l2": 7.0e-5,
            "pressure_correctable_divergence_max_abs": 8.0e-5,
            "pressure_correctable_divergence_cell_count": 12,
            "pressure_fixed_divergence_l2": 6.0e-5,
            "pressure_fixed_divergence_max_abs": 7.0e-5,
            "pressure_fixed_divergence_cell_count": 2,
            "interior_pressure_correctable_divergence_l2": 5.0e-5,
            "interior_pressure_correctable_divergence_max_abs": 6.0e-5,
            "interior_pressure_correctable_divergence_cell_count": 6,
            "interior_pressure_fixed_divergence_l2": 4.0e-5,
            "interior_pressure_fixed_divergence_max_abs": 5.0e-5,
            "interior_pressure_fixed_divergence_cell_count": 1,
        }
        sharp_summary = {
            "hibm_marker_primary_count": 3,
            "hibm_marker_secondary_count": 5,
            "hibm_marker_total_count": 8,
            "hibm_marker_primary_force_n": (1.0, 2.0, 3.0),
            "hibm_marker_secondary_force_n": (-0.25, 0.5, -1.0),
            "hibm_marker_total_force_n": (0.75, 2.5, 2.0),
            "hibm_marker_action_reaction_residual_n": 0.0,
            "hibm_mpm_scatter_action_reaction_residual_n": 0.0,
            "hibm_no_slip_residual_valid_marker_count": 8,
            "hibm_no_slip_residual_invalid_marker_count": 0,
            "hibm_no_slip_residual_l2_mps": 1.0e-5,
            "hibm_no_slip_residual_max_mps": 2.0e-5,
            "hibm_post_solid_kinematic_projection_applied": True,
            "hibm_post_solid_no_slip_residual_valid_marker_count": 8,
            "hibm_post_solid_no_slip_residual_invalid_marker_count": 0,
            "hibm_post_solid_no_slip_residual_l2_mps": 3.0e-6,
            "hibm_post_solid_no_slip_residual_max_mps": 4.0e-6,
            "hibm_velocity_dirichlet_invalid_reconstruction_count": 4,
            "hibm_velocity_dirichlet_invalid_no_fluid_sample_count": 1,
            "hibm_velocity_dirichlet_invalid_nonpositive_gap_count": 2,
            "hibm_velocity_dirichlet_invalid_node_behind_boundary_count": 0,
            "hibm_velocity_dirichlet_invalid_node_beyond_interior_count": 1,
            "hibm_ib_node_count": 11,
            "hibm_internal_obstacle_cell_count": 4,
            "hibm_pressure_neumann_max_raw_transmissibility_m": 25.0,
            "hibm_pressure_neumann_max_transmissibility_limit_m": 5.0,
            "hibm_pressure_neumann_transmissibility_capped_row_count": 2,
            "hibm_coupling_scheme": "explicit_loose",
            "hibm_added_mass_stability_status": "unmeasured",
            "hibm_added_mass_stability_measured": False,
            "hibm_added_mass_stabilization": "none",
            "hibm_semi_implicit_coupling_enabled": False,
            "hibm_semi_implicit_coupling_matrix_active": False,
            "hibm_pressure_correctable_divergence_l2": 7.0e-5,
            "hibm_pressure_correctable_divergence_max_abs": 8.0e-5,
            "hibm_pressure_correctable_divergence_cell_count": 12,
            "hibm_air_backed_cell_count": 17,
            "hibm_air_backed_component_count": 1,
            "hibm_air_backed_cell_volume_m3": 2.5e-5,
            "hibm_air_backed_seed_marker_count": 19,
            "hibm_air_backed_seed_missed_marker_count": 0,
            "hibm_air_backed_seed_fallback_cell_count": 4,
            "hibm_air_backed_reachability_barrier_cell_count": 23,
            "hibm_pressure_fixed_divergence_l2": 6.0e-5,
            "hibm_pressure_fixed_divergence_max_abs": 7.0e-5,
            "hibm_pressure_fixed_divergence_cell_count": 2,
            "hibm_interior_pressure_correctable_divergence_l2": 5.0e-5,
            "hibm_interior_pressure_correctable_divergence_max_abs": 6.0e-5,
            "hibm_interior_pressure_correctable_divergence_cell_count": 6,
            "hibm_interior_pressure_fixed_divergence_l2": 4.0e-5,
            "hibm_interior_pressure_fixed_divergence_max_abs": 5.0e-5,
            "hibm_interior_pressure_fixed_divergence_cell_count": 1,
        }
        projection_report = {
            "pressure_solver_requested": "fv_multigrid",
            "pressure_solver": "fv_cg",
            "pressure_solver_forced_to_fv_cg": True,
            "pressure_solver_force_reason": "hibm_pressure_neumann_requires_fv_solver",
            "pressure_nullspace_policy": "interface_matrix_anchored",
            "pressure_nullspace_compatibility_measured": True,
            "pressure_nullspace_zero_mean_projection_applied": False,
            "pressure_system_anchored_by_interface_matrix": True,
            "pressure_interface_neumann_active_rows": 2,
            "hibm_post_dirichlet_consistency_projection_applied": True,
            "hibm_post_dirichlet_consistency_projection_count": 1,
            "pressure_solve_failure_policy": "raise",
            "pressure_solve_failed": False,
            "pressure_solve_failure_action": "none",
            "cg_project_calls": 1,
            "cg_iterations_total": 12,
            "cg_iterations_max": 12,
            "cg_host_residual_checks": 3,
            "cg_restart_count": 0,
            "cg_restart_count_measured": False,
            "cg_restart_policy": "not_implemented",
            "cg_relative_residual_max": 5.0e-7,
            "cg_initial_relative_residual_max": 1.0,
            "cg_breakdown_count": 0,
            "cg_converged_all": True,
        }
        pressure_outlet_report = {
            "source_volume_flux_m3s": 2.5e-8,
            "zmin_reachable_source_volume_flux_m3s": 1.5e-8,
            "zmin_unreached_source_volume_flux_m3s": 1.0e-8,
            "zmin_reachable_source_cell_count": 3,
            "zmin_unreached_source_cell_count": 2,
            "zmin_unreached_source_abs_flux_m3s": 1.0e-8,
            "zmin_unreached_source_centroid_x_m": 1.0e-3,
            "zmin_unreached_source_centroid_y_m": 2.0e-3,
            "zmin_unreached_source_centroid_z_m": 3.0e-3,
            "zmin_unreached_source_min_x_m": 0.5e-3,
            "zmin_unreached_source_min_y_m": 1.5e-3,
            "zmin_unreached_source_min_z_m": 2.5e-3,
            "zmin_unreached_source_max_x_m": 1.5e-3,
            "zmin_unreached_source_max_y_m": 2.5e-3,
            "zmin_unreached_source_max_z_m": 3.5e-3,
            "zmin_pressure_outlet_flux_m3s": 1.0e-8,
            "zmin_velocity_outlet_flux_m3s": 2.0e-8,
            "zmin_pressure_outlet_to_source_ratio": 0.4,
            "zmin_velocity_outlet_to_source_ratio": 0.8,
            "zmin_projection_pre_velocity_outlet_flux_m3s": 3.0e-8,
            "zmin_projection_post_pressure_velocity_outlet_flux_m3s": 2.25e-8,
            "zmin_projection_post_boundary_velocity_outlet_flux_m3s": 2.0e-8,
        }
        solid_report = SimpleNamespace(
            particle_count=10,
            active_grid_nodes=8,
            grid_out_of_bounds_particle_count=2,
            particle_spacing_m=1.0e-3,
            grid_spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
            total_mass_kg=0.02,
            particle_momentum_kg_mps=(0.0, 0.0, -1.0e-4),
            grid_momentum_kg_mps=(0.0, 0.0, -1.0e-4),
            transfer_relative_error=0.0,
            max_speed_mps=0.01,
            external_force_n=(0.75, 2.5, 2.0),
            max_abs_j=1.0,
        )

        row = build_hibm_mpm_sharp_case_row(
            step=3,
            sample_report=sample_report,
            sharp_summary=sharp_summary,
            fluid_projection_report=projection_report,
            pressure_outlet_report=pressure_outlet_report,
            fluid_dt_s=2.0e-5,
            solid_mpm_report=solid_report,
            solid_model="neo_hookean_mpm",
            fsi_coupling_mode_report=fsi_coupling_mode_report(
                FSI_COUPLING_MODE_HIBM_MPM_SHARP
            ),
            fsi_coupling_iterations_requested=7,
        )

        self.assertEqual(row["step"], 3)
        self.assertEqual(row["fsi_coupling_mode"], FSI_COUPLING_MODE_HIBM_MPM_SHARP)
        self.assertTrue(row["fsi_coupling_mode_paper_hibm_mpm"])
        self.assertTrue(row["fsi_coupling_explicit_single_pass"])
        self.assertEqual(row["fsi_coupling_scheme"], "explicit_loose")
        self.assertEqual(row["fsi_added_mass_stability_status"], "unmeasured_single_pass")
        self.assertFalse(row["fsi_added_mass_stability_measured"])
        self.assertEqual(row["fsi_added_mass_stabilization"], "none")
        self.assertFalse(row["fsi_semi_implicit_coupling_enabled"])
        self.assertFalse(row["fsi_semi_implicit_coupling_matrix_active"])
        self.assertTrue(row["fsi_coupling_step_completed"])
        self.assertFalse(row["fsi_coupling_convergence_measured"])
        self.assertFalse(row["fsi_coupling_converged"])
        self.assertEqual(row["fsi_coupling_iterations_used"], 1)
        self.assertFalse(row["fsi_action_reaction_balance_measured"])
        self.assertEqual(
            row["fsi_coupling_residual_source"],
            "hibm_post_solid_no_slip_velocity_residual_l2_mps",
        )
        self.assertEqual(row["fsi_coupling_residual_units"], "m/s")
        self.assertTrue(math.isnan(row["fsi_coupling_residual_norm_n"]))
        self.assertAlmostEqual(row["fsi_coupling_residual_norm_mps"], 3.0e-6)
        self.assertAlmostEqual(row["fsi_coupling_residual_max_mps"], 4.0e-6)
        fixed_point_summary = {
            **sharp_summary,
            "hibm_coupling_scheme": "marker_fixed_point",
            "hibm_added_mass_stability_status": "not_converged",
            "hibm_added_mass_stability_measured": True,
            "hibm_added_mass_stabilization": "aitken_marker_state_under_relaxation",
            "hibm_semi_implicit_coupling_enabled": True,
            "hibm_semi_implicit_coupling_matrix_active": False,
            "hibm_fsi_coupling_iterations_used": 6,
            "hibm_fsi_coupling_converged": False,
            "hibm_fsi_coupling_explicit_single_pass": False,
            "hibm_fsi_coupling_residual_source": (
                "marker_surface_fixed_point_velocity_residual_l2_mps"
            ),
            "hibm_fsi_coupling_residual_l2_mps": 8.0e-4,
            "hibm_fsi_coupling_residual_max_mps": 2.5e-3,
        }
        fixed_point_row = build_hibm_mpm_sharp_case_row(
            step=3,
            sample_report=sample_report,
            sharp_summary=fixed_point_summary,
            fluid_projection_report=projection_report,
            pressure_outlet_report=pressure_outlet_report,
            fluid_dt_s=2.0e-5,
            solid_mpm_report=solid_report,
            solid_model="neo_hookean_mpm",
            fsi_coupling_mode_report=fsi_coupling_mode_report(
                FSI_COUPLING_MODE_HIBM_MPM_SHARP
            ),
            fsi_coupling_iterations_requested=6,
        )
        self.assertFalse(fixed_point_row["fsi_coupling_explicit_single_pass"])
        self.assertEqual(fixed_point_row["fsi_coupling_scheme"], "marker_fixed_point")
        self.assertEqual(fixed_point_row["fsi_coupling_iterations_used"], 6)
        self.assertEqual(
            fixed_point_row["fsi_coupling_residual_source"],
            "marker_surface_fixed_point_velocity_residual_l2_mps",
        )
        self.assertAlmostEqual(
            fixed_point_row["fsi_coupling_residual_norm_mps"],
            8.0e-4,
        )
        self.assertTrue(math.isnan(fixed_point_row["fsi_coupling_residual_norm_n"]))
        self.assertAlmostEqual(
            fixed_point_row["fsi_coupling_residual_max_mps"],
            2.5e-3,
        )
        unmeasured_no_slip_summary = {
            **sharp_summary,
            "hibm_no_slip_residual_valid_marker_count": 0,
            "hibm_no_slip_residual_invalid_marker_count": 8,
            "hibm_no_slip_residual_l2_mps": 0.0,
            "hibm_no_slip_residual_max_mps": 0.0,
            "hibm_post_solid_no_slip_residual_valid_marker_count": 0,
            "hibm_post_solid_no_slip_residual_invalid_marker_count": 8,
            "hibm_post_solid_no_slip_residual_l2_mps": 0.0,
            "hibm_post_solid_no_slip_residual_max_mps": 0.0,
        }
        unmeasured_no_slip_row = build_hibm_mpm_sharp_case_row(
            step=3,
            sample_report=sample_report,
            sharp_summary=unmeasured_no_slip_summary,
            fluid_projection_report=projection_report,
            pressure_outlet_report=pressure_outlet_report,
            fluid_dt_s=2.0e-5,
            solid_mpm_report=solid_report,
            solid_model="neo_hookean_mpm",
            fsi_coupling_mode_report=fsi_coupling_mode_report(
                FSI_COUPLING_MODE_HIBM_MPM_SHARP
            ),
            fsi_coupling_iterations_requested=7,
        )
        self.assertEqual(
            unmeasured_no_slip_row["fsi_coupling_residual_source"],
            "unmeasured_no_valid_post_solid_no_slip_markers",
        )
        self.assertTrue(
            math.isnan(unmeasured_no_slip_row["fsi_coupling_residual_norm_mps"])
        )
        self.assertTrue(
            math.isnan(unmeasured_no_slip_row["fsi_coupling_residual_max_mps"])
        )
        self.assertEqual(
            row["fsi_action_reaction_residual_source"],
            "marker_to_mpm_scatter_force_conservation",
        )
        self.assertTrue(math.isnan(row["fsi_fluid_reaction_action_reaction_relative_error"]))
        self.assertFalse(row["fsi_fluid_reaction_action_reaction_measured"])
        self.assertEqual(row["pressure_solver_requested"], "fv_multigrid")
        self.assertEqual(row["pressure_solver_actual"], "fv_cg")
        self.assertTrue(row["pressure_solver_forced_to_fv_cg"])
        self.assertEqual(
            row["pressure_solver_force_reason"],
            "hibm_pressure_neumann_requires_fv_solver",
        )
        self.assertEqual(row["pressure_nullspace_policy"], "interface_matrix_anchored")
        self.assertTrue(row["pressure_nullspace_compatibility_measured"])
        self.assertFalse(row["pressure_nullspace_zero_mean_projection_applied"])
        self.assertTrue(row["pressure_system_anchored_by_interface_matrix"])
        self.assertEqual(row["pressure_interface_neumann_active_rows"], 2)
        self.assertTrue(row["hibm_post_dirichlet_consistency_projection_applied"])
        self.assertEqual(row["hibm_post_dirichlet_consistency_projection_count"], 1)
        self.assertEqual(row["pressure_solve_failure_policy"], "raise")
        self.assertFalse(row["pressure_solve_failed"])
        self.assertEqual(row["pressure_solve_failure_action"], "none")
        self.assertEqual(row["post_boundary_divergence_l2"], 1.5e-4)
        self.assertAlmostEqual(
            row["projection_to_pre_divergence_l2_ratio"],
            1.0 / 3.0,
        )
        self.assertEqual(row["post_boundary_to_pre_divergence_l2_ratio"], 0.5)
        self.assertAlmostEqual(
            row["post_constraint_to_pre_divergence_l2_ratio"],
            1.0 / 3.0,
        )
        self.assertEqual(row["pressure_correctable_divergence_l2"], 7.0e-5)
        self.assertEqual(row["pressure_correctable_divergence_cell_count"], 12)
        self.assertEqual(row["pressure_fixed_divergence_l2"], 6.0e-5)
        self.assertEqual(row["pressure_fixed_divergence_cell_count"], 2)
        self.assertEqual(row["interior_pressure_correctable_divergence_l2"], 5.0e-5)
        self.assertEqual(row["interior_pressure_correctable_divergence_cell_count"], 6)
        self.assertEqual(row["interior_pressure_fixed_divergence_l2"], 4.0e-5)
        self.assertEqual(row["interior_pressure_fixed_divergence_cell_count"], 1)
        self.assertEqual(row["hibm_pressure_correctable_divergence_l2"], 7.0e-5)
        self.assertEqual(row["hibm_pressure_correctable_divergence_cell_count"], 12)
        self.assertEqual(row["hibm_pressure_fixed_divergence_l2"], 6.0e-5)
        self.assertEqual(row["hibm_pressure_fixed_divergence_cell_count"], 2)
        self.assertEqual(
            row["hibm_interior_pressure_correctable_divergence_l2"],
            5.0e-5,
        )
        self.assertEqual(
            row["hibm_interior_pressure_correctable_divergence_cell_count"],
            6,
        )
        self.assertEqual(row["hibm_interior_pressure_fixed_divergence_l2"], 4.0e-5)
        self.assertEqual(
            row["hibm_interior_pressure_fixed_divergence_cell_count"],
            1,
        )
        self.assertEqual(row["hibm_ib_node_count"], 11)
        self.assertEqual(row["hibm_internal_obstacle_cell_count"], 4)
        self.assertEqual(row["hibm_velocity_dirichlet_invalid_reconstruction_count"], 4)
        self.assertEqual(row["hibm_velocity_dirichlet_invalid_no_fluid_sample_count"], 1)
        self.assertEqual(row["hibm_velocity_dirichlet_invalid_nonpositive_gap_count"], 2)
        self.assertEqual(row["hibm_velocity_dirichlet_invalid_node_behind_boundary_count"], 0)
        self.assertEqual(row["hibm_velocity_dirichlet_invalid_node_beyond_interior_count"], 1)
        self.assertEqual(row["hibm_marker_primary_count"], 3)
        self.assertEqual(row["hibm_marker_secondary_count"], 5)
        self.assertEqual(row["hibm_marker_total_count"], 8)
        self.assertEqual(row["hibm_air_backed_cell_count"], 17)
        self.assertEqual(row["hibm_air_backed_component_count"], 1)
        self.assertEqual(row["hibm_air_backed_seed_marker_count"], 19)
        self.assertEqual(row["hibm_air_backed_seed_missed_marker_count"], 0)
        self.assertEqual(row["hibm_air_backed_seed_fallback_cell_count"], 4)
        self.assertEqual(
            row["hibm_air_backed_reachability_barrier_cell_count"],
            23,
        )
        self.assertEqual(
            row["fsi_volume_source_m3s"],
            pressure_outlet_report["source_volume_flux_m3s"],
        )
        self.assertNotIn("main_fsi_volume_source_m3s", row)
        self.assertNotIn("tail_fsi_volume_source_m3s", row)
        self.assertEqual(
            row["fsi_volume_source_semantics"],
            "computed_pressure_outlet_source_field_not_region_decomposed",
        )
        self.assertEqual(
            row["pressure_outlet_source_volume_flux_m3s"],
            pressure_outlet_report["source_volume_flux_m3s"],
        )
        self.assertEqual(
            row["pressure_outlet_reachable_source_volume_flux_m3s"],
            pressure_outlet_report["zmin_reachable_source_volume_flux_m3s"],
        )
        self.assertEqual(
            row["pressure_outlet_unreached_source_volume_flux_m3s"],
            pressure_outlet_report["zmin_unreached_source_volume_flux_m3s"],
        )
        for row_key, report_key in (
            (
                "pressure_outlet_reachable_source_cell_count",
                "zmin_reachable_source_cell_count",
            ),
            (
                "pressure_outlet_unreached_source_cell_count",
                "zmin_unreached_source_cell_count",
            ),
            (
                "pressure_outlet_unreached_source_abs_flux_m3s",
                "zmin_unreached_source_abs_flux_m3s",
            ),
            (
                "pressure_outlet_unreached_source_centroid_x_m",
                "zmin_unreached_source_centroid_x_m",
            ),
            (
                "pressure_outlet_unreached_source_centroid_y_m",
                "zmin_unreached_source_centroid_y_m",
            ),
            (
                "pressure_outlet_unreached_source_centroid_z_m",
                "zmin_unreached_source_centroid_z_m",
            ),
            (
                "pressure_outlet_unreached_source_min_x_m",
                "zmin_unreached_source_min_x_m",
            ),
            (
                "pressure_outlet_unreached_source_min_y_m",
                "zmin_unreached_source_min_y_m",
            ),
            (
                "pressure_outlet_unreached_source_min_z_m",
                "zmin_unreached_source_min_z_m",
            ),
            (
                "pressure_outlet_unreached_source_max_x_m",
                "zmin_unreached_source_max_x_m",
            ),
            (
                "pressure_outlet_unreached_source_max_y_m",
                "zmin_unreached_source_max_y_m",
            ),
            (
                "pressure_outlet_unreached_source_max_z_m",
                "zmin_unreached_source_max_z_m",
            ),
        ):
            self.assertEqual(row[row_key], pressure_outlet_report[report_key])
        self.assertEqual(
            row["pressure_outlet_velocity_flux_m3s"],
            pressure_outlet_report["zmin_velocity_outlet_flux_m3s"],
        )
        self.assertEqual(
            row["pressure_outlet_velocity_to_source_ratio"],
            pressure_outlet_report["zmin_velocity_outlet_to_source_ratio"],
        )
        self.assertEqual(
            row["pressure_outlet_pressure_flux_m3s"],
            pressure_outlet_report["zmin_pressure_outlet_flux_m3s"],
        )
        self.assertEqual(
            row["pressure_outlet_pressure_to_source_ratio"],
            pressure_outlet_report["zmin_pressure_outlet_to_source_ratio"],
        )
        self.assertEqual(row["main_fsi_fluid_force_z_n"], 3.0)
        self.assertEqual(row["tail_fsi_fluid_force_z_n"], -1.0)
        self.assertEqual(row["main_fsi_fluid_reaction_z_n"], -3.0)
        self.assertEqual(row["tail_fsi_fluid_reaction_z_n"], 1.0)
        self.assertEqual(row["pressure_projection_cg_project_calls"], 1)
        self.assertFalse(row["pressure_projection_cg_restart_count_measured"])
        self.assertEqual(row["pressure_projection_cg_restart_policy"], "not_implemented")
        self.assertEqual(
            row["hibm_pressure_neumann_max_raw_transmissibility_m"],
            25.0,
        )
        self.assertEqual(
            row["hibm_pressure_neumann_max_transmissibility_limit_m"],
            5.0,
        )
        self.assertEqual(
            row["hibm_pressure_neumann_transmissibility_capped_row_count"],
            2,
        )
        self.assertEqual(row["fluid_substep_dt_s"], 2.0e-5)
        self.assertEqual(row["solid_mpm_total_force_z_n"], 2.0)
        self.assertEqual(row["solid_mpm_grid_out_of_bounds_particle_count"], 2)
        self.assertNotIn("projected_ibm_residual_mps", row)
        self.assertNotIn("fsi_force_probe_valid_fraction", row)

    def test_sharp_required_row_fields_do_not_require_projected_ibm_reports(self) -> None:
        legacy_fields = finite_required_row_fields_for_mode(
            FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
            solid_model="neo_hookean_mpm",
        )
        sharp_fields = finite_required_row_fields_for_mode(
            FSI_COUPLING_MODE_HIBM_MPM_SHARP,
            solid_model="neo_hookean_mpm",
        )

        self.assertIn("projected_ibm_residual_mps", legacy_fields)
        self.assertIn("fsi_force_probe_valid_fraction", legacy_fields)
        self.assertIn("hibm_ib_node_count", sharp_fields)
        self.assertIn("hibm_internal_obstacle_cell_count", sharp_fields)
        self.assertIn("hibm_solid_band_interior_cell_count", sharp_fields)
        self.assertIn("hibm_solid_band_enclosed_water_cell_count", sharp_fields)
        self.assertIn(
            "hibm_solid_band_velocity_dirichlet_protected_cell_count",
            sharp_fields,
        )
        self.assertIn("hibm_solid_band_mask_protected_cell_count", sharp_fields)
        self.assertIn("hibm_next_solid_band_interior_cell_count", sharp_fields)
        self.assertIn("hibm_next_solid_band_enclosed_water_cell_count", sharp_fields)
        self.assertIn(
            "hibm_next_solid_band_velocity_dirichlet_protected_cell_count",
            sharp_fields,
        )
        self.assertIn("hibm_next_solid_band_mask_protected_cell_count", sharp_fields)
        self.assertIn("hibm_air_backed_cell_count", sharp_fields)
        self.assertIn("hibm_air_backed_component_count", sharp_fields)
        self.assertIn("hibm_air_backed_cell_volume_m3", sharp_fields)
        self.assertIn("hibm_air_backed_seed_marker_count", sharp_fields)
        self.assertIn("hibm_air_backed_seed_missed_marker_count", sharp_fields)
        self.assertIn("hibm_air_backed_seed_fallback_cell_count", sharp_fields)
        self.assertIn(
            "hibm_air_backed_reachability_barrier_cell_count",
            sharp_fields,
        )
        self.assertIn("hibm_marker_primary_count", sharp_fields)
        self.assertIn("hibm_marker_secondary_count", sharp_fields)
        self.assertIn("hibm_marker_total_count", sharp_fields)
        self.assertIn("hibm_marker_total_force_z_n", sharp_fields)
        self.assertIn("fsi_volume_source_m3s", sharp_fields)
        self.assertIn("fsi_coupling_residual_norm_mps", sharp_fields)
        self.assertNotIn("main_fsi_volume_source_m3s", sharp_fields)
        self.assertNotIn("tail_fsi_volume_source_m3s", sharp_fields)
        self.assertNotIn("fsi_coupling_residual_norm_n", sharp_fields)
        self.assertNotIn("hibm_coupling_scheme", sharp_fields)
        self.assertNotIn("hibm_added_mass_stability_status", sharp_fields)
        self.assertIn("hibm_added_mass_stability_measured", sharp_fields)
        self.assertNotIn("hibm_added_mass_stabilization", sharp_fields)
        self.assertIn("hibm_semi_implicit_coupling_enabled", sharp_fields)
        self.assertIn("hibm_semi_implicit_coupling_matrix_active", sharp_fields)
        self.assertNotIn("fsi_coupling_scheme", sharp_fields)
        self.assertNotIn("fsi_added_mass_stability_status", sharp_fields)
        self.assertIn("fsi_added_mass_stability_measured", sharp_fields)
        self.assertNotIn("fsi_added_mass_stabilization", sharp_fields)
        self.assertIn("fsi_semi_implicit_coupling_enabled", sharp_fields)
        self.assertIn("fsi_semi_implicit_coupling_matrix_active", sharp_fields)
        self.assertIn("hibm_no_slip_residual_valid_marker_count", sharp_fields)
        self.assertIn("hibm_no_slip_residual_invalid_marker_count", sharp_fields)
        self.assertIn("hibm_no_slip_residual_l2_mps", sharp_fields)
        self.assertIn("hibm_no_slip_residual_direct_sample_marker_count", sharp_fields)
        self.assertIn(
            "hibm_no_slip_residual_normal_walk_sample_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_no_slip_residual_nearest_fluid_sample_marker_count",
            sharp_fields,
        )
        self.assertIn("hibm_no_slip_residual_no_fluid_sample_marker_count", sharp_fields)
        self.assertIn(
            "hibm_no_slip_residual_primary_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_no_slip_residual_secondary_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_no_slip_residual_other_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn("hibm_post_solid_kinematic_projection_applied", sharp_fields)
        self.assertIn(
            "hibm_post_solid_no_slip_residual_valid_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn("hibm_post_solid_no_slip_residual_l2_mps", sharp_fields)
        self.assertIn(
            "hibm_post_solid_no_slip_residual_direct_sample_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_normal_walk_sample_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_nearest_fluid_sample_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_no_fluid_sample_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_primary_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_secondary_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_post_solid_no_slip_residual_other_region_invalid_marker_count",
            sharp_fields,
        )
        self.assertIn("fsi_coupling_residual_norm_mps", sharp_fields)
        self.assertIn("fsi_coupling_residual_max_mps", sharp_fields)
        self.assertIn("hibm_full_stress_two_sided_pressure_marker_count", sharp_fields)
        self.assertIn("hibm_full_stress_one_sided_pressure_marker_count", sharp_fields)
        self.assertIn("hibm_full_stress_one_sided_extended_marker_count", sharp_fields)
        self.assertIn(
            "hibm_full_stress_one_sided_gradient_missing_marker_count",
            sharp_fields,
        )
        self.assertIn("hibm_velocity_dirichlet_invalid_reconstruction_count", sharp_fields)
        self.assertIn("hibm_velocity_dirichlet_invalid_no_fluid_sample_count", sharp_fields)
        self.assertIn("hibm_velocity_dirichlet_invalid_nonpositive_gap_count", sharp_fields)
        self.assertIn("hibm_velocity_dirichlet_invalid_node_behind_boundary_count", sharp_fields)
        self.assertIn("hibm_velocity_dirichlet_invalid_node_beyond_interior_count", sharp_fields)
        self.assertIn("post_boundary_divergence_l2", sharp_fields)
        self.assertIn("post_boundary_divergence_max_abs", sharp_fields)
        self.assertIn("pressure_correctable_divergence_l2", sharp_fields)
        self.assertIn("pressure_correctable_divergence_cell_count", sharp_fields)
        self.assertIn("pressure_fixed_divergence_l2", sharp_fields)
        self.assertIn("pressure_fixed_divergence_cell_count", sharp_fields)
        self.assertIn("interior_pressure_correctable_divergence_l2", sharp_fields)
        self.assertIn(
            "interior_pressure_correctable_divergence_cell_count",
            sharp_fields,
        )
        self.assertIn("interior_pressure_fixed_divergence_l2", sharp_fields)
        self.assertIn("interior_pressure_fixed_divergence_cell_count", sharp_fields)
        self.assertIn("hibm_pressure_correctable_divergence_l2", sharp_fields)
        self.assertIn("hibm_pressure_correctable_divergence_cell_count", sharp_fields)
        self.assertIn("hibm_pressure_fixed_divergence_l2", sharp_fields)
        self.assertIn("hibm_pressure_fixed_divergence_cell_count", sharp_fields)
        self.assertIn(
            "hibm_interior_pressure_correctable_divergence_l2",
            sharp_fields,
        )
        self.assertIn(
            "hibm_interior_pressure_correctable_divergence_cell_count",
            sharp_fields,
        )
        self.assertIn("hibm_interior_pressure_fixed_divergence_l2", sharp_fields)
        self.assertIn(
            "hibm_interior_pressure_fixed_divergence_cell_count",
            sharp_fields,
        )
        self.assertIn("projection_to_pre_divergence_l2_ratio", sharp_fields)
        self.assertIn("post_boundary_to_pre_divergence_l2_ratio", sharp_fields)
        self.assertIn("post_constraint_to_pre_divergence_l2_ratio", sharp_fields)
        self.assertIn("pressure_projection_cg_restart_count_measured", sharp_fields)
        self.assertNotIn("pressure_projection_cg_restart_policy", sharp_fields)
        self.assertIn("hibm_pressure_neumann_max_raw_transmissibility_m", sharp_fields)
        self.assertIn(
            "hibm_pressure_neumann_max_transmissibility_limit_m",
            sharp_fields,
        )
        self.assertIn(
            "hibm_pressure_neumann_transmissibility_capped_row_count",
            sharp_fields,
        )
        self.assertIn(
            "hibm_pressure_neumann_gradient_raw_max_abs_pa_per_m",
            sharp_fields,
        )
        self.assertIn("hibm_pressure_neumann_gradient_limited_count", sharp_fields)
        self.assertIn("solid_mpm_grid_out_of_bounds_particle_count", sharp_fields)
        self.assertNotIn("projected_ibm_residual_mps", sharp_fields)
        self.assertNotIn("projected_ibm_residual_l2_mps", sharp_fields)
        self.assertNotIn("fsi_force_probe_valid_fraction", sharp_fields)
        self.assertNotIn("fsi_probe_valid_fraction", sharp_fields)

    def test_solid_response_constraint_force_mobility_ratio_uses_measured_solid_mobility(self) -> None:
        ratio = solid_response_constraint_force_mobility_ratio(
            previous_velocity_mps=(0.0, 0.0, 0.0),
            current_velocity_mps=(0.0, 0.0, 0.2),
            reaction_force_n=(0.0, 0.0, 4.0),
            interface_area_m2=0.02,
            probe_distance_m=0.001,
            density_kgm3=1000.0,
            dt_s=1.0e-4,
        )

        self.assertAlmostEqual(ratio, 10.0)
        self.assertEqual(
            solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=(0.0, 0.0, 0.0),
                current_velocity_mps=(0.0, 0.0, 0.2),
                reaction_force_n=(0.0, 0.0, 0.0),
                interface_area_m2=0.02,
                probe_distance_m=0.001,
                density_kgm3=1000.0,
                dt_s=1.0e-4,
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "current_velocity_mps"):
            solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=(0.0, 0.0, 0.0),
                current_velocity_mps=(0.0, 0.0, float("nan")),
                reaction_force_n=(0.0, 0.0, 4.0),
                interface_area_m2=0.02,
                probe_distance_m=0.001,
                density_kgm3=1000.0,
                dt_s=1.0e-4,
            )

    def test_solid_response_mobility_coupling_is_forwarded_to_projected_ibm(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("solid_response_constraint_force_mobility_ratio", source)
        self.assertIn("primary_response_constraint_force_solid_mobility_ratio", source)
        self.assertIn("secondary_response_constraint_force_solid_mobility_ratio", source)
        self.assertIn("primary_constraint_force_solid_mobility_ratio=", source)
        self.assertIn("secondary_constraint_force_solid_mobility_ratio=", source)
        self.assertIn("velocity_target_solid_mobility_ratios", source)
        self.assertIn("primary_velocity_target_solid_mobility_ratio=", source)
        self.assertIn("secondary_velocity_target_solid_mobility_ratio=", source)
        self.assertIn('"max_fsi_primary_velocity_target_solid_mobility_ratio"', source)
        self.assertIn('"max_fsi_secondary_velocity_target_solid_mobility_ratio"', source)

    def test_velocity_constraint_equivalent_force_uses_correction_dt(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("/ max(float(ibm_correction_dt_s), 1.0e-30)", source)
        self.assertNotIn("/ max(float(fluid_substep_dt_s), 1.0e-30)", source)

    def test_runner_delegates_accepted_reaction_update_to_core(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("update_interface_reaction_for_next_step", source)
        self.assertNotIn("aitken_relaxation_factor(", source)
        self.assertNotIn("relax_interface_reaction_forces(", source)
        self.assertNotIn("previous_interface_reaction_residual", source)
        self.assertNotIn("current_interface_reaction_relaxation", source)

    def test_runner_reports_nonuniform_spacing_without_pretending_uniform_grid(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn('"fluid_grid_min_spacing_m"', source)
        self.assertIn('"fluid_grid_max_spacing_m"', source)
        self.assertIn("fluid_probe_distance_m = min(fluid_grid_axis_min_spacing_m)", source)
        self.assertIn("fluid_grid_spacing_m = (\n        None", source)
        self.assertIn('summary_json = result.get("summary_json")', source)
        self.assertNotIn(
            'fluid_grid_spacing_m = [\n        float(simulator.fluid.dx),',
            source,
        )

    def test_runner_delegates_fixed_point_commit_lifecycle_to_core(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("solve_and_apply_interface_reaction_step", source)
        self.assertNotIn("solve_interface_reaction_step(", source)
        self.assertNotIn("fixed_point_result.force_n[0]", source)
        self.assertNotIn("fixed_point_result.force_n[1]", source)

    def test_runner_uses_projected_ibm_force_balance_report(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("primary_interface_reaction_balance", source)
        self.assertIn("secondary_interface_reaction_balance", source)
        self.assertNotIn(
            "main_full_reaction_balance = action_reaction_balance(primary_fluid_force_n",
            source,
        )
        self.assertNotIn(
            "tail_full_reaction_balance = action_reaction_balance(secondary_fluid_force_n",
            source,
        )

    def test_runner_action_reaction_uses_step_equivalent_ibm_force(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn(
            "primary_fluid_force_n = fluid_step_report.primary_equivalent_fluid_force_n",
            source,
        )
        self.assertIn(
            "secondary_fluid_force_n = fluid_step_report.secondary_equivalent_fluid_force_n",
            source,
        )
        self.assertIn(
            '"fsi_grid_force_x_n": primary_fluid_force_n[0] + secondary_fluid_force_n[0]',
            source,
        )
        self.assertIn('"fsi_last_correction_grid_force_x_n"', source)

    def test_primary_action_reaction_metric_is_not_pressure_grid_balance(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertNotIn("pressure_grid_balance = action_reaction_balance", source)
        self.assertNotIn('"fsi_action_reaction_note": (', source)
        self.assertIn("fsi_interface_balance = action_reaction_balance", source)
        self.assertIn('"fsi_grid_force_decomposition_source"', source)
        self.assertIn("fsi_grid_force_decomposition_* is also diagnostic only", source)
        self.assertNotIn('"fsi_grid_force_decomposition_consistent"', source)
        self.assertNotIn("max_fsi_grid_decomposition_relative_error <= 1.0e-6", source)
        self.assertNotIn('"fsi_action_reaction_consistent"', source)
        self.assertNotIn('"fluid_reaction_action_reaction_consistent"', source)
        self.assertNotIn('"fluid_reaction_full_3d_action_reaction_consistent"', source)
        self.assertNotIn('"solid_mpm_transfer_conservative"', source)

    def test_runner_reaction_target_comes_from_projected_ibm_not_pressure_traction(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn(
            "stabilized_primary_reaction_target_n = _vector3(\n"
            "            fluid_step_report.interface_reaction_target.primary_force_n",
            source,
        )
        self.assertIn(
            "stabilized_secondary_reaction_target_n = _vector3(\n"
            "            fluid_step_report.interface_reaction_target.secondary_force_n",
            source,
        )
        self.assertIn(
            "primary_interface_impedance_force_n=trial_primary_robin_impedance_force_n",
            source,
        )
        self.assertIn("trial_robin_impedance_force_n = robin_neumann_impedance_force", source)
        self.assertIn("robin_previous_velocity_mps = robin_previous_velocity_for_step", source)
        self.assertIn("previous_velocity_mps=robin_previous_velocity_mps", source)
        self.assertIn("selected_target_force_n = interface_reaction_target_for_mode", source)
        self.assertIn("target_force_n=selected_target_force_n", source)
        self.assertIn("robin_impedance_ns_per_m=0.0", source)
        self.assertNotIn(".component_pair(", source)
        self.assertIn('"raw_main_pressure_traction_z_n"', source)
        self.assertNotIn(
            "target_force_n=(tri_report.primary_pressure_traction_force_n",
            source,
        )
        self.assertNotIn(
            "raw_main_reaction_target_z_n = tri_report.primary_pressure_traction_force_n[2]",
            source,
        )
        self.assertNotIn(
            "raw_tail_reaction_target_z_n = tri_report.secondary_pressure_traction_force_n[2]",
            source,
        )

    def test_runner_records_full_3d_per_region_fsi_force_components(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        for field in (
            '"main_fsi_fluid_force_x_n"',
            '"main_fsi_fluid_force_y_n"',
            '"main_fsi_fluid_force_z_n"',
            '"tail_fsi_fluid_force_x_n"',
            '"tail_fsi_fluid_force_y_n"',
            '"tail_fsi_fluid_force_z_n"',
            '"main_fsi_fluid_reaction_x_n"',
            '"main_fsi_fluid_reaction_y_n"',
            '"main_fsi_fluid_reaction_z_n"',
            '"tail_fsi_fluid_reaction_x_n"',
            '"tail_fsi_fluid_reaction_y_n"',
            '"tail_fsi_fluid_reaction_z_n"',
        ):
            self.assertIn(field, source)

    def test_runner_passes_water_viscosity_to_surface_stress_diagnostics(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("viscosity_pa_s=spec.water_viscosity_pa_s", source)
        self.assertIn('"viscous_traction_force_x_n"', source)
        self.assertIn('"fluid_stress_traction_force_x_n"', source)
        self.assertIn('"max_viscous_traction_force_n"', source)
        self.assertIn('"max_fluid_stress_traction_force_n"', source)
        self.assertIn('"diagnostic_checks"', source)
        self.assertIn('"projection_pressure_traction_diagnostic_nonzero"', source)
        self.assertNotIn('"pressure_traction_nonzero"', source)
        self.assertIn('"fluid_stress_action_on_fluid_enabled": True', source)
        self.assertIn("Surface force spreading adds the opposite of sampled -pI + viscous", source)

    def test_runner_does_not_silently_zero_missing_projected_ibm_reports(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("required_projected_ibm_force_report", source)
        self.assertIn("required_fluid_impulse_report", source)
        self.assertIn('"fsi_force_probe_valid_fraction"', source)
        self.assertIn('"fsi_force_probe_valid_fraction_all_valid"', source)
        self.assertIn("min_force_probe_valid_fraction >= 1.0", source)
        self.assertNotIn('"fsi_force_probe_valid_fraction_positive"', source)
        for token in (
            "(0.0, 0.0, 0.0) if force_report is None",
            "0.0 if force_report is None",
            "0 if force_report is None",
            "0.0 if impulse_report is None",
            "if impulse_report is None",
        ):
            self.assertNotIn(token, source, msg=token)

    def test_runner_does_not_silently_zero_missing_final_summary_fields(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        for token in (
            'last.get("main_volume_flux_to_outlet_ratio", 0.0)',
            'last.get("main_volume_flux_to_downstream_ratio", 0.0)',
            'last.get("relaxed_main_interface_reaction_power_w_next", 0.0)',
            'last.get("outlet_flow_negative_z_m3s", 0.0)',
            'last.get("lip_flow_negative_z_m3s", 0.0)',
            'last.get("downstream_flow_negative_z_m3s", 0.0)',
            'last.get("solid_mpm_grid_dx_m", 0.0)',
            'last.get("solid_mpm_particle_spacing_m", 0.0)',
            'last.get("solid_mpm_particle_count", 0)',
            "solid_mpm_force_components_n = (0.0, 0.0, 0.0)",
        ):
            self.assertNotIn(token, source, msg=token)
        self.assertIn("_final_row_number(last, \"outlet_flow_negative_z_m3s\")", source)

    def test_reduced_squid_state_has_no_unused_taichi_pressure_schedule(self) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertNotIn("def _pressure_schedule(self, t)", source)

    def test_reported_pressure_uses_step_end_time_not_step_start_or_last_substep(self) -> None:
        current_time_s = 1.0
        dt_s = 0.25

        reported = pressure_schedule_step_end_pa(current_time_s, dt_s)

        self.assertAlmostEqual(reported, pressure_schedule_pa(current_time_s + dt_s))
        self.assertNotAlmostEqual(reported, pressure_schedule_pa(current_time_s))
        self.assertAlmostEqual(reported, 4000.0)

    def test_pressure_schedule_uses_configured_control_points(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="config.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
            pressure_t0_s=0.0,
            pressure_t1_s=0.5,
            pressure_t2_s=1.5,
            pressure_p0_pa=100.0,
            pressure_p1_pa=900.0,
            pressure_p2_pa=-300.0,
        )

        self.assertAlmostEqual(pressure_schedule_pa(0.25, spec), 500.0)
        self.assertAlmostEqual(pressure_schedule_step_end_pa(1.0, 0.25, spec), 0.0)

        delayed_spec = replace(spec, pressure_t0_s=0.25, pressure_t1_s=0.75)
        self.assertAlmostEqual(pressure_schedule_pa(0.0, delayed_spec), 100.0)
        self.assertAlmostEqual(pressure_schedule_pa(0.25, delayed_spec), 100.0)
        self.assertAlmostEqual(pressure_schedule_pa(0.5, delayed_spec), 500.0)

    def test_direct_nozzle_velocity_switch_is_not_supported(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--direct-nozzle-velocity"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_shell_transfer_solid_model_is_not_supported(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--solid-model", "shell_transfer"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_layered_transfer_solid_model_is_not_supported(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--solid-model", "layered_transfer"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_old_feedback_mode_switches_are_not_supported(self) -> None:
        for switch in (
            "--fsi-feedback-force-mode",
            "--disable-solid-constraint-reaction-feedback",
            "--solid-constraint-reaction-feedback",
            "--fluid-feedback-relaxation",
            "--fluid-feedback-aitken",
            "--fluid-feedback-passivity-limit",
            "--disable-fluid-feedback-passivity-limit",
            "--pressure-force-scale",
        ):
            with self.subTest(switch=switch):
                argv = ["squid_soft_robot.py", switch]
                if switch == "--fsi-feedback-force-mode":
                    argv.append("pressure_traction")
                with patch("sys.argv", argv):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parse_args()

    def test_aitken_relaxation_factor_uses_residual_delta_and_clamps(self) -> None:
        relaxation = aitken_relaxation_factor(
            0.5,
            previous_residual=(1.0, 0.0),
            current_residual=(0.5, 0.0),
        )
        self.assertAlmostEqual(relaxation, 1.0)

        clipped = aitken_relaxation_factor(
            0.5,
            previous_residual=(1.0, 0.0),
            current_residual=(0.99, 0.0),
        )
        self.assertAlmostEqual(clipped, 1.5)

    def test_interface_reaction_aitken_can_be_disabled_explicitly(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py", "--no-interface-reaction-aitken"]):
            args = parse_args()

        self.assertFalse(args.interface_reaction_aitken)

    def test_interface_reaction_aitken_lower_bound_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--interface-reaction-aitken-lower-bound",
                "0.005",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.interface_reaction_aitken_lower_bound, 0.005)

    def test_interface_reaction_aitken_upper_bound_can_be_selected_explicitly(self) -> None:
        with patch(
            "sys.argv",
            [
                "squid_soft_robot.py",
                "--interface-reaction-aitken-upper-bound",
                "0.25",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.interface_reaction_aitken_upper_bound, 0.25)

    def test_mooney_force_scale_cli_is_named_for_membrane_not_edge_springs(self) -> None:
        with patch("sys.argv", ["squid_soft_robot.py"]):
            default_args = parse_args()

        self.assertAlmostEqual(default_args.mooney_membrane_force_scale, 1.0)
        with patch("sys.argv", ["squid_soft_robot.py", "--mooney-membrane-force-scale", "0.75"]):
            args = parse_args()

        self.assertAlmostEqual(args.mooney_membrane_force_scale, 0.75)
        self.assertFalse(hasattr(args, "mooney_edge_force_scale"))
        with patch("sys.argv", ["squid_soft_robot.py", "--mooney-edge-force-scale", "0.75"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_interface_reaction_relaxation_applies_passivity_limit_after_under_relaxation(self) -> None:
        update = relax_interface_reaction_forces(
            previous_force_n=(0.0, 0.0),
            target_force_n=(10.0, -8.0),
            velocity_mps=(0.1, 0.2),
            relaxation=0.5,
            passivity_limit=True,
        )

        self.assertEqual(update.force_n, (5.0, -4.0))
        self.assertAlmostEqual(sum(update.power_w), -0.3)
        self.assertFalse(update.passivity_limited[0])
        self.assertAlmostEqual(update.force_n[1], -4.0)
        self.assertFalse(update.passivity_limited[1])
        self.assertAlmostEqual(update.residual_norm_n, (5.0 * 5.0 + 4.0 * 4.0) ** 0.5)

    def test_interface_reaction_target_uses_actual_fluid_reaction(self) -> None:
        target = interface_reaction_force((5.0, -6.0))

        self.assertEqual(target, (-5.0, 6.0))

    def test_generic_fixed_point_solver_uses_restore_callback_and_converges(self) -> None:
        restore_calls = 0

        def restore_state() -> None:
            nonlocal restore_calls
            restore_calls += 1

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            target = (0.5 * force_n[0] + 1.0, 0.5 * force_n[1] - 1.0)
            return InterfaceReactionTargetEvaluation(target_force_n=target, velocity_mps=(-1.0, 1.0))

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0, 0.0),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=20,
            tolerance_n=1.0e-5,
            initial_relaxation=1.0,
            use_aitken=True,
            passivity_limit=False,
        )

        self.assertTrue(result.converged)
        self.assertGreater(restore_calls, 1)
        self.assertAlmostEqual(result.force_n[0], 2.0, delta=1.0e-4)
        self.assertAlmostEqual(result.force_n[1], -2.0, delta=1.0e-4)

    def test_passivity_limiter_does_not_pollute_fixed_point_trial_guess(self) -> None:
        trial_forces: list[tuple[float, ...]] = []

        def restore_state() -> None:
            return None

        def evaluate_target(force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
            trial_forces.append(force_n)
            return InterfaceReactionTargetEvaluation(
                target_force_n=(13.333333333333334,),
                velocity_mps=(1.0,),
            )

        result = solve_interface_reaction_fixed_point(
            initial_force_n=(0.0,),
            evaluate_target=evaluate_target,
            restore_state=restore_state,
            max_iterations=3,
            tolerance_n=0.0,
            initial_relaxation=1.0,
            use_aitken=False,
            passivity_limit=True,
        )

        self.assertEqual(trial_forces[0], (0.0,))
        self.assertAlmostEqual(trial_forces[1][0], 13.333333333333334)
        self.assertEqual(result.force_n, (0.0,))

    def test_reduced_squid_vector_interface_reaction_snapshot_restores_trial_state(self) -> None:
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(0.0, 0.0, 0.0),
            fluid_bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
            dt_s=1.0e-3,
            water_density_kgm3=1025.0,
            water_viscosity_pa_s=1.05e-3,
        )
        simulator = ReducedSquidFSI(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        simulator.set_structure_state(
            time_s=0.25,
            pressure_pa=1000.0,
            hydraulic_pressure_pa=25.0,
            main_displacement_z_m=-0.001,
            main_velocity_z_mps=-0.02,
            tail_displacement_z_m=0.0003,
            tail_velocity_z_mps=0.01,
            volume_flux_m3s=1.0e-7,
            nozzle_velocity_z_mps=-0.03,
        )
        simulator.set_interface_reaction(
            primary_force_n=(-0.1, 0.2, -0.4),
            secondary_force_n=(0.3, -0.5, 0.2),
        )
        simulator.save_reduced_state()

        simulator.set_structure_state(
            time_s=0.5,
            pressure_pa=2000.0,
            hydraulic_pressure_pa=50.0,
            main_displacement_z_m=-0.004,
            main_velocity_z_mps=-0.08,
            tail_displacement_z_m=0.001,
            tail_velocity_z_mps=0.04,
            volume_flux_m3s=2.0e-7,
            nozzle_velocity_z_mps=-0.06,
        )
        simulator.set_interface_reaction(
            primary_force_n=(1.0, 2.0, 3.0),
            secondary_force_n=(4.0, 5.0, 6.0),
        )
        simulator.restore_reduced_state()

        self.assertAlmostEqual(float(simulator.time_s[None]), 0.25, places=6)
        self.assertAlmostEqual(float(simulator.main_w_m[None]), -0.001, places=7)
        self.assertAlmostEqual(float(simulator.main_v_mps[None]), -0.02, places=7)
        self.assertAlmostEqual(float(simulator.tail_w_m[None]), 0.0003, places=7)
        self.assertAlmostEqual(float(simulator.tail_v_mps[None]), 0.01, places=7)
        primary_reaction = simulator.primary_interface_reaction_force_n[None]
        secondary_reaction = simulator.secondary_interface_reaction_force_n[None]
        self.assertAlmostEqual(float(primary_reaction.x), -0.1, places=6)
        self.assertAlmostEqual(float(primary_reaction.y), 0.2, places=6)
        self.assertAlmostEqual(float(primary_reaction.z), -0.4, places=6)
        self.assertAlmostEqual(float(secondary_reaction.x), 0.3, places=6)
        self.assertAlmostEqual(float(secondary_reaction.y), -0.5, places=6)
        self.assertAlmostEqual(float(secondary_reaction.z), 0.2, places=6)
        self.assertFalse(hasattr(simulator, "main_interface_reaction_force_z_n"))
        self.assertFalse(hasattr(simulator, "tail_interface_reaction_force_z_n"))

    def test_partitioned_interface_reaction_passivity_can_be_disabled_explicitly(self) -> None:
        with patch(
            "sys.argv",
            ["squid_soft_robot.py", "--no-interface-reaction-passivity-limit"],
        ):
            args = parse_args()

        self.assertFalse(args.interface_reaction_passivity_limit)

    def test_partitioned_interface_reaction_passivity_is_explicit_opt_in(self) -> None:
        with patch(
            "sys.argv",
            ["squid_soft_robot.py", "--interface-reaction-passivity-limit"],
        ):
            args = parse_args()

        self.assertTrue(args.interface_reaction_passivity_limit)


class SquidRunCheckpointMarkerStateTests(unittest.TestCase):
    """C1/H1/H2/M1-M3 (2026-06-11): run checkpoints must carry the dynamic HIBM
    sharp marker state, reject stale formats, write history atomically, persist
    a closing checkpoint at loop exit, and guard solid out-of-bounds particles.
    """

    MARKER_STATE_FIELD_NAMES = ("x_gamma_m", "v_gamma_mps", "n_gamma", "A_gamma_m2")

    @staticmethod
    def _sharp_checkpoint_fixture():
        runtime = TaichiRuntimeConfig(arch="cuda")
        spec = SquidReducedSpec(
            source_config_path="dummy.json",
            fluid_bounds_min_m=(-0.01, -0.01, -0.01),
            fluid_bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(6, 6, 6),
            dt_s=1.0e-4,
            water_density_kgm3=1000.0,
            water_viscosity_pa_s=1.0e-3,
        )
        simulator = ReducedSquidFSI(spec, runtime=runtime)
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            grid_nodes=spec.grid_nodes,
            runtime=runtime,
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 4.0e-6
        solid.region_id[0] = 7
        return runtime, simulator, solid

    def test_run_checkpoint_version_is_3(self) -> None:
        # H1: S2 changed the drive physics in a way the arg fingerprint cannot
        # see, so pre-S2 checkpoints must be hard-rejected via a version bump.
        self.assertEqual(RUN_CHECKPOINT_VERSION, 3)

    def test_checkpoint_fingerprint_includes_diagnostic_neumann_rows_and_arch(
        self,
    ) -> None:
        # H2: both are real argparse dests that change the numerical trajectory.
        self.assertIn(
            "diagnostic_disable_pressure_neumann_matrix_rows",
            CHECKPOINT_ARG_FINGERPRINT_FIELDS,
        )
        self.assertIn("arch", CHECKPOINT_ARG_FINGERPRINT_FIELDS)

    def test_checkpoint_roundtrip_preserves_sharp_marker_state(self) -> None:
        # C1: markers advance by dt*v feedback and never re-converge to the
        # solid after a bad resume, so the checkpoint must carry their state.
        runtime, simulator, solid = self._sharp_checkpoint_fixture()
        coupling = build_hibm_mpm_sharp_coupling_state(
            fluid=simulator.fluid,
            solid_mpm=solid,
            runtime=runtime,
        )
        rest_position = coupling.markers.x_gamma_m.to_numpy()[:1].copy()
        deformed_position = rest_position + np.asarray(
            [[2.5e-3, -1.5e-3, 3.5e-3]], dtype=np.float32
        )
        deformed_velocity = np.asarray([[0.11, -0.07, 0.05]], dtype=np.float32)
        deformed_normal = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
        deformed_area = np.asarray([6.0e-6], dtype=np.float32)
        coupling.markers.x_gamma_m.from_numpy(deformed_position)
        coupling.markers.v_gamma_mps.from_numpy(deformed_velocity)
        coupling.markers.n_gamma.from_numpy(deformed_normal)
        coupling.markers.A_gamma_m2.from_numpy(deformed_area)
        state = InterfaceReactionRelaxationState(relaxation=1.0)
        args = SimpleNamespace(solid_model="neo_hookean_mpm")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "restart.npz"
            write_run_checkpoint(
                checkpoint_path,
                completed_step=5,
                step_count=100,
                full_pressure_waveform_steps=4000,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                interface_reaction_state=state,
                sharp_coupling_state=coupling,
            )
            with np.load(checkpoint_path, allow_pickle=False) as payload:
                for name in self.MARKER_STATE_FIELD_NAMES:
                    self.assertIn(f"marker_{name}", payload)
                self.assertTrue(bool(payload["has_marker_state"]))

            resumed_coupling = build_hibm_mpm_sharp_coupling_state(
                fluid=simulator.fluid,
                solid_mpm=solid,
                runtime=runtime,
            )
            completed_step, _ = load_run_checkpoint(
                checkpoint_path,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                sharp_coupling_state=resumed_coupling,
            )

        self.assertEqual(completed_step, 5)
        self.assertGreater(
            float(np.abs(deformed_position - rest_position).max()), 1.0e-4
        )
        np.testing.assert_allclose(
            resumed_coupling.markers.x_gamma_m.to_numpy()[:1],
            deformed_position,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            resumed_coupling.markers.v_gamma_mps.to_numpy()[:1],
            deformed_velocity,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            resumed_coupling.markers.n_gamma.to_numpy()[:1],
            deformed_normal,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            resumed_coupling.markers.A_gamma_m2.to_numpy()[:1],
            deformed_area,
            atol=1.0e-12,
        )

    def test_load_run_checkpoint_rejects_checkpoint_without_sharp_marker_state(
        self,
    ) -> None:
        # C1 double insurance behind the H1 version bump: a checkpoint written
        # without marker state must not silently resume a sharp-coupling run.
        runtime, simulator, solid = self._sharp_checkpoint_fixture()
        state = InterfaceReactionRelaxationState(relaxation=1.0)
        args = SimpleNamespace(solid_model="neo_hookean_mpm")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "restart.npz"
            write_run_checkpoint(
                checkpoint_path,
                completed_step=5,
                step_count=100,
                full_pressure_waveform_steps=4000,
                args=args,
                simulator=simulator,
                solid_mpm=solid,
                interface_reaction_state=state,
            )
            coupling = build_hibm_mpm_sharp_coupling_state(
                fluid=simulator.fluid,
                solid_mpm=solid,
                runtime=runtime,
            )
            with self.assertRaisesRegex(ValueError, "marker state"):
                load_run_checkpoint(
                    checkpoint_path,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid,
                    sharp_coupling_state=coupling,
                )

    def test_write_csv_is_atomic(self) -> None:
        # M2: history.csv must be written tmp-then-replace like the checkpoint
        # itself so a kill mid-write cannot truncate the resume history.
        rows: list[dict[str, object]] = [
            {"step": 1, "value": 0.5},
            {"step": 2, "value": 1.5, "extra": "x"},
        ]
        replace_calls: list[tuple[str, str]] = []
        real_replace = os.replace

        def recording_replace(src, dst, *replace_args, **replace_kwargs):
            replace_calls.append((str(src), str(dst)))
            return real_replace(src, dst, *replace_args, **replace_kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history.csv"
            with patch("os.replace", side_effect=recording_replace):
                write_csv(target, rows)

            self.assertTrue(target.exists())
            loaded = read_csv_rows(target)
            self.assertEqual([path for path in Path(temp_dir).iterdir()], [target])

        self.assertEqual(len(loaded), 2)
        self.assertEqual(int(loaded[0]["step"]), 1)
        self.assertEqual(loaded[1]["extra"], "x")
        self.assertEqual(len(replace_calls), 1)
        source_path, destination_path = replace_calls[0]
        self.assertTrue(source_path.endswith(".tmp"))
        self.assertEqual(destination_path, str(target))

    def test_step_guard_rejects_solid_out_of_bounds_particles(self) -> None:
        # M3: solid particles leaving the solid MPM grid must hard-stop the
        # step instead of silently logging a nonzero count to history.csv.
        from cases.squid_soft_robot import _raise_for_step_solid_out_of_bounds_guard

        _raise_for_step_solid_out_of_bounds_guard({"step": 4})
        _raise_for_step_solid_out_of_bounds_guard(
            {"step": 4, "solid_mpm_grid_out_of_bounds_particle_count": 0}
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"solid_mpm_grid_out_of_bounds_particle_count=3",
        ):
            _raise_for_step_solid_out_of_bounds_guard(
                {"step": 4, "solid_mpm_grid_out_of_bounds_particle_count": 3}
            )

    def test_step_guard_blocks_check_solid_out_of_bounds_particles(self) -> None:
        # M3 wiring: both per-step guard blocks (sharp + legacy) must call the
        # out-of-bounds guard inside the failure-artifact try block.
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        self.assertIn("def _raise_for_step_solid_out_of_bounds_guard(", source)
        guard_call_segments = source.split("cfl_limit=0.5,")[1:]
        self.assertEqual(
            len(guard_call_segments),
            2,
            msg="expected exactly the sharp and legacy per-step guard blocks",
        )
        for guard_call_segment in guard_call_segments:
            guard_block = guard_call_segment.split("except Exception as exc:", 1)[0]
            self.assertIn(
                "_raise_for_step_solid_out_of_bounds_guard(row)",
                guard_block,
            )

    def test_run_loop_exit_and_resume_wire_sharp_marker_checkpoint_state(self) -> None:
        # C1 wiring + M1: the in-loop checkpoint writes and the resume load
        # must pass the sharp coupling state, and the loop exit (wall-time
        # break or normal completion) must persist a closing checkpoint.
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        resume_block = source.split("if args.resume_from_checkpoint:", 1)[1].split(
            "first_step = completed_step + 1",
            1,
        )[0]
        self.assertIn("sharp_coupling_state=sharp_coupling_state", resume_block)

        closing_block = source.split(
            'partial_run_reason = "max_wall_time_s"',
            2,
        )[2].split("if sharp_case_runner_enabled:", 1)[0]
        self.assertIn("write_run_checkpoint(", closing_block)
        self.assertIn("sharp_coupling_state=sharp_coupling_state", closing_block)
        self.assertIn('completed_step=int(rows[-1]["step"])', closing_block)

        self.assertEqual(source.count("sharp_coupling_state=sharp_coupling_state"), 4)


class SquidSharpTwoSidedExtendedWalkContractTests(unittest.TestCase):
    def test_sharp_case_wires_two_sided_extended_walk_multiplier(self) -> None:
        # S2-A10 contract: the A8'' dedicated sampling view starved thin
        # features (the tail fin sits entirely inside its own row-cloud
        # envelope; per-step two-sided valid population 171-1017 avg 210
        # before A8'' -> ~0 after; tail_marker_participates True -> False),
        # so the case must arm the two-sided extended walk at the same 12.0
        # reach as the far-pressure closure multiplier on its sharp advance
        # call. Imitates test_sharp_case_drives_main_membrane_via_far_
        # pressure_closure, which pins the closure 12.0 on the same call
        # slice (the case's single sharp advance site; the checkpoint-resume
        # path re-enters the same loop).
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_advance_call = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("sharp_summary = hibm_mpm_sharp_step_summary", 1)[0]
        self.assertIn(
            "two_sided_probe_max_multiplier=12.0", sharp_advance_call
        )
        # The closure multiplier stays wired alongside it (the extension
        # complements the closure, it does not replace it).
        self.assertIn(
            "far_pressure_inside_probe_max_multiplier=12.0", sharp_advance_call
        )

    def test_sharp_case_treats_internal_nodes_as_thin_interface_not_obstacle(
        self,
    ) -> None:
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_advance_call = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("sharp_summary = hibm_mpm_sharp_step_summary", 1)[0]

        self.assertIn(
            "convert_internal_nodes_to_obstacles=False",
            sharp_advance_call,
        )
        self.assertIn(
            "far_pressure_air_backed_probe_normal_sign=pressure_far_side_normal_sign",
            sharp_advance_call,
        )


class SquidHistoryWriteRobustnessTests(unittest.TestCase):
    """2026-06-13 production incident: an external reader (a monitoring
    Import-Csv) held history.csv open at the instant of write_csv's atomic
    os.replace -> WinError 5 PermissionError -> the 4000-step production run
    died at step 506. Windows MoveFileEx(REPLACE_EXISTING) fails while ANY
    process holds the destination without FILE_SHARE_DELETE (monitors, Excel,
    antivirus, indexers). The atomic history write must absorb transient
    destination locks by retrying with backoff, and still raise once a
    persistent holder exhausts the budget (never hang, never drop the write
    silently).
    """

    def test_write_csv_retries_transient_replace_lock(self) -> None:
        rows: list[dict[str, object]] = [{"step": 1, "value": 0.5}]
        real_replace = os.replace
        calls: list[tuple[str, str]] = []

        def flaky_replace(src, dst, *args, **kwargs):
            calls.append((str(src), str(dst)))
            if len(calls) <= 2:
                raise PermissionError(5, "Access is denied", str(dst))
            return real_replace(src, dst, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history.csv"
            with patch("os.replace", side_effect=flaky_replace), patch(
                "time.sleep"
            ) as sleep_mock:
                write_csv(target, rows)

            loaded = read_csv_rows(target)
            self.assertEqual(
                [entry for entry in Path(temp_dir).iterdir()], [target]
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(int(loaded[0]["step"]), 1)
        # Backoff must separate the attempts (patched out for test speed).
        self.assertGreaterEqual(sleep_mock.call_count, 2)

    def test_write_csv_raises_after_replace_retry_budget(self) -> None:
        rows: list[dict[str, object]] = [{"step": 1}]
        attempts: list[int] = []

        def always_locked(src, dst, *args, **kwargs):
            attempts.append(1)
            raise PermissionError(5, "Access is denied", str(dst))

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history.csv"
            with patch("os.replace", side_effect=always_locked), patch(
                "time.sleep"
            ):
                with self.assertRaises(PermissionError):
                    write_csv(target, rows)

        # A real retry budget (not a one-shot), but finite (no hang).
        self.assertGreaterEqual(len(attempts), 8)
        self.assertLessEqual(len(attempts), 100)


class SquidClosureCoverageFloorGuardTests(unittest.TestCase):
    """Closure coverage below a configured floor is a fatal sharp-case signal.

    The guard lives beside the solid out-of-bounds guard and remains default
    off; this test uses synthetic row values, not historical run targets.
    """

    FIELD = "hibm_full_stress_far_pressure_closed_marker_count"

    @classmethod
    def _rows(cls, values: list[int]) -> list[dict[str, object]]:
        return [
            {"step": index + 1, cls.FIELD: value}
            for index, value in enumerate(values)
        ]

    def test_raises_after_patience_consecutive_rows_below_floor(self) -> None:
        from cases.squid_soft_robot import _raise_for_closure_coverage_floor

        rows = self._rows([10, 10, 6, 5, 4])

        with self.assertRaises(RuntimeError) as raised:
            _raise_for_closure_coverage_floor(rows, 7, 3)

        message = str(raised.exception)
        self.assertIn(self.FIELD, message)
        self.assertIn("7", message)
        self.assertIn("3", message)
        self.assertIn("4", message)

    def test_below_floor_for_patience_minus_one_is_silent(self) -> None:
        from cases.squid_soft_robot import _raise_for_closure_coverage_floor

        rows = self._rows([10, 10, 10, 6, 5])

        _raise_for_closure_coverage_floor(rows, 7, 3)
        # Fewer rows than the patience window can never establish a streak.
        _raise_for_closure_coverage_floor(self._rows([6, 5]), 7, 3)

    def test_disabled_floor_zero_is_silent(self) -> None:
        from cases.squid_soft_robot import _raise_for_closure_coverage_floor

        rows = self._rows([0, 0, 0, 0, 0])

        _raise_for_closure_coverage_floor(rows, 0, 3)

    def test_recovery_resets_the_streak(self) -> None:
        from cases.squid_soft_robot import _raise_for_closure_coverage_floor

        recovered = self._rows([5, 6, 10, 6, 5])

        _raise_for_closure_coverage_floor(recovered, 7, 3)

        relapsed = recovered + self._rows([4])
        with self.assertRaises(RuntimeError):
            _raise_for_closure_coverage_floor(relapsed, 7, 3)

    def test_sharp_guard_block_wires_floor_guard_and_neo_passes_fixed_rim(
        self,
    ) -> None:
        # Source-slicing contract, in the style of
        # SquidSharpTwoSidedExtendedWalkContractTests: the closure floor guard
        # must run inside the sharp per-step failure-artifact try block (so a
        # trip still writes step failure artifacts), and the neo_hookean_mpm
        # construction must honor the case's Fixed Support rim region the way
        # the tri_mooney_shell_mpm construction already does.
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_step_tail = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("reused_fluid_step_report = None", 1)[0]
        sharp_guard_try = sharp_step_tail.split("try:", 1)[1].split(
            "except Exception as exc:",
            1,
        )[0]
        self.assertIn("_raise_for_step_solid_out_of_bounds_guard(row)", sharp_guard_try)
        self.assertIn("_raise_for_closure_coverage_floor(", sharp_guard_try)
        self.assertIn("args.closure_coverage_floor", sharp_guard_try)
        self.assertIn("args.closure_coverage_floor_patience", sharp_guard_try)
        self.assertIn("_write_step_failure_artifacts(", sharp_step_tail)
        self.assertIn('"--closure-coverage-floor"', source)
        self.assertIn('"--closure-coverage-floor-patience"', source)

        neo_construction = source.split(
            "solid_mpm.initialize_layered_tri_surface(",
            1,
        )[1].split("raise ValueError", 1)[0]
        self.assertIn("fixed_region_id=5,", neo_construction)


class SquidNeoSolidSubsetContractTests(unittest.TestCase):
    def test_neo_solid_path_consumes_solid_region_subset(self) -> None:
        # S2-A11c: build_tri_surface_diagnostics builds TWO subsets - the FSI
        # diagnostic object from regions (7, 8) and the solid mesh from
        # (7, 8, 5) including the 878 rim faces ("Main membrane rim fixed to
        # chamber surface 6"). The first A11 wiring fed the neo path the
        # (7, 8) diagnostics object, so fixed_region_id=5 matched zero faces
        # and the rim constraint was VACUOUS - the membrane stayed a free
        # disc AND the 7.25 mm rim annulus stayed an open leak path around
        # the membrane edge. The neo construction must consume a solid-subset
        # diagnostics object (rim faces present: the constraint binds, and
        # the rim markers' velocity-Dirichlet rows seal the annulus).
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")

        builder = source.split("def build_tri_surface_diagnostics(", 1)[1]
        builder = builder.split("\ndef ", 1)[0]
        self.assertIn("solid_diagnostics", builder)

        # Anchor on the CONSTRUCTION occurrence specifically (the string
        # `elif args.solid_model == "neo_hookean_mpm":` also appears in the
        # checkpoint writer earlier in the file).
        window = source.split(
            'elif args.solid_model == "neo_hookean_mpm":\n'
            "        solid_mpm = NeoHookeanMpmState(",
            1,
        )[1]
        window = window.split("\n    else:", 1)[0]
        self.assertIn("solid_diagnostics", window)
        self.assertNotIn("tri_diagnostics", window)
        self.assertIn("fixed_region_id=5", window)


class SquidSharpAirBackedClosureContractTests(unittest.TestCase):
    def test_sharp_case_declares_air_backed_far_pressure_closure(self) -> None:
        # S2-A12 contract: the squid's far-pressure closure rides on the
        # generic HIBM air-backed classification. The case configures the
        # closure region, pressure, and probe reach; the solver computes the
        # selected cells and resulting flow at run time.
        source = Path("cases/squid_soft_robot.py").read_text(encoding="utf-8")
        sharp_advance_call = source.split(
            "sharp_report = sharp_coupling_state.advance_mpm_step(",
            1,
        )[1].split("sharp_summary = hibm_mpm_sharp_step_summary", 1)[0]
        self.assertIn("far_pressure_air_backed=True", sharp_advance_call)
        # The closure wiring the air zone rides on stays in place.
        self.assertIn("far_pressure_region_id=pressure_load_region_id", sharp_advance_call)
        self.assertIn("far_pressure_barrier_region_id=5", sharp_advance_call)
        self.assertIn("far_pressure_pa=pressure_pa", sharp_advance_call)
        self.assertIn(
            "far_pressure_air_backed_probe_normal_sign=pressure_far_side_normal_sign",
            sharp_advance_call,
        )
        self.assertIn(
            "far_pressure_inside_probe_max_multiplier=12.0", sharp_advance_call
        )


if __name__ == "__main__":
    unittest.main()
