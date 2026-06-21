from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PACKAGE_TOKEN = "simulation" + "_code"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class SourceStaticContractTests(unittest.TestCase):
    def test_simulation_core_has_no_case_specific_or_legacy_compatibility_api(self) -> None:
        forbidden_tokens = (
            "target_velocity_z",
            "primary_velocity_z",
            "secondary_velocity_z",
            "main_velocity_z",
            "tail_velocity_z",
            "nozzle_velocity_z",
            "surface_pressure_pa",
            "pressure_force_scale",
            "fluid_feedback",
            "pressure_feedback",
            "Feedback",
            "solve_interface_reaction_step",
            "run_squid",
            "squid_soft_robot",
            LEGACY_PACKAGE_TOKEN,
        )

        for path in (REPO_ROOT / "simulation_core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{path}: {token}")

    def test_projected_ibm_config_exposes_only_full_3d_region_velocities(self) -> None:
        source = _read("simulation_core/projected_ibm.py")

        self.assertIn("primary_velocity_mps: tuple[float, float, float]", source)
        self.assertIn("secondary_velocity_mps: tuple[float, float, float]", source)
        self.assertIn("_vector3(self.primary_velocity_mps", source)
        self.assertIn("_vector3(self.secondary_velocity_mps", source)
        self.assertNotIn("primary_velocity_z_mps", source)
        self.assertNotIn("secondary_velocity_z_mps", source)
        self.assertNotIn("(0.0, 0.0, z", source)

    def test_repository_has_generic_entrypoint_and_no_deleted_legacy_runner_dirs(self) -> None:
        run_simulation_source = _read("run_simulation.py")
        cases_source = _read("cases/__init__.py")

        self.assertIn("CASE_MODULES", cases_source)
        self.assertIn("_load_case_main(case_name)(args[1:])", run_simulation_source)
        self.assertFalse((REPO_ROOT / LEGACY_PACKAGE_TOKEN).exists())
        self.assertFalse((REPO_ROOT / "squid_soft_robot_latest_core_20260603").exists())
        self.assertFalse((REPO_ROOT / "squid_soft_robot_ecoflex0010_run_20260603").exists())

    def test_fluid_projection_source_uses_consistent_compact_anisotropic_operator(self) -> None:
        source = _read("simulation_core/fluid.py")

        self.assertIn(
            ") / self.cell_width_x_m[i]",
            source,
        )
        self.assertIn("def compute_divergence(self, *, pressure_outlet_zmin: bool = False)", source)
        self.assertIn("if pressure_outlet_zmin == 1 and k == 0:", source)
        self.assertIn("bottom_velocity_z = self.velocity[i, j, k].z", source)
        self.assertIn("(right_velocity_x - left_velocity_x) / self.cell_width_x_m[i]", source)
        self.assertIn("self.obstacle[i + 1, j, k] == 0", source)
        self.assertIn("grad.x = (center - self.pressure[im, j, k]) / self.center_distance_x_m[i]", source)
        self.assertIn("ti.cast(cell_width_x_m[i], ti.f64)", source)
        self.assertIn("ti.cast(center_distance_x_m[i + 1], ti.f64)", source)
        self.assertIn("fine_cell_width_x_m[fi]", source)
        self.assertIn("weighted_residual_sum += fine_residual[fi, fj, fk] * volume_m3", source)
        self.assertIn("def _cell_volume_m3(self, i, j, k):", source)
        self.assertIn("self._cell_volume_m3(ii, jj, kk)", source)
        self.assertIn("cell_volume_m3 = self._cell_volume_m3(i, j, k)", source)
        self.assertNotIn("float(self.spec.cell_volume_m3)", source)
        self.assertNotIn("cell_force / cell_volume_m3", source)
        self.assertIn("(flux_x_forward - flux_x_backward) / self.cell_width_x_m[i]", source)
        self.assertIn("self.center_distance_x_m[i + 1]", source)
        self.assertNotIn("non-uniform CartesianGrid requires non-uniform advection/diffusion", source)
        self.assertIn("grad.z = 2.0 * center / self.cell_width_z_m[k]", source)
        self.assertIn("def _apply_closed_boundary_no_normal_flow_kernel", source)
        self.assertIn("neighbor_sum += inv_dx2 * self.pressure[i - 1, j, k]", source)
        self.assertIn("neighbor_sum += inv_dx2 * self.pressure[i + 1, j, k]", source)
        self.assertIn("denominator += 2.0 * inv_dz2", source)
        self.assertIn("denominator += inv_dx2", source)
        self.assertIn("denominator += inv_dy2", source)
        self.assertIn("denominator += inv_dz2", source)
        self.assertIn("denominator += ti.cast(2.0, ti.f64) / (", source)
        self.assertIn("* ti.cast(cell_width_z_m[k], ti.f64)", source)
        self.assertIn("max_cells: int | None = None", source)
        self.assertIn("graded grid cell count", source)
        self.assertNotIn("_level_inv_spacing2", source)
        self.assertNotIn("_mg_spacing_m", source)
        self.assertIn("self.velocity[0, j, k].x = 0.0", source)
        self.assertIn("self.velocity[i, 0, k].y = 0.0", source)
        self.assertIn("self.velocity[i, j, 0].z = 0.0", source)
        self.assertNotIn("self.velocity[self.nx - 1, j, k].x = 0.0", source)
        self.assertIn("def _clear_pressure_correction_kernel", source)
        self.assertIn("self._clear_pressure_correction_kernel()", source)
        self.assertIn("while upper - lower > 1:", source)
        self.assertNotIn("for index in range(1, count):", source)
        self.assertIn("denominator <= 0.0", source)
        self.assertIn('"projection_l2": projection_stats["l2"]', source)
        self.assertIn('"post_boundary_l2": post_boundary_stats["l2"]', source)
        self.assertIn('"post_constraint_l2": final_stats["l2"]', source)
        self.assertIn('"interior_l2": final_interior_stats["l2"]', source)
        self.assertIn("def divergence_residual_stats(self, *, interior_only: bool = False)", source)
        self.assertIn("is_interior = i > 0 and i < self.nx - 1", source)
        self.assertNotIn("elif i > 0 and self.obstacle[im, j, k] == 0", source)
        self.assertNotIn("/ (2.0 * dx)", source)
        self.assertNotIn("/ (2.0 * dy)", source)
        self.assertNotIn("/ (2.0 * dz)", source)
        self.assertNotIn("neighbor_sum / 6.0", source)

    def test_forward_divergence_backward_gradient_composes_compact_laplacian(self) -> None:
        rng = np.random.default_rng(1234)
        pressure = rng.normal(size=(9, 10, 11))
        dx, dy, dz = 0.2, 0.3, 0.45

        grad_x = np.zeros_like(pressure)
        grad_y = np.zeros_like(pressure)
        grad_z = np.zeros_like(pressure)
        grad_x[1:, :, :] = (pressure[1:, :, :] - pressure[:-1, :, :]) / dx
        grad_y[:, 1:, :] = (pressure[:, 1:, :] - pressure[:, :-1, :]) / dy
        grad_z[:, :, 1:] = (pressure[:, :, 1:] - pressure[:, :, :-1]) / dz

        divergence_of_gradient = (
            (grad_x[2:, 1:-1, 1:-1] - grad_x[1:-1, 1:-1, 1:-1]) / dx
            + (grad_y[1:-1, 2:, 1:-1] - grad_y[1:-1, 1:-1, 1:-1]) / dy
            + (grad_z[1:-1, 1:-1, 2:] - grad_z[1:-1, 1:-1, 1:-1]) / dz
        )
        compact_laplacian = (
            (pressure[2:, 1:-1, 1:-1] - 2.0 * pressure[1:-1, 1:-1, 1:-1] + pressure[:-2, 1:-1, 1:-1])
            / (dx * dx)
            + (pressure[1:-1, 2:, 1:-1] - 2.0 * pressure[1:-1, 1:-1, 1:-1] + pressure[1:-1, :-2, 1:-1])
            / (dy * dy)
            + (pressure[1:-1, 1:-1, 2:] - 2.0 * pressure[1:-1, 1:-1, 1:-1] + pressure[1:-1, 1:-1, :-2])
            / (dz * dz)
        )

        np.testing.assert_allclose(divergence_of_gradient, compact_laplacian, rtol=1.0e-12, atol=1.0e-12)

    def test_obstacle_aware_operator_matches_local_poisson_stencil(self) -> None:
        rng = np.random.default_rng(4321)
        pressure = rng.normal(size=(9, 10, 11))
        obstacle = np.zeros(pressure.shape, dtype=bool)
        obstacle[4, 4, 5] = True
        obstacle[5, 4, 5] = True
        obstacle[3, 7, 6] = True
        dx, dy, dz = 0.2, 0.3, 0.45

        grad = np.zeros(pressure.shape + (3,))
        free = ~obstacle
        grad[1:, :, :, 0] = np.where(
            free[1:, :, :] & free[:-1, :, :],
            (pressure[1:, :, :] - pressure[:-1, :, :]) / dx,
            0.0,
        )
        grad[:, 1:, :, 1] = np.where(
            free[:, 1:, :] & free[:, :-1, :],
            (pressure[:, 1:, :] - pressure[:, :-1, :]) / dy,
            0.0,
        )
        grad[:, :, 1:, 2] = np.where(
            free[:, :, 1:] & free[:, :, :-1],
            (pressure[:, :, 1:] - pressure[:, :, :-1]) / dz,
            0.0,
        )

        divergence_of_gradient = np.zeros_like(pressure)
        local_laplacian = np.zeros_like(pressure)
        inv_dx2, inv_dy2, inv_dz2 = 1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz)
        nx, ny, nz = pressure.shape
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    if obstacle[i, j, k]:
                        continue
                    if not obstacle[i + 1, j, k]:
                        divergence_of_gradient[i, j, k] += (grad[i + 1, j, k, 0] - grad[i, j, k, 0]) / dx
                        local_laplacian[i, j, k] += inv_dx2 * (pressure[i + 1, j, k] - pressure[i, j, k])
                        if not obstacle[i - 1, j, k]:
                            local_laplacian[i, j, k] += inv_dx2 * (pressure[i - 1, j, k] - pressure[i, j, k])
                    if not obstacle[i, j + 1, k]:
                        divergence_of_gradient[i, j, k] += (grad[i, j + 1, k, 1] - grad[i, j, k, 1]) / dy
                        local_laplacian[i, j, k] += inv_dy2 * (pressure[i, j + 1, k] - pressure[i, j, k])
                        if not obstacle[i, j - 1, k]:
                            local_laplacian[i, j, k] += inv_dy2 * (pressure[i, j - 1, k] - pressure[i, j, k])
                    if not obstacle[i, j, k + 1]:
                        divergence_of_gradient[i, j, k] += (grad[i, j, k + 1, 2] - grad[i, j, k, 2]) / dz
                        local_laplacian[i, j, k] += inv_dz2 * (pressure[i, j, k + 1] - pressure[i, j, k])
                        if not obstacle[i, j, k - 1]:
                            local_laplacian[i, j, k] += inv_dz2 * (pressure[i, j, k - 1] - pressure[i, j, k])

        interior_free = free[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(
            divergence_of_gradient[1:-1, 1:-1, 1:-1][interior_free],
            local_laplacian[1:-1, 1:-1, 1:-1][interior_free],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_fsi_source_spreads_fluid_stress_action_and_preserves_grid_force(self) -> None:
        source = _read("simulation_core/tri_surface.py")

        self.assertIn("sample_has_fluid_support = _sampled_fluid_weight > 1.0e-12", source)
        self.assertIn("sample_area = area if sample_has_fluid_support else 0.0", source)
        self.assertIn("pressure_force_on_solid = -sampled_pressure * n * sample_area", source)
        self.assertIn("viscous_force_on_solid = (viscous_stress @ n) * sample_area", source)
        self.assertIn("fluid_stress_force_on_solid = pressure_force_on_solid + viscous_force_on_solid", source)
        self.assertIn("stress_action_on_fluid = -fluid_stress_force_on_solid", source)
        self.assertIn("total_fluid_force = constraint_force + stress_action_on_fluid", source)
        self.assertIn("valid_weight_sum", source)
        self.assertIn("renormalized_weight = weight / valid_weight_sum", source)

    def test_tri_surface_report_marks_inapplicable_shared_fields_nonfinite(self) -> None:
        source = _read("simulation_core/tri_surface.py")

        self.assertIn("def report(self, *,", source)
        self.assertIn("stress_fields_computed: bool", source)
        self.assertIn("force_fields_computed: bool", source)
        self.assertIn("nan_vector = self._nan_vector3()", source)
        self.assertIn(
            "return self.report(stress_fields_computed=True, force_fields_computed=True)",
            source,
        )
        self.assertIn(
            "return self.report(stress_fields_computed=False, force_fields_computed=False)",
            source,
        )
        self.assertIn(
            "return self.report(stress_fields_computed=True, force_fields_computed=False)",
            source,
        )

    def test_neo_hookean_region_normal_pressure_matches_fsi_traction_sign(self) -> None:
        tri_surface_source = _read("simulation_core/tri_surface.py")
        neo_hookean_source = _read("simulation_core/neo_hookean_mpm.py")

        self.assertIn(
            "pressure_force_on_solid = -sampled_pressure * n * sample_area",
            tri_surface_source,
        )
        self.assertIn(
            "self.external_force_n[p] = -pressure_pa * self.area_weight_m2[p] * self.surface_normal[p]",
            neo_hookean_source,
        )

    def test_tri_surface_uses_cartesian_grid_fields_for_probe_mapping(self) -> None:
        source = _read("simulation_core/tri_surface.py")
        projected_ibm_source = _read("simulation_core/projected_ibm.py")
        squid_source = _read("cases/squid_soft_robot.py")

        self.assertIn("_grid_coordinate_from_fields", source)
        self.assertIn("cell_width_x_m[ii] * cell_width_y_m[jj] * cell_width_z_m[kk]", source)
        self.assertIn("def _local_normal_probe_distance_m(", source)
        self.assertIn("local_width_x_m += weight * cell_width_x_m[i0 + oi]", source)
        self.assertIn("effective_probe_distance_m = probe_distance_m", source)
        self.assertIn("effective_probe_distance_m = local_probe_distance_m", source)
        self.assertIn("control_volume_m3 = sample_area * effective_probe_distance_m", source)
        self.assertNotIn("control_volume_m3 = sample_area * local_probe_distance_m", source)
        self.assertIn("while upper - lower > 1:", source)
        self.assertNotIn("for index in range(1, count):", source)
        self.assertNotIn("(position.x - bounds_min_x) / dx - 0.5", source)
        self.assertNotIn("(probe.x - bounds_min_x) / dx - 0.5", source)
        self.assertIn("grid_fields=fluid", projected_ibm_source)
        self.assertIn("grid_fields=simulator.fluid", squid_source)
        self.assertIn("cell_center_x_m[i]", squid_source)
        self.assertIn("cell_width_x_m[i] * cell_width_y_m[j]", squid_source)
        self.assertIn("outlet_radius_m: ti.f32", squid_source)
        self.assertIn("downstream_radius_m: ti.f32", squid_source)
        self.assertIn("float(spec.outlet_plume_radius_m)", squid_source)
        self.assertIn("simulator.fluid.obstacle_cell_count()", squid_source)
        self.assertNotIn("bounds_min_x + (ti.cast(i, ti.f32) + 0.5) * dx", squid_source)
        self.assertNotIn("obstacle_volume_m3() / simulator.fluid.spec.cell_volume_m3", squid_source)

    def test_projected_ibm_reports_time_averaged_action_reaction_force(self) -> None:
        source = _read("simulation_core/projected_ibm.py")

        self.assertIn("primary_fluid_impulse_n_s", source)
        self.assertIn("secondary_fluid_impulse_n_s", source)
        self.assertIn("impulse / float(config.dt_s)", source)
        self.assertIn("primary_equivalent_fluid_force_n", source)
        self.assertIn("secondary_equivalent_fluid_force_n", source)

    def test_core_fluid_b52_regressions_cover_global_mass_and_deep_graded_mg(self) -> None:
        source = _read("tests/test_core_fluid.py")

        self.assertIn(
            "test_fv_jacobi_obstacle_adjacent_volume_source_is_globally_conservative",
            source,
        )
        self.assertIn("residual_volume_flux_m3s", source)
        self.assertIn("np.sum((solver.divergence.to_numpy()[active] - source[active])", source)
        self.assertIn(
            "test_nonuniform_fv_cg_pressure_outlet_converges_where_multigrid_does_not",
            source,
        )
        self.assertIn("test_fv_cg_weighted_laplacian_is_self_adjoint_on_graded_obstacle_grid", source)
        self.assertIn("test_fv_cg_graded_obstacle_source_balances_pressure_outlet", source)
        self.assertIn('pressure_solver="fv_cg"', source)
        self.assertIn("solver.last_cg_relative_residual", source)
        self.assertIn(
            "test_uniform_fv_laplacian_has_second_order_richardson_convergence",
            source,
        )
        self.assertIn(
            "test_nonuniform_fv_laplacian_has_monotone_richardson_convergence",
            source,
        )
        self.assertIn("expected = -3.0 * np.pi * np.pi * pressure", source)
        self.assertIn("expected = 3.0 * np.pi * np.pi * pressure.astype", source)
        self.assertIn("self.assertLess(errors[2], errors[1] * 0.30)", source)
        self.assertIn(
            "test_pressure_outlet_source_ratio_stays_grid_independent_with_refinement",
            source,
        )
        self.assertIn("self.assertLess(max(errors), 5.0e-5)", source)
        self.assertIn("self.assertLess(max(ratios) - min(ratios), 1.0e-4)", source)
        self.assertNotIn("max_error_by_resolution", source)
        self.assertNotIn("self.assertLess(errors[-1], errors[0])", source)

    def test_squid_case_validates_real_outlet_flux_and_projection_tolerance(self) -> None:
        source = _read("cases/squid_soft_robot.py")

        self.assertIn('"final_outlet_to_fsi_volume_source_ratio_physical"', source)
        self.assertIn("physical_outlet_to_fsi_volume_source_passes(", source)
        self.assertIn('"pressure_outlet_velocity_to_source_ratio"', source)
        self.assertIn('"final_pressure_outlet_velocity_to_source_ratio"', source)
        self.assertIn('"final_pressure_outlet_pressure_to_source_ratio"', source)
        self.assertNotIn("abs(final_outlet_flux_ratio) >= float(args.min_outlet_to_main_volume_flux_ratio)", source)
        self.assertIn('"projection_divergence_below_tolerance"', source)
        self.assertNotIn("max_div_l2 <= float(args.projection_divergence_tolerance)", source)
        self.assertIn("max_interior_div_l2 <= float(args.projection_divergence_tolerance)", source)
        self.assertIn('"max_interior_divergence_l2"', source)
        self.assertIn('"interior_divergence_l2"', source)
        self.assertIn("default=3000", source)
        self.assertIn("default=1.0e-2", source)
        self.assertIn("outlet_ratio={main_volume_flux_to_outlet_ratio:.6e}", source)
        self.assertNotIn("nozzle_vz={nozzle_velocity_z_mps", source)
        self.assertIn("def spec_with_nozzle_graded_grid(", source)
        self.assertIn("RefinementRegion(", source)
        self.assertIn("target_spacing_m=args.graded_grid_target_spacing_m", source)
        self.assertIn("max_cells=args.graded_grid_max_cells", source)
        self.assertIn("--graded-grid-max-cells", source)
        self.assertIn("--use-graded-grid", source)
        self.assertIn("pressure_schedule_pa(sub_time_s, spec)", source)
        self.assertIn("pressure_schedule_step_end_pa(current_time_s, spec.dt_s, spec)", source)
        self.assertIn("if time <= t0_s:\n        return p0_pa", source)
        self.assertIn("def resolve_step_count(", source)
        self.assertIn("target_time_s = max(float(spec.pressure_t2_s), float(spec.dt_s))", source)
        self.assertIn("Default runs through the full configured", source)
        self.assertIn("pressure waveform; pass an explicit small value for smoke tests", source)
        self.assertIn("default=None", source)
        self.assertNotIn('parser.add_argument("--steps", type=int, default=8)', source)
        self.assertIn("not nozzle velocity, pressure, or flow", source)
        self.assertIn(
            "prescribed_velocity_boundary=fsi_velocity_constraint_blend > 0.0",
            source,
        )
        self.assertNotIn("prescribed_velocity_boundary=False", source)
        self.assertIn('return "fv_cg" if graded_grid_enabled else "fv_multigrid"', source)
        self.assertIn("auto uses fv_multigrid on uniform FV grids", source)
        self.assertIn("fv_cg on graded FV grids", source)
        self.assertIn('"fv_cg"', source)
        self.assertNotIn('return "fv_multigrid" if graded_grid_enabled else "jacobi"', source)
        self.assertIn("pressure_solver=pressure_solver_name", source)
        self.assertIn("multigrid_cycles=effective_multigrid_cycles", source)
        self.assertIn("--multigrid-cycles", source)


if __name__ == "__main__":
    unittest.main()
