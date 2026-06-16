from __future__ import annotations

import unittest
from types import MethodType

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluid import (
    HIBM_PRESSURE_COMPONENT_CAPACITY,
    CartesianGrid,
    GradedGridSpec,
    RefinementRegion,
    build_graded_grid,
)


class CoreCartesianFluidSolverTests(unittest.TestCase):
    @staticmethod
    def _graded_obstacle_source_case() -> tuple[CartesianGrid, np.ndarray, np.ndarray, float, np.ndarray]:
        box_m = 0.024
        region = RefinementRegion(
            bounds_min_m=(0.30 * box_m, 0.30 * box_m, 0.76 * box_m),
            bounds_max_m=(0.70 * box_m, 0.70 * box_m, 1.24 * box_m),
            target_spacing_m=6.0e-4,
        )
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(box_m, box_m, 2.0 * box_m),
                farfield_spacing_m=2.5e-3,
                max_growth_ratio=1.2,
                refinement_regions=(region,),
            )
        )
        nx, ny, nz = grid.grid_nodes
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        source = np.zeros(grid.grid_nodes, dtype=np.float32)
        slab_k = nz // 2
        aperture_half_width = max(1, nx // 12)
        cx, cy = nx // 2, ny // 2
        obstacle[:, :, slab_k] = 1
        obstacle[
            cx - aperture_half_width : cx + aperture_half_width + 1,
            cy - aperture_half_width : cy + aperture_half_width + 1,
            slab_k,
        ] = 0
        cell_volume_m3 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float64)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float64)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float64)[None, None, :]
        )
        source_total_m3s = 2.0e-6
        source_cells = np.s_[
            cx - aperture_half_width : cx + aperture_half_width + 1,
            cy - aperture_half_width : cy + aperture_half_width + 1,
            nz - 3 : nz - 1,
        ]
        source[source_cells] = source_total_m3s / float(np.sum(cell_volume_m3[source_cells]))
        return grid, obstacle, source, source_total_m3s, cell_volume_m3

    @staticmethod
    def _max_adjacent_ratio(widths: tuple[float, ...]) -> float:
        return max(
            max(widths[index], widths[index + 1]) / min(widths[index], widths[index + 1])
            for index in range(len(widths) - 1)
        )

    @staticmethod
    def _count_cells_with_centers_between(
        centers: tuple[float, ...],
        lower: float,
        upper: float,
    ) -> int:
        return sum(1 for value in centers if lower <= value <= upper)

    def test_fluid_domain_spec_builds_uniform_cartesian_grid(self) -> None:
        spec = FluidDomainSpec(
            bounds_min_m=(1.0, 2.0, 3.0),
            bounds_max_m=(2.0, 4.0, 6.0),
            grid_nodes=(4, 5, 6),
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
        )

        grid = spec.cartesian_grid

        self.assertIsNotNone(grid)
        self.assertEqual(grid.grid_nodes, (4, 5, 6))
        self.assertEqual(grid.uniform_spacing_m, (0.25, 0.4, 0.5))
        np.testing.assert_allclose(grid.cell_centers_x_m, (1.125, 1.375, 1.625, 1.875))
        np.testing.assert_allclose(grid.center_distances_z_m, (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

    def test_solver_loads_uniform_cartesian_grid_axis_fields(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(-1.0, -2.0, -3.0),
            bounds_max_m=(1.0, 2.0, 3.0),
            grid_nodes=(8, 10, 12),
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=grid.grid_nodes,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))

        np.testing.assert_allclose(solver.cell_width_x_m.to_numpy(), grid.cell_widths_x_m)
        np.testing.assert_allclose(solver.cell_face_x_m.to_numpy(), grid.cell_faces_x_m)
        np.testing.assert_allclose(solver.cell_center_y_m.to_numpy(), grid.cell_centers_y_m)
        np.testing.assert_allclose(solver.center_distance_z_m.to_numpy(), grid.center_distances_z_m)

    def test_cartesian_grid_coordinate_lookup_handles_nonuniform_axes(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.5, 1.0, 1.5, 1.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )

        np.testing.assert_allclose(grid.cell_faces_x_m, (0.0, 0.5, 1.5, 3.0, 4.0))
        np.testing.assert_allclose(grid.cell_centers_x_m, (0.25, 1.0, 2.25, 3.5))
        self.assertAlmostEqual(grid.grid_coordinate_x(0.25), 0.0)
        self.assertAlmostEqual(grid.grid_coordinate_x(1.0), 1.0)
        self.assertAlmostEqual(grid.grid_coordinate_x(2.25), 2.0)
        self.assertAlmostEqual(grid.grid_coordinate_x(0.0), -0.5)
        self.assertAlmostEqual(grid.grid_coordinate_x(4.0), 3.5)
        self.assertAlmostEqual(grid.grid_coordinate_x(1.5), 1.4)

    def test_cartesian_grid_uniform_coordinate_lookup_matches_legacy_formula(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(-1.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 4, 4),
        )

        for x_m in (-1.0, -0.875, -0.125, 0.0, 0.875, 1.0):
            self.assertAlmostEqual(grid.grid_coordinate_x(x_m), (x_m + 1.0) / 0.25 - 0.5)

    def test_build_graded_grid_without_regions_degenerates_to_uniform_grid(self) -> None:
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(0.012, 0.012, 0.012),
                farfield_spacing_m=0.003,
                max_growth_ratio=1.2,
            )
        )

        self.assertTrue(grid.is_uniform)
        self.assertEqual(grid.grid_nodes, (4, 4, 4))
        np.testing.assert_allclose(grid.uniform_spacing_m, (0.003, 0.003, 0.003))

    def test_build_graded_grid_rejects_cell_count_above_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "graded grid cell count 64 exceeds max_cells 63"):
            build_graded_grid(
                GradedGridSpec(
                    bounds_min_m=(0.0, 0.0, 0.0),
                    bounds_max_m=(0.012, 0.012, 0.012),
                    farfield_spacing_m=0.003,
                    max_growth_ratio=1.2,
                    max_cells=63,
                )
            )

    def test_build_graded_grid_resolves_nozzle_refinement_region(self) -> None:
        nozzle_radius_m = 0.003
        target_spacing_m = nozzle_radius_m / 5.0
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(-0.018, -0.018, 0.0),
                bounds_max_m=(0.018, 0.018, 0.060),
                farfield_spacing_m=0.003,
                max_growth_ratio=1.2,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(-nozzle_radius_m, -nozzle_radius_m, 0.020),
                        bounds_max_m=(nozzle_radius_m, nozzle_radius_m, 0.040),
                        target_spacing_m=target_spacing_m,
                    ),
                ),
            )
        )

        x_count = self._count_cells_with_centers_between(
            grid.cell_centers_x_m,
            -nozzle_radius_m,
            nozzle_radius_m,
        )
        y_count = self._count_cells_with_centers_between(
            grid.cell_centers_y_m,
            -nozzle_radius_m,
            nozzle_radius_m,
        )
        z_count = self._count_cells_with_centers_between(grid.cell_centers_z_m, 0.020, 0.040)
        refined_x_widths = [
            width
            for center, width in zip(grid.cell_centers_x_m, grid.cell_widths_x_m, strict=True)
            if -nozzle_radius_m <= center <= nozzle_radius_m
        ]

        self.assertGreaterEqual(x_count, 10)
        self.assertGreaterEqual(y_count, 10)
        self.assertGreaterEqual(z_count, 33)
        self.assertLessEqual(max(refined_x_widths), target_spacing_m * 1.0_000_001)
        self.assertLessEqual(self._max_adjacent_ratio(grid.cell_widths_x_m), 1.2 + 1.0e-9)
        self.assertLessEqual(self._max_adjacent_ratio(grid.cell_widths_y_m), 1.2 + 1.0e-9)
        self.assertLessEqual(self._max_adjacent_ratio(grid.cell_widths_z_m), 1.2 + 1.0e-9)
        self.assertLessEqual(max(grid.cell_widths_x_m), 0.003 * 1.0_000_001)

    def test_build_graded_grid_extends_refinement_over_subfine_boundary_sliver(self) -> None:
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                farfield_spacing_m=0.30,
                max_growth_ratio=1.2,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(0.05, 0.20, 0.20),
                        bounds_max_m=(0.40, 0.80, 0.80),
                        target_spacing_m=0.10,
                    ),
                ),
            )
        )

        self.assertLessEqual(self._max_adjacent_ratio(grid.cell_widths_x_m), 1.2 + 1.0e-12)
        self.assertGreaterEqual(min(grid.cell_widths_x_m), 0.10 / 1.2)

    def test_build_graded_grid_side_transition_preserves_growth_after_length_fit(self) -> None:
        growth_ratio = 1.2
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(0.0057, 1.0, 1.0),
                farfield_spacing_m=(0.001, 1.0, 1.0),
                max_growth_ratio=growth_ratio,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(0.0012, 0.0, 0.0),
                        bounds_max_m=(0.0037, 1.0, 1.0),
                        target_spacing_m=(0.0005, 1.0, 1.0),
                    ),
                ),
            )
        )

        self.assertLessEqual(
            self._max_adjacent_ratio(grid.cell_widths_x_m),
            growth_ratio + 1.0e-12,
        )

    def test_build_graded_grid_preserves_growth_ratio_between_refinement_regions(self) -> None:
        target_spacing_m = 1.0e-3
        growth_ratio = 1.2
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(0.10, 0.01, 0.01),
                farfield_spacing_m=(0.01, 0.01, 0.01),
                max_growth_ratio=growth_ratio,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(0.020, 0.0, 0.0),
                        bounds_max_m=(0.025, 0.01, 0.01),
                        target_spacing_m=(target_spacing_m, 0.01, 0.01),
                    ),
                    RefinementRegion(
                        bounds_min_m=(0.040, 0.0, 0.0),
                        bounds_max_m=(0.045, 0.01, 0.01),
                        target_spacing_m=(target_spacing_m, 0.01, 0.01),
                    ),
                ),
            )
        )

        self.assertGreaterEqual(min(grid.cell_widths_x_m), target_spacing_m / growth_ratio)
        self.assertLessEqual(
            self._max_adjacent_ratio(grid.cell_widths_x_m),
            growth_ratio + 1.0e-12,
        )

    def test_build_graded_grid_merges_unbridgeable_mismatched_refinement_gaps(self) -> None:
        growth_ratio = 1.2
        fine_spacing_m = 1.0e-3
        coarse_spacing_m = 1.0e-2
        first_right_m = 0.025
        gap_m = fine_spacing_m / growth_ratio * 1.1
        second_left_m = first_right_m + gap_m
        grid = build_graded_grid(
            GradedGridSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(0.10, 0.01, 0.01),
                farfield_spacing_m=(coarse_spacing_m, 0.01, 0.01),
                max_growth_ratio=growth_ratio,
                refinement_regions=(
                    RefinementRegion(
                        bounds_min_m=(0.020, 0.0, 0.0),
                        bounds_max_m=(first_right_m, 0.01, 0.01),
                        target_spacing_m=(fine_spacing_m, 0.01, 0.01),
                    ),
                    RefinementRegion(
                        bounds_min_m=(second_left_m, 0.0, 0.0),
                        bounds_max_m=(second_left_m + coarse_spacing_m, 0.01, 0.01),
                        target_spacing_m=(coarse_spacing_m, 0.01, 0.01),
                    ),
                ),
            )
        )

        self.assertLessEqual(
            self._max_adjacent_ratio(grid.cell_widths_x_m),
            growth_ratio + 1.0e-12,
        )
        self.assertGreaterEqual(
            self._count_cells_with_centers_between(
                grid.cell_centers_x_m,
                first_right_m,
                second_left_m,
            ),
            1,
        )

    def test_solver_accepts_nonuniform_grid_but_guards_unsupported_paths(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )

        spec = FluidDomainSpec(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )

        self.assertEqual(spec.grid_nodes, (4, 4, 4))
        self.assertIs(spec.cartesian_grid, grid)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))

        np.testing.assert_allclose(solver.cell_width_x_m.to_numpy(), grid.cell_widths_x_m)
        np.testing.assert_allclose(
            solver._mg_cell_width_x_m[1].to_numpy(),
            (0.3, 0.7),
        )
        with self.assertRaisesRegex(ValueError, "requires an FV pressure solver"):
            solver.project(iterations=2)
        with self.assertRaisesRegex(ValueError, "non-uniform CartesianGrid divergence cleanup"):
            solver.project(
                iterations=2,
                pressure_solver="fv_jacobi",
                divergence_cleanup_iterations=1,
            )

        solver.predict()
        np.testing.assert_allclose(solver.velocity.to_numpy(), np.zeros((4, 4, 4, 3)), atol=1.0e-7)

        stats = solver.project(iterations=2, pressure_solver="fv_jacobi", reset_pressure=True)
        self.assertEqual(stats["l2"], 0.0)
        stats = solver.project(iterations=2, pressure_solver="fv_multigrid", reset_pressure=True)
        self.assertEqual(stats["l2"], 0.0)

    def test_closed_boundary_projection_reports_post_clamp_divergence(self) -> None:
        spec = FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[3, :, :, 0] = 0.8
        velocity[:, 3, :, 1] = -0.6
        velocity[:, :, 3, 2] = 0.4
        solver.velocity.from_numpy(velocity)

        stats = solver.project(
            iterations=2,
            pressure_outlet_zmin=False,
            reset_pressure=True,
            pressure_solver="fv_jacobi",
        )
        project_report_reads = solver.last_divergence_report_host_reads
        solver.compute_divergence(pressure_outlet_zmin=False)
        recomputed = solver.divergence_residual_stats()
        recomputed_raw = solver.divergence_stats()

        self.assertAlmostEqual(stats["l2"], recomputed["l2"], delta=1.0e-7)
        self.assertAlmostEqual(stats["max_abs"], recomputed["max_abs"], delta=1.0e-7)
        self.assertAlmostEqual(stats["raw_l2"], recomputed_raw["l2"], delta=1.0e-7)
        self.assertAlmostEqual(stats["raw_max_abs"], recomputed_raw["max_abs"], delta=1.0e-7)
        self.assertLessEqual(project_report_reads, 5)

    def test_projection_can_skip_report_without_skipping_projection_step(self) -> None:
        spec = FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[2, 2, 2, 0] = 0.8
        velocity[1, 2, 2, 0] = -0.4
        solver.velocity.from_numpy(velocity)

        report = solver.project(
            iterations=3,
            pressure_solver="fv_jacobi",
            reset_pressure=True,
            read_report=False,
        )
        projected_velocity = solver.velocity.to_numpy()

        self.assertEqual(report, {})
        self.assertGreater(float(np.linalg.norm(projected_velocity - velocity)), 0.0)

    def test_pressure_outlet_projection_skip_report_avoids_full_divergence_report_reads(self) -> None:
        grid_nodes = (6, 6, 6)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[2:4, 2:4, 3:5] = 1.0e-6 / (8.0 * spec.cell_volume_m3)
        solver.volume_source_s.from_numpy(source)

        report = solver.project(
            iterations=12,
            pressure_outlet_zmin=True,
            pressure_solver="fv_multigrid",
            reset_pressure=True,
            read_report=False,
        )
        velocity = solver.velocity.to_numpy()

        self.assertEqual(report, {})
        self.assertGreater(float(np.linalg.norm(velocity)), 0.0)
        self.assertLessEqual(solver.last_divergence_report_host_reads, 4)

    def test_pressure_outlet_projection_skip_report_matches_full_report_state(self) -> None:
        grid_nodes = (6, 6, 6)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[2:4, 2:4, 3:5] = 1.0e-6 / (8.0 * spec.cell_volume_m3)
        full_report_solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        minimal_report_solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        full_report_solver.volume_source_s.from_numpy(source)
        minimal_report_solver.volume_source_s.from_numpy(source)

        full_report = full_report_solver.project(
            iterations=12,
            pressure_outlet_zmin=True,
            pressure_solver="fv_multigrid",
            reset_pressure=True,
            read_report=True,
        )
        minimal_report_solver.project(
            iterations=12,
            pressure_outlet_zmin=True,
            pressure_solver="fv_multigrid",
            reset_pressure=True,
            read_report=False,
        )

        self.assertIn("interior_l2", full_report)
        self.assertLessEqual(full_report_solver.last_divergence_report_host_reads, 16)
        np.testing.assert_allclose(
            minimal_report_solver.velocity.to_numpy(),
            full_report_solver.velocity.to_numpy(),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            minimal_report_solver.pressure.to_numpy(),
            full_report_solver.pressure.to_numpy(),
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    def test_pressure_correction_clear_preserves_accumulated_pressure(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base_pressure = np.zeros((4, 4, 4), dtype=np.float32)
        correction_pressure = np.zeros((4, 4, 4), dtype=np.float32)
        base_pressure[1, 1, 1] = 3.0
        base_pressure[2, 1, 1] = -2.0
        correction_pressure[1, 1, 1] = 0.25
        correction_pressure[2, 1, 1] = 0.75

        solver.pressure.from_numpy(base_pressure)
        solver._copy_pressure_to_accum_kernel()
        solver.pressure.from_numpy(correction_pressure)
        solver.pressure_tmp.from_numpy(correction_pressure)
        solver._clear_pressure_correction_kernel()

        np.testing.assert_allclose(solver.pressure_accum.to_numpy(), base_pressure)
        np.testing.assert_allclose(solver.pressure.to_numpy(), np.zeros_like(base_pressure))
        np.testing.assert_allclose(solver.pressure_tmp.to_numpy(), np.zeros_like(base_pressure))

        solver.pressure.from_numpy(correction_pressure)
        solver._accumulate_pressure_correction_kernel()
        expected_pressure = base_pressure + correction_pressure

        np.testing.assert_allclose(solver.pressure.to_numpy(), expected_pressure)
        np.testing.assert_allclose(solver.pressure_tmp.to_numpy(), expected_pressure)
        np.testing.assert_allclose(solver.pressure_accum.to_numpy(), expected_pressure)

    def test_nonuniform_fv_divergence_and_gradient_use_grid_metrics(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))

        velocity = np.zeros(grid.grid_nodes + (3,), dtype=np.float32)
        velocity[1, 1, 1, 0] = 1.0
        velocity[2, 1, 1, 0] = 1.6
        solver.velocity.from_numpy(velocity)
        solver.compute_divergence()
        divergence = solver.divergence.to_numpy()

        self.assertAlmostEqual(float(divergence[1, 1, 1]), 3.0, delta=1.0e-6)

        pressure = np.zeros(grid.grid_nodes, dtype=np.float32)
        pressure[0, 1, 1] = 4.0
        pressure[1, 1, 1] = 10.0
        solver.velocity.from_numpy(np.zeros_like(velocity))
        solver.pressure.from_numpy(pressure)
        solver._subtract_pressure_gradient_kernel(1.0, 0)
        projected_velocity = solver.velocity.to_numpy()

        self.assertAlmostEqual(float(projected_velocity[1, 1, 1, 0]), -40.0, delta=1.0e-5)

    def test_nonuniform_grid_uses_cell_volumes_for_obstacles_and_body_force(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.2, 0.2, 0.3, 0.3),
            cell_widths_z_m=(0.1, 0.2, 0.3, 0.4),
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        cell_volume_m3 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float32)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float32)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float32)[None, None, :]
        )
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        obstacle[1, 2, 3] = 1
        obstacle[3, 0, 1] = 1
        solver.obstacle.from_numpy(obstacle)

        self.assertAlmostEqual(
            solver.obstacle_volume_m3(),
            float(cell_volume_m3[1, 2, 3] + cell_volume_m3[3, 0, 1]),
            delta=1.0e-9,
        )
        self.assertEqual(solver.obstacle_cell_count(), 2)

        force = np.zeros(grid.grid_nodes + (3,), dtype=np.float32)
        force[0, 0, 0, 0] = 200.0
        force[2, 1, 3, 0] = 300.0
        solver.obstacle.from_numpy(np.zeros(grid.grid_nodes, dtype=np.int32))
        solver.force.from_numpy(force)
        report = solver.apply_body_force(dt_s=2.0e-3)
        expected_impulse_x = float(
            (
                200.0 * cell_volume_m3[0, 0, 0]
                + 300.0 * cell_volume_m3[2, 1, 3]
            )
            * 2.0e-3
        )

        self.assertAlmostEqual(report.grid_impulse_n_s[0], expected_impulse_x, delta=1.0e-8)
        self.assertAlmostEqual(report.momentum_delta_n_s[0], expected_impulse_x, delta=1.0e-8)
        self.assertLess(report.impulse_relative_error, 1.0e-6)

    def test_nonuniform_surface_force_spreading_uses_cell_volumes(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.2, 0.2, 0.3, 0.3),
            cell_widths_z_m=(0.1, 0.2, 0.3, 0.4),
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        surface_position_m = ti.Vector.field(3, dtype=ti.f32, shape=1)
        surface_force_n = ti.Vector.field(3, dtype=ti.f32, shape=1)
        surface_position_m.from_numpy(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        surface_force_n.from_numpy(np.array([[1.0, -2.0, 3.0]], dtype=np.float32))

        spread_report = solver.spread_surface_forces(
            surface_position_m,
            surface_force_n,
            1,
            center_m=(0.05, 0.1, 0.05),
            force_sign=-1.0,
        )
        impulse_report = solver.apply_body_force(dt_s=1.5e-3)

        self.assertEqual(spread_report.active_grid_cells, 1)
        np.testing.assert_allclose(
            spread_report.grid_force_n,
            (-1.0, 2.0, -3.0),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            impulse_report.momentum_delta_n_s,
            tuple(component * 1.5e-3 for component in spread_report.grid_force_n),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        self.assertLess(impulse_report.impulse_relative_error, 1.0e-6)

    def test_fluid_domain_spec_builds_graded_grid_and_derives_grid_nodes(self) -> None:
        graded_grid = GradedGridSpec(
            bounds_min_m=(-0.018, -0.018, 0.0),
            bounds_max_m=(0.018, 0.018, 0.060),
            farfield_spacing_m=0.003,
            max_growth_ratio=1.2,
            refinement_regions=(
                RefinementRegion(
                    bounds_min_m=(-0.003, -0.003, 0.020),
                    bounds_max_m=(0.003, 0.003, 0.040),
                    target_spacing_m=0.0006,
                ),
            ),
        )

        spec = FluidDomainSpec(
            bounds_min_m=graded_grid.bounds_min_m,
            bounds_max_m=graded_grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            graded_grid=graded_grid,
        )

        self.assertFalse(spec.cartesian_grid.is_uniform)
        self.assertEqual(spec.grid_nodes, spec.cartesian_grid.grid_nodes)
        self.assertGreaterEqual(
            self._count_cells_with_centers_between(spec.cartesian_grid.cell_centers_x_m, -0.003, 0.003),
            10,
        )
        self.assertGreaterEqual(
            self._count_cells_with_centers_between(spec.cartesian_grid.cell_centers_y_m, -0.003, 0.003),
            10,
        )
        self.assertGreaterEqual(
            self._count_cells_with_centers_between(spec.cartesian_grid.cell_centers_z_m, 0.020, 0.040),
            33,
        )

    def test_fluid_domain_spec_rejects_conflicting_grid_inputs(self) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
        )
        graded_grid = GradedGridSpec(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            farfield_spacing_m=0.25,
            max_growth_ratio=1.2,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            FluidDomainSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
                graded_grid=graded_grid,
            )

    def test_multigrid_default_cycles_cover_nonuniform_grids(self) -> None:
        uniform = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.08, 0.09, 0.10, 0.13, 0.14, 0.15, 0.16, 0.15),
            cell_widths_y_m=(0.125,) * 8,
            cell_widths_z_m=(0.08, 0.09, 0.10, 0.13, 0.14, 0.15, 0.16, 0.15),
        )
        nonuniform = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        self.assertEqual(
            uniform.default_multigrid_cycles(),
            CartesianFluidSolver.DEFAULT_MULTIGRID_CYCLES,
        )
        self.assertGreaterEqual(
            nonuniform.default_multigrid_cycles(),
            1,
        )
        self.assertGreater(
            nonuniform.default_multigrid_cycles(),
            uniform.default_multigrid_cycles(),
        )
        self.assertEqual(
            nonuniform.default_multigrid_cycles(),
            CartesianFluidSolver.DEFAULT_NONUNIFORM_MULTIGRID_CYCLES,
        )

    def test_zmin_pressure_outlet_clamps_only_backflow(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        solver.set_uniform_velocity((0.0, 0.0, 0.05))
        stats = solver.project(iterations=96, pressure_outlet_zmin=True)
        positive_boundary = solver.velocity.to_numpy()[:, :, 0, 2]
        self.assertLessEqual(float(positive_boundary.max()), 1.0e-8)
        self.assertIn("pre_projection_l2", stats)
        self.assertIn("projection_l2", stats)
        self.assertIn("post_boundary_l2", stats)
        self.assertIn("post_constraint_l2", stats)
        self.assertIn("interior_l2", stats)
        self.assertIn("interior_max_abs", stats)
        self.assertLessEqual(
            stats["post_boundary_l2"],
            max(stats["projection_l2"] * 1.10, 1.0e-7),
        )
        self.assertAlmostEqual(stats["post_constraint_l2"], stats["post_boundary_l2"])

        solver.set_uniform_velocity((0.0, 0.0, -0.05))
        solver._apply_zmin_no_backflow_kernel()
        negative_boundary = solver.velocity.to_numpy()[:, :, 0, 2]
        self.assertLess(float(negative_boundary.min()), -0.049)

    def test_predict_backtrace_ignores_obstacle_velocity_support(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                viscosity_pa_s=0.0,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros(solver.spec.grid_nodes, dtype=np.int32)
        obstacle[3, 4, 4] = 1
        velocity = np.zeros(solver.spec.grid_nodes + (3,), dtype=np.float32)
        velocity[..., 0] = -4.0
        velocity[3, 4, 4, :] = 0.0
        solver.obstacle.from_numpy(obstacle)
        solver.velocity.from_numpy(velocity)

        solver.predict(dt_s=1.0 / 64.0)
        predicted = solver.velocity.to_numpy()

        self.assertAlmostEqual(float(predicted[2, 4, 4, 0]), -4.0, delta=1.0e-5)
        np.testing.assert_allclose(predicted[3, 4, 4], (0.0, 0.0, 0.0), atol=1.0e-7)

    def test_projection_enforces_closed_boundary_no_normal_flow(self) -> None:
        for pressure_solver in ("jacobi", "compact_jacobi", "fv_jacobi"):
            with self.subTest(pressure_solver=pressure_solver):
                solver = CartesianFluidSolver(
                    FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
                    runtime=TaichiRuntimeConfig(arch="cuda"),
                )
                velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
                velocity[0, :, :, 0] = -0.2
                velocity[-1, :, :, 0] = 0.3
                velocity[:, 0, :, 1] = -0.4
                velocity[:, -1, :, 1] = 0.5
                velocity[:, :, 0, 2] = -0.6
                velocity[:, :, -1, 2] = 0.7
                solver.velocity.from_numpy(velocity)

                stats = solver.project(iterations=16, pressure_solver=pressure_solver)
                projected = solver.velocity.to_numpy()

                self.assertLessEqual(float(np.max(np.abs(projected[0, :, :, 0]))), 1.0e-8)
                self.assertLessEqual(float(np.max(np.abs(projected[:, 0, :, 1]))), 1.0e-8)
                self.assertLessEqual(float(np.max(np.abs(projected[:, :, 0, 2]))), 1.0e-8)
                solver.compute_divergence()
                actual_stats = solver.divergence_residual_stats()
                actual_interior_stats = solver.divergence_residual_stats(interior_only=True)
                self.assertAlmostEqual(stats["l2"], actual_stats["l2"], delta=1.0e-7)
                self.assertAlmostEqual(stats["max_abs"], actual_stats["max_abs"], delta=1.0e-7)
                self.assertAlmostEqual(stats["interior_l2"], actual_interior_stats["l2"], delta=1.0e-7)
                self.assertAlmostEqual(
                    stats["interior_max_abs"],
                    actual_interior_stats["max_abs"],
                    delta=1.0e-7,
                )
                self.assertAlmostEqual(stats["post_constraint_l2"], actual_stats["l2"], delta=1.0e-7)
                self.assertAlmostEqual(
                    stats["post_constraint_max_abs"],
                    actual_stats["max_abs"],
                    delta=1.0e-7,
                )

    def test_closed_boundary_clamp_preserves_shifted_mac_internal_max_faces(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[0, 1, 1, 0] = -0.2
        velocity[-1, 1, 1, 0] = 0.3
        velocity[1, 0, 1, 1] = -0.4
        velocity[1, -1, 1, 1] = 0.5
        velocity[1, 1, 0, 2] = -0.6
        velocity[1, 1, -1, 2] = 0.7
        solver.velocity.from_numpy(velocity)

        solver._apply_closed_boundary_no_normal_flow_kernel(0)
        clamped = solver.velocity.to_numpy()

        self.assertEqual(float(clamped[0, 1, 1, 0]), 0.0)
        self.assertEqual(float(clamped[1, 0, 1, 1]), 0.0)
        self.assertEqual(float(clamped[1, 1, 0, 2]), 0.0)
        self.assertAlmostEqual(float(clamped[-1, 1, 1, 0]), 0.3, delta=1.0e-7)
        self.assertAlmostEqual(float(clamped[1, -1, 1, 1]), 0.5, delta=1.0e-7)
        self.assertAlmostEqual(float(clamped[1, 1, -1, 2]), 0.7, delta=1.0e-7)

    def test_zmin_pressure_outlet_reclamps_after_preserved_constraints(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        target_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        target_weight = np.zeros((4, 4, 4), dtype=np.float32)
        target_sum[:, :, 0, 2] = 0.25
        target_weight[:, :, 0] = 1.0
        solver.velocity_constraint_sum.from_numpy(target_sum)
        solver.velocity_constraint_weight.from_numpy(target_weight)

        solver.project(
            iterations=16,
            pressure_outlet_zmin=True,
            pressure_solver="fv_jacobi",
            preserve_velocity_constraints=True,
            velocity_constraint_blend=1.0,
        )
        zmin_velocity = solver.velocity.to_numpy()[:, :, 0, 2]

        self.assertLessEqual(float(np.max(zmin_velocity)), 1.0e-8)

    def test_obstacle_pressure_is_neumann_for_projection(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[4, 4, 4] = 1
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[4, 4, 4] = 1000.0
        solver.obstacle.from_numpy(obstacle)
        solver.pressure.from_numpy(pressure)

        solver.project(iterations=1)

        velocity = solver.velocity.to_numpy()
        self.assertLess(float(np.linalg.norm(velocity, axis=-1).max()), 1.0e-8)

    def test_project_uses_explicit_dt_for_pressure_gradient_correction(self) -> None:
        full_dt_solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        half_dt_solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        full_dt_solver.set_vertical_pressure_gradient(
            reference_height_m=0.5,
            gradient_z_pa_per_m=100.0,
        )
        half_dt_solver.set_vertical_pressure_gradient(
            reference_height_m=0.5,
            gradient_z_pa_per_m=100.0,
        )

        full_dt_solver.project(iterations=1, dt_s=1.0e-3)
        half_dt_solver.project(iterations=1, dt_s=5.0e-4)

        full_speed = float(np.linalg.norm(full_dt_solver.velocity.to_numpy(), axis=-1).max())
        half_speed = float(np.linalg.norm(half_dt_solver.velocity.to_numpy(), axis=-1).max())
        self.assertGreater(full_speed, 0.0)
        self.assertAlmostEqual(half_speed / full_speed, 0.5, delta=0.05)

    def test_projection_solves_consistent_divergence_gradient_system(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(24, 24, 24), dt_s=2.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        solver.set_sinusoidal_divergent_velocity(0.15)
        solver.compute_divergence()
        before = solver.divergence_stats()

        after = solver.project(iterations=480)

        self.assertGreater(before["l2"], 0.1)
        self.assertLess(after["l2"], before["l2"] * 0.05)
        self.assertLess(after["max_abs"], before["max_abs"] * 0.08)

    def test_projection_reduces_divergence_next_to_obstacle(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.16)
        solver.set_sinusoidal_divergent_velocity(0.15)
        solver.compute_divergence()
        before = solver.divergence_stats()

        after = solver.project(iterations=480, divergence_cleanup_iterations=8)

        self.assertGreater(before["l2"], 0.5)
        self.assertLess(after["projection_l2"], before["l2"] * 0.10)
        self.assertLess(after["l2"], before["l2"] * 0.03)
        self.assertLess(after["max_abs"], before["max_abs"] * 0.10)
        self.assertLessEqual(after["l2"], after["post_constraint_l2"])

    def test_local_divergence_cleanup_reduces_post_projection_residual(self) -> None:
        base_velocity_solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base_velocity_solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.16)
        base_velocity_solver.set_sinusoidal_divergent_velocity(0.15)
        velocity = base_velocity_solver.velocity.to_numpy()
        obstacle = base_velocity_solver.obstacle.to_numpy()

        projection_only = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        with_cleanup = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        for solver in (projection_only, with_cleanup):
            solver.velocity.from_numpy(velocity)
            solver.obstacle.from_numpy(obstacle)

        projected = projection_only.project(iterations=240)
        cleaned = with_cleanup.project(
            iterations=240,
            divergence_cleanup_iterations=48,
            divergence_cleanup_relaxation=0.5,
        )

        self.assertLess(cleaned["l2"], projected["l2"] * 0.97)
        self.assertLess(cleaned["max_abs"], projected["max_abs"] * 0.9)

    def test_projection_tracks_explicit_volume_source_constraint(self) -> None:
        grid_nodes = (24, 24, 24)
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        nx, ny, nz = grid_nodes
        x = (np.arange(nx, dtype=np.float32) + 0.5) / nx
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[:, :, :] = 0.08 * np.sin(2.0 * np.pi * x)[:, None, None]
        solver.volume_source_s.from_numpy(source)

        projected = solver.project(iterations=480)
        solver.compute_divergence()
        raw_divergence = solver.divergence.to_numpy()
        residual = raw_divergence - source

        self.assertGreater(projected["pre_projection_l2"], 0.05)
        self.assertLess(projected["l2"], projected["pre_projection_l2"] * 0.25)
        interior_residual = residual[2:-2, 2:-2, 2:-2]
        interior_l2 = float(np.linalg.norm(interior_residual) / interior_residual.size**0.5)
        self.assertLess(interior_l2, projected["pre_projection_l2"] * 0.25)
        self.assertGreater(projected["raw_l2"], 0.02)

    def test_pressure_outlet_divergence_counts_zmin_face_velocity(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[2, 2, 1] = 1
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[2, 2, 0, 2] = -0.25
        solver.obstacle.from_numpy(obstacle)
        solver.velocity.from_numpy(velocity)

        solver.compute_divergence()
        default_divergence = solver.divergence.to_numpy()
        solver.compute_divergence(pressure_outlet_zmin=True)
        outlet_divergence = solver.divergence.to_numpy()

        self.assertAlmostEqual(float(default_divergence[2, 2, 0]), 0.0)
        self.assertAlmostEqual(float(outlet_divergence[2, 2, 0]), 1.0, delta=1.0e-6)

    def test_divergence_uses_zero_flux_on_obstacle_faces(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[2, 1, 1] = 1
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[1, 1, 1, 0] = 1.25
        solver.obstacle.from_numpy(obstacle)
        solver.velocity.from_numpy(velocity)

        solver.compute_divergence()
        divergence = solver.divergence.to_numpy()

        self.assertAlmostEqual(float(divergence[0, 1, 1]), 5.0, delta=1.0e-6)
        self.assertAlmostEqual(float(divergence[1, 1, 1]), -5.0, delta=1.0e-6)
        self.assertAlmostEqual(float(divergence[2, 1, 1]), 0.0, delta=1.0e-6)

    def test_fv_divergence_is_pressure_gradient_adjoint_next_to_obstacles(self) -> None:
        grid_nodes = (5, 4, 4)
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros(grid_nodes, dtype=np.int32)
        obstacle[3, 1:3, 1:3] = 1
        pressure = np.zeros(grid_nodes, dtype=np.float32)
        for i in range(grid_nodes[0]):
            for j in range(grid_nodes[1]):
                for k in range(grid_nodes[2]):
                    pressure[i, j, k] = 0.7 * i - 0.3 * j + 0.11 * k + 0.05 * i * j
        pressure[obstacle == 1] = 0.0

        solver.obstacle.from_numpy(obstacle)
        solver.pressure.from_numpy(pressure)
        solver.velocity.from_numpy(np.zeros(grid_nodes + (3,), dtype=np.float32))
        dt_over_rho = 0.25
        solver._subtract_pressure_gradient_kernel(dt_over_rho, 0)
        solver.compute_divergence()
        divergence = solver.divergence.to_numpy()

        width_x = solver.cell_width_x_m.to_numpy()
        width_y = solver.cell_width_y_m.to_numpy()
        width_z = solver.cell_width_z_m.to_numpy()
        distance_x = solver.center_distance_x_m.to_numpy()
        distance_y = solver.center_distance_y_m.to_numpy()
        distance_z = solver.center_distance_z_m.to_numpy()
        expected = np.zeros(grid_nodes, dtype=np.float32)
        nx, ny, nz = grid_nodes
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if obstacle[i, j, k] != 0:
                        continue
                    applied = 0.0
                    center = float(pressure[i, j, k])
                    if i > 0 and obstacle[i - 1, j, k] == 0:
                        applied += (
                            float(pressure[i - 1, j, k]) - center
                        ) / (float(width_x[i]) * float(distance_x[i]))
                    if i < nx - 1 and obstacle[i + 1, j, k] == 0:
                        applied += (
                            float(pressure[i + 1, j, k]) - center
                        ) / (float(width_x[i]) * float(distance_x[i + 1]))
                    if j > 0 and obstacle[i, j - 1, k] == 0:
                        applied += (
                            float(pressure[i, j - 1, k]) - center
                        ) / (float(width_y[j]) * float(distance_y[j]))
                    if j < ny - 1 and obstacle[i, j + 1, k] == 0:
                        applied += (
                            float(pressure[i, j + 1, k]) - center
                        ) / (float(width_y[j]) * float(distance_y[j + 1]))
                    if k > 0 and obstacle[i, j, k - 1] == 0:
                        applied += (
                            float(pressure[i, j, k - 1]) - center
                        ) / (float(width_z[k]) * float(distance_z[k]))
                    if k < nz - 1 and obstacle[i, j, k + 1] == 0:
                        applied += (
                            float(pressure[i, j, k + 1]) - center
                        ) / (float(width_z[k]) * float(distance_z[k + 1]))
                    expected[i, j, k] = -dt_over_rho * applied

        active = obstacle == 0
        np.testing.assert_allclose(divergence[active], expected[active], rtol=2.0e-5, atol=1.0e-5)

    def test_fv_divergence_pressure_gradient_is_globally_conservative_with_obstacles(self) -> None:
        grid_nodes = (6, 5, 4)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        obstacle = np.zeros(grid_nodes, dtype=np.int32)
        obstacle[3, 1:4, 1:3] = 1
        pressure = np.zeros(grid_nodes, dtype=np.float32)
        for i in range(grid_nodes[0]):
            for j in range(grid_nodes[1]):
                for k in range(grid_nodes[2]):
                    pressure[i, j, k] = 0.17 * i * i - 0.31 * j + 0.13 * k
        pressure[obstacle == 1] = 0.0
        solver.obstacle.from_numpy(obstacle)
        solver.pressure.from_numpy(pressure)
        solver.velocity.from_numpy(np.zeros(grid_nodes + (3,), dtype=np.float32))

        solver._subtract_pressure_gradient_kernel(0.4, 0)
        solver.compute_divergence()
        net_volume_flux_m3s = float(
            np.sum(solver.divergence.to_numpy()[obstacle == 0] * spec.cell_volume_m3)
        )

        self.assertAlmostEqual(net_volume_flux_m3s, 0.0, delta=1.0e-7)

    def test_legacy_jacobi_obstacle_stencil_uses_reverse_open_neighbor(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[2, 1, 1] = 1
        pressure = np.zeros((4, 4, 4), dtype=np.float32)
        pressure[0, 1, 1] = 5.0
        solver.obstacle.from_numpy(obstacle)
        solver.pressure.from_numpy(pressure)

        solver._pressure_jacobi_kernel(1.0, 1.0, 1.0, 1.0, 0)
        solver._copy_pressure_kernel()
        updated = solver.pressure.to_numpy()

        self.assertGreater(float(updated[1, 1, 1]), 0.0)

    def test_multigrid_restriction_marks_mixed_obstacle_blocks_blocked(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[1, 1, 1] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.compute_divergence()
        solver._prepare_fv_multigrid_rhs(1.0)
        solver._compute_fv_residual_level(0, pressure_outlet_zmin=False)
        solver._mg_restrict_residual_kernel(
            solver._mg_residual[0],
            solver._mg_pressure_interface_matrix_diagonal[0],
            solver._mg_obstacle[0],
            solver._mg_velocity_dirichlet_boundary_active[0],
            solver._mg_velocity_dirichlet_boundary_projection_weight[0],
            solver._mg_cell_width_x_m[0],
            solver._mg_cell_width_y_m[0],
            solver._mg_cell_width_z_m[0],
            solver._mg_rhs[1],
            solver._mg_pressure_interface_matrix_diagonal[1],
            solver._mg_pressure[1],
            solver._mg_tmp[1],
            solver._mg_residual[1],
            solver._mg_obstacle[1],
            solver._mg_velocity_dirichlet_boundary_active[1],
            solver._mg_velocity_dirichlet_boundary_projection_weight[1],
            8,
            8,
            8,
        )

        coarse_obstacle = solver._mg_obstacle[1].to_numpy()
        coarse_velocity_dirichlet = solver._mg_velocity_dirichlet_boundary_active[1].to_numpy()

        self.assertEqual(int(coarse_obstacle[0, 0, 0]), 1)
        self.assertEqual(int(coarse_velocity_dirichlet[1, 1, 1]), 1)

    def test_multigrid_restriction_volume_averages_interface_diagonal(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fine_diagonal = np.zeros((8, 8, 8), dtype=np.float32)
        fine_diagonal[2, 2, 2] = 80.0
        solver.pressure_interface_matrix_diagonal.from_numpy(fine_diagonal)
        solver.compute_divergence()
        solver._prepare_fv_multigrid_rhs(1.0)
        solver._compute_fv_residual_level(0, pressure_outlet_zmin=False)
        solver._mg_restrict_residual_kernel(
            solver._mg_residual[0],
            solver._mg_pressure_interface_matrix_diagonal[0],
            solver._mg_obstacle[0],
            solver._mg_velocity_dirichlet_boundary_active[0],
            solver._mg_velocity_dirichlet_boundary_projection_weight[0],
            solver._mg_cell_width_x_m[0],
            solver._mg_cell_width_y_m[0],
            solver._mg_cell_width_z_m[0],
            solver._mg_rhs[1],
            solver._mg_pressure_interface_matrix_diagonal[1],
            solver._mg_pressure[1],
            solver._mg_tmp[1],
            solver._mg_residual[1],
            solver._mg_obstacle[1],
            solver._mg_velocity_dirichlet_boundary_active[1],
            solver._mg_velocity_dirichlet_boundary_projection_weight[1],
            8,
            8,
            8,
        )

        coarse_diagonal = solver._mg_pressure_interface_matrix_diagonal[1].to_numpy()
        self.assertAlmostEqual(float(coarse_diagonal[1, 1, 1]), 10.0, delta=1.0e-5)
        self.assertAlmostEqual(float(coarse_diagonal[0, 0, 0]), 0.0, delta=1.0e-6)

    def test_fv_multigrid_coarse_smoother_does_not_read_fine_interface_diagonal_by_index(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fine_diagonal = np.zeros((8, 8, 8), dtype=np.float32)
        fine_diagonal[1, 1, 1] = 1.0e8
        solver.pressure_interface_matrix_diagonal.from_numpy(fine_diagonal)
        solver._prepare_fv_multigrid_rhs(1.0)

        coarse_rhs = np.zeros(solver._mg_shapes[1], dtype=np.float32)
        coarse_rhs[1, 1, 1] = 1.0
        solver._mg_rhs[1].from_numpy(coarse_rhs)
        solver._smooth_fv_pressure_level(
            1,
            iterations=1,
            pressure_outlet_zmin=False,
            omega=1.0,
        )

        coarse_pressure = solver._mg_pressure[1].to_numpy()
        self.assertLess(float(coarse_pressure[1, 1, 1]), -1.0e-3)

    def test_legacy_jacobi_pressure_outlet_balances_explicit_volume_source(self) -> None:
        grid_nodes = (12, 12, 12)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source_total_m3s = 1.0e-6
        source = np.zeros(grid_nodes, dtype=np.float32)
        source_cells = np.s_[5:7, 5:7, 7:9]
        source[source_cells] = source_total_m3s / (8.0 * spec.cell_volume_m3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.volume_source_s.from_numpy(source)

        solver.project(
            iterations=960,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="jacobi",
        )
        report = solver.pressure_outlet_fv_flux_report()

        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=0.12,
        )

    def test_fv_multigrid_pressure_outlet_balances_explicit_volume_source(self) -> None:
        grid_nodes = (16, 16, 16)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source_total_m3s = 2.0e-6
        source = np.zeros(grid_nodes, dtype=np.float32)
        source_cells = np.s_[7:9, 7:9, 10:12]
        source[source_cells] = source_total_m3s / (8.0 * spec.cell_volume_m3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.volume_source_s.from_numpy(source)

        solver.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_multigrid",
            multigrid_cycles=8,
        )
        report = solver.pressure_outlet_fv_flux_report()
        velocity = solver.velocity.to_numpy()
        outlet_area_m2 = solver.cell_width_x_m.to_numpy()[:, None] * solver.cell_width_y_m.to_numpy()[None, :]
        velocity_field_outlet_flux_m3s = float(
            np.sum(-velocity[:, :, 0, 2] * outlet_area_m2)
        )
        solver.compute_divergence(pressure_outlet_zmin=True)
        residual_volume_flux_m3s = float(
            np.sum((solver.divergence.to_numpy() - source) * spec.cell_volume_m3)
        )

        self.assertAlmostEqual(
            report["source_volume_flux_m3s"],
            source_total_m3s,
            delta=source_total_m3s * 1.0e-5,
        )
        self.assertGreater(report["zmin_pressure_outlet_flux_m3s"], 0.0)
        self.assertGreater(report["zmin_velocity_outlet_flux_m3s"], 0.0)
        self.assertAlmostEqual(
            report["zmin_pressure_outlet_to_source_ratio"],
            1.0,
            delta=0.08,
        )
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=0.08,
        )
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_flux_m3s"],
            report["zmin_pressure_outlet_flux_m3s"],
            delta=source_total_m3s * 0.02,
        )
        self.assertAlmostEqual(
            velocity_field_outlet_flux_m3s,
            source_total_m3s,
            delta=source_total_m3s * 0.08,
        )
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.02,
        )

    def test_uniform_fv_laplacian_has_second_order_richardson_convergence(self) -> None:
        errors: list[float] = []
        for resolution in (16, 32, 64):
            spacing = 1.0 / resolution
            centers = (np.arange(resolution, dtype=np.float64) + 0.5) * spacing
            x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
            pressure = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
            expected = -3.0 * np.pi * np.pi * pressure
            discrete = np.zeros_like(pressure)
            discrete[1:-1, 1:-1, 1:-1] = (
                pressure[2:, 1:-1, 1:-1]
                + pressure[:-2, 1:-1, 1:-1]
                + pressure[1:-1, 2:, 1:-1]
                + pressure[1:-1, :-2, 1:-1]
                + pressure[1:-1, 1:-1, 2:]
                + pressure[1:-1, 1:-1, :-2]
                - 6.0 * pressure[1:-1, 1:-1, 1:-1]
            ) / (spacing * spacing)
            error = discrete[1:-1, 1:-1, 1:-1] - expected[1:-1, 1:-1, 1:-1]
            errors.append(float(np.sqrt(np.mean(error * error))))

        self.assertLess(errors[1], errors[0] * 0.30)
        self.assertLess(errors[2], errors[1] * 0.30)

    def test_nonuniform_fv_laplacian_has_monotone_richardson_convergence(self) -> None:
        relative_errors: list[float] = []
        for resolution in (12, 18, 24):
            q = (np.arange(resolution, dtype=np.float64) + 0.5) / float(resolution)
            raw_widths = 1.0 + 0.3 * np.sin(2.0 * np.pi * q)
            widths = tuple(float(value) for value in raw_widths / np.sum(raw_widths))
            grid = CartesianGrid(
                bounds_min_m=(0.0, 0.0, 0.0),
                cell_widths_x_m=widths,
                cell_widths_y_m=widths,
                cell_widths_z_m=widths,
            )
            spec = FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            )
            solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
            x = np.asarray(grid.cell_centers_x_m, dtype=np.float64)[:, None, None]
            y = np.asarray(grid.cell_centers_y_m, dtype=np.float64)[None, :, None]
            z = np.asarray(grid.cell_centers_z_m, dtype=np.float64)[None, None, :]
            pressure = (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)).astype(np.float32)
            expected = 3.0 * np.pi * np.pi * pressure.astype(np.float64)

            solver.pressure_tmp.from_numpy(pressure)
            solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
            applied = solver.cg_r.to_numpy().astype(np.float64)
            interior = np.zeros(grid.grid_nodes, dtype=bool)
            interior[2:-2, 2:-2, 2:-2] = True
            error = applied[interior] - expected[interior]
            relative_errors.append(
                float(np.sqrt(np.mean(error * error)) / np.sqrt(np.mean(expected[interior] ** 2)))
            )

        self.assertLess(relative_errors[1], relative_errors[0])
        self.assertLess(relative_errors[2], relative_errors[1])
        self.assertLess(relative_errors[2], 4.0e-3)

    def test_pressure_outlet_source_ratio_stays_grid_independent_with_refinement(self) -> None:
        source_total_m3s = 2.0e-6
        ratios: list[float] = []
        errors: list[float] = []
        for resolution in (8, 12, 16):
            grid_nodes = (resolution, resolution, resolution)
            spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
            source = np.zeros(grid_nodes, dtype=np.float32)
            x0 = resolution // 2 - 1
            y0 = resolution // 2 - 1
            z0 = max(2, (3 * resolution) // 4 - 1)
            source_cells = np.s_[x0 : x0 + 2, y0 : y0 + 2, z0 : z0 + 2]
            source[source_cells] = source_total_m3s / (8.0 * spec.cell_volume_m3)
            solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
            solver.volume_source_s.from_numpy(source)

            solver.project(
                iterations=8,
                pressure_outlet_zmin=True,
                reset_pressure=True,
                pressure_solver="fv_multigrid",
                multigrid_cycles=CartesianFluidSolver.DEFAULT_MULTIGRID_CYCLES,
            )
            ratio = solver.pressure_outlet_fv_flux_report()["zmin_velocity_outlet_to_source_ratio"]
            ratios.append(ratio)
            errors.append(abs(1.0 - ratio))

        self.assertTrue(all(ratio > 0.0 for ratio in ratios))
        self.assertLess(max(errors), 5.0e-5)
        self.assertLess(max(ratios) - min(ratios), 1.0e-4)

    def test_pressure_outlet_flux_report_uses_net_section_flux_without_reusing_final_pressure(self) -> None:
        grid_nodes = (4, 4, 4)
        spec = FluidDomainSpec.unit_box(
            grid_nodes=grid_nodes,
            density_kgm3=1.0,
            dt_s=1.0,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        source_total_m3s = 0.25
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[0, 0, 1] = source_total_m3s / spec.cell_volume_m3
        velocity = np.zeros((*grid_nodes, 3), dtype=np.float32)
        velocity[0, 0, 0, 2] = -8.0
        velocity[1, 0, 0, 2] = 4.0
        pressure = np.zeros(grid_nodes, dtype=np.float32)
        pressure[0, 0, 0] = 1.0
        pressure[1, 0, 0] = -0.5
        solver.volume_source_s.from_numpy(source)
        solver.velocity.from_numpy(velocity)
        solver.pressure.from_numpy(pressure)

        report = solver.pressure_outlet_fv_flux_report()

        self.assertEqual(solver.last_pressure_outlet_report_host_reads, 1)
        self.assertAlmostEqual(report["source_volume_flux_m3s"], source_total_m3s)
        self.assertAlmostEqual(report["zmin_velocity_outlet_flux_m3s"], source_total_m3s)
        self.assertAlmostEqual(report["zmin_pressure_outlet_flux_m3s"], 0.0)
        self.assertAlmostEqual(report["zmin_velocity_outlet_to_source_ratio"], 1.0)
        self.assertAlmostEqual(report["zmin_pressure_outlet_to_source_ratio"], 0.0)

    def test_nonuniform_fv_jacobi_pressure_outlet_balances_explicit_volume_source(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.08, 0.09, 0.10, 0.12, 0.16, 0.17, 0.14, 0.14),
            cell_widths_y_m=(0.125,) * 8,
            cell_widths_z_m=(0.10, 0.10, 0.11, 0.12, 0.14, 0.15, 0.14, 0.14),
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        source_total_m3s = 8.0e-7
        source = np.zeros(grid.grid_nodes, dtype=np.float32)
        cell_volume_m3 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float32)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float32)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float32)[None, None, :]
        )
        source_cells = np.s_[3:5, 3:5, 5:7]
        source[source_cells] = source_total_m3s / float(np.sum(cell_volume_m3[source_cells]))
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.volume_source_s.from_numpy(source)

        solver.project(
            iterations=1200,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_jacobi",
        )
        report = solver.pressure_outlet_fv_flux_report()
        velocity = solver.velocity.to_numpy()
        outlet_area_m2 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float32)[:, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float32)[None, :]
        )
        velocity_field_outlet_flux_m3s = float(
            np.sum(-velocity[:, :, 0, 2] * outlet_area_m2)
        )
        solver.compute_divergence(pressure_outlet_zmin=True)
        residual_volume_flux_m3s = float(np.sum((solver.divergence.to_numpy() - source) * cell_volume_m3))

        self.assertAlmostEqual(
            report["source_volume_flux_m3s"],
            source_total_m3s,
            delta=source_total_m3s * 0.02,
        )
        self.assertGreater(report["zmin_velocity_outlet_flux_m3s"], 0.0)
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=0.15,
        )
        self.assertAlmostEqual(
            velocity_field_outlet_flux_m3s,
            source_total_m3s,
            delta=source_total_m3s * 0.15,
        )
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.05,
        )

    def test_fv_jacobi_obstacle_adjacent_volume_source_is_globally_conservative(self) -> None:
        grid_nodes = (14, 10, 10)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source_total_m3s = 1.0e-6
        source = np.zeros(grid_nodes, dtype=np.float32)
        obstacle = np.zeros(grid_nodes, dtype=np.int32)
        source_cells = np.s_[7:8, 4:6, 5:7]
        obstacle[8, 4:6, 5:7] = 1
        source[source_cells] = source_total_m3s / (4.0 * spec.cell_volume_m3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)

        solver.project(
            iterations=2400,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_jacobi",
        )
        report = solver.pressure_outlet_fv_flux_report()
        solver.compute_divergence(pressure_outlet_zmin=True)
        active = obstacle == 0
        residual_volume_flux_m3s = float(
            np.sum((solver.divergence.to_numpy()[active] - source[active]) * spec.cell_volume_m3)
        )

        self.assertAlmostEqual(
            report["source_volume_flux_m3s"],
            source_total_m3s,
            delta=source_total_m3s * 1.0e-5,
        )
        self.assertGreater(report["zmin_velocity_outlet_flux_m3s"], 0.0)
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=0.12,
        )
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.01,
        )

    def test_nonuniform_fv_cg_pressure_outlet_converges_where_multigrid_does_not(self) -> None:
        widths = tuple(0.02 * (1.0 + 0.15 * ((index % 4) / 3.0)) for index in range(24))
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=widths,
            cell_widths_y_m=widths,
            cell_widths_z_m=widths,
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        source_total_m3s = 2.0e-6
        source = np.zeros(grid.grid_nodes, dtype=np.float32)
        cell_volume_m3 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float32)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float32)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float32)[None, None, :]
        )
        source_cells = np.s_[11:13, 11:13, 18:20]
        source[source_cells] = source_total_m3s / float(np.sum(cell_volume_m3[source_cells]))
        jacobi = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        cg = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        jacobi.volume_source_s.from_numpy(source)
        cg.volume_source_s.from_numpy(source)

        jacobi.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_jacobi",
        )
        cg.project(
            iterations=160,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )
        jacobi_report = jacobi.pressure_outlet_fv_flux_report()
        cg_report = cg.pressure_outlet_fv_flux_report()
        jacobi_error = abs(1.0 - jacobi_report["zmin_velocity_outlet_to_source_ratio"])
        cg_error = abs(1.0 - cg_report["zmin_velocity_outlet_to_source_ratio"])

        self.assertGreaterEqual(len(cg._mg_shapes), 4)
        self.assertTrue(cg.last_cg_converged, cg.last_cg_breakdown)
        self.assertLess(cg.last_cg_iterations, 160)
        self.assertLess(cg.last_cg_relative_residual, 1.0e-6)
        self.assertGreater(cg_report["zmin_velocity_outlet_flux_m3s"], 0.0)
        self.assertLess(cg_error, jacobi_error * 0.05)
        self.assertLess(cg_error, 1.0e-3)

    def test_fv_cg_weighted_laplacian_is_self_adjoint_on_graded_obstacle_grid(self) -> None:
        grid, obstacle, _source, _source_total_m3s, _cell_volume_m3 = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        rng = np.random.default_rng(17)
        x = rng.normal(size=grid.grid_nodes).astype(np.float32)
        y = rng.normal(size=grid.grid_nodes).astype(np.float32)
        x[obstacle == 1] = 0.0
        y[obstacle == 1] = 0.0
        solver.obstacle.from_numpy(obstacle)
        solver.pressure_tmp.from_numpy(x)
        solver.cg_d.from_numpy(y)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 1)
        solver._fv_laplacian_apply_kernel(solver.cg_d, solver.cg_Ad, 1)
        lhs = float(solver._weighted_dot_kernel(solver.cg_r, solver.cg_d))
        rhs = float(solver._weighted_dot_kernel(solver.pressure_tmp, solver.cg_Ad))
        relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)

        self.assertLess(relative_error, 1.0e-5)

    def test_fv_cg_interface_matrix_diagonal_enters_operator_and_preconditioner(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.ones((5, 5, 5), dtype=np.float32)
        diagonal = np.zeros((5, 5, 5), dtype=np.float32)
        diagonal[2, 2, 2] = 7.5
        residual = np.zeros((5, 5, 5), dtype=np.float32)
        residual[2, 2, 2] = 15.0
        solver.pressure_tmp.from_numpy(pressure)
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.cg_r.from_numpy(residual)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)
        solver._fv_diagonal_kernel(solver.fv_diag, 0)
        solver._apply_jacobi_preconditioner_kernel(solver.cg_r, solver.cg_z)

        operator = solver.cg_Ad.to_numpy()
        fv_diag = solver.fv_diag.to_numpy()
        preconditioned = solver.cg_z.to_numpy()
        base_diag = 6.0 / (0.2 * 0.2)
        self.assertAlmostEqual(operator[2, 2, 2], 7.5, delta=1.0e-5)
        self.assertAlmostEqual(fv_diag[2, 2, 2], base_diag + 7.5, delta=3.0e-5)
        self.assertAlmostEqual(
            preconditioned[2, 2, 2],
            15.0 / (base_diag + 7.5),
            delta=1.0e-6,
        )

    def test_fv_cg_uses_f64_work_vectors_for_high_contrast_interface_rows(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        self.assertEqual(solver.cg_r.dtype, ti.f64)
        self.assertEqual(solver.cg_rhs.dtype, ti.f64)
        self.assertEqual(solver.cg_z.dtype, ti.f64)
        self.assertEqual(solver.cg_d.dtype, ti.f64)
        self.assertEqual(solver.cg_Ad.dtype, ti.f64)
        self.assertEqual(solver.cg_r_old.dtype, ti.f64)
        self.assertEqual(solver.bicgstab_s.dtype, ti.f64)
        self.assertEqual(solver.bicgstab_t.dtype, ti.f64)
        self.assertEqual(solver.pressure.dtype, ti.f64)
        self.assertEqual(solver.pressure_interface_projection_divergence_s.dtype, ti.f64)
        self.assertEqual(solver.pressure_interface_matrix_diagonal.dtype, ti.f64)
        self.assertEqual(solver.pressure_interface_matrix_rhs.dtype, ti.f64)
        self.assertEqual(solver.hibm_pressure_fill_next.dtype, ti.f64)
        self.assertEqual(solver.reduction_sum.dtype, ti.f64)
        self.assertEqual(solver.reduction_max.dtype, ti.f64)
        self.assertEqual(solver.divergence_combined_sum.dtype, ti.f64)
        self.assertEqual(solver.divergence_combined_max.dtype, ti.f64)
        self.assertEqual(solver._mg_rhs[0].dtype, ti.f64)
        self.assertEqual(solver._mg_residual[0].dtype, ti.f64)

    def test_interface_matrix_forces_fv_multigrid_projection_to_fv_cg(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        interface_diagonal = np.zeros((5, 5, 5), dtype=np.float64)
        interface_diagonal[2, 2, 2] = 1.0
        solver.pressure_interface_matrix_diagonal.from_numpy(interface_diagonal)

        report = solver.project(
            iterations=4,
            pressure_solver="fv_multigrid",
            pressure_solve_failure_policy="report",
        )

        self.assertEqual(report["pressure_solver_requested"], "fv_multigrid")
        self.assertEqual(report["pressure_solver"], "fv_cg")
        self.assertTrue(report["pressure_solver_forced_to_fv_cg"])
        self.assertEqual(
            report["pressure_solver_force_reason"],
            "pressure_interface_matrix_requires_rowlist_fv_cg",
        )

    def test_fv_cg_interface_coupling_enters_operator_symmetrically(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.ones((5, 5, 5), dtype=np.float32)
        pressure[2, 2, 2] = 3.0
        pressure[2, 2, 3] = 1.0
        solver.pressure_tmp.from_numpy(pressure)
        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        base_operator = solver.cg_r.to_numpy()
        diagonal = np.zeros((5, 5, 5), dtype=np.float32)
        diagonal[2, 2, 2] = 4.0
        diagonal[2, 2, 3] = 4.0
        cell_volume_m3 = 0.2 * 0.2 * 0.2
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_coupling_active[2, 2, 2] = 1
        solver.pressure_interface_coupling_neighbor[2, 2, 2] = (2, 2, 3)
        solver.pressure_interface_coupling_coefficient[2, 2, 2] = (
            4.0 * cell_volume_m3
        )

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)

        coupled_operator = solver.cg_Ad.to_numpy()
        self.assertAlmostEqual(
            coupled_operator[2, 2, 2] - base_operator[2, 2, 2],
            8.0,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            coupled_operator[2, 2, 3] - base_operator[2, 2, 3],
            -8.0,
            delta=1.0e-5,
        )

    def test_fv_cg_interface_coupling_uses_transmissibility_on_nonuniform_cells(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.35, 0.35),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        rng = np.random.default_rng(41)
        x = rng.normal(size=grid.grid_nodes).astype(np.float32)
        y = rng.normal(size=grid.grid_nodes).astype(np.float32)
        transmissibility = 0.375
        left = (1, 2, 2)
        right = (2, 2, 2)
        left_volume = 0.2 * 0.25 * 0.25
        right_volume = 0.35 * 0.25 * 0.25
        diagonal = np.zeros(grid.grid_nodes, dtype=np.float32)
        diagonal[left] = transmissibility / left_volume
        diagonal[right] = transmissibility / right_volume
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_coupling_active[left] = 1
        solver.pressure_interface_coupling_neighbor[left] = right
        solver.pressure_interface_coupling_coefficient[left] = transmissibility
        solver.pressure_tmp.from_numpy(x)
        solver.cg_d.from_numpy(y)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        solver._fv_laplacian_apply_kernel(solver.cg_d, solver.cg_Ad, 0)

        lhs = float(solver._weighted_dot_kernel(solver.cg_r, solver.cg_d))
        rhs = float(solver._weighted_dot_kernel(solver.pressure_tmp, solver.cg_Ad))
        relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
        self.assertLess(relative_error, 1.0e-5)

    def test_fv_cg_interface_coupling_applies_multiple_owner_slots(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.ones((5, 5, 5), dtype=np.float32)
        owner = (2, 2, 2)
        neighbor0 = (2, 2, 3)
        neighbor1 = (2, 3, 2)
        pressure[owner] = 3.0
        pressure[neighbor0] = 1.0
        pressure[neighbor1] = 2.0
        solver.pressure_tmp.from_numpy(pressure)
        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        base_operator = solver.cg_r.to_numpy()

        cell_volume_m3 = 0.2**3
        coefficient0 = 4.0 * cell_volume_m3
        coefficient1 = 7.0 * cell_volume_m3
        diagonal = np.zeros((5, 5, 5), dtype=np.float32)
        diagonal[owner] = (coefficient0 + coefficient1) / cell_volume_m3
        diagonal[neighbor0] = coefficient0 / cell_volume_m3
        diagonal[neighbor1] = coefficient1 / cell_volume_m3
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_coupling_active[owner] = 2
        solver.pressure_interface_coupling_neighbor[owner] = neighbor0
        solver.pressure_interface_coupling_coefficient[owner] = coefficient0
        solver.pressure_interface_coupling_extra_neighbor[
            owner[0],
            owner[1],
            owner[2],
            0,
        ] = neighbor1
        solver.pressure_interface_coupling_extra_coefficient[
            owner[0],
            owner[1],
            owner[2],
            0,
        ] = coefficient1

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)

        operator = solver.cg_Ad.to_numpy()
        self.assertAlmostEqual(
            operator[owner] - base_operator[owner],
            15.0,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            operator[neighbor0] - base_operator[neighbor0],
            -8.0,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            operator[neighbor1] - base_operator[neighbor1],
            -7.0,
            delta=5.0e-5,
        )

    def test_fv_cg_interface_matrix_rhs_enters_positive_cg_rhs(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        interface_rhs = np.zeros((5, 5, 5), dtype=np.float32)
        interface_rhs[2, 2, 2] = 3.25
        solver.pressure_interface_matrix_rhs.from_numpy(interface_rhs)

        solver._prepare_fv_multigrid_rhs(rhs_scale=1.0)
        solver._cg_build_positive_rhs_kernel(solver._mg_rhs[0], solver.cg_z, 0.0)

        positive_rhs = solver.cg_z.to_numpy()
        self.assertAlmostEqual(positive_rhs[2, 2, 2], 3.25, delta=1.0e-6)

    def test_pressure_gradient_does_not_modify_hibm_velocity_dirichlet_row(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((5, 5, 5), 10.0, dtype=np.float32)
        pressure[2, 2, 1] = 2.0
        solver.pressure.from_numpy(pressure)
        solver.velocity[2, 2, 2] = (0.25, -0.5, 0.75)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_value_mps[2, 2, 2] = (0.25, -0.5, 0.75)

        solver._subtract_pressure_gradient_kernel(1.0e-6, 0)

        velocity = tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3))
        self.assertEqual(velocity, (0.25, -0.5, 0.75))

    def test_preserve_projected_rows_reclamps_reconstructed_hibm_dirichlet_row(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        solver.velocity[node] = (9.0, 9.0, 9.0)
        solver.velocity_dirichlet_boundary_active[node] = 1
        solver.velocity_dirichlet_boundary_value_mps[node] = (0.25, -0.5, 0.75)
        solver.velocity_dirichlet_boundary_projection_weight[node] = 0.5

        solver._apply_velocity_dirichlet_boundary_rows_kernel(1, 1)

        velocity = tuple(float(solver.velocity[node][axis]) for axis in range(3))
        self.assertEqual(velocity, (0.25, -0.5, 0.75))
        report = solver.velocity_dirichlet_boundary_report()
        self.assertEqual(report.active_cells, 1)
        self.assertGreater(report.max_delta_mps, 0.0)

    def test_fv_cg_projection_reduces_velocity_dirichlet_injected_divergence(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(6, 6, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (3, 3, 3)
        solver.velocity_dirichlet_boundary_active[node] = 1
        solver.velocity_dirichlet_boundary_value_mps[node] = (0.0, 0.0, -0.1)
        solver.velocity_dirichlet_boundary_projection_weight[node] = 1.0

        report = solver.project(
            iterations=512,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            reset_pressure=True,
            read_report=True,
        )

        self.assertGreater(report["pre_projection_l2"], 1.0e-4)
        self.assertLess(report["projection_l2"], report["pre_projection_l2"] * 0.25)
        self.assertLess(report["l2"], report["pre_projection_l2"] * 0.25)

    def test_fv_cg_laplacian_applies_neumann_row_at_hibm_dirichlet(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((5, 5, 5), 3.0, dtype=np.float32)
        pressure[2, 2, 1] = 1.0
        solver.pressure_tmp.from_numpy(pressure)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        base_operator = solver.cg_r.to_numpy()

        solver.pressure_interface_matrix_diagonal[2, 2, 2] = 4.0
        solver.pressure_interface_matrix_diagonal[2, 2, 1] = 4.0
        solver.pressure_interface_coupling_active[2, 2, 2] = 1
        solver.pressure_interface_coupling_neighbor[2, 2, 2] = (2, 2, 1)
        solver.pressure_interface_coupling_coefficient[2, 2, 2] = 4.0 * (0.2**3)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)

        operator = solver.cg_Ad.to_numpy()
        self.assertAlmostEqual(
            operator[2, 2, 2] - base_operator[2, 2, 2],
            8.0,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            operator[2, 2, 1] - base_operator[2, 2, 1],
            -8.0,
            delta=5.0e-5,
        )

    def test_fv_diagonal_uses_reconstructed_hibm_dirichlet_projection_weight(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[1, 2, 2] = 0
        obstacle[2, 2, 2] = 0
        solver.obstacle.from_numpy(obstacle)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_projection_weight[2, 2, 2] = 0.5

        solver._fv_diagonal_kernel(solver.fv_diag, 0)

        diagonal = solver.fv_diag.to_numpy()
        half_face_coefficient = 0.5 / (0.2 * 0.2)
        self.assertAlmostEqual(
            float(diagonal[2, 2, 2]),
            half_face_coefficient,
            delta=1.0e-5,
        )

    def test_fv_cg_laplacian_keeps_hibm_dirichlet_face_prescribed_with_reconstruction_weight(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((5, 5, 5), 3.0, dtype=np.float32)
        pressure[2, 2, 1] = 1.0
        solver.pressure_tmp.from_numpy(pressure)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_projection_weight[2, 2, 2] = 0.5
        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        base_operator = solver.cg_r.to_numpy()

        solver.pressure_interface_matrix_diagonal[2, 2, 2] = 4.0
        solver.pressure_interface_matrix_diagonal[2, 2, 1] = 4.0
        solver.pressure_interface_coupling_active[2, 2, 2] = 1
        solver.pressure_interface_coupling_neighbor[2, 2, 2] = (2, 2, 1)
        solver.pressure_interface_coupling_coefficient[2, 2, 2] = 4.0 * (0.2**3)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)

        operator = solver.cg_Ad.to_numpy()
        self.assertAlmostEqual(
            operator[2, 2, 2] - base_operator[2, 2, 2],
            8.0,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            operator[2, 2, 1] - base_operator[2, 2, 1],
            -8.0,
            delta=5.0e-5,
        )

    def test_fv_laplacian_matches_gradient_response_at_reconstructed_dirichlet_row(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[2, 2, 2] = 3.0
        pressure[1, 2, 2] = 1.0
        solver.pressure_tmp.from_numpy(pressure)
        solver.pressure.from_numpy(pressure)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_projection_weight[2, 2, 2] = 0.5
        solver.velocity_dirichlet_boundary_value_mps[2, 2, 2] = (0.0, 0.0, 0.0)

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)
        solver._subtract_pressure_gradient_kernel(1.0, 0)
        solver.compute_divergence()

        operator = solver.cg_Ad.to_numpy()
        divergence_response = solver.divergence.to_numpy()
        np.testing.assert_allclose(
            operator,
            divergence_response,
            rtol=1.0e-6,
            atol=5.0e-5,
        )

    def test_fv_laplacian_includes_pressure_neumann_term_on_fixed_dirichlet_face(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[2, 2, 2] = 3.0
        pressure[2, 2, 1] = 1.0
        solver.pressure_tmp.from_numpy(pressure)
        solver.pressure.from_numpy(pressure)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_value_mps[2, 2, 2] = (0.0, 0.0, 0.0)
        transmissibility = 4.0 * (0.2**3)
        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_r, 0)
        base_operator = solver.cg_r.to_numpy()

        solver.pressure_interface_matrix_diagonal[2, 2, 2] = 4.0
        solver.pressure_interface_matrix_diagonal[2, 2, 1] = 4.0
        solver.pressure_interface_coupling_active[2, 2, 2] = 1
        solver.pressure_interface_coupling_neighbor[2, 2, 2] = (2, 2, 1)
        solver.pressure_interface_coupling_coefficient[2, 2, 2] = transmissibility

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)
        solver._subtract_pressure_gradient_kernel(1.0, 0)
        solver.compute_divergence()

        operator = solver.cg_Ad.to_numpy()
        self.assertAlmostEqual(
            operator[2, 2, 2] - base_operator[2, 2, 2],
            8.0,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            operator[2, 2, 1] - base_operator[2, 2, 1],
            -8.0,
            delta=5.0e-5,
        )
        self.assertEqual(
            tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )

    def test_pressure_interface_row_list_projection_flux_enters_divergence_report(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[2, 2, 2] = 3.0
        pressure[2, 2, 1] = 1.0
        solver.pressure.from_numpy(pressure)
        solver.pressure_tmp.from_numpy(pressure)

        owner = (2, 2, 2)
        neighbor = (2, 2, 1)
        cell_volume_m3 = 0.2**3
        transmissibility = 4.0 * cell_volume_m3
        solver.pressure_interface_matrix_diagonal[owner] = transmissibility / cell_volume_m3
        solver.pressure_interface_matrix_diagonal[neighbor] = (
            transmissibility / cell_volume_m3
        )
        solver.pressure_interface_row_count[None] = 1
        solver.pressure_interface_row_owner[0] = owner
        solver.pressure_interface_row_neighbor[0] = neighbor
        solver.pressure_interface_row_transmissibility[0] = transmissibility

        solver._fv_laplacian_apply_kernel(solver.pressure_tmp, solver.cg_Ad, 0)
        solver._subtract_pressure_gradient_kernel(1.0, 0)
        solver._update_pressure_interface_projection_divergence_kernel(1.0)
        solver.compute_divergence()

        operator = solver.cg_Ad.to_numpy()
        total_divergence = (
            solver.divergence.to_numpy()
            + solver.pressure_interface_projection_divergence_s.to_numpy()
        )
        np.testing.assert_allclose(total_divergence, operator, rtol=1.0e-6, atol=5.0e-5)
        stats = solver.divergence_residual_stats()
        self.assertAlmostEqual(
            stats["max_abs"],
            float(np.max(np.abs(operator))),
            delta=5.0e-5,
        )

    def test_hibm_solid_band_nonprojectable_cells_are_masked(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[3, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[2, 3, 2] = 1
        solver.velocity_dirichlet_boundary_active[2, 2, 3] = 1
        solver.velocity[2, 2, 2] = (1.0, -2.0, 3.0)
        solver.velocity_prev[2, 2, 2] = (4.0, -5.0, 6.0)
        solver.volume_source_s[2, 2, 2] = 7.0

        masked = solver.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=False,
        )

        self.assertEqual(masked, 1)
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 1)
        self.assertEqual(int(solver.obstacle[1, 2, 2]), 0)
        self.assertEqual(
            tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            tuple(float(solver.velocity_prev[2, 2, 2][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(float(solver.volume_source_s[2, 2, 2]), 0.0)

    def test_hibm_solid_band_nonprojectable_boundary_cells_are_masked(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[0, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[1, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 3, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 2, 3] = 1
        solver.velocity[0, 2, 2] = (1.0, -2.0, 3.0)
        solver.velocity_prev[0, 2, 2] = (4.0, -5.0, 6.0)
        solver.volume_source_s[0, 2, 2] = 7.0

        masked = solver.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=False,
        )

        self.assertEqual(masked, 1)
        self.assertEqual(int(solver.obstacle[0, 2, 2]), 1)
        self.assertEqual(int(solver.obstacle[0, 1, 2]), 0)
        self.assertEqual(
            tuple(float(solver.velocity[0, 2, 2][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            tuple(float(solver.velocity_prev[0, 2, 2][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(float(solver.volume_source_s[0, 2, 2]), 0.0)

    def test_hibm_pressure_outlet_disconnected_component_is_anchored_not_frozen(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((5, 5, 5), dtype=np.int32)
        obstacle[:, :, 2] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.velocity_dirichlet_boundary_active[2, 2, 4] = 1
        solver.velocity[2, 2, 4] = (1.0, -2.0, 3.0)
        solver.velocity_prev[2, 2, 4] = (4.0, -5.0, 6.0)
        solver.volume_source_s[2, 2, 4] = 7.0

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 50)
        self.assertEqual(solver.last_hibm_pressure_unreached_cell_count, 50)
        self.assertTrue(solver.last_hibm_pressure_reachability_converged)
        kept_obstacle = solver.obstacle.to_numpy()
        self.assertTrue(np.all(kept_obstacle[:, :, 3:] == 0))
        self.assertEqual(int(kept_obstacle[2, 2, 1]), 0)
        self.assertEqual(
            tuple(float(solver.velocity[2, 2, 4][axis]) for axis in range(3)),
            (1.0, -2.0, 3.0),
        )
        self.assertEqual(
            tuple(float(solver.velocity_prev[2, 2, 4][axis]) for axis in range(3)),
            (4.0, -5.0, 6.0),
        )
        self.assertAlmostEqual(float(solver.volume_source_s[2, 2, 4]), 7.0)

    def test_fv_cg_anchors_pressure_mean_over_outlet_unreached_set(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((5, 5, 5), dtype=np.int32)
        obstacle[:, :, 2] = 1
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 50)
        velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
        velocity[2, 2, 4, 0] = 0.4
        solver.velocity.from_numpy(velocity)

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertGreaterEqual(
            int(report.get("cg_unreached_set_mean_projection_count", 0)),
            1,
        )

    def test_fv_cg_anchors_multiple_disconnected_components_independently(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[6, 2, 6] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 3)
        self.assertEqual(solver.last_hibm_pressure_unreached_component_count, 2)
        self.assertFalse(solver.last_hibm_pressure_unreached_component_overflow)
        solver.volume_source_s[2, 2, 4] = 1.0
        solver.volume_source_s[2, 2, 5] = 3.0
        solver.volume_source_s[6, 2, 6] = -2.0

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertEqual(int(report.get("cg_unreached_component_count", 0)), 2)
        self.assertGreaterEqual(
            int(report.get("cg_unreached_set_mean_projection_count", 0)),
            1,
        )

    def test_fv_cg_reports_unreached_component_size_distribution(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[6, 2, 6] = 0
        solver.obstacle.from_numpy(obstacle)

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 3)
        self.assertEqual(solver.last_hibm_pressure_unreached_component_count, 2)
        self.assertEqual(solver.last_hibm_pressure_unreached_component_raw_count, 2)
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_largest_cell_count,
            2,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_singleton_count,
            1,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_small_threshold_cells,
            128,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_small_count,
            2,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_small_cell_count,
            3,
        )

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertEqual(int(report["cg_unreached_component_raw_count"]), 2)
        self.assertEqual(
            int(report["cg_unreached_component_largest_cell_count"]),
            2,
        )
        self.assertEqual(int(report["cg_unreached_component_singleton_count"]), 1)
        self.assertEqual(
            int(report["cg_unreached_component_small_threshold_cells"]),
            128,
        )
        self.assertEqual(int(report["cg_unreached_component_small_count"]), 2)
        self.assertEqual(int(report["cg_unreached_component_small_cell_count"]), 3)

    def test_fv_cg_cleans_raw_singleton_overflow_before_projection(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(23, 23, 23), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((23, 23, 23), dtype=np.int32)
        obstacle[:, :, 0] = 0
        singleton_count = 0
        for i in range(1, 23, 2):
            for j in range(1, 23, 2):
                for k in range(2, 23, 2):
                    obstacle[i, j, k] = 0
                    singleton_count += 1
        solver.obstacle.from_numpy(obstacle)

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, singleton_count)
        self.assertGreater(singleton_count, HIBM_PRESSURE_COMPONENT_CAPACITY)
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_count,
            HIBM_PRESSURE_COMPONENT_CAPACITY,
        )
        self.assertTrue(solver.last_hibm_pressure_unreached_component_overflow)
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_raw_count,
            singleton_count,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_largest_cell_count,
            1,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_singleton_count,
            singleton_count,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_small_count,
            singleton_count,
        )
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_small_cell_count,
            singleton_count,
        )

        report = solver.project(
            iterations=64,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
            pressure_solve_failure_policy="report",
        )

        self.assertFalse(report["pressure_projection_physical_failure"])
        self.assertEqual(report["pressure_projection_physical_failure_reason"], "")
        self.assertEqual(
            int(report["hibm_projection_overflow_singleton_cleanup_cell_count"]),
            singleton_count,
        )
        self.assertEqual(
            int(
                report[
                    "hibm_projection_overflow_singleton_cleanup_component_count"
                ]
            ),
            singleton_count,
        )
        self.assertEqual(int(report["cg_unreached_component_count"]), 0)
        self.assertEqual(int(report["cg_unreached_component_raw_count"]), 0)
        self.assertEqual(
            int(report["cg_unreached_component_largest_cell_count"]),
            0,
        )
        self.assertEqual(
            int(report["cg_unreached_component_singleton_count"]),
            0,
        )

    def test_fv_cg_reports_unreached_component_rhs_incompatibility(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[6, 2, 6] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 3)
        solver.volume_source_s[2, 2, 4] = 1.0
        solver.volume_source_s[2, 2, 5] = 3.0
        solver.volume_source_s[6, 2, 6] = -2.0

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertFalse(report["pressure_solve_failed"])
        self.assertTrue(report["pressure_projection_physical_failure"])
        self.assertEqual(
            report["pressure_projection_physical_failure_reason"],
            "unreached_component_rhs_incompatible",
        )
        self.assertEqual(
            int(report["hibm_unreached_incompatible_component_count"]),
            2,
        )
        self.assertGreater(
            float(report["hibm_unreached_component_rhs_mean_max_abs"]),
            0.0,
        )

    def test_fv_cg_tiny_unreached_cleanup_is_opt_in(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        tiny_cells = ((2, 2, 4), (2, 2, 5))
        large_cells = tuple((6, 2, k) for k in range(2, 7))
        for cell in tiny_cells + large_cells:
            obstacle[cell] = 0
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s[2, 2, 4] = 1.0
        solver.volume_source_s[2, 2, 5] = 3.0
        solver.volume_source_s[6, 2, 3] = -2.0

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
            hibm_tiny_unreached_cleanup_component_cells=4,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertTrue(report["pressure_projection_physical_failure"])
        self.assertEqual(
            report["pressure_projection_physical_failure_reason"],
            "unreached_component_rhs_incompatible",
        )
        self.assertEqual(
            int(report["hibm_projection_tiny_unreached_cleanup_cell_count"]),
            len(tiny_cells),
        )
        self.assertEqual(
            int(report["hibm_projection_tiny_unreached_cleanup_component_count"]),
            1,
        )
        self.assertEqual(int(report["cg_unreached_component_raw_count"]), 1)
        self.assertEqual(
            int(report["cg_unreached_component_largest_cell_count"]),
            len(large_cells),
        )
        self.assertEqual(
            int(report["hibm_unreached_incompatible_component_count"]),
            1,
        )
        obstacle_after = solver.obstacle.to_numpy()
        self.assertTrue(all(int(obstacle_after[cell]) == 1 for cell in tiny_cells))
        self.assertTrue(all(int(obstacle_after[cell]) == 0 for cell in large_cells))

    def test_fv_cg_reports_unreached_component_label_overflow_as_physical_failure(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(31, 31, 31), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((31, 31, 31), dtype=np.int32)
        obstacle[:, :, 0] = 0
        component_count = 0
        target_component_count = HIBM_PRESSURE_COMPONENT_CAPACITY + 1
        for i in range(1, 30, 2):
            if component_count >= target_component_count:
                break
            for j in range(1, 30, 2):
                if component_count >= target_component_count:
                    break
                for k in range(2, 29, 3):
                    if component_count >= target_component_count:
                        break
                    obstacle[i, j, k] = 0
                    obstacle[i, j, k + 1] = 0
                    component_count += 1
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 2 * component_count)
        self.assertGreater(component_count, HIBM_PRESSURE_COMPONENT_CAPACITY)
        self.assertTrue(solver.last_hibm_pressure_unreached_component_overflow)
        self.assertEqual(
            solver.last_hibm_pressure_unreached_component_singleton_count,
            0,
        )

        report = solver.project(
            iterations=64,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertTrue(report["pressure_projection_physical_failure"])
        self.assertEqual(
            report["pressure_projection_physical_failure_reason"],
            "unreached_component_label_overflow",
        )
        self.assertTrue(report["cg_unreached_component_overflow"])
        self.assertEqual(
            int(report["hibm_projection_overflow_singleton_cleanup_cell_count"]),
            0,
        )

    def test_fv_cg_raises_on_unreached_component_rhs_incompatibility(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 2)
        solver.volume_source_s[2, 2, 4] = 1.0
        solver.volume_source_s[2, 2, 5] = 3.0

        with self.assertRaisesRegex(
            RuntimeError,
            "unreached component RHS is incompatible",
        ):
            solver.project(
                iterations=200,
                pressure_outlet_zmin=True,
                pressure_solver="fv_cg",
                cg_tolerance=1.0e-6,
                pressure_solve_failure_policy="raise",
            )

    def test_final_divergence_report_excludes_anchored_unreached_cells(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[6, 2, 6] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 3)
        solver.volume_source_s[2, 2, 4] = 1.0
        solver.volume_source_s[2, 2, 5] = 3.0
        solver.volume_source_s[6, 2, 6] = -2.0

        residual_stats = solver.divergence_residual_stats(interior_only=True)
        (
            _final_raw,
            final_stats,
            _final_interior_raw,
            final_interior_stats,
        ) = solver.final_divergence_report_stats(pressure_outlet_zmin=True)

        self.assertAlmostEqual(residual_stats["l2"], 0.0, delta=1.0e-12)
        self.assertAlmostEqual(final_stats["l2"], 0.0, delta=1.0e-12)
        self.assertAlmostEqual(final_interior_stats["l2"], 0.0, delta=1.0e-12)
        self.assertEqual(int(solver.last_unreached_divergence_stats["count"]), 3)
        self.assertGreater(float(solver.last_unreached_divergence_stats["l2"]), 1.0)

    def test_pressure_outlet_reachability_fill_reaches_fixed_point_on_long_path(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 4, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((16, 4, 16), dtype=np.int32)
        obstacle[:, :, 4] = 1
        obstacle[15, :, 4] = 0
        obstacle[:, :, 8] = 1
        obstacle[0, :, 8] = 0
        obstacle[:, :, 12] = 1
        obstacle[15, :, 12] = 0
        solver.obstacle.from_numpy(obstacle)

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 0)
        self.assertTrue(solver.last_hibm_pressure_reachability_converged)
        self.assertEqual(int(solver.obstacle[8, 1, 15]), 0)

    def test_divergence_partition_reports_velocity_dirichlet_near_field(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        divergence = np.zeros((4, 4, 4), dtype=np.float32)
        divergence[1, 1, 1] = 2.0
        divergence[3, 3, 3] = 3.0
        solver.divergence.from_numpy(divergence)
        solver.velocity_dirichlet_boundary_active[1, 1, 1] = 1

        near_raw, near_residual, far_raw, far_residual = (
            solver.divergence_dirichlet_partition_report_stats()
        )

        self.assertAlmostEqual(near_raw["max_abs"], 2.0, delta=1.0e-6)
        self.assertAlmostEqual(near_residual["max_abs"], 2.0, delta=1.0e-6)
        self.assertAlmostEqual(far_raw["max_abs"], 3.0, delta=1.0e-6)
        self.assertAlmostEqual(far_residual["max_abs"], 3.0, delta=1.0e-6)

    def test_project_report_splits_pressure_correctable_and_fixed_dirichlet_divergence(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        divergence = np.zeros((4, 4, 4), dtype=np.float32)
        divergence[1, 1, 1] = 2.0
        divergence[2, 2, 2] = 3.0
        solver.divergence.from_numpy(divergence)
        for node in (
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
        ):
            solver.velocity_dirichlet_boundary_active[node] = 1

        (
            *unused,
            pressure_correctable_raw,
            pressure_correctable_residual,
            pressure_fixed_raw,
            pressure_fixed_residual,
            interior_pressure_correctable_raw,
            interior_pressure_correctable_residual,
            interior_pressure_fixed_raw,
            interior_pressure_fixed_residual,
        ) = solver.final_and_dirichlet_partition_report_stats()

        self.assertAlmostEqual(pressure_correctable_raw["max_abs"], 3.0, delta=1.0e-6)
        self.assertAlmostEqual(pressure_correctable_residual["max_abs"], 3.0, delta=1.0e-6)
        self.assertAlmostEqual(pressure_fixed_raw["max_abs"], 2.0, delta=1.0e-6)
        self.assertAlmostEqual(pressure_fixed_residual["max_abs"], 2.0, delta=1.0e-6)
        self.assertGreater(pressure_correctable_residual["count"], 0)
        self.assertGreater(pressure_fixed_residual["count"], 0)
        self.assertAlmostEqual(
            interior_pressure_correctable_raw["max_abs"],
            3.0,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            interior_pressure_correctable_residual["max_abs"],
            3.0,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(interior_pressure_fixed_raw["max_abs"], 2.0, delta=1.0e-6)
        self.assertAlmostEqual(
            interior_pressure_fixed_residual["max_abs"],
            2.0,
            delta=1.0e-6,
        )

    def test_pressure_interface_matrix_terms_report_integrates_grid_fields(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        diagonal = np.zeros((5, 5, 5), dtype=np.float32)
        rhs = np.zeros((5, 5, 5), dtype=np.float32)
        cell_volume = 0.2**3
        diagonal[2, 2, 2] = 4.0 / cell_volume
        rhs[2, 2, 2] = -6.0 / cell_volume
        diagonal[1, 2, 2] = 2.0 / cell_volume
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_matrix_rhs.from_numpy(rhs)

        report = solver.pressure_interface_matrix_terms_report()

        self.assertAlmostEqual(report["diagonal_integral"], 6.0, delta=1.0e-5)
        self.assertAlmostEqual(report["rhs_integral"], -6.0, delta=1.0e-5)
        self.assertAlmostEqual(report["max_abs_diagonal"], 4.0 / cell_volume, delta=1.0e-4)
        self.assertEqual(report["active_cells"], 2)
        self.assertEqual(report["row_count"], 0)
        self.assertEqual(report["row_active_count"], 0)
        self.assertAlmostEqual(report["row_diagonal_integral"], 0.0, delta=1.0e-8)
        self.assertAlmostEqual(
            report["row_diagonal_integral_abs_mismatch"],
            6.0,
            delta=1.0e-5,
        )

    def test_pressure_interface_matrix_terms_report_checks_rowlist_diagonal_closure(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        owner = (2, 2, 2)
        neighbor = (2, 2, 3)
        cell_volume = 0.2**3
        transmissibility = 0.5
        diagonal = np.zeros((5, 5, 5), dtype=np.float32)
        diagonal[owner] = transmissibility / cell_volume
        diagonal[neighbor] = transmissibility / cell_volume
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_row_count[None] = 1
        solver.pressure_interface_row_owner[0] = owner
        solver.pressure_interface_row_neighbor[0] = neighbor
        solver.pressure_interface_row_transmissibility[0] = transmissibility

        closed_report = solver.pressure_interface_matrix_terms_report()

        self.assertEqual(closed_report["row_count"], 1)
        self.assertEqual(closed_report["row_active_count"], 1)
        self.assertEqual(closed_report["row_invalid_count"], 0)
        self.assertAlmostEqual(
            closed_report["row_diagonal_integral"],
            2.0 * transmissibility,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            closed_report["row_diagonal_integral_abs_mismatch"],
            0.0,
            delta=1.0e-6,
        )

        diagonal[neighbor] = 0.0
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)

        open_report = solver.pressure_interface_matrix_terms_report()

        self.assertEqual(open_report["row_count"], 1)
        self.assertEqual(open_report["row_active_count"], 1)
        self.assertGreater(
            open_report["row_diagonal_integral_abs_mismatch"],
            0.49,
        )
        self.assertGreater(
            open_report["row_diagonal_max_abs_density_mismatch"],
            0.49 / cell_volume,
        )

    def test_fv_cg_graded_obstacle_source_balances_pressure_outlet(self) -> None:
        grid, obstacle, source, source_total_m3s, cell_volume_m3 = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)

        solver.project(
            iterations=160,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )
        report = solver.pressure_outlet_fv_flux_report()
        solver.compute_divergence(pressure_outlet_zmin=True)
        active = obstacle == 0
        residual_volume_flux_m3s = float(
            np.sum((solver.divergence.to_numpy()[active] - source[active]) * cell_volume_m3[active])
        )

        self.assertGreater(max(grid.cell_widths_z_m) / min(grid.cell_widths_z_m), 4.0)
        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertLess(solver.last_cg_iterations, 120)
        self.assertLess(solver.last_cg_relative_residual, 1.0e-6)
        self.assertLessEqual(solver.last_cg_host_residual_checks, 6)
        self.assertAlmostEqual(
            report["source_volume_flux_m3s"],
            source_total_m3s,
            delta=source_total_m3s * 1.0e-5,
        )
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=2.0e-3,
        )
        self.assertTrue(np.isfinite(report["zmin_pressure_outlet_to_source_ratio"]))
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.01,
        )

    def test_fv_cg_jacobi_preconditioner_keeps_graded_outlet_balance(self) -> None:
        grid, obstacle, source, source_total_m3s, cell_volume_m3 = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)

        project_report = solver.project(
            iterations=512,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
            cg_preconditioner="jacobi",
        )
        report = solver.pressure_outlet_fv_flux_report()
        solver.compute_divergence(pressure_outlet_zmin=True)
        active = obstacle == 0
        residual_volume_flux_m3s = float(
            np.sum((solver.divergence.to_numpy()[active] - source[active]) * cell_volume_m3[active])
        )

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertEqual(project_report["cg_project_calls"], 1)
        self.assertLess(solver.last_cg_relative_residual, 1.0e-6)
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=2.0e-3,
        )
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.01,
        )

    def test_fv_cg_light_multigrid_preconditioner_keeps_graded_outlet_balance(self) -> None:
        grid, obstacle, source, source_total_m3s, cell_volume_m3 = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)

        project_report = solver.project(
            iterations=256,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
            cg_preconditioner="fv_multigrid_light",
        )
        report = solver.pressure_outlet_fv_flux_report()
        solver.compute_divergence(pressure_outlet_zmin=True)
        active = obstacle == 0
        residual_volume_flux_m3s = float(
            np.sum((solver.divergence.to_numpy()[active] - source[active]) * cell_volume_m3[active])
        )

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertEqual(project_report["cg_project_calls"], 1)
        self.assertLess(solver.last_cg_relative_residual, 1.0e-6)
        self.assertAlmostEqual(
            report["zmin_velocity_outlet_to_source_ratio"],
            1.0,
            delta=2.0e-3,
        )
        self.assertAlmostEqual(
            residual_volume_flux_m3s,
            0.0,
            delta=source_total_m3s * 0.01,
        )

    def test_fv_cg_auto_uses_jacobi_when_interface_couplings_are_active(self) -> None:
        grid, obstacle, source, *_ = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)
        nx, ny, nz = grid.grid_nodes
        owner = (nx // 2, ny // 2, nz // 2 + 1)
        neighbor = (nx // 2, ny // 2, nz // 2)
        owner_volume = (
            grid.cell_widths_x_m[owner[0]]
            * grid.cell_widths_y_m[owner[1]]
            * grid.cell_widths_z_m[owner[2]]
        )
        neighbor_volume = (
            grid.cell_widths_x_m[neighbor[0]]
            * grid.cell_widths_y_m[neighbor[1]]
            * grid.cell_widths_z_m[neighbor[2]]
        )
        transmissibility = 0.25 * min(owner_volume, neighbor_volume)
        solver.pressure_interface_matrix_diagonal[owner] = transmissibility / owner_volume
        solver.pressure_interface_matrix_diagonal[neighbor] = (
            transmissibility / neighbor_volume
        )
        solver.pressure_interface_coupling_active[owner] = 1
        solver.pressure_interface_coupling_neighbor[owner] = neighbor
        solver.pressure_interface_coupling_coefficient[owner] = transmissibility
        multigrid_calls: list[object] = []

        original_multigrid_preconditioner = solver._apply_fv_multigrid_preconditioner

        def counted_multigrid_preconditioner(self, *_args, **_kwargs) -> None:
            multigrid_calls.append(_args)
            return original_multigrid_preconditioner(*_args, **_kwargs)

        solver._apply_fv_multigrid_preconditioner = MethodType(
            counted_multigrid_preconditioner,
            solver,
        )

        solver.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_preconditioner="auto",
            pressure_solve_failure_policy="report",
            read_report=False,
        )

        self.assertEqual(len(multigrid_calls), 0)
        self.assertEqual(solver.last_project_cg_project_calls, 1)

    def test_fv_cg_auto_uses_jacobi_when_interface_rowlist_is_active(self) -> None:
        grid, obstacle, source, *_ = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)
        nx, ny, nz = grid.grid_nodes
        owner = (nx // 2, ny // 2, nz // 2 + 1)
        neighbor = (nx // 2, ny // 2, nz // 2)
        owner_volume = (
            grid.cell_widths_x_m[owner[0]]
            * grid.cell_widths_y_m[owner[1]]
            * grid.cell_widths_z_m[owner[2]]
        )
        neighbor_volume = (
            grid.cell_widths_x_m[neighbor[0]]
            * grid.cell_widths_y_m[neighbor[1]]
            * grid.cell_widths_z_m[neighbor[2]]
        )
        transmissibility = 0.25 * min(owner_volume, neighbor_volume)
        solver.pressure_interface_matrix_diagonal[owner] = transmissibility / owner_volume
        solver.pressure_interface_matrix_diagonal[neighbor] = (
            transmissibility / neighbor_volume
        )
        solver.pressure_interface_row_count[None] = 1
        solver.pressure_interface_row_owner[0] = owner
        solver.pressure_interface_row_neighbor[0] = neighbor
        solver.pressure_interface_row_transmissibility[0] = transmissibility
        multigrid_calls: list[object] = []

        original_multigrid_preconditioner = solver._apply_fv_multigrid_preconditioner

        def counted_multigrid_preconditioner(self, *_args, **_kwargs) -> None:
            multigrid_calls.append(_args)
            return original_multigrid_preconditioner(*_args, **_kwargs)

        solver._apply_fv_multigrid_preconditioner = MethodType(
            counted_multigrid_preconditioner,
            solver,
        )

        report = solver.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_preconditioner="auto",
            pressure_solve_failure_policy="report",
            read_report=True,
        )

        self.assertEqual(len(multigrid_calls), 0)
        self.assertEqual(report["pressure_interface_matrix_row_active_count"], 1)
        self.assertTrue(report["pressure_interface_matrix_active"])
        self.assertEqual(solver.last_project_cg_project_calls, 1)

    def test_fv_cg_explicit_multigrid_preconditioner_is_disabled_for_interface_couplings(
        self,
    ) -> None:
        grid, obstacle, source, *_ = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)
        nx, ny, nz = grid.grid_nodes
        owner = (nx // 2, ny // 2, nz // 2 + 1)
        neighbor = (nx // 2, ny // 2, nz // 2)
        owner_volume = (
            grid.cell_widths_x_m[owner[0]]
            * grid.cell_widths_y_m[owner[1]]
            * grid.cell_widths_z_m[owner[2]]
        )
        neighbor_volume = (
            grid.cell_widths_x_m[neighbor[0]]
            * grid.cell_widths_y_m[neighbor[1]]
            * grid.cell_widths_z_m[neighbor[2]]
        )
        transmissibility = 0.25 * min(owner_volume, neighbor_volume)
        solver.pressure_interface_matrix_diagonal[owner] = transmissibility / owner_volume
        solver.pressure_interface_matrix_diagonal[neighbor] = (
            transmissibility / neighbor_volume
        )
        solver.pressure_interface_coupling_active[owner] = 1
        solver.pressure_interface_coupling_neighbor[owner] = neighbor
        solver.pressure_interface_coupling_coefficient[owner] = transmissibility
        multigrid_calls: list[object] = []

        original_multigrid_preconditioner = solver._apply_fv_multigrid_preconditioner

        def counted_multigrid_preconditioner(self, *_args, **_kwargs) -> None:
            multigrid_calls.append(_args)
            return original_multigrid_preconditioner(*_args, **_kwargs)

        solver._apply_fv_multigrid_preconditioner = MethodType(
            counted_multigrid_preconditioner,
            solver,
        )

        solver.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_preconditioner="fv_multigrid_light",
            pressure_solve_failure_policy="report",
            read_report=False,
        )

        self.assertEqual(len(multigrid_calls), 0)
        self.assertEqual(solver.last_project_cg_project_calls, 1)

    def test_fv_cg_auto_uses_multigrid_on_graded_grid_without_interface_couplings(
        self,
    ) -> None:
        grid, obstacle, source, *_ = self._graded_obstacle_source_case()
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.obstacle.from_numpy(obstacle)
        solver.volume_source_s.from_numpy(source)
        multigrid_calls: list[object] = []
        original_multigrid_preconditioner = solver._apply_fv_multigrid_preconditioner

        def counted_multigrid_preconditioner(self, *_args, **_kwargs) -> None:
            multigrid_calls.append(_args)
            return original_multigrid_preconditioner(*_args, **_kwargs)

        solver._apply_fv_multigrid_preconditioner = MethodType(
            counted_multigrid_preconditioner,
            solver,
        )

        solver.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_preconditioner="auto",
            pressure_solve_failure_policy="report",
            read_report=False,
        )

        self.assertGreater(len(multigrid_calls), 0)
        self.assertEqual(solver.last_project_cg_project_calls, 1)

    def test_fv_cg_closed_domain_mean_projection_stays_device_side(self) -> None:
        widths = tuple(0.03 * (1.0 + 0.1 * ((index % 3) - 1)) for index in range(10))
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=widths,
            cell_widths_y_m=widths,
            cell_widths_z_m=widths,
        )
        spec = FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        )
        cell_volume_m3 = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float64)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float64)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float64)[None, None, :]
        )
        source = np.zeros(grid.grid_nodes, dtype=np.float32)
        source_total_m3s = 1.0e-7
        positive_cells = np.s_[2:4, 2:4, 2:4]
        negative_cells = np.s_[6:8, 6:8, 6:8]
        source[positive_cells] = source_total_m3s / float(np.sum(cell_volume_m3[positive_cells]))
        source[negative_cells] = -source_total_m3s / float(np.sum(cell_volume_m3[negative_cells]))
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        solver.volume_source_s.from_numpy(source)

        report = solver.project(
            iterations=96,
            pressure_outlet_zmin=False,
            reset_pressure=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        pressure = solver.pressure.to_numpy().astype(np.float64)
        weighted_pressure_mean = float(np.sum(pressure * cell_volume_m3) / np.sum(cell_volume_m3))

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertLess(solver.last_cg_relative_residual, 1.0e-6)
        self.assertEqual(report["cg_project_calls"], 1)
        self.assertEqual(report["cg_iterations_total"], solver.last_cg_iterations)
        self.assertLess(report["cg_iterations_total"], 96)
        self.assertTrue(report["cg_converged_all"])
        self.assertLess(report["cg_relative_residual_max"], 1.0e-6)
        self.assertAlmostEqual(weighted_pressure_mean, 0.0, delta=1.0e-7)
        self.assertEqual(solver.last_cg_mean_host_reads, 0)
        self.assertGreater(solver.last_cg_mean_projection_count, 0)
        self.assertEqual(
            report["cg_mean_projection_count"],
            solver.last_cg_mean_projection_count,
        )
        self.assertEqual(
            report["cg_mean_projection_count"],
            solver.last_project_cg_mean_projection_count,
        )

    def test_fv_cg_zero_rhs_discards_stale_pressure_without_reset_pressure(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(6, 6, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        stale_pressure = np.zeros((6, 6, 6), dtype=np.float32)
        stale_pressure[:, :, :] = np.linspace(0.0, 1.0, 6, dtype=np.float32)[None, None, :]
        solver.pressure.from_numpy(stale_pressure)
        velocity_before = solver.velocity.to_numpy()

        report = solver.project(
            iterations=8,
            pressure_outlet_zmin=False,
            reset_pressure=False,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        np.testing.assert_allclose(solver.velocity.to_numpy(), velocity_before, atol=1.0e-12)
        np.testing.assert_allclose(solver.pressure.to_numpy(), np.zeros_like(stale_pressure), atol=1.0e-12)
        self.assertTrue(report["cg_converged_all"])
        self.assertAlmostEqual(report["cg_initial_relative_residual_max"], 0.0)

    def test_fv_multigrid_outlet_balance_converges_faster_than_fv_jacobi(self) -> None:
        grid_nodes = (16, 16, 16)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source_total_m3s = 2.0e-6
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[7:9, 7:9, 10:12] = source_total_m3s / (8.0 * spec.cell_volume_m3)
        jacobi = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        multigrid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
        jacobi.volume_source_s.from_numpy(source)
        multigrid.volume_source_s.from_numpy(source)

        jacobi.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_jacobi",
        )
        multigrid.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_multigrid",
            multigrid_cycles=8,
        )
        jacobi_error = abs(1.0 - jacobi.pressure_outlet_fv_flux_report()["zmin_velocity_outlet_to_source_ratio"])
        multigrid_error = abs(
            1.0 - multigrid.pressure_outlet_fv_flux_report()["zmin_velocity_outlet_to_source_ratio"]
        )

        self.assertLess(multigrid_error, jacobi_error * 0.5)

    def test_fv_multigrid_default_cycles_are_decoupled_from_iteration_budget(self) -> None:
        grid_nodes = (16, 16, 16)
        spec = FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3)
        source_total_m3s = 2.0e-6
        source = np.zeros(grid_nodes, dtype=np.float32)
        source[7:9, 7:9, 10:12] = source_total_m3s / (8.0 * spec.cell_volume_m3)
        default_cycles = CartesianFluidSolver(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        explicit_cycles = CartesianFluidSolver(
            spec,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        default_cycles.volume_source_s.from_numpy(source)
        explicit_cycles.volume_source_s.from_numpy(source)

        default_cycles.project(
            iterations=3000,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_multigrid",
        )
        explicit_cycles.project(
            iterations=8,
            pressure_outlet_zmin=True,
            reset_pressure=True,
            pressure_solver="fv_multigrid",
            multigrid_cycles=CartesianFluidSolver.DEFAULT_MULTIGRID_CYCLES,
        )
        default_report = default_cycles.pressure_outlet_fv_flux_report()
        explicit_report = explicit_cycles.pressure_outlet_fv_flux_report()

        self.assertAlmostEqual(
            default_report["zmin_velocity_outlet_to_source_ratio"],
            explicit_report["zmin_velocity_outlet_to_source_ratio"],
            delta=1.0e-6,
        )
        self.assertGreater(default_report["zmin_velocity_outlet_to_source_ratio"], 0.98)

    def test_predict_preserves_uniform_velocity_without_forces(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        solver.set_uniform_velocity((0.12, -0.03, 0.04))
        before = solver.velocity.to_numpy()
        solver.predict()
        after = solver.velocity.to_numpy()

        self.assertLess(float(np.max(np.abs(after - before))), 1.0e-6)

    def test_nonuniform_predict_preserves_uniform_velocity_without_forces(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.08, 0.10, 0.12, 0.16, 0.18, 0.17, 0.11, 0.08),
            cell_widths_y_m=(0.09, 0.11, 0.15, 0.16, 0.14, 0.13, 0.12, 0.10),
            cell_widths_z_m=(0.07, 0.09, 0.12, 0.15, 0.17, 0.16, 0.14, 0.10),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        solver.set_uniform_velocity((0.01, -0.015, 0.02))
        before = solver.velocity.to_numpy()
        solver.predict()
        after = solver.velocity.to_numpy()

        self.assertLess(float(np.max(np.abs(after - before))), 1.0e-6)

    def test_device_state_snapshot_restores_velocity_and_pressure(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros((8, 8, 8, 3), dtype=np.float32)
        velocity[:, :, :, 0] = 0.03
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :] = 12.5
        force = np.ones((8, 8, 8, 3), dtype=np.float32)
        constraint_weight = np.ones((8, 8, 8), dtype=np.float32)
        solver.velocity.from_numpy(velocity)
        solver.pressure.from_numpy(pressure)
        solver.save_state()

        solver.set_uniform_velocity((1.0, 2.0, 3.0))
        solver.pressure.from_numpy(np.full_like(pressure, 3.25))
        solver.snapshot_pressure()
        solver.pressure.from_numpy(np.zeros_like(pressure))
        solver.force.from_numpy(force)
        solver.velocity_constraint_weight.from_numpy(constraint_weight)
        solver.restore_state()

        np.testing.assert_allclose(solver.velocity.to_numpy(), velocity, atol=1.0e-7)
        np.testing.assert_allclose(solver.pressure.to_numpy(), pressure, atol=1.0e-7)
        np.testing.assert_allclose(solver.force.to_numpy(), np.zeros_like(force), atol=1.0e-7)
        np.testing.assert_allclose(
            solver.velocity_constraint_weight.to_numpy(),
            np.zeros_like(constraint_weight),
            atol=1.0e-7,
        )

    def test_restore_state_clears_transient_velocity_dirichlet_projection_weight(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        active = np.zeros((4, 4, 4), dtype=np.int32)
        value = np.zeros((4, 4, 4, 3), dtype=np.float32)
        projection_weight = np.zeros((4, 4, 4), dtype=np.float32)
        active[1, 1, 1] = 1
        value[1, 1, 1, :] = (0.1, -0.2, 0.3)
        projection_weight[1, 1, 1] = 0.75

        solver.save_state()
        solver.velocity_dirichlet_boundary_active.from_numpy(active)
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(value)
        solver.velocity_dirichlet_boundary_projection_weight.from_numpy(projection_weight)
        solver.restore_state()

        np.testing.assert_array_equal(
            solver.velocity_dirichlet_boundary_active.to_numpy(),
            np.zeros_like(active),
        )
        np.testing.assert_allclose(
            solver.velocity_dirichlet_boundary_value_mps.to_numpy(),
            np.zeros_like(value),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            solver.velocity_dirichlet_boundary_projection_weight.to_numpy(),
            np.zeros_like(projection_weight),
            atol=1.0e-7,
        )

    def test_fv_cg_negative_curvature_sets_device_breakdown_code(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.cg_rz[None] = 1.0
        solver.cg_rz_new[None] = 1.0
        solver.cg_dAd[None] = 0.0

        solver._cg_compute_alpha_kernel()
        self.assertEqual(int(solver.cg_breakdown_code[None]), 1)
        self.assertEqual(float(solver.cg_alpha[None]), 0.0)

        solver._cg_compute_beta_kernel(0)
        self.assertEqual(int(solver.cg_breakdown_code[None]), 1)

    def test_fv_cg_project_report_exposes_device_breakdown_code(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = solver.project(pressure_solver="fv_cg", iterations=4)

        self.assertIn("cg_breakdown_code", report)
        self.assertIn("cg_breakdown_dAd", report)
        self.assertIn("cg_restart_count", report)
        self.assertIn("cg_restart_count_measured", report)
        self.assertIn("cg_restart_policy", report)
        self.assertEqual(report["cg_breakdown_code"], 0)
        self.assertEqual(report["cg_breakdown_dAd"], 0.0)
        self.assertEqual(report["cg_restart_count"], 0)
        self.assertTrue(report["cg_restart_count_measured"])
        self.assertEqual(report["cg_restart_policy"], "periodic_exact_residual")

    def test_closed_domain_non_cg_projection_reports_unmeasured_nullspace_policy(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = solver.project(
            pressure_solver="fv_multigrid",
            pressure_outlet_zmin=False,
            iterations=2,
        )

        self.assertEqual(report["pressure_nullspace_policy"], "closed_neumann_non_cg_unmeasured")
        self.assertFalse(report["pressure_nullspace_compatibility_measured"])
        self.assertFalse(report["pressure_nullspace_zero_mean_projection_applied"])
        self.assertFalse(report["pressure_system_anchored_by_interface_matrix"])

    def test_closed_domain_fv_cg_interface_matrix_reports_anchored_pressure_system(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.pressure_interface_matrix_diagonal[1, 1, 1] = 5.0

        report = solver.project(
            pressure_solver="fv_cg",
            pressure_outlet_zmin=False,
            iterations=4,
        )

        self.assertEqual(report["pressure_nullspace_policy"], "interface_matrix_anchored")
        self.assertTrue(report["pressure_system_anchored_by_interface_matrix"])
        self.assertFalse(report["pressure_nullspace_zero_mean_projection_applied"])
        self.assertTrue(report["pressure_nullspace_compatibility_measured"])

    def test_pressure_outlet_cleanup_uses_homogeneous_interface_rhs(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.pressure_interface_matrix_rhs[1, 1, 1] = 7.0
        zero_stats = {"l2": 0.0, "max_abs": 0.0}
        residual_stats = iter(
            (
                {"l2": 1.0, "max_abs": 1.0},
                {"l2": 1.0, "max_abs": 1.0},
                {"l2": 2.0, "max_abs": 2.0},
                {"l2": 0.0, "max_abs": 0.0},
            )
        )
        solve_rhs_values: list[float] = []

        def fake_solve(**_kwargs) -> None:
            solve_rhs_values.append(float(solver.pressure_interface_matrix_rhs[1, 1, 1]))

        def fake_divergence_stats(*, interior_only: bool = False) -> dict[str, float]:
            return dict(zero_stats)

        def fake_divergence_residual_stats() -> dict[str, float]:
            return dict(next(residual_stats, zero_stats))

        solver._solve_pressure_poisson_with_solver = fake_solve
        solver.compute_divergence = lambda **_kwargs: None
        solver.divergence_stats = fake_divergence_stats
        solver.divergence_residual_stats = fake_divergence_residual_stats
        solver.final_divergence_report_stats = lambda: (
            dict(zero_stats),
            dict(zero_stats),
            dict(zero_stats),
            dict(zero_stats),
        )
        solver.divergence_dirichlet_partition_report_stats = lambda: (
            dict(zero_stats),
            dict(zero_stats),
            dict(zero_stats),
            dict(zero_stats),
        )

        solver.project(
            pressure_solver="fv_cg",
            pressure_outlet_zmin=True,
            iterations=8,
            read_report=True,
        )

        self.assertGreaterEqual(len(solve_rhs_values), 2)
        self.assertEqual(solve_rhs_values[0], 7.0)
        self.assertEqual(solve_rhs_values[1], 0.0)

    def test_fv_cg_pressure_failure_report_policy_keeps_diagnostic_path(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        def fake_solve(**_kwargs) -> None:
            solver.last_cg_iterations = 4
            solver.last_cg_host_residual_checks = 2
            solver.last_cg_mean_host_reads = 0
            solver.last_cg_restart_count = 0
            solver.last_cg_initial_relative_residual = 1.0
            solver.last_cg_relative_residual = 0.25
            solver.last_cg_converged = False
            solver.last_cg_breakdown = "forced nonconvergence"
            solver.last_cg_breakdown_dAd = 0.0
            solver.cg_breakdown_code[None] = 3

        solver._solve_pressure_poisson_with_solver = fake_solve

        report = solver.project(
            iterations=4,
            pressure_solver="fv_cg",
            pressure_solve_failure_policy="report",
        )

        self.assertFalse(report["cg_converged_all"])
        self.assertEqual(report["cg_breakdown_count"], 1)
        self.assertEqual(report["pressure_solve_failure_policy"], "report")
        self.assertTrue(report["pressure_solve_failed"])
        self.assertEqual(
            report["pressure_solve_failure_action"],
            "reported_cleared_pressure_correction",
        )

    def test_fv_cg_pressure_failure_raise_aborts_before_velocity_correction(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        gradient_calls: list[tuple[float, int]] = []

        def fake_solve(**_kwargs) -> None:
            solver.last_cg_iterations = 4
            solver.last_cg_host_residual_checks = 2
            solver.last_cg_mean_host_reads = 0
            solver.last_cg_restart_count = 0
            solver.last_cg_initial_relative_residual = 1.0
            solver.last_cg_relative_residual = 0.25
            solver.last_cg_converged = False
            solver.last_cg_breakdown = "forced nonconvergence"
            solver.last_cg_breakdown_dAd = 0.0
            solver.cg_breakdown_code[None] = 3

        def fail_if_gradient_applied(self, dt_over_rho: float, pressure_outlet_zmin: int) -> None:
            gradient_calls.append((float(dt_over_rho), int(pressure_outlet_zmin)))
            raise AssertionError("pressure gradient was applied after a failed FV-CG solve")

        solver._solve_pressure_poisson_with_solver = fake_solve
        solver._subtract_pressure_gradient_kernel = MethodType(fail_if_gradient_applied, solver)

        with self.assertRaisesRegex(RuntimeError, "FV-CG pressure solve did not converge"):
            solver.project(
                iterations=4,
                pressure_solver="fv_cg",
                pressure_solve_failure_policy="raise",
            )

        self.assertEqual(gradient_calls, [])
        self.assertFalse(solver.last_project_cg_converged_all)
        self.assertEqual(solver.last_project_cg_breakdown_count, 1)

    def test_cg_alpha_preserves_first_nonpositive_dAd_on_device(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.cg_breakdown_code[None] = 0
        solver.cg_dAd[None] = -3.5

        solver._cg_compute_alpha_kernel()
        solver.cg_dAd[None] = 5.0
        solver._cg_compute_alpha_kernel()

        self.assertEqual(int(solver.cg_breakdown_code[None]), 1)
        self.assertAlmostEqual(float(solver.cg_breakdown_dAd[None]), -3.5)

    def test_fv_cg_residual_convergence_wins_over_latched_device_breakdown(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        residual_checks = 0

        def no_op(self, *_args, **_kwargs) -> None:
            return None

        def fake_weighted_dot(self, *_args, **_kwargs) -> float:
            return 1.0

        def fake_weighted_dot_to_field(self, _a, _b, out) -> None:
            nonlocal residual_checks
            if out is self.cg_rr:
                residual_checks += 1
                self.cg_rr[None] = 1.0 if residual_checks == 1 else 1.0e-16
            elif out is self.cg_rz:
                self.cg_rz[None] = 1.0
            elif out is self.cg_dAd:
                self.cg_dAd[None] = 0.0
            elif out is self.cg_rz_new:
                self.cg_rz_new[None] = 1.0
            elif out is self.cg_beta_numerator:
                self.cg_beta_numerator[None] = 1.0

        for name in (
            "_prepare_fv_multigrid_rhs",
            "_fv_diagonal_kernel",
            "_cg_build_positive_rhs_kernel",
            "_fv_laplacian_apply_kernel",
            "_axpby_scalar_field_kernel",
            "_apply_jacobi_preconditioner_kernel",
            "_copy_scalar_field_kernel",
            "_cg_apply_alpha_kernel",
        ):
            setattr(solver, name, MethodType(no_op, solver))
        solver._weighted_dot_kernel = MethodType(fake_weighted_dot, solver)
        solver._weighted_dot_to_field_kernel = MethodType(fake_weighted_dot_to_field, solver)

        solver._solve_pressure_poisson_fv_cg(
            iterations=1,
            rhs_scale=1.0,
            pressure_outlet_zmin=True,
            tolerance=1.0e-6,
            preconditioner="jacobi",
        )

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertEqual(solver.last_cg_breakdown, "")
        self.assertLessEqual(solver.last_cg_relative_residual, 1.0e-6)

    def test_fv_cg_restarts_exact_residual_after_device_breakdown(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        residual_checks = 0
        d_ad_calls = 0

        def no_op(self, *_args, **_kwargs) -> None:
            return None

        def fake_weighted_dot(self, *_args, **_kwargs) -> float:
            return 1.0

        def fake_weighted_dot_to_field(self, _a, _b, out) -> None:
            nonlocal residual_checks, d_ad_calls
            if out is self.cg_rr:
                residual_checks += 1
                if residual_checks <= 3:
                    self.cg_rr[None] = 0.25
                else:
                    self.cg_rr[None] = 1.0e-16
            elif out is self.cg_rz:
                self.cg_rz[None] = 1.0
            elif out is self.cg_dAd:
                d_ad_calls += 1
                self.cg_dAd[None] = 0.0 if d_ad_calls == 1 else 1.0
            elif out is self.cg_rz_new:
                self.cg_rz_new[None] = 1.0
            elif out is self.cg_beta_numerator:
                self.cg_beta_numerator[None] = 1.0

        for name in (
            "_prepare_fv_multigrid_rhs",
            "_fv_diagonal_kernel",
            "_cg_build_positive_rhs_kernel",
            "_fv_laplacian_apply_kernel",
            "_axpby_scalar_field_kernel",
            "_apply_jacobi_preconditioner_kernel",
            "_copy_scalar_field_kernel",
            "_cg_apply_alpha_kernel",
        ):
            setattr(solver, name, MethodType(no_op, solver))
        solver._weighted_dot_kernel = MethodType(fake_weighted_dot, solver)
        solver._weighted_dot_to_field_kernel = MethodType(fake_weighted_dot_to_field, solver)

        solver._solve_pressure_poisson_fv_cg(
            iterations=32,
            rhs_scale=1.0,
            pressure_outlet_zmin=True,
            tolerance=1.0e-6,
            preconditioner="jacobi",
        )

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertEqual(solver.last_cg_breakdown, "")
        self.assertEqual(solver.last_cg_restart_count, 1)
        self.assertEqual(int(solver.cg_breakdown_code[None]), 0)

    def test_fv_cg_interface_matrix_does_not_enter_bicgstab_fallback_after_pcg_budget(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fallback_calls: list[dict[str, object]] = []

        def no_op(self, *_args, **_kwargs) -> None:
            return None

        def fake_weighted_dot(self, *_args, **_kwargs) -> float:
            return 1.0

        def fake_weighted_dot_to_field(self, _a, _b, out) -> None:
            if out is self.cg_rr:
                self.cg_rr[None] = 1.0
            elif out is self.cg_rz:
                self.cg_rz[None] = 1.0
            elif out is self.cg_dAd:
                self.cg_dAd[None] = 1.0
            elif out is self.cg_rz_new:
                self.cg_rz_new[None] = 1.0
            elif out is self.cg_beta_numerator:
                self.cg_beta_numerator[None] = 1.0

        def fake_fallback(self, **kwargs) -> None:
            fallback_calls.append(dict(kwargs))
            self.last_cg_iterations = int(kwargs["start_iteration"]) + 1
            self.last_cg_relative_residual = 0.0
            self.last_cg_converged = True
            self.last_cg_breakdown = ""

        for name in (
            "_prepare_fv_multigrid_rhs",
            "_fv_diagonal_kernel",
            "_cg_build_positive_rhs_kernel",
            "_fv_laplacian_apply_kernel",
            "_axpby_scalar_field_kernel",
            "_apply_jacobi_preconditioner_kernel",
            "_copy_scalar_field_kernel",
            "_cg_apply_alpha_kernel",
            "_cg_update_direction_and_rz_kernel",
        ):
            setattr(solver, name, MethodType(no_op, solver))
        solver._weighted_dot_kernel = MethodType(fake_weighted_dot, solver)
        solver._weighted_dot_to_field_kernel = MethodType(fake_weighted_dot_to_field, solver)
        solver._continue_pressure_poisson_fv_bicgstab = MethodType(fake_fallback, solver)

        solver._solve_pressure_poisson_fv_cg(
            iterations=1,
            rhs_scale=1.0,
            pressure_outlet_zmin=True,
            tolerance=1.0e-6,
            preconditioner="jacobi",
            pressure_interface_matrix_active=True,
        )

        self.assertEqual(fallback_calls, [])
        self.assertFalse(solver.last_cg_converged)
        self.assertEqual(solver.last_cg_iterations, 1)

    def test_fv_cg_interface_rowlist_can_enter_bicgstab_fallback_after_pcg_budget(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.pressure_interface_row_count[None] = 1
        fallback_calls: list[dict[str, object]] = []

        def no_op(self, *_args, **_kwargs) -> None:
            return None

        def fake_weighted_dot(self, *_args, **_kwargs) -> float:
            return 1.0

        def fake_weighted_dot_to_field(self, _a, _b, out) -> None:
            if out is self.cg_rr:
                self.cg_rr[None] = 1.0
            elif out is self.cg_rz:
                self.cg_rz[None] = 1.0
            elif out is self.cg_dAd:
                self.cg_dAd[None] = 1.0
            elif out is self.cg_rz_new:
                self.cg_rz_new[None] = 1.0
            elif out is self.cg_beta_numerator:
                self.cg_beta_numerator[None] = 1.0

        def fake_fallback(self, **kwargs) -> None:
            fallback_calls.append(dict(kwargs))
            self.last_cg_iterations = int(kwargs["start_iteration"]) + 1
            self.last_cg_relative_residual = 0.0
            self.last_cg_converged = True
            self.last_cg_breakdown = ""

        for name in (
            "_prepare_fv_multigrid_rhs",
            "_fv_diagonal_kernel",
            "_cg_build_positive_rhs_kernel",
            "_fv_laplacian_apply_kernel",
            "_axpby_scalar_field_kernel",
            "_apply_jacobi_preconditioner_kernel",
            "_copy_scalar_field_kernel",
            "_cg_apply_alpha_kernel",
            "_cg_update_direction_and_rz_kernel",
        ):
            setattr(solver, name, MethodType(no_op, solver))
        solver._weighted_dot_kernel = MethodType(fake_weighted_dot, solver)
        solver._weighted_dot_to_field_kernel = MethodType(fake_weighted_dot_to_field, solver)
        solver._continue_pressure_poisson_fv_bicgstab = MethodType(fake_fallback, solver)

        solver._solve_pressure_poisson_fv_cg(
            iterations=1,
            rhs_scale=1.0,
            pressure_outlet_zmin=True,
            tolerance=1.0e-6,
            preconditioner="jacobi",
            pressure_interface_matrix_active=True,
        )

        self.assertEqual(len(fallback_calls), 1)
        self.assertTrue(solver.last_cg_converged)
        self.assertEqual(solver.last_cg_iterations, 2)

    def test_bicgstab_fallback_solves_generic_fv_rhs_on_device(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        rhs = np.zeros((4, 4, 4), dtype=np.float64)
        rhs[2, 2, 2] = 1.0
        solver.cg_rhs.from_numpy(rhs)
        solver._fv_diagonal_kernel(solver.fv_diag, 1)
        b_norm = float(np.sqrt(max(solver._weighted_dot_kernel(solver.cg_rhs, solver.cg_rhs), 0.0)))

        solver._continue_pressure_poisson_fv_bicgstab(
            start_iteration=0,
            max_iters=80,
            b_norm=b_norm,
            outlet=1,
            tolerance=1.0e-8,
            anchor_unreached=False,
            remove_nullspace_mean=False,
        )

        self.assertTrue(solver.last_cg_converged, solver.last_cg_breakdown)
        self.assertLessEqual(solver.last_cg_relative_residual, 1.0e-8)
        self.assertIn("bicgstab_fallback", solver.last_cg_restart_policy)

    def test_predict_viscosity_damps_shear_mode(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(32, 32, 32),
                density_kgm3=1.0,
                viscosity_pa_s=5.0e-2,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros((32, 32, 32, 3), dtype=np.float32)
        y = (np.arange(32, dtype=np.float32) + 0.5) / 32.0
        velocity[:, :, :, 0] = np.sin(6.0 * np.pi * y)[None, :, None]
        solver.velocity.from_numpy(velocity)

        before_l2 = float(np.linalg.norm(velocity[2:-2, 2:-2, 2:-2, 0]))
        solver.predict()
        after = solver.velocity.to_numpy()
        after_l2 = float(np.linalg.norm(after[2:-2, 2:-2, 2:-2, 0]))

        self.assertLess(after_l2, before_l2 * 0.99)
        self.assertGreater(after_l2, before_l2 * 0.9)

    def test_nonuniform_predict_viscosity_damps_shear_mode(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.04,) * 8,
            cell_widths_y_m=(0.020, 0.024, 0.029, 0.035, 0.042, 0.050, 0.060, 0.072, 0.086, 0.102, 0.122, 0.145),
            cell_widths_z_m=(0.05,) * 6,
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=5.0e-2,
                dt_s=2.0e-4,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        y = np.asarray(grid.cell_centers_y_m, dtype=np.float32)
        y_min = float(grid.bounds_min_m[1])
        y_extent = float(grid.bounds_max_m[1] - grid.bounds_min_m[1])
        mode = np.sin(3.0 * np.pi * (y - y_min) / y_extent).astype(np.float32)
        velocity = np.zeros(grid.grid_nodes + (3,), dtype=np.float32)
        velocity[:, :, :, 0] = mode[None, :, None]
        solver.velocity.from_numpy(velocity)

        before_l2 = float(np.linalg.norm(velocity[:, 2:-2, :, 0]))
        solver.predict()
        after = solver.velocity.to_numpy()
        after_l2 = float(np.linalg.norm(after[:, 2:-2, :, 0]))

        self.assertLess(after_l2, before_l2)
        self.assertGreater(after_l2, before_l2 * 0.9)

    def test_predict_rk2_backtrace_reduces_rotational_advection_error(self) -> None:
        grid_nodes = (32, 32, 8)
        dt_s = 0.12
        omega = 1.5
        x = (np.arange(grid_nodes[0], dtype=np.float32) + 0.5) / grid_nodes[0]
        y = (np.arange(grid_nodes[1], dtype=np.float32) + 0.5) / grid_nodes[1]
        xx, yy = np.meshgrid(x, y, indexing="ij")
        velocity_slice = np.zeros((grid_nodes[0], grid_nodes[1], 3), dtype=np.float32)
        velocity_slice[:, :, 0] = -omega * (yy - 0.5)
        velocity_slice[:, :, 1] = omega * (xx - 0.5)
        velocity = np.repeat(velocity_slice[:, :, None, :], grid_nodes[2], axis=2)

        angle = -omega * dt_s
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        dx = xx - 0.5
        dy = yy - 0.5
        back_x = 0.5 + cos_angle * dx - sin_angle * dy
        back_y = 0.5 + sin_angle * dx + cos_angle * dy
        exact_slice = np.zeros_like(velocity_slice)
        exact_slice[:, :, 0] = -omega * (back_y - 0.5)
        exact_slice[:, :, 1] = omega * (back_x - 0.5)
        exact = np.repeat(exact_slice[:, :, None, :], grid_nodes[2], axis=2)
        interior = (
            slice(6, -6),
            slice(6, -6),
            slice(1, -1),
            slice(None),
        )

        def run_scheme(advection_scheme: str) -> float:
            solver = CartesianFluidSolver(
                FluidDomainSpec.unit_box(
                    grid_nodes=grid_nodes,
                    density_kgm3=1.0,
                    viscosity_pa_s=0.0,
                    dt_s=dt_s,
                ),
                runtime=TaichiRuntimeConfig(arch="cuda"),
            )
            solver.velocity.from_numpy(velocity)
            solver.predict(advection_scheme=advection_scheme)
            predicted = solver.velocity.to_numpy()
            error = predicted[interior] - exact[interior]
            reference = exact[interior]
            return float(np.linalg.norm(error) / np.linalg.norm(reference))

        euler_error = run_scheme("euler")
        rk2_error = run_scheme("rk2")

        self.assertLess(rk2_error, euler_error * 0.55)

    def test_taylor_green_predict_project_error_decreases_on_finer_grid(self) -> None:
        def run_one_step(grid_nodes: tuple[int, int, int]) -> float:
            nx, ny, nz = grid_nodes
            amplitude = 0.05
            viscosity_m2_s = 1.0e-2
            dt_s = 5.0e-4
            x = (np.arange(nx, dtype=np.float32) + 0.5) / nx
            y = (np.arange(ny, dtype=np.float32) + 0.5) / ny
            xx, yy = np.meshgrid(x, y, indexing="ij")
            exact_slice = np.zeros((nx, ny, 3), dtype=np.float32)
            exact_slice[:, :, 0] = amplitude * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
            exact_slice[:, :, 1] = -amplitude * np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
            velocity = np.repeat(exact_slice[:, :, None, :], nz, axis=2)
            decay = np.exp(-8.0 * np.pi * np.pi * viscosity_m2_s * dt_s)
            exact_after = velocity * np.float32(decay)

            solver = CartesianFluidSolver(
                FluidDomainSpec.unit_box(
                    grid_nodes=grid_nodes,
                    density_kgm3=1.0,
                    viscosity_pa_s=viscosity_m2_s,
                    dt_s=dt_s,
                ),
                runtime=TaichiRuntimeConfig(arch="cuda"),
            )
            solver.velocity.from_numpy(velocity)
            solver.predict()
            predicted = solver.project(iterations=12 * nx)
            after = solver.velocity.to_numpy()
            interior = (
                slice(nx // 4, 3 * nx // 4),
                slice(ny // 4, 3 * ny // 4),
                slice(1, nz - 1),
                slice(None),
            )
            error = after[interior] - exact_after[interior]
            reference = exact_after[interior]
            relative_error = float(np.linalg.norm(error) / np.linalg.norm(reference))
            self.assertLess(predicted["projection_l2"], 1.0e-2)
            return relative_error

        coarse_error = run_one_step((24, 24, 8))
        fine_error = run_one_step((48, 48, 8))

        self.assertLess(fine_error, coarse_error * 0.85)

    def test_taylor_green_predict_project_keeps_smooth_flow_bounded(self) -> None:
        grid_nodes = (32, 32, 16)
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=grid_nodes,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-2,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        nx, ny, nz = grid_nodes
        x = (np.arange(nx, dtype=np.float32) + 0.5) / nx
        y = (np.arange(ny, dtype=np.float32) + 0.5) / ny
        xx, yy = np.meshgrid(x, y, indexing="ij")
        velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        amplitude = 0.15
        velocity[:, :, :, 0] = (
            amplitude * np.sin(2.0 * np.pi * xx)[:, :, None] * np.cos(2.0 * np.pi * yy)[:, :, None]
        )
        velocity[:, :, :, 1] = (
            -amplitude * np.cos(2.0 * np.pi * xx)[:, :, None] * np.sin(2.0 * np.pi * yy)[:, :, None]
        )
        solver.velocity.from_numpy(velocity)
        initial_speed_l2 = float(np.linalg.norm(velocity[2:-2, 2:-2, 2:-2]))

        solver.predict()
        solver.compute_divergence()
        predicted = solver.divergence_stats()
        projected = solver.project(iterations=360)
        after = solver.velocity.to_numpy()
        after_speed_l2 = float(np.linalg.norm(after[2:-2, 2:-2, 2:-2]))

        self.assertGreater(predicted["l2"], 1.0e-4)
        self.assertLess(projected["projection_l2"], predicted["l2"] * 0.25)
        self.assertLess(after_speed_l2, initial_speed_l2 * 1.02)
        self.assertGreater(after_speed_l2, initial_speed_l2 * 0.85)

    def test_apply_body_force_ignores_obstacle_impulse_in_report(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[3, 3, 3] = 1
        force = np.zeros((8, 8, 8, 3), dtype=np.float32)
        force[3, 3, 3, 0] = 1000.0
        solver.obstacle.from_numpy(obstacle)
        solver.force.from_numpy(force)

        report = solver.apply_body_force()

        self.assertEqual(report.active_velocity_cells, 0)
        np.testing.assert_allclose(report.grid_impulse_n_s, (0.0, 0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(report.momentum_delta_n_s, (0.0, 0.0, 0.0), atol=1.0e-12)
        self.assertAlmostEqual(report.impulse_relative_error, 0.0, delta=1.0e-12)

    def test_apply_body_force_can_skip_report_without_skipping_kernel(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        force = np.zeros((8, 8, 8, 3), dtype=np.float32)
        force[2, 2, 2, 0] = 1000.0
        solver.force.from_numpy(force)

        report = solver.apply_body_force(dt_s=2.0e-3, read_report=False)
        velocity = solver.velocity.to_numpy()

        self.assertIsNone(report)
        self.assertAlmostEqual(float(velocity[2, 2, 2, 0]), 2.0e-3, delta=1.0e-9)
        self.assertAlmostEqual(float(velocity[2, 2, 2, 1]), 0.0, delta=1.0e-12)

    def test_body_force_impulse_report_can_be_read_after_skipped_report_without_reapplying(self) -> None:
        force = np.zeros((8, 8, 8, 3), dtype=np.float32)
        force[2, 2, 2, 0] = 1000.0
        force[3, 2, 2, 1] = -500.0

        reference = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        reference.force.from_numpy(force)
        expected_report = reference.apply_body_force(dt_s=2.0e-3, read_report=True)
        expected_velocity = reference.velocity.to_numpy()

        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.force.from_numpy(force)
        skipped_report = solver.apply_body_force(dt_s=2.0e-3, read_report=False)
        velocity_after_apply = solver.velocity.to_numpy()

        self.assertIsNone(skipped_report)
        report = solver.body_force_impulse_report()
        velocity_after_report = solver.velocity.to_numpy()

        np.testing.assert_allclose(velocity_after_apply, expected_velocity, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(velocity_after_report, expected_velocity, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(report.grid_impulse_n_s, expected_report.grid_impulse_n_s, atol=1.0e-12)
        np.testing.assert_allclose(
            report.momentum_delta_n_s,
            expected_report.momentum_delta_n_s,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            report.impulse_relative_error,
            expected_report.impulse_relative_error,
            delta=1.0e-12,
        )
        self.assertEqual(report.active_velocity_cells, expected_report.active_velocity_cells)

    def test_released_hibm_internal_cell_reconstructs_velocity_from_fluid_neighbors(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
        for index in (
            (1, 2, 2),
            (3, 2, 2),
            (2, 1, 2),
            (2, 3, 2),
            (2, 2, 1),
            (2, 2, 3),
        ):
            velocity[index] = (0.25, -0.5, 0.75)
        solver.velocity.from_numpy(velocity)
        solver.velocity_prev.from_numpy(velocity)
        node_kind = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        internal = np.zeros((5, 5, 5), dtype=np.int32)
        internal[2, 2, 2] = 7
        node_kind.from_numpy(internal)

        self.assertEqual(solver.apply_hibm_internal_obstacles(node_kind, internal_node_code=7), 1)
        internal[2, 2, 2] = 0
        node_kind.from_numpy(internal)
        self.assertEqual(solver.apply_hibm_internal_obstacles(node_kind, internal_node_code=7), 0)

        np.testing.assert_allclose(
            solver.velocity.to_numpy()[2, 2, 2],
            (0.25, -0.5, 0.75),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            solver.velocity_prev.to_numpy()[2, 2, 2],
            (0.25, -0.5, 0.75),
            atol=1.0e-7,
        )
        self.assertEqual(int(solver.report_hibm_fresh_fluid_cells[None]), 1)

    def test_hibm_internal_obstacle_conversion_can_be_disabled_for_thin_interfaces(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node_kind = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        internal = np.zeros((5, 5, 5), dtype=np.int32)
        internal[2, 2, 2] = 7
        node_kind.from_numpy(internal)

        converted = solver.apply_hibm_internal_obstacles(
            node_kind,
            internal_node_code=7,
            convert_internal_nodes=False,
        )

        self.assertEqual(converted, 0)
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 0)
        self.assertEqual(int(solver.report_hibm_internal_obstacle_cells[None]), 0)

    def test_released_hibm_internal_cell_uses_dirichlet_velocity_without_fluid_neighbors(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base_obstacle = np.zeros((5, 5, 5), dtype=np.int32)
        for index in (
            (1, 2, 2),
            (3, 2, 2),
            (2, 1, 2),
            (2, 3, 2),
            (2, 2, 1),
            (2, 2, 3),
        ):
            base_obstacle[index] = 1
        solver.obstacle.from_numpy(base_obstacle)
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_value_mps[2, 2, 2] = (0.1, 0.2, -0.3)
        node_kind = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        internal = np.zeros((5, 5, 5), dtype=np.int32)
        internal[2, 2, 2] = 7
        node_kind.from_numpy(internal)

        self.assertEqual(solver.apply_hibm_internal_obstacles(node_kind, internal_node_code=7), 1)
        internal[2, 2, 2] = 0
        node_kind.from_numpy(internal)
        self.assertEqual(solver.apply_hibm_internal_obstacles(node_kind, internal_node_code=7), 0)

        np.testing.assert_allclose(
            solver.velocity.to_numpy()[2, 2, 2],
            (0.1, 0.2, -0.3),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            solver.velocity_prev.to_numpy()[2, 2, 2],
            (0.1, 0.2, -0.3),
            atol=1.0e-7,
        )
        self.assertEqual(int(solver.report_hibm_fresh_fluid_cells[None]), 1)

    def test_apply_velocity_constraints_reports_momentum_delta(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        target_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        target_weight = np.zeros((4, 4, 4), dtype=np.float32)
        target_sum[1, 1, 1] = (0.25, -0.1, 0.05)
        target_weight[1, 1, 1] = 1.0
        solver.velocity_constraint_sum.from_numpy(target_sum)
        solver.velocity_constraint_weight.from_numpy(target_weight)

        report = solver.apply_velocity_constraints(blend=1.0, read_report=True)
        cell_volume_m3 = solver.dx * solver.dy * solver.dz
        expected_momentum = tuple(
            solver.rho * cell_volume_m3 * component
            for component in (0.25, -0.1, 0.05)
        )

        np.testing.assert_allclose(
            report.momentum_delta_n_s,
            expected_momentum,
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_velocity_dirichlet_boundary_rows_apply_without_legacy_constraint(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[1, 1, 1] = 1
        solver.velocity_dirichlet_boundary_value_mps[1, 1, 1] = (0.25, -0.1, 0.05)

        report = solver.apply_velocity_dirichlet_boundary_rows(read_report=True)
        cell_mass_kg = solver.rho * solver.dx * solver.dy * solver.dz
        expected_momentum = tuple(
            cell_mass_kg * component for component in (0.25, -0.1, 0.05)
        )

        self.assertEqual(report.active_cells, 1)
        np.testing.assert_allclose(
            report.momentum_delta_n_s,
            expected_momentum,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        velocity = tuple(float(solver.velocity[1, 1, 1][axis]) for axis in range(3))
        np.testing.assert_allclose(velocity, (0.25, -0.1, 0.05), atol=1.0e-7)
        self.assertEqual(solver.velocity_constraint_weight[1, 1, 1], 0.0)

    def test_projection_consumes_velocity_dirichlet_boundary_rows(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[1, 1, 1] = 1
        solver.velocity_dirichlet_boundary_value_mps[1, 1, 1] = (0.1, 0.2, -0.3)

        report = solver.project(
            iterations=2,
            pressure_solver="fv_jacobi",
            preserve_velocity_constraints=False,
            reset_pressure=True,
            read_report=True,
        )

        velocity = tuple(float(solver.velocity[1, 1, 1][axis]) for axis in range(3))
        np.testing.assert_allclose(velocity, (0.1, 0.2, -0.3), atol=1.0e-7)
        self.assertEqual(solver.velocity_constraint_weight[1, 1, 1], 0.0)
        self.assertGreaterEqual(report["velocity_dirichlet_boundary_apply_calls"], 1)
        self.assertGreater(report["velocity_dirichlet_boundary_active_cells_total"], 0)
        self.assertGreater(report["velocity_dirichlet_boundary_active_cells_max"], 0)
        self.assertGreater(report["velocity_dirichlet_boundary_max_delta_mps"], 0.0)
        self.assertGreater(report["velocity_dirichlet_boundary_mean_delta_mps"], 0.0)

    def test_apply_velocity_constraints_reports_region_momentum_deltas(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        primary_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        secondary_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        primary_weight = np.zeros((4, 4, 4), dtype=np.float32)
        secondary_weight = np.zeros((4, 4, 4), dtype=np.float32)
        primary_sum[1, 1, 1] = (1.0, 0.0, 0.0)
        secondary_sum[1, 1, 1] = (0.0, 3.0, 0.0)
        primary_weight[1, 1, 1] = 1.0
        secondary_weight[1, 1, 1] = 1.0
        solver.velocity_constraint_primary_sum.from_numpy(primary_sum)
        solver.velocity_constraint_secondary_sum.from_numpy(secondary_sum)
        solver.velocity_constraint_primary_weight.from_numpy(primary_weight)
        solver.velocity_constraint_secondary_weight.from_numpy(secondary_weight)
        solver.velocity_constraint_sum.from_numpy(primary_sum + secondary_sum)
        solver.velocity_constraint_weight.from_numpy(primary_weight + secondary_weight)

        report = solver.apply_velocity_constraints(blend=1.0, read_report=True)
        cell_mass_kg = solver.rho * solver.dx * solver.dy * solver.dz

        np.testing.assert_allclose(
            report.primary_momentum_delta_n_s,
            (0.5 * cell_mass_kg, 0.0, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            report.secondary_momentum_delta_n_s,
            (0.0, 1.5 * cell_mass_kg, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(report.primary_momentum_delta_n_s)
            + np.asarray(report.secondary_momentum_delta_n_s),
            report.momentum_delta_n_s,
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_velocity_constraint_impulse_accumulator_spans_report_skipped_applications(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        primary_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        secondary_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        primary_weight = np.zeros((4, 4, 4), dtype=np.float32)
        secondary_weight = np.zeros((4, 4, 4), dtype=np.float32)
        primary_sum[1, 1, 1] = (1.0, 0.0, 0.0)
        secondary_sum[2, 1, 1] = (0.0, 3.0, 0.0)
        primary_weight[1, 1, 1] = 1.0
        secondary_weight[2, 1, 1] = 1.0
        solver.velocity_constraint_primary_sum.from_numpy(primary_sum)
        solver.velocity_constraint_secondary_sum.from_numpy(secondary_sum)
        solver.velocity_constraint_primary_weight.from_numpy(primary_weight)
        solver.velocity_constraint_secondary_weight.from_numpy(secondary_weight)
        solver.velocity_constraint_sum.from_numpy(primary_sum + secondary_sum)
        solver.velocity_constraint_weight.from_numpy(primary_weight + secondary_weight)
        solver.reset_velocity_constraint_impulse_accumulator()

        skipped_report = solver.apply_velocity_constraints(blend=0.5, read_report=False)
        final_report = solver.apply_velocity_constraints(blend=1.0, read_report=True)
        primary_impulse, secondary_impulse = solver.velocity_constraint_impulse_report()
        cell_mass_kg = solver.rho * solver.dx * solver.dy * solver.dz

        self.assertIsNone(skipped_report)
        np.testing.assert_allclose(
            final_report.primary_momentum_delta_n_s,
            (0.5 * cell_mass_kg, 0.0, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            final_report.secondary_momentum_delta_n_s,
            (0.0, 1.5 * cell_mass_kg, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            primary_impulse,
            (1.0 * cell_mass_kg, 0.0, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            secondary_impulse,
            (0.0, 3.0 * cell_mass_kg, 0.0),
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_apply_velocity_constraints_can_skip_report_without_skipping_kernel(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        target_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        target_weight = np.zeros((4, 4, 4), dtype=np.float32)
        target_sum[1, 1, 1] = (0.25, -0.1, 0.05)
        target_weight[1, 1, 1] = 1.0
        solver.velocity_constraint_sum.from_numpy(target_sum)
        solver.velocity_constraint_weight.from_numpy(target_weight)

        report = solver.apply_velocity_constraints(blend=1.0, read_report=False)
        velocity = solver.velocity.to_numpy()

        self.assertIsNone(report)
        np.testing.assert_allclose(
            velocity[1, 1, 1],
            (0.25, -0.1, 0.05),
            rtol=1.0e-7,
            atol=1.0e-7,
        )

    def test_velocity_constraint_solid_mobility_ratio_reduces_correction_impulse(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        target_sum = np.zeros((4, 4, 4, 3), dtype=np.float32)
        target_weight = np.zeros((4, 4, 4), dtype=np.float32)
        target_sum[1, 1, 1] = (4.0, -2.0, 1.0)
        target_weight[1, 1, 1] = 1.0
        solver.velocity_constraint_sum.from_numpy(target_sum)
        solver.velocity_constraint_weight.from_numpy(target_weight)

        report = solver.apply_velocity_constraints(
            blend=1.0,
            solid_mobility_ratio=3.0,
            read_report=True,
        )
        velocity = solver.velocity.to_numpy()
        cell_mass_kg = solver.rho * solver.dx * solver.dy * solver.dz
        expected_velocity = (1.0, -0.5, 0.25)

        np.testing.assert_allclose(
            velocity[1, 1, 1],
            expected_velocity,
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            report.momentum_delta_n_s,
            tuple(cell_mass_kg * value for value in expected_velocity),
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_velocity_constraint_rejects_invalid_solid_mobility_ratio(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "solid_mobility_ratio"):
                    solver.apply_velocity_constraints(
                        blend=1.0,
                        solid_mobility_ratio=value,
                        read_report=False,
                    )


class UnreachedSetInterfaceHitObservabilityTests(unittest.TestCase):
    """R2-H1: the flood-unreachable set used for nullspace anchoring can overlap
    rows that the pressure-interface matrix terms actually anchor/connect. The
    projection report must count those overlaps so the mean-subtraction policy
    can be audited before any numerical-semantics change."""

    def test_fv_cg_reports_unreached_set_interface_hit_diagnostics(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[6, 2, 6] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 3)
        self.assertEqual(solver.last_hibm_pressure_unreached_component_count, 2)
        # Pocket component A = {(2,2,4), (2,2,5)}; pocket component B = {(6,2,6)}.
        # Interface diagonal anchors one cell of component A directly.
        solver.pressure_interface_matrix_diagonal[2, 2, 4] = 4.0
        # Interface coupling edge owned by a reachable outlet-layer cell points
        # into pocket component B, so B is matrix-connected to the reachable
        # region even though the 6-neighbor flood cannot see the edge.
        cell_volume_m3 = float(solver.dx * solver.dy * solver.dz)
        transmissibility = 0.25
        solver.pressure_interface_coupling_active[3, 2, 0] = 1
        solver.pressure_interface_coupling_neighbor[3, 2, 0] = (6, 2, 6)
        solver.pressure_interface_coupling_coefficient[3, 2, 0] = transmissibility
        solver.pressure_interface_matrix_diagonal[3, 2, 0] = (
            transmissibility / cell_volume_m3
        )
        solver.pressure_interface_matrix_diagonal[6, 2, 6] = (
            transmissibility / cell_volume_m3
        )

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        # (2,2,4) and (6,2,6) carry interface diagonal terms while unreached.
        self.assertEqual(int(report["unreached_cells_with_interface_diagonal"]), 2)
        # (6,2,6) is the target row of an active interface coupling edge.
        self.assertEqual(int(report["unreached_cells_with_interface_coupling"]), 1)
        # Both pocket components are hit; each is counted exactly once.
        self.assertEqual(int(report["unreached_components_with_interface_hits"]), 2)
        self.assertEqual(
            int(solver.last_hibm_unreached_cells_with_interface_diagonal), 2
        )
        self.assertEqual(
            int(solver.last_hibm_unreached_cells_with_interface_coupling), 1
        )
        self.assertEqual(
            int(solver.last_hibm_unreached_components_with_interface_hits), 2
        )

    def test_fv_cg_reports_zero_interface_hits_for_purely_disconnected_pockets(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(9, 4, 9), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((9, 4, 9), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 2)

        report = solver.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertEqual(int(report["unreached_cells_with_interface_diagonal"]), 0)
        self.assertEqual(int(report["unreached_cells_with_interface_coupling"]), 0)
        self.assertEqual(int(report["unreached_components_with_interface_hits"]), 0)

    def test_projection_report_exposes_band_and_label_budget_flags(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[0, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[1, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 3, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 2, 3] = 1
        masked = solver.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=False,
        )
        self.assertEqual(masked, 1)
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 1)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 0)
        self.assertTrue(solver.last_hibm_pressure_reachability_converged)
        self.assertTrue(solver.last_hibm_pressure_component_labels_converged)

        report = solver.project(
            iterations=60,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertEqual(int(report["hibm_solid_band_last_marked_increment"]), 1)
        self.assertTrue(bool(report["hibm_pressure_component_labels_converged"]))
        self.assertTrue(bool(report["hibm_pressure_reachability_converged"]))

    def test_restore_state_resets_reachability_and_interface_hit_flags(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.save_state()
        solver.last_hibm_pressure_unreached_component_overflow = True
        solver.last_hibm_pressure_reachability_converged = False
        solver.last_hibm_pressure_reachability_sweeps = 7
        solver.last_hibm_pressure_reachability_reused = True
        solver.last_hibm_pressure_component_labels_converged = False
        solver.last_hibm_unreached_cells_with_interface_diagonal = 3
        solver.last_hibm_unreached_cells_with_interface_coupling = 2
        solver.last_hibm_unreached_components_with_interface_hits = 1
        solver.last_hibm_solid_band_marked_increment = 4

        solver.restore_state()

        self.assertFalse(solver.last_hibm_pressure_unreached_component_overflow)
        self.assertTrue(solver.last_hibm_pressure_reachability_converged)
        self.assertEqual(int(solver.last_hibm_pressure_reachability_sweeps), 0)
        self.assertFalse(solver.last_hibm_pressure_reachability_reused)
        self.assertTrue(solver.last_hibm_pressure_component_labels_converged)
        self.assertEqual(
            int(solver.last_hibm_unreached_cells_with_interface_diagonal), 0
        )
        self.assertEqual(
            int(solver.last_hibm_unreached_cells_with_interface_coupling), 0
        )
        self.assertEqual(
            int(solver.last_hibm_unreached_components_with_interface_hits), 0
        )
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 0)

class DivergenceFinalReportStagingShapeTests(unittest.TestCase):
    def test_staging_arrays_cover_all_eighteen_report_slots(self) -> None:
        # The final divergence report kernel statically iterates 18 slots
        # (16 partition slots + 2 anchored-unreached slots added by the
        # guard-split round). Staging arrays declared narrower than the
        # static slot range are written and read out of bounds on every
        # reported projection - undefined behavior in release kernels that
        # pollutes the max/count lanes the long-run divergence guards read.
        spec = FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3)
        solver = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))

        self.assertEqual(tuple(solver.divergence_combined_sum.shape), (18,))
        self.assertEqual(tuple(solver.divergence_combined_max.shape), (18,))
        self.assertEqual(tuple(solver.divergence_combined_count.shape), (18,))


class HibmSolidBandPopulationSplitTests(unittest.TestCase):
    """S2-A8': split the solid band by population.

    The legacy band kernel converts every zero-correctable active cell to
    an obstacle, which confuses two populations: membrane-interior sliver
    cells (quasi-solid cells the IB node search classified near the marker
    surface) and real enclosed water (cells with no marker within the
    search radius, sealed off only by the velocity-Dirichlet row cloud).
    The split keeps the enclosed water active so the per-component
    anchoring chain solves its pressure, while slivers still convert.

    Mode table (environment gates read per call, A8 pattern):

    - default (both gates unset): convert every candidate, bitwise-
      unchanged legacy band; the population mirrors stay -1 (unmeasured).
    - ``HIBM_BAND_INTERIOR_ONLY=1``: convert classified slivers only;
      requires the node classification field; the returned increment only
      covers conversions so the caller's fixed-round loop still saturates.
    - ``HIBM_BAND_COUNT_ONLY=1``: convert nothing (A8 diagnostic, wins
      over the interior-only gate); with a classification field the two
      populations are still counted.
    """

    # Cell-shaped IB node classification codes (HibmMpmIbNodeSearch
    # publishes _NODE_NONE=0 for "no marker within the search radius";
    # any other code means the search classified the cell near the
    # marker surface).
    _NODE_NONE = 0
    _NODE_CLASSIFIED = 1

    @staticmethod
    def _solver_with_two_band_candidates():
        # (2, 2, 2): zero-correctable candidate sealed by velocity-
        # Dirichlet rows on all six stencil faces (same construction as
        # the legacy band test). (0, 0, 0): zero-correctable candidate
        # sealed by the domain boundary on the minus faces and Dirichlet
        # rows on the plus faces. No other cell loses every correctable
        # face.
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[3, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[2, 3, 2] = 1
        solver.velocity_dirichlet_boundary_active[2, 2, 3] = 1
        solver.velocity[2, 2, 2] = (1.0, -2.0, 3.0)
        solver.velocity_prev[2, 2, 2] = (4.0, -5.0, 6.0)
        solver.volume_source_s[2, 2, 2] = 7.0
        solver.velocity_dirichlet_boundary_active[1, 0, 0] = 1
        solver.velocity_dirichlet_boundary_active[0, 1, 0] = 1
        solver.velocity_dirichlet_boundary_active[0, 0, 1] = 1
        solver.velocity[0, 0, 0] = (1.0, -2.0, 3.0)
        solver.velocity_prev[0, 0, 0] = (4.0, -5.0, 6.0)
        solver.volume_source_s[0, 0, 0] = 7.0
        node_kind_code = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        # The sliver candidate sits inside the search-classified band
        # around the marker surface; the enclosed-water candidate keeps
        # the unclassified code (no marker within the search radius).
        node_kind_code[2, 2, 2] = HibmSolidBandPopulationSplitTests._NODE_CLASSIFIED
        return solver, node_kind_code

    def test_interior_only_band_split_converts_slivers_and_keeps_enclosed_water(
        self,
    ) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        with mock.patch.dict(os.environ, {"HIBM_BAND_INTERIOR_ONLY": "1"}):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
            )

            self.assertEqual(marked, 1)
            self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 1)
            self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 1)
            self.assertEqual(
                int(solver.last_hibm_solid_band_enclosed_water_cells), 1
            )
            # The membrane-interior sliver converts exactly like the
            # legacy band cell.
            self.assertEqual(int(solver.obstacle[2, 2, 2]), 1)
            self.assertEqual(
                tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3)),
                (0.0, 0.0, 0.0),
            )
            self.assertEqual(
                tuple(
                    float(solver.velocity_prev[2, 2, 2][axis]) for axis in range(3)
                ),
                (0.0, 0.0, 0.0),
            )
            self.assertAlmostEqual(float(solver.volume_source_s[2, 2, 2]), 0.0)
            # The enclosed-water cell stays active fluid with its state
            # intact, left for the per-component anchoring chain.
            self.assertEqual(int(solver.obstacle[0, 0, 0]), 0)
            self.assertEqual(
                tuple(float(solver.velocity[0, 0, 0][axis]) for axis in range(3)),
                (1.0, -2.0, 3.0),
            )
            self.assertEqual(
                tuple(
                    float(solver.velocity_prev[0, 0, 0][axis]) for axis in range(3)
                ),
                (4.0, -5.0, 6.0),
            )
            self.assertAlmostEqual(float(solver.volume_source_s[0, 0, 0]), 7.0)

            # Second sweep: conversions are monotone and the sliver set
            # saturates, so the increment the caller's fixed-round band
            # loop breaks on reaches zero while the enclosed-water
            # population stays observable.
            marked_again = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
            )
            self.assertEqual(marked_again, 0)
            self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 0)
            self.assertEqual(
                int(solver.last_hibm_solid_band_enclosed_water_cells), 1
            )
            self.assertEqual(int(solver.obstacle[0, 0, 0]), 0)

    def test_interior_only_band_split_requires_node_classification_field(
        self,
    ) -> None:
        import os
        from unittest import mock

        solver, _ = self._solver_with_two_band_candidates()
        with mock.patch.dict(os.environ, {"HIBM_BAND_INTERIOR_ONLY": "1"}):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            with self.assertRaises(ValueError):
                solver.mark_hibm_solid_band_nonprojectable_cells(
                    pressure_outlet_zmin=False,
                )

    def test_interior_only_band_split_rejects_mismatched_classification_shape(
        self,
    ) -> None:
        import os
        from unittest import mock

        solver, _ = self._solver_with_two_band_candidates()
        wrong_shape = ti.field(dtype=ti.i32, shape=(4, 4, 4))
        with mock.patch.dict(os.environ, {"HIBM_BAND_INTERIOR_ONLY": "1"}):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            with self.assertRaises(ValueError):
                solver.mark_hibm_solid_band_nonprojectable_cells(
                    pressure_outlet_zmin=False,
                    node_kind_code=wrong_shape,
                    unclassified_node_code=self._NODE_NONE,
                )

    def test_count_only_with_classification_reports_population_split(self) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        with mock.patch.dict(os.environ, {"HIBM_BAND_COUNT_ONLY": "1"}):
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
            )

        # A8 semantics preserved: every candidate is counted, nothing
        # converts - and the two populations are now observable.
        self.assertEqual(marked, 2)
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 2)
        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 1)
        self.assertEqual(int(solver.last_hibm_solid_band_enclosed_water_cells), 1)
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 0)
        self.assertEqual(int(solver.obstacle[0, 0, 0]), 0)
        self.assertEqual(
            tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3)),
            (1.0, -2.0, 3.0),
        )
        self.assertAlmostEqual(float(solver.volume_source_s[2, 2, 2]), 7.0)

    def test_default_band_mode_uses_classification_and_keeps_enclosed_water(
        self,
    ) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        with mock.patch.dict(os.environ):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
            )

        # Default sharp diagnostics use the classification field: membrane
        # interior slivers convert, but enclosed water stays active and is
        # counted instead of disappearing behind the -1 sentinel.
        self.assertEqual(marked, 1)
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 1)
        self.assertEqual(int(solver.obstacle[0, 0, 0]), 0)
        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 1)
        self.assertEqual(int(solver.last_hibm_solid_band_enclosed_water_cells), 1)

    def test_band_split_can_protect_classified_no_slip_row_neighborhood(
        self,
    ) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        solver.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        solver.velocity_dirichlet_boundary_marker_region_id[2, 2, 2] = 7
        with mock.patch.dict(os.environ):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
                protect_velocity_dirichlet_radius_cells=0,
                protect_velocity_dirichlet_marker_region_id=7,
            )

        self.assertEqual(marked, 0)
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 0)
        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 0)
        self.assertEqual(int(solver.last_hibm_solid_band_enclosed_water_cells), 1)
        self.assertEqual(
            int(solver.last_hibm_solid_band_velocity_dirichlet_protected_cells),
            1,
        )
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 0)
        self.assertEqual(
            tuple(float(solver.velocity[2, 2, 2][axis]) for axis in range(3)),
            (1.0, -2.0, 3.0),
        )
        self.assertAlmostEqual(float(solver.volume_source_s[2, 2, 2]), 7.0)

    def test_band_split_protection_respects_marker_region_filter(self) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        solver.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        solver.velocity_dirichlet_boundary_marker_region_id[2, 2, 2] = 7
        with mock.patch.dict(os.environ):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
                protect_velocity_dirichlet_radius_cells=0,
                protect_velocity_dirichlet_marker_region_id=8,
            )

        self.assertEqual(marked, 1)
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 1)
        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 1)
        self.assertEqual(
            int(solver.last_hibm_solid_band_velocity_dirichlet_protected_cells),
            0,
        )
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 1)

    def test_band_split_can_protect_source_probe_clear_mask_cells(self) -> None:
        import os
        from unittest import mock

        solver, node_kind_code = self._solver_with_two_band_candidates()
        protection_mask = np.zeros(tuple(solver.obstacle.shape), dtype=np.int32)
        protection_mask[2, 2, 2] = 1
        solver.set_hibm_solid_band_protection_mask_from_numpy(protection_mask)
        with mock.patch.dict(os.environ):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            marked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
                node_kind_code=node_kind_code,
                unclassified_node_code=self._NODE_NONE,
                protect_solid_band_mask=True,
            )

        self.assertEqual(marked, 0)
        self.assertEqual(int(solver.last_hibm_solid_band_marked_increment), 0)
        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), 0)
        self.assertEqual(int(solver.last_hibm_solid_band_mask_protected_cells), 1)
        self.assertEqual(int(solver.obstacle[2, 2, 2]), 0)

    def test_no_slip_sampling_obstacle_exposes_dirichlet_rows_only(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.obstacle[1, 1, 1] = 1
        solver.obstacle[2, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[2, 2, 2] = 1

        sampling_obstacle = solver.build_hibm_no_slip_sampling_obstacle()

        self.assertEqual(int(sampling_obstacle[1, 1, 1]), 1)
        self.assertEqual(int(sampling_obstacle[2, 2, 2]), 0)

    def test_projection_report_exposes_band_population_split_mirrors(self) -> None:
        import os
        from unittest import mock

        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.velocity_dirichlet_boundary_active[0, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[1, 2, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 3, 2] = 1
        solver.velocity_dirichlet_boundary_active[0, 2, 3] = 1
        with mock.patch.dict(os.environ):
            os.environ.pop("HIBM_BAND_COUNT_ONLY", None)
            os.environ.pop("HIBM_BAND_INTERIOR_ONLY", None)
            masked = solver.mark_hibm_solid_band_nonprojectable_cells(
                pressure_outlet_zmin=False,
            )
        self.assertEqual(masked, 1)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 0)

        report = solver.project(
            iterations=60,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertEqual(int(report["hibm_solid_band_last_marked_increment"]), 1)
        # The unclassified legacy sweep reports the populations as
        # unmeasured rather than as misleading zeros.
        self.assertEqual(int(report["hibm_solid_band_interior_cells"]), -1)
        self.assertEqual(int(report["hibm_solid_band_enclosed_water_cells"]), -1)
        self.assertEqual(
            int(report["hibm_solid_band_velocity_dirichlet_protected_cells"]),
            -1,
        )
        self.assertEqual(int(report["hibm_solid_band_mask_protected_cells"]), -1)

    def test_restore_state_resets_band_population_split_mirrors(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.save_state()
        solver.last_hibm_solid_band_interior_cells = 5
        solver.last_hibm_solid_band_enclosed_water_cells = 6
        solver.last_hibm_solid_band_velocity_dirichlet_protected_cells = 7
        solver.last_hibm_solid_band_mask_protected_cells = 8

        solver.restore_state()

        self.assertEqual(int(solver.last_hibm_solid_band_interior_cells), -1)
        self.assertEqual(int(solver.last_hibm_solid_band_enclosed_water_cells), -1)
        self.assertEqual(
            int(solver.last_hibm_solid_band_velocity_dirichlet_protected_cells),
            -1,
        )
        self.assertEqual(int(solver.last_hibm_solid_band_mask_protected_cells), -1)


class HibmConvertedCellPressureFillTests(unittest.TestCase):
    """S2-A8'' red tests: post-projection pressure fill + sampling view.

    The A4->A8' experiment chain established that the band's full
    conversion is the CORRECT projection behavior (zero-correctable cells
    are zero matrix rows; A8' interior-only measured a CG residual floor
    of 0.518), and that the closure starvation instead comes from two
    sampling-side defects: (a) the stress sampler shares the projection's
    obstacle view, so sealed water that was converted under the membrane
    is invisible to it, and (b) the ``pressure`` value of converted cells
    is stale (they do not participate in the solve).

    Fix under test (fluid side):

    - ``fill_hibm_converted_cell_pressures(sweeps=8)``: Jacobi-style
      (read-old / write-new, expand+commit double buffer) iterative
      6-neighbor averaging over "hibm-converted non-base" cells
      (``obstacle != 0 and hibm_base_obstacle == 0``). Per sweep, every
      such cell with at least one AVAILABLE neighbor recomputes
      ``pressure = mean(available neighbor pressures)`` and is marked in
      the i32 flag field ``hibm_pressure_filled``; available = a
      6-neighbor that is solved water (``obstacle == 0``) OR a converted
      cell already marked in a PREVIOUS sweep. A cell with no available
      neighbor keeps its value and stays unmarked, so the fill front
      advances exactly one cell layer per sweep from the solved water
      into the sealed interior.
    - ``last_hibm_pressure_filled_cell_count`` host mirror: -1 means
      "fill never ran since construction / restore_state"; project()
      report key ``hibm_pressure_filled_cell_count`` is a pure mirror
      passthrough (the fill is NOT mounted inside project() - the HIBM
      assemble calls it explicitly after the projection returns, so
      within one assemble step the projection report shows the previous
      step's fill count).
    - ``build_hibm_sampling_obstacle(node_kind_code,
      unclassified_node_code)``: dedicated sampling view
      ``sampling_obstacle`` with the S2-A8'' truth table
      ``view = 1 iff hibm_base_obstacle != 0 OR node_kind_code !=
      unclassified_node_code`` - base geometry and the classified row
      cloud envelope stay dry for sampling (the A8 lesson: opening the
      envelope made every marker sample zero-pressure dead water on both
      sides, killing the drive), while NONE-classified converted sealed
      water becomes samplable with its back-filled pressure.
    """

    @staticmethod
    def _solver_with_converted_chain():
        # 8^3 unit box. 1D chain along x at (y, z) = (4, 4):
        #   W  = (1,4,4): solved water (obstacle 0), pressure 2.0
        #   C1 = (2,4,4), C2 = (3,4,4), C3 = (4,4,4): originally water in
        #        the base snapshot, then hibm-converted (obstacle 1,
        #        base 0) with stale pressures 50 / 60 / 70
        #   everything else: base obstacle (dry in the base snapshot),
        #        so the chain's only available contact is W at the C1 end
        #        and the (5,4,4) end is a base-obstacle wall.
        # Plus one isolated converted cell I = (6,6,6) (base 0, converted)
        # whose 6 neighbors are all base obstacles: it must never fill.
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base = np.ones((8, 8, 8), dtype=np.int32)
        base[1, 4, 4] = 0
        base[2, 4, 4] = 0
        base[3, 4, 4] = 0
        base[4, 4, 4] = 0
        base[6, 6, 6] = 0
        solver.obstacle.from_numpy(base)
        solver.snapshot_hibm_base_obstacle()
        # hibm conversion: the chain and the isolated cell become
        # obstacles while the base snapshot still remembers them as water.
        solver.obstacle[2, 4, 4] = 1
        solver.obstacle[3, 4, 4] = 1
        solver.obstacle[4, 4, 4] = 1
        solver.obstacle[6, 6, 6] = 1
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[1, 4, 4] = 2.0
        pressure[2, 4, 4] = 50.0
        pressure[3, 4, 4] = 60.0
        pressure[4, 4, 4] = 70.0
        pressure[6, 6, 6] = 9.0
        solver.pressure.from_numpy(pressure)
        return solver

    def test_pressure_fill_diffuses_one_layer_per_sweep_into_converted_chain(
        self,
    ) -> None:
        solver = self._solver_with_converted_chain()

        # Sweep-by-sweep hand calculation (Jacobi: availability and
        # neighbor pressures are read from the PREVIOUS sweep's state):
        #
        # fill(sweeps=1), from stale (C1, C2, C3) = (50, 60, 70):
        #   sweep 1: C1 sees solved water W (filled flags all old=0, so
        #            C2 is not available) -> C1 = mean(2.0) = 2.0, marked;
        #            C2 sees no available neighbor (C1 and C3 both
        #            converted and unmarked in the previous state) ->
        #            keeps 60.0, unmarked; C3 likewise keeps 70.0.
        filled = solver.fill_hibm_converted_cell_pressures(sweeps=1)
        self.assertEqual(filled, 1)
        self.assertEqual(int(solver.last_hibm_pressure_filled_cell_count), 1)
        self.assertAlmostEqual(float(solver.pressure[2, 4, 4]), 2.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[3, 4, 4]), 60.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[4, 4, 4]), 70.0, delta=1.0e-6)
        self.assertEqual(int(solver.hibm_pressure_filled[2, 4, 4]), 1)
        self.assertEqual(int(solver.hibm_pressure_filled[3, 4, 4]), 0)
        self.assertEqual(int(solver.hibm_pressure_filled[4, 4, 4]), 0)

        # fill(sweeps=2) re-initializes the flags, then from
        # (C1, C2, C3) = (2, 60, 70):
        #   sweep 1: C1 = mean(W=2.0) = 2.0, marked; C2/C3 unmarked.
        #   sweep 2: C1 = mean(W=2.0) = 2.0 (C2 still unmarked in the old
        #            state); C2 = mean(C1_old=2.0) = 2.0, marked
        #            (C1 was marked in sweep 1); C3 still has no
        #            available neighbor -> keeps 70.0.
        filled = solver.fill_hibm_converted_cell_pressures(sweeps=2)
        self.assertEqual(filled, 2)
        self.assertEqual(int(solver.last_hibm_pressure_filled_cell_count), 2)
        self.assertAlmostEqual(float(solver.pressure[2, 4, 4]), 2.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[3, 4, 4]), 2.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[4, 4, 4]), 70.0, delta=1.0e-6)
        self.assertEqual(int(solver.hibm_pressure_filled[4, 4, 4]), 0)

        # fill(sweeps=8) (the production default), from (2, 2, 70):
        #   sweep 1: C1 = mean(W) = 2.0 marked; C2/C3 hold (flags were
        #            re-initialized, so C2's neighbors are unmarked).
        #   sweep 2: C2 = mean(C1_old=2.0) = 2.0 marked.
        #   sweep 3: C3 = mean(C2_old=2.0) = 2.0 marked; C2 now averages
        #            its marked neighbor C1 (2.0) -> 2.0.
        #   sweeps 4-8: stationary - every chain cell averages marked /
        #            water neighbors that all read 2.0.
        filled = solver.fill_hibm_converted_cell_pressures(sweeps=8)
        self.assertEqual(filled, 3)
        self.assertEqual(int(solver.last_hibm_pressure_filled_cell_count), 3)
        self.assertAlmostEqual(float(solver.pressure[2, 4, 4]), 2.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[3, 4, 4]), 2.0, delta=1.0e-6)
        self.assertAlmostEqual(float(solver.pressure[4, 4, 4]), 2.0, delta=1.0e-6)
        self.assertEqual(int(solver.hibm_pressure_filled[2, 4, 4]), 1)
        self.assertEqual(int(solver.hibm_pressure_filled[3, 4, 4]), 1)
        self.assertEqual(int(solver.hibm_pressure_filled[4, 4, 4]), 1)

        # The water-untouched isolated converted cell holds its stale
        # value and stays unmarked through every call above.
        self.assertAlmostEqual(float(solver.pressure[6, 6, 6]), 9.0, delta=1.0e-6)
        self.assertEqual(int(solver.hibm_pressure_filled[6, 6, 6]), 0)
        # Solved water and base obstacles are copy-through: the fill only
        # rewrites converted cells.
        self.assertAlmostEqual(float(solver.pressure[1, 4, 4]), 2.0, delta=1.0e-6)
        self.assertEqual(int(solver.hibm_pressure_filled[1, 4, 4]), 0)
        self.assertAlmostEqual(float(solver.pressure[0, 0, 0]), 0.0, delta=1.0e-6)

    def test_pressure_fill_rejects_non_positive_sweeps(self) -> None:
        solver = self._solver_with_converted_chain()
        with self.assertRaises(ValueError):
            solver.fill_hibm_converted_cell_pressures(sweeps=0)

    def test_pressure_fill_is_not_mounted_in_project_and_mirrors_report_key(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        # -1 = "fill never ran"; a plain projection must keep it there
        # (the fill is explicitly NOT mounted inside project(); the HIBM
        # assemble calls it after the projection returns).
        self.assertEqual(int(solver.last_hibm_pressure_filled_cell_count), -1)
        report = solver.project(iterations=4)
        self.assertIn("hibm_pressure_filled_cell_count", report)
        self.assertEqual(int(report["hibm_pressure_filled_cell_count"]), -1)

        # The report key is a pure host-mirror passthrough: within one
        # HIBM assemble step the projection report therefore shows the
        # PREVIOUS step's fill count (the fill runs post-projection).
        solver.last_hibm_pressure_filled_cell_count = 7
        report = solver.project(iterations=4)
        self.assertEqual(int(report["hibm_pressure_filled_cell_count"]), 7)

        solver.save_state()
        solver.restore_state()
        self.assertEqual(int(solver.last_hibm_pressure_filled_cell_count), -1)

    def test_build_hibm_sampling_obstacle_truth_table(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base = np.zeros((5, 5, 5), dtype=np.int32)
        base[0, 0, 0] = 1
        solver.obstacle.from_numpy(base)
        solver.snapshot_hibm_base_obstacle()
        # (1,1,1): hibm-converted sealed water (obstacle 1, base 0,
        # unclassified) - the population the fill makes samplable.
        solver.obstacle[1, 1, 1] = 1
        node_kind_code = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        # (2,2,2): row-cloud envelope - ACTIVE fluid the IB node search
        # classified near the marker surface. (0,0,0): base obstacle that
        # also got classified (both conditions hold).
        node_kind_code[2, 2, 2] = 1
        node_kind_code[0, 0, 0] = 2

        solver.build_hibm_sampling_obstacle(
            node_kind_code,
            unclassified_node_code=0,
        )

        # Truth table (view = 1 iff base != 0 OR classified):
        #   base obstacle (+ classified)        -> dry
        self.assertEqual(int(solver.sampling_obstacle[0, 0, 0]), 1)
        #   classified envelope (active fluid)  -> dry (A8 lesson)
        self.assertEqual(int(solver.sampling_obstacle[2, 2, 2]), 1)
        #   converted sealed water (NONE class) -> water (samplable)
        self.assertEqual(int(solver.sampling_obstacle[1, 1, 1]), 0)
        #   free solved water                   -> water
        self.assertEqual(int(solver.sampling_obstacle[3, 3, 3]), 0)

    def test_build_hibm_sampling_obstacle_rejects_mismatched_shape(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        wrong_shape = ti.field(dtype=ti.i32, shape=(4, 4, 4))
        with self.assertRaises(ValueError):
            solver.build_hibm_sampling_obstacle(
                wrong_shape,
                unclassified_node_code=0,
            )


if __name__ == "__main__":
    unittest.main()
