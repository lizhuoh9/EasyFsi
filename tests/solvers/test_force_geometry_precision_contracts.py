from __future__ import annotations

import inspect
import re
import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


def _solver_for_grid(
    grid: CartesianGrid,
    *,
    density_kgm3: float = 1000.0,
    viscosity_pa_s: float = 1.0e-3,
) -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=density_kgm3,
            viscosity_pa_s=viscosity_pa_s,
            dt_s=5.0e-4,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _device_width(value: float) -> np.float64:
    """Mirror the solver's f32 geometry ledger, then promote for arithmetic."""

    return np.float64(np.float32(value))


class ForceGeometryStaticPrecisionContracts(unittest.TestCase):
    """Integrated volume, area, and impulse ledgers must multiply in f64."""

    def test_authoritative_face_area_helpers_cast_each_width_before_product(
        self,
    ) -> None:
        helper_widths = {
            "_x_face_area_f64_m2": (
                "self.cell_width_y_m[j]",
                "self.cell_width_z_m[k]",
            ),
            "_y_face_area_f64_m2": (
                "self.cell_width_x_m[i]",
                "self.cell_width_z_m[k]",
            ),
            "_z_face_area_f64_m2": (
                "self.cell_width_x_m[i]",
                "self.cell_width_y_m[j]",
            ),
        }
        for helper_name, widths in helper_widths.items():
            with self.subTest(helper=helper_name):
                helper = getattr(CartesianFluidSolver, helper_name, None)
                self.assertIsNotNone(
                    helper,
                    f"{helper_name} must be the shared f64 face-area authority",
                )
                if helper is None:
                    continue
                source = " ".join(inspect.getsource(helper).split())
                for width in widths:
                    self.assertIn(f"ti.cast({width}, ti.f64)", source)
                self.assertNotIn(
                    f"ti.cast({widths[0]} * {widths[1]}, ti.f64)",
                    source,
                )

    def test_obstacle_volume_uses_authoritative_f64_cell_volume(self) -> None:
        source = inspect.getsource(CartesianFluidSolver._sum_obstacle_volume_kernel)
        self.assertIn("self._cell_volume_f64_m3(i, j, k)", source)
        self.assertNotIn("self._cell_volume_m3(i, j, k)", source)

    def test_obstacle_surface_force_kernels_use_face_area_authorities(self) -> None:
        for kernel_name in (
            "_compute_obstacle_surface_pressure_force_kernel",
            "_compute_obstacle_surface_viscous_force_kernel",
        ):
            with self.subTest(kernel=kernel_name):
                source = inspect.getsource(getattr(CartesianFluidSolver, kernel_name))
                self.assertIn("self._x_face_area_f64_m2", source)
                self.assertIn("self._y_face_area_f64_m2", source)
                self.assertIn("self._z_face_area_f64_m2", source)
                for direct_product in (
                    "self.cell_width_y_m[j] * self.cell_width_z_m[k]",
                    "self.cell_width_x_m[i] * self.cell_width_z_m[k]",
                    "self.cell_width_x_m[i] * self.cell_width_y_m[j]",
                ):
                    self.assertNotIn(direct_product, source)

    def test_primary_and_secondary_impulses_promote_before_multiplication(
        self,
    ) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._apply_velocity_constraints_kernel
        )
        primary_start = source.index("primary_momentum_delta =")
        secondary_start = source.index("secondary_momentum_delta =")
        accumulation_start = source.index(
            "self._atomic_add_report_vector(", secondary_start
        )
        primary_block = source[primary_start:secondary_start]
        secondary_block = source[secondary_start:accumulation_start]

        for name, block in (
            ("primary_velocity_delta", primary_block),
            ("secondary_velocity_delta", secondary_block),
        ):
            with self.subTest(momentum=name):
                self.assertIn("_cell_volume_f64_m3", block)
                self.assertNotIn("_cell_volume_m3", block)
                self.assertRegex(
                    block,
                    re.compile(rf"ti\.cast\(\s*{name}\s*,\s*ti\.f64\s*\)"),
                )
                self.assertRegex(
                    block,
                    re.compile(r"ti\.cast\(\s*self\.rho\s*,\s*ti\.f64\s*\)"),
                )

    def test_multigrid_restriction_has_no_floor_for_positive_free_volume(
        self,
    ) -> None:
        source = inspect.getsource(CartesianFluidSolver._mg_restrict_residual_kernel)
        average_start = source.index("if obstacle_count > 0:")
        average_end = source.index("# Legacy scalar metadata", average_start)
        average_block = source[average_start:average_end]
        self.assertIn("free_volume_m3", average_block)
        self.assertNotIn("ti.max", average_block)
        self.assertNotIn("1.0e-30", average_block)


class ForceGeometryNumericalPrecisionContracts(unittest.TestCase):
    @staticmethod
    def _underflow_area_grid(nodes: int = 4) -> CartesianGrid:
        return CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.2e-3,) * nodes,
            cell_widths_y_m=(1.0e-24,) * nodes,
            cell_widths_z_m=(1.3e-24,) * nodes,
        )

    def test_tiny_positive_obstacle_volume_survives_device_f32_geometry(self) -> None:
        grid = self._underflow_area_grid()
        solver = _solver_for_grid(grid)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        obstacle[1, 1, 1] = 1
        solver.obstacle.from_numpy(obstacle)

        expected_volume_m3 = (
            _device_width(grid.cell_widths_x_m[1])
            * _device_width(grid.cell_widths_y_m[1])
            * _device_width(grid.cell_widths_z_m[1])
        )
        measured_volume_m3 = solver.obstacle_volume_m3()

        self.assertGreater(expected_volume_m3, 0.0)
        self.assertAlmostEqual(
            measured_volume_m3,
            expected_volume_m3,
            delta=float(expected_volume_m3 * np.float64(1.0e-12)),
        )

    def test_pressure_surface_force_preserves_tiny_positive_face_area(self) -> None:
        grid = self._underflow_area_grid()
        solver = _solver_for_grid(grid)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        obstacle[1, 1, 1] = 1
        pressure = np.zeros(grid.grid_nodes, dtype=np.float32)
        pressure[0, 1, 1] = np.float32(1.0e30)
        solver.obstacle.from_numpy(obstacle)
        solver.pressure.from_numpy(pressure)

        expected_force_x_n = (
            np.float64(pressure[0, 1, 1])
            * _device_width(grid.cell_widths_y_m[1])
            * _device_width(grid.cell_widths_z_m[1])
        )
        force_n = solver.compute_obstacle_surface_pressure_force_n()

        self.assertGreater(expected_force_x_n, 0.0)
        self.assertAlmostEqual(
            force_n[0],
            expected_force_x_n,
            delta=float(expected_force_x_n * np.float64(1.0e-12)),
        )
        self.assertEqual(force_n[1], 0.0)
        self.assertEqual(force_n[2], 0.0)

    def test_graded_viscous_surface_force_uses_promoted_face_areas(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.0011, 0.0012, 0.0013, 0.0014, 0.0015),
            cell_widths_y_m=(0.00036, 0.00039, 0.00042, 0.00047, 0.00051),
            cell_widths_z_m=(0.00043, 0.00046, 0.00049, 0.00054, 0.00061),
        )
        mu_pa_s = 2.0e12
        speed_mps = np.float32(3.0e8)
        solver = _solver_for_grid(grid, viscosity_pa_s=mu_pa_s)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        obstacle[2, 2, 2] = 1
        velocity = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
        velocity[..., 1] = speed_mps
        solver.obstacle.from_numpy(obstacle)
        solver.velocity.from_numpy(velocity)

        dx1, dx2, dx3 = (
            _device_width(grid.cell_widths_x_m[index]) for index in (1, 2, 3)
        )
        dy1, dy2, dy3 = (
            _device_width(grid.cell_widths_y_m[index]) for index in (1, 2, 3)
        )
        dz1, dz2, dz3 = (
            _device_width(grid.cell_widths_z_m[index]) for index in (1, 2, 3)
        )
        area_yz = dy2 * dz2
        area_xz = dx2 * dz2
        area_xy = dx2 * dy2
        expected_force_y_n = (
            np.float64(2.0)
            * np.float64(mu_pa_s)
            * np.float64(speed_mps)
            * (
                area_yz * (np.float64(1.0) / dx1 + np.float64(1.0) / dx3)
                + area_xz * (np.float64(1.0) / dy1 + np.float64(1.0) / dy3)
                + area_xy * (np.float64(1.0) / dz1 + np.float64(1.0) / dz3)
            )
        )

        force_n = solver.compute_obstacle_surface_viscous_force_n()

        self.assertAlmostEqual(
            force_n[1],
            expected_force_y_n,
            delta=float(abs(expected_force_y_n) * np.float64(5.0e-13)),
        )
        self.assertAlmostEqual(force_n[0], 0.0, delta=1.0e-18)
        self.assertAlmostEqual(force_n[2], 0.0, delta=1.0e-18)

    def test_primary_secondary_impulses_survive_f32_product_overflow_and_volume_underflow(
        self,
    ) -> None:
        grid = self._underflow_area_grid()
        rho_kgm3 = 1000.0
        solver = _solver_for_grid(grid, density_kgm3=rho_kgm3)
        shape = grid.grid_nodes
        cell = (1, 1, 1)
        weight = np.zeros(shape, dtype=np.float32)
        primary_weight = np.zeros(shape, dtype=np.float32)
        secondary_weight = np.zeros(shape, dtype=np.float32)
        total_sum = np.zeros((*shape, 3), dtype=np.float32)
        primary_sum = np.zeros((*shape, 3), dtype=np.float32)
        secondary_sum = np.zeros((*shape, 3), dtype=np.float32)
        primary_sum[cell + (0,)] = np.float32(2.0e38)
        secondary_sum[cell + (1,)] = np.float32(2.0e38)
        total_sum[cell] = primary_sum[cell] + secondary_sum[cell]
        weight[cell] = np.float32(2.0)
        primary_weight[cell] = np.float32(1.0)
        secondary_weight[cell] = np.float32(1.0)
        solver.velocity_constraint_sum.from_numpy(total_sum)
        solver.velocity_constraint_weight.from_numpy(weight)
        solver.velocity_constraint_primary_sum.from_numpy(primary_sum)
        solver.velocity_constraint_primary_weight.from_numpy(primary_weight)
        solver.velocity_constraint_secondary_sum.from_numpy(secondary_sum)
        solver.velocity_constraint_secondary_weight.from_numpy(secondary_weight)

        solver.apply_velocity_constraints(read_report=False)

        volume_m3 = (
            _device_width(grid.cell_widths_x_m[cell[0]])
            * _device_width(grid.cell_widths_y_m[cell[1]])
            * _device_width(grid.cell_widths_z_m[cell[2]])
        )
        device_delta_mps = np.float64(
            np.float32(np.float32(2.0e38) / np.float32(2.0))
        )
        expected_impulse_n_s = np.float64(rho_kgm3) * device_delta_mps * volume_m3
        primary = solver.velocity_constraint_primary_impulse_n_s[None]
        secondary = solver.velocity_constraint_secondary_impulse_n_s[None]

        self.assertTrue(np.isfinite(float(primary.x)))
        self.assertTrue(np.isfinite(float(secondary.y)))
        self.assertAlmostEqual(
            float(primary.x),
            expected_impulse_n_s,
            delta=float(expected_impulse_n_s * np.float64(1.0e-12)),
        )
        self.assertAlmostEqual(
            float(secondary.y),
            expected_impulse_n_s,
            delta=float(expected_impulse_n_s * np.float64(1.0e-12)),
        )

    def test_multigrid_restriction_uses_true_tiny_positive_free_volume(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=tuple(1.0e-11 * (1.0 + 0.03 * i) for i in range(8)),
            cell_widths_y_m=tuple(1.8e-11 * (1.0 + 0.02 * i) for i in range(8)),
            cell_widths_z_m=tuple(2.4e-11 * (1.0 + 0.01 * i) for i in range(8)),
        )
        solver = _solver_for_grid(grid)
        fine_residual = np.zeros(grid.grid_nodes, dtype=np.float64)
        fine_diagonal = np.zeros(grid.grid_nodes, dtype=np.float64)
        child_values = np.arange(1.0, 9.0, dtype=np.float64).reshape((2, 2, 2))
        fine_residual[:2, :2, :2] = child_values
        fine_diagonal[:2, :2, :2] = 2.0 * child_values
        solver._mg_residual[0].from_numpy(fine_residual)
        solver._mg_pressure_interface_matrix_diagonal[0].from_numpy(fine_diagonal)

        solver._mg_restrict_residual_kernel(
            solver._mg_residual[0],
            solver._mg_pressure_interface_matrix_diagonal[0],
            solver._mg_obstacle[0],
            solver._mg_velocity_dirichlet_boundary_active[0],
            solver._mg_velocity_dirichlet_boundary_projection_weight[0],
            solver._mg_velocity_dirichlet_boundary_marker_region_id[0],
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
            solver._mg_velocity_dirichlet_boundary_marker_region_id[1],
            solver._mg_velocity_dirichlet_pressure_hard_fixed_component_mask[1],
            8,
            8,
            8,
            0,
        )

        wx = np.asarray(grid.cell_widths_x_m[:2], dtype=np.float32).astype(np.float64)
        wy = np.asarray(grid.cell_widths_y_m[:2], dtype=np.float32).astype(np.float64)
        wz = np.asarray(grid.cell_widths_z_m[:2], dtype=np.float32).astype(np.float64)
        child_volume_m3 = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        self.assertLess(float(child_volume_m3.sum()), 1.0e-30)
        expected_rhs = float(np.sum(child_values * child_volume_m3) / child_volume_m3.sum())
        expected_diagonal = 2.0 * expected_rhs
        coarse_rhs = solver._mg_rhs[1].to_numpy()
        coarse_diagonal = solver._mg_pressure_interface_matrix_diagonal[1].to_numpy()

        self.assertAlmostEqual(float(coarse_rhs[0, 0, 0]), expected_rhs, delta=1.0e-6)
        self.assertAlmostEqual(
            float(coarse_diagonal[0, 0, 0]),
            expected_diagonal,
            delta=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
