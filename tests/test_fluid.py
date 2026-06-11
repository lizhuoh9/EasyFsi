from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core.fluid import CartesianFluidSolver, FluidDomainSpec


class CartesianFluidSolverTests(unittest.TestCase):
    def test_obstacle_mask_approximates_sphere_volume(self) -> None:
        spec = FluidDomainSpec(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(24, 24, 24),
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=1.0e-3,
        )
        solver = CartesianFluidSolver(spec)
        solver.mark_sphere_obstacle(center_m=(0.5, 0.5, 0.5), radius_m=0.25)

        measured = solver.obstacle_volume_m3()
        expected = 4.0 / 3.0 * 3.141592653589793 * 0.25**3
        self.assertLess(abs(measured - expected) / expected, 0.25)

    def test_uniform_velocity_has_near_zero_divergence(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16)))
        solver.set_uniform_velocity((0.2, -0.1, 0.05))
        solver.compute_divergence()

        stats = solver.divergence_stats(interior_only=True)
        self.assertLess(stats["max_abs"], 1.0e-5)
        self.assertLess(stats["l2"], 1.0e-5)

    def test_simple_shear_velocity_has_near_zero_divergence(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(21, 21, 21)))
        solver.set_simple_shear_velocity(75.0, center_y_m=0.5)
        solver.compute_divergence()

        stats = solver.divergence_stats(interior_only=True)
        self.assertLess(stats["max_abs"], 1.0e-5)
        self.assertLess(stats["l2"], 1.0e-5)

    def test_pressure_projection_reduces_device_side_divergence(self) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(24, 24, 24), dt_s=2.0e-3)
        )
        solver.set_sinusoidal_divergent_velocity(0.15)
        solver.compute_divergence()
        before = solver.divergence_stats()

        after = solver.project(iterations=240)

        self.assertGreater(before["l2"], 0.1)
        self.assertLess(after["l2"], before["l2"] * 0.45)
        self.assertLess(after["max_abs"], before["max_abs"] * 0.55)

    def test_device_state_snapshot_restores_velocity_and_pressure(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8)))
        velocity = np.zeros((8, 8, 8, 3), dtype=np.float32)
        velocity[:, :, :, 1] = -0.02
        pressure = np.full((8, 8, 8), 7.5, dtype=np.float32)
        solver.velocity.from_numpy(velocity)
        solver.pressure.from_numpy(pressure)
        solver.save_state()

        solver.set_uniform_velocity((0.5, 0.5, 0.5))
        solver.pressure.from_numpy(np.zeros_like(pressure))
        solver.restore_state()

        np.testing.assert_allclose(solver.velocity.to_numpy(), velocity, atol=1.0e-7)
        np.testing.assert_allclose(solver.pressure.to_numpy(), pressure, atol=1.0e-7)

    def test_obstacle_no_slip_velocity_is_enforced(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(18, 18, 18)))
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.mark_sphere_obstacle(center_m=(0.5, 0.5, 0.5), radius_m=0.2)
        solver.apply_obstacle_velocity((0.0, 0.0, -0.12))

        error = solver.obstacle_velocity_error((0.0, 0.0, -0.12))
        self.assertLess(error["max_abs"], 1.0e-6)

    def test_surface_force_spreading_conserves_action_reaction_on_grid(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(32, 32, 32)))
        surface_position_m = ti.Vector.field(3, dtype=ti.f32, shape=4)
        surface_force_n = ti.Vector.field(3, dtype=ti.f32, shape=4)
        surface_position_m.from_numpy(
            np.array(
                [
                    [-0.02, 0.0, 0.0],
                    [0.02, 0.0, 0.0],
                    [0.0, -0.02, 0.0],
                    [0.0, 0.02, 0.0],
                ],
                dtype=np.float32,
            )
        )
        surface_force_n.from_numpy(
            np.array(
                [
                    [0.0, 0.0, -0.25],
                    [0.0, 0.0, -0.50],
                    [0.0, 0.0, -0.75],
                    [0.0, 0.0, -1.00],
                ],
                dtype=np.float32,
            )
        )

        spread_report = solver.spread_surface_forces(
            surface_position_m,
            surface_force_n,
            4,
            center_m=(0.5, 0.5, 0.5),
            force_sign=-1.0,
        )

        self.assertGreater(spread_report.active_grid_cells, 0)
        self.assertLess(spread_report.action_reaction_relative_error, 1.0e-5)
        self.assertAlmostEqual(spread_report.surface_force_n[2], -2.5, delta=1.0e-6)
        self.assertAlmostEqual(
            spread_report.grid_force_n[2],
            -spread_report.surface_force_n[2],
            delta=abs(spread_report.surface_force_n[2]) * 1.0e-5,
        )

        impulse_report = solver.apply_body_force(dt_s=2.5e-3)

        self.assertGreater(impulse_report.active_velocity_cells, 0)
        self.assertLess(impulse_report.impulse_relative_error, 1.0e-5)
        self.assertAlmostEqual(
            impulse_report.momentum_delta_n_s[2],
            spread_report.grid_force_n[2] * 2.5e-3,
            delta=abs(spread_report.grid_force_n[2] * 2.5e-3) * 1.0e-5,
        )

    def test_surface_force_spreading_renormalizes_around_obstacles(self) -> None:
        solver = CartesianFluidSolver(FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8)))
        centers_x = solver.cell_center_x_m.to_numpy()
        centers_y = solver.cell_center_y_m.to_numpy()
        centers_z = solver.cell_center_z_m.to_numpy()

        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[3, 4, 4] = 1
        solver.obstacle.from_numpy(obstacle)

        surface_position_m = ti.Vector.field(3, dtype=ti.f32, shape=1)
        surface_force_n = ti.Vector.field(3, dtype=ti.f32, shape=1)
        surface_position_m.from_numpy(
            np.array(
                [[0.5 * (centers_x[3] + centers_x[4]), centers_y[4], centers_z[4]]],
                dtype=np.float32,
            )
        )
        surface_force_n.from_numpy(np.array([[0.0, 0.0, -2.0]], dtype=np.float32))

        spread_report = solver.spread_surface_forces(
            surface_position_m,
            surface_force_n,
            1,
            center_m=(0.0, 0.0, 0.0),
            force_sign=-1.0,
        )
        impulse_report = solver.apply_body_force(dt_s=2.5e-3)

        self.assertEqual(spread_report.active_grid_cells, 1)
        self.assertLess(spread_report.action_reaction_relative_error, 1.0e-5)
        self.assertAlmostEqual(spread_report.grid_force_n[2], 2.0, delta=2.0e-5)
        self.assertAlmostEqual(
            impulse_report.momentum_delta_n_s[2],
            spread_report.grid_force_n[2] * 2.5e-3,
            delta=abs(spread_report.grid_force_n[2] * 2.5e-3) * 1.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
