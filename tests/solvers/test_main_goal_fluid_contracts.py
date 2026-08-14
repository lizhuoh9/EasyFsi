from __future__ import annotations

import math
import unittest

import numpy as np
import taichi as ti

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fluids.grid import CartesianGrid
from simulation_core.fluids.solver import CartesianFluidSolver
from simulation_core.fluids.spec import FluidDomainSpec


_OPEN_WALLS = (False,) * 6


def _cuda_solver(grid: CartesianGrid) -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-3,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _exact_obstacle_distance(
    grid: CartesianGrid,
    obstacle: np.ndarray,
) -> np.ndarray:
    faces = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            grid.cell_faces_x_m,
            grid.cell_faces_y_m,
            grid.cell_faces_z_m,
        )
    )
    centers = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            grid.cell_centers_x_m,
            grid.cell_centers_y_m,
            grid.cell_centers_z_m,
        )
    )
    obstacle_cells = np.argwhere(obstacle != 0)
    expected = np.full(obstacle.shape, np.inf, dtype=np.float64)
    for cell in np.ndindex(obstacle.shape):
        position = tuple(centers[axis][cell[axis]] for axis in range(3))
        for source in obstacle_cells:
            squared_distance = 0.0
            for axis in range(3):
                source_index = int(source[axis])
                offset = max(
                    faces[axis][source_index] - position[axis],
                    0.0,
                    position[axis] - faces[axis][source_index + 1],
                )
                squared_distance += offset * offset
            expected[cell] = min(expected[cell], math.sqrt(squared_distance))
    return expected


class SSTWallDistanceUnionContracts(unittest.TestCase):
    def test_save_restore_preserves_nonzero_volume_source(self) -> None:
        solver = _cuda_solver(
            CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
            )
        )
        expected = np.zeros((4, 4, 4), dtype=np.float32)
        expected[1, 2, 3] = 2.5
        expected_velocity = np.full((4, 4, 4, 3), 1.25, dtype=np.float32)
        expected_velocity_prev = np.full((4, 4, 4, 3), -0.75, dtype=np.float32)
        solver.volume_source_s.from_numpy(expected)
        solver.velocity.from_numpy(expected_velocity)
        solver.velocity_prev.from_numpy(expected_velocity_prev)

        solver.save_state()
        solver.volume_source_s.fill(0.0)
        solver.velocity.fill(0.0)
        solver.velocity_prev.fill(0.0)
        solver.restore_state()

        np.testing.assert_array_equal(solver.volume_source_s.to_numpy(), expected)
        np.testing.assert_array_equal(solver.velocity.to_numpy(), expected_velocity)
        np.testing.assert_array_equal(
            solver.velocity_prev.to_numpy(),
            expected_velocity_prev,
        )

    def test_sst_configuration_defers_wall_distance_until_geometry_is_final(self) -> None:
        solver = _cuda_solver(
            CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
            )
        )

        def unexpected_wall_distance_build(**_kwargs: object) -> None:
            raise AssertionError("configuration must not build provisional wall distance")

        solver.prepare_sst_wall_distance = unexpected_wall_distance_build  # type: ignore[method-assign]
        solver.configure_sst_2003(
            inlet_velocity_mps=10.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="fluent_correlation",
            defer_wall_distance=True,
        )

        self.assertEqual(solver.turbulence_model, "sst_2003")
        self.assertFalse(solver._sst_wall_distance_valid)

    def test_static_obstacle_remains_in_union_with_marker_or_segment_geometry(
        self,
    ) -> None:
        grid = CartesianGrid.uniform(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
        )
        solver = _cuda_solver(grid)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        obstacle[0, 1, 1] = 1
        solver.obstacle.from_numpy(obstacle)

        markers = ti.Vector.field(3, dtype=ti.f32, shape=2)
        markers.from_numpy(
            np.asarray(((0.0, 0.99, 0.99), (1.0, 0.99, 0.99)), dtype=np.float32)
        )
        segments = ti.Vector.field(3, dtype=ti.i32, shape=1)
        segments.from_numpy(np.asarray(((0, 1, -1),), dtype=np.int32))

        cases = (
            {
                "marker_position_m": markers,
                "marker_count": 1,
            },
            {
                "marker_position_m": markers,
                "marker_count": 2,
                "projection_segment_indices": segments,
                "projection_segment_count": 1,
            },
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                solver._sst_wall_distance_valid = False
                solver.prepare_sst_wall_distance(
                    no_slip_domain_walls=_OPEN_WALLS,
                    **arguments,
                )
                self.assertAlmostEqual(
                    float(solver.sst_wall_distance_m[2, 1, 1]),
                    0.375,
                    places=6,
                )

    def test_graded_obstacle_distance_matches_brute_force_reference(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(
                0.0944780480,
                0.1236522134,
                0.0424950093,
                0.5030133411,
                0.1839919760,
                0.0523694122,
            ),
            cell_widths_y_m=(
                0.1692490746,
                0.0656558885,
                0.0277519390,
                0.0506783822,
                0.6077031052,
                0.0789616105,
            ),
            cell_widths_z_m=(
                0.0503074683,
                0.6385186312,
                0.0382240849,
                0.0389215981,
                0.0719565500,
                0.1620716675,
            ),
        )
        solver = _cuda_solver(grid)
        obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
        for index in (
            (4, 5, 3),
            (4, 0, 5),
            (1, 1, 3),
            (2, 2, 3),
            (5, 2, 4),
            (4, 4, 1),
        ):
            obstacle[index] = 1
        solver.obstacle.from_numpy(obstacle)

        solver.prepare_sst_wall_distance(no_slip_domain_walls=_OPEN_WALLS)

        expected = _exact_obstacle_distance(grid, obstacle)
        active = obstacle == 0
        np.testing.assert_allclose(
            solver.sst_wall_distance_m.to_numpy()[active],
            expected[active],
            rtol=2.0e-6,
            atol=2.0e-7,
        )


class FluidMutatorPreflightContracts(unittest.TestCase):
    @staticmethod
    def _device_work_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("device work must not start")

    def test_public_mutators_reject_nonfinite_or_non_f32_inputs_before_device_work(
        self,
    ) -> None:
        cases = (
            (
                "_set_vertical_pressure_gradient_kernel",
                lambda solver: solver.set_vertical_pressure_gradient(float("nan"), 1.0),
            ),
            (
                "_set_vertical_pressure_gradient_kernel",
                lambda solver: solver.set_vertical_pressure_gradient(0.0, 1.0e100),
            ),
            (
                "_set_uniform_velocity_kernel",
                lambda solver: solver.set_uniform_velocity((0.0, float("inf"), 0.0)),
            ),
            (
                "_set_sinusoidal_divergent_velocity_kernel",
                lambda solver: solver.set_sinusoidal_divergent_velocity(float("nan")),
            ),
            (
                "_set_sinusoidal_divergent_velocity_kernel",
                lambda solver: solver.set_sinusoidal_divergent_velocity(1.0e100),
            ),
            (
                "_set_simple_shear_velocity_kernel",
                lambda solver: solver.set_simple_shear_velocity(1.0e100),
            ),
            (
                "_set_simple_shear_velocity_kernel",
                lambda solver: solver.set_simple_shear_velocity(
                    1.0,
                    center_y_m=float("nan"),
                ),
            ),
            (
                "_apply_obstacle_velocity_kernel",
                lambda solver: solver.apply_obstacle_velocity((0.0, 1.0e-100, 0.0)),
            ),
            (
                "_obstacle_velocity_error_kernel",
                lambda solver: solver.obstacle_velocity_error((float("nan"), 0.0, 0.0)),
            ),
        )
        for kernel_name, invoke in cases:
            with self.subTest(kernel=kernel_name):
                solver = object.__new__(CartesianFluidSolver)
                setattr(solver, kernel_name, self._device_work_must_not_start)
                with self.assertRaisesRegex(ValueError, "f32-representable"):
                    invoke(solver)


class SurfaceForceDerivedPreflightContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = _cuda_solver(
            CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
            )
        )
        self.position = ti.Vector.field(3, dtype=ti.f64, shape=1)
        self.force = ti.Vector.field(3, dtype=ti.f64, shape=1)
        self.position.from_numpy(
            np.asarray(((0.125, 0.125, 0.125),), dtype=np.float64)
        )
        self.solver.force.fill(7.0)
        self.force_before = self.solver.force.to_numpy().copy()

    def _assert_rejected_without_force_mutation(
        self,
        force_n: float,
        *,
        force_sign: float,
    ) -> None:
        self.force.from_numpy(
            np.asarray(((force_n, 0.0, 0.0),), dtype=np.float64)
        )
        with self.assertRaisesRegex(ValueError, "derived"):
            self.solver.spread_surface_forces(
                self.position,
                self.force,
                vertex_count=1,
                center_m=(0.0, 0.0, 0.0),
                force_sign=force_sign,
            )
        np.testing.assert_array_equal(self.solver.force.to_numpy(), self.force_before)

    def test_force_sign_product_overflow_fails_before_force_mutation(self) -> None:
        self._assert_rejected_without_force_mutation(2.0e38, force_sign=2.0)

    def test_force_density_overflow_fails_before_force_mutation(self) -> None:
        self._assert_rejected_without_force_mutation(1.0e38, force_sign=1.0)


class MarkerFeedbackPreflightContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = _cuda_solver(
            CartesianGrid.uniform(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
            )
        )
        self.position = ti.Vector.field(3, dtype=ti.f64, shape=1)
        self.velocity = ti.Vector.field(3, dtype=ti.f64, shape=1)
        self.region = ti.field(dtype=ti.i32, shape=1)
        self.position.from_numpy(
            np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
        )
        self.velocity.fill(0.0)
        self.region[0] = 101
        self.solver.marker_feedback_owned[0, 0, 0] = 1
        self.solver.marker_feedback_target_weight[0, 0, 0] = 7.0

    def _apply(self, marker_count: object) -> None:
        self.solver.apply_marker_feedback_constraints(
            self.position,
            self.velocity,
            self.region,
            marker_count,  # type: ignore[arg-type]
            feedback_available=True,
            preserve_velocity_constraints=True,
            primary_region_id=101,
            secondary_region_id=202,
        )

    def _assert_constraints_unchanged(self) -> None:
        self.assertEqual(int(self.solver.marker_feedback_owned[0, 0, 0]), 1)
        self.assertEqual(
            float(self.solver.marker_feedback_target_weight[0, 0, 0]),
            7.0,
        )

    def test_fractional_negative_or_over_capacity_count_fails_before_clear(self) -> None:
        for marker_count in (1.5, -1, 2):
            with self.subTest(marker_count=marker_count):
                with self.assertRaises((TypeError, ValueError)):
                    self._apply(marker_count)
                self._assert_constraints_unchanged()

    def test_nonfinite_active_marker_fails_before_clear(self) -> None:
        self.position.from_numpy(
            np.asarray(((float("nan"), 0.25, 0.25),), dtype=np.float64)
        )
        with self.assertRaisesRegex(ValueError, "marker feedback"):
            self._apply(1)
        self._assert_constraints_unchanged()

    def test_region_ids_require_i32_storage_before_clear(self) -> None:
        invalid_region = ti.field(dtype=ti.f32, shape=1)
        invalid_region[0] = 101.0
        with self.assertRaisesRegex(TypeError, "marker_region_id"):
            self.solver.apply_marker_feedback_constraints(
                self.position,
                self.velocity,
                invalid_region,
                1,
                feedback_available=True,
                preserve_velocity_constraints=True,
                primary_region_id=101,
                secondary_region_id=202,
            )
        self._assert_constraints_unchanged()


if __name__ == "__main__":
    unittest.main()
