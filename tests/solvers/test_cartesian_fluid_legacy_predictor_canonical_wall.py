from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


_WALL_OWNER = (2, 2, 2)
_FREE_SIDE = (1, 2, 2)
_WALL_TARGET_MPS = 0.25


def _probe_storage_path_crossing(
    solver: ti.template(),
    start_i: ti.i32,
    start_j: ti.i32,
    start_k: ti.i32,
    target_i: ti.i32,
    target_j: ti.i32,
    target_k: ti.i32,
    output: ti.template(),
):
    output[None] = solver._storage_path_crosses_obstacle_or_canonical_hibm_wall(
        start_i,
        start_j,
        start_k,
        target_i,
        target_j,
        target_k,
    )


_probe_storage_path_crossing.__annotations__ = {
    "solver": ti.template(),
    "start_i": ti.i32,
    "start_j": ti.i32,
    "start_k": ti.i32,
    "target_i": ti.i32,
    "target_j": ti.i32,
    "target_k": ti.i32,
    "output": ti.template(),
}
_probe_storage_path_crossing = ti.kernel(_probe_storage_path_crossing)


def _probe_velocity_prev_trilinear(
    solver: ti.template(),
    gx: ti.f32,
    gy: ti.f32,
    gz: ti.f32,
    output: ti.template(),
):
    output[None] = solver._sample_velocity_prev_trilinear(
        gx,
        gy,
        gz,
        ti.Vector([0.0, 0.0, 0.0]),
    )


_probe_velocity_prev_trilinear.__annotations__ = {
    "solver": ti.template(),
    "gx": ti.f32,
    "gy": ti.f32,
    "gz": ti.f32,
    "output": ti.template(),
}
_probe_velocity_prev_trilinear = ti.kernel(_probe_velocity_prev_trilinear)


def _probe_laminar_diffusion_face_flux(
    solver: ti.template(), output: ti.template()
):
    output[0] = solver._laminar_diffusion_face_flux(
        ti.Vector([1.0, 2.0, 3.0]),
        ti.Vector([10.0, 20.0, 30.0]),
        2,
        2,
        2,
        1.0,
        0.5,
        0,
        True,
    )


_probe_laminar_diffusion_face_flux.__annotations__ = {
    "solver": ti.template(),
    "output": ti.template(),
}
_probe_laminar_diffusion_face_flux = ti.kernel(_probe_laminar_diffusion_face_flux)


def _seal_canonical_component_ledger(solver: CartesianFluidSolver) -> None:
    for consumer, capability in (
        solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES.items()
    ):
        solver._register_velocity_dirichlet_component_ledger_consumer_generation(
            consumer,
            capability=capability,
        )
    solver.seal_velocity_dirichlet_component_ledger()


def _stamp_canonical_wall_ledger(solver: CartesianFluidSolver) -> None:
    shape = (solver.nx, solver.ny, solver.nz)
    component_mask = np.zeros(shape, dtype=np.int32)
    component_mask[_WALL_OWNER] = 1
    solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
        component_mask
    )
    solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
        component_mask
    )
    solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
        component_mask
    )

    targets = np.zeros((*shape, 3), dtype=np.float32)
    targets[_WALL_OWNER + (0,)] = _WALL_TARGET_MPS
    solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
    mobility = np.ones((*shape, 3), dtype=np.float32)
    mobility[_WALL_OWNER + (0,)] = 0.0
    solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(mobility)
    enforcement = np.zeros((*shape, 3), dtype=np.float32)
    enforcement[_WALL_OWNER + (0,)] = 1.0
    solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
        enforcement
    )


def _canonical_wall_solver() -> CartesianFluidSolver:
    grid = CartesianGrid(
        bounds_min_m=(0.0, 0.0, 0.0),
        cell_widths_x_m=(1.0,) * 5,
        cell_widths_y_m=(1.0,) * 5,
        cell_widths_z_m=(1.0,) * 5,
    )
    solver = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1.0,
            viscosity_pa_s=0.0,
            dt_s=0.5,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )
    solver.set_velocity_dirichlet_boundary_authority("canonical")
    _stamp_canonical_wall_ledger(solver)
    _seal_canonical_component_ledger(solver)
    return solver


class CartesianFluidLegacyPredictorCanonicalWallContracts(unittest.TestCase):
    def test_storage_path_detects_diagonal_canonical_face_from_both_directions(
        self,
    ) -> None:
        solver = _canonical_wall_solver()
        mask = np.zeros((5, 5, 5), dtype=np.int32)
        mask[2, 1, 1] = 1
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        blocked = ti.field(dtype=ti.i32, shape=())

        _probe_storage_path_crossing(solver, 1, 1, 1, 2, 2, 1, blocked)
        self.assertEqual(int(blocked[None]), 1)
        _probe_storage_path_crossing(solver, 2, 2, 1, 1, 1, 1, blocked)
        self.assertEqual(int(blocked[None]), 1)

    def test_trilinear_corner_keeps_an_unconnected_component_wall_row(self) -> None:
        solver = _canonical_wall_solver()
        mask = np.zeros((5, 5, 5), dtype=np.int32)
        mask[_WALL_OWNER] = 2
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        velocity_prev = np.zeros((5, 5, 5, 3), dtype=np.float32)
        velocity_prev[_WALL_OWNER + (0,)] = 99.0
        solver.velocity_prev.from_numpy(velocity_prev)
        sampled = ti.Vector.field(3, dtype=ti.f32, shape=())

        _probe_velocity_prev_trilinear(solver, 1.4, 2.0, 2.0, sampled)

        self.assertGreater(float(sampled[None].x), 30.0)

    def test_trilinear_diagonal_corner_requires_connected_face_bit(self) -> None:
        solver = _canonical_wall_solver()
        velocity_prev = np.zeros((5, 5, 5, 3), dtype=np.float32)
        velocity_prev[2, 2, 1, 0] = 99.0
        solver.velocity_prev.from_numpy(velocity_prev)
        sampled = ti.Vector.field(3, dtype=ti.f32, shape=())

        mask = np.zeros((5, 5, 5), dtype=np.int32)
        mask[2, 1, 1] = 1
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        _probe_velocity_prev_trilinear(solver, 1.4, 1.4, 1.0, sampled)
        self.assertAlmostEqual(float(sampled[None].x), 0.0, places=6)

        mask[2, 1, 1] = 2
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        _probe_velocity_prev_trilinear(solver, 1.4, 1.4, 1.0, sampled)
        self.assertGreater(float(sampled[None].x), 10.0)

    def test_laminar_face_mask_is_normal_axis_scoped_and_zeroes_partial_wall_tangents(
        self,
    ) -> None:
        solver = _canonical_wall_solver()
        targets = np.zeros((5, 5, 5, 3), dtype=np.float32)
        targets[_WALL_OWNER] = (4.0, 50.0, 60.0)
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
        probed = ti.Vector.field(3, dtype=ti.f32, shape=1)

        mask = np.zeros((5, 5, 5), dtype=np.int32)
        mask[_WALL_OWNER] = 2
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        _probe_laminar_diffusion_face_flux(solver, probed)
        np.testing.assert_allclose(
            probed.to_numpy()[0], (9.0, 18.0, 27.0), rtol=0.0, atol=1.0e-6
        )

        mask[_WALL_OWNER] = 1
        solver.canonical_exact_hard_face_component_mask.from_numpy(mask)
        _probe_laminar_diffusion_face_flux(solver, probed)
        np.testing.assert_allclose(
            probed.to_numpy()[0], (6.0, 0.0, 0.0), rtol=0.0, atol=1.0e-6
        )

    def test_canonical_predict_rejects_restored_unsealed_ledger_without_mutation(
        self,
    ) -> None:
        solver = _canonical_wall_solver()
        velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
        velocity[_FREE_SIDE + (0,)] = -0.8
        solver.velocity.from_numpy(velocity)
        solver.save_state()
        solver.restore_state()
        before = solver.velocity.to_numpy()

        with self.assertRaisesRegex(RuntimeError, "not sealed"):
            solver.predict(
                dt_s=0.5,
                advection_scheme="euler",
                kinematic_viscosity_m2_s=0.0,
            )

        np.testing.assert_array_equal(solver.velocity.to_numpy(), before)

    def test_canonical_hard_face_mask_clears_on_restore_and_rebuilds_on_reseal(
        self,
    ) -> None:
        solver = _canonical_wall_solver()
        self.assertEqual(
            int(
                solver.canonical_exact_hard_face_component_mask.to_numpy()[
                    _WALL_OWNER
                ]
            ),
            1,
        )

        solver.save_state()
        solver.restore_state()

        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertEqual(
            int(
                solver.canonical_exact_hard_face_component_mask.to_numpy()[
                    _WALL_OWNER
                ]
            ),
            0,
        )

        _stamp_canonical_wall_ledger(solver)
        _seal_canonical_component_ledger(solver)
        self.assertEqual(
            int(
                solver.canonical_exact_hard_face_component_mask.to_numpy()[
                    _WALL_OWNER
                ]
            ),
            1,
        )

    def test_sealed_pressure_mask_refresh_cannot_rebuild_hard_face_mask(self) -> None:
        solver = _canonical_wall_solver()
        solver.velocity_dirichlet_boundary_active_component_mask.fill(0)
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()

        self.assertEqual(
            int(
                solver.canonical_exact_hard_face_component_mask.to_numpy()[
                    _WALL_OWNER
                ]
            ),
            1,
        )

    def test_legacy_predictor_does_not_backtrace_or_interpolate_through_canonical_wall(
        self,
    ) -> None:
        for scheme in ("euler", "rk2"):
            with self.subTest(scheme=scheme, contract="multi-cell backtrace"):
                solver = _canonical_wall_solver()
                velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
                velocity[_FREE_SIDE + (0,)] = -4.0
                velocity[3, 2, 2, 0] = 6.0
                velocity[_WALL_OWNER + (0,)] = 5.0
                solver.velocity.from_numpy(velocity)

                solver.predict(
                    dt_s=0.5,
                    advection_scheme=scheme,
                    kinematic_viscosity_m2_s=0.0,
                )
                predicted = solver.velocity.to_numpy()

                self.assertAlmostEqual(
                    float(predicted[_FREE_SIDE + (0,)]), -4.0, places=5
                )
                self.assertAlmostEqual(
                    float(predicted[_WALL_OWNER + (0,)]), _WALL_TARGET_MPS, places=6
                )

            with self.subTest(scheme=scheme, contract="one-sided interpolation"):
                solver = _canonical_wall_solver()
                velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
                velocity[_FREE_SIDE + (0,)] = -0.8
                velocity[_WALL_OWNER + (0,)] = 6.0
                solver.velocity.from_numpy(velocity)

                solver.predict(
                    dt_s=0.5,
                    advection_scheme=scheme,
                    kinematic_viscosity_m2_s=0.0,
                )
                predicted = solver.velocity.to_numpy()

                self.assertAlmostEqual(
                    float(predicted[_FREE_SIDE + (0,)]), -0.8, places=5
                )
                self.assertAlmostEqual(
                    float(predicted[_WALL_OWNER + (0,)]), _WALL_TARGET_MPS, places=6
                )

    def test_laminar_predictor_diffusion_does_not_read_through_canonical_wall(
        self,
    ) -> None:
        def free_side_velocity_after_diffusion(opposite_sentinel: float) -> float:
            solver = _canonical_wall_solver()
            velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
            velocity[_FREE_SIDE + (0,)] = 1.0
            velocity[_WALL_OWNER + (0,)] = opposite_sentinel
            solver.velocity.from_numpy(velocity)
            solver.predict(
                dt_s=0.05,
                advection_scheme="euler",
                kinematic_viscosity_m2_s=0.2,
            )
            return float(solver.velocity.to_numpy()[_FREE_SIDE + (0,)])

        low_sentinel = free_side_velocity_after_diffusion(-7.0)
        high_sentinel = free_side_velocity_after_diffusion(9.0)

        self.assertAlmostEqual(low_sentinel, high_sentinel, places=6)

    def test_laminar_canonical_wall_diffusion_uses_the_exact_moving_target(
        self,
    ) -> None:
        def free_side_velocity_after_diffusion(wall_target: float) -> float:
            solver = _canonical_wall_solver()
            targets = solver.velocity_dirichlet_boundary_value_mps.to_numpy()
            targets[_WALL_OWNER + (0,)] = wall_target
            solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
            solver._invalidate_velocity_dirichlet_component_ledger()
            _seal_canonical_component_ledger(solver)

            velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
            velocity[_FREE_SIDE + (0,)] = 1.0
            solver.velocity.from_numpy(velocity)
            solver.predict(
                dt_s=0.05,
                advection_scheme="euler",
                kinematic_viscosity_m2_s=0.2,
            )
            return float(solver.velocity.to_numpy()[_FREE_SIDE + (0,)])

        stationary_wall = free_side_velocity_after_diffusion(0.0)
        moving_wall = free_side_velocity_after_diffusion(0.5)

        self.assertGreater(moving_wall, stationary_wall + 5.0e-3)


if __name__ == "__main__":
    unittest.main()
