from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


_OPEN_WALLS = (False, False, False, False, False, False)


def _cuda_solver(*, dt_s: float = 0.05) -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-8,
            dt_s=dt_s,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _set_minimum_normal_row(
    velocity: np.ndarray,
    *,
    axis: int,
    value: float,
) -> np.ndarray:
    updated = np.array(velocity, copy=True)
    index = [slice(None), slice(None), slice(None), axis]
    index[axis] = 0
    updated[tuple(index)] = value
    return updated


def _interior_normal_component(velocity: np.ndarray, *, axis: int) -> np.ndarray:
    index = [slice(None), slice(None), slice(None), axis]
    index[axis] = slice(1, None)
    return velocity[tuple(index)]


class CartesianFluidMusclBoundaryContracts(unittest.TestCase):
    def _predict_from_stale_minimum_row(
        self,
        solver: CartesianFluidSolver,
        *,
        axis: int,
        stale_value: float,
        wall_flags: tuple[bool, bool, bool, bool, bool, bool],
    ) -> np.ndarray:
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., axis] = 1.0
        solver.velocity.from_numpy(
            _set_minimum_normal_row(
                velocity,
                axis=axis,
                value=stale_value,
            )
        )
        solver.predict(
            dt_s=0.05,
            advection_scheme="muscl_tvd",
            kinematic_viscosity_m2_s=0.0,
            no_slip_domain_walls=wall_flags,
        )
        return _interior_normal_component(solver.velocity.to_numpy(), axis=axis)

    def test_exact_minimum_face_makes_muscl_independent_of_stale_compact_row(
        self,
    ) -> None:
        solver = _cuda_solver()
        face_masks = (
            solver.external_velocity_boundary_x_face_active_component_mask,
            solver.external_velocity_boundary_y_face_active_component_mask,
            solver.external_velocity_boundary_z_face_active_component_mask,
        )

        for axis in range(3):
            with self.subTest(axis=axis):
                for field in face_masks:
                    field.fill(0)
                target = [0.0, 0.0, 0.0]
                target[axis] = 1.0
                solver.refresh_external_velocity_boundary_face_uniform(
                    axis_index=axis,
                    side_index=0,
                    target_velocity_mps=tuple(target),
                    active_component_mask=1 << axis,
                )
                negative = self._predict_from_stale_minimum_row(
                    solver,
                    axis=axis,
                    stale_value=-7.0,
                    wall_flags=_OPEN_WALLS,
                )
                positive = self._predict_from_stale_minimum_row(
                    solver,
                    axis=axis,
                    stale_value=11.0,
                    wall_flags=_OPEN_WALLS,
                )
                np.testing.assert_allclose(negative, positive, rtol=0.0, atol=2.0e-6)
                np.testing.assert_allclose(negative, 1.0, rtol=0.0, atol=2.0e-6)

    def test_no_slip_minimum_face_ignores_stale_compact_normal_row(self) -> None:
        solver = _cuda_solver()
        for axis in range(3):
            with self.subTest(axis=axis):
                wall_flags = list(_OPEN_WALLS)
                wall_flags[2 * axis] = True
                negative = self._predict_from_stale_minimum_row(
                    solver,
                    axis=axis,
                    stale_value=-7.0,
                    wall_flags=tuple(wall_flags),
                )
                positive = self._predict_from_stale_minimum_row(
                    solver,
                    axis=axis,
                    stale_value=11.0,
                    wall_flags=tuple(wall_flags),
                )
                np.testing.assert_allclose(negative, positive, rtol=0.0, atol=2.0e-6)

    def test_canonical_ab_owner_target_reaches_adjacent_upwind_dual_flux(self) -> None:
        solver = _cuda_solver()
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        for obstacle_side in (-1, 1):
            with self.subTest(obstacle_side=obstacle_side):
                fluid_k = 2
                obstacle_k = fluid_k + obstacle_side
                storage_k = fluid_k if obstacle_side < 0 else obstacle_k
                obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                obstacle[:, :, obstacle_k] = 1
                solver.obstacle.from_numpy(obstacle)

                component_mask = np.zeros((4, 4, 4), dtype=np.int32)
                component_mask[:, :, storage_k] = 4
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    component_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    component_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    component_mask
                )
                targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                targets[:, :, storage_k, 2] = 0.4
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((4, 4, 4, 3), dtype=np.float32)
                mobility[:, :, storage_k, 2] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(mobility)
                enforcement = np.zeros((4, 4, 4, 3), dtype=np.float32)
                enforcement[:, :, storage_k, 2] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                directed_speed = 1.0 if obstacle_side < 0 else -1.0
                measured_fluxes = []
                for stale_owner in (37.0, -41.0):
                    velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
                    velocity[..., 2] = directed_speed
                    velocity[:, :, storage_k, 2] = stale_owner
                    solver.velocity.from_numpy(velocity)
                    solver._compute_muscl_momentum_fluxes(
                        solver.velocity,
                        (0, 0, 0, 0, 0, 0),
                    )
                    measured_fluxes.append(
                        float(
                            solver.muscl_momentum_flux_z.to_numpy()[
                                1, 1, storage_k + (1 if obstacle_side < 0 else 0), 2
                            ]
                        )
                    )

                np.testing.assert_allclose(
                    measured_fluxes,
                    (0.0125 * directed_speed, 0.0125 * directed_speed),
                    rtol=0.0,
                    atol=2.0e-6,
                )

    def test_soft_moving_interface_source_survives_both_ssp_stages(self) -> None:
        solver = _cuda_solver()
        source_value = 0.3

        for authority in ("canonical", "legacy"):
            for obstacle_side in (-1, 1):
                with self.subTest(authority=authority, obstacle_side=obstacle_side):
                    fluid_k = 2
                    obstacle_k = fluid_k + obstacle_side
                    storage_k = fluid_k if obstacle_side < 0 else obstacle_k
                    obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                    obstacle[:, :, obstacle_k] = 1
                    solver.obstacle.from_numpy(obstacle)
                    solver.set_velocity_dirichlet_boundary_authority(authority)

                    active = np.zeros((4, 4, 4), dtype=np.int32)
                    active[:, :, storage_k] = 1
                    component_active = 4 * active
                    targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                    targets[:, :, storage_k, 2] = 0.4
                    solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                    solver.velocity_dirichlet_boundary_hard_fixed_component_mask.fill(0)
                    if authority == "canonical":
                        solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                            component_active
                        )
                        solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                            component_active
                        )
                        component_weight = np.zeros((4, 4, 4, 3), dtype=np.float32)
                        component_weight[:, :, storage_k, 2] = 0.25
                        solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                            component_weight
                        )
                    else:
                        solver.velocity_dirichlet_boundary_active.from_numpy(active)
                        solver.velocity_dirichlet_boundary_owned_row.from_numpy(active)
                        weight = np.zeros((4, 4, 4), dtype=np.float32)
                        weight[:, :, storage_k] = 0.25
                        solver.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
                            weight
                        )

                    velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
                    velocity[:, :, storage_k, 2] = source_value
                    solver.velocity.from_numpy(velocity)
                    solver.velocity_transport_base.from_numpy(velocity)
                    solver._compute_muscl_momentum_fluxes(
                        solver.velocity_transport_base,
                        (0, 0, 0, 0, 0, 0),
                    )
                    solver._muscl_momentum_ssp_stage_kernel(
                        solver.velocity_transport_base,
                        0.1,
                        0,
                    )
                    np.testing.assert_allclose(
                        solver.velocity.to_numpy()[:, :, storage_k, 2],
                        source_value,
                        rtol=0.0,
                        atol=2.0e-7,
                    )

                    solver._copy_velocity_to_prev_kernel()
                    solver._compute_muscl_momentum_fluxes(
                        solver.velocity_prev,
                        (0, 0, 0, 0, 0, 0),
                    )
                    solver._muscl_momentum_ssp_stage_kernel(
                        solver.velocity_prev,
                        0.1,
                        1,
                    )
                    np.testing.assert_allclose(
                        solver.velocity.to_numpy()[:, :, storage_k, 2],
                        source_value,
                        rtol=0.0,
                        atol=2.0e-7,
                    )

    def test_sst_scalar_transport_does_not_rebuild_momentum_dual_ledgers(self) -> None:
        solver = _cuda_solver(dt_s=1.0e-4)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, -0.1))
        solver.muscl_momentum_dual_volume_m3.fill(-123.0)
        solver.muscl_momentum_volume_flux_x_half_m3_s.fill(-123.0)
        solver.muscl_momentum_volume_flux_y_half_m3_s.fill(-123.0)
        solver.muscl_momentum_volume_flux_z_half_m3_s.fill(-123.0)

        solver.advance_sst_transport(
            dt_s=1.0e-4,
            kinematic_viscosity_m2_s=1.0e-8,
            no_slip_domain_walls=_OPEN_WALLS,
            advection_scheme="euler",
        )

        np.testing.assert_array_equal(
            solver.muscl_momentum_dual_volume_m3.to_numpy(),
            -123.0,
        )
        np.testing.assert_array_equal(
            solver.muscl_momentum_volume_flux_x_half_m3_s.to_numpy(),
            -123.0,
        )
        np.testing.assert_array_equal(
            solver.muscl_momentum_volume_flux_y_half_m3_s.to_numpy(),
            -123.0,
        )
        np.testing.assert_array_equal(
            solver.muscl_momentum_volume_flux_z_half_m3_s.to_numpy(),
            -123.0,
        )


if __name__ == "__main__":
    unittest.main()
