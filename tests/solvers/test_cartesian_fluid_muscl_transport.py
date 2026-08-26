from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


_OPEN_WALLS = (False, False, False, False, False, False)


def _probe_muscl_x_face_states(
    solver: ti.template(),
    field: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    output: ti.template(),
):
    output[0] = solver._muscl_vector_face_state(field, i, j, k, 0, -1)
    output[1] = solver._muscl_vector_face_state(field, i, j, k, 0, 1)


# ``from __future__ import annotations`` turns these Taichi annotations into
# strings before the decorator inspects them.  Restore the concrete Taichi
# types explicitly so this test-only probe remains importable on Taichi 1.7.
_probe_muscl_x_face_states.__annotations__ = {
    "solver": ti.template(),
    "field": ti.template(),
    "i": ti.i32,
    "j": ti.i32,
    "k": ti.i32,
    "output": ti.template(),
}
_probe_muscl_x_face_states = ti.kernel(_probe_muscl_x_face_states)


def _cuda_solver(
    *,
    grid_nodes: tuple[int, int, int],
    density_kgm3: float = 1.0,
    viscosity_pa_s: float = 0.0,
    dt_s: float,
) -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=grid_nodes,
            density_kgm3=density_kgm3,
            viscosity_pa_s=viscosity_pa_s,
            dt_s=dt_s,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


class CartesianFluidMusclTransportContracts(unittest.TestCase):
    """Accuracy contracts for the production finite-volume transport path."""

    @staticmethod
    def _smooth_transverse_shear_result(
        cells_z: int,
    ) -> tuple[float, float]:
        cfl = 0.25
        transport_time_s = 0.125
        dt_s = cfl / float(cells_z)
        step_count = int(round(transport_time_s / dt_s))
        solver = _cuda_solver(
            grid_nodes=(4, 4, cells_z),
            dt_s=dt_s,
        )

        z = (np.arange(cells_z, dtype=np.float32) + 0.5) / cells_z
        initial = 0.1 * np.sin(4.0 * np.pi * z)
        velocity = np.zeros((4, 4, cells_z, 3), dtype=np.float32)
        velocity[..., 0] = initial
        velocity[..., 2] = -1.0
        solver.velocity.from_numpy(velocity)

        for _ in range(step_count):
            solver.predict(
                dt_s=dt_s,
                advection_scheme="muscl_tvd",
                kinematic_viscosity_m2_s=0.0,
            )

        transported = solver.velocity.to_numpy()[1, 1, :, 0]
        exact = 0.1 * np.sin(4.0 * np.pi * (z + transport_time_s))
        interior = (z >= 0.10) & (z <= 0.75)
        relative_l2 = float(
            np.linalg.norm(transported[interior] - exact[interior])
            / np.linalg.norm(exact[interior])
        )
        retained_amplitude = float(
            np.dot(transported[interior], exact[interior])
            / np.dot(exact[interior], exact[interior])
        )
        return relative_l2, retained_amplitude

    def test_muscl_native_mac_reconstruction_is_affine_exact_on_a_graded_grid(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 2.0, 4.0, 8.0, 16.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=0.0,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        probed_states = ti.Vector.field(3, dtype=ti.f32, shape=2)

        alpha = 0.25
        beta = 1.0
        faces_x = np.asarray(grid.cell_faces_x_m, dtype=np.float32)
        centers_x = np.asarray(grid.cell_centers_x_m, dtype=np.float32)
        velocity = np.zeros((5, 4, 4, 3), dtype=np.float32)
        # Native backward-MAC locations: u_x lives on F_x while the transverse
        # u_y component is cell-centred in x.  One fixture therefore locks both
        # the new normal contract and the already-correct transverse contract.
        velocity[..., 0] = (alpha * faces_x[:-1] + beta)[:, None, None]
        velocity[..., 1] = (alpha * centers_x + beta)[:, None, None]
        solver.velocity.from_numpy(velocity)

        solver._prepare_muscl_velocity_component_reconstruction_kernel(
            solver.velocity, 0, 0
        )
        solver._prepare_muscl_velocity_component_reconstruction_kernel(
            solver.velocity, 0, 1
        )
        target_i, target_j, target_k = 2, 1, 1
        _probe_muscl_x_face_states(
            solver,
            solver.velocity,
            target_i,
            target_j,
            target_k,
            probed_states,
        )

        slopes = solver.muscl_velocity_slope.to_numpy()
        states = probed_states.to_numpy()
        with self.subTest(contract="normal slope uses primal face spacing"):
            self.assertAlmostEqual(
                float(slopes[target_i, target_j, target_k, 0, 0]),
                alpha,
                places=6,
            )
        with self.subTest(contract="normal states live on dual cell centres"):
            np.testing.assert_allclose(
                states[:, 0],
                alpha
                * np.asarray(
                    (centers_x[target_i - 1], centers_x[target_i]),
                    dtype=np.float32,
                )
                + beta,
                rtol=0.0,
                atol=2.0e-6,
            )
        with self.subTest(contract="transverse slope keeps centre spacing"):
            self.assertAlmostEqual(
                float(slopes[target_i, target_j, target_k, 0, 1]),
                alpha,
                places=6,
            )
        with self.subTest(contract="transverse states stay on scalar faces"):
            np.testing.assert_allclose(
                states[:, 1],
                alpha
                * np.asarray(
                    (faces_x[target_i], faces_x[target_i + 1]),
                    dtype=np.float32,
                )
                + beta,
                rtol=0.0,
                atol=2.0e-6,
            )

    def test_muscl_native_mac_dual_volume_and_half_flux_ledgers_are_graded_exact(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(2.0, 6.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=0.0,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        # u_x lives at F_x=(0,2,8,12).  Its row i=1 dual CV spans
        # C_x[0]..C_x[1], hence V_x=(1+3)*1*1=4.  The two normal dual-face
        # volume fluxes are the face-velocity averages 1 and 5.
        velocity[..., 0] = np.asarray((0.0, 2.0, 8.0, 12.0), dtype=np.float32)[
            :, None, None
        ]
        # Across y faces 1 and 2, the x-minus/x-plus half cells carry distinct
        # velocities.  Their graded half areas are 1 and 3, so the integrated
        # Q halves must be (1,9) and (2,12), not one collocated-cell sample.
        velocity[0, 1, :, 1] = 1.0
        velocity[1, 1, :, 1] = 3.0
        velocity[0, 2, :, 1] = 2.0
        velocity[1, 2, :, 1] = 4.0
        solver.velocity.from_numpy(velocity)
        solver._compute_muscl_momentum_fluxes(solver.velocity, (0, 0, 0, 0, 0, 0))

        self.assertTrue(
            hasattr(solver, "muscl_momentum_dual_volume_m3"),
            "momentum MUSCL still has no component-native dual-CV volume ledger",
        )
        self.assertTrue(
            hasattr(solver, "muscl_momentum_volume_flux_y_half_m3_s"),
            "momentum MUSCL still collapses the two graded transverse half faces",
        )
        dual_volume = solver.muscl_momentum_dual_volume_m3.to_numpy()
        qx_half = solver.muscl_momentum_volume_flux_x_half_m3_s.to_numpy()
        qy_half = solver.muscl_momentum_volume_flux_y_half_m3_s.to_numpy()
        qx_minus = qx_half[..., 0]
        qx_plus = qx_half[..., 1]
        qy_minus = qy_half[..., 0]
        qy_plus = qy_half[..., 1]

        target = (1, 1, 1)
        self.assertAlmostEqual(float(dual_volume[target + (0,)]), 4.0, places=6)
        np.testing.assert_allclose(
            (
                qx_minus[1, 1, 1, 0] + qx_plus[1, 1, 1, 0],
                qx_minus[2, 1, 1, 0] + qx_plus[2, 1, 1, 0],
            ),
            (1.0, 5.0),
            rtol=0.0,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            (
                qy_minus[1, 1, 1, 0],
                qy_plus[1, 1, 1, 0],
                qy_minus[1, 2, 1, 0],
                qy_plus[1, 2, 1, 0],
            ),
            (1.0, 9.0, 2.0, 12.0),
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_muscl_native_mac_stage_uses_the_same_dual_q_for_continuity(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(2.0, 6.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=0.0,
                dt_s=0.1,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 0] = 5.0
        # For the u_x row i=1, the y-minus integrated Q is 1*1 + 3*3=10
        # and y-plus is 1*2 + 3*4=14.  A constant transported u_x must remain
        # exactly five because div(Q*u_x)-u_x*div(Q) vanishes on that same
        # component dual volume V=4.
        velocity[0, 1, :, 1] = 1.0
        velocity[1, 1, :, 1] = 3.0
        velocity[0, 2, :, 1] = 2.0
        velocity[1, 2, :, 1] = 4.0
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

        self.assertAlmostEqual(
            float(solver.velocity.to_numpy()[1, 1, 1, 0]),
            5.0,
            places=5,
        )

    def test_muscl_canonical_moving_wall_has_zero_relative_q_and_fixed_ab_owner(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0,) * 5,
            cell_widths_y_m=(1.0,) * 5,
            cell_widths_z_m=(1.0,) * 5,
        )
        wall_target_mps = 0.4

        for obstacle_side, stale_owner_mps in ((-1, 37.0), (1, -41.0)):
            with self.subTest(obstacle_side=obstacle_side):
                solver = CartesianFluidSolver(
                    FluidDomainSpec(
                        bounds_min_m=grid.bounds_min_m,
                        bounds_max_m=grid.bounds_max_m,
                        grid_nodes=None,
                        density_kgm3=1.0,
                        viscosity_pa_s=0.0,
                        dt_s=0.1,
                        cartesian_grid=grid,
                    ),
                    runtime=TaichiRuntimeConfig(arch="cuda"),
                )
                fluid_k = 2
                obstacle_k = fluid_k + obstacle_side
                storage_k = fluid_k if obstacle_side < 0 else obstacle_k
                obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
                obstacle[:, :, obstacle_k] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")

                component_mask = np.zeros(grid.grid_nodes, dtype=np.int32)
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
                targets = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                targets[:, :, storage_k, 2] = wall_target_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((*grid.grid_nodes, 3), dtype=np.float32)
                mobility[:, :, storage_k, 2] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                enforcement[:, :, storage_k, 2] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                stale = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                stale[:, :, storage_k, 2] = stale_owner_mps
                solver.velocity.from_numpy(stale)
                solver.velocity_transport_base.from_numpy(stale)
                solver._compute_muscl_momentum_fluxes(
                    solver.velocity_transport_base,
                    (0, 0, 0, 0, 0, 0),
                )

                np.testing.assert_allclose(
                    solver.muscl_normal_velocity_z.to_numpy()[:, :, storage_k],
                    0.0,
                    rtol=0.0,
                    atol=1.0e-7,
                )
                wall_half_q = (
                    solver.muscl_momentum_volume_flux_z_half_m3_s.to_numpy()[
                        :, :, storage_k, :, :
                    ]
                )
                np.testing.assert_allclose(
                    wall_half_q,
                    0.0,
                    rtol=0.0,
                    atol=1.0e-7,
                )

                solver._muscl_momentum_ssp_stage_kernel(
                    solver.velocity_transport_base,
                    0.1,
                    0,
                )
                np.testing.assert_allclose(
                    solver.velocity.to_numpy()[:, :, storage_k, 2],
                    wall_target_mps,
                    rtol=0.0,
                    atol=2.0e-7,
                )

    def test_muscl_half_flux_counterflow_cannot_cancel_out_of_momentum_or_cfl(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(2.0, 6.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=0.0,
                dt_s=0.1,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        # On y face 1 of the u_x dual CV at i=1, the x-minus half has
        # Q=1*(+6)=+6 and the x-plus half has Q=3*(-2)=-6.  Net Q is zero,
        # but the signs use donor states u_x=1 and 3: F=+6*1-6*3=-12.
        velocity[1, 0, :, 0] = 1.0
        velocity[1, 1:, :, 0] = 3.0
        velocity[0, 1, :, 1] = 6.0
        velocity[1, 1, :, 1] = -2.0
        solver.velocity.from_numpy(velocity)
        solver._compute_muscl_momentum_fluxes(
            solver.velocity,
            (0, 0, 0, 0, 0, 0),
        )

        half_q = solver.muscl_momentum_volume_flux_y_half_m3_s.to_numpy()
        np.testing.assert_allclose(
            half_q[1, 1, 1, 0, :],
            (6.0, -6.0),
            rtol=0.0,
            atol=2.0e-6,
        )
        self.assertAlmostEqual(
            float(solver.muscl_momentum_flux_y.to_numpy()[1, 1, 1, 0]),
            -12.0,
            places=5,
        )

        # Erase the legacy scalar-face ledger after flux construction.  A
        # momentum CFL path that still reads it reports zero; the native path
        # must consume the signed half-Q ledger and see the 6/4 local rate.
        solver.muscl_normal_velocity_x.fill(0.0)
        solver.muscl_normal_velocity_y.fill(0.0)
        solver.muscl_normal_velocity_z.fill(0.0)
        self.assertGreaterEqual(
            float(solver._muscl_momentum_advection_rate_kernel()),
            1.5 - 2.0e-6,
        )

    def test_muscl_momentum_is_second_order_and_retains_smooth_shear(self) -> None:
        coarse_error, coarse_amplitude = self._smooth_transverse_shear_result(64)
        fine_error, fine_amplitude = self._smooth_transverse_shear_result(128)

        observed_order = math.log(coarse_error / fine_error, 2.0)
        self.assertGreaterEqual(observed_order, 1.8)
        self.assertGreaterEqual(coarse_amplitude, 0.95)
        self.assertGreaterEqual(fine_amplitude, coarse_amplitude)

    def test_muscl_sst_transport_is_bounded_and_retains_scalar_amplitude(
        self,
    ) -> None:
        cells_z = 64
        cfl = 0.25
        transport_time_s = 0.125
        dt_s = cfl / float(cells_z)
        step_count = int(round(transport_time_s / dt_s))
        solver = _cuda_solver(
            grid_nodes=(4, 4, cells_z),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-8,
            dt_s=dt_s,
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        z = (np.arange(cells_z, dtype=np.float32) + 0.5) / cells_z
        initial_profile = 1.0e-6 * (1.0 + 0.2 * np.sin(4.0 * np.pi * z))
        initial_k = np.broadcast_to(
            initial_profile[None, None, :], (4, 4, cells_z)
        ).copy()
        solver.sst_turbulent_kinetic_energy.from_numpy(initial_k.astype(np.float32))
        solver.sst_specific_dissipation_rate.fill(10.0)
        solver.set_uniform_velocity((0.0, 0.0, -1.0))

        for _ in range(step_count):
            solver.advance_sst_transport(
                dt_s=dt_s,
                kinematic_viscosity_m2_s=1.0e-8,
                no_slip_domain_walls=_OPEN_WALLS,
                advection_scheme="muscl_tvd",
            )

        transported = np.mean(
            solver.sst_turbulent_kinetic_energy.to_numpy(), axis=(0, 1)
        )
        self.assertTrue(np.all(np.isfinite(transported)))
        self.assertTrue(np.all(transported > 0.0))

        exact_mode = np.sin(4.0 * np.pi * (z + transport_time_s))
        interior = (z >= 0.10) & (z <= 0.75)
        centered = transported[interior] - float(np.mean(transported[interior]))
        retained_relative_amplitude = float(
            np.dot(centered, exact_mode[interior])
            / np.dot(exact_mode[interior], exact_mode[interior])
            / (0.2 * float(np.mean(transported[interior])))
        )
        self.assertGreaterEqual(retained_relative_amplitude, 0.95)

    def test_muscl_sst_preserves_uniform_state_with_projection_residual(
        self,
    ) -> None:
        """Finite projection residuals must not create artificial SST energy."""

        cells_z = 8
        dt_s = 0.01
        solver = _cuda_solver(
            grid_nodes=(4, 4, cells_z),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-8,
            dt_s=dt_s,
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        velocity = np.zeros((4, 4, cells_z, 3), dtype=np.float32)
        velocity[..., 2] = -0.02 * (
            np.arange(cells_z, dtype=np.float32)[None, None, :] + 1.0
        ) ** 2
        solver.velocity.from_numpy(velocity)
        for side_index in (0, 1):
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=2,
                side_index=side_index,
                target_velocity_mps=(0.0, 0.0, 0.0),
                active_component_mask=4,
            )
        solver.sst_turbulent_kinetic_energy.fill(2.0)
        solver.sst_specific_dissipation_rate.fill(10.0)
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        solver.sst_strain_rate_magnitude_s.fill(0.0)
        solver.sst_blending_f1.fill(1.0)
        solver.sst_blending_f2.fill(1.0)
        solver.sst_beta.fill(0.0)
        solver.sst_gamma.fill(0.0)
        solver.sst_sigma_k.fill(1.0)
        solver.sst_sigma_omega.fill(1.0)

        with mock.patch.object(
            CartesianFluidSolver,
            "_update_sst_coefficients_from_prepared_inputs_checked",
            # Preserve the former separate max-diffusivity reduction while
            # keeping this test's prescribed coefficient fields frozen.
            new=lambda _solver, molecular_nu: float(molecular_nu),
        ):
            solver.advance_sst_transport(
                dt_s=dt_s,
                kinematic_viscosity_m2_s=1.0e-8,
                no_slip_domain_walls=_OPEN_WALLS,
                advection_scheme="muscl_tvd",
            )

        transported_k = solver.sst_turbulent_kinetic_energy.to_numpy()[:, :, 1:-1]
        transported_omega = solver.sst_specific_dissipation_rate.to_numpy()[:, :, 1:-1]
        self.assertLessEqual(float(np.ptp(transported_k)), 2.0e-6)
        self.assertLessEqual(float(np.ptp(transported_omega)), 1.0e-5)

    def test_muscl_face_ledger_cfl_and_compact_pulse_contracts(self) -> None:
        cells_z = 16
        solver = _cuda_solver(
            grid_nodes=(4, 4, cells_z),
            dt_s=0.25 / cells_z,
        )
        wall_codes = (0, 0, 0, 0, 0, 0)

        # Exact physical-face data must participate in CFL even when every
        # compact cell row is stationary.
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, -4.0),
            active_component_mask=4,
        )
        solver._compute_muscl_momentum_fluxes(solver.velocity, wall_codes)
        exact_face_rate = float(solver._muscl_momentum_advection_rate_kernel())
        self.assertGreaterEqual(exact_face_rate, 4.0 * cells_z * (1.0 - 1.0e-6))

        # Internal normal velocity is the backward-MAC plus-storage row, not
        # an average of adjacent transported cell vectors.
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, 0.0),
            active_component_mask=0,
        )
        velocity = np.zeros((4, 4, cells_z, 3), dtype=np.float32)
        internal_face = cells_z // 2
        velocity[:, :, internal_face, 2] = 3.0
        solver.velocity.from_numpy(velocity)
        solver._compute_muscl_momentum_fluxes(solver.velocity, wall_codes)
        normal_faces = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_allclose(
            normal_faces[:, :, internal_face],
            3.0,
            rtol=0.0,
            atol=1.0e-7,
        )

        # A discontinuous transverse pulse exercises the non-smooth MC branch:
        # conservative transport may sharpen it but must neither create new
        # extrema nor lose volume before the pulse reaches an open boundary.
        velocity.fill(0.0)
        velocity[..., 2] = -1.0
        velocity[:, :, 9:12, 0] = 1.0
        solver.velocity.from_numpy(velocity)
        initial_integral = float(np.sum(velocity[..., 0]))
        for _ in range(4):
            solver.predict(
                dt_s=0.25 / cells_z,
                advection_scheme="muscl_tvd",
                kinematic_viscosity_m2_s=0.0,
            )
        transported = solver.velocity.to_numpy()[..., 0]
        self.assertGreaterEqual(float(np.min(transported)), -2.0e-6)
        self.assertLessEqual(float(np.max(transported)), 1.0 + 2.0e-6)
        self.assertAlmostEqual(
            float(np.sum(transported)),
            initial_integral,
            delta=max(1.0e-5, 2.0e-6 * initial_integral),
        )

    def test_muscl_momentum_reuses_committed_final_flux_for_next_substep(
        self,
    ) -> None:
        solver = _cuda_solver(grid_nodes=(4, 4, 8), dt_s=0.1)
        solver.set_uniform_velocity((0.0, 0.0, 0.0))

        original_flux_builder = (
            CartesianFluidSolver._compute_muscl_momentum_fluxes
        )
        flux_call_count = 0

        def counted_flux_builder(
            active_solver: CartesianFluidSolver,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal flux_call_count
            flux_call_count += 1
            original_flux_builder(active_solver, *args, **kwargs)

        with (
            mock.patch.object(
                CartesianFluidSolver,
                "_compute_muscl_momentum_fluxes",
                new=counted_flux_builder,
            ),
            mock.patch.object(
                CartesianFluidSolver,
                "_muscl_momentum_advection_rate_s",
                new=lambda _solver: 10.0,
            ),
        ):
            solver.predict(
                dt_s=0.1,
                advection_scheme="muscl_tvd",
                kinematic_viscosity_m2_s=0.0,
            )

        substeps = solver._last_momentum_advection_substeps
        self.assertGreater(substeps, 1)
        self.assertEqual(flux_call_count, 2 * substeps + 1)

    def test_muscl_momentum_retries_a_stage_cfl_spike(self) -> None:
        solver = _cuda_solver(grid_nodes=(4, 4, 8), dt_s=0.1)
        solver.set_uniform_velocity((0.0, 0.0, 0.0))

        rate_call_count = 0

        def synthetic_rate() -> float:
            nonlocal rate_call_count
            rate_call_count += 1
            # The first RK intermediate state is deliberately more restrictive
            # than the beginning of the slice.  A long-run controller must roll
            # that trial back and reduce dt instead of terminating the run.
            return 30.0 if rate_call_count == 2 else 10.0

        with mock.patch.object(
            CartesianFluidSolver,
            "_muscl_momentum_advection_rate_s",
            new=lambda _solver: synthetic_rate(),
        ):
            solver.predict(
                dt_s=0.1,
                advection_scheme="muscl_tvd",
                kinematic_viscosity_m2_s=0.0,
            )

        self.assertGreater(rate_call_count, 2)
        self.assertGreaterEqual(solver._last_momentum_advection_substeps, 3)
        self.assertEqual(
            solver._last_momentum_advection_rejected_trial_count,
            1,
        )
        self.assertEqual(solver._last_momentum_advection_requested_time_s, 0.1)
        self.assertEqual(solver._last_momentum_advection_accepted_time_s, 0.1)
        self.assertEqual(
            solver._last_momentum_advection_remaining_unadvanced_time_s,
            0.0,
        )
        self.assertLessEqual(
            solver._last_momentum_advection_max_substep_cfl,
            0.900001,
        )
        # A rejected 0.09-s trial is not accepted physical time.  Force only
        # the committed-slice sum to under-report so the real MUSCL path must
        # fail closed instead of echoing the requested dt in its audit fields.
        with mock.patch(
            "simulation_core.fluids.solver.math.fsum",
            return_value=0.0,
        ):
            with self.assertRaisesRegex(
                FloatingPointError,
                "MUSCL momentum advection physical-time accounting",
            ):
                solver.predict(
                    dt_s=0.1,
                    advection_scheme="muscl_tvd",
                    kinematic_viscosity_m2_s=0.0,
                )
        self.assertEqual(solver._last_momentum_advection_requested_time_s, 0.1)
        self.assertEqual(solver._last_momentum_advection_accepted_time_s, 0.0)
        self.assertEqual(
            solver._last_momentum_advection_remaining_unadvanced_time_s,
            0.1,
        )
        self.assertTrue(np.all(np.isfinite(solver.velocity.to_numpy())))

    def test_muscl_momentum_retries_before_committing_an_invalid_final_stage(
        self,
    ) -> None:
        solver = _cuda_solver(grid_nodes=(4, 4, 8), dt_s=0.1)
        solver.set_uniform_velocity((0.0, 0.0, 0.0))

        rate_call_count = 0

        def synthetic_rate() -> float:
            nonlocal rate_call_count
            rate_call_count += 1
            # Calls one and two are the input and RK intermediate state.  The
            # first trial's completed second stage is invalid.  It must be
            # rejected while the finite transport base is still available.
            return float("inf") if rate_call_count == 3 else 10.0

        with mock.patch.object(
            CartesianFluidSolver,
            "_muscl_momentum_advection_rate_s",
            new=lambda _solver: synthetic_rate(),
        ):
            solver.predict(
                dt_s=0.1,
                advection_scheme="muscl_tvd",
                kinematic_viscosity_m2_s=0.0,
            )

        self.assertGreater(rate_call_count, 3)
        self.assertGreaterEqual(solver._last_momentum_advection_substeps, 3)
        self.assertEqual(
            solver._last_momentum_advection_rejected_trial_count,
            1,
        )
        self.assertEqual(solver._last_momentum_advection_requested_time_s, 0.1)
        self.assertEqual(solver._last_momentum_advection_accepted_time_s, 0.1)
        self.assertEqual(
            solver._last_momentum_advection_remaining_unadvanced_time_s,
            0.0,
        )
        self.assertLessEqual(
            solver._last_momentum_advection_max_substep_cfl,
            0.900001,
        )
        self.assertTrue(np.all(np.isfinite(solver.velocity.to_numpy())))

    def test_muscl_momentum_preserves_constant_transverse_velocity_with_projection_residual(
        self,
    ) -> None:
        """A pressure residual must not become a compressive momentum source.

        The incompressible momentum equation transports velocity in advective
        form.  Conservative face fluxes are equivalent only when the discrete
        face field is exactly solenoidal.  HIBM/projection rows have a finite
        residual, so the production path must include the matching continuity
        correction and preserve a constant transverse component.
        """

        cells_z = 8
        dt_s = 0.01
        solver = _cuda_solver(grid_nodes=(4, 4, cells_z), dt_s=dt_s)
        velocity = np.zeros((4, 4, cells_z, 3), dtype=np.float32)
        velocity[..., 0] = 2.0
        velocity[..., 2] = -0.1 * (
            np.arange(cells_z, dtype=np.float32)[None, None, :] + 1.0
        )
        solver.velocity.from_numpy(velocity)

        solver.predict(
            dt_s=dt_s,
            advection_scheme="muscl_tvd",
            kinematic_viscosity_m2_s=0.0,
        )

        transported = solver.velocity.to_numpy()
        np.testing.assert_allclose(
            transported[:, :, 1:-1, 0],
            2.0,
            rtol=0.0,
            atol=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
