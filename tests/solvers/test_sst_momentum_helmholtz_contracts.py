from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


def _cuda_solver(grid: CartesianGrid) -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1.0,
            viscosity_pa_s=1.0,
            dt_s=1.0,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _unit_grid() -> CartesianGrid:
    return CartesianGrid(
        bounds_min_m=(0.0, 0.0, 0.0),
        cell_widths_x_m=(1.0, 1.0, 1.0, 1.0),
        cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
        cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
    )


def _probe_x_sweep_face_terms(
    solver: ti.template(),
    output: ti.template(),
    i: ti.i32,
    side: ti.i32,
    molecular_nu_m2_s: ti.f32,
    external_no_slip: ti.i32,
):
    terms = solver._sst_momentum_lod_face_terms(
        i,
        1,
        1,
        0,
        side,
        1.0,
        molecular_nu_m2_s,
        external_no_slip,
    )
    for component in ti.static(range(9)):
        output[component] = terms[component]


# ``from __future__ import annotations`` stringifies the probe annotations
# before Taichi 1.7 inspects them.  Restore concrete Taichi types explicitly.
_probe_x_sweep_face_terms.__annotations__ = {
    "solver": ti.template(),
    "output": ti.template(),
    "i": ti.i32,
    "side": ti.i32,
    "molecular_nu_m2_s": ti.f32,
    "external_no_slip": ti.i32,
}
_probe_x_sweep_face_terms = ti.kernel(_probe_x_sweep_face_terms)


class SSTMomentumHelmholtzContracts(unittest.TestCase):
    """Discrete geometry contracts for the variable-viscosity MAC solve."""

    def test_variable_viscosity_normal_mac_uses_physical_face_coefficients(
        self,
    ) -> None:
        solver = _cuda_solver(_unit_grid())
        mu_t_x = np.asarray((1.0, 3.0, 9.0, 27.0), dtype=np.float32)
        solver.sst_eddy_viscosity_pa_s.from_numpy(
            np.broadcast_to(mu_t_x[:, None, None], (4, 4, 4)).copy()
        )
        terms = ti.field(dtype=ti.f32, shape=9)

        _probe_x_sweep_face_terms(solver, terms, 2, -1, 0.0, 0)
        backward = terms.to_numpy()
        _probe_x_sweep_face_terms(solver, terms, 2, 1, 0.0, 0)
        forward = terms.to_numpy()

        # The x component at row i=2 owns the MAC dual cell bounded by scalar
        # centers i-1 and i.  Its two viscosity coefficients therefore use
        # those physical-face values directly.  The transverse components
        # remain cell-centered and retain arithmetic face interpolation.
        for side, observed, expected in (
            ("backward", backward[:3], (3.0, 6.0, 6.0)),
            ("forward", forward[:3], (9.0, 18.0, 18.0)),
        ):
            with self.subTest(side=side):
                np.testing.assert_allclose(
                    observed,
                    np.asarray(expected, dtype=np.float32),
                    rtol=0.0,
                    atol=1.0e-6,
                )

    def test_exact_normal_owner_survives_every_lod_axis_sweep(self) -> None:
        target_mps = 2.0
        for orientation, driven_owner_i in (
            ("fluid_storage", 1),
            ("obstacle_storage", 3),
        ):
            with self.subTest(orientation=orientation):
                solver = _cuda_solver(_unit_grid())
                obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                obstacle[0, :, :] = 1
                obstacle[3, :, :] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")
                solver.sst_eddy_viscosity_pa_s.fill(0.0)

                owner_mask = np.zeros((4, 4, 4), dtype=np.int32)
                owner_mask[1, 1, 1] = 1
                owner_mask[3, 1, 1] = 1
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    owner_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    owner_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    owner_mask
                )

                targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                targets[driven_owner_i, 1, 1, 0] = target_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((4, 4, 4, 3), dtype=np.float32)
                mobility[1, 1, 1, 0] = 0.0
                mobility[3, 1, 1, 0] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((4, 4, 4, 3), dtype=np.float32)
                enforcement[1, 1, 1, 0] = 1.0
                enforcement[3, 1, 1, 0] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                velocity_rhs = np.zeros((4, 4, 4, 3), dtype=np.float32)
                velocity_rhs[driven_owner_i, 1, 1, 0] = target_mps
                solver.velocity.from_numpy(velocity_rhs)
                solver.velocity_prev.fill((0.0, 0.0, 0.0))
                solver.velocity_transport_base.fill((0.0, 0.0, 0.0))
                solver.sst_obstacle_interface_wall_target_component_mask.fill(0)
                solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
                cached_mask = (
                    solver.sst_obstacle_interface_wall_target_component_mask.to_numpy()
                )
                self.assertEqual(
                    int(cached_mask[driven_owner_i, 1, 1, 0]) & 1,
                    1,
                )

                for axis in range(3):
                    solver._sst_momentum_lod_backward_euler_axis_kernel(
                        0.25,
                        1.0,
                        axis,
                        0,
                        0,
                    )
                    with self.subTest(orientation=orientation, after_axis=axis):
                        self.assertAlmostEqual(
                            float(
                                solver.velocity.to_numpy()[
                                    driven_owner_i, 1, 1, 0
                                ]
                            ),
                            target_mps,
                            places=6,
                        )

    def test_moving_normal_ab_owner_is_eliminated_into_unsplit_free_row(
        self,
    ) -> None:
        target_mps = 2.25
        stale_owner_mps = -17.0
        dt_s = 0.25

        for orientation, driven_owner_i in (
            ("fluid_storage", 1),
            ("obstacle_storage", 3),
        ):
            with self.subTest(orientation=orientation):
                solver = _cuda_solver(_unit_grid())
                obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                obstacle[0, :, :] = 1
                obstacle[3, :, :] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")
                solver.sst_eddy_viscosity_pa_s.fill(0.0)

                owner_mask = np.zeros((4, 4, 4), dtype=np.int32)
                owner_mask[1, :, :] = 1
                owner_mask[3, :, :] = 1
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    owner_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    owner_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    owner_mask
                )

                targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                targets[driven_owner_i, 1, 1, 0] = target_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((4, 4, 4, 3), dtype=np.float32)
                mobility[1, :, :, 0] = 0.0
                mobility[3, :, :, 0] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((4, 4, 4, 3), dtype=np.float32)
                enforcement[1, :, :, 0] = 1.0
                enforcement[3, :, :, 0] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                stale_velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
                stale_velocity[driven_owner_i, 1, 1, 0] = stale_owner_mps
                solver.velocity.from_numpy(stale_velocity)
                solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
                solver._compute_muscl_momentum_dual_geometry_kernel()
                solver._initialize_sst_momentum_helmholtz_component_kernel(
                    0, 0, 2
                )

                row_kinds = solver.bicgstab_t.to_numpy()
                row_targets = solver.cg_mg_rhs.to_numpy()
                self.assertEqual(int(row_kinds[driven_owner_i, 1, 1]), 2)
                self.assertAlmostEqual(
                    float(row_targets[driven_owner_i, 1, 1]),
                    target_mps,
                    places=6,
                )
                self.assertNotAlmostEqual(
                    float(row_targets[driven_owner_i, 1, 1]),
                    stale_owner_mps,
                    places=6,
                )

                solver._assemble_sst_momentum_helmholtz_axis_kernel(
                    solver.cg_r_old,
                    0,
                    0,
                    dt_s,
                    1.0,
                    0,
                    0,
                    0,
                    2,
                )

                # Row i=2 is the sole free x-MAC face between the two exact
                # interface rows.  Each unit-area/unit-distance side adds
                # dt*nu=0.25 to its diagonal; only the selected moving owner
                # contributes a nonzero eliminated Dirichlet RHS.
                self.assertEqual(int(row_kinds[2, 1, 1]), 1)
                self.assertAlmostEqual(
                    float(solver.fv_diag.to_numpy()[2, 1, 1]),
                    1.0 + 2.0 * dt_s,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(solver.cg_rhs.to_numpy()[2, 1, 1]),
                    dt_s * target_mps,
                    places=6,
                )

    def test_xmax_wall_uses_normal_mac_and_transverse_distances(self) -> None:
        solver = _cuda_solver(_unit_grid())
        solver.sst_eddy_viscosity_pa_s.fill(0.0)

        exact_mask = np.zeros((2, 4, 4), dtype=np.int32)
        exact_mask[1, 1, 1] = 0b111
        solver.external_velocity_boundary_x_face_active_component_mask.from_numpy(
            exact_mask
        )
        exact_target = np.zeros((2, 4, 4, 3), dtype=np.float32)
        exact_target[1, 1, 1] = (4.0, 5.0, 6.0)
        solver.external_velocity_boundary_x_face_value_mps.from_numpy(exact_target)
        terms = ti.field(dtype=ti.f32, shape=9)

        _probe_x_sweep_face_terms(solver, terms, 3, 1, 1.0, 0)
        observed = terms.to_numpy()

        # At xmax, packed x at row nx-1 owns the last internal normal face;
        # the physical boundary face is one full cell width away.  Packed y/z
        # remain cell-centered and see the usual half-cell wall distance.
        for quantity, actual, expected in (
            ("diagonal", observed[3:6], (1.0, 2.0, 2.0)),
            ("prescribed_rhs", observed[6:9], (4.0, 10.0, 12.0)),
        ):
            with self.subTest(quantity=quantity):
                np.testing.assert_allclose(
                    actual,
                    np.asarray(expected, dtype=np.float32),
                    rtol=0.0,
                    atol=1.0e-6,
                )

    def test_transverse_shared_edge_sums_both_component_half_patches(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 2.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = _cuda_solver(grid)
        mu_t_x = np.asarray((1.0, 3.0, 9.0, 27.0), dtype=np.float32)
        solver.sst_eddy_viscosity_pa_s.from_numpy(
            np.broadcast_to(mu_t_x[:, None, None], (4, 4, 4)).copy()
        )
        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(0)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver._initialize_sst_momentum_helmholtz_component_kernel(0, 0, 2)
        solver._assemble_sst_momentum_helmholtz_axis_kernel(
            solver.cg_mg_residual,
            0,
            1,
            1.0,
            0.0,
            0,
            0,
            0,
            2,
        )

        observed = float(solver.cg_mg_residual.to_numpy()[2, 1, 1])
        # x-MAC at i=2 contains half of scalar cells i=1 and i=2.  Across
        # this y edge their areas are 0.5*2 and 0.5*4, with nu 3 and 9.
        expected = 0.5 * 2.0 * 3.0 + 0.5 * 4.0 * 9.0
        self.assertAlmostEqual(observed, expected, places=6)

    def test_z_component_x_edge_uses_y_as_the_remaining_face_width(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_y_m=(1.0, 2.0, 3.0, 4.0),
            cell_widths_z_m=(1.0, 2.0, 4.0, 8.0),
        )
        solver = _cuda_solver(grid)
        mu_t_z = np.asarray((1.0, 3.0, 9.0, 27.0), dtype=np.float32)
        solver.sst_eddy_viscosity_pa_s.from_numpy(
            np.broadcast_to(mu_t_z[None, None, :], (4, 4, 4)).copy()
        )
        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(0)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver._initialize_sst_momentum_helmholtz_component_kernel(2, 0, 2)
        solver._assemble_sst_momentum_helmholtz_axis_kernel(
            solver.cg_r_old,
            2,
            0,
            1.0,
            0.0,
            0,
            0,
            0,
            2,
        )

        observed = float(solver.cg_r_old.to_numpy()[1, 1, 2])
        expected = (0.5 * 2.0 * 3.0 + 0.5 * 4.0 * 9.0) * 2.0
        self.assertAlmostEqual(observed, expected, places=6)


if __name__ == "__main__":
    unittest.main()
