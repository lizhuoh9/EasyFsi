from __future__ import annotations

from collections.abc import Mapping
import unittest
from unittest import mock

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from simulation_core.fluids import CartesianGrid


_OPEN_WALLS = (False, False, False, False, False, False)


def _cuda_solver(spec: FluidDomainSpec) -> CartesianFluidSolver:
    return CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cuda"))


def _report_value(report: object, name: str) -> object:
    """Read a transport diagnostic without prescribing its report container."""

    if isinstance(report, Mapping):
        return report[name]
    return getattr(report, name)


class CartesianFluidSSTTransportContracts(unittest.TestCase):
    """End-to-end contracts for SST state owned by the Cartesian solver."""

    def test_laminar_is_default_and_sst_requires_explicit_physical_initialization(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.2,
                viscosity_pa_s=1.8e-5,
                dt_s=1.0e-3,
            )
        )

        self.assertEqual(solver.turbulence_model, "laminar")
        solver.set_uniform_velocity((3.0, -0.25, 0.5))
        laminar_before = solver.velocity.to_numpy().copy()
        solver.predict(kinematic_viscosity_m2_s=0.0)
        np.testing.assert_allclose(
            solver.velocity.to_numpy(),
            laminar_before,
            rtol=0.0,
            atol=1.0e-6,
        )

        solver.configure_sst_2003(
            inlet_velocity_mps=10.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        self.assertEqual(solver.turbulence_model, "sst_2003")
        expected_k_m2_s2 = 1.5 * (10.0 * 0.05) ** 2
        expected_mu_t_pa_s = 10.0 * 1.8e-5
        expected_omega_s = 1.2 * expected_k_m2_s2 / expected_mu_t_pa_s
        np.testing.assert_allclose(
            solver.sst_turbulent_kinetic_energy.to_numpy(),
            expected_k_m2_s2,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            solver.sst_specific_dissipation_rate.to_numpy(),
            expected_omega_s,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            solver.sst_eddy_viscosity_pa_s.to_numpy(),
            expected_mu_t_pa_s,
            rtol=2.0e-6,
        )

    def test_predictor_consumes_spatial_eddy_viscosity_not_its_scalar_mean(
        self,
    ) -> None:
        spec = FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-6,
            dt_s=5.0e-3,
        )
        near_gradient = _cuda_solver(spec)
        away_from_gradient = _cuda_solver(spec)
        for solver in (near_gradient, away_from_gradient):
            solver.configure_sst_2003(
                inlet_velocity_mps=1.0,
                turbulence_intensity=0.05,
                turbulent_viscosity_ratio=10.0,
                no_slip_domain_walls=_OPEN_WALLS,
            )

        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[1, 1, 1, 0] = 1.0
        near_gradient.velocity.from_numpy(velocity)
        away_from_gradient.velocity.from_numpy(velocity)

        local_mu_t = np.zeros((4, 4, 4), dtype=np.float32)
        remote_mu_t = np.zeros_like(local_mu_t)
        local_mu_t[1, 1, 1] = 0.5
        remote_mu_t[3, 3, 3] = 0.5
        self.assertAlmostEqual(float(np.mean(local_mu_t)), float(np.mean(remote_mu_t)))
        near_gradient.sst_eddy_viscosity_pa_s.from_numpy(local_mu_t)
        away_from_gradient.sst_eddy_viscosity_pa_s.from_numpy(remote_mu_t)

        near_gradient.predict(dt_s=5.0e-3, kinematic_viscosity_m2_s=0.0)
        away_from_gradient.predict(dt_s=5.0e-3, kinematic_viscosity_m2_s=0.0)

        local_result = near_gradient.velocity.to_numpy()
        remote_result = away_from_gradient.velocity.to_numpy()
        self.assertLess(
            float(local_result[1, 1, 1, 0]),
            float(remote_result[1, 1, 1, 0]) - 1.0e-4,
        )
        self.assertGreater(float(np.max(np.abs(local_result - remote_result))), 1.0e-4)

    def test_sst_stiff_momentum_diffusion_is_implicit_not_a_substep_cap(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        x = solver.cell_center_x_m.to_numpy()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 1] = np.sin(np.pi * x)[:, None, None]
        solver.velocity.from_numpy(velocity)
        solver.sst_turbulent_kinetic_energy.fill(0.25)
        solver.sst_specific_dissipation_rate.fill(1.0)
        solver.sst_eddy_viscosity_pa_s.fill(100.0)
        solver._sst_max_automatic_substeps = 2
        energy_before = float(np.sum(velocity * velocity))

        solver.predict(
            dt_s=1.0e-2,
            kinematic_viscosity_m2_s=1.0e-5,
            no_slip_domain_walls=_OPEN_WALLS,
            advection_scheme="muscl_tvd",
        )

        velocity_after = solver.velocity.to_numpy()
        self.assertEqual(
            solver._sst_last_momentum_diffusion_integrator,
            "unsplit_volume_symmetric_pcg_jacobi_frozen_coefficients",
        )
        self.assertEqual(solver._sst_last_momentum_diffusion_substeps, 1)
        self.assertGreater(solver._sst_last_momentum_diffusion_cfl, 90.0)
        self.assertTrue(solver._sst_last_momentum_helmholtz_converged)
        self.assertLessEqual(
            solver._sst_last_momentum_helmholtz_relative_residual,
            1.0e-7,
        )
        self.assertTrue(np.all(np.isfinite(velocity_after)))
        self.assertLess(float(np.sum(velocity_after * velocity_after)), energy_before)

    def test_failed_unsplit_momentum_trial_rolls_back_the_whole_velocity_field(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-3,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        initial_velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        initial_velocity[..., 0] = np.linspace(
            0.0, 0.3, 4, dtype=np.float32
        )[:, None, None]
        initial_velocity[..., 1] = -0.125
        solver.velocity.from_numpy(initial_velocity)
        solver._sst_max_automatic_substeps = 1

        transaction_bases: list[np.ndarray] = []
        unsplit_calls: list[dict[str, object]] = []
        lod_calls: list[tuple[object, ...]] = []

        def fail_unsplit(
            active_solver: CartesianFluidSolver,
            **kwargs: object,
        ) -> dict[str, object]:
            transaction_bases.append(
                active_solver.velocity_transport_base.to_numpy().copy()
            )
            unsplit_calls.append(dict(kwargs))
            # Model a failed component scatter corrupting every component and
            # every row.  ``predict`` must restore the transaction base before
            # exposing the failure to its caller.
            active_solver.velocity.fill((101.0, -202.0, 303.0))
            return {
                "converged": False,
                "iterations": 7,
                "relative_residual": 0.25,
                "components": [
                    {
                        "component": 0,
                        "converged": False,
                        "iterations": 7,
                        "relative_residual": 0.25,
                        "breakdown": "invalid_or_nonpositive_diagonal",
                    }
                ],
            }

        def forbid_lod(*args: object, **kwargs: object) -> None:
            lod_calls.append((*args, kwargs))
            raise AssertionError("failed unsplit solve must not fall back to LOD")

        with (
            mock.patch.object(
                CartesianFluidSolver,
                "_solve_sst_momentum_unsplit_helmholtz",
                new=fail_unsplit,
            ),
            mock.patch.object(
                CartesianFluidSolver,
                "_sst_momentum_lod_backward_euler_axis_kernel",
                new=forbid_lod,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "transactional retries"):
                solver.predict(
                    dt_s=1.0e-3,
                    kinematic_viscosity_m2_s=1.0e-5,
                    no_slip_domain_walls=_OPEN_WALLS,
                    advection_scheme="rk2",
                )

        self.assertEqual(len(unsplit_calls), 1)
        self.assertEqual(lod_calls, [])
        self.assertEqual(len(transaction_bases), 1)
        np.testing.assert_array_equal(
            solver.velocity.to_numpy(),
            transaction_bases[0],
        )
        self.assertEqual(
            solver._sst_last_momentum_diffusion_integrator,
            "unsplit_volume_symmetric_pcg_jacobi_frozen_coefficients",
        )
        self.assertEqual(solver._sst_last_momentum_diffusion_substeps, 1)
        self.assertGreater(solver._sst_last_momentum_diffusion_cfl, 0.0)
        self.assertFalse(solver._sst_last_momentum_helmholtz_converged)
        self.assertEqual(solver._sst_last_momentum_helmholtz_iterations, 7)
        self.assertEqual(solver._sst_last_momentum_helmholtz_iterations_total, 7)
        self.assertEqual(
            solver._sst_last_momentum_helmholtz_relative_residual,
            0.25,
        )
        self.assertEqual(
            solver._sst_last_momentum_helmholtz_rejected_trial_count,
            1,
        )

    def test_nonfatal_unsplit_trial_retries_from_transaction_base_and_commits(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-3,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        initial_velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        initial_velocity[..., 0] = np.linspace(
            0.0, 0.3, 4, dtype=np.float32
        )[:, None, None]
        initial_velocity[..., 1] = -0.125
        solver.velocity.from_numpy(initial_velocity)
        solver._sst_max_automatic_substeps = 2

        trial_dt_s: list[float] = []
        trial_states: list[np.ndarray] = []
        transaction_base: list[np.ndarray] = []
        lod_calls: list[tuple[object, ...]] = []

        def no_explicit_stress(*args: object, **kwargs: object) -> None:
            del args, kwargs

        def solve_with_one_retry(
            active_solver: CartesianFluidSolver,
            **kwargs: object,
        ) -> dict[str, object]:
            dt_s = float(kwargs["dt_s"])
            trial_dt_s.append(dt_s)
            current = active_solver.velocity.to_numpy().copy()
            trial_states.append(current)
            if not transaction_base:
                transaction_base.append(
                    active_solver.velocity_transport_base.to_numpy().copy()
                )
                active_solver.velocity.fill((101.0, -202.0, 303.0))
                return {
                    "converged": False,
                    "iterations": 5,
                    "relative_residual": 0.2,
                    "components": [
                        {
                            "component": 0,
                            "converged": False,
                            "iterations": 5,
                            "relative_residual": 0.2,
                            "breakdown": "iteration_limit",
                        }
                    ],
                }

            committed = current + np.float32(1.0)
            active_solver.velocity.from_numpy(committed)
            return {
                "converged": True,
                "iterations": 3,
                "relative_residual": 1.0e-8,
                "components": [
                    {"component": 0, "converged": True, "iterations": 1},
                    {"component": 1, "converged": True, "iterations": 2},
                    {"component": 2, "converged": True, "iterations": 3},
                ],
            }

        def forbid_lod(*args: object, **kwargs: object) -> None:
            lod_calls.append((*args, kwargs))
            raise AssertionError("transactional retry must not call the LOD solver")

        with (
            mock.patch.object(
                CartesianFluidSolver,
                "_sst_momentum_explicit_stress_rhs_checked",
                new=no_explicit_stress,
            ),
            mock.patch.object(
                CartesianFluidSolver,
                "_sst_momentum_transpose_stress_checked",
                new=no_explicit_stress,
            ),
            mock.patch.object(
                CartesianFluidSolver,
                "_solve_sst_momentum_unsplit_helmholtz",
                new=solve_with_one_retry,
            ),
            mock.patch.object(
                CartesianFluidSolver,
                "_sst_momentum_lod_backward_euler_axis_kernel",
                new=forbid_lod,
            ),
        ):
            solver.predict(
                dt_s=1.0e-3,
                kinematic_viscosity_m2_s=1.0e-5,
                no_slip_domain_walls=_OPEN_WALLS,
                advection_scheme="rk2",
            )

        np.testing.assert_allclose(
            trial_dt_s,
            [1.0e-3, 5.0e-4, 5.0e-4],
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertEqual(lod_calls, [])
        self.assertEqual(len(transaction_base), 1)
        expected_after_first_commit = (
            transaction_base[0] + np.float32(1.0)
        ).astype(np.float32)
        expected_after_second_commit = (
            expected_after_first_commit + np.float32(1.0)
        ).astype(np.float32)
        np.testing.assert_array_equal(trial_states[0], transaction_base[0])
        np.testing.assert_array_equal(trial_states[1], transaction_base[0])
        np.testing.assert_array_equal(
            trial_states[2],
            expected_after_first_commit,
        )
        np.testing.assert_array_equal(
            solver.velocity.to_numpy(),
            expected_after_second_commit,
        )
        self.assertEqual(
            solver._sst_last_momentum_diffusion_integrator,
            "unsplit_volume_symmetric_pcg_jacobi_frozen_coefficients",
        )
        self.assertEqual(solver._sst_last_momentum_diffusion_substeps, 2)
        self.assertTrue(solver._sst_last_momentum_helmholtz_converged)
        self.assertEqual(solver._sst_last_momentum_helmholtz_iterations, 5)
        self.assertEqual(solver._sst_last_momentum_helmholtz_iterations_total, 17)
        self.assertAlmostEqual(
            solver._sst_last_momentum_helmholtz_relative_residual,
            1.0e-8,
        )
        self.assertEqual(
            solver._sst_last_momentum_helmholtz_rejected_trial_count,
            1,
        )

    def test_sst_momentum_helmholtz_solve_is_unsplit_for_a_2d_neumann_mode(
        self,
    ) -> None:
        """Recover a manufactured mode of the full multidimensional operator.

        A sequential backward-Euler LOD factorization solves
        ``(I-dt*Ly)(I-dt*Lz) u = rhs`` and therefore adds a spurious
        ``dt**2*Ly*Lz`` term.  This fixture chooses comparable y/z diffusion
        numbers so that the old one-pass factorization attenuates the exact
        response by roughly 24 percent even though every directional solve is
        individually converged.
        """

        grid_nodes = (4, 8, 8)
        dt_s = 0.1
        nu_m2_s = 1.0
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=grid_nodes,
                density_kgm3=1.0,
                viscosity_pa_s=nu_m2_s,
                dt_s=dt_s,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.sst_eddy_viscosity_pa_s.fill(0.0)

        y_m = solver.cell_center_y_m.to_numpy()
        z_m = solver.cell_center_z_m.to_numpy()
        exact_mode = (
            np.cos(np.pi * y_m)[:, None]
            * np.cos(np.pi * z_m)[None, :]
        )
        dy_m = float(solver.cell_width_y_m.to_numpy()[0])
        dz_m = float(solver.cell_width_z_m.to_numpy()[0])
        lambda_y_s = 4.0 * np.sin(np.pi / (2.0 * grid_nodes[1])) ** 2 / dy_m**2
        lambda_z_s = 4.0 * np.sin(np.pi / (2.0 * grid_nodes[2])) ** 2 / dz_m**2
        directional_y = dt_s * nu_m2_s * lambda_y_s
        directional_z = dt_s * nu_m2_s * lambda_z_s
        old_factorized_response_ratio = (
            1.0 + directional_y + directional_z
        ) / ((1.0 + directional_y) * (1.0 + directional_z))
        self.assertLess(old_factorized_response_ratio, 0.8)

        rhs_scale = 1.0 + directional_y + directional_z
        rhs = np.zeros((*grid_nodes, 3), dtype=np.float32)
        rhs[..., 0] = rhs_scale * exact_mode[None, :, :]
        solver.velocity.from_numpy(rhs)

        report = solver._solve_sst_momentum_unsplit_helmholtz(
            dt_s=dt_s,
            molecular_nu_m2_s=nu_m2_s,
            wall_flag_codes=(0, 0, 0, 0, 0, 0),
            relative_tolerance=2.0e-6,
            max_iterations=256,
        )

        solved = solver.velocity.to_numpy()[..., 0]
        exact = np.broadcast_to(exact_mode[None, :, :], solved.shape)
        relative_l2 = float(np.linalg.norm(solved - exact) / np.linalg.norm(exact))
        self.assertTrue(bool(report["converged"]))
        self.assertLess(float(report["relative_residual"]), 2.0e-6)
        self.assertLess(relative_l2, 2.0e-5)

    def test_sst_momentum_lod_x_axis_preserves_transverse_cell_cv_geometry(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 2.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0,
                dt_s=1.0,
                cartesian_grid=grid,
            )
        )
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        expected_solution = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float32)
        transverse_rhs = np.asarray(
            (0.0, -1.0 / 6.0, 9.0 / 8.0, -1.0 / 48.0),
            dtype=np.float32,
        )
        velocity_rhs = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity_rhs[..., 1] = transverse_rhs[:, None, None]
        solver.velocity.from_numpy(velocity_rhs)

        solver._sst_momentum_lod_backward_euler_axis_kernel(
            1.0,
            1.0,
            0,
            0,
            0,
        )

        np.testing.assert_allclose(
            solver.velocity.to_numpy()[:, 1, 1, 1],
            expected_solution,
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_sst_momentum_lod_x_axis_uses_normal_face_dual_cv_geometry(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 2.0, 4.0, 8.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0,
                dt_s=1.0,
                cartesian_grid=grid,
            )
        )
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        expected_solution = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float32)
        normal_rhs = np.asarray(
            (0.0, -1.0 / 3.0, 5.0 / 4.0, -1.0 / 24.0),
            dtype=np.float32,
        )
        velocity_rhs = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity_rhs[..., 0] = normal_rhs[:, None, None]
        solver.velocity.from_numpy(velocity_rhs)

        solver._sst_momentum_lod_backward_euler_axis_kernel(
            1.0,
            1.0,
            0,
            0,
            0,
        )

        np.testing.assert_allclose(
            solver.velocity.to_numpy()[:, 1, 1, 0],
            expected_solution,
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_sst_momentum_lod_preserves_canonical_normal_owner_both_orientations(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_y_m=(1.0, 1.0, 1.0, 1.0),
            cell_widths_z_m=(1.0, 1.0, 1.0, 1.0),
        )
        target_mps = 2.0
        dt_s = 0.25
        wall_coupling = dt_s
        expected_free_face_mps = (
            wall_coupling * target_mps / (1.0 + 2.0 * wall_coupling)
        )

        for orientation, driven_owner_i in (
            ("fluid_storage", 1),
            ("obstacle_storage", 3),
        ):
            with self.subTest(orientation=orientation):
                solver = _cuda_solver(
                    FluidDomainSpec(
                        bounds_min_m=grid.bounds_min_m,
                        bounds_max_m=grid.bounds_max_m,
                        grid_nodes=None,
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0,
                        dt_s=dt_s,
                        cartesian_grid=grid,
                    )
                )
                obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                obstacle[0, :, :] = 1
                obstacle[3, :, :] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")
                solver.sst_eddy_viscosity_pa_s.fill(0.0)

                x_mask = np.zeros((4, 4, 4), dtype=np.int32)
                x_mask[1, :, :] = 1
                x_mask[3, :, :] = 1
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    x_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    x_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    x_mask
                )
                targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                targets[driven_owner_i, :, :, 0] = target_mps
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

                velocity_rhs = np.zeros((4, 4, 4, 3), dtype=np.float32)
                velocity_rhs[driven_owner_i, :, :, 0] = target_mps
                solver.velocity.from_numpy(velocity_rhs)
                solver.velocity_prev.fill((0.0, 0.0, 0.0))
                solver.velocity_transport_base.fill((0.0, 0.0, 0.0))
                solver.sst_obstacle_interface_wall_target_component_mask.fill(0)
                solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
                owner_masks = (
                    solver.sst_obstacle_interface_wall_target_component_mask.to_numpy()
                )
                self.assertEqual(int(owner_masks[1, 1, 1, 0]) & 1, 1)
                self.assertEqual(int(owner_masks[3, 1, 1, 0]) & 1, 1)

                solver._sst_momentum_lod_backward_euler_axis_kernel(
                    dt_s,
                    1.0,
                    0,
                    0,
                    0,
                )

                expected_line = np.zeros(4, dtype=np.float32)
                expected_line[driven_owner_i] = target_mps
                expected_line[2] = expected_free_face_mps
                np.testing.assert_allclose(
                    solver.velocity.to_numpy()[:, 1, 1, 0],
                    expected_line,
                    rtol=0.0,
                    atol=2.0e-6,
                )

    def test_predictor_uses_full_symmetric_boussinesq_stress(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-6,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        centers_x = solver.cell_center_x_m.to_numpy()
        centers_y = solver.cell_center_y_m.to_numpy()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 1] = centers_x[:, None, None] * centers_y[None, :, None]
        solver.velocity.from_numpy(velocity)
        solver.sst_turbulent_kinetic_energy.fill(0.25)
        solver.sst_eddy_viscosity_pa_s.fill(0.1)

        solver.predict(dt_s=1.0e-4, kinematic_viscosity_m2_s=0.0)

        # div(nu_eff * grad(u)) has no x component for u=(0, x*y, 0),
        # while div(nu_eff * grad(u)^T)=grad(div(u)) accelerates x.
        self.assertGreater(
            float(np.max(np.abs(solver.velocity.to_numpy()[1:3, 1:3, 1:3, 0]))),
            1.0e-6,
        )

    def test_sst_strain_uses_true_mac_faces_for_oblique_quadratic_shear(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        centers_x = solver.cell_center_x_m.to_numpy().astype(np.float64)
        centers_y = solver.cell_center_y_m.to_numpy().astype(np.float64)
        centers_z = solver.cell_center_z_m.to_numpy().astype(np.float64)
        widths_x = solver.cell_width_x_m.to_numpy().astype(np.float64)
        widths_y = solver.cell_width_y_m.to_numpy().astype(np.float64)
        widths_z = solver.cell_width_z_m.to_numpy().astype(np.float64)
        minus_faces_x = centers_x - 0.5 * widths_x
        minus_faces_y = centers_y - 0.5 * widths_y
        minus_faces_z = centers_z - 0.5 * widths_z

        normal = np.asarray((2.0, 1.0, 0.0), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        tangent = np.asarray((-1.0, 2.0, 0.0), dtype=np.float64)
        tangent /= np.linalg.norm(tangent)
        self.assertAlmostEqual(float(np.dot(normal, tangent)), 0.0)

        target = (3, 3, 2)
        phase_origin_m = float(
            normal[0] * centers_x[target[0]]
            + normal[1] * centers_y[target[1]]
            + normal[2] * centers_z[target[2]]
        )
        amplitude_per_m_s = 64.0
        velocity = np.zeros((8, 8, 4, 3), dtype=np.float32)

        # The compact vector stores each component on its own negative MAC face.
        xi_x = (
            normal[0] * minus_faces_x[:, None, None]
            + normal[1] * centers_y[None, :, None]
            + normal[2] * centers_z[None, None, :]
            - phase_origin_m
        )
        xi_y = (
            normal[0] * centers_x[:, None, None]
            + normal[1] * minus_faces_y[None, :, None]
            + normal[2] * centers_z[None, None, :]
            - phase_origin_m
        )
        xi_z = (
            normal[0] * centers_x[:, None, None]
            + normal[1] * centers_y[None, :, None]
            + normal[2] * minus_faces_z[None, None, :]
            - phase_origin_m
        )
        velocity[..., 0] = (
            tangent[0] * amplitude_per_m_s * xi_x * xi_x
        ).astype(np.float32)
        velocity[..., 1] = (
            tangent[1] * amplitude_per_m_s * xi_y * xi_y
        ).astype(np.float32)
        velocity[..., 2] = (
            tangent[2] * amplitude_per_m_s * xi_z * xi_z
        ).astype(np.float32)
        solver.velocity.from_numpy(velocity)

        solver._update_sst_coefficients_checked(1.0e-5)

        strain_s = solver.sst_strain_rate_magnitude_s.to_numpy()
        target_strain_s = float(strain_s[target])
        # U=A*xi^2 has dU/dxi=0 at this scalar-cell center, so its exact
        # symmetric strain and the corresponding SST production are both zero.
        self.assertLess(abs(target_strain_s), 1.0e-5)
        nu_t_m2_s = float(solver.sst_eddy_viscosity_pa_s.to_numpy()[target])
        raw_production_m2_s3 = nu_t_m2_s * target_strain_s * target_strain_s
        self.assertLess(abs(raw_production_m2_s3), 1.0e-10)

    def test_sst_mac_reconstruction_preserves_constant_and_linear_shear(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(6, 6, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        solver.set_uniform_velocity((1.25, -0.75, 0.5))
        solver._update_sst_coefficients_checked(1.0e-5)
        self.assertLess(
            float(np.max(np.abs(solver.sst_strain_rate_magnitude_s.to_numpy()))),
            1.0e-6,
        )

        shear_rate_s = 3.5
        centers_x = solver.cell_center_x_m.to_numpy()
        velocity = np.zeros((6, 6, 4, 3), dtype=np.float32)
        # v lives on y-normal faces; because this shear varies only with x,
        # its true face x coordinate is the scalar-cell x center.
        velocity[..., 1] = shear_rate_s * centers_x[:, None, None]
        solver.velocity.from_numpy(velocity)
        solver._update_sst_coefficients_checked(1.0e-5)
        self.assertAlmostEqual(
            float(solver.sst_strain_rate_magnitude_s.to_numpy()[3, 3, 2]),
            shear_rate_s,
            places=5,
        )

    def test_sst_normal_strain_uses_same_cell_mac_face_jump(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(7, 6, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        target = (3, 3, 2)
        face_jump_mps = 2.0
        velocity = np.zeros((7, 6, 4, 3), dtype=np.float32)
        # Packed backward MAC ownership: row i owns the scalar cell's x-minus
        # face and row i+1 owns its x-plus face.  A normal derivative must use
        # this same-cell face pair directly; averaging to centers first is a
        # different, smoothing operator for non-polynomial/local face states.
        velocity[target[0] + 1, target[1], target[2], 0] = face_jump_mps
        solver.velocity.from_numpy(velocity)

        solver._update_sst_coefficients_checked(1.0e-5)

        cell_width_m = float(solver.cell_width_x_m.to_numpy()[target[0]])
        expected_strain_s = np.sqrt(2.0) * face_jump_mps / cell_width_m
        measured_strain_s = float(
            solver.sst_strain_rate_magnitude_s.to_numpy()[target]
        )
        self.assertAlmostEqual(
            measured_strain_s,
            expected_strain_s,
            places=4,
        )

    def test_sst_report_exposes_graded_volume_mean_and_rms_state(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.15, 0.20, 0.25, 0.40),
            cell_widths_z_m=(0.1, 0.2, 0.3, 0.4),
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-5,
                cartesian_grid=grid,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        k_state = np.broadcast_to(
            np.asarray((0.1, 0.4, 0.9, 1.6), dtype=np.float32)[:, None, None],
            (4, 4, 4),
        ).copy()
        omega_state = np.broadcast_to(
            np.asarray((10.0, 20.0, 40.0, 80.0), dtype=np.float32)[:, None, None],
            (4, 4, 4),
        ).copy()
        solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
        solver.sst_specific_dissipation_rate.from_numpy(omega_state)

        report = solver.advance_sst_transport(
            dt_s=1.0e-5,
            kinematic_viscosity_m2_s=1.0e-5,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        volume_m3 = (
            solver.cell_width_x_m.to_numpy().astype(np.float64)[:, None, None]
            * solver.cell_width_y_m.to_numpy().astype(np.float64)[None, :, None]
            * solver.cell_width_z_m.to_numpy().astype(np.float64)[None, None, :]
        )
        total_volume_m3 = float(np.sum(volume_m3, dtype=np.float64))
        fields_and_report_names = (
            (
                solver.sst_turbulent_kinetic_energy.to_numpy(),
                "turbulent_kinetic_energy_volume_mean_m2_s2",
                "turbulent_kinetic_energy_volume_rms_m2_s2",
            ),
            (
                solver.sst_specific_dissipation_rate.to_numpy(),
                "specific_dissipation_rate_volume_mean_s",
                "specific_dissipation_rate_volume_rms_s",
            ),
            (
                solver.sst_eddy_viscosity_pa_s.to_numpy(),
                "eddy_viscosity_volume_mean_pa_s",
                "eddy_viscosity_volume_rms_pa_s",
            ),
        )
        for state, mean_name, rms_name in fields_and_report_names:
            state_f64 = state.astype(np.float64)
            expected_mean = float(
                np.sum(state_f64 * volume_m3, dtype=np.float64)
                / total_volume_m3
            )
            expected_rms = float(
                np.sqrt(
                    np.sum(state_f64 * state_f64 * volume_m3, dtype=np.float64)
                    / total_volume_m3
                )
            )
            self.assertAlmostEqual(float(report[mean_name]), expected_mean, places=6)
            self.assertAlmostEqual(float(report[rms_name]), expected_rms, places=6)

    def test_sst_moving_obstacle_face_target_is_side_symmetric_and_not_stale(
        self,
    ) -> None:
        wall_velocity_mps = 1.5
        opposite_face_velocity_mps = 0.25
        target_fluid_i = 2
        measured_strain_s: list[float] = []

        for obstacle_side, stale_storage_velocity_mps in ((-1, 23.0), (1, -31.0)):
            with self.subTest(obstacle_side=obstacle_side):
                solver = _cuda_solver(
                    FluidDomainSpec.unit_box(
                        grid_nodes=(5, 4, 4),
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0e-5,
                        dt_s=1.0e-4,
                    )
                )
                obstacle_i = target_fluid_i + obstacle_side
                obstacle = np.zeros((5, 4, 4), dtype=np.int32)
                obstacle[obstacle_i, :, :] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.configure_sst_2003(
                    inlet_velocity_mps=1.0,
                    turbulence_intensity=0.05,
                    turbulent_viscosity_ratio=10.0,
                    no_slip_domain_walls=_OPEN_WALLS,
                )
                solver.set_velocity_dirichlet_boundary_authority("canonical")

                # A backward-MAC interface face belongs to the plus-side row:
                # the fluid row when the obstacle is on the minus side, and
                # the obstacle row when the obstacle is on the plus side.
                storage_i = (
                    target_fluid_i if obstacle_side < 0 else obstacle_i
                )
                velocity = np.zeros((5, 4, 4, 3), dtype=np.float32)
                velocity[storage_i, :, :, 0] = stale_storage_velocity_mps
                if obstacle_side < 0:
                    velocity[target_fluid_i + 1, :, :, 0] = (
                        opposite_face_velocity_mps
                    )
                else:
                    velocity[target_fluid_i, :, :, 0] = (
                        opposite_face_velocity_mps
                    )
                solver.velocity.from_numpy(velocity)

                active_mask = np.zeros((5, 4, 4), dtype=np.int32)
                hard_mask = np.zeros_like(active_mask)
                owned_mask = np.zeros_like(active_mask)
                active_mask[storage_i, :, :] = 1
                hard_mask[storage_i, :, :] = 1
                owned_mask[storage_i, :, :] = 1
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    active_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    hard_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    owned_mask
                )

                targets = np.zeros((5, 4, 4, 3), dtype=np.float32)
                targets[storage_i, :, :, 0] = wall_velocity_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((5, 4, 4, 3), dtype=np.float32)
                mobility[storage_i, :, :, 0] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((5, 4, 4, 3), dtype=np.float32)
                enforcement[storage_i, :, :, 0] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                solver._update_sst_coefficients_checked(1.0e-5)
                strain_s = float(
                    solver.sst_strain_rate_magnitude_s.to_numpy()[
                        target_fluid_i, 1, 1
                    ]
                )
                measured_strain_s.append(strain_s)

                half_cell_width_m = 0.5 * float(
                    solver.cell_width_x_m.to_numpy()[target_fluid_i]
                )
                center_velocity_mps = 0.5 * (
                    wall_velocity_mps + opposite_face_velocity_mps
                )
                expected_strain_s = np.sqrt(2.0) * abs(
                    wall_velocity_mps - center_velocity_mps
                ) / half_cell_width_m
                self.assertAlmostEqual(
                    strain_s,
                    expected_strain_s,
                    places=4,
                )

        self.assertAlmostEqual(
            measured_strain_s[0],
            measured_strain_s[1],
            places=5,
        )

    def test_sst_canonical_x_interface_uses_same_owner_row_tangential_target(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(5, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        fluid_i = 2
        storage_i = 3
        obstacle = np.zeros((5, 4, 4), dtype=np.int32)
        obstacle[storage_i, :, :] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        solver.set_uniform_velocity((0.0, 0.0, 0.0))

        # Bit x proves that this plus-side storage row owns the physical
        # x-normal obstacle face.  Bit y supplies the tangential component of
        # that same wall-velocity vector; it must not be reinterpreted as a
        # separate y-normal interface claim.
        xy_mask = np.zeros((5, 4, 4), dtype=np.int32)
        xy_mask[storage_i, :, :] = 3
        solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
            xy_mask
        )
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            xy_mask
        )
        solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
            xy_mask
        )
        tangential_wall_velocity_mps = 2.0
        targets = np.zeros((5, 4, 4, 3), dtype=np.float32)
        targets[storage_i, :, :, 1] = tangential_wall_velocity_mps
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
        mobility = np.ones((5, 4, 4, 3), dtype=np.float32)
        mobility[storage_i, :, :, :2] = 0.0
        solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(mobility)
        enforcement = np.zeros((5, 4, 4, 3), dtype=np.float32)
        enforcement[storage_i, :, :, :2] = 1.0
        solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
            enforcement
        )

        solver._update_sst_coefficients_checked(1.0e-5)

        half_cell_width_m = 0.5 * float(
            solver.cell_width_x_m.to_numpy()[fluid_i]
        )
        expected_strain_s = tangential_wall_velocity_mps / half_cell_width_m
        self.assertAlmostEqual(
            float(solver.sst_strain_rate_magnitude_s.to_numpy()[fluid_i, 1, 1]),
            expected_strain_s,
            places=5,
        )

    def test_sst_external_tangential_target_uses_componentwise_half_cell_strain(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        tangential_wall_velocity_mps = 2.0
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=0,
            side_index=0,
            # x is an inactive lure: bit 2 owns only the y component.
            target_velocity_mps=(91.0, tangential_wall_velocity_mps, -73.0),
            active_component_mask=2,
        )

        solver._update_sst_coefficients_checked(1.0e-5)

        half_cell_width_m = 0.5 * float(solver.cell_width_x_m.to_numpy()[0])
        expected_strain_s = tangential_wall_velocity_mps / half_cell_width_m
        strain_s = solver.sst_strain_rate_magnitude_s.to_numpy()
        self.assertAlmostEqual(
            float(strain_s[0, 1, 1]),
            expected_strain_s,
            places=5,
        )
        self.assertLess(abs(float(strain_s[1, 1, 1])), 1.0e-6)

    def test_sst_grad_k_rhs_uses_backward_mac_face_on_graded_grid(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4, 0.5),
            cell_widths_y_m=(0.25,) * 4,
            cell_widths_z_m=(0.25,) * 4,
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
                cartesian_grid=grid,
            )
        )
        centers_x = solver.cell_center_x_m.to_numpy().astype(np.float64)
        k_state = np.broadcast_to(
            centers_x[:, None, None] ** 2,
            (5, 4, 4),
        ).copy().astype(np.float32)
        solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
        solver.velocity_prev.fill((0.0, 0.0, 0.0))

        dt_s = 0.3
        solver._sst_momentum_explicit_stress_rhs_checked(dt_s)

        face_i = 2
        center_distance_m = float(
            solver.center_distance_x_m.to_numpy()[face_i]
        )
        expected_face_gradient_m_s2 = float(
            (k_state[face_i, 1, 1] - k_state[face_i - 1, 1, 1])
            / center_distance_m
        )
        expected_face_rhs_mps = (
            -(2.0 / 3.0) * dt_s * expected_face_gradient_m_s2
        )
        legacy_cell_gradient_m_s2 = float(
            (k_state[face_i + 1, 1, 1] - k_state[face_i - 1, 1, 1])
            / (centers_x[face_i + 1] - centers_x[face_i - 1])
        )
        self.assertGreater(
            abs(legacy_cell_gradient_m_s2 - expected_face_gradient_m_s2),
            0.1,
            msg="graded fixture must separate cell and two-point face gradients",
        )
        self.assertAlmostEqual(
            float(solver.velocity.to_numpy()[face_i, 1, 1, 0]),
            expected_face_rhs_mps,
            places=6,
        )

    def test_sst_grad_k_rhs_uniform_quadratic_locks_mac_face_location(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(5, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        centers_x = solver.cell_center_x_m.to_numpy().astype(np.float64)
        k_state = np.broadcast_to(
            centers_x[:, None, None] ** 2,
            (5, 4, 4),
        ).copy().astype(np.float32)
        solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
        solver.velocity_prev.fill((0.0, 0.0, 0.0))

        dt_s = 0.3
        solver._sst_momentum_explicit_stress_rhs_checked(dt_s)

        face_i = 2
        face_x_m = float(solver.cell_face_x_m.to_numpy()[face_i])
        expected_face_gradient_m_s2 = 2.0 * face_x_m
        expected_face_rhs_mps = (
            -(2.0 / 3.0) * dt_s * expected_face_gradient_m_s2
        )
        self.assertAlmostEqual(
            float(solver.velocity.to_numpy()[face_i, 1, 1, 0]),
            expected_face_rhs_mps,
            places=6,
        )

    def test_sst_grad_k_rhs_does_not_cross_solid_or_domain_boundary(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4, 0.5),
            cell_widths_y_m=(0.25,) * 4,
            cell_widths_z_m=(0.25,) * 4,
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
                cartesian_grid=grid,
            )
        )
        centers_x = solver.cell_center_x_m.to_numpy().astype(np.float64)
        base_k_state = np.broadcast_to(
            centers_x[:, None, None] ** 2,
            (5, 4, 4),
        ).copy().astype(np.float32)
        obstacle = np.zeros((5, 4, 4), dtype=np.int32)
        obstacle[1, :, :] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.velocity_prev.fill((0.0, 0.0, 0.0))

        dt_s = 0.3
        interface_face_i = 2
        interface_results_mps: list[float] = []
        for solid_k_m2_s2 in (1.0e-12, 1.0e6):
            k_state = base_k_state.copy()
            k_state[obstacle != 0] = solid_k_m2_s2
            solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
            solver._sst_momentum_explicit_stress_rhs_checked(dt_s)
            velocity = solver.velocity.to_numpy()
            interface_results_mps.append(
                float(velocity[interface_face_i, 1, 1, 0])
            )
            self.assertEqual(float(velocity[1, 1, 1, 0]), 0.0)
        self.assertAlmostEqual(
            interface_results_mps[0],
            interface_results_mps[1],
            places=6,
            msg="a fluid/solid MAC face must not sample k inside the solid",
        )

        solver.obstacle.fill(0)
        solver.sst_turbulent_kinetic_energy.from_numpy(base_k_state)
        solver._sst_momentum_explicit_stress_rhs_checked(dt_s)
        expected_xmin_rhs_mps = 0.0
        self.assertAlmostEqual(
            float(solver.velocity.to_numpy()[0, 1, 1, 0]),
            expected_xmin_rhs_mps,
            places=6,
        )

    def test_resolved_sst_grad_k_keeps_algebraic_constraint_out_of_wall_geometry(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(5, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.obstacle.fill(0)
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=(False,) * 6,
            near_wall_treatment="resolved",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        storage_i = 2
        target_mps = 1.75
        x_mask = np.zeros((5, 4, 4), dtype=np.int32)
        x_mask[storage_i, :, :] = 1
        solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
            x_mask
        )
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            x_mask
        )
        solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
            x_mask
        )
        targets = np.zeros((5, 4, 4, 3), dtype=np.float32)
        targets[storage_i, :, :, 0] = target_mps
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
        mobility = np.ones((5, 4, 4, 3), dtype=np.float32)
        mobility[storage_i, :, :, 0] = 0.0
        solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
            mobility
        )
        enforcement = np.zeros((5, 4, 4, 3), dtype=np.float32)
        enforcement[storage_i, :, :, 0] = 1.0
        solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
            enforcement
        )

        velocity_prev = np.zeros((5, 4, 4, 3), dtype=np.float32)
        velocity_prev[storage_i, :, :, 0] = -9.0
        solver.velocity_prev.from_numpy(velocity_prev)

        measured_mps: list[float] = []
        for plus_side_k_m2_s2 in (1.0, 2.0):
            k_state = np.ones((5, 4, 4), dtype=np.float32)
            k_state[storage_i, :, :] = plus_side_k_m2_s2
            solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
            solver._sst_momentum_explicit_stress_rhs_checked(0.3)
            measured_mps.append(
                float(solver.velocity.to_numpy()[storage_i, 1, 1, 0])
            )

        expected_mps = (
            -9.0,
            -9.0
            - (2.0 / 3.0)
            * 0.3
            * (2.0 - 1.0)
            / float(solver.center_distance_x_m.to_numpy()[storage_i]),
        )
        np.testing.assert_allclose(
            measured_mps,
            expected_mps,
            rtol=0.0,
            atol=1.0e-6,
            err_msg=(
                "a fluid-fluid HIBM interpolation constraint must not invent "
                "a resolved-SST wall normal or suppress the physical grad-k RHS"
            ),
        )

    def test_correlation_sst_grad_k_wall_is_componentwise_on_all_axes(
        self,
    ) -> None:
        shape = (5, 5, 5)
        storage_coordinate = 2
        sentinel_velocity_mps = np.array((17.0, -19.0, 23.0), dtype=np.float32)

        for axis_index in range(3):
            with self.subTest(axis_index=axis_index):
                solver = _cuda_solver(
                    FluidDomainSpec.unit_box(
                        grid_nodes=shape,
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0e-5,
                        dt_s=1.0e-4,
                    )
                )
                solver.obstacle.fill(0)
                solver.configure_sst_2003(
                    inlet_velocity_mps=3.0,
                    turbulence_intensity=0.05,
                    turbulent_viscosity_ratio=10.0,
                    no_slip_domain_walls=(False,) * 6,
                    near_wall_treatment="fluent_correlation",
                )
                solver.set_velocity_dirichlet_boundary_authority("canonical")

                component_bit = 1 << axis_index
                storage_plane = [slice(None)] * 3
                storage_plane[axis_index] = storage_coordinate
                storage_plane_index = tuple(storage_plane)
                probe = [1, 1, 1]
                probe[axis_index] = storage_coordinate
                probe_index = tuple(probe)

                exact_mask = np.zeros(shape, dtype=np.int32)
                exact_mask[storage_plane_index] = component_bit
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    exact_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    exact_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    exact_mask
                )

                target_mps = 1.75 + axis_index
                targets = np.zeros((*shape, 3), dtype=np.float32)
                targets[storage_plane_index + (axis_index,)] = target_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((*shape, 3), dtype=np.float32)
                mobility[storage_plane_index + (axis_index,)] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((*shape, 3), dtype=np.float32)
                enforcement[storage_plane_index + (axis_index,)] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )

                velocity_prev = np.broadcast_to(
                    sentinel_velocity_mps,
                    (*shape, 3),
                ).copy()
                solver.velocity_prev.from_numpy(velocity_prev)
                expected_velocity_mps = sentinel_velocity_mps.copy()
                expected_velocity_mps[axis_index] = target_mps

                minus_plane = [slice(None)] * 3
                minus_plane[axis_index] = storage_coordinate - 1
                minus_plane_index = tuple(minus_plane)
                for high_k_side in ("minus", "plus"):
                    with self.subTest(
                        axis_index=axis_index,
                        high_k_side=high_k_side,
                    ):
                        k_state = np.ones(shape, dtype=np.float32)
                        high_plane_index = (
                            minus_plane_index
                            if high_k_side == "minus"
                            else storage_plane_index
                        )
                        k_state[high_plane_index] = 1.0e6
                        solver.sst_turbulent_kinetic_energy.from_numpy(k_state)
                        solver._sst_momentum_explicit_stress_rhs_checked(0.3)
                        np.testing.assert_allclose(
                            solver.velocity.to_numpy()[probe_index],
                            expected_velocity_mps,
                            rtol=0.0,
                            atol=1.0e-6,
                            err_msg=(
                                "the exact normal component must restore its "
                                "target without freezing tangential components"
                            ),
                        )

    def test_sst_grad_k_rhs_restores_canonical_owner_for_both_face_orientations(
        self,
    ) -> None:
        target_fluid_i = 2
        authoritative_target_mps = 1.75
        measured_target_mps: list[float] = []

        for obstacle_side, stale_face_mps in ((-1, -9.0), (1, 13.0)):
            with self.subTest(obstacle_side=obstacle_side):
                solver = _cuda_solver(
                    FluidDomainSpec.unit_box(
                        grid_nodes=(5, 4, 4),
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0e-5,
                        dt_s=1.0e-4,
                    )
                )
                obstacle_i = target_fluid_i + obstacle_side
                obstacle = np.zeros((5, 4, 4), dtype=np.int32)
                obstacle[obstacle_i, :, :] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")
                storage_i = (
                    target_fluid_i if obstacle_side < 0 else obstacle_i
                )

                velocity_prev = np.zeros((5, 4, 4, 3), dtype=np.float32)
                velocity_prev[storage_i, :, :, 0] = stale_face_mps
                solver.velocity_prev.from_numpy(velocity_prev)
                centers_x = solver.cell_center_x_m.to_numpy()
                k_state = np.broadcast_to(
                    1.0 + centers_x[:, None, None] ** 2,
                    (5, 4, 4),
                ).copy().astype(np.float32)
                solver.sst_turbulent_kinetic_energy.from_numpy(k_state)

                x_mask = np.zeros((5, 4, 4), dtype=np.int32)
                x_mask[storage_i, :, :] = 1
                solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
                    x_mask
                )
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                    x_mask
                )
                solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
                    x_mask
                )
                targets = np.zeros((5, 4, 4, 3), dtype=np.float32)
                targets[storage_i, :, :, 0] = authoritative_target_mps
                solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
                mobility = np.ones((5, 4, 4, 3), dtype=np.float32)
                mobility[storage_i, :, :, 0] = 0.0
                solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
                    mobility
                )
                enforcement = np.zeros((5, 4, 4, 3), dtype=np.float32)
                enforcement[storage_i, :, :, 0] = 1.0
                solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
                    enforcement
                )
                solver.sst_obstacle_interface_wall_target_component_mask.fill(0)
                solver._sst_momentum_explicit_stress_rhs_checked(0.3)
                self.assertEqual(
                    int(
                        solver.sst_obstacle_interface_wall_target_component_mask.to_numpy()[
                            storage_i, 1, 1, 0
                        ]
                    )
                    & 1,
                    1,
                )

                actual_target_mps = float(
                    solver.velocity.to_numpy()[storage_i, 1, 1, 0]
                )
                measured_target_mps.append(actual_target_mps)
                self.assertAlmostEqual(
                    actual_target_mps,
                    authoritative_target_mps,
                    places=6,
                )

        self.assertAlmostEqual(
            measured_target_mps[0],
            measured_target_mps[1],
            places=6,
        )

    def test_predict_none_reuses_configured_sst_wall_flags(self) -> None:
        spec = FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-4,
        )
        implicit = _cuda_solver(spec)
        explicit = _cuda_solver(spec)
        walls = (False, False, True, False, False, False)
        for solver in (implicit, explicit):
            solver.configure_sst_2003(
                inlet_velocity_mps=1.0,
                turbulence_intensity=0.05,
                turbulent_viscosity_ratio=10.0,
                no_slip_domain_walls=walls,
            )
            solver.set_uniform_velocity((1.0, 0.0, 0.0))

        implicit.predict(dt_s=1.0e-4)
        explicit.predict(dt_s=1.0e-4, no_slip_domain_walls=walls)

        np.testing.assert_allclose(
            implicit.velocity.to_numpy(),
            explicit.velocity.to_numpy(),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_sst_transport_auto_substeps_stiff_diffusion_and_preserves_state(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=2.0e-2,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        k = np.ones((4, 4, 4), dtype=np.float32)
        k[1, 1, 1] = 4.0
        omega = np.full((4, 4, 4), 0.5, dtype=np.float32)
        solver.sst_turbulent_kinetic_energy.from_numpy(k)
        solver.sst_specific_dissipation_rate.from_numpy(omega)

        report = solver.advance_sst_transport(
            dt_s=2.0e-2,
            kinematic_viscosity_m2_s=1.0e-5,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        self.assertEqual(
            _report_value(report, "diffusion_integrator"),
            "lod_backward_euler_frozen_coefficients",
        )
        self.assertEqual(int(_report_value(report, "diffusion_substeps")), 1)
        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        omega_after = solver.sst_specific_dissipation_rate.to_numpy()
        mu_t_after = solver.sst_eddy_viscosity_pa_s.to_numpy()
        for name, values in (
            ("k", k_after),
            ("omega", omega_after),
            ("eddy viscosity", mu_t_after),
        ):
            with self.subTest(field=name):
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values > 0.0))
        self.assertLess(float(k_after[1, 1, 1]), float(k[1, 1, 1]))
        self.assertGreater(float(k_after[0, 1, 1]), float(k[0, 1, 1]))

    def test_sst_stiff_diffusion_is_implicit_not_an_explicit_substep_cap(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=2.0e-2,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        k = np.full((4, 4, 4), 250.0, dtype=np.float32)
        k[1, 1, 1] = 500.0
        solver.sst_turbulent_kinetic_energy.from_numpy(k)
        solver.sst_specific_dissipation_rate.fill(0.1)
        # This is an algorithmic contract, not permission to hide the problem
        # behind a larger explicit-substep ceiling.
        solver._sst_max_automatic_substeps = 2

        report = solver.advance_sst_transport(
            dt_s=2.0e-2,
            kinematic_viscosity_m2_s=1.0e-5,
            no_slip_domain_walls=_OPEN_WALLS,
            advection_scheme="muscl_tvd",
        )

        self.assertEqual(
            _report_value(report, "diffusion_integrator"),
            "lod_backward_euler_frozen_coefficients",
        )
        self.assertEqual(int(_report_value(report, "explicit_transport_substeps")), 1)
        self.assertEqual(
            int(_report_value(report, "implicit_diffusion_directional_solves")),
            3,
        )
        self.assertGreater(
            float(_report_value(report, "diffusion_cfl_before_implicit_solve")),
            4096.0,
        )
        for name, values in (
            ("k", solver.sst_turbulent_kinetic_energy.to_numpy()),
            ("omega", solver.sst_specific_dissipation_rate.to_numpy()),
        ):
            with self.subTest(field=name):
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values > 0.0))

    def test_sst_invalid_explicit_candidate_rolls_back_and_retries(self) -> None:
        class RejectFirstCandidateSolver(CartesianFluidSolver):
            def __init__(self, spec: FluidDomainSpec) -> None:
                super().__init__(spec, runtime=TaichiRuntimeConfig(arch="cuda"))
                self._forced_candidate_rejections_remaining = 1

            def _sst_explicit_candidate_invalid_count_host(self) -> int:
                if self._forced_candidate_rejections_remaining > 0:
                    self._forced_candidate_rejections_remaining -= 1
                    return 1
                return super()._sst_explicit_candidate_invalid_count_host()

        solver = RejectFirstCandidateSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        report = solver.advance_sst_transport(
            dt_s=1.0e-2,
            advection_scheme="muscl_tvd",
        )

        self.assertEqual(int(report["rejected_transport_trial_count"]), 1)
        self.assertEqual(solver._sst_last_transport_rejected_trial_count, 1)
        self.assertEqual(int(report["explicit_transport_substeps"]), 2)
        self.assertTrue(
            np.all(np.isfinite(solver.sst_turbulent_kinetic_energy.to_numpy()))
        )
        self.assertTrue(
            np.all(solver.sst_turbulent_kinetic_energy.to_numpy() > 0.0)
        )

    def test_sst_transport_accepts_laminar_zero_k_without_retry(
        self,
    ) -> None:
        walls = (False, False, True, False, False, False)
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=2.93691e-11,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=walls,
            near_wall_treatment="fluent_correlation",
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.sst_turbulent_kinetic_energy.fill(0.0)
        solver.sst_specific_dissipation_rate.fill(404679.90625)

        # Zero turbulent kinetic energy is a valid laminar-limit state; only
        # omega must remain strictly positive.
        report = solver.advance_sst_transport(
            dt_s=2.93691e-11,
            no_slip_domain_walls=walls,
        )

        self.assertEqual(int(report["rejected_transport_trial_count"]), 0)
        self.assertEqual(solver._sst_last_transport_rejected_trial_count, 0)
        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        omega_after = solver.sst_specific_dissipation_rate.to_numpy()
        mu_t_after = solver.sst_eddy_viscosity_pa_s.to_numpy()
        self.assertTrue(np.all(np.isfinite(k_after)))
        self.assertTrue(np.all(k_after >= 0.0))
        self.assertTrue(np.all(np.isfinite(omega_after)))
        self.assertTrue(np.all(omega_after > 0.0))
        zero_k = k_after == 0.0
        self.assertTrue(np.any(zero_k))
        np.testing.assert_array_equal(
            mu_t_after[zero_k],
            np.zeros_like(mu_t_after[zero_k]),
        )

    def test_sst_lod_diffusion_conserves_on_a_graded_variable_coefficient_grid(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.3, 0.4),
            cell_widths_y_m=(0.15, 0.2, 0.25, 0.4),
            cell_widths_z_m=(0.1, 0.2, 0.3, 0.4),
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
                cartesian_grid=grid,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        ii, jj, kk = np.indices((4, 4, 4), dtype=np.float32)
        k_before = 0.5 + 0.2 * ii + 0.1 * jj + 0.05 * kk
        omega_before = 2.0 + 0.3 * ii + 0.2 * jj + 0.1 * kk
        solver.sst_turbulent_kinetic_energy.from_numpy(k_before)
        solver.sst_specific_dissipation_rate.from_numpy(omega_before)
        mu_t = (0.01 + 0.015 * ii + 0.01 * jj + 0.005 * kk).astype(np.float32)
        sigma_k = (0.6 + 0.05 * ii + 0.03 * jj).astype(np.float32)
        sigma_omega = (0.5 + 0.04 * jj + 0.02 * kk).astype(np.float32)
        solver.sst_eddy_viscosity_pa_s.from_numpy(mu_t)
        solver.sst_sigma_k.from_numpy(sigma_k)
        solver.sst_sigma_omega.from_numpy(sigma_omega)
        volume = (
            np.asarray(grid.cell_widths_x_m, dtype=np.float64)[:, None, None]
            * np.asarray(grid.cell_widths_y_m, dtype=np.float64)[None, :, None]
            * np.asarray(grid.cell_widths_z_m, dtype=np.float64)[None, None, :]
        )
        k_integral_before = float(np.sum(volume * k_before))
        omega_integral_before = float(np.sum(volume * omega_before))

        def solve_expected_x_lines(
            values: np.ndarray,
            diffusivity: np.ndarray,
        ) -> np.ndarray:
            expected = np.empty_like(values, dtype=np.float64)
            widths = np.asarray(grid.cell_widths_x_m, dtype=np.float64)
            for y_index in range(4):
                for z_index in range(4):
                    matrix = np.zeros((4, 4), dtype=np.float64)
                    for x_index in range(4):
                        lower = 0.0
                        upper = 0.0
                        if x_index > 0:
                            distance = 0.5 * (widths[x_index - 1] + widths[x_index])
                            lower = (
                                0.5
                                * 0.5
                                * (
                                    diffusivity[x_index - 1, y_index, z_index]
                                    + diffusivity[x_index, y_index, z_index]
                                )
                                / (widths[x_index] * distance)
                            )
                            matrix[x_index, x_index - 1] = -lower
                        if x_index + 1 < 4:
                            distance = 0.5 * (widths[x_index] + widths[x_index + 1])
                            upper = (
                                0.5
                                * 0.5
                                * (
                                    diffusivity[x_index, y_index, z_index]
                                    + diffusivity[x_index + 1, y_index, z_index]
                                )
                                / (widths[x_index] * distance)
                            )
                            matrix[x_index, x_index + 1] = -upper
                        matrix[x_index, x_index] = 1.0 + lower + upper
                    expected[:, y_index, z_index] = np.linalg.solve(
                        matrix,
                        values[:, y_index, z_index].astype(np.float64),
                    )
            return expected

        expected_k_after_x = solve_expected_x_lines(k_before, 1.0e-5 + sigma_k * mu_t)
        expected_omega_after_x = solve_expected_x_lines(
            omega_before, 1.0e-5 + sigma_omega * mu_t
        )
        solver._sst_lod_backward_euler_axis_kernel(0.5, 1.0e-5, 0, 0, 0)
        np.testing.assert_allclose(
            solver.sst_turbulent_kinetic_energy.to_numpy(),
            expected_k_after_x,
            rtol=3.0e-5,
            atol=3.0e-5,
        )
        np.testing.assert_allclose(
            solver.sst_specific_dissipation_rate.to_numpy(),
            expected_omega_after_x,
            rtol=3.0e-5,
            atol=3.0e-5,
        )
        for axis_index in (1, 2):
            solver._sst_lod_backward_euler_axis_kernel(
                0.5, 1.0e-5, axis_index, 0, 0
            )

        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        omega_after = solver.sst_specific_dissipation_rate.to_numpy()
        self.assertAlmostEqual(
            float(np.sum(volume * k_after)), k_integral_before, delta=5.0e-5
        )
        self.assertAlmostEqual(
            float(np.sum(volume * omega_after)),
            omega_integral_before,
            delta=5.0e-5,
        )
        self.assertGreaterEqual(float(np.min(k_after)), float(np.min(k_before)) - 1.0e-5)
        self.assertLessEqual(float(np.max(k_after)), float(np.max(k_before)) + 1.0e-5)
        self.assertGreaterEqual(
            float(np.min(omega_after)), float(np.min(omega_before)) - 1.0e-5
        )
        self.assertLessEqual(
            float(np.max(omega_after)), float(np.max(omega_before)) + 1.0e-5
        )

    def test_sst_lod_obstacle_splits_a_diffusion_line(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(5, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
            )
        )
        obstacle = np.zeros((5, 4, 4), dtype=np.int32)
        obstacle[2, :, :] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.sst_eddy_viscosity_pa_s.fill(0.25)
        solver.sst_sigma_k.fill(0.85)
        solver.sst_sigma_omega.fill(0.5)

        right_segment_results: list[np.ndarray] = []
        left_segment_results: list[np.ndarray] = []
        for left_pulse in (1.0, 100.0):
            k_initial = np.ones((5, 4, 4), dtype=np.float32)
            k_initial[0, :, :] = left_pulse
            solver.sst_turbulent_kinetic_energy.from_numpy(k_initial)
            solver.sst_specific_dissipation_rate.fill(2.0)
            solver._sst_lod_backward_euler_axis_kernel(
                0.2,
                1.0e-5,
                0,
                0,
                0,
            )
            k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
            left_segment_results.append(k_after[:2].copy())
            right_segment_results.append(k_after[3:].copy())
            self.assertTrue(np.all(np.isfinite(k_after)))
            self.assertTrue(np.all(k_after > 0.0))

        np.testing.assert_allclose(
            right_segment_results[1],
            right_segment_results[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertGreater(
            float(np.max(np.abs(left_segment_results[1] - left_segment_results[0]))),
            1.0,
        )

    def test_sst_wall_omega_numerical_limit_fails_closed_without_clipping(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-5,
            )
        )
        walls = (True, False, False, False, False, False)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=walls,
        )
        wall_distance = solver.sst_wall_distance_m.to_numpy()
        wall_distance[0, :, :] = 1.0e-12
        solver.sst_wall_distance_m.from_numpy(wall_distance)

        with self.assertRaisesRegex(FloatingPointError, "wall-omega"):
            solver.advance_sst_transport(
                dt_s=1.0e-5,
                no_slip_domain_walls=walls,
            )

    def test_sst_inlet_is_a_face_flux_even_below_half_cell_cfl(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            inlet_face="zmax",
            outlet_face="zmin",
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, -1.0))
        solver.sst_turbulent_kinetic_energy.fill(1.0)
        solver.sst_specific_dissipation_rate.fill(1.0)

        solver.advance_sst_transport(dt_s=1.0e-2)

        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        self.assertLess(float(np.mean(k_after[:, :, -1])), float(np.mean(k_after[:, :, -2])))

    def test_sst_transport_uses_exact_external_face_velocity_ledger(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-2,
            )
        )
        # Production velocity inlets live on the exact external-face ledger;
        # the adjacent cell centre is deliberately stationary here.  SST
        # scalar transport must therefore use the face velocity, not the cell
        # velocity, when assembling the conservative boundary flux.
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            inlet_face="zmax",
            outlet_face="zmin",
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.sst_turbulent_kinetic_energy.fill(1.0)
        solver.sst_specific_dissipation_rate.fill(1.0)

        solver.advance_sst_transport(dt_s=1.0e-2)

        baseline_k = solver.sst_turbulent_kinetic_energy.to_numpy()
        baseline_boundary_mean = float(np.mean(baseline_k[:, :, -1]))

        # Replay the same state with only the exact external-face ledger
        # changed.  This paired comparison isolates the ledger contribution
        # from SST diffusion/source splitting and remains valid when the
        # automatic stability partition changes its substep sizes.
        solver.sst_turbulent_kinetic_energy.fill(1.0)
        solver.sst_specific_dissipation_rate.fill(1.0)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, -1.0),
        )
        solver.advance_sst_transport(dt_s=1.0e-2)

        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        # This deliberately isolated face ledger is not divergence-free: the
        # cell-centre velocity and all other faces remain stationary.  The
        # conservative scalar response therefore has no physically prescribed
        # sign.  What the contract owns is that the exact face ledger is
        # consumed and produces a resolvable change.
        self.assertGreater(
            abs(float(np.mean(k_after[:, :, -1])) - baseline_boundary_mean),
            1.0e-5,
        )

    def test_sst_euler_cfl_uses_exact_physical_faces_not_cell_speed(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-1,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            max_automatic_substeps=64,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.sst_turbulent_kinetic_energy.fill(1.0)
        solver.sst_specific_dissipation_rate.fill(1.0)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=0,
            side_index=0,
            target_velocity_mps=(10.0, 0.0, 0.0),
        )

        report = solver.advance_sst_transport(
            dt_s=1.0e-1,
            advection_scheme="euler",
        )

        self.assertGreater(
            float(_report_value(report, "advection_cfl_before_substeps")),
            3.9,
        )
        self.assertGreaterEqual(
            int(_report_value(report, "explicit_transport_substeps")),
            5,
        )
        self.assertLessEqual(
            float(_report_value(report, "maximum_substep_advection_cfl")),
            0.900001,
        )

    def test_sst_euler_cfl_counts_both_backward_mac_outflow_faces(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0e-3,) * 4,
            cell_widths_y_m=(0.25,) * 4,
            cell_widths_z_m=(0.25,) * 4,
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
                cartesian_grid=grid,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            max_automatic_substeps=16,
        )
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        # For cell i=1 the backward-MAC x-minus face is stored at i=1 and
        # x-plus at i=2.  Both point outwards, so its exact rate is
        # (10+10)/0.001 = 20000 /s.  A max-cell-speed estimate sees only
        # about 10080 /s on this anisotropic grid and is not a safe CFL gate.
        velocity[1, :, :, 0] = -10.0
        velocity[2, :, :, 0] = 10.0
        solver.velocity.from_numpy(velocity)
        solver.sst_turbulent_kinetic_energy.fill(1.0)
        solver.sst_specific_dissipation_rate.fill(1.0)

        report = solver.advance_sst_transport(
            dt_s=1.0e-4,
            advection_scheme="euler",
        )

        self.assertGreater(
            float(_report_value(report, "advection_cfl_before_substeps")),
            1.99,
        )
        self.assertGreaterEqual(
            int(_report_value(report, "explicit_transport_substeps")),
            3,
        )

    def test_sst_momentum_diffusion_uses_exact_external_face_velocity_ledger(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            inlet_face="zmax",
            outlet_face="zmin",
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, -1.0),
        )
        solver.sst_turbulent_kinetic_energy.fill(0.25)
        solver.sst_eddy_viscosity_pa_s.fill(0.1)

        solver.predict(dt_s=1.0e-4, kinematic_viscosity_m2_s=0.0)

        velocity = solver.velocity.to_numpy()
        self.assertLess(float(np.mean(velocity[:, :, -1, 2])), -1.0e-6)

    def test_sst_transpose_stress_places_cross_derivative_on_x_mac_face(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0,) * 7,
            cell_widths_y_m=(1.0,) * 5,
            cell_widths_z_m=(1.0,) * 5,
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=2.0,
                viscosity_pa_s=1.0,
                dt_s=0.1,
                cartesian_grid=grid,
            )
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        frozen_velocity = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
        centers_x_m = np.asarray(grid.cell_centers_x_m, dtype=np.float32)
        backward_faces_y_m = np.asarray(
            grid.cell_faces_y_m[:-1],
            dtype=np.float32,
        )
        frozen_velocity[:, :, :, 1] = (
            centers_x_m[:, None, None] ** 2
            * backward_faces_y_m[None, :, None] ** 2
        )
        solver.velocity_prev.from_numpy(frozen_velocity)
        solver.sst_eddy_viscosity_pa_s.fill(1.0)

        solver._sst_momentum_transpose_stress_checked(
            0.1,
            0.5,
            0,
            0,
            0,
            0,
            0,
            0,
        )

        # u_y=C_x^2 F_y^2 gives d_y(d_x u_y)=4*x*y.  The x velocity
        # stored at [4,2,2] lives at (F_x[4], C_y[2], C_z[2]), not at
        # the packed row's (C_x[4], F_y[2], C_z[2]) coordinates.
        target_i, target_j, target_k = 4, 2, 2
        expected_increment_mps = (
            0.1
            * 4.0
            * 1.0
            * grid.cell_faces_x_m[target_i]
            * grid.cell_centers_y_m[target_j]
        )
        self.assertAlmostEqual(
            float(solver.velocity.to_numpy()[target_i, target_j, target_k, 0]),
            expected_increment_mps,
            places=5,
        )

    def test_sst_transpose_stress_uses_canonical_z_wall_flux_for_both_owners(
        self,
    ) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0,) * 5,
            cell_widths_y_m=(1.0,) * 5,
            cell_widths_z_m=(1.0,) * 5,
        )
        slope_s = 2.0
        target_fluid_k = 2

        for obstacle_side, stale_owner_mps in ((-1, 37.0), (1, -41.0)):
            with self.subTest(obstacle_side=obstacle_side):
                solver = _cuda_solver(
                    FluidDomainSpec(
                        bounds_min_m=grid.bounds_min_m,
                        bounds_max_m=grid.bounds_max_m,
                        grid_nodes=None,
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0,
                        dt_s=0.1,
                        cartesian_grid=grid,
                    )
                )
                obstacle_k = target_fluid_k + obstacle_side
                obstacle = np.zeros(grid.grid_nodes, dtype=np.int32)
                obstacle[:, :, obstacle_k] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.set_velocity_dirichlet_boundary_authority("canonical")

                # A backward-MAC z face is stored on its plus-side row.  The
                # minus-wall A owner is therefore the fluid row, while the
                # plus-wall B owner is the obstacle row.
                storage_k = (
                    target_fluid_k if obstacle_side < 0 else obstacle_k
                )
                wall_target_z_mps = (
                    slope_s * np.asarray(grid.cell_centers_y_m, dtype=np.float32)
                )

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
                targets[:, :, storage_k, 2] = wall_target_z_mps[None, :]
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

                # Keep every local cell-center value at zero so this fixture
                # isolates the physical-interface transpose flux.  The raw
                # owner value is deliberately stale and must be ignored.
                frozen_velocity = np.zeros(
                    (*grid.grid_nodes, 3), dtype=np.float32
                )
                frozen_velocity[:, :, storage_k, 2] = stale_owner_mps
                if obstacle_side < 0:
                    frozen_velocity[:, :, target_fluid_k + 1, 2] = (
                        -wall_target_z_mps[None, :]
                    )
                    frozen_velocity[:, :, target_fluid_k + 2, 2] = (
                        wall_target_z_mps[None, :]
                    )
                else:
                    frozen_velocity[:, :, target_fluid_k, 2] = (
                        -wall_target_z_mps[None, :]
                    )
                    frozen_velocity[:, :, target_fluid_k - 1, 2] = (
                        wall_target_z_mps[None, :]
                    )
                solver.velocity_prev.from_numpy(frozen_velocity)

                velocity = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
                velocity[:, :, storage_k, 2] = wall_target_z_mps[None, :]
                solver.velocity.from_numpy(velocity)
                solver.velocity_dirichlet_boundary_active.fill(0)
                solver.sst_obstacle_interface_wall_target_component_mask.fill(0)
                solver.sst_eddy_viscosity_pa_s.fill(0.0)

                solver._sst_momentum_transpose_stress_checked(
                    0.1,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

                actual_velocity = solver.velocity.to_numpy()
                expected_y_increment_mps = 0.1 * slope_s * obstacle_side
                self.assertAlmostEqual(
                    float(actual_velocity[2, 2, target_fluid_k, 1]),
                    expected_y_increment_mps,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(actual_velocity[2, 2, storage_k, 2]),
                    float(wall_target_z_mps[2]),
                    places=6,
                )
                self.assertEqual(
                    int(
                        solver.sst_obstacle_interface_wall_target_component_mask.to_numpy()[
                            2, 2, storage_k, 2
                        ]
                    )
                    & 4,
                    4,
                )

    def test_sst_transpose_stress_uses_moving_wall_tangential_gradient(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[:, :, 1] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.set_uniform_velocity((0.0, 0.0, 0.0))
        active = np.zeros((4, 4, 4), dtype=np.int32)
        targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
        active[:, :, 2] = 1
        wall_y = solver.cell_center_y_m.to_numpy()
        targets[:, :, 2, 2] = wall_y[None, :]
        solver.velocity_dirichlet_boundary_active.from_numpy(active)
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
        solver.sst_turbulent_kinetic_energy.fill(0.25)
        solver.sst_eddy_viscosity_pa_s.fill(0.1)

        solver.predict(dt_s=1.0e-4, kinematic_viscosity_m2_s=0.0)

        # At the z-normal moving wall, d(u_z,wall)/dy contributes a y
        # component through div(nu_eff * grad(u)^T).  A normal-only wall
        # closure leaves this component identically zero.
        self.assertGreater(
            float(np.max(np.abs(solver.velocity.to_numpy()[:, 1:3, 2, 1]))),
            1.0e-6,
        )

    def test_sst_continuity_correction_uses_zero_relative_flux_at_moving_obstacle(
        self,
    ) -> None:
        """Moving-wall MAC data must not masquerade as scalar through-flow."""

        for obstacle_k, adjacent_k, reference_k in ((1, 2, 3), (2, 1, 0)):
            with self.subTest(obstacle_k=obstacle_k):
                solver = _cuda_solver(
                    FluidDomainSpec.unit_box(
                        grid_nodes=(4, 4, 4),
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0e-12,
                        dt_s=1.0e-2,
                    )
                )
                obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                obstacle[:, :, obstacle_k] = 1
                solver.obstacle.from_numpy(obstacle)
                solver.configure_sst_2003(
                    inlet_velocity_mps=1.0,
                    turbulence_intensity=0.05,
                    turbulent_viscosity_ratio=10.0,
                    no_slip_domain_walls=_OPEN_WALLS,
                )
                velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
                # Face two is the shared obstacle/fluid face in both storage
                # orientations.  Its canonical backward-MAC row may belong to
                # either the fluid or the obstacle, but its wall-normal speed
                # is boundary motion rather than scalar through-flow.
                velocity[:, :, 2, 2] = 0.4
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
                    "_update_sst_coefficients_checked",
                    new=lambda _solver, _nu: None,
                ):
                    solver.advance_sst_transport(
                        dt_s=1.0e-2,
                        kinematic_viscosity_m2_s=1.0e-12,
                        no_slip_domain_walls=_OPEN_WALLS,
                        advection_scheme="muscl_tvd",
                    )

                k_state = solver.sst_turbulent_kinetic_energy.to_numpy()
                omega_state = solver.sst_specific_dissipation_rate.to_numpy()
                self.assertLessEqual(
                    float(
                        np.max(
                            np.abs(
                                k_state[:, :, adjacent_k]
                                - k_state[:, :, reference_k]
                            )
                        )
                    ),
                    1.0e-4,
                )
                self.assertLessEqual(
                    float(
                        np.max(
                            np.abs(
                                omega_state[:, :, adjacent_k]
                                - omega_state[:, :, reference_k]
                            )
                        )
                    ),
                    5.0e-4,
                )

    def test_wall_boundary_is_applied_at_face_not_overwritten_into_fluid_cell(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-5,
            )
        )
        walls = (False, False, True, False, False, False)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=walls,
        )
        solver.sst_turbulent_kinetic_energy.fill(0.5)
        solver.sst_specific_dissipation_rate.fill(2.0)

        solver.advance_sst_transport(dt_s=1.0e-5)

        k_after = solver.sst_turbulent_kinetic_energy.to_numpy()
        self.assertGreater(float(np.min(k_after[:, 0, :])), 1.0e-4)

    def test_fresh_fluid_reconstruction_restores_sst_state_from_fluid_neighbors(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        solver.sst_turbulent_kinetic_energy.fill(2.0)
        solver.sst_specific_dissipation_rate.fill(3.0)
        solver.sst_eddy_viscosity_pa_s.fill(4.0e-4)
        fresh = (2, 2, 2)
        solver.sst_turbulent_kinetic_energy[fresh] = 1.0e-12
        solver.sst_specific_dissipation_rate[fresh] = 1.0e12
        solver.sst_eddy_viscosity_pa_s[fresh] = 0.0
        solver.fresh_fluid_reconstruction_pending[fresh] = 1
        solver.hibm_fresh_fluid_cell[fresh] = 1
        solver.report_hibm_fresh_fluid_cells[None] = 1

        solver._reconstruct_fresh_fluid_cells()

        self.assertAlmostEqual(
            float(solver.sst_turbulent_kinetic_energy[fresh]), 2.0, places=5
        )
        self.assertAlmostEqual(
            float(solver.sst_specific_dissipation_rate[fresh]), 3.0, places=5
        )
        self.assertAlmostEqual(float(solver.sst_eddy_viscosity_pa_s[fresh]), 4.0e-4, places=8)

    def test_fresh_fluid_sst_reconstruction_ignores_velocity_only_boundary_weight(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        fresh = (2, 2, 2)
        donor = (1, 2, 2)
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[fresh] = 0
        obstacle[donor] = 0
        solver.obstacle.from_numpy(obstacle)
        solver.sst_turbulent_kinetic_energy[fresh] = 1.0e-12
        solver.sst_specific_dissipation_rate[fresh] = 1.0e12
        solver.sst_eddy_viscosity_pa_s[fresh] = 0.0
        solver.sst_turbulent_kinetic_energy[donor] = 2.0
        solver.sst_specific_dissipation_rate[donor] = 3.0
        solver.sst_eddy_viscosity_pa_s[donor] = 4.0e-4
        solver.velocity_dirichlet_boundary_active[fresh] = 1
        solver.velocity_dirichlet_boundary_value_mps[fresh] = (7.0, 8.0, 9.0)
        solver.fresh_fluid_reconstruction_pending[fresh] = 1

        solver._reconstruct_hibm_fresh_fluid_cells_kernel(1)

        self.assertEqual(int(solver.fresh_fluid_reconstruction_pending[fresh]), 0)
        self.assertAlmostEqual(
            float(solver.sst_turbulent_kinetic_energy[fresh]), 2.0, places=5
        )
        self.assertAlmostEqual(
            float(solver.sst_specific_dissipation_rate[fresh]), 3.0, places=5
        )
        self.assertAlmostEqual(
            float(solver.sst_eddy_viscosity_pa_s[fresh]), 4.0e-4, places=8
        )

    def test_fresh_fluid_reconstruction_advances_one_frozen_wave_per_pass(
        self,
    ) -> None:
        def run_layout(
            *,
            donor: tuple[int, int, int],
            first_layer: tuple[int, int, int],
            second_layer: tuple[int, int, int],
        ) -> tuple[dict[str, np.ndarray | int], dict[str, np.ndarray | int]]:
            solver = _cuda_solver(
                FluidDomainSpec.unit_box(
                    grid_nodes=(5, 4, 4),
                    density_kgm3=1.0,
                    viscosity_pa_s=1.0e-5,
                    dt_s=1.0e-4,
                )
            )
            solver.configure_sst_2003(
                inlet_velocity_mps=1.0,
                turbulence_intensity=0.05,
                turbulent_viscosity_ratio=10.0,
                no_slip_domain_walls=_OPEN_WALLS,
            )
            obstacle = np.ones((5, 4, 4), dtype=np.int32)
            obstacle[donor] = 0
            obstacle[first_layer] = 0
            obstacle[second_layer] = 0
            solver.obstacle.from_numpy(obstacle)
            solver.fresh_fluid_reconstruction_pending.fill(0)
            solver.fresh_fluid_reconstruction_pending[first_layer] = 1
            solver.fresh_fluid_reconstruction_pending[second_layer] = 1

            donor_velocity = np.array((1.25, -2.5, 3.75), dtype=np.float32)
            donor_velocity_prev = np.array((-4.0, 5.0, -6.0), dtype=np.float32)
            solver.velocity[donor] = donor_velocity
            solver.velocity_prev[donor] = donor_velocity_prev
            solver.sst_turbulent_kinetic_energy[donor] = 2.0
            solver.sst_specific_dissipation_rate[donor] = 3.0
            solver.sst_eddy_viscosity_pa_s[donor] = 4.0e-4
            solver.velocity[first_layer] = (10.0, 11.0, 12.0)
            solver.velocity_prev[first_layer] = (13.0, 14.0, 15.0)
            solver.sst_turbulent_kinetic_energy[first_layer] = 0.25
            solver.sst_specific_dissipation_rate[first_layer] = 100.0
            solver.sst_eddy_viscosity_pa_s[first_layer] = 1.0e-4
            solver.velocity[second_layer] = (20.0, 21.0, 22.0)
            solver.velocity_prev[second_layer] = (23.0, 24.0, 25.0)
            solver.sst_turbulent_kinetic_energy[second_layer] = 0.5
            solver.sst_specific_dissipation_rate[second_layer] = 200.0
            solver.sst_eddy_viscosity_pa_s[second_layer] = 2.0e-4

            def snapshot() -> dict[str, np.ndarray | int]:
                indices = (first_layer, second_layer)
                return {
                    "pending": np.array(
                        [
                            int(solver.fresh_fluid_reconstruction_pending[index])
                            for index in indices
                        ],
                        dtype=np.int32,
                    ),
                    "velocity": np.array(
                        [np.asarray(solver.velocity[index]) for index in indices]
                    ),
                    "velocity_prev": np.array(
                        [np.asarray(solver.velocity_prev[index]) for index in indices]
                    ),
                    "sst_k": np.array(
                        [
                            float(solver.sst_turbulent_kinetic_energy[index])
                            for index in indices
                        ]
                    ),
                    "sst_omega": np.array(
                        [
                            float(solver.sst_specific_dissipation_rate[index])
                            for index in indices
                        ]
                    ),
                    "sst_mu_t": np.array(
                        [
                            float(solver.sst_eddy_viscosity_pa_s[index])
                            for index in indices
                        ]
                    ),
                    "reconstructed_count": int(solver.reduction_count[None]),
                }

            solver._snapshot_fresh_fluid_reconstruction_pending_kernel()
            solver._reconstruct_hibm_fresh_fluid_cells_kernel(1)
            after_first_pass = snapshot()
            solver._snapshot_fresh_fluid_reconstruction_pending_kernel()
            solver._reconstruct_hibm_fresh_fluid_cells_kernel(1)
            after_second_pass = snapshot()
            return after_first_pass, after_second_pass

        layouts = (
            {
                "donor": (1, 2, 2),
                "first_layer": (2, 2, 2),
                "second_layer": (3, 2, 2),
            },
            {
                "donor": (3, 1, 1),
                "first_layer": (2, 1, 1),
                "second_layer": (1, 1, 1),
            },
        )
        results = [run_layout(**layout) for layout in layouts]
        donor_velocity = np.array((1.25, -2.5, 3.75), dtype=np.float32)
        donor_velocity_prev = np.array((-4.0, 5.0, -6.0), dtype=np.float32)
        untouched_second_velocity = np.array((20.0, 21.0, 22.0), dtype=np.float32)
        untouched_second_velocity_prev = np.array((23.0, 24.0, 25.0), dtype=np.float32)

        for after_first_pass, after_second_pass in results:
            np.testing.assert_array_equal(after_first_pass["pending"], (0, 1))
            self.assertEqual(after_first_pass["reconstructed_count"], 1)
            np.testing.assert_allclose(
                after_first_pass["velocity"],
                (donor_velocity, untouched_second_velocity),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                after_first_pass["velocity_prev"],
                (donor_velocity_prev, untouched_second_velocity_prev),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                after_first_pass["sst_k"], (2.0, 0.5), rtol=0.0, atol=1.0e-6
            )
            np.testing.assert_allclose(
                after_first_pass["sst_omega"],
                (3.0, 200.0),
                rtol=0.0,
                atol=1.0e-5,
            )
            np.testing.assert_allclose(
                after_first_pass["sst_mu_t"],
                (4.0e-4, 2.0e-4),
                rtol=0.0,
                atol=1.0e-8,
            )
            np.testing.assert_array_equal(after_second_pass["pending"], (0, 0))
            self.assertEqual(after_second_pass["reconstructed_count"], 1)
            np.testing.assert_allclose(
                after_second_pass["velocity"],
                (donor_velocity, donor_velocity),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                after_second_pass["velocity_prev"],
                (donor_velocity_prev, donor_velocity_prev),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                after_second_pass["sst_k"], (2.0, 2.0), rtol=0.0, atol=1.0e-6
            )
            np.testing.assert_allclose(
                after_second_pass["sst_omega"], (3.0, 3.0), rtol=0.0, atol=1.0e-6
            )
            np.testing.assert_allclose(
                after_second_pass["sst_mu_t"],
                (4.0e-4, 4.0e-4),
                rtol=0.0,
                atol=1.0e-8,
            )

        for field_name in (
            "pending",
            "velocity",
            "velocity_prev",
            "sst_k",
            "sst_omega",
            "sst_mu_t",
            "reconstructed_count",
        ):
            np.testing.assert_allclose(
                results[0][0][field_name],
                results[1][0][field_name],
                rtol=0.0,
                atol=1.0e-8,
            )
            np.testing.assert_allclose(
                results[0][1][field_name],
                results[1][1][field_name],
                rtol=0.0,
                atol=1.0e-8,
            )

    def test_fresh_fluid_sst_reconstruction_fails_closed_without_sst_donor(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        fresh = (2, 2, 2)
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[fresh] = 0
        solver.obstacle.from_numpy(obstacle)
        solver.sst_turbulent_kinetic_energy[fresh] = 0.25
        solver.sst_specific_dissipation_rate[fresh] = 1234.0
        solver.sst_eddy_viscosity_pa_s[fresh] = 1.0e-4
        solver.velocity_dirichlet_boundary_active[fresh] = 1
        solver.velocity_dirichlet_boundary_value_mps[fresh] = (7.0, 8.0, 9.0)
        solver.fresh_fluid_reconstruction_pending[fresh] = 1
        solver.hibm_fresh_fluid_cell[fresh] = 1
        solver.report_hibm_fresh_fluid_cells[None] = 1

        with self.assertRaisesRegex(RuntimeError, "without a reconstructable"):
            solver._reconstruct_fresh_fluid_cells()

        self.assertEqual(int(solver.fresh_fluid_reconstruction_pending[fresh]), 1)
        self.assertAlmostEqual(
            float(solver.sst_turbulent_kinetic_energy[fresh]), 0.25, places=6
        )
        self.assertAlmostEqual(
            float(solver.sst_specific_dissipation_rate[fresh]), 1234.0, places=3
        )
        self.assertAlmostEqual(
            float(solver.sst_eddy_viscosity_pa_s[fresh]), 1.0e-4, places=8
        )

    def test_fresh_fluid_reconstruction_advances_one_snapshot_wave_per_pass(
        self,
    ) -> None:
        for donor, first_layer, second_layer in (
            ((1, 2, 2), (2, 2, 2), (3, 2, 2)),
            ((3, 2, 2), (2, 2, 2), (1, 2, 2)),
        ):
            with self.subTest(donor=donor):
                solver = _cuda_solver(
                    FluidDomainSpec.unit_box(
                        grid_nodes=(5, 5, 5),
                        density_kgm3=1.0,
                        viscosity_pa_s=1.0e-5,
                        dt_s=1.0e-4,
                    )
                )
                solver.configure_sst_2003(
                    inlet_velocity_mps=1.0,
                    turbulence_intensity=0.05,
                    turbulent_viscosity_ratio=10.0,
                    no_slip_domain_walls=_OPEN_WALLS,
                )
                obstacle = np.ones((5, 5, 5), dtype=np.int32)
                for cell in (donor, first_layer, second_layer):
                    obstacle[cell] = 0
                solver.obstacle.from_numpy(obstacle)
                solver.velocity[donor] = (1.0, 2.0, 3.0)
                solver.velocity_prev[donor] = (-1.0, -2.0, -3.0)
                solver.sst_turbulent_kinetic_energy[donor] = 2.0
                solver.sst_specific_dissipation_rate[donor] = 3.0
                solver.sst_eddy_viscosity_pa_s[donor] = 4.0e-4
                for cell in (first_layer, second_layer):
                    solver.velocity[cell] = (91.0, 92.0, 93.0)
                    solver.velocity_prev[cell] = (-91.0, -92.0, -93.0)
                    solver.sst_turbulent_kinetic_energy[cell] = 1.0e-12
                    solver.sst_specific_dissipation_rate[cell] = 1.0e12
                    solver.sst_eddy_viscosity_pa_s[cell] = 0.0
                    solver.fresh_fluid_reconstruction_pending[cell] = 1

                solver._snapshot_fresh_fluid_reconstruction_pending_kernel()
                solver._reconstruct_hibm_fresh_fluid_cells_kernel(1)

                self.assertEqual(
                    int(solver.fresh_fluid_reconstruction_pending[first_layer]),
                    0,
                )
                self.assertEqual(
                    int(solver.fresh_fluid_reconstruction_pending[second_layer]),
                    1,
                )
                self.assertAlmostEqual(
                    float(solver.sst_turbulent_kinetic_energy[first_layer]),
                    2.0,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(solver.sst_specific_dissipation_rate[second_layer]),
                    float(np.float32(1.0e12)),
                    delta=0.0,
                )

                solver._snapshot_fresh_fluid_reconstruction_pending_kernel()
                solver._reconstruct_hibm_fresh_fluid_cells_kernel(1)

                self.assertEqual(
                    int(solver.fresh_fluid_reconstruction_pending[second_layer]),
                    0,
                )
                self.assertAlmostEqual(
                    float(solver.sst_turbulent_kinetic_energy[second_layer]),
                    2.0,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(solver.sst_specific_dissipation_rate[second_layer]),
                    3.0,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(solver.sst_eddy_viscosity_pa_s[second_layer]),
                    4.0e-4,
                    places=8,
                )

    def test_save_restore_round_trips_all_sst_physical_state(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.2,
                viscosity_pa_s=1.8e-5,
                dt_s=1.0e-3,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=2.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=(False, False, True, True, False, False),
        )
        shape = (4, 4, 4)
        ordinal = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        expected = {
            "sst_turbulent_kinetic_energy": 0.2 + 1.0e-3 * ordinal,
            "sst_specific_dissipation_rate": 10.0 + ordinal,
            "sst_eddy_viscosity_pa_s": 1.0e-4 + 1.0e-6 * ordinal,
            "sst_wall_distance_m": 0.01 + 1.0e-4 * ordinal,
        }
        for field_name, values in expected.items():
            getattr(solver, field_name).from_numpy(values)
        solver.save_state()

        for field_name in expected:
            getattr(solver, field_name).fill(9.0)
        solver.restore_state()

        for field_name, values in expected.items():
            with self.subTest(field=field_name):
                np.testing.assert_allclose(
                    getattr(solver, field_name).to_numpy(),
                    values,
                    rtol=2.0e-6,
                    atol=1.0e-7,
                )

    def test_graded_grid_wall_distance_uses_no_slip_faces_and_obstacle_surfaces(
        self,
    ) -> None:
        widths = (0.1, 0.2, 0.3, 0.4)
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=widths,
            cell_widths_y_m=widths,
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )
        solver = _cuda_solver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            )
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[2, 2, 1] = 1
        solver.obstacle.from_numpy(obstacle)
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=(False, False, True, True, False, False),
        )

        distance = solver.sst_wall_distance_m.to_numpy()
        self.assertAlmostEqual(float(distance[0, 0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(distance[0, 3, 0]), 0.20, places=6)
        # x-min is open: its 0.05 m center distance must not beat the 0.20 m
        # distance to the active y-min wall.
        self.assertAlmostEqual(float(distance[0, 1, 1]), 0.20, places=6)
        # Cell i=1 ends at x=0.30 and the adjacent obstacle starts there;
        # its center is x=0.20, so the physical obstacle-surface distance is
        # 0.10 m on this graded mesh (not an index-space distance).
        self.assertAlmostEqual(float(distance[1, 2, 1]), 0.10, places=6)
        self.assertTrue(np.all(np.isfinite(distance[obstacle == 0])))
        self.assertTrue(np.all(distance[obstacle == 0] > 0.0))

    def test_wall_distance_is_euclidean_to_diagonal_obstacle_surface(self) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, 1] = 1
        solver.obstacle.from_numpy(obstacle)

        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        expected = np.sqrt(3.0) * 0.375
        self.assertAlmostEqual(
            float(solver.sst_wall_distance_m[3, 3, 3]), expected, places=5
        )

    def test_wall_distance_uses_continuous_projection_segment_and_tip_vertex(
        self,
    ) -> None:
        solver = _cuda_solver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            )
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=4,
            projection_triangle_capacity=2,
        )
        markers.load_markers(
            positions_m=((0.5, 0.125, 0.5), (0.5, 0.9, 0.9)),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, 0.0, 1.0),) * 2,
            areas_m2=(0.1, 0.1),
            region_ids=(1, 1),
        )
        # Projection-only tip/cap vertices intentionally live beyond the
        # traction marker count.  Segment geometry must still consume them.
        markers.x_gamma_m[2] = (0.5, 0.625, 0.5)
        markers.projection_vertex_count = 3
        markers.projection_triangle_indices[0] = (0, 2, -1)
        markers.projection_segment_count = 1

        solver.prepare_sst_wall_distance(
            no_slip_domain_walls=_OPEN_WALLS,
            marker_position_m=markers.x_gamma_m,
            marker_count=int(markers.marker_count),
            projection_segment_indices=markers.projection_triangle_indices,
            projection_segment_count=int(markers.projection_segment_count),
            inactive_axis=0,
        )

        # Cell (0,1,2) is at (x,0.375,0.625).  Ignoring the extruded x axis,
        # its projection lies inside y=[0.125,0.625], so the exact distance to
        # the segment is the 0.125 m normal offset.  A point-cloud distance is
        # sqrt(0.25^2 + 0.125^2) and therefore fails this contract.
        self.assertAlmostEqual(
            float(solver.sst_wall_distance_m[0, 1, 2]), 0.125, places=6
        )


if __name__ == "__main__":
    unittest.main()
