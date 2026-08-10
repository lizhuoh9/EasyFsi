import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids.turbulence import sst_wall_correlation


_YMIN_WALL = (False, False, True, False, False, False)


def _cuda_solver() -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.225,
            viscosity_pa_s=1.5e-5,
            dt_s=1.0e-4,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _configure(
    solver: CartesianFluidSolver,
    *,
    near_wall_treatment: str = "resolved",
) -> None:
    solver.configure_sst_2003(
        inlet_velocity_mps=3.0,
        turbulence_intensity=0.05,
        turbulent_viscosity_ratio=10.0,
        no_slip_domain_walls=_YMIN_WALL,
        near_wall_treatment=near_wall_treatment,
    )


def _configure_legacy_resolved(solver: CartesianFluidSolver) -> None:
    solver.configure_sst_2003(
        inlet_velocity_mps=3.0,
        turbulence_intensity=0.05,
        turbulent_viscosity_ratio=10.0,
        no_slip_domain_walls=_YMIN_WALL,
    )


@ti.data_oriented
class _WallFaceCorrelationProbe:
    """Tiny device probe for the wall-face algebra used by SST transport."""

    def __init__(self) -> None:
        self.relative_tangential_velocity = ti.field(dtype=ti.f64, shape=2)
        self.wall_distance = ti.field(dtype=ti.f64, shape=2)
        self.turbulent_kinetic_energy = ti.field(dtype=ti.f64, shape=2)
        self.specific_dissipation_rate = ti.field(dtype=ti.f64, shape=2)
        self.result = ti.Vector.field(4, dtype=ti.f64, shape=2)

    @ti.kernel
    def evaluate(
        self,
        solver: ti.template(),
        density: ti.f64,
        kinematic_viscosity: ti.f64,
    ):
        for index in range(2):
            # Required shared device helper contract: return
            # [omega_wall, tau_wall, Pk_wall, tau/rho/U limit] in SI units.
            # wall-face path must use this same helper; a NumPy-only duplicate
            # would not satisfy this device integration contract.
            self.result[index] = solver._sst_wall_face_correlation(
                self.relative_tangential_velocity[index],
                self.wall_distance[index],
                self.turbulent_kinetic_energy[index],
                self.specific_dissipation_rate[index],
                density,
                kinematic_viscosity,
            )


@ti.data_oriented
class _WallFaceCorrelationF32Probe:
    """Exercise the wall-law algebra at the solver's production precision."""

    def __init__(self) -> None:
        self.result = ti.Vector.field(4, dtype=ti.f32, shape=())

    @ti.kernel
    def evaluate(self, solver: ti.template()):
        self.result[None] = solver._sst_wall_face_correlation(
            1.5e-25,
            1.0e-4,
            1.0e-30,
            20.0,
            1.225,
            1.5e-5,
        )


@ti.data_oriented
class _TransverseWallTermProbe:
    def __init__(self) -> None:
        self.result = ti.Vector.field(2, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate(
        self,
        solver: ti.template(),
        dt_s: ti.f64,
        kinematic_viscosity_m2_s: ti.f64,
    ):
        self.result[None] = solver._sst_momentum_transverse_boundary_face_terms(
            1,
            0,
            1,
            0,
            1,
            -1,
            dt_s,
            kinematic_viscosity_m2_s,
            1,
            2,
            0.0,
        )


class SSTWallCorrelationIntegrationContracts(unittest.TestCase):
    """Integration contracts for Fluent-style SST correlation wall treatment."""

    def test_configure_sst_correlation_is_opt_in_and_resolved_is_default(self) -> None:
        default_solver = _cuda_solver()
        _configure_legacy_resolved(default_solver)
        self.assertEqual(default_solver._sst_near_wall_treatment, "resolved")

        correlation_solver = _cuda_solver()
        _configure(correlation_solver, near_wall_treatment="fluent_correlation")
        self.assertEqual(
            correlation_solver._sst_near_wall_treatment,
            "fluent_correlation",
        )

    def test_correlation_ymin_k_diffusion_is_zero_flux_but_resolved_remains_dirichlet(
        self,
    ) -> None:
        """One y LOD factor preserves uniform k only for correlation walls."""

        correlation_solver = _cuda_solver()
        _configure(correlation_solver, near_wall_treatment="fluent_correlation")
        correlation_solver.sst_turbulent_kinetic_energy.fill(0.37)
        correlation_solver.sst_specific_dissipation_rate.fill(20.0)
        correlation_solver._update_sst_coefficients_checked(1.5e-5)
        correlation_solver._sst_lod_backward_euler_axis_kernel(
            1.0e-4,
            1.5e-5,
            1,
            1,
            0,
        )
        correlation_k_after = correlation_solver.sst_turbulent_kinetic_energy.to_numpy()
        np.testing.assert_allclose(correlation_k_after, 0.37, rtol=0.0, atol=2.0e-6)

        resolved_solver = _cuda_solver()
        _configure(resolved_solver)
        resolved_solver.sst_turbulent_kinetic_energy.fill(0.37)
        resolved_solver.sst_specific_dissipation_rate.fill(20.0)
        resolved_solver._update_sst_coefficients_checked(1.5e-5)
        resolved_solver._sst_lod_backward_euler_axis_kernel(
            1.0e-4,
            1.5e-5,
            1,
            1,
            0,
        )
        resolved_k_after = resolved_solver.sst_turbulent_kinetic_energy.to_numpy()
        self.assertLess(
            float(np.max(resolved_k_after[:, 0, :])),
            0.37 - 1.0e-5,
        )

    def test_taichi_wall_face_helper_matches_numpy_correlation_and_is_sensitive(
        self,
    ) -> None:
        solver = _cuda_solver()
        _configure_legacy_resolved(solver)
        probe = _WallFaceCorrelationProbe()
        relative_velocity = np.array([0.8, 3.2], dtype=np.float64)
        wall_distance = np.array([1.0e-4, 4.0e-4], dtype=np.float64)
        turbulent_kinetic_energy = np.array([0.04, 0.25], dtype=np.float64)
        specific_dissipation_rate = np.array([200.0, 80.0], dtype=np.float64)
        density = 1.225
        kinematic_viscosity = 1.5e-5
        probe.relative_tangential_velocity.from_numpy(relative_velocity)
        probe.wall_distance.from_numpy(wall_distance)
        probe.turbulent_kinetic_energy.from_numpy(turbulent_kinetic_energy)
        probe.specific_dissipation_rate.from_numpy(specific_dissipation_rate)

        probe.evaluate(solver, density, kinematic_viscosity)

        reference = sst_wall_correlation(
            relative_tangential_velocity=relative_velocity,
            wall_distance=wall_distance,
            turbulent_kinetic_energy=turbulent_kinetic_energy,
            specific_dissipation_rate=specific_dissipation_rate,
            density=density,
            kinematic_viscosity=kinematic_viscosity,
        )
        expected = np.stack(
            (
                reference.wall_specific_dissipation_rate,
                reference.wall_shear_stress,
                reference.wall_production,
                reference.kinematic_wall_traction_coefficient,
            ),
            axis=-1,
        )
        actual = probe.result.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=1.0e-10)
        self.assertGreater(
            float(np.max(np.abs(actual[1] - actual[0]))),
            1.0e-5,
        )

    def test_taichi_wall_face_helper_has_finite_zero_turbulence_laminar_limit(
        self,
    ) -> None:
        solver = _cuda_solver()
        _configure(solver, near_wall_treatment="fluent_correlation")
        probe = _WallFaceCorrelationProbe()
        wall_distance = np.array([2.0e-4, 4.0e-4], dtype=np.float64)
        kinematic_viscosity = 1.5e-5
        probe.relative_tangential_velocity.from_numpy(
            np.zeros(2, dtype=np.float64)
        )
        probe.wall_distance.from_numpy(wall_distance)
        probe.turbulent_kinetic_energy.from_numpy(np.zeros(2, dtype=np.float64))
        probe.specific_dissipation_rate.from_numpy(
            np.full(2, 20.0, dtype=np.float64)
        )

        probe.evaluate(solver, 1.225, kinematic_viscosity)

        actual = probe.result.to_numpy()
        expected_wall_omega = (
            (1.0 / 3.0)
            * 6.0
            * kinematic_viscosity
            / (0.075 * wall_distance**2.0)
        )
        self.assertTrue(np.all(np.isfinite(actual)), actual)
        np.testing.assert_allclose(
            actual[:, 0],
            expected_wall_omega,
            rtol=2.0e-6,
            atol=0.0,
        )
        np.testing.assert_array_equal(actual[:, 1:3], 0.0)
        np.testing.assert_allclose(
            actual[:, 3],
            kinematic_viscosity / wall_distance,
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_taichi_f32_low_y_plus_blend_is_finite_and_has_laminar_limit(
        self,
    ) -> None:
        solver = _cuda_solver()
        _configure(solver, near_wall_treatment="fluent_correlation")
        probe = _WallFaceCorrelationF32Probe()

        probe.evaluate(solver)

        actual = probe.result.to_numpy()
        self.assertTrue(np.all(np.isfinite(actual)), actual)
        # As y+ -> 0, u+ -> y+ and u*/u+ -> nu / wall_distance.
        self.assertAlmostEqual(
            float(actual[3]),
            1.5e-5 / 1.0e-4,
            delta=2.0e-5,
        )

    def test_exact_moving_domain_wall_uses_correlation_traction_and_target(self) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            # Wall identity remains explicit; the exact external ledger
            # supplies this no-slip wall's nonzero moving target.
            no_slip_domain_walls=_YMIN_WALL,
            near_wall_treatment="fluent_correlation",
        )
        relative_speed = 2.0
        wall_speed = 0.25
        stored_nearest_wall_distance = 4.0e-4
        turbulent_kinetic_energy = 0.25
        kinematic_viscosity = 1.5e-5
        dt_s = 1.0e-4
        solver.sst_turbulent_kinetic_energy.fill(turbulent_kinetic_energy)
        solver.sst_specific_dissipation_rate.fill(20.0)
        solver.sst_wall_distance_m.fill(stored_nearest_wall_distance)
        solver.sst_cell_center_velocity_mps.fill(
            (wall_speed + relative_speed, 0.0, 0.0)
        )
        for i in range(solver.nx):
            for k in range(solver.nz):
                solver.external_velocity_boundary_y_face_active_component_mask[
                    0, i, k
                ] = 1
                solver.external_velocity_boundary_y_face_value_mps[0, i, k] = (
                    wall_speed,
                    0.0,
                    0.0,
                )

        probe = _TransverseWallTermProbe()
        probe.evaluate(solver, dt_s, kinematic_viscosity)

        reference = sst_wall_correlation(
            relative_tangential_velocity=relative_speed,
            wall_distance=0.5 * float(solver.cell_width_y_m[0]),
            turbulent_kinetic_energy=turbulent_kinetic_energy,
            specific_dissipation_rate=20.0,
            density=solver.rho,
            kinematic_viscosity=kinematic_viscosity,
        )
        physical_wall_area = solver.dx * solver.dz
        expected_diagonal = (
            dt_s
            * physical_wall_area
            * float(reference.wall_shear_stress)
            / solver.rho
            / relative_speed
        )
        actual = probe.result.to_numpy()
        self.assertAlmostEqual(actual[0], expected_diagonal, delta=2.0e-10)
        self.assertAlmostEqual(
            actual[1],
            expected_diagonal * wall_speed,
            delta=2.0e-10,
        )

    def test_zero_relative_speed_retains_finite_correlation_wall_damping(self) -> None:
        solver = _cuda_solver()
        _configure(solver, near_wall_treatment="fluent_correlation")
        wall_speed = 0.25
        k_value = 0.25
        omega_value = 20.0
        kinematic_viscosity = 1.5e-5
        dt_s = 1.0e-4
        solver.sst_turbulent_kinetic_energy.fill(k_value)
        solver.sst_specific_dissipation_rate.fill(omega_value)
        solver.sst_cell_center_velocity_mps.fill((wall_speed, 0.0, 0.0))
        for i in range(solver.nx):
            for k in range(solver.nz):
                solver.external_velocity_boundary_y_face_active_component_mask[
                    0, i, k
                ] = 1
                solver.external_velocity_boundary_y_face_value_mps[0, i, k] = (
                    wall_speed,
                    0.0,
                    0.0,
                )

        probe = _TransverseWallTermProbe()
        probe.evaluate(solver, dt_s, kinematic_viscosity)

        reference = sst_wall_correlation(
            relative_tangential_velocity=0.0,
            wall_distance=0.5 * float(solver.cell_width_y_m[0]),
            turbulent_kinetic_energy=k_value,
            specific_dissipation_rate=omega_value,
            density=solver.rho,
            kinematic_viscosity=kinematic_viscosity,
        )
        expected_diagonal = (
            dt_s
            * solver.dx
            * solver.dz
            * float(reference.kinematic_wall_traction_coefficient)
        )
        actual = probe.result.to_numpy()
        self.assertGreater(actual[0], 0.0)
        self.assertAlmostEqual(actual[0], expected_diagonal, delta=2.0e-10)
        self.assertAlmostEqual(
            actual[1],
            expected_diagonal * wall_speed,
            delta=2.0e-10,
        )

    def test_wall_k_and_omega_sources_share_the_limited_correlation_production(
        self,
    ) -> None:
        solver = _cuda_solver()
        _configure(solver, near_wall_treatment="fluent_correlation")
        k_value = 0.25
        omega_value = 20.0
        dt_s = 1.0e-6
        kinematic_viscosity = 1.5e-5
        velocity = np.zeros((solver.nx, solver.ny, solver.nz, 3), dtype=np.float32)
        velocity[..., 0] = 2.0
        solver.velocity.from_numpy(velocity)
        solver.sst_turbulent_kinetic_energy.fill(k_value)
        solver.sst_specific_dissipation_rate.fill(omega_value)
        solver._update_sst_coefficients_checked(kinematic_viscosity)
        solver._copy_sst_state_to_prev_kernel()

        solver._advance_sst_transport_kernel(
            dt_s,
            kinematic_viscosity,
            0,
            solver._sst_inlet_turbulent_kinetic_energy_m2_s2,
            solver._sst_inlet_specific_dissipation_rate_s,
            solver._sst_backflow_turbulent_kinetic_energy_m2_s2,
            solver._sst_backflow_specific_dissipation_rate_s,
            solver._sst_inlet_face_code,
            solver._sst_outlet_face_code,
            0,
            0,
            1,
            0,
            0,
            0,
        )

        cell = (1, 0, 1)
        local_velocity = solver.sst_cell_center_velocity_mps.to_numpy()[cell]
        wall_distance = float(solver.sst_wall_distance_m[cell])
        reference = sst_wall_correlation(
            relative_tangential_velocity=float(
                np.linalg.norm(local_velocity[[0, 2]])
            ),
            wall_distance=wall_distance,
            turbulent_kinetic_energy=k_value,
            specific_dissipation_rate=omega_value,
            density=solver.rho,
            kinematic_viscosity=kinematic_viscosity,
        )
        limited_production = min(
            float(reference.wall_production),
            10.0 * 0.09 * solver.rho * k_value * omega_value,
        )
        expected_k = (
            k_value + dt_s * limited_production / solver.rho
        ) / (1.0 + dt_s * 0.09 * omega_value)
        mu_t = float(solver.sst_eddy_viscosity_pa_s[cell])
        gamma = float(solver.sst_gamma[cell])
        beta = float(solver.sst_beta[cell])
        expected_omega = (
            omega_value + dt_s * gamma * limited_production / mu_t
        ) / (1.0 + dt_s * beta * omega_value)

        self.assertAlmostEqual(
            float(solver.sst_turbulent_kinetic_energy_next[cell]),
            expected_k,
            delta=2.0e-6,
        )
        self.assertAlmostEqual(
            float(solver.sst_specific_dissipation_rate_next[cell]),
            expected_omega,
            delta=2.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
