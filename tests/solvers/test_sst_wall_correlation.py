from __future__ import annotations

import unittest

import numpy as np

from simulation_core.fluids.turbulence import sst_wall_correlation


def _reference_wall_correlation(
    *,
    relative_tangential_velocity: np.ndarray | float,
    wall_distance: np.ndarray | float,
    turbulent_kinetic_energy: np.ndarray | float,
    specific_dissipation_rate: np.ndarray | float,
    density: np.ndarray | float,
    kinematic_viscosity: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Independent scalar-equation reference for Fluent's correlation law."""

    delta_u, dy, k, omega, rho, nu = np.broadcast_arrays(
        np.asarray(relative_tangential_velocity, dtype=np.float64),
        np.asarray(wall_distance, dtype=np.float64),
        np.asarray(turbulent_kinetic_energy, dtype=np.float64),
        np.asarray(specific_dissipation_rate, dtype=np.float64),
        np.asarray(density, dtype=np.float64),
        np.asarray(kinematic_viscosity, dtype=np.float64),
    )
    kappa = 0.4187
    e_constant = 9.793
    c_mu = beta_star = 0.09
    beta_i = 0.075
    c_calib = 1.0 / 3.0
    c_exp = 1.3

    speed = np.abs(delta_u)
    u_star = np.sqrt(nu * speed / dy + np.sqrt(c_mu) * k)
    y_plus = dy * u_star / nu
    u_laminar_plus = y_plus
    u_turbulent_plus = np.log(e_constant * np.maximum(y_plus, 0.2)) / kappa
    u_plus = (u_laminar_plus**-4.0 + u_turbulent_plus**-4.0) ** -0.25
    u_tau = speed / u_plus
    tau_wall = rho * u_tau * u_star
    kinematic_wall_traction_coefficient = u_star / u_plus
    d_u_turbulent_plus_d_y_plus = np.where(
        y_plus > 0.2,
        1.0 / (kappa * y_plus),
        0.0,
    )
    omega_laminar_plus = c_calib * 6.0 / (beta_i * y_plus**2.0)
    omega_turbulent_plus = (
        d_u_turbulent_plus_d_y_plus / np.sqrt(beta_star)
    )
    omega_plus = omega_laminar_plus * (
        1.0 + (omega_turbulent_plus / omega_laminar_plus) ** c_exp
    ) ** (1.0 / c_exp)
    omega_wall = u_star**2.0 / nu * omega_plus
    dynamic_viscosity = rho * nu
    production_laminar = (
        rho * k / omega * (tau_wall / dynamic_viscosity) ** 2.0
    )
    production_turbulent = (
        tau_wall**2.0
        / dynamic_viscosity
        * d_u_turbulent_plus_d_y_plus
    )
    production_wall = np.divide(
        production_laminar * production_turbulent,
        production_laminar + production_turbulent,
        out=np.zeros_like(production_laminar),
        where=(production_laminar + production_turbulent) > 0.0,
    )
    return {
        "u_star": u_star,
        "y_plus": y_plus,
        "u_laminar_plus": u_laminar_plus,
        "u_turbulent_plus": u_turbulent_plus,
        "u_plus": u_plus,
        "u_tau": u_tau,
        "wall_shear_stress": tau_wall,
        "kinematic_wall_traction_coefficient": kinematic_wall_traction_coefficient,
        "omega_laminar_plus": omega_laminar_plus,
        "omega_turbulent_plus": omega_turbulent_plus,
        "omega_plus": omega_plus,
        "wall_specific_dissipation_rate": omega_wall,
        "production_laminar": production_laminar,
        "production_turbulent": production_turbulent,
        "wall_production": production_wall,
    }


class SSTWallCorrelationTests(unittest.TestCase):
    def assertMatchesReference(self, actual, expected: dict[str, np.ndarray]) -> None:
        for name, reference in expected.items():
            np.testing.assert_allclose(getattr(actual, name), reference, rtol=2.0e-13, atol=0.0)

    def test_low_y_plus_uses_laminar_log_law_branch(self) -> None:
        expected = _reference_wall_correlation(
            relative_tangential_velocity=0.01,
            wall_distance=1.0e-6,
            turbulent_kinetic_energy=1.0e-8,
            specific_dissipation_rate=40.0,
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )

        actual = sst_wall_correlation(
            relative_tangential_velocity=0.01,
            wall_distance=1.0e-6,
            turbulent_kinetic_energy=1.0e-8,
            specific_dissipation_rate=40.0,
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )

        self.assertLess(float(actual.y_plus), 0.2)
        self.assertEqual(float(actual.d_u_turbulent_plus_d_y_plus), 0.0)
        self.assertMatchesReference(actual, expected)

    def test_buffer_and_high_y_plus_match_direct_reference(self) -> None:
        expected = _reference_wall_correlation(
            relative_tangential_velocity=np.array([0.3, 25.0]),
            wall_distance=np.array([1.0e-4, 5.0e-3]),
            turbulent_kinetic_energy=np.array([0.04, 3.0]),
            specific_dissipation_rate=np.array([200.0, 80.0]),
            density=np.array([1.225, 1.225]),
            kinematic_viscosity=1.5e-5,
        )

        actual = sst_wall_correlation(
            relative_tangential_velocity=np.array([0.3, 25.0]),
            wall_distance=np.array([1.0e-4, 5.0e-3]),
            turbulent_kinetic_energy=np.array([0.04, 3.0]),
            specific_dissipation_rate=np.array([200.0, 80.0]),
            density=np.array([1.225, 1.225]),
            kinematic_viscosity=1.5e-5,
        )

        self.assertTrue(np.all(actual.y_plus > 0.2))
        self.assertMatchesReference(actual, expected)

    def test_zero_relative_tangential_velocity_has_zero_shear_and_production(self) -> None:
        expected = _reference_wall_correlation(
            relative_tangential_velocity=0.0,
            wall_distance=2.0e-4,
            turbulent_kinetic_energy=0.09,
            specific_dissipation_rate=20.0,
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )
        actual = sst_wall_correlation(
            relative_tangential_velocity=0.0,
            wall_distance=2.0e-4,
            turbulent_kinetic_energy=0.09,
            specific_dissipation_rate=20.0,
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )

        self.assertEqual(float(actual.u_tau), 0.0)
        self.assertEqual(float(actual.wall_shear_stress), 0.0)
        self.assertEqual(float(actual.wall_production), 0.0)
        self.assertGreater(float(actual.wall_specific_dissipation_rate), 0.0)
        self.assertGreater(
            float(actual.kinematic_wall_traction_coefficient),
            0.0,
        )
        np.testing.assert_allclose(
            actual.kinematic_wall_traction_coefficient,
            expected["kinematic_wall_traction_coefficient"],
            rtol=2.0e-13,
            atol=0.0,
        )

    def test_zero_turbulence_and_relative_speed_use_finite_laminar_omega_limit(
        self,
    ) -> None:
        wall_distance = 2.0e-4
        kinematic_viscosity = 1.5e-5
        beta_i = 0.075
        c_calib = 1.0 / 3.0
        expected_wall_omega = (
            c_calib
            * 6.0
            * kinematic_viscosity
            / (beta_i * wall_distance**2.0)
        )

        actual = sst_wall_correlation(
            relative_tangential_velocity=0.0,
            wall_distance=wall_distance,
            turbulent_kinetic_energy=0.0,
            specific_dissipation_rate=20.0,
            density=1.225,
            kinematic_viscosity=kinematic_viscosity,
        )

        self.assertTrue(np.isfinite(float(actual.wall_specific_dissipation_rate)))
        self.assertAlmostEqual(
            float(actual.wall_specific_dissipation_rate),
            expected_wall_omega,
            delta=expected_wall_omega * 2.0e-13,
        )
        self.assertEqual(float(actual.u_tau), 0.0)
        self.assertEqual(float(actual.wall_shear_stress), 0.0)
        self.assertEqual(float(actual.wall_production), 0.0)
        self.assertAlmostEqual(
            float(actual.kinematic_wall_traction_coefficient),
            kinematic_viscosity / wall_distance,
            delta=(kinematic_viscosity / wall_distance) * 2.0e-13,
        )

    def test_wall_production_uses_cell_omega_not_wall_omega(self) -> None:
        common = {
            "relative_tangential_velocity": 3.2,
            "wall_distance": 4.0e-4,
            "turbulent_kinetic_energy": 0.25,
            "density": 1.225,
            "kinematic_viscosity": 1.5e-5,
        }

        low_omega = sst_wall_correlation(
            **common,
            specific_dissipation_rate=20.0,
        )
        high_omega = sst_wall_correlation(
            **common,
            specific_dissipation_rate=200.0,
        )

        self.assertAlmostEqual(
            float(low_omega.wall_specific_dissipation_rate),
            float(high_omega.wall_specific_dissipation_rate),
        )
        self.assertGreater(
            float(low_omega.production_laminar),
            float(high_omega.production_laminar),
        )
        self.assertGreater(
            float(low_omega.wall_production),
            float(high_omega.wall_production),
        )

    def test_broadcasts_inputs_and_returns_read_only_arrays(self) -> None:
        actual = sst_wall_correlation(
            relative_tangential_velocity=np.array([[0.3], [0.6]]),
            wall_distance=np.array([1.0e-4, 2.0e-4, 3.0e-4]),
            turbulent_kinetic_energy=0.04,
            specific_dissipation_rate=30.0,
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )

        self.assertEqual(actual.y_plus.shape, (2, 3))
        self.assertFalse(actual.y_plus.flags.writeable)
        with self.assertRaises(ValueError):
            actual.y_plus[0, 0] = 0.0

    def test_rejects_nonfinite_and_nonpositive_physical_inputs(self) -> None:
        baseline = {
            "relative_tangential_velocity": 1.0,
            "wall_distance": 1.0e-4,
            "turbulent_kinetic_energy": 0.1,
            "specific_dissipation_rate": 20.0,
            "density": 1.225,
            "kinematic_viscosity": 1.5e-5,
        }
        for name, value in (
            ("relative_tangential_velocity", np.nan),
            ("wall_distance", 0.0),
            ("turbulent_kinetic_energy", -1.0e-12),
            ("specific_dissipation_rate", 0.0),
            ("density", -1.0),
            ("kinematic_viscosity", np.inf),
        ):
            with self.subTest(name=name):
                inputs = dict(baseline)
                inputs[name] = value
                with self.assertRaises(ValueError):
                    sst_wall_correlation(**inputs)


if __name__ == "__main__":
    unittest.main()
