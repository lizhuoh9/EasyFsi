from __future__ import annotations

import unittest

import numpy as np

from simulation_core.fluids.turbulence import (
    SST2003_CONSTANTS,
    SSTConstants,
    blend_sst_coefficients,
    sst_blending_functions,
    sst_eddy_viscosity,
    sst_local_source_step,
    sst_production_limiter,
    validate_sst_state,
)


class SSTTurbulenceModelTests(unittest.TestCase):
    def test_sst_2003_constants_and_coefficient_blending(self) -> None:
        constants = SST2003_CONSTANTS

        self.assertAlmostEqual(constants.beta_star, 0.09)
        self.assertAlmostEqual(constants.a1, 0.31)
        self.assertAlmostEqual(constants.gamma_1, 5.0 / 9.0)
        self.assertAlmostEqual(constants.gamma_2, 0.44)
        self.assertAlmostEqual(constants.production_limit_factor, 10.0)

        blended = blend_sst_coefficients(np.array([1.0, 0.0, 0.25]))
        np.testing.assert_allclose(blended.sigma_k, [0.85, 1.0, 0.9625])
        np.testing.assert_allclose(blended.sigma_omega, [0.5, 0.856, 0.767])
        np.testing.assert_allclose(blended.beta, [0.075, 0.0828, 0.08085])
        np.testing.assert_allclose(
            blended.gamma,
            [5.0 / 9.0, 0.44, 0.25 * (5.0 / 9.0) + 0.75 * 0.44],
        )

    def test_blending_functions_match_sst_2003_definition_for_arrays(self) -> None:
        k = np.array([0.5, 0.8])
        omega = np.array([2.0, 4.0])
        wall_distance = np.array([0.1, 0.4])
        nu = np.array([1.0e-5, 1.5e-5])
        density = np.array([1.0, 1.2])
        grad_dot = np.array([0.2, -0.3])

        result = sst_blending_functions(
            turbulent_kinetic_energy=k,
            specific_dissipation_rate=omega,
            wall_distance=wall_distance,
            kinematic_viscosity=nu,
            density=density,
            grad_k_dot_grad_omega=grad_dot,
        )

        c = SST2003_CONSTANTS
        cd_kw = np.maximum(
            2.0 * density * c.sigma_omega_2 * grad_dot / omega,
            c.cd_kw_floor,
        )
        arg1 = np.minimum(
            np.maximum(
                np.sqrt(k) / (c.beta_star * omega * wall_distance),
                500.0 * nu / (wall_distance**2 * omega),
            ),
            4.0 * density * c.sigma_omega_2 * k / (cd_kw * wall_distance**2),
        )
        arg2 = np.maximum(
            2.0 * np.sqrt(k) / (c.beta_star * omega * wall_distance),
            500.0 * nu / (wall_distance**2 * omega),
        )

        np.testing.assert_allclose(result.cd_kw, cd_kw)
        np.testing.assert_allclose(result.f1, np.tanh(arg1**4))
        np.testing.assert_allclose(result.f2, np.tanh(arg2**2))
        self.assertTrue(np.all((result.f1 >= 0.0) & (result.f1 <= 1.0)))
        self.assertTrue(np.all((result.f2 >= 0.0) & (result.f2 <= 1.0)))

    def test_eddy_viscosity_uses_shear_stress_transport_limiter(self) -> None:
        result = sst_eddy_viscosity(
            density=np.array([2.0, 2.0]),
            turbulent_kinetic_energy=np.array([3.0, 3.0]),
            specific_dissipation_rate=np.array([4.0, 4.0]),
            strain_rate_magnitude=np.array([0.1, 10.0]),
            f2=np.ones(2),
        )

        np.testing.assert_allclose(result, [1.5, 0.186])

    def test_production_limiter_caps_both_transport_equations(self) -> None:
        limited = sst_production_limiter(
            production=np.array([1.0, 1000.0]),
            density=2.0,
            turbulent_kinetic_energy=3.0,
            specific_dissipation_rate=4.0,
        )

        np.testing.assert_allclose(limited, [1.0, 21.6])

    def test_local_source_step_matches_near_wall_reference_values(self) -> None:
        result = sst_local_source_step(
            turbulent_kinetic_energy=np.array([0.5]),
            specific_dissipation_rate=np.array([2.0]),
            density=np.array([1.0]),
            kinematic_viscosity=np.array([1.0e-5]),
            wall_distance=np.array([0.1]),
            strain_rate_magnitude=np.array([3.0]),
            grad_k_dot_grad_omega=np.array([0.0]),
            dt_s=0.01,
        )

        # At this near-wall point F1 and F2 are saturated to one.  Therefore
        # mu_t = 0.31*0.5/3, P = mu_t*3^2, and the cross term is zero.
        np.testing.assert_allclose(result.f1, [1.0], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.f2, [1.0], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.eddy_viscosity, [0.051666666666666666])
        np.testing.assert_allclose(result.raw_production, [0.465])
        np.testing.assert_allclose(result.limited_production, [0.465])
        np.testing.assert_allclose(result.k_source, [0.375])
        np.testing.assert_allclose(result.omega_source, [4.7])
        np.testing.assert_allclose(result.k_next, [0.50375])
        np.testing.assert_allclose(result.omega_next, [2.047])

    def test_local_source_step_includes_outer_cross_diffusion(self) -> None:
        result = sst_local_source_step(
            turbulent_kinetic_energy=np.array([0.5]),
            specific_dissipation_rate=np.array([2.0]),
            density=np.array([1.0]),
            kinematic_viscosity=np.array([1.0e-5]),
            wall_distance=np.array([1.0e6]),
            strain_rate_magnitude=np.array([3.0]),
            grad_k_dot_grad_omega=np.array([0.2]),
            dt_s=0.01,
        )

        self.assertLess(float(result.f1[0]), 1.0e-30)
        self.assertLess(float(result.f2[0]), 1.0e-9)
        np.testing.assert_allclose(result.eddy_viscosity, [0.25], rtol=1.0e-12)
        np.testing.assert_allclose(result.raw_production, [2.25], rtol=1.0e-12)
        np.testing.assert_allclose(result.limited_production, [0.9], rtol=1.0e-12)
        np.testing.assert_allclose(result.k_source, [0.81], rtol=1.0e-12)
        np.testing.assert_allclose(result.omega_source, [1.424], rtol=1.0e-12)
        np.testing.assert_allclose(result.k_next, [0.5081], rtol=1.0e-12)
        np.testing.assert_allclose(result.omega_next, [2.01424], rtol=1.0e-12)

    def test_array_inputs_are_not_mutated_and_return_arrays_are_read_only(self) -> None:
        k = np.array([0.5, 0.8])
        omega = np.array([2.0, 4.0])
        k_before = k.copy()
        omega_before = omega.copy()

        result = sst_local_source_step(
            turbulent_kinetic_energy=k,
            specific_dissipation_rate=omega,
            density=1.0,
            kinematic_viscosity=1.0e-5,
            wall_distance=0.1,
            strain_rate_magnitude=1.0,
            grad_k_dot_grad_omega=0.0,
            dt_s=1.0e-3,
        )

        np.testing.assert_array_equal(k, k_before)
        np.testing.assert_array_equal(omega, omega_before)
        for value in vars(result).values():
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(ValueError):
            result.k_next[0] = 0.0

    def test_state_validation_rejects_non_physical_or_non_finite_values(self) -> None:
        invalid_cases = (
            ({"turbulent_kinetic_energy": [-1.0]}, "turbulent_kinetic_energy"),
            ({"specific_dissipation_rate": [0.0]}, "specific_dissipation_rate"),
            ({"density": [-1.0]}, "density"),
            ({"kinematic_viscosity": [0.0]}, "kinematic_viscosity"),
            ({"wall_distance": [0.0]}, "wall_distance"),
            ({"turbulent_kinetic_energy": [np.nan]}, "turbulent_kinetic_energy"),
            ({"specific_dissipation_rate": [np.inf]}, "specific_dissipation_rate"),
        )

        defaults = {
            "turbulent_kinetic_energy": [0.5],
            "specific_dissipation_rate": [2.0],
            "density": [1.0],
            "kinematic_viscosity": [1.0e-5],
            "wall_distance": [0.1],
        }
        for override, field_name in invalid_cases:
            with self.subTest(field_name=field_name, value=next(iter(override.values()))):
                values = {**defaults, **override}
                with self.assertRaisesRegex(ValueError, field_name):
                    validate_sst_state(**values)

    def test_constants_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "a1"):
            SSTConstants(a1=float("nan"))
        with self.assertRaisesRegex(ValueError, "production_limit_factor"):
            SSTConstants(production_limit_factor=0.0)

    def test_local_step_rejects_time_step_that_breaks_positivity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive specific_dissipation_rate"):
            sst_local_source_step(
                turbulent_kinetic_energy=np.array([0.5]),
                specific_dissipation_rate=np.array([2.0]),
                density=1.0,
                kinematic_viscosity=1.0e-5,
                wall_distance=0.1,
                strain_rate_magnitude=0.0,
                grad_k_dot_grad_omega=0.0,
                dt_s=10.0,
            )

    def test_zero_k_uses_continuous_omega_production_limit(self) -> None:
        result = sst_local_source_step(
            turbulent_kinetic_energy=np.array([0.0]),
            specific_dissipation_rate=np.array([2.0]),
            density=1.0,
            kinematic_viscosity=1.0e-5,
            wall_distance=0.1,
            strain_rate_magnitude=3.0,
            grad_k_dot_grad_omega=0.0,
            dt_s=0.01,
        )

        # mu_t and limited P both vanish linearly with k.  P/mu_t must
        # therefore use the k->0 continuous limit instead of being set to 0.
        np.testing.assert_allclose(result.eddy_viscosity, [0.0])
        np.testing.assert_allclose(result.limited_production, [0.0])
        np.testing.assert_allclose(result.omega_source, [1.2528])
        np.testing.assert_allclose(result.omega_next, [2.012528])


if __name__ == "__main__":
    unittest.main()
