from __future__ import annotations

import math
import unittest

from simulation_core.coupling.fsi_coupling import _least_squares_coefficients
from simulation_core.diagnostics.validation import (
    ReferenceCurve,
    force_nonzero_when_loaded,
    vector_norm,
)
from simulation_core.materials.hyperelastic import (
    NeoHookeanMaterial,
    incompressible_uniaxial_nominal_stress_pa,
)
from simulation_core.solids.mooney_shell.core import (
    _raise_if_out_of_bounds_exceeds_tolerance,
)


class MainAuditNumericContractTests(unittest.TestCase):
    def test_iqn_least_squares_is_scale_invariant_for_large_finite_values(self) -> None:
        coefficients = _least_squares_coefficients(
            ((1.0e200, 0.0), (0.0, 1.0e200)),
            (1.0e200, 1.0e200),
        )

        self.assertIsNotNone(coefficients)
        assert coefficients is not None
        self.assertAlmostEqual(coefficients[0], 1.0)
        self.assertAlmostEqual(coefficients[1], 1.0)

    def test_validation_helpers_reject_nonfinite_queries_and_negative_tolerance(self) -> None:
        curve = ReferenceCurve(
            name="force",
            units="N",
            source="unit test",
            points=((0.0, 0.0), (1.0, 1.0)),
        )

        with self.assertRaisesRegex(ValueError, "time_s must be finite"):
            curve.value_at(math.nan)
        with self.assertRaisesRegex(ValueError, "computed_value must be finite"):
            curve.relative_error_at(time_s=0.5, computed_value=math.inf)
        with self.assertRaisesRegex(ValueError, "tolerance_n"):
            force_nonzero_when_loaded(
                force_components_n=(0.0, 0.0, 0.0),
                load_value=1.0,
                force_required=True,
                tolerance_n=-1.0,
            )
        self.assertTrue(math.isfinite(vector_norm((1.0e200, 1.0e200))))

    def test_hyperelastic_public_inputs_reject_nonfinite_physical_values(self) -> None:
        material = NeoHookeanMaterial(
            name="invalid",
            density_kgm3=math.nan,
            shear_modulus_pa=1.0,
            bulk_modulus_pa=2.0,
            youngs_modulus_pa=1.0,
            poissons_ratio=0.3,
        )

        with self.assertRaisesRegex(ValueError, "density_kgm3"):
            material.validate()
        with self.assertRaisesRegex(ValueError, "stretch"):
            incompressible_uniaxial_nominal_stress_pa(math.nan, 1.0)
        with self.assertRaisesRegex(ValueError, "shear_modulus_pa"):
            incompressible_uniaxial_nominal_stress_pa(1.0, math.inf)

    def test_all_particles_out_of_bounds_always_fails_even_with_high_tolerance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "3 of 3 MPM particles"):
            _raise_if_out_of_bounds_exceeds_tolerance(3, 3, tolerance=3)


if __name__ == "__main__":
    unittest.main()
