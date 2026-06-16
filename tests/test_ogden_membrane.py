from __future__ import annotations

import unittest

from simulation_core.benchmarking.ogden_membrane import (
    OgdenMembraneMaterial,
    stretch_from_volume_ratio,
)


class OgdenMembraneTests(unittest.TestCase):
    def test_equibiaxial_stress_uses_material_terms(self) -> None:
        material = OgdenMembraneMaterial.from_sequences(
            alpha=(1.3, 5.0, -2.0),
            shear_modulus_pa=(6.3e5, 0.012e5, -0.1e5),
        )

        self.assertAlmostEqual(material.equibiaxial_cauchy_stress_pa(1.0), 0.0)
        self.assertGreater(material.equibiaxial_cauchy_stress_pa(1.5), 0.0)

    def test_stretch_from_volume_ratio_is_computed(self) -> None:
        self.assertAlmostEqual(
            stretch_from_volume_ratio(current_volume_m3=8.0, rest_volume_m3=1.0),
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
