from __future__ import annotations

import math
import unittest

from simulation_core import (
    NeoHookeanStressProbe,
    TaichiRuntimeConfig,
    ecoflex_0010_material,
    incompressible_uniaxial_nominal_stress_pa,
    psi_to_pa,
)


class EcoflexHyperelasticTests(unittest.TestCase):
    def test_datasheet_unit_conversion_and_default_material(self) -> None:
        material = ecoflex_0010_material()

        self.assertAlmostEqual(psi_to_pa(8.0), 55158.058345344, places=6)
        self.assertEqual(material.name, "Ecoflex 00-10")
        self.assertEqual(material.density_kgm3, 1040.0)
        self.assertEqual(material.shore_hardness, "Shore 00-10")
        self.assertAlmostEqual(material.modulus_100_pa, psi_to_pa(8.0), places=5)
        self.assertAlmostEqual(material.tensile_strength_pa, psi_to_pa(120.0), places=4)
        self.assertEqual(material.elongation_at_break_percent, 800.0)
        self.assertGreater(material.shear_modulus_pa, 0.0)
        self.assertGreater(material.bulk_modulus_pa, material.shear_modulus_pa)
        self.assertGreater(material.stable_explicit_dt_s(1.0e-4), 0.0)

    def test_100_percent_modulus_calibrates_uniaxial_nominal_stress(self) -> None:
        material = ecoflex_0010_material()
        nominal_stress_pa = incompressible_uniaxial_nominal_stress_pa(
            stretch=2.0,
            shear_modulus_pa=material.shear_modulus_pa,
        )

        self.assertAlmostEqual(nominal_stress_pa, material.modulus_100_pa, delta=1.0e-3)

    def test_stress_probe_has_zero_reference_stress(self) -> None:
        probe = NeoHookeanStressProbe(
            ecoflex_0010_material(),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = probe.evaluate_diagonal_stretch((1.0, 1.0, 1.0))

        self.assertAlmostEqual(report.jacobian, 1.0, places=6)
        self.assertAlmostEqual(report.strain_energy_density_jm3, 0.0, delta=1.0e-3)
        self.assertAlmostEqual(report.first_piola_x_pa, 0.0, delta=1.0e-3)
        self.assertAlmostEqual(report.cauchy_x_pa, 0.0, delta=1.0e-3)

    def test_stress_probe_reports_positive_incompressible_tension(self) -> None:
        material = ecoflex_0010_material()
        probe = NeoHookeanStressProbe(material, runtime=TaichiRuntimeConfig(arch="cuda"))
        transverse = 1.0 / math.sqrt(2.0)

        report = probe.evaluate_diagonal_stretch((2.0, transverse, transverse))

        self.assertAlmostEqual(report.jacobian, 1.0, delta=1.0e-5)
        self.assertGreater(report.strain_energy_density_jm3, 0.0)
        self.assertGreater(report.first_piola_x_pa, 0.0)
        self.assertGreater(report.cauchy_x_pa, report.first_piola_x_pa)


if __name__ == "__main__":
    unittest.main()
