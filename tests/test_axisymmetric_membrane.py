from __future__ import annotations

import unittest

from simulation_core.benchmarking.axisymmetric_membrane import (
    SmoothAxisymmetricMembraneStressConfig,
    smooth_axisymmetric_membrane_stress_report,
)
from simulation_core.benchmarking.ogden_membrane import (
    OgdenMembraneMaterial,
    stretch_from_volume_ratio,
)


class AxisymmetricMembraneStressTests(unittest.TestCase):
    def test_smooth_profile_reports_local_stress_from_target_volume(self) -> None:
        initial_volume_m3 = 4.7115599468997e-5
        target_volume_m3 = 1.53260483578228e-4
        report = smooth_axisymmetric_membrane_stress_report(
            SmoothAxisymmetricMembraneStressConfig(
                neck_radius_m=0.005,
                initial_height_m=0.08,
                final_height_m=0.08,
                initial_bulb_radius_m=0.02,
                target_volume_m3=target_volume_m3,
                ogden_alpha=(1.3, 5.0, -2.0),
                ogden_shear_modulus_pa=(6.3e5, 0.012e5, -0.1e5),
                sample_count=512,
            )
        )
        global_stretch = stretch_from_volume_ratio(
            current_volume_m3=target_volume_m3,
            rest_volume_m3=initial_volume_m3,
        )
        global_stress_pa = OgdenMembraneMaterial.from_sequences(
            alpha=(1.3, 5.0, -2.0),
            shear_modulus_pa=(6.3e5, 0.012e5, -0.1e5),
        ).equibiaxial_cauchy_stress_pa(global_stretch)

        self.assertAlmostEqual(report.final_volume_m3, target_volume_m3, delta=2.0e-8)
        self.assertGreater(report.final_bulb_radius_m, 0.02)
        self.assertGreater(report.max_meridional_stretch, 1.0)
        self.assertGreater(report.max_mises_stress_pa, global_stress_pa)


if __name__ == "__main__":
    unittest.main()
