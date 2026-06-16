from __future__ import annotations

import unittest


class MembraneInflationFsiTests(unittest.TestCase):
    def test_uv_membrane_inflation_computes_pressure_volume_and_stretch(self) -> None:
        from simulation_core.benchmarking.membrane_inflation_fsi import (
            MembraneInflationConfig,
            run_uv_membrane_inflation_smoke,
        )

        report = run_uv_membrane_inflation_smoke(
            MembraneInflationConfig(
                radius_m=0.03,
                thickness_m=0.001,
                density_kgm3=1000.0,
                c1_pa=2.0e5,
                c2_pa=5.0e4,
                inlet_flow_m3s=1.0e-5,
                fill_duration_s=0.01,
                dt_s=2.0e-4,
                step_count=10,
                pressure_bulk_modulus_pa=2.0e5,
                latitude_bands=4,
                longitude_segments=8,
                grid_nodes=(12, 12, 12),
            )
        )

        self.assertEqual(report["case"], "generic-uv-membrane-inflation-fsi")
        self.assertEqual(
            report["computed_result_sources"]["pressure_pa"],
            "bulk_modulus * (target_volume-current_volume) / rest_volume",
        )
        self.assertGreater(report["final_target_volume_m3"], report["rest_volume_m3"])
        self.assertGreater(report["final_volume_m3"], report["rest_volume_m3"])
        self.assertGreater(report["final_pressure_pa"], 0.0)
        self.assertGreater(report["max_mean_radial_stretch"], 1.0)
        self.assertEqual(len(report["history"]), 10)


if __name__ == "__main__":
    unittest.main()
