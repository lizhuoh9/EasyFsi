from __future__ import annotations

import unittest

from cases.comsol_water_balloon_fsi import (
    CASE_SPEC,
    COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS,
    COMSOL_WATER_BALLOON_CASE_METADATA,
    COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA,
    COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML,
    WaterBalloonFsiConfig,
    run_water_balloon_fsi_smoke,
)


class ComsolWaterBalloonFsiCaseTests(unittest.TestCase):
    def test_case_spec_preserves_official_boundary_conditions(self) -> None:
        bc = COMSOL_WATER_BALLOON_BOUNDARY_CONDITIONS
        metadata = COMSOL_WATER_BALLOON_CASE_METADATA

        self.assertEqual(CASE_SPEC.case_id, "comsol-water-balloon-fsi")
        self.assertEqual(CASE_SPEC.coordinate_model, "axisymmetric-2d")
        self.assertEqual(CASE_SPEC.acceptance_tolerance, 0.05)
        self.assertEqual(bc["inlet"]["type"], "time-windowed-velocity-inlet")
        self.assertAlmostEqual(bc["inlet"]["peak_velocity_mps"], 0.15)
        self.assertAlmostEqual(bc["inlet"]["nominal_flow_lpm"], 1.4)
        self.assertEqual(bc["axis"]["type"], "axisymmetry")
        self.assertEqual(bc["fluid_structure_interface"]["type"], "two-way-fsi")
        self.assertTrue(bc["gravity"]["enabled"])
        self.assertEqual(metadata["solid"]["model"], "Ogden hyperelastic membrane")
        self.assertEqual(metadata["study"]["parametric_sweep"], "initial-size-factor")
        self.assertEqual(metadata["study"]["case_count"], 3)
        self.assertFalse(metadata["reference_results"]["extraction_required"])

    def test_case_spec_contains_official_water_content_reference(self) -> None:
        self.assertEqual(
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.source,
            "COMSOL water_balloon_solved.mph Probe Table 1, dom1 water content",
        )
        self.assertEqual(
            CASE_SPEC.reference_results["final_water_content_ml"],
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.value_at(15.0),
        )
        self.assertAlmostEqual(
            CASE_SPEC.reference_results["initial_water_content_ml"],
            COMSOL_WATER_BALLOON_WATER_CONTENT_REFERENCE_ML.value_at(0.0),
        )
        self.assertEqual(
            CASE_SPEC.reference_results["max_mises_stress_pa"],
            COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA,
        )

    def test_smoke_uses_generic_membrane_inflation_runner(self) -> None:
        report = run_water_balloon_fsi_smoke(
            WaterBalloonFsiConfig(
                step_count=6,
                dt_s=2.0e-4,
                fill_start_s=0.0,
                fill_duration_s=0.005,
                latitude_bands=4,
                longitude_segments=8,
                grid_nodes=(12, 12, 12),
            )
        )

        self.assertEqual(report["case"], "comsol-water-balloon-fsi")
        self.assertNotIn("reference_max_displacement_m", report)
        self.assertGreater(report["final_target_volume_m3"], report["rest_volume_m3"])
        self.assertGreater(report["final_volume_m3"], report["rest_volume_m3"])
        self.assertGreater(
            max(float(row["pressure_pa"]) for row in report["history"]),
            0.0,
        )
        self.assertEqual(
            report["computed_result_sources"]["pressure_pa"],
            "bulk_modulus * (target_volume-current_volume) / rest_volume",
        )

    def test_water_content_observables_are_computed_against_official_curve(self) -> None:
        report = run_water_balloon_fsi_smoke(
            WaterBalloonFsiConfig(
                initial_size_factor=2.0,
                step_count=3,
                dt_s=2.0e-4,
                latitude_bands=4,
                longitude_segments=8,
                grid_nodes=(12, 12, 12),
            )
        )

        self.assertEqual(
            report["computed_result_sources"]["initial_water_content_ml"],
            "Taichi axisymmetric profile volume integral",
        )
        self.assertEqual(
            report["computed_result_sources"]["final_water_content_ml"],
            "Taichi integral of inlet_velocity * inlet_area over time",
        )
        self.assertLessEqual(
            report["volume_reference_errors"]["initial_water_content_ml"],
            CASE_SPEC.acceptance_tolerance,
        )
        self.assertLessEqual(
            report["volume_reference_errors"]["final_water_content_ml"],
            CASE_SPEC.acceptance_tolerance,
        )
        self.assertTrue(report["official_volume_reference_passed"])
        for relative_error in report["water_content_curve_relative_errors"].values():
            self.assertLessEqual(relative_error, CASE_SPEC.acceptance_tolerance)

    def test_structure_stress_reference_uses_local_axisymmetric_profile(self) -> None:
        report = run_water_balloon_fsi_smoke(
            WaterBalloonFsiConfig(
                initial_size_factor=2.0,
                step_count=3,
                dt_s=2.0e-4,
                latitude_bands=4,
                longitude_segments=8,
                grid_nodes=(12, 12, 12),
            )
        )

        self.assertEqual(
            report["computed_result_sources"]["global_equibiaxial_mises_stress_pa"],
            "Ogden membrane stress from computed volume stretch",
        )
        self.assertEqual(
            report["computed_result_sources"]["local_axisymmetric_mises_stress_pa"],
            "Taichi local Ogden stress from smooth axisymmetric neck/ellipse profile",
        )
        self.assertGreater(report["global_equibiaxial_stretch"], 1.0)
        self.assertGreater(report["global_equibiaxial_mises_stress_pa"], 0.0)
        self.assertGreater(
            report["local_axisymmetric_mises_stress_pa"],
            report["global_equibiaxial_mises_stress_pa"],
        )
        self.assertGreater(report["local_axisymmetric_blend_exponent"], 2.0)
        self.assertAlmostEqual(
            report["local_axisymmetric_fillet_radius_m"],
            0.013333333333333334,
        )
        self.assertGreater(report["local_axisymmetric_max_meridional_stretch"], 1.0)
        self.assertEqual(
            report["official_max_mises_stress_pa"],
            COMSOL_WATER_BALLOON_MAX_MISES_STRESS_PA,
        )
        self.assertLess(
            report["structure_reference_errors"]["max_mises_stress_pa"],
            report["global_equibiaxial_mises_stress_relative_error"],
        )
        self.assertLessEqual(
            report["structure_reference_errors"]["max_mises_stress_pa"],
            CASE_SPEC.acceptance_tolerance,
        )
        self.assertTrue(report["official_structure_reference_passed"])
        self.assertTrue(report["official_reference_passed"])
        self.assertIn("official fillet geometry", report["structure_reference_gap"])


if __name__ == "__main__":
    unittest.main()
