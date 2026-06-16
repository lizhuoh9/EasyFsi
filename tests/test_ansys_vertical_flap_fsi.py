import inspect
import math
import unittest

import cases.ansys_vertical_flap_fsi as vertical_flap_case
from cases import CASE_MODULES
from cases.ansys_vertical_flap_fsi import (
    ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS,
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS,
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


class AnsysVerticalFlapFsiSmokeTests(unittest.TestCase):
    def test_benchmark_results_must_be_computed_not_assigned_from_reference(self):
        config_fields = VerticalFlapFsiConfig.__dataclass_fields__
        self.assertNotIn("pressure_scale", config_fields)
        self.assertFalse(
            hasattr(vertical_flap_case, "_reference_equivalent_pressure_jump_pa")
        )

        run_source = inspect.getsource(vertical_flap_case.run_vertical_flap_fsi_smoke)
        self.assertNotIn("_load_reference_pressure_and_velocity", run_source)
        self.assertNotIn("pressure_jump_pa", run_source)

    def test_case_metadata_matches_ansys_tutorial_boundaries_and_targets(self):
        bc = ANSYS_VERTICAL_FLAP_BOUNDARY_CONDITIONS
        metadata = ANSYS_VERTICAL_FLAP_CASE_METADATA
        reference = ANSYS_VERTICAL_FLAP_REFERENCE_RESULTS

        self.assertAlmostEqual(metadata["geometry"]["duct_length_m"], 0.10)
        self.assertAlmostEqual(metadata["geometry"]["duct_height_m"], 0.04)
        self.assertAlmostEqual(metadata["geometry"]["modeled_height_m"], 0.02)
        self.assertEqual(metadata["geometry"]["modeled_domain"], "lower-symmetry-half")
        self.assertAlmostEqual(metadata["geometry"]["flap_height_m"], 0.01)
        self.assertAlmostEqual(metadata["geometry"]["flap_thickness_m"], 0.003)
        self.assertEqual(metadata["fluid"]["material"], "air")
        self.assertAlmostEqual(metadata["fluid"]["inlet_velocity_mps"], 10.0)
        self.assertAlmostEqual(metadata["solid"]["density_kgm3"], 1600.0)
        self.assertAlmostEqual(metadata["solid"]["young_modulus_pa"], 1.0e6)
        self.assertAlmostEqual(metadata["solid"]["poisson_ratio"], 0.47)
        self.assertEqual(metadata["time_integration"]["dt_s"], 5.0e-4)
        self.assertEqual(metadata["time_integration"]["step_count"], 50)
        self.assertAlmostEqual(metadata["time_integration"]["total_time_s"], 0.025)
        self.assertEqual(bc["inlet"]["type"], "velocity-inlet")
        self.assertAlmostEqual(bc["inlet"]["velocity_mps"], 10.0)
        self.assertEqual(bc["outlet"]["type"], "pressure-outlet")
        self.assertEqual(bc["symmetry"]["type"], "symmetry")
        self.assertEqual(bc["flap_root"]["structure"], "fixed-displacement")
        self.assertEqual(bc["flap_wall"]["coupling"], "intrinsic-two-way-fsi")
        self.assertEqual(
            CASE_MODULES["ansys-vertical-flap-fsi"],
            "cases.ansys_vertical_flap_fsi",
        )
        self.assertAlmostEqual(reference["max_displacement_m"], 5.1e-5)
        self.assertEqual(reference["time_step_s"], 5.0e-4)
        self.assertEqual(reference["step_count"], 50)
        self.assertGreaterEqual(reference["local_velocity_peak_mps"], 20.0)
        self.assertLessEqual(reference["local_velocity_peak_mps"], 29.0)

    def test_smoke_fsi_chain_matches_reference_displacement_tolerance(self):
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(step_count=50, displacement_tolerance=0.05)
        )

        self.assertEqual(report["flow_solution_mode"], "computed_projection")
        self.assertEqual(report["streamwise_axis"], "z")
        self.assertEqual(report["out_of_plane_axis"], "x")
        self.assertEqual(
            report["computed_result_sources"]["pressure_pa"], "fluid.pressure"
        )
        self.assertEqual(
            report["computed_result_sources"]["local_velocity_peak_mps"],
            "max(norm(fluid.velocity))",
        )
        self.assertEqual(
            report["computed_result_sources"]["max_displacement_m"], "solid.x-rest_x"
        )
        self.assertNotIn("pressure_jump_pa", report)
        self.assertNotIn("pressure_scale", report["config"])
        self.assertGreater(report["stress_valid_marker_count"], 0)
        self.assertEqual(report["stress_invalid_marker_count"], 0)
        self.assertEqual(report["scatter_invalid_marker_count"], 0)
        self.assertGreater(report["surface_feedback_updated_marker_count"], 0)
        self.assertLess(report["total_marker_force_n"][2], 0.0)
        self.assertLess(report["tip_mean_displacement_m"][2], 0.0)
        self.assertAlmostEqual(report["root_max_displacement_m"], 0.0, delta=1.0e-8)
        self.assertTrue(
            all(
                step["scatter_invalid_marker_count"] == 0
                and step["feedback_invalid_marker_count"] == 0
                for step in report["history"]
            )
        )
        tip_streamwise_history = [
            step["tip_mean_displacement_m"][2] for step in report["history"]
        ]
        self.assertTrue(all(value <= 0.0 for value in tip_streamwise_history))
        self.assertTrue(
            all(
                later <= earlier + 1.0e-8
                for earlier, later in zip(
                    tip_streamwise_history,
                    tip_streamwise_history[1:],
                )
            )
        )
        self.assertLess(
            max(abs(step["tip_mean_displacement_m"][0]) for step in report["history"]),
            1.0e-6,
        )
        self.assertTrue(math.isfinite(report["max_displacement_m"]))
        self.assertLessEqual(
            report["max_displacement_relative_error"],
            report["displacement_tolerance"],
        )
        self.assertLessEqual(
            report["local_velocity_peak_relative_error"],
            report["velocity_peak_tolerance"],
        )
        self.assertLess(report["scatter_action_reaction_residual_n"], 1.0e-9)


if __name__ == "__main__":
    unittest.main()
