from __future__ import annotations

import math
import unittest

from cases.comsol_multibody_mechanism_fsi import (
    CASE_SPEC,
    COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS,
    COMSOL_MULTIBODY_MECHANISM_CASE_METADATA,
    OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS,
    MultibodyMechanismFsiConfig,
    run_multibody_mechanism_fsi_smoke,
)


class ComsolMultibodyMechanismFsiCaseTests(unittest.TestCase):
    def test_case_spec_preserves_official_pair_and_motion_conditions(self) -> None:
        bc = COMSOL_MULTIBODY_MECHANISM_BOUNDARY_CONDITIONS
        metadata = COMSOL_MULTIBODY_MECHANISM_CASE_METADATA

        self.assertEqual(CASE_SPEC.case_id, "comsol-multibody-mechanism-fsi")
        self.assertEqual(CASE_SPEC.coordinate_model, "cartesian-3d")
        self.assertEqual(CASE_SPEC.acceptance_tolerance, 0.05)
        self.assertEqual(metadata["interface"]["type"], "FSI Pair")
        self.assertEqual(metadata["study"]["duration_s"], 1.0)
        self.assertAlmostEqual(metadata["solid"]["maximum_fin_rotation_rad"], math.radians(15.0))
        self.assertAlmostEqual(metadata["solid"]["rotation_ramp_duration_s"], 0.25)
        self.assertEqual(bc["fluid_domain"]["pressure_anchor"], "point-constraint-zero")
        self.assertEqual(bc["solid_to_mesh_full_displacement"]["components"], "all")
        self.assertEqual(bc["rear_curved_faces"]["components"], "normal-only")
        self.assertEqual(bc["fins"]["final_angle_deg"], 15.0)

    def test_smoke_computes_pair_displacement_force_and_forward_velocity(self) -> None:
        report = run_multibody_mechanism_fsi_smoke(
            MultibodyMechanismFsiConfig(
                step_count=6,
                dt_s=0.05,
                active_until_s=0.25,
                max_fin_rotation_deg=15.0,
            )
        )

        self.assertEqual(report["case"], "comsol-multibody-mechanism-fsi")
        self.assertEqual(
            report["computed_result_sources"]["fin_rotation_rad"],
            "EqualOppositeHingePair.angles_at(time)",
        )
        self.assertGreater(report["max_abs_fin_rotation_rad"], 0.0)
        self.assertGreater(report["max_mesh_displacement_m"], 0.0)
        self.assertGreater(report["max_fluid_force_norm_n"], 0.0)
        self.assertNotEqual(report["final_forward_velocity_mps"], 0.0)
        self.assertLessEqual(report["max_action_reaction_relative_error"], 1.0e-12)

    def test_default_smoke_final_velocity_matches_official_figure7_within_five_percent(
        self,
    ) -> None:
        report = run_multibody_mechanism_fsi_smoke(MultibodyMechanismFsiConfig())

        reference_final_mps = (
            OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS.value_at(1.0) / 100.0
        )
        relative_error = abs(
            float(report["final_forward_velocity_mps"]) - reference_final_mps
        ) / reference_final_mps

        self.assertLessEqual(relative_error, CASE_SPEC.acceptance_tolerance)

    def test_default_smoke_early_velocity_is_not_one_step_late_against_figure7(
        self,
    ) -> None:
        report = run_multibody_mechanism_fsi_smoke(MultibodyMechanismFsiConfig())

        first_sample = next(
            row for row in report["history"] if math.isclose(float(row["time_s"]), 0.05)
        )
        computed_cmps = float(first_sample["forward_velocity_mps"]) * 100.0
        relative_error = OFFICIAL_FORWARD_VELOCITY_REFERENCE_CMPS.relative_error_at(
            time_s=0.05,
            computed_value=computed_cmps,
        )

        self.assertLessEqual(relative_error, CASE_SPEC.acceptance_tolerance)

    def test_default_smoke_keeps_computed_fluid_drag_after_fin_motion_stops(self) -> None:
        report = run_multibody_mechanism_fsi_smoke(MultibodyMechanismFsiConfig())

        post_actuation = [
            row
            for row in report["history"]
            if float(row["time_s"]) > COMSOL_MULTIBODY_MECHANISM_CASE_METADATA[
                "solid"
            ]["rotation_ramp_duration_s"]
        ]

        self.assertTrue(post_actuation)
        self.assertTrue(
            any(abs(row["body_drag_force_n"][0]) > 0.0 for row in post_actuation)
        )


if __name__ == "__main__":
    unittest.main()
