from __future__ import annotations

import unittest
from unittest import mock

from validation_runs.ansys_vertical_flap_fsi.scripts import (
    run_traction_selected_formulation_coupled_step50 as step50,
)


class AnsysVerticalFlapStep50CliTests(unittest.TestCase):
    def test_summary_rows_do_not_end_in_whitespace(self) -> None:
        shared_row = {
            "smoke_status": "passed",
            "completed_step_count": 10,
            "requested_step_count": 10,
            "invalid_marker_count_max": 0,
            "one_sided_marker_count_min": 24,
            "anchor_selected_marker_count_min": 24,
            "anchor_fallback_marker_count_max": 0,
            "force_action_reaction_residual_max_n": 0.0,
            "max_velocity_growth_ratio": 1.0,
            "max_pressure_growth_ratio": 1.0,
            "max_displacement_growth_ratio": 1.0,
            "force_sign_flip_count": 0,
            "first_failed_gate": "",
        }
        row_by_scenario = {
            scenario: {**shared_row, "scenario": scenario}
            for scenario in (
                step50.STEP10_SCENARIO,
                step50.STEP30_SCENARIO,
                step50.STEP50_SCENARIO,
            )
        }

        rows = step50._summary_stage_rows(row_by_scenario)

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row == row.rstrip() for row in rows))

    def test_main_succeeds_only_for_the_full_step50_pass_status(self) -> None:
        expected_codes = {
            "selected_formulation_coupled_step50_passed": 0,
            "selected_formulation_coupled_step30_passed": 1,
            "selected_formulation_coupled_step10_passed": 1,
            "selected_formulation_coupled_step50_pending": 1,
            "selected_formulation_coupled_step50_failed": 1,
        }

        for candidate_status, expected_code in expected_codes.items():
            with self.subTest(candidate_status=candidate_status):
                with mock.patch.object(
                    step50,
                    "run",
                    return_value={"candidate_status": candidate_status},
                ):
                    self.assertEqual(step50.main(), expected_code)


if __name__ == "__main__":
    unittest.main()
