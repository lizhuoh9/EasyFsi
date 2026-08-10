from __future__ import annotations

import unittest
from unittest import mock

from validation_runs.ansys_vertical_flap_fsi.scripts import (
    run_traction_selected_formulation_coupled_step50 as step50,
)


class AnsysVerticalFlapStep50CliTests(unittest.TestCase):
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
