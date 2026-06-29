from __future__ import annotations

import unittest
from pathlib import Path


STATUS_REPORT = (
    Path("validation_runs")
    / "ansys_vertical_flap_fsi"
    / "real_solver_validation_status"
    / "real_solver_validation_status_2026-07-01.md"
)


class AnsysVerticalFlapRealSolverValidationStatusTests(unittest.TestCase):
    def test_status_report_records_solver_and_fluent_boundaries(self):
        text = STATUS_REPORT.read_text(encoding="utf-8")

        self.assertIn("generic_solver_selected_formulation_step50_passed", text)
        self.assertIn("selected_formulation_coupled_step50_timeout", text)
        self.assertIn("real_fluent_bundle_unavailable", text)
        self.assertIn("fluent_parity_claimed: false", text)
        self.assertIn("No source_exports promotion was performed", text)


if __name__ == "__main__":
    unittest.main()
