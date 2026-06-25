from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "flow_collapse_diagnostics"
PREFLOW_JSON = DIAG_ROOT / "preflow_only_sweep" / "preflow_only_sweep.json"
PREFLOW_CSV = DIAG_ROOT / "preflow_only_sweep" / "preflow_only_sweep.csv"
MATRIX_JSON = (
    DIAG_ROOT / "diagnostic_matrix" / "flow_collapse_diagnostic_matrix.json"
)
MATRIX_CSV = DIAG_ROOT / "diagnostic_matrix" / "flow_collapse_diagnostic_matrix.csv"
MATRIX_SUMMARY = (
    DIAG_ROOT / "diagnostic_matrix" / "flow_collapse_diagnostic_matrix_summary.md"
)


class AnsysVerticalFlapFlowCollapseArtifactTests(unittest.TestCase):
    def test_preflow_only_sweep_records_fixed_solid_projection_history(self):
        payload = _read_json(PREFLOW_JSON)
        rows = payload["rows"]

        self.assertEqual(
            {row["preflow_steps"] for row in rows},
            {1, 2, 5, 10, 20},
        )
        self.assertIn("primary_observation", payload)
        self.assertIn("current_best_hypothesis", payload)
        self.assertIn("next_action", payload)
        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["preflow_status"], "max_steps")
            self.assertFalse(row["solid_advanced"])
            self.assertFalse(row["feedback_applied"])
            self.assertIn("fluid_speed_p999_mps", row)
            self.assertIn("projection_l2", row)
            self.assertIn("projection_max_abs", row)
            self.assertIn("marker_force_z_N", row)

        csv_rows = _read_csv(PREFLOW_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_diagnostic_matrix_answers_feedback_versus_projection_question(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        scenarios = {row["scenario"] for row in rows}

        self.assertEqual(
            scenarios,
            {
                "feedback_on_step10",
                "feedback_off_step10",
                "solver_fv_jacobi_1080_step10",
                "solver_fv_cg_1080_step10",
                "solver_fv_cg_4096_step10",
                "reset_pressure_first_only_step10",
                "reset_pressure_every_step_step10",
                "reinitialize_inlet_each_step_step10",
            },
        )
        self.assertIn("primary_observation", payload)
        self.assertIn("current_best_hypothesis", payload)
        self.assertIn("next_action", payload)
        self.assertTrue(MATRIX_SUMMARY.read_text(encoding="utf-8").startswith("#"))

        feedback_on = _row(rows, "feedback_on_step10")
        feedback_off = _row(rows, "feedback_off_step10")
        self.assertEqual(feedback_on["run_status"], "completed")
        self.assertEqual(feedback_off["run_status"], "completed")
        self.assertIn(
            payload["current_best_hypothesis"],
            {
                "feedback constraints are the primary suspect for flow collapse",
                "projection-only flow path is the primary suspect for flow collapse",
                "coarse 10-step matrix did not reproduce final p999 collapse",
            },
        )

        for row in rows:
            self.assertEqual(row["step_count"], 10)
            self.assertIn("final_velocity_p999_mps", row)
            self.assertIn("max_velocity_p999_mps", row)
            self.assertIn("collapse_ratio_p999", row)
            self.assertIn("flow_status", row)
            self.assertIn("projection_max_abs", row)

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))


def _row(rows: list[dict], scenario: str) -> dict:
    return next(row for row in rows if row["scenario"] == scenario)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
