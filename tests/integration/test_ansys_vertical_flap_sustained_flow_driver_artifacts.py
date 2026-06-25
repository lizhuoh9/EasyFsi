from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "sustained_flow_driver_diagnostics"
MATRIX_JSON = DIAG_ROOT / "sustained_flow_driver_matrix.json"
MATRIX_CSV = DIAG_ROOT / "sustained_flow_driver_matrix.csv"
MATRIX_SUMMARY = DIAG_ROOT / "sustained_flow_driver_matrix_summary.md"


class AnsysVerticalFlapSustainedFlowDriverArtifactTests(unittest.TestCase):
    def test_matrix_records_explicit_driver_modes_and_source_fields(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        scenarios = {row["scenario"] for row in rows}

        self.assertEqual(
            scenarios,
            {
                "projection_only_step10",
                "reinitialize_inlet_each_step_step10",
                "sustained_boundary_inlet_step10",
                "sustained_volume_source_inlet_step10",
                "sustained_inlet_predictor_step10",
                "sustained_inlet_predictor_feedback_off_step10",
                "reset_pressure_every_step_step10",
            },
        )
        self.assertIn("primary_observation", payload)
        self.assertIn("current_best_hypothesis", payload)
        self.assertIn("next_action", payload)
        self.assertIn("scope_limits", payload)
        self.assertTrue(MATRIX_SUMMARY.read_text(encoding="utf-8").startswith("#"))

        diagnostic = _row(rows, "reinitialize_inlet_each_step_step10")
        self.assertEqual(
            diagnostic["flow_driver_mode"],
            "reinitialize_inlet_each_step_diagnostic",
        )
        self.assertTrue(diagnostic["flow_driver_diagnostic_only"])
        self.assertTrue(diagnostic["flow_driver_uses_full_velocity_reset"])
        self.assertTrue(diagnostic["flow_reinitialize_inlet_each_step"])

        projection = _row(rows, "projection_only_step10")
        self.assertEqual(projection["flow_driver_mode"], "projection_only")
        self.assertFalse(projection["flow_driver_uses_full_velocity_reset"])

        sustained = [
            row
            for row in rows
            if str(row["flow_driver_mode"]).startswith("sustained_")
        ]
        self.assertGreaterEqual(len(sustained), 3)
        for row in sustained:
            self.assertEqual(row["run_status"], "completed")
            self.assertFalse(row["flow_driver_diagnostic_only"])
            self.assertFalse(row["flow_driver_uses_full_velocity_reset"])
            self.assertIn("final_velocity_p999_mps", row)
            self.assertIn("source_volume_flux_m3s", row)
            self.assertIn("zmin_pressure_outlet_flux_m3s", row)
            self.assertIn("pressure_outlet_flux_ratio", row)
            self.assertIn("stress_invalid_marker_count", row)
            self.assertIn("scatter_invalid_marker_count", row)
            self.assertIn("feedback_invalid_marker_count", row)

        source = _row(rows, "sustained_volume_source_inlet_step10")
        predictor = _row(rows, "sustained_inlet_predictor_step10")
        self.assertTrue(source["flow_volume_source_applied"])
        self.assertTrue(predictor["flow_volume_source_applied"])
        self.assertFalse(source["flow_reinitialize_inlet_each_step"])
        self.assertFalse(predictor["flow_reinitialize_inlet_each_step"])

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
