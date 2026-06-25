from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "source_candidate_step20_diagnostics"
MATRIX_JSON = DIAG_ROOT / "source_candidate_step20_matrix.json"
MATRIX_CSV = DIAG_ROOT / "source_candidate_step20_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "source_candidate_step20_summary.md"
HISTORY_JSON = DIAG_ROOT / "source_candidate_step20_history.json"
CANDIDATE_HISTORY_CSV = DIAG_ROOT / "source_strength_0p75_step20_history.csv"
VERIFICATION_MD = DIAG_ROOT / "verification_source_candidate_step20_2026-06-25.md"

REQUIRED_SCENARIOS = {
    "projection_only_step20_baseline",
    "diagnostic_reinitialize_step20_upper_bound",
    "source_0p70_constant_step20",
    "source_0p75_constant_step20",
    "source_0p80_constant_step20",
    "source_0p75_reset_pressure_step20",
    "source_0p75_ramp2_step20",
    "source_0p80_ramp2_step20",
}

HISTORY_FIELDS = {
    "scenario",
    "step",
    "source_factor",
    "source_normal_velocity_mps",
    "velocity_peak_mps",
    "velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "projection_l2",
    "projection_max_abs",
    "marker_force_z_N",
    "tip_dz_m",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
}


class AnsysVerticalFlapSourceCandidateStep20ArtifactTests(unittest.TestCase):
    def test_step20_matrix_records_required_scenarios_and_gates(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        scenarios = {row["scenario"] for row in rows}

        self.assertEqual(payload["step_count"], 20)
        self.assertTrue(REQUIRED_SCENARIOS.issubset(scenarios))
        self.assertIn(payload["candidate_status"], {"candidate_found", "no_candidate"})
        self.assertEqual(
            payload["mass_balance_primary_metric"],
            "velocity_outlet_flux_ratio",
        )
        self.assertEqual(
            payload["pressure_outlet_flux_interpretation"],
            "diagnostic_only_until_pressure_outlet_model_reviewed",
        )
        self.assertIn("primary_observation", payload)
        self.assertIn("current_best_hypothesis", payload)
        self.assertIn("next_action", payload)

        for row in rows:
            self.assertEqual(row["step_count"], 20)
            self.assertNotIn("step50", row["scenario"].lower())
            self.assertNotIn("step050", row["scenario"].lower())
            self.assertIn("pressure_outlet_flux_ratio", row)
            self.assertIn("velocity_outlet_flux_ratio", row)
            self.assertIn("max_velocity_peak_mps", row)
            self.assertIn("candidate_status", row)
            if row["candidate_status"] == "candidate":
                _assert_step20_candidate_gate(self, row)

        diagnostic = _row(rows, "diagnostic_reinitialize_step20_upper_bound")
        self.assertTrue(diagnostic["flow_driver_uses_full_velocity_reset"])
        self.assertEqual(diagnostic["candidate_status"], "diagnostic_excluded")

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_step20_history_and_verification_are_reviewable(self):
        history_payload = _read_json(HISTORY_JSON)
        histories = history_payload["histories"]
        candidate_history = histories["source_0p75_constant_step20"]

        self.assertEqual(history_payload["step_count"], 20)
        self.assertEqual(len(candidate_history), 20)
        self.assertTrue(HISTORY_FIELDS.issubset(candidate_history[0]))

        csv_rows = _read_csv(CANDIDATE_HISTORY_CSV)
        self.assertEqual(len(csv_rows), 20)
        self.assertTrue(HISTORY_FIELDS.issubset(csv_rows[0]))

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("mass_balance_primary_metric = velocity_outlet_flux_ratio", summary)
        self.assertIn(
            "pressure_outlet_flux_interpretation = "
            "diagnostic_only_until_pressure_outlet_model_reviewed",
            summary,
        )
        self.assertIn("No 50-step run was performed", verification)
        self.assertIn("No Fluent parity claim is made", verification)
        self.assertIn("No solid parameters were tuned", verification)


def _assert_step20_candidate_gate(
    test_case: unittest.TestCase,
    row: dict,
) -> None:
    test_case.assertFalse(row["flow_driver_uses_full_velocity_reset"])
    test_case.assertGreaterEqual(float(row["final_velocity_p999_mps"]), 20.0)
    test_case.assertLessEqual(float(row["final_velocity_p999_mps"]), 29.0)
    test_case.assertLessEqual(float(row["final_velocity_peak_mps"]), 35.0)
    test_case.assertLessEqual(float(row["max_velocity_peak_mps"]), 40.0)
    test_case.assertGreaterEqual(float(row["velocity_outlet_flux_ratio"]), 0.80)
    test_case.assertLessEqual(float(row["velocity_outlet_flux_ratio"]), 1.20)
    test_case.assertEqual(int(row["stress_invalid_marker_count"]), 0)
    test_case.assertEqual(int(row["scatter_invalid_marker_count"]), 0)
    test_case.assertEqual(int(row["feedback_invalid_marker_count"]), 0)
    test_case.assertLess(float(row["marker_force_z_N"]), 0.0)
    test_case.assertLess(float(row["tip_dz_final_m"]), 0.0)


def _row(rows: list[dict], scenario: str) -> dict:
    return next(row for row in rows if row["scenario"] == scenario)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
