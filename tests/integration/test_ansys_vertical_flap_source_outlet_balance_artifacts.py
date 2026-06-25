from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "source_outlet_balance_diagnostics"
SOURCE_JSON = DIAG_ROOT / "source_strength_sweep.json"
SOURCE_CSV = DIAG_ROOT / "source_strength_sweep.csv"
OUTLET_JSON = DIAG_ROOT / "outlet_balance_sweep.json"
OUTLET_CSV = DIAG_ROOT / "outlet_balance_sweep.csv"
SUMMARY_MD = DIAG_ROOT / "source_outlet_balance_summary.md"
VERIFICATION_MD = DIAG_ROOT / "verification_source_outlet_balance_2026-06-25.md"


class AnsysVerticalFlapSourceOutletBalanceArtifactTests(unittest.TestCase):
    def test_source_strength_sweep_records_required_balance_fields(self):
        payload = _read_json(SOURCE_JSON)
        rows = payload["rows"]
        strengths = {round(float(row["source_strength"]), 2) for row in rows}

        self.assertEqual(strengths, {0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00})
        self.assertTrue(any(float(row["source_strength"]) < 1.0 for row in rows))
        self.assertIn(payload["candidate_status"], {"candidate_found", "no_candidate"})
        self.assertIn("best_candidate", payload)
        self.assertIn("primary_observation", payload)
        self.assertIn("current_best_hypothesis", payload)
        self.assertIn("next_action", payload)

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["flow_driver_mode"], "sustained_volume_source_inlet")
            self.assertFalse(row["flow_driver_uses_full_velocity_reset"])
            self.assertIn("source_volume_flux_m3s", row)
            self.assertIn("positive_source_volume_flux_m3s", row)
            self.assertIn("abs_source_volume_flux_m3s", row)
            self.assertIn("zmin_pressure_outlet_flux_m3s", row)
            self.assertIn("zmin_velocity_outlet_flux_m3s", row)
            self.assertIn("pressure_outlet_flux_ratio", row)
            self.assertIn("velocity_outlet_flux_ratio", row)
            self.assertIn("final_velocity_peak_mps", row)
            self.assertIn("final_velocity_p999_mps", row)
            self.assertIn("projection_l2", row)
            self.assertIn("projection_max_abs", row)
            self.assertIn("candidate_status", row)

        csv_rows = _read_csv(SOURCE_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_outlet_balance_sweep_excludes_diagnostic_reset_from_candidate(self):
        payload = _read_json(OUTLET_JSON)
        rows = payload["rows"]
        scenarios = {row["scenario"] for row in rows}

        self.assertTrue(
            {
                "projection_only_baseline_step10",
                "diagnostic_reinitialize_upper_bound_step10",
                "selected_source_strength_step10",
                "selected_source_strength_reset_pressure_step10",
                "selected_source_strength_ramp5_step10",
            }.issubset(scenarios)
        )

        diagnostic = _row(rows, "diagnostic_reinitialize_upper_bound_step10")
        self.assertTrue(diagnostic["flow_driver_uses_full_velocity_reset"])
        self.assertEqual(diagnostic["candidate_status"], "diagnostic_excluded")

        for row in rows:
            self.assertIn("pressure_outlet_flux_ratio", row)
            self.assertIn("velocity_outlet_flux_ratio", row)

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("best_candidate =", summary)
        self.assertIn("candidate_status =", summary)
        self.assertIn("No 50-step run was performed", verification)

        csv_rows = _read_csv(OUTLET_CSV)
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
