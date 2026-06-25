from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
STEP050_REPORT = ROOT / "easyfsi" / "easyfsi_step050_after_halfdomain_repair.json"
STEP050_PROCESS = ROOT / "easyfsi" / "easyfsi_step050_after_halfdomain_repair_process.json"
STEP050_SUMMARY = ROOT / "compare_after_halfdomain_repair" / "easyfsi_summary.json"
STEP050_HISTORY = ROOT / "compare_after_halfdomain_repair" / "easyfsi_history.csv"
STEP050_STAGE = ROOT / "compare_after_halfdomain_repair" / "stage_check.md"
PREFLOW_REPORT = ROOT / "easyfsi" / "easyfsi_step001_preflow001_after_halfdomain_repair.json"
PREFLOW_STAGE = ROOT / "compare_preflow001_smoke" / "stage_check.md"


class AnsysVerticalFlapPostRepairArtifactTests(unittest.TestCase):
    def test_repaired_step050_coarse_run_completed_and_records_fail_flow(self):
        report = _read_json(STEP050_REPORT)
        process = _read_json(STEP050_PROCESS)
        summary = _read_json(STEP050_SUMMARY)[0]
        stage_check = STEP050_STAGE.read_text(encoding="utf-8")

        self.assertEqual(process["status"], "completed")
        self.assertEqual(process["history_rows"], 50)
        self.assertEqual(len(report["history"]), 50)
        self.assertEqual(summary["steps"], 50)
        self.assertEqual(summary["status"], "FAIL_FLOW")
        self.assertLess(summary["velocity_peak_mps"], 20.0)
        self.assertLess(summary["velocity_p999_mps"], 20.0)
        self.assertIn(
            "diagnosis = check fluid solver / BC / obstacle / outlet / projection",
            stage_check,
        )
        self.assertEqual(summary["stress_invalid"], 0)
        self.assertEqual(summary["scatter_invalid"], 0)
        self.assertEqual(summary["feedback_invalid"], 0)
        self.assertLess(summary["marker_force_z_N"], 0.0)
        self.assertAlmostEqual(summary["root_max_disp_m"], 0.0)

    def test_repaired_step050_history_contains_requested_diagnostics(self):
        with STEP050_HISTORY.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 50)
        last = rows[-1]
        for column in [
            "local_velocity_peak_mps",
            "fluid_speed_p99_mps",
            "fluid_speed_p999_mps",
            "pressure_min_pa",
            "pressure_max_pa",
            "solid_substeps_selected",
            "solid_estimated_cfl",
            "total_marker_force_z_N",
            "mpm_external_force_z_N",
            "tip_mean_dz_m",
        ]:
            self.assertIn(column, last)
            self.assertNotEqual(last[column], "")

    def test_preflow_smoke_records_fixed_solid_preflow_history(self):
        report = _read_json(PREFLOW_REPORT)
        stage_check = PREFLOW_STAGE.read_text(encoding="utf-8")

        self.assertEqual(report["preflow_steps_requested"], 1)
        self.assertEqual(report["preflow_steps_completed"], 1)
        self.assertEqual(len(report["preflow_history"]), 1)
        self.assertTrue(report["preflow_history"][0]["solid_fixed"])
        self.assertFalse(report["preflow_history"][0]["solid_advanced"])
        self.assertEqual(len(report["history"]), 1)
        self.assertIn("steps_requested = 1", stage_check)
        self.assertIn("history_rows = 1", stage_check)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
