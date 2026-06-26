from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "fixed_solid_load_temporal_diagnostics"
MATRIX_JSON = DIAG_ROOT / "fixed_solid_load_temporal_matrix.json"
MATRIX_CSV = DIAG_ROOT / "fixed_solid_load_temporal_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "fixed_solid_load_temporal_summary.md"
HISTORY_JSON = DIAG_ROOT / "fixed_solid_load_temporal_history.json"
HISTORIES_DIR = DIAG_ROOT / "histories"
VERIFICATION_MD = DIAG_ROOT / "verification_fixed_solid_load_temporal_2026-06-26.md"

REQUIRED_SCENARIOS = {
    "fixed_load_0p75_constant_step60",
    "fixed_load_0p80_constant_step60",
    "fixed_load_0p75_ramp2_step60",
    "fixed_load_0p80_ramp2_step60",
    "projection_only_step60_baseline",
    "diagnostic_reinitialize_step60_upper_bound",
}

REQUIRED_HISTORY_FIELDS = {
    "total_force_z_N",
    "primary_face_force_z_N",
    "secondary_face_force_z_N",
    "fluid_reaction_force_z_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_valid_marker_count",
    "secondary_face_valid_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
    "max_abs_traction_pa",
    "two_sided_pressure_marker_count",
    "one_sided_pressure_marker_count",
}


class AnsysVerticalFlapFixedSolidLoadArtifactTests(unittest.TestCase):
    def test_fixed_solid_load_matrix_is_reviewable(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]

        self.assertEqual(payload["preflow_steps"], 60)
        self.assertEqual(payload["last_window_steps"], 20)
        self.assertEqual(
            payload["scope_limit"],
            "fixed-solid load diagnostic only; no coupled 50-step or Fluent parity claim",
        )
        self.assertTrue(REQUIRED_SCENARIOS.issubset({row["scenario"] for row in rows}))
        self.assertIn("best_fixed_solid_load_candidate", payload)
        self.assertIn("fixed_solid_load_candidate_count", payload)
        self.assertIn("candidate_status", payload)
        self.assertEqual(
            payload["best_fixed_solid_load_candidate"],
            "fixed_load_0p80_ramp2_step60",
        )
        self.assertEqual(payload["fixed_solid_load_candidate_count"], 1)
        self.assertEqual(payload["candidate_status"], "candidate_found")

        for row in rows:
            self.assertEqual(row["step_count"], 0)
            self.assertEqual(row["preflow_steps"], 60)
            self.assertFalse(_truthy(row["solid_advanced"]))
            self.assertFalse(_truthy(row["feedback_applied"]))
            self.assertIn("flow_temporal_status", row)
            self.assertIn("hydrodynamic_load_status", row)
            self.assertIn("hydrodynamic_load_fail_reasons", row)
            self.assertIn("force_z_zero_crossing_count", row)
            self.assertIn("force_z_negative_fraction", row)
            self.assertIn("last20_force_z_mean_N", row)
            self.assertIn("last20_marker_action_reaction_residual_max_N", row)
            self.assertIn("last20_scatter_action_reaction_residual_max_N", row)
            self.assertEqual(row["worker_mode"], "isolated_subprocess")
            self.assertEqual(int(row["worker_returncode"]), 0)
            self.assertFalse(_truthy(row["worker_timed_out"]))
            self.assertGreater(float(row["worker_elapsed_s"]), 0.0)
            self.assertTrue(Path(row["worker_stdout_log"]).exists())
            self.assertTrue(Path(row["worker_stderr_log"]).exists())

        diagnostic_row = next(
            row
            for row in rows
            if row["scenario"] == "diagnostic_reinitialize_step60_upper_bound"
        )
        self.assertEqual(
            diagnostic_row["flow_driver_mode"],
            "reinitialize_inlet_each_step_diagnostic",
        )
        self.assertEqual(diagnostic_row["hydrodynamic_load_status"], "load_temporal_strict")
        self.assertEqual(diagnostic_row["flow_temporal_status"], "flow_temporal_failed")
        self.assertNotEqual(
            payload["best_fixed_solid_load_candidate"],
            diagnostic_row["scenario"],
        )

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_fixed_solid_load_histories_include_force_and_residuals(self):
        payload = _read_json(HISTORY_JSON)
        histories = payload["histories"]

        self.assertEqual(payload["preflow_steps"], 60)
        for scenario in REQUIRED_SCENARIOS:
            self.assertIn(scenario, histories)
            self.assertEqual(len(histories[scenario]), 60)
            history_path = HISTORIES_DIR / f"{scenario}_history.csv"
            self.assertTrue(history_path.exists(), msg=str(history_path))
            rows = _read_csv(history_path)
            self.assertEqual(len(rows), 60)
            self.assertTrue(REQUIRED_HISTORY_FIELDS.issubset(rows[0]))
            self.assertEqual([int(row["flow_step_index_global"]) for row in rows], list(range(60)))
            for row in rows:
                self.assertNotEqual(row["total_force_z_N"], "")
                self.assertNotEqual(row["primary_face_force_z_N"], "")
                self.assertNotEqual(row["secondary_face_force_z_N"], "")
                self.assertNotEqual(row["marker_action_reaction_residual_N"], "")
                self.assertNotEqual(row["scatter_action_reaction_residual_N"], "")
                total = float(row["total_force_z_N"])
                primary = float(row["primary_face_force_z_N"])
                secondary = float(row["secondary_face_force_z_N"])
                self.assertAlmostEqual(primary + secondary, total, places=8)

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("Fixed-Solid Load Temporal", summary)
        self.assertIn("candidate_rule = completed, non-diagnostic", summary)
        self.assertIn("face-resolved marker force", verification)
        self.assertIn("diagnostic upper-bound rows are never release candidates", verification)
        self.assertIn("No 50-step run was performed", verification)
        self.assertIn("No Fluent parity claim is made", verification)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


if __name__ == "__main__":
    unittest.main()
