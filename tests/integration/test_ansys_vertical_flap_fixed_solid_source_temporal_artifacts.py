from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "fixed_solid_source_temporal_diagnostics"
MATRIX_JSON = DIAG_ROOT / "fixed_solid_source_temporal_matrix.json"
MATRIX_CSV = DIAG_ROOT / "fixed_solid_source_temporal_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "fixed_solid_source_temporal_summary.md"
HISTORY_JSON = DIAG_ROOT / "fixed_solid_source_temporal_history.json"
HISTORIES_DIR = DIAG_ROOT / "histories"
VERIFICATION_MD = DIAG_ROOT / "verification_fixed_solid_source_temporal_2026-06-25.md"

REQUIRED_SCENARIOS = {
    "fixed_source_0p75_constant_step30",
    "fixed_source_0p80_constant_step30",
    "fixed_source_0p75_ramp2_step30",
    "fixed_source_0p80_ramp2_step30",
    "fixed_source_0p75_ramp5_step30",
    "projection_only_step30_baseline",
    "diagnostic_reinitialize_step30_upper_bound",
}


class AnsysVerticalFlapFixedSolidSourceTemporalArtifactTests(unittest.TestCase):
    def test_fixed_solid_source_temporal_matrix_is_reviewable(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]

        self.assertEqual(payload["step_count"], 0)
        self.assertEqual(payload["preflow_steps"], 30)
        self.assertEqual(payload["scope_limit"], "fixed-solid preflow-only diagnostic; not coupled FSI validation")
        self.assertTrue(REQUIRED_SCENARIOS.issubset({row["scenario"] for row in rows}))
        self.assertIn("best_fixed_solid_flow_candidate", payload)
        self.assertIn("fixed_solid_flow_candidate_count", payload)

        for row in rows:
            self.assertEqual(row["step_count"], 0)
            self.assertEqual(row["preflow_steps"], 30)
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertIn("flow_temporal_status", row)
            self.assertIn("flow_temporal_fail_reasons", row)
            self.assertIn("flow_last_window_min_p999_mps", row)
            self.assertIn("flow_last_window_mean_outlet_ratio", row)
            self.assertEqual(row["worker_mode"], "isolated_subprocess")
            self.assertEqual(int(row["worker_returncode"]), 0)
            self.assertFalse(bool(row["worker_timed_out"]))
            self.assertGreater(float(row["worker_elapsed_s"]), 0.0)
            self.assertTrue(Path(row["worker_stdout_log"]).exists())
            self.assertTrue(Path(row["worker_stderr_log"]).exists())

        diagnostic = _row(rows, "diagnostic_reinitialize_step30_upper_bound")
        self.assertTrue(bool(diagnostic["flow_driver_uses_full_velocity_reset"]))
        self.assertEqual(
            diagnostic["flow_temporal_status"],
            "flow_temporal_not_applicable",
        )

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_fixed_solid_histories_and_docs_are_scope_limited(self):
        payload = _read_json(HISTORY_JSON)
        histories = payload["histories"]

        self.assertEqual(payload["preflow_steps"], 30)
        for scenario in REQUIRED_SCENARIOS:
            self.assertIn(scenario, histories)
            self.assertEqual(len(histories[scenario]), 30)
            history_path = HISTORIES_DIR / f"{scenario}_history.csv"
            self.assertTrue(history_path.exists(), msg=str(history_path))
            history_rows = _read_csv(history_path)
            self.assertEqual(len(history_rows), 30)
            self.assertIn("flow_step_index_global", history_rows[0])
            self.assertIn("flow_source_schedule_step_index", history_rows[0])
            self.assertIn("flow_source_schedule_scope", history_rows[0])
            self.assertEqual(
                [int(row["flow_step_index_local"]) for row in history_rows],
                list(range(30)),
            )
            self.assertEqual(
                [int(row["flow_step_index_global"]) for row in history_rows],
                list(range(30)),
            )
            self.assertEqual(
                [int(row["flow_source_schedule_step_index"]) for row in history_rows],
                list(range(30)),
            )

        ramp5_rows = _read_csv(
            HISTORIES_DIR / "fixed_source_0p75_ramp5_step30_history.csv"
        )
        self.assertEqual(
            [round(float(row["source_factor"]), 2) for row in ramp5_rows[:5]],
            [0.15, 0.30, 0.45, 0.60, 0.75],
        )

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("fixed-solid source temporal step30", summary.lower())
        self.assertIn("No coupled FSI step was run", verification)
        self.assertIn("No Fluent parity claim is made", verification)
        self.assertIn("not coupled FSI validation", verification)
        self.assertIn("does not skip to 0, 2, 4 during preflow", verification)


def _row(rows: list[dict], scenario: str) -> dict:
    return next(row for row in rows if row["scenario"] == scenario)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
