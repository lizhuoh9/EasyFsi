from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "source_candidate_step20_diagnostics"
MATRIX_JSON = DIAG_ROOT / "source_candidate_step20_matrix.json"
SUMMARY_MD = DIAG_ROOT / "source_candidate_step20_summary.md"
BEST_HISTORY_CSV = DIAG_ROOT / "best_candidate_step20_history.csv"
BEST_FINAL_GATE_HISTORY_CSV = DIAG_ROOT / "best_final_gate_candidate_history.csv"
BEST_FLOW_TEMPORAL_HISTORY_CSV = DIAG_ROOT / "best_flow_temporal_candidate_history.csv"
BEST_COMBINED_TEMPORAL_HISTORY_CSV = (
    DIAG_ROOT / "best_combined_temporal_candidate_history.csv"
)
ALL_CANDIDATE_HISTORIES_CSV = DIAG_ROOT / "all_candidate_step20_histories.csv"
HISTORIES_DIR = DIAG_ROOT / "histories"
VERIFICATION_MD = DIAG_ROOT / "verification_source_candidate_step20_2026-06-25.md"

FINAL_GATE_SOURCE_CANDIDATES = {
    "source_0p75_constant_step20",
    "source_0p80_constant_step20",
    "source_0p75_ramp2_step20",
    "source_0p80_ramp2_step20",
    "source_0p75_ramp5_step20",
}


class AnsysVerticalFlapTemporalGateArtifactTests(unittest.TestCase):
    def test_matrix_records_temporal_candidate_gate(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]

        self.assertEqual(payload["candidate_status"], "no_temporal_candidate")
        self.assertEqual(payload["best_final_gate_candidate"], "source_0p75_ramp5_step20")
        self.assertEqual(payload["best_temporal_candidate"], "none")
        self.assertEqual(payload["best_flow_temporal_candidate"], "source_0p80_ramp2_step20")
        self.assertEqual(payload["best_combined_temporal_candidate"], "none")
        self.assertEqual(payload["promotion_candidate"], "none")
        self.assertEqual(payload["promotion_candidate_status"], "no_promotion_candidate")
        self.assertEqual(payload["diagnostic_fallback_candidate"], "source_0p75_ramp5_step20")
        self.assertEqual(payload["temporal_candidate_status"], "no_temporal_candidate")
        self.assertEqual(payload["temporal_candidate_count"], 0)
        self.assertGreater(payload["flow_temporal_candidate_count"], 0)

        for row in rows:
            self.assertIn("temporal_warmup_steps", row)
            self.assertIn("temporal_evaluation_start_step", row)
            self.assertIn("temporal_candidate_status", row)
            self.assertIn("temporal_fail_reasons", row)
            self.assertIn("temporal_last_window_min_p999_mps", row)
            self.assertIn("temporal_last_window_mean_velocity_outlet_flux_ratio", row)
            self.assertIn("flow_temporal_status", row)
            self.assertIn("flow_temporal_fail_reasons", row)
            self.assertIn("flow_last_window_min_p999_mps", row)
            self.assertIn("flow_last_window_mean_outlet_ratio", row)
            self.assertIn("coupling_settling_status", row)
            self.assertIn("coupling_first_permanently_valid_step", row)
            self.assertIn("coupling_longest_consecutive_pass_steps", row)
            self.assertIn("promotion_candidate_status", row)
            if row["scenario"] == "diagnostic_reinitialize_step20_upper_bound":
                self.assertEqual(row["temporal_candidate_status"], "temporal_not_applicable")
                self.assertEqual(row["flow_temporal_status"], "flow_temporal_not_applicable")
                self.assertEqual(row["coupling_settling_status"], "coupling_not_applicable")
            elif row["run_status"] == "completed":
                self.assertNotEqual(row["temporal_candidate_status"], "temporal_not_applicable")

        flow_best = _row(rows, "source_0p80_ramp2_step20")
        self.assertEqual(flow_best["flow_temporal_status"], "flow_temporal_strict")
        self.assertEqual(flow_best["flow_last_window_failed_step_count"], 0)
        self.assertEqual(flow_best["coupling_settling_status"], "coupling_settled_late")
        self.assertEqual(flow_best["promotion_candidate_status"], "not_promotion_candidate")

        ramp5 = _row(rows, "source_0p75_ramp5_step20")
        self.assertEqual(ramp5["candidate_status"], "candidate")
        self.assertEqual(ramp5["temporal_candidate_status"], "temporal_failed")
        self.assertEqual(ramp5["flow_temporal_status"], "flow_temporal_failed")
        self.assertGreater(int(ramp5["temporal_last_window_failed_step_count"]), 0)
        self.assertIn("p999_below_20", ramp5["temporal_fail_reasons"])

    def test_best_and_candidate_history_csvs_match_matrix(self):
        payload = _read_json(MATRIX_JSON)
        best_candidate = payload["best_candidate"]
        best_rows = _read_csv(BEST_HISTORY_CSV)

        self.assertEqual(len(best_rows), 20)
        self.assertEqual({row["scenario"] for row in best_rows}, {best_candidate})

        final_rows = _read_csv(BEST_FINAL_GATE_HISTORY_CSV)
        self.assertEqual({row["scenario"] for row in final_rows}, {"source_0p75_ramp5_step20"})
        flow_rows = _read_csv(BEST_FLOW_TEMPORAL_HISTORY_CSV)
        self.assertEqual({row["scenario"] for row in flow_rows}, {"source_0p80_ramp2_step20"})
        combined_rows = _read_csv(BEST_COMBINED_TEMPORAL_HISTORY_CSV)
        self.assertEqual(combined_rows, [])

        for scenario in FINAL_GATE_SOURCE_CANDIDATES:
            history_path = HISTORIES_DIR / f"{scenario}_history.csv"
            self.assertTrue(history_path.exists(), msg=str(history_path))
            rows = _read_csv(history_path)
            self.assertEqual(len(rows), 20)
            self.assertEqual({row["scenario"] for row in rows}, {scenario})

        all_rows = _read_csv(ALL_CANDIDATE_HISTORIES_CSV)
        self.assertGreaterEqual(len(all_rows), 20 * len(FINAL_GATE_SOURCE_CANDIDATES))
        self.assertTrue(
            FINAL_GATE_SOURCE_CANDIDATES.issubset(
                {row["scenario"] for row in all_rows}
            )
        )

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("temporal_candidate_status = no_temporal_candidate", summary)
        self.assertIn("best_flow_temporal_candidate = source_0p80_ramp2_step20", summary)
        self.assertIn("promotion_candidate = none", summary)
        self.assertIn("best_candidate_history_csv =", summary)
        self.assertIn("No 50-step run was performed", verification)
        self.assertIn("No Fluent parity claim is made", verification)
        self.assertIn("No promotion-ready combined temporal candidate is claimed", verification)
        self.assertIn("STEP30", verification)


def _row(rows: list[dict], scenario: str) -> dict:
    return next(row for row in rows if row["scenario"] == scenario)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
