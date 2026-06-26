from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "preflow_release_coupling_diagnostics"
MATRIX_JSON = DIAG_ROOT / "preflow_release_step20_matrix.json"
MATRIX_CSV = DIAG_ROOT / "preflow_release_step20_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "preflow_release_step20_summary.md"
HISTORY_JSON = DIAG_ROOT / "preflow_release_step20_history.json"
HISTORIES_DIR = DIAG_ROOT / "histories"
VERIFICATION_MD = DIAG_ROOT / "verification_preflow_release_step20_2026-06-25.md"

REQUIRED_SCENARIOS = {
    "no_preflow_release20_source_0p80_ramp2",
    "preflow10_release20_source_0p80_ramp2",
    "preflow20_release20_source_0p80_ramp2",
    "preflow30_release20_source_0p80_ramp2",
    "preflow20_release20_source_0p75_constant",
    "preflow30_release20_source_0p75_constant",
    "preflow20_release20_source_0p75_ramp2",
    "preflow20_release20_source_0p80_ramp2_feedback_off",
    "preflow20_release20_source_0p80_ramp2_phase_local",
}


class AnsysVerticalFlapPreflowReleaseStep20ArtifactTests(unittest.TestCase):
    def test_preflow_release_matrix_records_required_scenarios_and_scope(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]

        self.assertEqual(payload["release_steps"], 20)
        self.assertEqual(
            payload["scope_limit"],
            "coupled STEP20 diagnostic only; no 50-step or Fluent parity claim",
        )
        self.assertTrue(REQUIRED_SCENARIOS.issubset({row["scenario"] for row in rows}))
        self.assertIn("best_preflow_release_candidate", payload)
        self.assertIn("best_release_flow_candidate", payload)
        self.assertIn("best_release_coupling_candidate", payload)
        self.assertIn("best_release_promotion_candidate", payload)
        self.assertIn("promotion_candidate_count", payload)
        self.assertIn("candidate_status", payload)

        for row in rows:
            self.assertEqual(row["release_steps"], 20)
            self.assertNotIn("step50", row["scenario"].lower())
            self.assertNotIn("step050", row["scenario"].lower())
            self.assertIn("preflow_flow_temporal_status", row)
            self.assertIn("release_flow_temporal_status", row)
            self.assertIn("release_temporal_candidate_status", row)
            self.assertIn("release_coupling_settling_status", row)
            self.assertIn("promotion_candidate_status", row)
            self.assertIn("preflow_release_index_continuity_ok", row)
            self.assertNotIn("preflow_release_state_continuity_ok", row)
            self.assertIn("release_ramp_restarted_after_preflow", row)
            self.assertIn("first_release_pressure_reset", row)
            self.assertIn("first_release_full_field_reinitialized", row)
            self.assertIn("release_final_root_max_displacement_m", row)
            self.assertIn("release_final_marker_action_reaction_residual_N", row)
            self.assertIn("release_final_scatter_action_reaction_residual_N", row)
            self.assertEqual(row["worker_mode"], "isolated_subprocess")
            self.assertEqual(int(row["worker_returncode"]), 0)
            self.assertFalse(_truthy(row["worker_timed_out"]))
            self.assertGreater(float(row["worker_elapsed_s"]), 0.0)
            self.assertTrue(Path(row["worker_stdout_log"]).exists())
            self.assertTrue(Path(row["worker_stderr_log"]).exists())

        preflow20 = _row(rows, "preflow20_release20_source_0p80_ramp2")
        self.assertTrue(_truthy(preflow20["preflow_release_index_continuity_ok"]))
        self.assertTrue(
            _truthy(preflow20["preflow_release_source_factor_continuity_ok"])
        )
        self.assertFalse(_truthy(preflow20["release_ramp_restarted_after_preflow"]))
        self.assertEqual(int(preflow20["last_preflow_global_step"]), 19)
        self.assertEqual(int(preflow20["first_release_global_step"]), 20)
        self.assertEqual(int(preflow20["first_release_source_schedule_step"]), 20)

        phase_local = _row(rows, "preflow20_release20_source_0p80_ramp2_phase_local")
        self.assertEqual(phase_local["flow_source_schedule_scope"], "phase_local")
        self.assertTrue(_truthy(phase_local["release_ramp_restarted_after_preflow"]))
        self.assertEqual(int(phase_local["last_preflow_global_step"]), 19)
        self.assertEqual(int(phase_local["first_release_global_step"]), 20)
        self.assertEqual(int(phase_local["first_release_source_schedule_step"]), 0)
        self.assertEqual(
            phase_local["promotion_candidate_status"],
            "not_promotion_candidate",
        )

        feedback_off = _row(rows, "preflow20_release20_source_0p80_ramp2_feedback_off")
        self.assertFalse(_truthy(feedback_off["apply_marker_feedback_to_fluid"]))
        self.assertEqual(
            feedback_off["promotion_candidate_status"],
            "not_promotion_candidate",
        )

        no_preflow = _row(rows, "no_preflow_release20_source_0p80_ramp2")
        self.assertEqual(no_preflow["last_preflow_global_step"], "")
        self.assertEqual(int(no_preflow["first_release_global_step"]), 0)
        self.assertEqual(int(no_preflow["first_release_source_schedule_step"]), 0)

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_preflow_release_histories_and_docs_are_reviewable(self):
        payload = _read_json(HISTORY_JSON)
        histories = payload["histories"]

        self.assertEqual(payload["release_steps"], 20)
        for scenario in REQUIRED_SCENARIOS:
            self.assertIn(scenario, histories)
            matrix_row = _row(_read_json(MATRIX_JSON)["rows"], scenario)
            expected_steps = int(matrix_row["preflow_steps"]) + 20
            self.assertEqual(len(histories[scenario]), expected_steps)

            history_path = HISTORIES_DIR / f"{scenario}_history.csv"
            self.assertTrue(history_path.exists(), msg=str(history_path))
            rows = _read_csv(history_path)
            self.assertEqual(len(rows), expected_steps)
            self.assertIn("flow_phase", rows[0])
            self.assertIn("global_step", rows[0])
            self.assertIn("source_schedule_step", rows[0])
            self.assertIn("flow_pressure_reset_applied", rows[0])
            self.assertIn("flow_full_field_reinitialized", rows[0])
            self.assertIn("marker_action_reaction_residual_N", rows[0])
            self.assertIn("scatter_action_reaction_residual_N", rows[0])

            preflow_rows = [row for row in rows if row["flow_phase"] == "preflow"]
            release_rows = [row for row in rows if row["flow_phase"] == "release"]
            self.assertEqual(len(preflow_rows), int(matrix_row["preflow_steps"]))
            self.assertEqual(len(release_rows), 20)
            self.assertEqual(
                [int(row["global_step"]) for row in release_rows],
                list(range(int(matrix_row["preflow_steps"]), expected_steps)),
            )

        global_rows = _read_csv(
            HISTORIES_DIR / "preflow20_release20_source_0p80_ramp2_history.csv"
        )
        release_global = [row for row in global_rows if row["flow_phase"] == "release"]
        self.assertEqual(int(release_global[0]["source_schedule_step"]), 20)
        self.assertFalse(
            _truthy(release_global[0]["flow_source_ramp_restarted_after_preflow"])
        )

        local_rows = _read_csv(
            HISTORIES_DIR
            / "preflow20_release20_source_0p80_ramp2_phase_local_history.csv"
        )
        release_local = [row for row in local_rows if row["flow_phase"] == "release"]
        self.assertEqual(int(release_local[0]["source_schedule_step"]), 0)
        self.assertTrue(
            _truthy(release_local[0]["flow_source_ramp_restarted_after_preflow"])
        )

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("Preflow-Release STEP20", summary)
        self.assertIn("No 50-step run was performed", verification)
        self.assertIn("No L2/L3 matrix was run", verification)
        self.assertIn("No Fluent parity claim is made", verification)
        self.assertIn(
            "No solid material, damping, support-radius, or gate threshold was tuned",
            verification,
        )


def _row(rows: list[dict], scenario: str) -> dict:
    return next(row for row in rows if row["scenario"] == scenario)


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
