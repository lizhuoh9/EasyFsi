from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from tools.validation.print_ansys_vertical_flap_diagnostics import (
    build_stage_check,
    build_summary_row,
)


BASELINE_DIR = Path("validation_runs") / "ansys_vertical_flap_fsi"
REPORT_JSON = BASELINE_DIR / "easyfsi" / "easyfsi_step050.json"
SUMMARY_JSON = BASELINE_DIR / "compare" / "easyfsi_summary.json"
STAGE_CHECK = BASELINE_DIR / "compare" / "stage_check.md"
DISPLACEMENT_COMPARE = BASELINE_DIR / "compare" / "displacement_compare.csv"


class AnsysVerticalFlapClosedLoopFeedbackTests(unittest.TestCase):
    def test_diagnostics_accept_closed_loop_report_contract(self) -> None:
        report = _closed_loop_report()

        summary = build_summary_row(report)
        stage_check = build_stage_check(report, summary, fluent_csv=None)

        self.assertEqual(summary["status"], "PASS_SMOKE")
        self.assertEqual(summary["tip_dz_monotonic_violation_count"], 0)
        self.assertIn("fluid_recomputed_after_feedback = true", stage_check)
        self.assertIn(
            "feedback_closure_status = CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK",
            stage_check,
        )

    def test_current_web_baseline_artifacts_are_available(self) -> None:
        self.assertTrue(REPORT_JSON.is_file())
        self.assertTrue(SUMMARY_JSON.is_file())
        self.assertTrue(STAGE_CHECK.is_file())
        self.assertTrue(DISPLACEMENT_COMPARE.is_file())

    def test_regenerated_web_baseline_reports_closed_loop_feedback(self) -> None:
        stage_check = STAGE_CHECK.read_text(encoding="utf-8")
        report = _read_report()
        history = report.get("history", [])

        self.assertIn("fluid_recomputed_after_feedback = true", stage_check)
        self.assertIn(
            "feedback_closure_status = CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK",
            stage_check,
        )
        self.assertIs(report["fluid_recomputed_after_feedback"], True)
        self.assertEqual(
            report["feedback_closure_status"],
            "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK",
        )
        self.assertEqual(report["fluid_recompute_count"], len(history))
        self.assertEqual(report["fluid_projection_count"], len(history))
        self.assertEqual(report["fluid_projection_after_feedback_count"], len(history) - 1)
        self.assertEqual(report["fluid_recompute_count"], 50)
        self.assertTrue(all(entry["fluid_recomputed"] is True for entry in history))
        self.assertIs(history[0]["fluid_recomputed_after_feedback"], False)
        self.assertIs(history[0]["feedback_available_before_projection"], False)
        for entry in history[1:]:
            self.assertIs(entry["fluid_recomputed_after_feedback"], True)
            self.assertIs(entry["feedback_available_before_projection"], True)
        for entry in history:
            self.assertIn("local_velocity_peak_mps", entry)
            self.assertIn("pressure_min_pa", entry)
            self.assertIn("pressure_max_pa", entry)
            self.assertIn("flow_projection_report", entry)

    @unittest.expectedFailure
    def test_current_web_baseline_requires_no_solid_history_rebound(self) -> None:
        summary = _read_summary()

        self.assertNotEqual(summary["status"], "FAIL_SOLID_HISTORY")
        self.assertEqual(summary["tip_dz_monotonic_violation_count"], 0)

    @unittest.expectedFailure
    def test_current_web_baseline_targets_twenty_percent_displacement_error(self) -> None:
        compare = _read_displacement_compare()

        self.assertLessEqual(float(compare["rel_error"]), 0.20)


def _closed_loop_report() -> dict:
    return {
        "case": "ansys-vertical-flap-fsi",
        "case_metadata": {
            "geometry": {
                "duct_length_m": 0.10,
                "duct_height_m": 0.04,
                "flap_height_m": 0.01,
                "flap_thickness_m": 0.003,
            },
            "solid": {
                "density_kgm3": 1600.0,
                "young_modulus_pa": 1.0e6,
                "poisson_ratio": 0.47,
            },
        },
        "config": {
            "dt_s": 5.0e-4,
            "step_count": 3,
            "grid_nodes": [4, 32, 64],
            "solid_particle_counts": [1, 12, 4],
            "marker_count": 12,
            "mpm_support_radius_m": 0.0015,
        },
        "reference_results": {
            "max_displacement_m": 5.1e-5,
            "local_velocity_peak_mps": 28.1,
            "local_velocity_peak_range_mps": [20.0, 29.0],
            "time_step_s": 5.0e-4,
        },
        "flow_projection_report": {"final_residual": 1.0e-7},
        "computed_pressure_min_pa": -100.0,
        "computed_pressure_max_pa": 200.0,
        "local_velocity_peak_mps": 28.0,
        "local_velocity_peak_relative_error": 0.0035587188612099642,
        "velocity_peak_tolerance": 0.05,
        "stress_valid_marker_count": 12,
        "stress_invalid_marker_count": 0,
        "two_sided_pressure_marker_count": 12,
        "scatter_invalid_marker_count": 0,
        "surface_feedback_updated_marker_count": 12,
        "surface_feedback_invalid_marker_count": 0,
        "surface_feedback_max_marker_displacement_m": 4.0e-5,
        "total_marker_force_n": [0.0, 0.0, -1.2],
        "mpm_external_force_n": [0.0, 0.0, -1.2],
        "scatter_action_reaction_residual_n": 0.0,
        "root_max_displacement_m": 0.0,
        "tip_mean_displacement_m": [0.0, 1.0e-6, -4.8e-5],
        "max_displacement_m": 5.0e-5,
        "reference_max_displacement_m": 5.1e-5,
        "max_displacement_relative_error": 0.0196078431372549,
        "displacement_tolerance": 0.05,
        "fluid_recomputed_after_feedback": True,
        "feedback_closure_status": "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK",
        "fluid_recompute_count": 3,
        "fluid_projection_count": 3,
        "fluid_projection_after_feedback_count": 2,
        "history": [
            _history_entry(1, -2.0e-5, -0.4),
            _history_entry(2, -3.5e-5, -0.8),
            _history_entry(3, -4.8e-5, -1.2),
        ],
    }


def _history_entry(step: int, tip_dz_m: float, force_z_n: float) -> dict:
    return {
        "step": step,
        "stress_valid_marker_count": 12,
        "scatter_invalid_marker_count": 0,
        "feedback_invalid_marker_count": 0,
        "total_marker_force_n": [0.0, 0.0, force_z_n],
        "mpm_external_force_n": [0.0, 0.0, force_z_n],
        "max_displacement_m": abs(tip_dz_m),
        "tip_mean_displacement_m": [0.0, 1.0e-6, tip_dz_m],
        "fluid_recomputed": True,
        "fluid_recomputed_after_feedback": step > 1,
        "feedback_available_before_projection": step > 1,
        "local_velocity_peak_mps": 28.0,
        "pressure_min_pa": -100.0,
        "pressure_max_pa": 200.0,
        "flow_projection_report": {"final_residual": 1.0e-7},
    }


def _read_summary() -> dict:
    rows = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    if not rows:
        raise AssertionError(f"{SUMMARY_JSON} is empty")
    return rows[0]


def _read_report() -> dict:
    text = REPORT_JSON.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    if start < 0:
        raise AssertionError(f"{REPORT_JSON} does not contain a JSON object")
    report, _ = json.JSONDecoder().raw_decode(text[start:])
    return report


def _read_displacement_compare() -> dict[str, str]:
    with DISPLACEMENT_COMPARE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"{DISPLACEMENT_COMPARE} is empty")
    return rows[-1]


if __name__ == "__main__":
    unittest.main()
