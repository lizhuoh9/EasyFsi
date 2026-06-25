from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.validation.print_ansys_vertical_flap_diagnostics import (
    build_history_rows,
    build_summary_row,
    load_report,
    main,
    write_diagnostics,
)


class AnsysVerticalFlapDiagnosticsTests(unittest.TestCase):
    def test_summary_status_records_magnitude_failure_after_lower_layers_pass(self) -> None:
        report = _fixture_report()

        summary = build_summary_row(report)

        self.assertEqual(summary["case"], "ansys-vertical-flap-fsi")
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["grid"], "4x32x64")
        self.assertEqual(summary["particles"], "1x12x4")
        self.assertEqual(summary["markers"], 12)
        self.assertLess(summary["marker_force_z_N"], 0.0)
        self.assertEqual(summary["status"], "FAIL_MAGNITUDE")

    def test_history_rows_extract_vectors_and_derive_time(self) -> None:
        report = _fixture_report()

        rows = build_history_rows(report)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["time_s"], 5.0e-4)
        self.assertAlmostEqual(rows[1]["total_marker_force_z_N"], -1.2)
        self.assertAlmostEqual(rows[1]["mpm_external_force_z_N"], -1.2)
        self.assertAlmostEqual(rows[1]["tip_mean_dz_m"], -4.0e-5)
        self.assertAlmostEqual(rows[1]["tip_norm_m"], 4.001249804748512e-5)
        self.assertEqual(rows[0]["root_max_displacement_m"], "")
        self.assertAlmostEqual(rows[1]["local_velocity_peak_mps"], 28.0)
        self.assertAlmostEqual(rows[1]["pressure_min_pa"], -12.0)
        self.assertAlmostEqual(rows[1]["pressure_max_pa"], 42.0)
        self.assertAlmostEqual(rows[1]["projection_l2"], 2.0e-7)
        self.assertAlmostEqual(rows[1]["projection_max_abs"], 3.0e-7)
        self.assertAlmostEqual(rows[1]["pre_projection_l2"], 4.0e-7)
        self.assertAlmostEqual(rows[1]["post_boundary_l2"], 5.0e-7)
        self.assertAlmostEqual(
            rows[1]["velocity_dirichlet_boundary_max_delta_mps"],
            6.0e-7,
        )

    def test_write_diagnostics_creates_summary_history_and_stage_check(self) -> None:
        report = _fixture_report()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            outputs = write_diagnostics([report], output_dir)

            self.assertEqual(
                set(outputs),
                {"summary_csv", "summary_json", "history_csv", "stage_check"},
            )
            summary_rows = _read_csv(output_dir / "easyfsi_summary.csv")
            history_rows = _read_csv(output_dir / "easyfsi_history.csv")
            stage_check = (output_dir / "stage_check.md").read_text(encoding="utf-8")
            summary_json = json.loads(
                (output_dir / "easyfsi_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary_rows[0]["status"], "FAIL_MAGNITUDE")
            self.assertEqual(history_rows[1]["step"], "2")
            self.assertEqual(history_rows[1]["local_velocity_peak_mps"], "28.0")
            self.assertEqual(history_rows[1]["projection_l2"], "2e-07")
            self.assertEqual(history_rows[1]["projection_max_abs"], "3e-07")
            self.assertIn("[SETUP]", stage_check)
            self.assertIn("[FLOW_ONLY]", stage_check)
            self.assertIn("[INTERFACE_FORCE]", stage_check)
            self.assertIn("[SOLID_RESPONSE]", stage_check)
            self.assertIn("[FSI_FEEDBACK]", stage_check)
            self.assertIn("[COORDINATE_MAPPING]", stage_check)
            self.assertNotIn("projection_final_residual =", stage_check)
            self.assertIn("projection_l2 = 1e-07", stage_check)
            self.assertIn("projection_max_abs = 2e-07", stage_check)
            self.assertIn("Fluent x <-> EasyFsi z", stage_check)
            self.assertIn("fluent_comparison = not run", stage_check)
            self.assertEqual(summary_json[0]["status"], "FAIL_MAGNITUDE")

    def test_summary_records_tip_history_monotonic_violation(self) -> None:
        report = _history_violation_report()

        summary = build_summary_row(report)

        self.assertAlmostEqual(summary["tip_dz_final_m"], -3.0e-5)
        self.assertAlmostEqual(summary["tip_dz_min_m"], -4.0e-5)
        self.assertAlmostEqual(summary["tip_dz_max_m"], -2.0e-5)
        self.assertEqual(summary["tip_dz_monotonic_violation_count"], 1)
        self.assertEqual(summary["first_tip_dz_violation_step"], 3)
        self.assertAlmostEqual(summary["max_tip_dz_rebound_m"], 1.0e-5)
        self.assertEqual(summary["tip_dz_sign_violation_count"], 0)
        self.assertEqual(summary["status"], "FAIL_SOLID_HISTORY")

    def test_status_returns_fail_solid_history_before_fail_magnitude(self) -> None:
        report = _history_violation_report()
        report["max_displacement_relative_error"] = 4.0
        report["displacement_tolerance"] = 0.05

        summary = build_summary_row(report)

        self.assertEqual(summary["status"], "FAIL_SOLID_HISTORY")

    def test_stage_check_reports_open_loop_load_reuse(self) -> None:
        report = _history_violation_report()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            write_diagnostics([report], output_dir)

            stage_check = (output_dir / "stage_check.md").read_text(encoding="utf-8")
            self.assertIn("fluid_recomputed_after_feedback = false", stage_check)
            self.assertIn(
                "feedback_closure_status = OPEN_LOOP_OR_PREFEEDBACK_ONLY",
                stage_check,
            )
            self.assertIn("tip_dz_monotonic_violation_count = 1", stage_check)
            self.assertIn("diagnosis = check solid history monotonicity", stage_check)

    def test_history_health_metrics_are_blank_or_zero_for_missing_history(self) -> None:
        report = _fixture_report()
        report.pop("history")

        summary = build_summary_row(report)

        self.assertEqual(summary["tip_dz_final_m"], "")
        self.assertEqual(summary["tip_dz_min_m"], "")
        self.assertEqual(summary["tip_dz_max_m"], "")
        self.assertEqual(summary["tip_dz_monotonic_violation_count"], 0)
        self.assertEqual(summary["first_tip_dz_violation_step"], "")
        self.assertEqual(summary["max_tip_dz_rebound_m"], "")
        self.assertEqual(summary["tip_dz_sign_violation_count"], 0)

    def test_optional_fluent_tip_csv_writes_displacement_compare(self) -> None:
        report = _fixture_report()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fluent_csv = output_dir / "fluent_tip_displacement.csv"
            _write_csv(
                fluent_csv,
                [
                    "step",
                    "time_s",
                    "tip_total_displacement_m",
                    "tip_x_displacement_m",
                    "tip_y_displacement_m",
                ],
                [
                    {
                        "step": 2,
                        "time_s": 1.0e-3,
                        "tip_total_displacement_m": 5.1e-5,
                        "tip_x_displacement_m": 4.0e-5,
                        "tip_y_displacement_m": 2.0e-6,
                    }
                ],
            )

            outputs = write_diagnostics([report], output_dir, fluent_csv=fluent_csv)

            self.assertIn("displacement_compare_csv", outputs)
            compare_rows = _read_csv(output_dir / "displacement_compare.csv")
            self.assertEqual(compare_rows[0]["step"], "2")
            self.assertEqual(compare_rows[0]["fluent_tip_total_m"], "5.1e-05")
            self.assertEqual(compare_rows[0]["easyfsi_tip_streamwise_m"], "-4e-05")
            self.assertEqual(compare_rows[0]["easyfsi_tip_vertical_m"], "1e-06")
            self.assertEqual(compare_rows[0]["local_velocity_peak_mps"], "28.0")
            self.assertEqual(compare_rows[0]["projection_l2"], "2e-07")
            self.assertGreater(float(compare_rows[0]["rel_error"]), 0.0)

    def test_load_report_accepts_prefix_text_before_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(
                "[Taichi] version banner\n" + json.dumps(_fixture_report()),
                encoding="utf-8",
            )

            report = load_report(path)

            self.assertEqual(report["case"], "ansys-vertical-flap-fsi")

    def test_cli_returns_nonzero_for_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = Path(tmp) / "bad.json"
            bad_json.write_text("not json", encoding="utf-8")

            exit_code = main(
                [
                    "--easyfsi-json",
                    str(bad_json),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                ]
            )

            self.assertEqual(exit_code, 1)


def _fixture_report() -> dict:
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
            "step_count": 2,
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
        "flow_projection_report": {
            "projection_l2": 1.0e-7,
            "projection_max_abs": 2.0e-7,
            "pre_projection_l2": 3.0e-7,
            "post_boundary_l2": 4.0e-7,
            "velocity_dirichlet_boundary_max_delta_mps": 5.0e-7,
        },
        "computed_pressure_min_pa": -12.0,
        "computed_pressure_max_pa": 42.0,
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
        "tip_mean_displacement_m": [0.0, 1.0e-6, -4.0e-5],
        "max_displacement_m": 6.0e-5,
        "reference_max_displacement_m": 5.1e-5,
        "max_displacement_relative_error": 0.1764705882352941,
        "displacement_tolerance": 0.05,
        "history": [
            {
                "step": 1,
                "stress_valid_marker_count": 12,
                "scatter_invalid_marker_count": 0,
                "feedback_invalid_marker_count": 0,
                "total_marker_force_n": [0.0, 0.0, -0.6],
                "mpm_external_force_n": [0.0, 0.0, -0.6],
                "max_displacement_m": 2.0e-5,
                "tip_mean_displacement_m": [0.0, 5.0e-7, -2.0e-5],
                "local_velocity_peak_mps": 27.0,
                "pressure_min_pa": -10.0,
                "pressure_max_pa": 40.0,
                "flow_projection_report": {
                    "projection_l2": 1.0e-7,
                    "projection_max_abs": 2.0e-7,
                    "pre_projection_l2": 3.0e-7,
                    "post_boundary_l2": 4.0e-7,
                    "velocity_dirichlet_boundary_max_delta_mps": 5.0e-7,
                },
            },
            {
                "step": 2,
                "stress_valid_marker_count": 12,
                "scatter_invalid_marker_count": 0,
                "feedback_invalid_marker_count": 0,
                "total_marker_force_n": [0.0, 0.0, -1.2],
                "mpm_external_force_n": [0.0, 0.0, -1.2],
                "max_displacement_m": 6.0e-5,
                "tip_mean_displacement_m": [0.0, 1.0e-6, -4.0e-5],
                "local_velocity_peak_mps": 28.0,
                "pressure_min_pa": -12.0,
                "pressure_max_pa": 42.0,
                "flow_projection_report": {
                    "projection_l2": 2.0e-7,
                    "projection_max_abs": 3.0e-7,
                    "pre_projection_l2": 4.0e-7,
                    "post_boundary_l2": 5.0e-7,
                    "velocity_dirichlet_boundary_max_delta_mps": 6.0e-7,
                },
            },
        ],
    }


def _history_violation_report() -> dict:
    report = copy.deepcopy(_fixture_report())
    report["config"]["step_count"] = 3
    report["tip_mean_displacement_m"] = [0.0, 1.0e-6, -3.0e-5]
    report["history"] = [
        {
            "step": 1,
            "stress_valid_marker_count": 12,
            "scatter_invalid_marker_count": 0,
            "feedback_invalid_marker_count": 0,
            "total_marker_force_n": [0.0, 0.0, -0.6],
            "mpm_external_force_n": [0.0, 0.0, -0.6],
            "max_displacement_m": 2.0e-5,
            "tip_mean_displacement_m": [0.0, 5.0e-7, -2.0e-5],
        },
        {
            "step": 2,
            "stress_valid_marker_count": 12,
            "scatter_invalid_marker_count": 0,
            "feedback_invalid_marker_count": 0,
            "total_marker_force_n": [0.0, 0.0, -1.2],
            "mpm_external_force_n": [0.0, 0.0, -1.2],
            "max_displacement_m": 6.0e-5,
            "tip_mean_displacement_m": [0.0, 1.0e-6, -4.0e-5],
        },
        {
            "step": 3,
            "stress_valid_marker_count": 12,
            "scatter_invalid_marker_count": 0,
            "feedback_invalid_marker_count": 0,
            "total_marker_force_n": [0.0, 0.0, -1.0],
            "mpm_external_force_n": [0.0, 0.0, -1.0],
            "max_displacement_m": 6.0e-5,
            "tip_mean_displacement_m": [0.0, 1.0e-6, -3.0e-5],
        },
    ]
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
