from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from validation_runs.ansys_vertical_flap_fsi.scripts import (
    run_traction_formulation_validation_matrix as traction_matrix,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "traction_formulation_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_formulation_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_formulation_matrix.csv"
SUMMARY_MD = DIAG_ROOT / "traction_formulation_summary.md"
HISTORY_JSON = DIAG_ROOT / "traction_formulation_history.json"
HISTORIES_DIR = DIAG_ROOT / "histories"
VERIFICATION_MD = DIAG_ROOT / "verification_traction_formulation_2026-06-26.md"

REQUIRED_SCENARIOS = {
    "dual_two_sided_offset0p51_pressure_only",
    "dual_one_sided_offset0p51_pressure_only",
    "single_mid_two_sided_offset0p00_pressure_only",
    "dual_two_sided_offset0p25_pressure_only",
    "dual_two_sided_offset1p00_pressure_only",
    "dual_two_sided_offset0p51_viscous_air",
}

REQUIRED_ROW_FIELDS = {
    "marker_layout",
    "pressure_sampling_mode",
    "include_viscous_traction",
    "viscosity_pa_s",
    "marker_face_offset_cells",
    "primary_face_force_z_N",
    "secondary_face_force_z_N",
    "total_force_z_N",
    "primary_plus_secondary_force_z_N",
    "force_decomposition_residual_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_mean_pressure_pa",
    "secondary_face_mean_pressure_pa",
    "primary_face_mean_traction_z_pa",
    "secondary_face_mean_traction_z_pa",
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "pressure_min_pa",
    "pressure_max_pa",
    "flow_snapshot_signature",
    "force_difference_from_baseline_N",
    "force_ratio_to_baseline",
    "force_difference_from_reference_N",
    "force_ratio_to_reference",
    "face_force_ratio",
    "scope_limit",
}

REQUIRED_HISTORY_FIELDS = {
    "total_force_z_N",
    "primary_face_force_z_N",
    "secondary_face_force_z_N",
    "primary_plus_secondary_force_z_N",
    "force_decomposition_residual_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_mean_traction_z_pa",
    "secondary_face_mean_traction_z_pa",
    "max_abs_traction_pa",
    "two_sided_pressure_marker_count",
    "one_sided_pressure_marker_count",
}


class AnsysVerticalFlapTractionFormulationArtifactTests(unittest.TestCase):
    def test_traction_formulation_matrix_is_reviewable(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_name = {row["scenario"]: row for row in rows}

        self.assertEqual(payload["preflow_steps"], 20)
        self.assertEqual(
            payload["scope_limit"],
            "fixed-solid traction formulation diagnostic only; no coupled 50-step or Fluent parity claim",
        )
        self.assertTrue(REQUIRED_SCENARIOS.issubset(by_name))
        self.assertEqual(
            payload["baseline_scenario"],
            "dual_two_sided_offset0p51_pressure_only",
        )
        self.assertEqual(payload["reference_formulation_candidate"], "none")
        self.assertEqual(
            payload["candidate_status"],
            "no_reference_formulation_candidate",
        )
        self.assertIn(
            "dual_face_one_sided_unsupported",
            payload["candidate_blockers"],
        )
        self.assertIn(
            "dual_two_sided_offset_sensitivity_above_tolerance",
            payload["candidate_blockers"],
        )
        self.assertIn(
            "pressure_probe_diagnostics_incomplete",
            payload["candidate_blockers"],
        )
        self.assertEqual(
            payload["offset_sensitivity_status"],
            "offset_sensitivity_above_tolerance",
        )
        self.assertLess(float(payload["offset_force_ratio_min"]), 0.2)
        self.assertGreater(float(payload["offset_force_ratio_max"]), 1.5)
        self.assertEqual(
            payload["formulation_agreement_status"],
            "blocked_dual_one_sided_unsupported",
        )
        self.assertEqual(
            payload["flow_snapshot_identity_status"],
            "flow_metrics_match_completed_rows",
        )
        self.assertGreaterEqual(payload["supported_formulation_count"], 1)
        self.assertEqual(payload["unsupported_formulation_count"], 1)
        self.assertIn("pressure_mean_status", payload)

        unsupported = by_name["dual_one_sided_offset0p51_pressure_only"]
        self.assertEqual(unsupported["run_status"], "unsupported")
        self.assertEqual(unsupported["worker_mode"], "not_run")
        self.assertIn("one_sided_pressure_region_id", unsupported["status_reason"])

        reference = by_name["dual_two_sided_offset0p51_pressure_only"]
        self.assertEqual(reference["run_status"], "completed")
        self.assertEqual(reference["force_difference_from_baseline_N"], 0.0)
        self.assertEqual(reference["force_ratio_to_baseline"], 1.0)
        self.assertEqual(reference["force_difference_from_reference_N"], 0.0)
        self.assertEqual(reference["force_ratio_to_reference"], 1.0)

        viscous = by_name["dual_two_sided_offset0p51_viscous_air"]
        self.assertTrue(_truthy(viscous["include_viscous_traction"]))
        self.assertGreater(float(viscous["viscosity_pa_s"]), 0.0)

        pressure_only = [
            row
            for row in rows
            if row["scenario"].endswith("pressure_only")
            and row["run_status"] == "completed"
        ]
        self.assertGreaterEqual(len(pressure_only), 1)
        for row in pressure_only:
            self.assertFalse(_truthy(row["include_viscous_traction"]))
            self.assertEqual(float(row["viscosity_pa_s"]), 0.0)

        for row in rows:
            self.assertTrue(REQUIRED_ROW_FIELDS.issubset(row))
            self.assertEqual(row["step_count"], 0)
            self.assertEqual(row["preflow_steps"], 20)
            self.assertFalse(_truthy(row["solid_advanced"]))
            self.assertFalse(_truthy(row["feedback_applied"]))
            self.assertEqual(row["flow_driver_mode"], "sustained_volume_source_inlet")
            self.assertEqual(row["source_profile"], "linear_ramp")
            self.assertEqual(int(row["source_ramp_steps"]), 2)
            self.assertEqual(
                row["scope_limit"],
                "fixed-solid traction formulation diagnostic only; no coupled 50-step or Fluent parity claim",
            )
            if row["run_status"] == "completed":
                self.assertEqual(row["worker_mode"], "isolated_subprocess")
                self.assertEqual(int(row["worker_returncode"]), 0)
                self.assertFalse(_truthy(row["worker_timed_out"]))
                self.assertGreater(float(row["worker_elapsed_s"]), 0.0)
                self.assertTrue(Path(row["worker_stdout_log"]).exists())
                self.assertTrue(Path(row["worker_stderr_log"]).exists())
                primary = float(row["primary_face_force_z_N"])
                secondary = float(row["secondary_face_force_z_N"])
                total = float(row["total_force_z_N"])
                split_total = float(row["primary_plus_secondary_force_z_N"])
                self.assertAlmostEqual(primary + secondary, total, places=8)
                self.assertAlmostEqual(split_total, total, places=8)
                self.assertLessEqual(
                    float(row["force_decomposition_residual_N"]),
                    1.0e-8,
                )
                self.assertEqual(row["primary_face_mean_pressure_pa"], "")
                self.assertEqual(row["secondary_face_mean_pressure_pa"], "")

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))

    def test_candidate_gate_blocks_failed_required_rows(self):
        rows = _stable_synthetic_rows()
        rows[-1] = dict(rows[-1], run_status="failed", status_reason="run_not_completed")
        payload = traction_matrix._payload(traction_matrix._apply_baseline_comparisons(rows))

        self.assertEqual(payload["reference_formulation_candidate"], "none")
        self.assertIn("required_formulation_failed", payload["candidate_blockers"])

    def test_candidate_gate_blocks_offset_instability_when_rows_complete(self):
        rows = _stable_synthetic_rows(
            offset025_force_z=-195.0,
            offset100_force_z=-6.7,
        )
        payload = traction_matrix._payload(traction_matrix._apply_baseline_comparisons(rows))

        self.assertEqual(payload["reference_formulation_candidate"], "none")
        self.assertEqual(
            payload["offset_sensitivity_status"],
            "offset_sensitivity_above_tolerance",
        )
        self.assertIn(
            "dual_two_sided_offset_sensitivity_above_tolerance",
            payload["candidate_blockers"],
        )
        self.assertAlmostEqual(float(payload["offset_force_ratio_max"]), 1.95)
        self.assertAlmostEqual(float(payload["offset_force_ratio_min"]), 0.067)

    def test_candidate_gate_promotes_stable_synthetic_baseline_only(self):
        rows = _stable_synthetic_rows()
        payload = traction_matrix._payload(traction_matrix._apply_baseline_comparisons(rows))

        self.assertEqual(
            payload["reference_formulation_candidate"],
            "dual_two_sided_offset0p51_pressure_only",
        )
        self.assertEqual(payload["candidate_status"], "candidate_found")
        self.assertEqual(payload["candidate_blockers"], [])
        self.assertEqual(payload["offset_sensitivity_status"], "offset_sensitivity_passed")
        self.assertEqual(payload["formulation_agreement_status"], "formulation_agreement_passed")
        self.assertEqual(
            payload["flow_snapshot_identity_status"],
            "flow_metrics_match_completed_rows",
        )

    def test_traction_formulation_histories_and_docs_are_scope_limited(self):
        payload = _read_json(HISTORY_JSON)
        histories = payload["histories"]

        self.assertEqual(payload["preflow_steps"], 20)
        for scenario in REQUIRED_SCENARIOS:
            self.assertIn(scenario, histories)
            history_path = HISTORIES_DIR / f"{scenario}_history.csv"
            self.assertTrue(history_path.exists(), msg=str(history_path))
            rows = _read_csv(history_path)
            if scenario == "dual_one_sided_offset0p51_pressure_only":
                self.assertEqual(rows, [])
                continue
            self.assertEqual(len(rows), 20)
            self.assertTrue(REQUIRED_HISTORY_FIELDS.issubset(rows[0]))
            for row in rows:
                primary = float(row["primary_face_force_z_N"])
                secondary = float(row["secondary_face_force_z_N"])
                total = float(row["total_force_z_N"])
                self.assertAlmostEqual(primary + secondary, total, places=8)

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("Traction Formulation", summary)
        self.assertIn("candidate_rule = all A/B/C rows supported", summary)
        self.assertIn("reference_formulation_candidate = none", verification)
        self.assertIn("one_sided_pressure_region_id", verification)
        self.assertIn("per-face pressure means", verification)
        self.assertIn("No 50-step run was performed", verification)
        self.assertIn("No Fluent parity claim is made", verification)


def _stable_synthetic_rows(
    *,
    offset025_force_z: float = -95.0,
    offset100_force_z: float = -105.0,
) -> list[dict[str, object]]:
    return [
        _synthetic_completed_row(
            "dual_two_sided_offset0p51_pressure_only",
            -100.0,
        ),
        _synthetic_completed_row(
            "dual_one_sided_offset0p51_pressure_only",
            -99.0,
            pressure_sampling_mode="one_sided_surface_pressure",
        ),
        _synthetic_completed_row(
            "single_mid_two_sided_offset0p00_pressure_only",
            -100.0,
            marker_layout="single_mid_surface",
            marker_face_offset_cells=0.0,
        ),
        _synthetic_completed_row(
            "dual_two_sided_offset0p25_pressure_only",
            offset025_force_z,
            marker_face_offset_cells=0.25,
        ),
        _synthetic_completed_row(
            "dual_two_sided_offset1p00_pressure_only",
            offset100_force_z,
            marker_face_offset_cells=1.0,
        ),
        _synthetic_completed_row(
            "dual_two_sided_offset0p51_viscous_air",
            -101.0,
            include_viscous_traction=True,
            viscosity_pa_s=1.8e-5,
        ),
    ]


def _synthetic_completed_row(
    scenario: str,
    total_force_z: float,
    *,
    marker_layout: str = "dual_physical_faces",
    pressure_sampling_mode: str = "two_sided_pressure_jump",
    include_viscous_traction: bool = False,
    viscosity_pa_s: float = 0.0,
    marker_face_offset_cells: float = 0.51,
) -> dict[str, object]:
    primary = total_force_z * 0.6
    secondary = total_force_z - primary
    return {
        "scenario": scenario,
        "run_status": "completed",
        "marker_layout": marker_layout,
        "pressure_sampling_mode": pressure_sampling_mode,
        "include_viscous_traction": include_viscous_traction,
        "viscosity_pa_s": viscosity_pa_s,
        "marker_face_offset_cells": marker_face_offset_cells,
        "step_count": 0,
        "preflow_steps": 20,
        "primary_face_invalid_marker_count": 0,
        "secondary_face_invalid_marker_count": 0,
        "primary_face_force_z_N": primary,
        "secondary_face_force_z_N": secondary,
        "total_force_z_N": total_force_z,
        "primary_plus_secondary_force_z_N": total_force_z,
        "force_decomposition_residual_N": 0.0,
        "marker_action_reaction_residual_N": 0.0,
        "scatter_action_reaction_residual_N": 0.0,
        "primary_face_mean_pressure_pa": 1.0,
        "secondary_face_mean_pressure_pa": 1.0,
        "flow_driver_uses_full_velocity_reset": False,
        "final_velocity_peak_mps": 22.0,
        "final_velocity_p999_mps": 20.0,
        "velocity_outlet_flux_ratio": 0.98,
        "pressure_outlet_flux_ratio": 0.02,
        "pressure_min_pa": -10.0,
        "pressure_max_pa": 12.0,
        "status_reason": "completed",
    }


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
