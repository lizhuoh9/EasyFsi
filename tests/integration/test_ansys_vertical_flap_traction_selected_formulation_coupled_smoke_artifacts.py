from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
REFERENCE_SELECTION = (
    ROOT
    / "traction_reference_formulation_selection_diagnostics"
    / "traction_reference_formulation_selection_matrix.json"
)
FIXED_SOLID_SELECTION = (
    ROOT
    / "traction_fixed_solid_selected_formulation_diagnostics"
    / "traction_fixed_solid_selected_formulation_matrix.json"
)
SELECTED_ANCHOR_MARKERS_JSON = (
    ROOT
    / "traction_fixed_solid_selected_formulation_diagnostics"
    / "marker_diagnostics"
    / "fixed_solid_selected_per_face_one_sided_probe0p51_markers.json"
)
DIAG_ROOT = ROOT / "traction_selected_formulation_coupled_smoke_diagnostics"
SCENARIO_DIAGNOSTICS_ROOT = DIAG_ROOT / "scenario_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_selected_formulation_coupled_smoke_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_selected_formulation_coupled_smoke_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_selected_formulation_coupled_smoke_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_selected_formulation_coupled_smoke_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_selected_formulation_coupled_smoke.py"
)
EXPECTED_CANDIDATE = "anchored_dual_face_pressure_pair_with_per_face_one_sided"
EXPECTED_PREFLIGHT_SCENARIO = "selected_formulation_coupled_preflight_1step"
EXPECTED_SMOKE_SCENARIO = "selected_formulation_coupled_smoke_5step"
EXPECTED_PENDING_BASE_BLOCKERS = {
    "coupled_fsi_validation_pending",
    "no_fluent_parity_claim",
}
EXPECTED_PENDING_SMOKE_BLOCKERS = {
    "blocked_nan_or_inf",
    "blocked_invalid_marker_sampling",
    "blocked_anchor_fallback",
    "blocked_one_sided_incomplete",
    "blocked_force_residual",
    "blocked_velocity_threshold",
    "blocked_pressure_threshold",
    "blocked_solid_displacement_threshold",
    "blocked_requested_5step_not_completed",
    "not_run",
}
EXPECTED_PASS_BLOCKERS = {
    "long_coupled_validation_pending",
    "no_fluent_parity_claim",
}


class AnsysVerticalFlapSelectedFormulationCoupledSmokeArtifactTests(
    unittest.TestCase
):
    def test_smoke_matrix_records_two_selected_formulation_rows(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}
        self.assertEqual(payload["purpose"], "selected_formulation_coupled_smoke_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["scenario_count"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(by_scenario),
            {EXPECTED_PREFLIGHT_SCENARIO, EXPECTED_SMOKE_SCENARIO},
        )
        self.assertEqual(payload["reference_formulation_candidate"], EXPECTED_CANDIDATE)
        self.assertEqual(
            payload["pressure_pair_policy_candidate"],
            "baseline_anchored_cell_pair",
        )
        self.assertEqual(
            payload["one_sided_pressure_policy_candidate"],
            "per_face_mirrored",
        )
        self.assertEqual(
            payload["reference_selection_source"],
            REFERENCE_SELECTION.as_posix(),
        )
        self.assertEqual(
            payload["reference_selection_source_sha256"],
            _sha256_file(REFERENCE_SELECTION),
        )
        self.assertEqual(
            payload["fixed_solid_selected_formulation_source"],
            FIXED_SOLID_SELECTION.as_posix(),
        )
        self.assertEqual(
            payload["fixed_solid_selected_formulation_source_sha256"],
            _sha256_file(FIXED_SOLID_SELECTION),
        )
        self.assertEqual(
            payload["selected_anchor_markers_source"],
            SELECTED_ANCHOR_MARKERS_JSON.as_posix(),
        )
        self.assertEqual(
            payload["selected_anchor_markers_source_sha256"],
            _sha256_file(SELECTED_ANCHOR_MARKERS_JSON),
        )

        for row in rows:
            self.assertEqual(row["reference_formulation_candidate"], EXPECTED_CANDIDATE)
            self.assertEqual(
                row["reference_selection_source_sha256"],
                _sha256_file(REFERENCE_SELECTION),
            )
            self.assertEqual(
                row["fixed_solid_selected_formulation_source_sha256"],
                _sha256_file(FIXED_SOLID_SELECTION),
            )
            self.assertEqual(
                row["selected_anchor_markers_source"],
                SELECTED_ANCHOR_MARKERS_JSON.as_posix(),
            )
            self.assertEqual(
                row["selected_anchor_markers_source_sha256"],
                _sha256_file(SELECTED_ANCHOR_MARKERS_JSON),
            )
            self.assertEqual(int(row["requested_step_count"]), 5)
            self.assertGreaterEqual(int(row["completed_step_count"]), 0)
            self.assertTrue(Path(row["scenario_diagnostics_json"]).exists())

        self.assertEqual(
            int(by_scenario[EXPECTED_PREFLIGHT_SCENARIO]["diagnostic_step_count"]),
            1,
        )
        self.assertEqual(
            int(by_scenario[EXPECTED_SMOKE_SCENARIO]["diagnostic_step_count"]),
            5,
        )

    def test_preflight_row_hardens_anchor_injected_one_step_checkpoint(self):
        payload = _read_json(MATRIX_JSON)
        row = _row_by_scenario(payload)[EXPECTED_PREFLIGHT_SCENARIO]
        acceptance = payload["row_acceptance"][EXPECTED_PREFLIGHT_SCENARIO]
        diagnostics = _read_json(Path(row["scenario_diagnostics_json"]))
        history = diagnostics["report"]["history"]
        first_step = history[0]
        report = diagnostics["report"]

        self.assertIn(
            payload["candidate_status"],
            {
                "selected_formulation_coupled_smoke_pending",
                "selected_formulation_coupled_smoke_passed",
            },
        )
        self.assertEqual(row["smoke_status"], "blocked_requested_5step_not_completed")
        self.assertEqual(row["run_status"], "blocked")
        self.assertEqual(int(row["completed_step_count"]), 1)
        self.assertEqual(int(row["requested_step_count"]), 5)
        self.assertEqual(row["first_failed_step"], 2)
        self.assertEqual(row["first_failed_gate"], "completed_requested_steps")
        self.assertEqual(int(row["invalid_marker_count_max"]), 0)
        self.assertGreaterEqual(
            int(row["pressure_pair_anchor_active_marker_count_min"]),
            24,
        )
        self.assertGreaterEqual(int(row["anchor_selected_marker_count_min"]), 24)
        self.assertEqual(int(row["anchor_fallback_marker_count_max"]), 0)
        self.assertGreaterEqual(int(row["one_sided_marker_count_min"]), 24)
        self.assertEqual(int(row["one_sided_anchor_fallback_marker_count_max"]), 0)

        self.assertTrue(acceptance["finite_fields"])
        self.assertTrue(acceptance["no_marker_invalid"])
        self.assertTrue(acceptance["anchor_selected_all"])
        self.assertTrue(acceptance["anchor_fallback_zero"])
        self.assertTrue(acceptance["one_sided_complete"])
        self.assertTrue(acceptance["one_sided_fallback_zero"])
        self.assertTrue(acceptance["residual_within_tolerance"])
        self.assertFalse(acceptance["completed_requested_steps"])
        self.assertFalse(acceptance["accepted"])

        self.assertEqual(int(first_step["stress_invalid_marker_count"]), 0)
        self.assertEqual(int(first_step["stress_valid_marker_count"]), 24)
        self.assertEqual(int(first_step["primary_face_invalid_marker_count"]), 0)
        self.assertEqual(int(first_step["secondary_face_invalid_marker_count"]), 0)
        self.assertEqual(int(first_step["one_sided_pressure_marker_count"]), 24)
        self.assertEqual(int(report["surface_feedback_updated_marker_count"]), 24)

    def test_five_step_status_is_pass_or_exact_fail_closed(self):
        payload = _read_json(MATRIX_JSON)
        row = _row_by_scenario(payload)[EXPECTED_SMOKE_SCENARIO]
        acceptance = payload["smoke_acceptance"]
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}

        if payload["candidate_status"] == "selected_formulation_coupled_smoke_passed":
            self.assertTrue(acceptance["accepted"])
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["smoke_status"], "passed")
            self.assertEqual(blockers, EXPECTED_PASS_BLOCKERS)
            self.assertEqual(
                int(row["completed_step_count"]),
                int(row["requested_step_count"]),
            )
            self.assertEqual(int(row["completed_step_count"]), 5)
            self.assertTrue(bool(row["fluid_finite"]))
            self.assertTrue(bool(row["pressure_finite"]))
            self.assertTrue(bool(row["solid_position_finite"]))
            self.assertEqual(int(row["invalid_marker_count_max"]), 0)
            self.assertGreaterEqual(
                int(row["pressure_pair_anchor_active_marker_count_min"]),
                24,
            )
            self.assertGreaterEqual(int(row["anchor_selected_marker_count_min"]), 24)
            self.assertEqual(int(row["anchor_fallback_marker_count_max"]), 0)
            self.assertGreaterEqual(int(row["one_sided_marker_count_min"]), 24)
            self.assertEqual(int(row["one_sided_anchor_fallback_marker_count_max"]), 0)
            self.assertLessEqual(
                float(row["force_action_reaction_residual_max_n"]),
                float(payload["stable_candidate_gate"]["force_action_reaction_residual_max_n"]),
            )
            self.assertIn(
                "coupled_fsi_validation_pending",
                payload["historical_blockers_retired"],
            )
        else:
            self.assertEqual(
                payload["candidate_status"],
                "selected_formulation_coupled_smoke_pending",
            )
            self.assertFalse(acceptance["accepted"])
            self.assertTrue(EXPECTED_PENDING_BASE_BLOCKERS.issubset(blockers))
            self.assertIn(row["smoke_status"], EXPECTED_PENDING_SMOKE_BLOCKERS)
            self.assertIn(row["smoke_status"], blockers)
            self.assertEqual(row["run_status"], "blocked")
            self.assertNotEqual(row["smoke_status"], "passed")
            self.assertIn("coupled_fsi_validation_pending", blockers)
            self.assertNotEqual(row["first_failed_step"], "")
            self.assertNotEqual(row["first_failed_gate"], "")

        self.assertIn("no_fluent_parity_claim", blockers)
        completed_steps = int(row["completed_step_count"])
        for key in (
            "invalid_marker_count_by_step",
            "one_sided_marker_count_by_step",
            "anchor_selected_marker_count_by_step",
            "anchor_fallback_marker_count_by_step",
            "one_sided_anchor_fallback_marker_count_by_step",
            "force_action_reaction_residual_by_step",
            "max_velocity_by_step",
            "max_pressure_abs_by_step",
            "max_displacement_by_step",
        ):
            self.assertEqual(len(row[key]), completed_steps, key)

    def test_history_summary_csv_and_checksums_are_consistent(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        by_scenario = _row_by_scenario(payload)

        self.assertEqual(
            set(history["histories"]),
            {EXPECTED_PREFLIGHT_SCENARIO, EXPECTED_SMOKE_SCENARIO},
        )
        for scenario in (EXPECTED_PREFLIGHT_SCENARIO, EXPECTED_SMOKE_SCENARIO):
            self.assertEqual(history["histories"][scenario]["scenario"], scenario)
            self.assertEqual(
                history["histories"][scenario]["reference_formulation_candidate"],
                EXPECTED_CANDIDATE,
            )

        self.assertIn("selected-formulation coupled smoke", summary)
        self.assertIn("does not claim 50-step validation", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertNotIn("50-step validation passed", summary)
        self.assertIn(payload["candidate_status"], summary)
        self.assertIn(by_scenario[EXPECTED_PREFLIGHT_SCENARIO]["smoke_status"], summary)
        self.assertIn(by_scenario[EXPECTED_SMOKE_SCENARIO]["smoke_status"], summary)
        self.assertIn("step | invalid | one-sided | anchor selected", summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(
            {row["scenario"] for row in csv_rows},
            {EXPECTED_PREFLIGHT_SCENARIO, EXPECTED_SMOKE_SCENARIO},
        )

        checksum_rows = _read_checksums(CHECKSUMS)
        for artifact in (
            MATRIX_JSON.name,
            MATRIX_CSV.name,
            HISTORY_JSON.name,
            SUMMARY_MD.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))
        for row in payload["rows"]:
            diagnostics_rel = Path(row["scenario_diagnostics_json"]).relative_to(
                DIAG_ROOT
            )
            self.assertIn(diagnostics_rel.as_posix(), checksum_rows)
            self.assertEqual(
                checksum_rows[diagnostics_rel.as_posix()],
                _sha256_file(DIAG_ROOT / diagnostics_rel),
            )


def _row_by_scenario(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["scenario"]: row for row in payload["rows"]}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
