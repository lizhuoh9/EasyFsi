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
SOURCE_SMOKE_MATRIX = (
    ROOT
    / "traction_selected_formulation_coupled_smoke_diagnostics"
    / "traction_selected_formulation_coupled_smoke_matrix.json"
)
DIAG_ROOT = ROOT / "traction_selected_formulation_coupled_step50_diagnostics"
SCENARIO_DIAGNOSTICS_ROOT = DIAG_ROOT / "scenario_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_selected_formulation_coupled_step50_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_selected_formulation_coupled_step50_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_selected_formulation_coupled_step50_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_selected_formulation_coupled_step50_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_selected_formulation_coupled_step50.py"
)
EXPECTED_CANDIDATE = "anchored_dual_face_pressure_pair_with_per_face_one_sided"
EXPECTED_SCENARIOS = {
    "selected_formulation_coupled_step10": 10,
    "selected_formulation_coupled_step30": 30,
    "selected_formulation_coupled_step50": 50,
}
EXPECTED_CANDIDATE_STATUS = "selected_formulation_coupled_step50_passed"
EXPECTED_ACTIVE_BLOCKERS = {"no_fluent_parity_claim"}
EXPECTED_RETIRED_BLOCKERS = ["long_coupled_validation_pending"]


class AnsysVerticalFlapSelectedFormulationCoupledStep50ArtifactTests(
    unittest.TestCase
):
    def test_step50_matrix_records_three_staged_rows(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}
        self.assertEqual(payload["purpose"], "selected_formulation_coupled_step50_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["scenario_count"], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(by_scenario), set(EXPECTED_SCENARIOS))
        self.assertEqual(payload["reference_formulation_candidate"], EXPECTED_CANDIDATE)
        self.assertEqual(
            payload["pressure_pair_policy_candidate"],
            "baseline_anchored_cell_pair",
        )
        self.assertEqual(
            payload["one_sided_pressure_policy_candidate"],
            "per_face_mirrored",
        )
        self.assertEqual(payload["source_5step_smoke_matrix"], SOURCE_SMOKE_MATRIX.as_posix())
        self.assertEqual(
            payload["source_5step_smoke_matrix_sha256"],
            _sha256_file(SOURCE_SMOKE_MATRIX),
        )
        self.assertEqual(
            payload["source_5step_smoke_candidate_status"],
            "selected_formulation_coupled_smoke_passed",
        )
        self.assertEqual(
            payload["reference_selection_source_sha256"],
            _sha256_file(REFERENCE_SELECTION),
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

        for scenario, requested_step_count in EXPECTED_SCENARIOS.items():
            row = by_scenario[scenario]
            self.assertEqual(row["reference_formulation_candidate"], EXPECTED_CANDIDATE)
            self.assertEqual(
                row["pressure_pair_policy_candidate"],
                "baseline_anchored_cell_pair",
            )
            self.assertEqual(
                row["one_sided_pressure_policy_candidate"],
                "per_face_mirrored",
            )
            self.assertEqual(int(row["requested_step_count"]), requested_step_count)
            self.assertEqual(
                int(row["diagnostic_step_count"]),
                requested_step_count,
            )
            self.assertEqual(
                row["selected_anchor_markers_source"],
                SELECTED_ANCHOR_MARKERS_JSON.as_posix(),
            )
            self.assertEqual(
                row["selected_anchor_markers_source_sha256"],
                _sha256_file(SELECTED_ANCHOR_MARKERS_JSON),
            )
            self.assertEqual(
                row["source_5step_smoke_matrix"],
                SOURCE_SMOKE_MATRIX.as_posix(),
            )
            self.assertEqual(
                row["source_5step_smoke_matrix_sha256"],
                _sha256_file(SOURCE_SMOKE_MATRIX),
            )
            self.assertTrue(Path(row["scenario_diagnostics_json"]).exists())
            self.assertEqual(
                Path(row["scenario_diagnostics_json"]).parent,
                SCENARIO_DIAGNOSTICS_ROOT,
            )

    def test_step50_candidate_status_and_blockers_are_exact(self):
        payload = _read_json(MATRIX_JSON)
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}

        self.assertEqual(payload["candidate_status"], EXPECTED_CANDIDATE_STATUS)
        self.assertNotIn("fluent_parity_claim", payload)
        self.assertEqual(blockers, EXPECTED_ACTIVE_BLOCKERS)
        self.assertEqual(payload["historical_blockers_retired"], EXPECTED_RETIRED_BLOCKERS)
        self.assertIn("no_fluent_parity_claim", blockers)
        self.assertNotIn("long_coupled_validation_pending", blockers)
        self.assertEqual(payload["first_failed_scenario"], "")
        self.assertEqual(payload["first_failed_step"], "")
        self.assertEqual(payload["first_failed_gate"], "")

    def test_step50_pass_does_not_claim_fluent_parity(self):
        payload = _read_json(MATRIX_JSON)
        by_scenario = _row_by_scenario(payload)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        step50 = by_scenario["selected_formulation_coupled_step50"]

        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertNotIn("fluent_parity_claim", payload)
        self.assertEqual(
            {item["blocker"] for item in payload["candidate_blockers"]},
            {"no_fluent_parity_claim"},
        )
        self.assertEqual(
            payload["historical_blockers_retired"],
            ["long_coupled_validation_pending"],
        )
        self.assertEqual(step50["smoke_status"], "passed")
        self.assertEqual(step50["run_status"], "completed")
        self.assertEqual(int(step50["completed_step_count"]), 50)
        self.assertEqual(int(step50["requested_step_count"]), 50)
        self.assertEqual(int(step50["invalid_marker_count_max"]), 0)
        self.assertGreaterEqual(int(step50["one_sided_marker_count_min"]), 24)
        self.assertGreaterEqual(int(step50["anchor_selected_marker_count_min"]), 24)
        self.assertEqual(int(step50["anchor_fallback_marker_count_max"]), 0)

    def test_step50_row_acceptance_matches_gate_fields(self):
        payload = _read_json(MATRIX_JSON)
        by_scenario = _row_by_scenario(payload)
        acceptance_by_scenario = payload["row_acceptance"]

        for scenario, requested_step_count in EXPECTED_SCENARIOS.items():
            row = by_scenario[scenario]
            acceptance = acceptance_by_scenario[scenario]
            completed_steps = int(row["completed_step_count"])
            self.assertTrue(acceptance["accepted"])
            self.assertTrue(acceptance["completed_requested_steps"])
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["smoke_status"], "passed")
            self.assertEqual(int(row["completed_step_count"]), requested_step_count)
            self.assertEqual(row["first_failed_step"], "")
            self.assertEqual(row["first_failed_gate"], "")
            self.assertEqual(row["first_failed_gate_value"], "")
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
            self.assertEqual(
                int(row["one_sided_anchor_fallback_marker_count_max"]),
                0,
            )
            self.assertLessEqual(
                float(row["force_action_reaction_residual_max_n"]),
                float(
                    payload["stable_candidate_gate"][
                        "force_action_reaction_residual_max_n"
                    ]
                ),
            )

            for key in (
                "invalid_marker_count_by_step",
                "one_sided_marker_count_by_step",
                "anchor_selected_marker_count_by_step",
                "anchor_fallback_marker_count_by_step",
                "one_sided_anchor_fallback_marker_count_by_step",
                "force_action_reaction_residual_by_step",
                "marker_force_z_by_step",
                "max_velocity_by_step",
                "max_pressure_abs_by_step",
                "max_displacement_by_step",
            ):
                self.assertEqual(len(row[key]), completed_steps, key)

            self.assertGreaterEqual(float(row["max_velocity_growth_ratio"]), 0.0)
            self.assertGreaterEqual(float(row["max_pressure_growth_ratio"]), 0.0)
            self.assertGreaterEqual(float(row["max_displacement_growth_ratio"]), 0.0)
            self.assertGreaterEqual(int(row["force_sign_flip_count"]), 0)

    def test_history_summary_csv_and_checksums_are_consistent(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")

        self.assertEqual(set(history["histories"]), set(EXPECTED_SCENARIOS))
        for scenario in EXPECTED_SCENARIOS:
            self.assertEqual(history["histories"][scenario]["scenario"], scenario)
            self.assertEqual(
                history["histories"][scenario]["reference_formulation_candidate"],
                EXPECTED_CANDIDATE,
            )
            self.assertEqual(
                history["histories"][scenario]["source_5step_smoke_matrix"],
                SOURCE_SMOKE_MATRIX.as_posix(),
            )

        self.assertIn("selected-formulation coupled step50", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertIn(payload["candidate_status"], summary)
        self.assertIn("scenario | status | completed/requested", summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(
            {row["scenario"] for row in csv_rows},
            set(EXPECTED_SCENARIOS),
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
