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
EXPECTED_SCENARIO = "selected_formulation_coupled_smoke_5step"
EXPECTED_PENDING_BLOCKERS = {
    "coupled_fsi_validation_pending",
    "no_fluent_parity_claim",
    "blocked_invalid_marker_sampling",
}
EXPECTED_PASS_BLOCKERS = {
    "long_coupled_validation_pending",
    "no_fluent_parity_claim",
}


class AnsysVerticalFlapSelectedFormulationCoupledSmokeArtifactTests(
    unittest.TestCase
):
    def test_smoke_matrix_records_selected_formulation_sources(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        self.assertEqual(payload["purpose"], "selected_formulation_coupled_smoke_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["scenario_count"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], EXPECTED_SCENARIO)
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

        row = rows[0]
        self.assertEqual(row["reference_formulation_candidate"], EXPECTED_CANDIDATE)
        self.assertEqual(
            row["reference_selection_source_sha256"],
            _sha256_file(REFERENCE_SELECTION),
        )
        self.assertEqual(
            row["fixed_solid_selected_formulation_source_sha256"],
            _sha256_file(FIXED_SOLID_SELECTION),
        )
        self.assertEqual(int(row["requested_step_count"]), 5)
        self.assertGreaterEqual(int(row["completed_step_count"]), 0)
        self.assertTrue(Path(row["scenario_diagnostics_json"]).exists())

    def test_smoke_status_is_fail_closed_until_gates_pass(self):
        payload = _read_json(MATRIX_JSON)
        row = payload["rows"][0]
        acceptance = payload["smoke_acceptance"]
        blockers = {item["blocker"] for item in payload["candidate_blockers"]}

        if payload["candidate_status"] == "selected_formulation_coupled_smoke_passed":
            self.assertTrue(acceptance["accepted"])
            self.assertEqual(blockers, EXPECTED_PASS_BLOCKERS)
            self.assertEqual(
                int(row["completed_step_count"]),
                int(row["requested_step_count"]),
            )
            self.assertTrue(bool(row["fluid_finite"]))
            self.assertTrue(bool(row["pressure_finite"]))
            self.assertTrue(bool(row["solid_position_finite"]))
            self.assertEqual(int(row["invalid_marker_count_max"]), 0)
            self.assertEqual(int(row["anchor_fallback_marker_count_max"]), 0)
            self.assertGreaterEqual(int(row["one_sided_marker_count_min"]), 24)
            self.assertLessEqual(
                float(row["force_action_reaction_residual_max_n"]),
                float(payload["stable_candidate_gate"]["force_action_reaction_residual_max_n"]),
            )
        else:
            self.assertEqual(
                payload["candidate_status"],
                "selected_formulation_coupled_smoke_pending",
            )
            self.assertFalse(acceptance["accepted"])
            self.assertEqual(blockers, EXPECTED_PENDING_BLOCKERS)
            self.assertEqual(row["run_status"], "blocked")
            self.assertNotEqual(row["smoke_status"], "passed")
            self.assertIn("coupled_fsi_validation_pending", blockers)

        self.assertIn("no_fluent_parity_claim", blockers)

    def test_history_summary_csv_and_checksums_are_consistent(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        row = payload["rows"][0]

        self.assertEqual(set(history["histories"]), {EXPECTED_SCENARIO})
        self.assertEqual(
            history["histories"][EXPECTED_SCENARIO]["scenario"],
            EXPECTED_SCENARIO,
        )
        self.assertEqual(
            history["histories"][EXPECTED_SCENARIO][
                "reference_formulation_candidate"
            ],
            EXPECTED_CANDIDATE,
        )
        self.assertIn("selected-formulation coupled smoke", summary)
        self.assertIn("does not claim 50-step validation", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertNotIn("Fluent parity validated", summary)
        self.assertNotIn("50-step validation passed", summary)
        self.assertIn(payload["candidate_status"], summary)
        self.assertIn(row["smoke_status"], summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["scenario"], EXPECTED_SCENARIO)

        checksum_rows = _read_checksums(CHECKSUMS)
        for artifact in (
            MATRIX_JSON.name,
            MATRIX_CSV.name,
            HISTORY_JSON.name,
            SUMMARY_MD.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))
        diagnostics_rel = Path(row["scenario_diagnostics_json"]).relative_to(DIAG_ROOT)
        self.assertIn(diagnostics_rel.as_posix(), checksum_rows)
        self.assertEqual(
            checksum_rows[diagnostics_rel.as_posix()],
            _sha256_file(DIAG_ROOT / diagnostics_rel),
        )


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
