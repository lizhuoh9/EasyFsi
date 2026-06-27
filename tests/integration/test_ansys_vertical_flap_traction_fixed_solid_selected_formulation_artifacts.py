from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SELECTION_ROOT = ROOT / "traction_reference_formulation_selection_diagnostics"
DIAG_ROOT = ROOT / "traction_fixed_solid_selected_formulation_diagnostics"
MARKER_DIAGNOSTICS_ROOT = DIAG_ROOT / "marker_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_fixed_solid_selected_formulation_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_fixed_solid_selected_formulation_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_fixed_solid_selected_formulation_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_fixed_solid_selected_formulation_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SELECTION_MATRIX = SELECTION_ROOT / "traction_reference_formulation_selection_matrix.json"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_fixed_solid_selected_formulation_matrix.py"
)
EXPECTED_SELECTION_SOURCE = SELECTION_MATRIX.as_posix()
EXPECTED_SHARED_SHA = (
    "3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968"
)
EXPECTED_CANDIDATE = "anchored_dual_face_pressure_pair_with_per_face_one_sided"
EXPECTED_FIXED_SOLID_POLICY = "confirmed_shared_fixed_solid_snapshot_reused"
EXPECTED_ACTIVE_BLOCKERS = {
    "coupled_fsi_validation_pending",
    "no_fluent_parity_claim",
}
EXPECTED_RETIRED_BLOCKERS = {
    "fixed_solid_regenerated_validation_pending",
}
EXPECTED_BASELINE_SCENARIO = "fixed_solid_selected_baseline_probe0p51"
EXPECTED_ANCHORED_SCENARIOS = {
    "fixed_solid_selected_anchored_probe0p00",
    "fixed_solid_selected_anchored_probe0p25",
    "fixed_solid_selected_anchored_probe0p51",
    "fixed_solid_selected_anchored_probe0p625",
    "fixed_solid_selected_anchored_probe1p00",
}
EXPECTED_PER_FACE_SCENARIOS = {
    "fixed_solid_selected_per_face_one_sided_probe0p51",
    "fixed_solid_selected_per_face_one_sided_probe0p625",
    "fixed_solid_selected_per_face_one_sided_probe1p00",
}
EXPECTED_SCENARIOS = (
    {EXPECTED_BASELINE_SCENARIO}
    | EXPECTED_ANCHORED_SCENARIOS
    | EXPECTED_PER_FACE_SCENARIOS
)


class AnsysVerticalFlapFixedSolidSelectedFormulationArtifactTests(
    unittest.TestCase
):
    def test_fixed_solid_selected_formulation_matrix_is_reviewable(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(payload["purpose"], "fixed_solid_selected_formulation_matrix")
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertEqual(payload["selection_source"], EXPECTED_SELECTION_SOURCE)
        self.assertEqual(payload["selection_source_sha256"], _sha256_file(SELECTION_MATRIX))
        self.assertEqual(
            payload["fixed_solid_snapshot_policy"],
            EXPECTED_FIXED_SOLID_POLICY,
        )
        self.assertNotIn("coupled_fsi_validated", payload["candidate_status"])
        self.assertNotIn("fluent_parity", payload["candidate_status"])
        self.assertEqual(
            payload["new_or_confirmed_flow_snapshot_sha256"],
            EXPECTED_SHARED_SHA,
        )
        self.assertEqual(
            payload["anchor_source_flow_snapshot_sha256"],
            EXPECTED_SHARED_SHA,
        )
        self.assertNotEqual(payload["marker_geometry_sha256"], "")
        self.assertNotEqual(payload["anchor_source_marker_geometry_sha256"], "")
        self.assertEqual(
            payload["marker_geometry_sha256"],
            payload["anchor_source_marker_geometry_sha256"],
        )
        self.assertNotEqual(payload["anchor_map_sha256"], "")

        self.assertEqual(
            payload["candidate_status"],
            "fixed_solid_selected_formulation_validated",
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
        self.assertEqual(payload["completed_formulation_count"], len(EXPECTED_SCENARIOS))
        self.assertEqual(payload["unsupported_formulation_count"], 0)
        self.assertEqual(
            {item["blocker"] for item in payload["candidate_blockers"]},
            EXPECTED_ACTIVE_BLOCKERS,
        )
        self.assertEqual(
            set(payload["historical_blockers_retired"]),
            EXPECTED_RETIRED_BLOCKERS,
        )

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["fixed_solid_validation_status"], "completed")
            self.assertEqual(
                row["worker_mode"],
                "fixed_solid_selected_formulation_validation",
            )
            self.assertEqual(row["reference_formulation_candidate"], EXPECTED_CANDIDATE)
            self.assertEqual(row["pressure_pair_policy"], "baseline_anchored_cell_pair")
            self.assertEqual(row["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
            self.assertEqual(
                row["new_or_confirmed_flow_snapshot_sha256"],
                EXPECTED_SHARED_SHA,
            )
            self.assertEqual(
                row["anchor_source_flow_snapshot_sha256"],
                EXPECTED_SHARED_SHA,
            )
            self.assertEqual(
                row["anchor_source_marker_geometry_sha256"],
                row["marker_geometry_sha256"],
            )
            self.assertEqual(row["anchor_map_sha256"], payload["anchor_map_sha256"])
            self.assertEqual(row["source_artifact_json"], EXPECTED_SELECTION_SOURCE)
            self.assertEqual(row["source_artifact_sha256"], _sha256_file(SELECTION_MATRIX))
            self.assertTrue(Path(row["source_marker_diagnostics_json"]).exists())
            self.assertEqual(
                row["source_marker_diagnostics_sha256"],
                _sha256_file(Path(row["source_marker_diagnostics_json"])),
            )
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))

        baseline = by_scenario[EXPECTED_BASELINE_SCENARIO]
        self.assertEqual(baseline["one_sided_pressure_policy"], "disabled")

        for scenario in EXPECTED_ANCHORED_SCENARIOS:
            row = by_scenario[scenario]
            self.assertEqual(row["selection_component"], "pressure_pair_preselection")
            self.assertEqual(row["one_sided_pressure_policy"], "disabled")
            self.assertEqual(
                int(row["pressure_pair_anchor_selected_marker_count"]),
                int(row["total_marker_count"]),
            )
            self.assertEqual(int(row["pressure_pair_anchor_fallback_marker_count"]), 0)

        for scenario in EXPECTED_PER_FACE_SCENARIOS:
            row = by_scenario[scenario]
            self.assertEqual(row["selection_component"], "per_face_one_sided_pressure")
            self.assertEqual(row["one_sided_pressure_policy"], "per_face_mirrored")
            self.assertEqual(int(row["one_sided_marker_count"]), 24)
            self.assertEqual(int(row["one_sided_primary_marker_count"]), 12)
            self.assertEqual(int(row["one_sided_secondary_marker_count"]), 12)
            self.assertEqual(int(row["one_sided_anchor_selected_marker_count"]), 24)
            self.assertEqual(int(row["one_sided_anchor_fallback_marker_count"]), 0)

    def test_fixed_solid_candidate_gate_is_artifact_backed(self):
        payload = _read_json(MATRIX_JSON)
        acceptance = payload["fixed_solid_validation_acceptance"]
        gate = payload["stable_candidate_gate"]

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(
            acceptance["completed_row_count"],
            acceptance["expected_completed_row_count"],
        )
        self.assertTrue(acceptance["selected_reference_formulation_found"])
        self.assertTrue(acceptance["same_fixed_solid_snapshot_sha"])
        self.assertTrue(acceptance["anchor_source_matches_fixed_solid_snapshot"])
        self.assertTrue(acceptance["anchor_source_matches_marker_geometry"])
        self.assertTrue(acceptance["pressure_complete"])
        self.assertTrue(acceptance["invalid_marker_counts_zero"])
        self.assertTrue(acceptance["anchor_selected_all_markers"])
        self.assertTrue(acceptance["anchor_fallback_zero"])
        self.assertTrue(acceptance["one_sided_rows_complete"])
        self.assertLessEqual(
            float(acceptance["absolute_baseline_bias"]),
            float(gate["absolute_baseline_bias_max"]),
        )
        self.assertLessEqual(
            float(acceptance["force_ratio_relative_span"]),
            float(gate["force_ratio_relative_span_max"]),
        )
        self.assertLessEqual(
            float(acceptance["max_face_traction_decomposition_residual_pa"]),
            float(gate["traction_decomposition_residual_max"]),
        )

    def test_history_and_marker_wrappers_preserve_fixed_solid_provenance(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)

        self.assertEqual(set(history["histories"]), EXPECTED_SCENARIOS)
        self.assertEqual(set(payload["histories"]), EXPECTED_SCENARIOS)
        for scenario in EXPECTED_SCENARIOS:
            for source in (history["histories"][scenario], payload["histories"][scenario]):
                self.assertEqual(source["scenario"], scenario)
                self.assertTrue(source["source_history_present"])
                self.assertNotEqual(source["source_flow_phase"], "")
                self.assertEqual(
                    source["flow_phase"],
                    "fixed_solid_selected_formulation_validation",
                )
                self.assertEqual(source["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
                self.assertEqual(
                    source["fixed_solid_snapshot_policy"],
                    EXPECTED_FIXED_SOLID_POLICY,
                )
                self.assertEqual(
                    source["reference_formulation_candidate"],
                    EXPECTED_CANDIDATE,
                )

        for row in payload["rows"]:
            wrapper = _read_json(Path(row["marker_diagnostics_json"]))
            source_path = Path(row["source_marker_diagnostics_json"])
            self.assertEqual(wrapper["scenario"], row["scenario"])
            self.assertEqual(wrapper["source_scenario"], row["source_scenario"])
            self.assertEqual(wrapper["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
            self.assertEqual(
                wrapper["fixed_solid_snapshot_policy"],
                EXPECTED_FIXED_SOLID_POLICY,
            )
            self.assertEqual(
                wrapper["reference_formulation_candidate"],
                EXPECTED_CANDIDATE,
            )
            self.assertEqual(wrapper["source_marker_diagnostics_json"], source_path.as_posix())
            self.assertEqual(
                wrapper["source_marker_diagnostics_sha256"],
                _sha256_file(source_path),
            )
            self.assertEqual(wrapper["marker_count"], int(row["total_marker_count"]))

    def test_summary_csv_and_checksums_match_fixed_solid_artifacts(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("fixed-solid selected formulation", summary)
        self.assertIn("does not claim coupled FSI", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn(EXPECTED_FIXED_SOLID_POLICY, summary)
        self.assertIn(EXPECTED_CANDIDATE, summary)
        self.assertNotIn("fresh regenerated coupled run", summary)
        self.assertNotIn("coupled FSI validated", summary)
        self.assertNotIn("Fluent parity validated", summary)
        for scenario in EXPECTED_SCENARIOS:
            self.assertIn(scenario, summary)

        with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), len(payload["rows"]))
        self.assertEqual(
            {row["scenario"] for row in csv_rows},
            {row["scenario"] for row in payload["rows"]},
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
            marker_rel = Path(row["marker_diagnostics_json"]).relative_to(DIAG_ROOT)
            self.assertIn(marker_rel.as_posix(), checksum_rows)


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
