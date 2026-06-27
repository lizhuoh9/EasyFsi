from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_reference_formulation_selection_diagnostics"
MARKER_DIAGNOSTICS_ROOT = DIAG_ROOT / "marker_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_reference_formulation_selection_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_reference_formulation_selection_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_reference_formulation_selection_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_reference_formulation_selection_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_reference_formulation_selection_matrix.py"
)
EXPECTED_SHARED_SHA = (
    "3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968"
)
EXPECTED_CANDIDATE = "anchored_dual_face_pressure_pair_with_per_face_one_sided"
EXPECTED_BASELINE_SCENARIO = "reference_baseline_anchored_two_sided_probe0p51"
EXPECTED_ANCHORED_SCENARIOS = {
    "reference_anchored_two_sided_probe0p00",
    "reference_anchored_two_sided_probe0p25",
    "reference_anchored_two_sided_probe0p375",
    "reference_anchored_two_sided_probe0p625",
    "reference_anchored_two_sided_probe1p00",
}
EXPECTED_PER_FACE_SCENARIOS = {
    "reference_per_face_one_sided_probe0p51",
    "reference_per_face_one_sided_probe0p625",
    "reference_per_face_one_sided_probe1p00",
}
EXPECTED_SCENARIOS = (
    {EXPECTED_BASELINE_SCENARIO}
    | EXPECTED_ANCHORED_SCENARIOS
    | EXPECTED_PER_FACE_SCENARIOS
)
EXPECTED_ALLOWED_BLOCKERS = {
    "sampling_only_no_coupled_fsi",
    "no_fluent_parity_claim",
    "fixed_solid_regenerated_validation_pending",
    "coupled_fsi_validation_pending",
}
FORBIDDEN_ACTIVE_BLOCKERS = {
    "dual_face_one_sided_unsupported",
    "reference_selection_deferred",
}
SCOPE_REQUIRED_FRAGMENTS = (
    "shared snapshot",
    "sampling-only",
    "does not claim Fluent parity",
)


class AnsysVerticalFlapReferenceFormulationSelectionArtifactTests(
    unittest.TestCase
):
    def test_selection_matrix_is_shared_snapshot_sampling_only(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(payload["baseline_scenario"], EXPECTED_BASELINE_SCENARIO)
        self.assertEqual(set(payload["anchored_scenarios"]), EXPECTED_ANCHORED_SCENARIOS)
        self.assertEqual(set(payload["per_face_scenarios"]), EXPECTED_PER_FACE_SCENARIOS)
        self.assertEqual(
            payload["purpose"],
            "shared_flow_snapshot_reference_formulation_selection_matrix",
        )
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertNotIn("D:", payload["source_script"])
        self.assertEqual(
            payload["flow_snapshot_identity_status"],
            "shared_snapshot_sha256_identical_completed_rows",
        )
        self.assertEqual(payload["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
        self.assertEqual(payload["flow_snapshot_sha256"], manifest["field_sha256"])
        self.assertTrue(SHARED_FIELDS.exists())
        self.assertEqual(_sha256_file(SHARED_FIELDS), EXPECTED_SHARED_SHA)

        self.assertEqual(
            payload["candidate_status"],
            "reference_formulation_candidate_selected",
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
        self.assertIn(
            "dual_face_one_sided_unsupported",
            payload["historical_blockers_retired"],
        )
        _assert_scope(self, payload["scope_limit"])

        blockers = {item["blocker"] for item in payload["candidate_blockers"]}
        self.assertEqual(blockers, EXPECTED_ALLOWED_BLOCKERS)
        self.assertTrue(FORBIDDEN_ACTIVE_BLOCKERS.isdisjoint(blockers))

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["formulation_status"], "completed")
            self.assertEqual(
                row["worker_mode"],
                "shared_snapshot_reference_formulation_selection",
            )
            self.assertEqual(row["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["reference_formulation_candidate"], EXPECTED_CANDIDATE)
            self.assertEqual(row["pressure_pair_policy"], "baseline_anchored_cell_pair")
            self.assertTrue(Path(row["source_artifact_json"]).exists())
            self.assertTrue(Path(row["source_marker_diagnostics_json"]).exists())
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            _assert_scope(self, row["scope_limit"])

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
            self.assertEqual(row["pressure_sampling_mode"], "one_sided_surface_pressure")
            self.assertEqual(int(row["one_sided_marker_count"]), int(row["total_marker_count"]))
            self.assertEqual(int(row["one_sided_primary_marker_count"]), 12)
            self.assertEqual(int(row["one_sided_secondary_marker_count"]), 12)
            self.assertEqual(int(row["one_sided_anchor_selected_marker_count"]), 24)
            self.assertEqual(int(row["one_sided_anchor_fallback_marker_count"]), 0)
            self.assertEqual(
                row["one_sided_side_selection_counts"],
                {"inside": 0, "outside": 24},
            )

    def test_selection_candidate_gate_is_artifact_backed(self):
        payload = _read_json(MATRIX_JSON)
        acceptance = payload["reference_selection_acceptance"]
        gate = payload["stable_candidate_gate"]

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(
            acceptance["completed_row_count"],
            acceptance["expected_completed_row_count"],
        )
        self.assertTrue(acceptance["pressure_pair_preselection_candidate_found"])
        self.assertTrue(acceptance["per_face_one_sided_pressure_completed"])
        self.assertTrue(acceptance["same_shared_snapshot_sha"])
        self.assertTrue(acceptance["pressure_complete"])
        self.assertTrue(acceptance["invalid_marker_counts_zero"])
        self.assertTrue(acceptance["anchor_selected_all_markers"])
        self.assertTrue(acceptance["anchor_fallback_zero"])
        self.assertTrue(acceptance["scope_sampling_only"])
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

    def test_history_and_marker_wrappers_preserve_source_evidence(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)

        self.assertEqual(set(history["histories"]), EXPECTED_SCENARIOS)
        self.assertEqual(set(payload["histories"]), EXPECTED_SCENARIOS)
        for scenario in EXPECTED_SCENARIOS:
            for source in (history["histories"][scenario], payload["histories"][scenario]):
                self.assertEqual(source["scenario"], scenario)
                self.assertEqual(
                    source["flow_phase"],
                    "shared_snapshot_reference_formulation_selection",
                )
                self.assertEqual(source["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
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
            self.assertEqual(wrapper["reference_formulation_candidate"], EXPECTED_CANDIDATE)
            self.assertEqual(wrapper["source_marker_diagnostics_json"], source_path.as_posix())
            self.assertEqual(
                wrapper["source_marker_diagnostics_sha256"],
                _sha256_file(source_path),
            )
            self.assertEqual(wrapper["marker_count"], int(row["total_marker_count"]))
            if row["selection_component"] == "per_face_one_sided_pressure":
                self.assertEqual(
                    wrapper["one_sided_stats"]["one_sided_side_selection_counts"],
                    {"inside": 0, "outside": 24},
                )
                self.assertEqual(wrapper["one_sided_stats"]["one_sided_marker_count"], 24)

    def test_summary_csv_and_checksums_match_selection_artifacts(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("shared snapshot", summary)
        self.assertIn("marker-traction sampling", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn(EXPECTED_CANDIDATE, summary)
        self.assertIn("fixed-solid and coupled validations remain pending", summary)
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


def _assert_scope(testcase: unittest.TestCase, scope: str) -> None:
    for fragment in SCOPE_REQUIRED_FRAGMENTS:
        testcase.assertIn(fragment, scope)


if __name__ == "__main__":
    unittest.main()
