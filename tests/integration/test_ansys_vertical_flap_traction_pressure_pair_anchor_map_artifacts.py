from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_pressure_pair_anchor_map_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_pressure_pair_anchor_map_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_pressure_pair_anchor_map_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_pressure_pair_anchor_map_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_pressure_pair_anchor_map_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_pressure_pair_anchor_map_matrix.py"
)
EXPECTED_SHARED_SHA = (
    "3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968"
)
EXPECTED_BASELINE_SCENARIO = "baseline_independent_probe0p51"
EXPECTED_ANCHORED_SCENARIOS = {
    "anchored_from_baseline_probe0p00",
    "anchored_from_baseline_probe0p25",
    "anchored_from_baseline_probe0p375",
    "anchored_from_baseline_probe0p51",
    "anchored_from_baseline_probe0p625",
    "anchored_from_baseline_probe0p75",
    "anchored_from_baseline_probe1p00",
    "anchored_from_baseline_probe1p50",
}
EXPECTED_SCENARIOS = {EXPECTED_BASELINE_SCENARIO} | EXPECTED_ANCHORED_SCENARIOS
ANCHOR_REQUIRED_FIELDS = {
    "pressure_pair_anchor_active",
    "pressure_pair_anchor_inside_cell",
    "pressure_pair_anchor_outside_cell",
    "pressure_pair_anchor_source",
    "pressure_pair_anchor_fallback_used",
}
PAIR_REQUIRED_FIELDS = {
    "pressure_pair_policy",
    "pressure_pair_selected",
    "pressure_pair_fallback_used",
    "pressure_pair_inside_cell",
    "pressure_pair_outside_cell",
    "pressure_pair_cell_delta",
    "pressure_pair_symmetry_residual_cells",
}
SCOPE_REQUIRED_FRAGMENTS = (
    "shared snapshot",
    "sampling-only",
    "does not claim Fluent parity",
)


class AnsysVerticalFlapPressurePairAnchorMapArtifactTests(unittest.TestCase):
    def test_anchor_map_matrix_is_shared_snapshot_sampling_only(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(payload["baseline_scenario"], EXPECTED_BASELINE_SCENARIO)
        self.assertEqual(set(payload["anchored_scenarios"]), EXPECTED_ANCHORED_SCENARIOS)
        self.assertEqual(
            payload["purpose"],
            "shared_flow_snapshot_pressure_pair_anchor_map_matrix",
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
        self.assertEqual(payload["flow_snapshot_source_commit"], manifest["source_commit"])
        self.assertTrue(SHARED_FIELDS.exists())
        self.assertEqual(_sha256_file(SHARED_FIELDS), EXPECTED_SHARED_SHA)
        self.assertEqual(payload["completed_formulation_count"], len(EXPECTED_SCENARIOS))
        self.assertEqual(payload["scenario_count"], len(EXPECTED_SCENARIOS))
        self.assertEqual(
            payload["candidate_status"],
            "pressure_pair_anchor_map_stable_candidate_found",
        )
        self.assertEqual(
            payload["stable_pressure_pair_policy"],
            "baseline_anchored_cell_pair",
        )
        self.assertIsNone(payload["reference_formulation_candidate"])
        _assert_scope(self, payload["scope_limit"])

        blockers = {
            item["blocker"]: item.get("detail", "")
            for item in payload["candidate_blockers"]
        }
        for blocker in (
            "reference_selection_deferred",
            "dual_face_one_sided_unsupported",
            "sampling_only_no_coupled_fsi",
            "no_fluent_parity_claim",
        ):
            self.assertIn(blocker, blockers)
            self.assertNotEqual(blockers[blocker].strip(), "")

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["formulation_status"], "completed")
            self.assertEqual(row["worker_mode"], "shared_snapshot_pressure_pair_anchor_map")
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["flow_snapshot_sha256"], EXPECTED_SHARED_SHA)
            self.assertEqual(row["flow_snapshot_source_commit"], manifest["source_commit"])
            self.assertEqual(int(row["flow_snapshot_preflow_steps"]), 20)
            self.assertAlmostEqual(float(row["marker_face_offset_cells"]), 0.51)
            self.assertEqual(row["pressure_probe_origin_mode"], "physical_face_offset")
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            _assert_scope(self, row["scope_limit"])

        self.assertEqual(
            by_scenario[EXPECTED_BASELINE_SCENARIO]["pressure_pair_policy"],
            "independent_ladder",
        )
        for scenario in EXPECTED_ANCHORED_SCENARIOS:
            self.assertEqual(
                by_scenario[scenario]["pressure_pair_policy"],
                "baseline_anchored_cell_pair",
            )
            self.assertEqual(
                by_scenario[scenario]["anchor_source_scenario"],
                EXPECTED_BASELINE_SCENARIO,
            )
            self.assertNotEqual(by_scenario[scenario]["anchor_map_sha256"], "")
            self.assertEqual(
                int(
                    by_scenario[scenario][
                        "pressure_pair_anchor_selected_marker_count"
                    ]
                ),
                24,
            )
            self.assertEqual(
                int(
                    by_scenario[scenario][
                        "pressure_pair_anchor_fallback_marker_count"
                    ]
                ),
                0,
            )
        anchored_ratios = {
            round(float(by_scenario[scenario]["force_ratio_to_anchor_baseline"]), 12)
            for scenario in EXPECTED_ANCHORED_SCENARIOS
        }
        self.assertEqual(len(anchored_ratios), 1)

    def test_anchor_map_candidate_gate_is_artifact_backed(self):
        payload = _read_json(MATRIX_JSON)
        acceptance = payload["anchor_map_acceptance"]
        gate = payload["stable_candidate_gate"]

        self.assertEqual(acceptance["row_count"], len(EXPECTED_ANCHORED_SCENARIOS))
        self.assertEqual(acceptance["expected_row_count"], len(EXPECTED_ANCHORED_SCENARIOS))
        self.assertIsInstance(acceptance["accepted"], bool)
        self.assertIsInstance(acceptance["pressure_complete"], bool)
        self.assertIsInstance(acceptance["invalid_marker_counts_zero"], bool)
        self.assertIsInstance(acceptance["anchor_selected_all_markers"], bool)
        self.assertIsInstance(acceptance["anchor_fallback_zero"], bool)
        self.assertIsInstance(acceptance["scope_sampling_only"], bool)

        self.assertEqual(
            payload["candidate_status"],
            "pressure_pair_anchor_map_stable_candidate_found",
        )
        self.assertEqual(
            payload["stable_pressure_pair_policy"],
            "baseline_anchored_cell_pair",
        )
        self.assertTrue(acceptance["accepted"])
        self.assertEqual(float(acceptance["force_ratio_relative_span"]), 0.0)
        self.assertLessEqual(
            float(acceptance["force_ratio_relative_span"]),
            float(gate["force_ratio_relative_span_max"]),
        )
        self.assertLessEqual(
            float(acceptance["max_face_traction_decomposition_residual_pa"]),
            float(gate["traction_decomposition_residual_max"]),
        )
        self.assertTrue(acceptance["anchor_selected_all_markers"])
        self.assertTrue(acceptance["anchor_fallback_zero"])
        self.assertTrue(acceptance["scope_sampling_only"])

    def test_marker_diagnostics_include_anchor_fields(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        self.assertEqual(set(history["histories"]), EXPECTED_SCENARIOS)

        for row in payload["rows"]:
            marker_payload = _read_json(Path(row["marker_diagnostics_json"]))
            self.assertEqual(marker_payload["scenario"], row["scenario"])
            self.assertEqual(marker_payload["pressure_pair_policy"], row["pressure_pair_policy"])
            self.assertEqual(
                marker_payload["flow_snapshot_sha256"],
                row["flow_snapshot_sha256"],
            )
            self.assertEqual(
                marker_payload["marker_geometry_sha256"],
                row["marker_geometry_sha256"],
            )
            self.assertTrue(
                ANCHOR_REQUIRED_FIELDS.issubset(marker_payload["marker_required_fields"])
            )
            self.assertTrue(
                PAIR_REQUIRED_FIELDS.issubset(marker_payload["marker_required_fields"])
            )
            self.assertGreater(marker_payload["marker_count"], 0)
            self.assertEqual(marker_payload["marker_count"], len(marker_payload["markers"]))
            self.assertEqual(
                marker_payload["anchor_stats"]["pressure_pair_anchor_selected_marker_count"],
                row["pressure_pair_anchor_selected_marker_count"],
            )
            self.assertEqual(
                marker_payload["anchor_stats"]["pressure_pair_anchor_fallback_marker_count"],
                row["pressure_pair_anchor_fallback_marker_count"],
            )
            for marker in marker_payload["markers"]:
                self.assertTrue(ANCHOR_REQUIRED_FIELDS.issubset(marker))
                self.assertTrue(PAIR_REQUIRED_FIELDS.issubset(marker))
                self.assertEqual(marker["pressure_pair_policy"], row["pressure_pair_policy"])
                self.assertEqual(len(marker["pressure_pair_anchor_inside_cell"]), 3)
                self.assertEqual(len(marker["pressure_pair_anchor_outside_cell"]), 3)
                if row["pressure_pair_policy"] == "baseline_anchored_cell_pair":
                    self.assertTrue(marker["pressure_pair_anchor_active"])
                    self.assertEqual(marker["pressure_pair_anchor_source"], "api")
                    self.assertFalse(marker["pressure_pair_anchor_fallback_used"])
                    self.assertEqual(
                        marker["pressure_pair_inside_cell"],
                        marker["pressure_pair_anchor_inside_cell"],
                    )
                    self.assertEqual(
                        marker["pressure_pair_outside_cell"],
                        marker["pressure_pair_anchor_outside_cell"],
                    )

    def test_summary_and_checksums_match_anchor_map_artifacts(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("reuses one archived shared preflow snapshot", summary)
        self.assertIn("marker traction sampling", summary)
        self.assertIn("does not advance", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn("reference_formulation_candidate", summary)
        self.assertIn("stable_pressure_pair_policy", summary)
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
