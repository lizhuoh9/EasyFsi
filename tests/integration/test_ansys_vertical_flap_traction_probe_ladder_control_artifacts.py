from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_probe_ladder_control_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_probe_ladder_control_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_probe_ladder_control_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_probe_ladder_control_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_probe_ladder_control_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_probe_ladder_control_matrix.py"
)
EXPECTED_STRATEGIES = {
    "current_control_baseline",
    "origin0p51_start1p00_spacing0p50",
    "origin0p51_start0p75_spacing0p50",
    "origin0p51_start0p625_spacing0p25",
    "origin0p51_start0p51_spacing0p25",
}
EXPECTED_PROBE_OFFSETS = {0.51, 0.625, 1.0}
EXPECTED_SCENARIOS = {
    f"{strategy}_{probe}"
    for strategy in EXPECTED_STRATEGIES
    for probe in ("probe0p51", "probe0p625", "probe1p00")
}
SCOPE_REQUIRED_FRAGMENTS = (
    "shared snapshot",
    "sampling-only",
    "does not claim Fluent parity",
)
MARKER_REQUIRED_FIELDS = {
    "marker_index",
    "region_id",
    "position_m",
    "pressure_probe_origin_m",
    "pressure_probe_origin_source",
    "pressure_probe_origin_explicit",
    "normal",
    "valid",
    "invalid_reason_code",
    "invalid_reason",
    "base_pressure_found",
    "inside_pressure_found",
    "outside_pressure_found",
    "base_pressure_pa",
    "inside_pressure_pa",
    "outside_pressure_pa",
    "pressure_jump_pa",
    "fluid_side_pressure_defined",
    "fluid_side_pressure_pa",
    "reference_pressure_pa",
    "inside_probe_ladder_mode",
    "outside_probe_ladder_mode",
    "inside_probe_rung",
    "outside_probe_rung",
    "inside_probe_multiplier",
    "outside_probe_multiplier",
    "inside_probe_distance_m",
    "outside_probe_distance_m",
    "inside_probe_grid_coordinate",
    "outside_probe_grid_coordinate",
    "inside_probe_nearest_cell",
    "outside_probe_nearest_cell",
    "inside_probe_fluid_weight",
    "outside_probe_fluid_weight",
    "pressure_traction_pa",
    "viscous_traction_pa",
    "total_traction_pa",
    "traction_decomposition_residual_pa",
}


class AnsysVerticalFlapTractionProbeLadderControlArtifactTests(unittest.TestCase):
    def test_ladder_control_matrix_is_shared_snapshot_sampling_only(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(set(payload["strategies"]), EXPECTED_STRATEGIES)
        self.assertEqual(
            {round(float(value), 6) for value in payload["probe_origin_offsets_cells"]},
            {round(value, 6) for value in EXPECTED_PROBE_OFFSETS},
        )
        self.assertEqual(
            payload["purpose"],
            "shared_flow_snapshot_traction_probe_ladder_control_matrix",
        )
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertNotIn("D:", payload["source_script"])
        self.assertEqual(
            payload["flow_snapshot_identity_status"],
            "shared_snapshot_sha256_identical_completed_rows",
        )
        self.assertEqual(payload["flow_snapshot_sha256"], manifest["field_sha256"])
        self.assertEqual(payload["flow_snapshot_source_commit"], manifest["source_commit"])
        self.assertTrue(SHARED_FIELDS.exists())
        self.assertEqual(_sha256_file(SHARED_FIELDS), manifest["field_sha256"])
        self.assertEqual(payload["completed_formulation_count"], len(EXPECTED_SCENARIOS))
        self.assertEqual(payload["scenario_count"], len(EXPECTED_SCENARIOS))
        self.assertIn(
            payload["candidate_status"],
            {
                "probe_ladder_control_no_stable_candidate",
                "probe_ladder_control_stable_candidate_found",
            },
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
            self.assertEqual(row["worker_mode"], "shared_snapshot_probe_ladder_control")
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["flow_snapshot_sha256"], manifest["field_sha256"])
            self.assertEqual(row["flow_snapshot_source_commit"], manifest["source_commit"])
            self.assertEqual(int(row["flow_snapshot_preflow_steps"]), 20)
            self.assertAlmostEqual(float(row["marker_face_offset_cells"]), 0.51)
            self.assertEqual(row["pressure_probe_origin_mode"], "physical_face_offset")
            self.assertEqual(row["pressure_probe_ladder_mode"], "current_normal_cell_ladder")
            self.assertIn(row["strategy"], EXPECTED_STRATEGIES)
            self.assertNotEqual(row["marker_geometry_sha256"], "")
            self.assertNotEqual(row["pressure_probe_origin_sha256"], "")
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            _assert_scope(self, row["scope_limit"])

    def test_strategy_gate_is_artifact_backed(self):
        payload = _read_json(MATRIX_JSON)
        summaries = payload["strategy_summaries"]
        details = payload["strategy_acceptance_details"]

        self.assertEqual(set(summaries), EXPECTED_STRATEGIES)
        self.assertEqual(set(details), EXPECTED_STRATEGIES)
        for strategy in EXPECTED_STRATEGIES:
            summary = summaries[strategy]
            detail = details[strategy]
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["completed_row_count"], 3)
            self.assertEqual(detail["row_count"], 3)
            self.assertEqual(detail["expected_row_count"], 3)
            self.assertIsInstance(detail["accepted"], bool)
            self.assertNotEqual(summary["force_ratio_span"]["relative_span"], "")
            self.assertIn("force_ratio_relative_span", detail)
            self.assertIn("max_face_traction_decomposition_residual_pa", detail)

        accepted = payload["accepted_strategies"]
        if payload["candidate_status"] == "probe_ladder_control_stable_candidate_found":
            self.assertIn(payload["stable_ladder_candidate"], accepted)
            accepted_detail = details[payload["stable_ladder_candidate"]]
            self.assertTrue(accepted_detail["accepted"])
            self.assertLessEqual(
                float(accepted_detail["force_ratio_relative_span"]),
                float(payload["stable_candidate_gate"]["force_ratio_relative_span_max"]),
            )
            self.assertLessEqual(
                float(accepted_detail["max_face_traction_decomposition_residual_pa"]),
                float(
                    payload["stable_candidate_gate"][
                        "traction_decomposition_residual_max"
                    ]
                ),
            )
        else:
            self.assertEqual(accepted, [])
            self.assertIsNone(payload["stable_ladder_candidate"])

    def test_marker_diagnostics_archive_ladder_control_fields(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        self.assertEqual(set(history["histories"]), EXPECTED_SCENARIOS)

        for row in payload["rows"]:
            marker_payload = _read_json(Path(row["marker_diagnostics_json"]))
            self.assertEqual(marker_payload["scenario"], row["scenario"])
            self.assertEqual(marker_payload["strategy"], row["strategy"])
            self.assertEqual(
                marker_payload["flow_snapshot_sha256"],
                row["flow_snapshot_sha256"],
            )
            self.assertEqual(
                marker_payload["marker_geometry_sha256"],
                row["marker_geometry_sha256"],
            )
            self.assertEqual(
                marker_payload["pressure_probe_origin_sha256"],
                row["pressure_probe_origin_sha256"],
            )
            self.assertEqual(
                set(marker_payload["marker_required_fields"]),
                MARKER_REQUIRED_FIELDS,
            )
            self.assertEqual(
                marker_payload["pressure_probe_ladder_mode"],
                "current_normal_cell_ladder",
            )
            self.assertGreater(marker_payload["marker_count"], 0)
            self.assertEqual(marker_payload["marker_count"], len(marker_payload["markers"]))
            for marker in marker_payload["markers"]:
                self.assertEqual(set(marker), MARKER_REQUIRED_FIELDS)
                self.assertEqual(marker["pressure_probe_origin_source"], "explicit")
                self.assertTrue(marker["pressure_probe_origin_explicit"])
                self.assertEqual(len(marker["pressure_probe_origin_m"]), 3)
                self.assertEqual(len(marker["inside_probe_nearest_cell"]), 3)
                self.assertEqual(len(marker["outside_probe_nearest_cell"]), 3)

    def test_summary_and_checksums_match_ladder_control_artifacts(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("reuses one archived shared preflow snapshot", summary)
        self.assertIn("marker traction sampling", summary)
        self.assertIn("does not advance", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn("reference_formulation_candidate", summary)
        self.assertIn("stable_ladder_candidate", summary)
        self.assertIn("origin0p51_symmetric_cell_pair_policy", summary)
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
