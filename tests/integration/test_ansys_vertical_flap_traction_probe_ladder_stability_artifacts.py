from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_probe_ladder_stability_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_probe_ladder_stability_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_probe_ladder_stability_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_probe_ladder_stability_history.json"
TRANSITION_MAP_JSON = DIAG_ROOT / "traction_probe_ladder_transition_map.json"
SUMMARY_MD = DIAG_ROOT / "traction_probe_ladder_stability_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_probe_ladder_stability_matrix.py"
)
EXPECTED_SCENARIOS = {
    "probe_offset0p00",
    "probe_offset0p125",
    "probe_offset0p25",
    "probe_offset0p375",
    "probe_offset0p51",
    "probe_offset0p625",
    "probe_offset0p75",
    "probe_offset0p875",
    "probe_offset1p00",
    "probe_offset1p25",
    "probe_offset1p50",
}
EXPECTED_OFFSETS = {
    0.0,
    0.125,
    0.25,
    0.375,
    0.51,
    0.625,
    0.75,
    0.875,
    1.0,
    1.25,
    1.5,
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


class AnsysVerticalFlapTractionProbeLadderStabilityArtifactTests(
    unittest.TestCase,
):
    def test_ladder_stability_matrix_uses_one_shared_snapshot(self):
        for path in (
            MATRIX_JSON,
            MATRIX_CSV,
            HISTORY_JSON,
            TRANSITION_MAP_JSON,
            SUMMARY_MD,
            CHECKSUMS,
        ):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(
            payload["purpose"],
            "shared_flow_snapshot_traction_probe_ladder_stability_matrix",
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
        self.assertEqual(
            payload["candidate_status"],
            "probe_ladder_stability_diagnostic_only",
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
            "probe_ladder_stability_diagnostic_only",
            "sampling_only_no_coupled_fsi",
            "no_fluent_parity_claim",
        ):
            self.assertIn(blocker, blockers)
            self.assertNotEqual(blockers[blocker].strip(), "")

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["formulation_status"], "completed")
            self.assertEqual(row["worker_mode"], "shared_snapshot_probe_ladder_stability")
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["flow_snapshot_sha256"], manifest["field_sha256"])
            self.assertEqual(row["flow_snapshot_source_commit"], manifest["source_commit"])
            self.assertEqual(int(row["flow_snapshot_preflow_steps"]), 20)
            self.assertAlmostEqual(float(row["marker_face_offset_cells"]), 0.51)
            self.assertEqual(row["pressure_probe_origin_mode"], "physical_face_offset")
            self.assertNotEqual(row["marker_geometry_sha256"], "")
            self.assertNotEqual(row["pressure_probe_origin_sha256"], "")
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            _assert_scope(self, row["scope_limit"])

    def test_ladder_stability_preserves_current_probe_pathology(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertAlmostEqual(
            float(by_scenario["probe_offset0p51"]["force_ratio_to_baseline"]),
            1.0,
            delta=1.0e-12,
        )
        self.assertGreater(
            float(by_scenario["probe_offset0p25"]["force_ratio_to_baseline"]),
            1.5,
        )
        self.assertLess(
            float(by_scenario["probe_offset1p00"]["force_ratio_to_baseline"]),
            0.1,
        )
        self.assertGreater(
            float(payload["probe_origin_force_ratio_span"]["relative_span"]),
            20.0,
        )
        self.assertEqual(
            {row["marker_geometry_sha256"] for row in rows},
            {rows[0]["marker_geometry_sha256"]},
        )
        self.assertGreater(
            len({row["pressure_probe_origin_sha256"] for row in rows}),
            1,
        )

    def test_transition_map_covers_offsets_and_probe_classifications(self):
        payload = _read_json(MATRIX_JSON)
        transition_map = _read_json(TRANSITION_MAP_JSON)
        entries = transition_map["entries"]
        by_scenario = {row["scenario"]: row for row in payload["rows"]}

        self.assertEqual(transition_map["flow_snapshot_sha256"], payload["flow_snapshot_sha256"])
        self.assertEqual(transition_map["baseline_scenario"], "probe_offset0p51")
        self.assertEqual(
            {round(float(entry["offset_cells"]), 6) for entry in entries},
            {round(value, 6) for value in EXPECTED_OFFSETS},
        )
        self.assertEqual({entry["scenario"] for entry in entries}, EXPECTED_SCENARIOS)
        for entry in entries:
            row = by_scenario[entry["scenario"]]
            self.assertEqual(
                entry["force_ratio_to_baseline"],
                row["force_ratio_to_baseline"],
            )
            for field in (
                "primary_inside_nearest_cell_histogram",
                "primary_outside_nearest_cell_histogram",
                "secondary_inside_nearest_cell_histogram",
                "secondary_outside_nearest_cell_histogram",
                "primary_inside_rung_histogram",
                "primary_outside_rung_histogram",
                "secondary_inside_rung_histogram",
                "secondary_outside_rung_histogram",
            ):
                self.assertIsInstance(entry[field], dict)
            self.assertGreaterEqual(
                int(entry["primary_pressure_complete_marker_count"]),
                0,
            )
            self.assertGreaterEqual(
                int(entry["secondary_pressure_complete_marker_count"]),
                0,
            )
        summary = transition_map["transition_summary"]
        self.assertEqual(summary["force_amplification_threshold"], 1.5)
        self.assertEqual(summary["force_collapse_threshold"], 0.1)
        self.assertNotEqual(
            summary["first_force_amplification_offset_cells"],
            "",
        )
        self.assertNotEqual(summary["first_force_collapse_offset_cells"], "")
        self.assertIn(
            summary["collapse_0p51_to_1p00_has_probe_classification_change"],
            (True, False),
        )

    def test_marker_diagnostics_archive_probe_ladder_fields(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        self.assertEqual(set(history["histories"]), EXPECTED_SCENARIOS)

        for row in payload["rows"]:
            marker_payload = _read_json(Path(row["marker_diagnostics_json"]))
            self.assertEqual(marker_payload["scenario"], row["scenario"])
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
            self.assertGreater(marker_payload["marker_count"], 0)
            self.assertEqual(marker_payload["marker_count"], len(marker_payload["markers"]))
            for marker in marker_payload["markers"]:
                self.assertEqual(set(marker), MARKER_REQUIRED_FIELDS)
                self.assertEqual(marker["pressure_probe_origin_source"], "explicit")
                self.assertTrue(marker["pressure_probe_origin_explicit"])
                self.assertEqual(len(marker["pressure_probe_origin_m"]), 3)
                self.assertEqual(len(marker["inside_probe_nearest_cell"]), 3)
                self.assertEqual(len(marker["outside_probe_nearest_cell"]), 3)

    def test_summary_and_checksums_match_ladder_artifacts(self):
        payload = _read_json(MATRIX_JSON)
        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("reuses one archived shared preflow snapshot", summary)
        self.assertIn("marker traction sampling", summary)
        self.assertIn("does not advance", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn("reference_formulation_candidate", summary)
        self.assertIn("none", summary)
        self.assertIn("nearest-cell", summary)
        self.assertIn("rung", summary)
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
            TRANSITION_MAP_JSON.name,
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
