from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_probe_offset_decoupling_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_probe_offset_decoupling_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_probe_offset_decoupling_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_probe_offset_decoupling_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_probe_offset_decoupling_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_probe_offset_decoupling_matrix.py"
)
EXPECTED_SCENARIOS = {
    "fixed_marker0p51_probe0p00",
    "fixed_marker0p51_probe0p25",
    "fixed_marker0p51_probe0p51",
    "fixed_marker0p51_probe1p00",
    "fixed_probe0p51_marker0p00",
    "fixed_probe0p51_marker0p25",
    "fixed_probe0p51_marker0p51",
    "fixed_probe0p51_marker1p00",
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


class AnsysVerticalFlapTractionProbeOffsetDecouplingArtifactTests(
    unittest.TestCase,
):
    def test_probe_offset_decoupling_matrix_uses_one_shared_snapshot(self):
        for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), EXPECTED_SCENARIOS)
        self.assertEqual(
            payload["purpose"],
            "shared_flow_snapshot_traction_probe_offset_decoupling_matrix",
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
            "probe_offset_decoupling_diagnostic_only",
        )
        self.assertIsNone(payload["reference_formulation_candidate"])
        _assert_scope(self, payload["scope_limit"])

        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["formulation_status"], "completed")
            self.assertEqual(row["worker_mode"], "shared_snapshot_probe_offset_decoupling")
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["flow_snapshot_sha256"], manifest["field_sha256"])
            self.assertEqual(row["flow_snapshot_source_commit"], manifest["source_commit"])
            self.assertEqual(int(row["flow_snapshot_preflow_steps"]), 20)
            self.assertNotEqual(row["marker_geometry_sha256"], "")
            self.assertNotEqual(row["pressure_probe_origin_sha256"], "")
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())
            _assert_scope(self, row["scope_limit"])

    def test_fixed_marker_and_fixed_probe_hashes_decouple(self):
        payload = _read_json(MATRIX_JSON)
        fixed_marker = [
            row for row in payload["rows"] if row["scenario_group"] == "fixed_marker"
        ]
        fixed_probe = [
            row for row in payload["rows"] if row["scenario_group"] == "fixed_probe"
        ]
        by_scenario = {row["scenario"]: row for row in payload["rows"]}

        self.assertEqual(len(fixed_marker), 4)
        self.assertEqual(len(fixed_probe), 4)
        self.assertEqual({row["marker_geometry_sha256"] for row in fixed_marker}, {
            fixed_marker[0]["marker_geometry_sha256"],
        })
        self.assertGreater(
            len({row["pressure_probe_origin_sha256"] for row in fixed_marker}),
            1,
        )
        self.assertEqual({row["pressure_probe_origin_sha256"] for row in fixed_probe}, {
            fixed_probe[0]["pressure_probe_origin_sha256"],
        })
        self.assertGreater(
            len({row["marker_geometry_sha256"] for row in fixed_probe}),
            1,
        )
        self.assertGreater(
            float(payload["fixed_marker_probe_origin_ratio_span"]["relative_span"]),
            20.0,
        )
        self.assertLessEqual(
            abs(float(payload["fixed_probe_marker_ratio_span"]["relative_span"])),
            1.0e-12,
        )
        for row in fixed_probe:
            self.assertAlmostEqual(
                float(row["force_ratio_to_group_baseline"]),
                1.0,
                delta=1.0e-12,
                msg=row["scenario"],
            )
        self.assertGreater(
            float(
                by_scenario["fixed_marker0p51_probe0p25"][
                    "force_ratio_to_group_baseline"
                ]
            ),
            1.5,
        )
        self.assertLess(
            float(
                by_scenario["fixed_marker0p51_probe1p00"][
                    "force_ratio_to_group_baseline"
                ]
            ),
            0.1,
        )

    def test_marker_diagnostics_archive_probe_origin_fields(self):
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

    def test_summary_and_blockers_stay_diagnostic_only(self):
        payload = _read_json(MATRIX_JSON)
        blockers = {
            item["blocker"]: item.get("detail", "")
            for item in payload["candidate_blockers"]
        }
        for blocker in (
            "reference_selection_deferred",
            "dual_face_one_sided_unsupported",
            "probe_offset_decoupling_diagnostic_only",
            "sampling_only_no_coupled_fsi",
        ):
            self.assertIn(blocker, blockers)
            self.assertNotEqual(blockers[blocker].strip(), "")

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("reuses one archived shared preflow snapshot", summary)
        self.assertIn("does not claim Fluent parity", summary)
        self.assertIn("does not advance", summary)
        self.assertIn("reference_formulation_candidate", summary)
        self.assertIn("none", summary)
        for scenario in EXPECTED_SCENARIOS:
            self.assertIn(scenario, summary)

    def test_csv_and_checksums_match_json_artifacts(self):
        payload = _read_json(MATRIX_JSON)
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
