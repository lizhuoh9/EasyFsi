from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validation_runs.ansys_vertical_flap_fsi.scripts import (
    run_traction_snapshot_resampling_matrix as resampling_runner,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
DIAG_ROOT = ROOT / "traction_snapshot_resampling_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_snapshot_resampling_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_snapshot_resampling_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_snapshot_resampling_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_snapshot_resampling_summary.md"
VERIFICATION_MD = DIAG_ROOT / "verification_snapshot_resampling_2026-06-26.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"
SHARED_MANIFEST = SHARED_ROOT / "snapshot_manifest.json"
SHARED_FIELDS = SHARED_ROOT / "step020_fields.npz"

SUPPORTED_SCENARIOS = {
    "dual_two_sided_offset0p51_pressure_only",
    "single_mid_two_sided_offset0p00_pressure_only",
    "dual_two_sided_offset0p25_pressure_only",
    "dual_two_sided_offset1p00_pressure_only",
    "dual_two_sided_offset0p51_viscous_air",
}

UNSUPPORTED_SCENARIO = "dual_one_sided_offset0p51_pressure_only"
EXPECTED_SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_snapshot_resampling_matrix.py"
)
SCOPE_REQUIRED_FRAGMENTS = (
    "shared snapshot",
    "sampling-only",
    "does not claim Fluent parity",
)
OLD_SCOPE_FRAGMENT = "fixed-solid traction formulation diagnostic only"
CORE_CANDIDATE_BLOCKERS = {
    "required_formulation_unsupported",
    "dual_face_one_sided_unsupported",
    "dual_two_sided_offset_sensitivity_above_tolerance",
}

MARKER_REQUIRED_FIELDS = {
    "marker_index",
    "region_id",
    "position_m",
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


class AnsysVerticalFlapTractionSnapshotResamplingArtifactTests(unittest.TestCase):
    def test_resampling_matrix_is_tied_to_one_shared_snapshot(self):
        for path in (
            MATRIX_JSON,
            MATRIX_CSV,
            HISTORY_JSON,
            SUMMARY_MD,
            VERIFICATION_MD,
            CHECKSUMS,
        ):
            self.assertTrue(path.exists(), path)

        manifest = _read_json(SHARED_MANIFEST)
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(set(by_scenario), SUPPORTED_SCENARIOS | {UNSUPPORTED_SCENARIO})
        self.assertEqual(payload["purpose"], "shared_flow_snapshot_traction_resampling_matrix")
        self.assertEqual(
            payload["flow_snapshot_identity_status"],
            "shared_snapshot_sha256_identical_completed_rows",
        )
        self.assertEqual(payload["flow_snapshot_sha256"], manifest["field_sha256"])
        self.assertEqual(payload["flow_snapshot_source_commit"], manifest["source_commit"])
        self.assertTrue(SHARED_FIELDS.exists())
        self.assertEqual(_sha256_file(SHARED_FIELDS), manifest["field_sha256"])
        self.assertEqual(payload["completed_formulation_count"], len(SUPPORTED_SCENARIOS))
        self.assertEqual(payload["unsupported_formulation_count"], 1)
        _assert_resampling_scope(self, payload["scope_limit"])
        self.assertEqual(payload["source_script"], EXPECTED_SOURCE_SCRIPT)
        self.assertFalse(Path(payload["source_script"]).is_absolute())
        self.assertNotIn("\\", payload["source_script"])
        self.assertNotIn("D:", payload["source_script"])
        self.assertNotIn("Users", payload["source_script"])
        self.assertNotIn("lizhu", payload["source_script"])
        self.assertEqual(
            payload["candidate_status"],
            "snapshot_resampling_no_reference_selection",
        )
        self.assertIsNone(payload["reference_formulation_candidate"])

        for row in rows:
            _assert_resampling_scope(self, row["scope_limit"])

        for scenario in SUPPORTED_SCENARIOS:
            row = by_scenario[scenario]
            self.assertEqual(row["formulation_status"], "completed")
            self.assertEqual(row["worker_mode"], "shared_snapshot_resampling")
            self.assertFalse(bool(row["solid_advanced"]))
            self.assertFalse(bool(row["feedback_applied"]))
            self.assertEqual(row["flow_snapshot_sha256"], manifest["field_sha256"])
            self.assertEqual(row["flow_snapshot_source_commit"], manifest["source_commit"])
            self.assertEqual(int(row["flow_snapshot_preflow_steps"]), 20)
            self.assertEqual(row["flow_snapshot_path"], SHARED_FIELDS.as_posix())
            self.assertGreater(float(row["total_marker_count"]), 0.0)
            self.assertEqual(float(row["primary_face_invalid_marker_count"]), 0.0)
            if row["marker_layout"] == "dual_physical_faces":
                self.assertEqual(float(row["secondary_face_invalid_marker_count"]), 0.0)
            self.assertLessEqual(float(row["marker_action_reaction_residual_N"]), 1.0e-8)
            self.assertNotEqual(row["marker_geometry_sha256"], "")
            self.assertTrue(Path(row["marker_diagnostics_json"]).exists())

        unsupported = by_scenario[UNSUPPORTED_SCENARIO]
        self.assertEqual(unsupported["formulation_status"], "unsupported")
        self.assertEqual(unsupported["worker_mode"], "not_run")
        self.assertIn("one-sided pressure", unsupported["status_reason"])
        self.assertEqual(unsupported["marker_diagnostics_json"], "")

    def test_resampling_preserves_offset_sensitivity_and_fail_closed_blockers(self):
        payload = _read_json(MATRIX_JSON)
        by_scenario = {row["scenario"]: row for row in payload["rows"]}

        low_offset = by_scenario["dual_two_sided_offset0p25_pressure_only"]
        high_offset = by_scenario["dual_two_sided_offset1p00_pressure_only"]
        blockers = {
            item["blocker"]: item.get("detail", "")
            for item in payload["candidate_blockers"]
        }

        self.assertGreater(float(low_offset["force_ratio_to_baseline"]), 1.5)
        self.assertLess(float(high_offset["force_ratio_to_baseline"]), 0.2)
        self.assertIn("dual_face_one_sided_unsupported", blockers)
        self.assertIn("formulation_resampling_only", blockers)
        self.assertIn("required_formulation_unsupported", blockers)
        self.assertIn("reference_selection_deferred", blockers)
        for blocker, detail in blockers.items():
            self.assertNotEqual(detail.strip(), "", blocker)
        for blocker in CORE_CANDIDATE_BLOCKERS:
            self.assertNotEqual(blockers[blocker].strip(), "", blocker)

        summary = SUMMARY_MD.read_text(encoding="utf-8")
        self.assertIn("reuses one archived shared preflow", summary)
        self.assertIn("force differences come from sampling formulation", summary)
        self.assertIn("dual-face one-sided pressure scenario remains fail-closed", summary)
        self.assertIn("reference_formulation_candidate", summary)
        self.assertIn("none", summary)
        self.assertIn("candidate_blockers", summary)
        for scenario in SUPPORTED_SCENARIOS:
            self.assertIn(scenario, summary)
        self.assertIn(UNSUPPORTED_SCENARIO, summary)
        self.assertIn("Fluent parity", summary)
        self.assertIn("coupled 50-step FSI", summary)
        self.assertIn("split marker offset from pressure-probe", summary)
        verification = VERIFICATION_MD.read_text(encoding="utf-8")
        self.assertIn("checks the archived NPZ checksum", verification)

    def test_marker_diagnostics_are_archived_for_completed_rows(self):
        payload = _read_json(MATRIX_JSON)
        history = _read_json(HISTORY_JSON)
        self.assertEqual(set(history["histories"]), SUPPORTED_SCENARIOS)

        for row in payload["rows"]:
            if row["formulation_status"] != "completed":
                continue
            marker_path = Path(row["marker_diagnostics_json"])
            marker_payload = _read_json(marker_path)
            self.assertEqual(marker_payload["scenario"], row["scenario"])
            self.assertEqual(marker_payload["flow_snapshot_sha256"], row["flow_snapshot_sha256"])
            self.assertEqual(
                marker_payload["marker_geometry_sha256"],
                row["marker_geometry_sha256"],
            )
            self.assertEqual(
                set(marker_payload["marker_required_fields"]),
                MARKER_REQUIRED_FIELDS,
            )
            self.assertGreater(marker_payload["marker_count"], 0)
            self.assertEqual(marker_payload["marker_count"], len(marker_payload["markers"]))
            for marker in marker_payload["markers"]:
                self.assertEqual(set(marker), MARKER_REQUIRED_FIELDS)

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
            VERIFICATION_MD.name,
        ):
            self.assertIn(artifact, checksum_rows)
            self.assertEqual(checksum_rows[artifact], _sha256_file(DIAG_ROOT / artifact))
        for row in payload["rows"]:
            if row["formulation_status"] == "completed":
                marker_rel = Path(row["marker_diagnostics_json"]).relative_to(DIAG_ROOT)
                self.assertIn(marker_rel.as_posix(), checksum_rows)

    def test_snapshot_field_shape_validator_rejects_mismatches_without_taichi(self):
        manifest = {"grid_nodes": [2, 3, 4]}
        config = SimpleNamespace(grid_nodes=(2, 3, 4))
        fields = _synthetic_snapshot_fields(2, 3, 4)

        resampling_runner._validate_snapshot_fields(fields, manifest, config)

        bad_pressure = dict(fields)
        bad_pressure["pressure"] = np.zeros((2, 3, 5), dtype=np.float64)
        with self.assertRaisesRegex(
            resampling_runner.SnapshotResamplingError,
            "pressure",
        ):
            resampling_runner._validate_snapshot_fields(
                bad_pressure,
                manifest,
                config,
            )

        with self.assertRaisesRegex(
            resampling_runner.SnapshotResamplingError,
            "grid_nodes",
        ):
            resampling_runner._validate_snapshot_fields(
                fields,
                {"grid_nodes": [2, 3, 5]},
                config,
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


def _assert_resampling_scope(
    testcase: unittest.TestCase,
    scope: str,
) -> None:
    for fragment in SCOPE_REQUIRED_FRAGMENTS:
        testcase.assertIn(fragment, scope)
    testcase.assertNotIn(OLD_SCOPE_FRAGMENT, scope)


def _synthetic_snapshot_fields(nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    return {
        "velocity": np.zeros((nx, ny, nz, 3), dtype=np.float64),
        "pressure": np.zeros((nx, ny, nz), dtype=np.float64),
        "obstacle": np.zeros((nx, ny, nz), dtype=np.int32),
        "cell_face_x_m": np.zeros((nx + 1,), dtype=np.float64),
        "cell_face_y_m": np.zeros((ny + 1,), dtype=np.float64),
        "cell_face_z_m": np.zeros((nz + 1,), dtype=np.float64),
        "cell_center_x_m": np.zeros((nx,), dtype=np.float64),
        "cell_center_y_m": np.zeros((ny,), dtype=np.float64),
        "cell_center_z_m": np.zeros((nz,), dtype=np.float64),
        "cell_width_x_m": np.ones((nx,), dtype=np.float64),
        "cell_width_y_m": np.ones((ny,), dtype=np.float64),
        "cell_width_z_m": np.ones((nz,), dtype=np.float64),
    }
