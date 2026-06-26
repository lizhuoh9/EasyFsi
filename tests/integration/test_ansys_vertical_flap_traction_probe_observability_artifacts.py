from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
DIAG_ROOT = ROOT / "traction_probe_observability_diagnostics"
MATRIX_JSON = DIAG_ROOT / "traction_probe_observability_matrix.json"
MATRIX_CSV = DIAG_ROOT / "traction_probe_observability_matrix.csv"
HISTORY_JSON = DIAG_ROOT / "traction_probe_observability_history.json"
SUMMARY_MD = DIAG_ROOT / "traction_probe_observability_summary.md"
CHECKSUMS = DIAG_ROOT / "CHECKSUMS.sha256"

SUPPORTED_SCENARIOS = {
    "dual_two_sided_offset0p25_pressure_only",
    "dual_two_sided_offset0p51_pressure_only",
    "dual_two_sided_offset1p00_pressure_only",
    "single_mid_two_sided_offset0p00_pressure_only",
    "dual_two_sided_offset0p51_viscous_air",
}
UNSUPPORTED_SCENARIO = "dual_one_sided_offset0p51_pressure_only"

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


class AnsysVerticalFlapTractionProbeObservabilityArtifactTests(unittest.TestCase):
    def test_observability_matrix_archives_real_probe_evidence(self):
        payload = _read_json(MATRIX_JSON)
        rows = payload["rows"]
        by_name = {row["scenario"]: row for row in rows}

        self.assertEqual(
            payload["purpose"],
            "fixed-solid traction probe observability diagnostics",
        )
        self.assertEqual(payload["preflow_steps"], 20)
        self.assertEqual(payload["reference_formulation_candidate"], "none")
        self.assertEqual(
            payload["candidate_status"],
            "no_reference_formulation_candidate",
        )
        self.assertIn(
            "dual_face_one_sided_unsupported",
            payload["candidate_blockers"],
        )
        self.assertIn(
            "dual_two_sided_offset_sensitivity_above_tolerance",
            payload["candidate_blockers"],
        )
        self.assertNotIn(
            "pressure_probe_diagnostics_incomplete",
            payload["candidate_blockers"],
        )
        self.assertNotIn(
            "probe_rung_diagnostics_incomplete",
            payload["candidate_blockers"],
        )
        self.assertNotIn(
            "probe_cell_diagnostics_incomplete",
            payload["candidate_blockers"],
        )
        self.assertNotIn(
            "traction_decomposition_missing",
            payload["candidate_blockers"],
        )

        self.assertTrue(SUPPORTED_SCENARIOS.issubset(by_name))
        for scenario in SUPPORTED_SCENARIOS:
            row = by_name[scenario]
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["worker_mode"], "isolated_subprocess")
            self.assertEqual(int(row["worker_returncode"]), 0)
            self.assertFalse(_truthy(row["worker_timed_out"]))
            self.assertEqual(int(row["preflow_steps"]), 20)
            self.assertFalse(_truthy(row["solid_advanced"]))
            self.assertFalse(_truthy(row["feedback_applied"]))
            self.assertEqual(row["flow_driver_mode"], "sustained_volume_source_inlet")
            self.assertNotEqual(row["marker_diagnostics_json"], "")
            self.assert_marker_json_is_complete(Path(row["marker_diagnostics_json"]))
            self.assert_probe_fields_are_complete(row)

        unsupported = by_name[UNSUPPORTED_SCENARIO]
        self.assertEqual(unsupported["run_status"], "unsupported")
        self.assertEqual(unsupported["worker_mode"], "not_run")
        self.assertEqual(unsupported.get("marker_diagnostics_json", ""), "")

        self.assertGreater(
            float(by_name["dual_two_sided_offset0p25_pressure_only"]["force_ratio_to_baseline"]),
            1.5,
        )
        self.assertLess(
            float(by_name["dual_two_sided_offset1p00_pressure_only"]["force_ratio_to_baseline"]),
            0.2,
        )

        histories = _read_json(HISTORY_JSON)
        for scenario in SUPPORTED_SCENARIOS:
            self.assertEqual(len(histories[scenario]), 20)

        csv_rows = _read_csv(MATRIX_CSV)
        self.assertEqual(len(csv_rows), len(rows))
        self.assertTrue(CHECKSUMS.exists())
        self.assertIn("marker_diagnostics/", CHECKSUMS.read_text(encoding="utf-8"))

    def test_observability_summary_explains_offset_pathology(self):
        summary = SUMMARY_MD.read_text(encoding="utf-8")

        self.assertIn("fixed-solid traction probe observability only", summary)
        self.assertIn("offset0p25 mechanism", summary)
        self.assertIn("duplicates pressure jump", summary)
        self.assertIn("offset0p51 mechanism", summary)
        self.assertIn("secondary face near zero", summary)
        self.assertIn("offset1p00 mechanism", summary)
        self.assertIn("loses the thin-wall pressure jump", summary)
        self.assertIn("does not claim Fluent parity", summary)

    def assert_marker_json_is_complete(self, path: Path) -> None:
        self.assertTrue(path.exists(), path)
        payload = _read_json(path)

        self.assertEqual(payload["preflow_step"], 20)
        self.assertGreater(payload["marker_count"], 0)
        self.assertEqual(payload["marker_count"], len(payload["markers"]))
        for marker in payload["markers"]:
            self.assertTrue(MARKER_REQUIRED_FIELDS.issubset(marker))
            self.assertIsInstance(marker["invalid_reason_code"], int)
            self.assertEqual(len(marker["position_m"]), 3)
            self.assertEqual(len(marker["normal"]), 3)
            self.assertEqual(len(marker["inside_probe_grid_coordinate"]), 3)
            self.assertEqual(len(marker["outside_probe_grid_coordinate"]), 3)
            self.assertEqual(len(marker["inside_probe_nearest_cell"]), 3)
            self.assertEqual(len(marker["outside_probe_nearest_cell"]), 3)
            self.assertEqual(len(marker["pressure_traction_pa"]), 3)
            self.assertEqual(len(marker["viscous_traction_pa"]), 3)
            self.assertEqual(len(marker["total_traction_pa"]), 3)
            self.assertLessEqual(
                abs(float(marker["traction_decomposition_residual_pa"])),
                1.0e-8,
            )

    def assert_probe_fields_are_complete(self, row: dict[str, object]) -> None:
        required_faces = ["primary"]
        if row["marker_layout"] == "dual_physical_faces":
            required_faces.append("secondary")

        for face in required_faces:
            valid_count = int(row[f"{face}_face_valid_marker_count"])
            self.assertGreater(valid_count, 0)
            self.assertEqual(
                int(row[f"{face}_face_pressure_complete_marker_count"]),
                valid_count,
            )
            self.assertEqual(
                int(row[f"{face}_face_pressure_missing_marker_count"]),
                0,
            )
            self.assertEqual(
                int(row[f"{face}_face_inside_pressure_found_marker_count"]),
                valid_count,
            )
            self.assertEqual(
                int(row[f"{face}_face_outside_pressure_found_marker_count"]),
                valid_count,
            )
            self.assertNotEqual(row[f"{face}_face_inside_probe_rung_histogram"], "")
            self.assertNotEqual(row[f"{face}_face_outside_probe_rung_histogram"], "")
            self.assertGreater(
                int(row[f"{face}_face_inside_unique_nearest_cell_count"]),
                0,
            )
            self.assertGreater(
                int(row[f"{face}_face_outside_unique_nearest_cell_count"]),
                0,
            )
            self.assertLessEqual(
                abs(float(row[f"{face}_face_traction_decomposition_max_abs_residual_pa"])),
                1.0e-8,
            )
            self.assertEqual(
                int(row[f"{face}_face_traction_decomposition_invalid_marker_count"]),
                0,
            )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


if __name__ == "__main__":
    unittest.main()

