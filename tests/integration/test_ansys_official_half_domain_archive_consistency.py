import csv
import json
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = (
    REPO_ROOT / "validation" / "ansys-fluent-official-half-domain-hibm-mpm-2026-06-25"
)
DATA_ROOT = ARCHIVE_ROOT / "data"
PREFIX = "official_half_grid4x320x640_step1_p4096_s1000"
CASE_ID = "ansys-fluent-official-half-domain-single-flap"
MODELED_GRID = [4, 320, 640]
DISPLAY_GRID = [4, 640, 640]
MARKERS_PER_FACE = 84
MARKER_COUNT_ACTUAL = 2 * MARKERS_PER_FACE
PROJECTION_ITERATIONS = 4096


def _read_json(name: str) -> dict[str, object]:
    return json.loads((DATA_ROOT / name).read_text(encoding="utf-8"))


class AnsysOfficialHalfDomainArchiveConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = _read_json(f"{PREFIX}_manifest.json")
        cls.summary = _read_json(f"{PREFIX}_summary.json")
        cls.report = _read_json(f"{PREFIX}_report.json")
        cls.process = _read_json(f"{PREFIX}_process.json")
        cls.render = _read_json(f"{PREFIX}_fields_mirrored_velocity_pipe_style.json")
        with (DATA_ROOT / f"{PREFIX}_history.csv").open(newline="", encoding="utf-8") as handle:
            cls.history_rows = list(csv.DictReader(handle))
        cls.fields = np.load(DATA_ROOT / f"{PREFIX}_fields.npz", allow_pickle=True)

    def test_json_artifacts_use_one_official_half_domain_schema(self):
        payloads = [self.manifest, self.summary, self.report, self.process, self.render]
        for payload in payloads:
            self.assertEqual(payload["case"], CASE_ID)
            self.assertTrue(payload["official_half_domain"])
            self.assertFalse(payload["full_domain_two_flap"])
            self.assertEqual(payload["flap_count_modeled"], 1)
            self.assertEqual(
                payload["flap_count_displayed_after_symmetry_mirror"],
                2,
            )

    def test_grid_projection_and_marker_counts_are_consistent(self):
        for payload in [self.manifest, self.summary, self.report, self.process]:
            self.assertEqual(payload["modeled_grid_nodes"], MODELED_GRID)
            self.assertEqual(payload["display_grid_after_symmetry_mirror"], DISPLAY_GRID)
            self.assertEqual(payload["markers_per_face"], MARKERS_PER_FACE)
            self.assertEqual(payload["marker_count_actual"], MARKER_COUNT_ACTUAL)
            self.assertEqual(
                payload["flow_projection_iterations_actual"],
                PROJECTION_ITERATIONS,
            )

        self.assertEqual(self.report["config"]["grid_nodes"], MODELED_GRID)
        self.assertEqual(self.report["config"]["marker_count"], MARKER_COUNT_ACTUAL)
        self.assertEqual(
            self.report["config"]["flow_projection_iterations"],
            PROJECTION_ITERATIONS,
        )

    def test_history_and_field_snapshot_match_schema_counts(self):
        self.assertEqual(len(self.history_rows), 1)
        row = self.history_rows[0]
        self.assertEqual(int(row["stress_valid_marker_count"]), MARKER_COUNT_ACTUAL)
        self.assertEqual(int(row["stress_invalid_marker_count"]), 0)

        self.assertEqual(self.fields["pressure_pa"].shape, tuple(MODELED_GRID))
        self.assertEqual(self.fields["velocity_mps"].shape, tuple(MODELED_GRID + [3]))
        self.assertEqual(self.fields["obstacle"].shape, tuple(MODELED_GRID))
        self.assertEqual(self.fields["marker_position_m"].shape[0], MARKER_COUNT_ACTUAL)
        self.assertEqual(int(self.fields["marker_count_actual"]), MARKER_COUNT_ACTUAL)
        self.assertEqual(int(self.fields["markers_per_face"]), MARKERS_PER_FACE)
        self.assertEqual(
            int(self.fields["flow_projection_iterations_actual"]),
            PROJECTION_ITERATIONS,
        )
        self.assertEqual(self.fields["display_grid_after_symmetry_mirror"].tolist(), DISPLAY_GRID)

    def test_render_metadata_matches_modeled_and_mirrored_grids(self):
        self.assertEqual(self.render["modeled_grid_nodes"], MODELED_GRID)
        self.assertEqual(self.render["grid_nodes_modeled_half"], MODELED_GRID)
        self.assertEqual(self.render["display_grid_after_symmetry_mirror"], DISPLAY_GRID)
        self.assertEqual(self.render["marker_count_actual"], MARKER_COUNT_ACTUAL)
        self.assertAlmostEqual(self.render["inlet_velocity_mps"], 10.0)

    def test_gate_a_b_archive_artifact_quality_is_non_expected(self):
        history = self.report["history"][-1]
        self.assertEqual(history["stress_invalid_marker_count"], 0)
        self.assertEqual(self.summary["stress_invalid_marker_count"], 0)
        self.assertAlmostEqual(history["fixed_root_max_displacement_m"], 0.0)
        self.assertLess(history["marker_force_z_n"], 0.0)
        self.assertLess(history["lower_tip_mean_displacement_m"][2], 0.0)

        finite_values = [
            history["projection_l2"],
            history["projection_max_abs"],
            self.summary["fluid_speed_p99_mps"],
            self.summary["fluid_speed_p999_mps"],
            self.summary["max_displacement_m"],
            self.summary["marker_force_z_n"],
        ]
        for value in finite_values:
            self.assertTrue(np.isfinite(float(value)))

        self.assertGreaterEqual(self.summary["fluid_speed_p999_mps"], 20.0)
        self.assertLessEqual(self.summary["fluid_speed_p999_mps"], 29.0)


if __name__ == "__main__":
    unittest.main()
