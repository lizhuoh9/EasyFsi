from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (
    compare_structure_monitor,
)


class CompareStructureMonitorTemporalOverlapTests(unittest.TestCase):
    def test_one_solver_point_vs_fifty_fluent_points_does_not_pass(self) -> None:
        # Audit probe: `count = min(len(solver), len(fluent))` used to
        # truncate BOTH series to the shorter one, so comparing 1 solver
        # sample against 50 Fluent samples silently discarded 49 reference
        # points and compared only the first row of each series -- which can
        # trivially "pass" no matter what the other 49 Fluent points show.
        solver_rows = [{"step": 1, "time_s": 0.0, "tip_disp": 0.0}]
        fluent_rows = [
            {
                "step": step,
                "time_s": step * 5.0e-4,
                "monitor_avg_total_col0_col6_m": 0.0,
            }
            for step in range(1, 51)
        ]

        result = compare_structure_monitor(
            solver_rows,
            fluent_rows,
            solver_displacement_key="tip_disp",
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("insufficient temporal overlap", result["reason"])
        self.assertIn("1", result["reason"])
        self.assertIn("50", result["reason"])
        self.assertEqual(result["solver_sample_count"], 1)
        self.assertEqual(result["fluent_sample_count"], 50)
        self.assertEqual(result["metrics"], {})
        self.assertEqual(result["gates"], {})

    def test_matching_length_series_still_compare_normally(self) -> None:
        solver_rows = [
            {"step": step, "time_s": step * 5.0e-4, "tip_disp": 0.01 * step}
            for step in range(1, 51)
        ]
        fluent_rows = [
            {
                "step": step,
                "time_s": step * 5.0e-4,
                "monitor_avg_total_col0_col6_m": 0.01 * step,
            }
            for step in range(1, 51)
        ]

        result = compare_structure_monitor(
            solver_rows,
            fluent_rows,
            solver_displacement_key="tip_disp",
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["sample_count"], 50)
        self.assertNotIn("reason", result)

    def test_slightly_short_series_within_default_tolerance_still_compares(
        self,
    ) -> None:
        # 46 of 50 = 92% overlap, above the default 90% threshold, so the
        # comparison must still run (not be rejected outright).
        solver_rows = [
            {"step": step, "time_s": step * 5.0e-4, "tip_disp": 0.01 * step}
            for step in range(1, 47)
        ]
        fluent_rows = [
            {
                "step": step,
                "time_s": step * 5.0e-4,
                "monitor_avg_total_col0_col6_m": 0.01 * step,
            }
            for step in range(1, 51)
        ]

        result = compare_structure_monitor(
            solver_rows,
            fluent_rows,
            solver_displacement_key="tip_disp",
        )

        self.assertNotIn("reason", result)
        self.assertEqual(result["sample_count"], 46)
        self.assertEqual(result["status"], "passed")

    def test_custom_overlap_fraction_is_honored(self) -> None:
        # 45 of 50 = 90% overlap: passes the default threshold but fails a
        # caller-supplied stricter 95% requirement.
        solver_rows = [
            {"step": step, "time_s": step * 5.0e-4, "tip_disp": 0.01 * step}
            for step in range(1, 46)
        ]
        fluent_rows = [
            {
                "step": step,
                "time_s": step * 5.0e-4,
                "monitor_avg_total_col0_col6_m": 0.01 * step,
            }
            for step in range(1, 51)
        ]

        default_result = compare_structure_monitor(
            solver_rows,
            fluent_rows,
            solver_displacement_key="tip_disp",
        )
        strict_result = compare_structure_monitor(
            solver_rows,
            fluent_rows,
            solver_displacement_key="tip_disp",
            min_temporal_overlap_fraction=0.95,
        )

        self.assertNotIn("reason", default_result)
        self.assertEqual(strict_result["status"], "failed")
        self.assertIn("insufficient temporal overlap", strict_result["reason"])

    def test_empty_series_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            compare_structure_monitor(
                [],
                [{"step": 1, "monitor_avg_total_col0_col6_m": 0.0}],
                solver_displacement_key="tip_disp",
            )


if __name__ == "__main__":
    unittest.main()
