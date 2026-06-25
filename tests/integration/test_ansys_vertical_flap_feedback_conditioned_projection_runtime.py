from __future__ import annotations

import math
import unittest

from cases.ansys_vertical_flap_fsi import (
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


class AnsysVerticalFlapFeedbackConditionedProjectionRuntimeTests(unittest.TestCase):
    def test_two_step_runtime_consumes_feedback_after_first_step(self) -> None:
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(
                step_count=2,
                flow_projection_iterations=2,
                solid_substeps=1,
            )
        )

        history = report["history"]

        self.assertEqual(len(history), 2)
        self.assertIs(history[0]["fluid_projection_consumed_feedback"], False)
        self.assertIs(history[1]["fluid_projection_consumed_feedback"], True)
        self.assertEqual(report["fluid_projection_consumed_feedback_count"], 1)
        self.assertEqual(
            report["fluid_projection_consumed_feedback_count"],
            report["config"]["step_count"] - 1,
        )
        self.assertGreater(history[1]["fluid_feedback_constraint_marker_count"], 0)
        self.assertGreater(history[1]["fluid_feedback_constraint_active_cell_count"], 0)
        self.assertGreaterEqual(
            history[1]["fluid_feedback_constraint_cleared_cell_count"],
            0,
        )
        self.assertGreaterEqual(
            history[1]["fluid_feedback_constraint_non_obstacle_cell_count"],
            0,
        )
        self.assertGreaterEqual(
            history[1]["fluid_feedback_constraint_projection_participating_cell_count"],
            0,
        )
        self.assertTrue(
            math.isfinite(history[1]["no_slip_target_residual_after_assembly_mps"])
        )
        self.assertTrue(
            math.isfinite(history[1]["no_slip_projected_residual_after_projection_mps"])
        )

    def test_three_step_runtime_clears_previous_feedback_constraints(self) -> None:
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(
                step_count=3,
                flow_projection_iterations=2,
                solid_substeps=1,
            )
        )

        history = report["history"]
        step_1 = history[0]
        step_2 = history[1]
        step_3 = history[2]

        self.assertEqual(len(history), 3)
        self.assertIs(step_1["fluid_projection_consumed_feedback"], False)
        self.assertIs(step_2["fluid_projection_consumed_feedback"], True)
        self.assertIs(step_3["fluid_projection_consumed_feedback"], True)
        self.assertEqual(report["fluid_projection_consumed_feedback_count"], 2)
        self.assertEqual(
            report["fluid_projection_consumed_feedback_count"],
            report["config"]["step_count"] - 1,
        )
        self.assertGreater(step_2["fluid_feedback_constraint_active_cell_count"], 0)
        self.assertGreater(step_3["fluid_feedback_constraint_cleared_cell_count"], 0)
        self.assertLessEqual(
            step_3["fluid_feedback_constraint_cleared_cell_count"],
            step_2["fluid_feedback_constraint_active_cell_count"],
        )
        self.assertGreaterEqual(
            step_3["fluid_feedback_constraint_projection_participating_cell_count"],
            0,
        )
        self.assertTrue(
            math.isfinite(step_3["no_slip_target_residual_after_assembly_mps"])
        )
        self.assertTrue(
            math.isfinite(step_3["no_slip_projected_residual_after_projection_mps"])
        )


if __name__ == "__main__":
    unittest.main()
