from __future__ import annotations

import unittest
from pathlib import Path


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"


class AnsysVerticalFlapFeedbackConditionedProjectionTests(unittest.TestCase):
    def test_runner_applies_marker_feedback_to_fluid_before_projection(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())

        apply_index = loop_body.index("_apply_marker_feedback_to_fluid(")
        project_index = loop_body.index("_project_current_flow(")
        stress_index = loop_body.index("_sample_stress_to_marker_forces(markers, fluid)")

        self.assertLess(apply_index, project_index)
        self.assertLess(project_index, stress_index)

    def test_runner_tracks_feedback_consumed_projection_count(self) -> None:
        source = _runner_source()

        self.assertIn("fluid_projection_consumed_feedback_count = 0", source)
        self.assertIn("fluid_projection_consumed_feedback_count += 1", source)
        self.assertIn('"fluid_projection_consumed_feedback_count"', source)
        self.assertIn('"fluid_projection_consumed_feedback"', source)

    def test_runner_reports_feedback_constraint_metrics_per_step(self) -> None:
        history_body = _history_append_body(_runner_source())

        for field in (
            '"fluid_projection_consumed_feedback"',
            '"fluid_feedback_constraint_marker_count"',
            '"fluid_feedback_constraint_active_cell_count"',
            '"no_slip_residual_before_mps"',
            '"no_slip_residual_after_mps"',
        ):
            self.assertIn(field, history_body)

    def test_adapter_reads_marker_feedback_and_updates_fluid_constraints(self) -> None:
        source = _runner_source()
        adapter_body = _function_body(source, "def _apply_marker_feedback_to_fluid(")

        self.assertIn("markers.x_gamma_m.to_numpy()", adapter_body)
        self.assertIn("markers.v_gamma_mps.to_numpy()", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_active.to_numpy()", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_value_mps.to_numpy()", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_active.from_numpy", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_value_mps.from_numpy", adapter_body)
        self.assertIn("fluid.velocity_dirichlet_boundary_projection_weight.from_numpy", adapter_body)


def _runner_source() -> str:
    return RUNNER_SOURCE.read_text(encoding="utf-8")


def _fsi_loop_body(source: str) -> str:
    loop_start = source.index("for step_index in range(config.step_count):")
    loop_end = source.index("    if (\n        latest_stress_report is None", loop_start)
    return source[loop_start:loop_end]


def _history_append_body(source: str) -> str:
    loop_body = _fsi_loop_body(source)
    append_start = loop_body.index("history.append(")
    append_end = loop_body.index("\n        )", append_start) + len("\n        )")
    return loop_body[append_start:append_end]


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    next_function = source.find("\ndef ", start + len(signature))
    if next_function < 0:
        return source[start:]
    return source[start:next_function]


if __name__ == "__main__":
    unittest.main()
