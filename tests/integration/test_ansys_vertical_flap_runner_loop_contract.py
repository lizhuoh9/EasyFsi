from __future__ import annotations

import unittest
from pathlib import Path


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"


class AnsysVerticalFlapRunnerLoopContractTests(unittest.TestCase):
    def test_current_runner_solves_computed_flow_before_fsi_loop(self) -> None:
        source = _runner_source()
        solve_index = source.index("_solve_computed_flow(fluid, config)")
        loop_index = source.index("for step_index in range(config.step_count):")
        stress_index = source.index("_sample_stress_to_marker_forces(markers, fluid)")

        self.assertLess(solve_index, loop_index)
        self.assertGreater(stress_index, loop_index)

    @unittest.expectedFailure
    def test_closed_loop_solver_must_report_fluid_recompute_count(self) -> None:
        source = _runner_source()

        self.assertIn('"fluid_recomputed_after_feedback"', source)
        self.assertIn('"feedback_closure_status"', source)
        self.assertIn('"CLOSED_LOOP_RECOMPUTED_FLOW"', source)
        self.assertIn('"fluid_recompute_count"', source)

    @unittest.expectedFailure
    def test_closed_loop_solver_must_record_per_step_flow_recompute_fields(self) -> None:
        history_body = _history_append_body(_runner_source())

        self.assertIn('"fluid_recomputed"', history_body)
        self.assertIn('"local_velocity_peak_mps"', history_body)
        self.assertIn('"pressure_min_pa"', history_body)
        self.assertIn('"pressure_max_pa"', history_body)
        self.assertIn('"flow_projection_report"', history_body)

    @unittest.expectedFailure
    def test_closed_loop_solver_must_project_fluid_inside_fsi_loop(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())
        stress_index = loop_body.index("_sample_stress_to_marker_forces(markers, fluid)")

        recompute_indices = [
            loop_body.find(token)
            for token in (
                "fluid.project(",
                "_project_current_flow(",
                "_recompute_current_flow(",
                "_project_flow_for_step(",
                "_recompute_flow_for_step(",
            )
            if loop_body.find(token) >= 0
        ]
        self.assertTrue(
            recompute_indices,
            "FSI loop must project or recompute fluid before stress sampling",
        )
        self.assertLess(min(recompute_indices), stress_index)


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


if __name__ == "__main__":
    unittest.main()
