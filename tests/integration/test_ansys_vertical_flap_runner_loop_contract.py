from __future__ import annotations

import unittest
from pathlib import Path


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"


class AnsysVerticalFlapRunnerLoopContractTests(unittest.TestCase):
    def test_runtime_projects_flow_inside_each_trial_before_stress_sampling(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())
        project_index = loop_body.index("_flow_advance_current_step(")
        stress_index = loop_body.index(
            "_sample_stress_to_marker_forces(\n            markers,"
        )

        self.assertLess(project_index, stress_index)
        self.assertIn("_project_current_flow(", _flow_advance_body(_runner_source()))

    def test_closed_loop_solver_must_report_fluid_recompute_count(self) -> None:
        source = _runner_source()

        self.assertIn("fluid_projection_count = 0", source)
        self.assertIn("fluid_projection_after_feedback_count = 0", source)
        self.assertNotIn("feedback_available_for_projection = False", source)
        self.assertIn("feedback_available_before_projection = apply_feedback", source)
        self.assertIn("fluid_projection_after_feedback_count > 0", source)
        self.assertIn('"fluid_recomputed_after_feedback"', source)
        self.assertIn('"feedback_closure_status"', source)
        self.assertIn('"CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK"', source)
        self.assertIn('"OPEN_LOOP_OR_PREFEEDBACK_ONLY"', source)
        self.assertIn('"fluid_recompute_count"', source)
        self.assertIn('"fluid_projection_count"', source)
        self.assertIn('"fluid_projection_after_feedback_count"', source)

    def test_closed_loop_solver_must_record_per_step_flow_recompute_fields(self) -> None:
        history_body = _history_append_body(_runner_source())

        self.assertIn('"fluid_recomputed"', history_body)
        self.assertIn('"fluid_recomputed_after_feedback"', history_body)
        self.assertIn('"feedback_available_before_projection"', history_body)
        self.assertIn('"local_velocity_peak_mps"', history_body)
        self.assertIn('"pressure_min_pa"', history_body)
        self.assertIn('"pressure_max_pa"', history_body)
        self.assertIn('"flow_projection_report"', history_body)

    def test_closed_loop_solver_must_project_fluid_inside_each_trial(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())
        stress_index = loop_body.index(
            "_sample_stress_to_marker_forces(\n            markers,"
        )

        recompute_indices = [
            loop_body.find(token)
            for token in (
                "fluid.project(",
                "_project_current_flow(",
                "_flow_advance_current_step(",
                "_recompute_current_flow(",
                "_project_flow_for_step(",
                "_recompute_flow_for_step(",
            )
            if loop_body.find(token) >= 0
        ]
        self.assertTrue(
            recompute_indices,
            "FSI trial must project or recompute fluid before stress sampling",
        )
        self.assertLess(min(recompute_indices), stress_index)

    def test_runtime_invalidates_step_transaction_around_begin_commit_and_rollback(
        self,
    ) -> None:
        source = _runner_source()
        runtime_start = source.index("    class RectangularMarkerVelocityRuntime")
        runtime_end = source.index("    runtime = RectangularMarkerVelocityRuntime()")
        runtime_source = source[runtime_start:runtime_end]
        begin_start = runtime_source.index("        def begin_step(")
        restore_start = runtime_source.index("        def restore_step_base(")
        begin_source = runtime_source[begin_start:restore_start]
        commit_start = runtime_source.index("        def commit_step(")
        publish_start = runtime_source.index("        def publish_step(")
        commit_source = runtime_source[commit_start:publish_start]
        rollback_start = runtime_source.index("        def rollback_step(")
        finalize_start = runtime_source.index("        def finalize_run(")
        rollback_source = runtime_source[rollback_start:finalize_start]

        self.assertIn("self.step_transaction_ready = False", runtime_source)
        self.assertIn("self.clear_step_transaction()", begin_source)
        self.assertIn("self.step_transaction_ready = True", begin_source)
        self.assertLess(
            begin_source.index("marker_base_state = capture_marker_interface_state"),
            begin_source.index("self.step_transaction_ready = True"),
        )
        self.assertIn("self.clear_step_transaction()", commit_source)
        self.assertIn("if not self.step_transaction_ready:", rollback_source)
        self.assertIn("self.clear_step_transaction()", rollback_source)

    def test_trial_restore_preserves_hibm_resources_and_invalidates_topology(
        self,
    ) -> None:
        source = _runner_source()
        runtime_start = source.index("    class RectangularMarkerVelocityRuntime")
        runtime_end = source.index("    runtime = RectangularMarkerVelocityRuntime()")
        runtime_source = source[runtime_start:runtime_end]
        restore_start = runtime_source.index("        def restore_step_base(")
        evaluate_start = runtime_source.index("        def evaluate_trial(")
        restore_source = runtime_source[restore_start:evaluate_start]

        self.assertNotIn("sharp_boundary_cache.clear()", restore_source)
        self.assertIn(
            'sharp_boundary_cache.get("hibm_sharp_marker_boundary")',
            restore_source,
        )
        for key in (
            "classified_topology_key",
            "search_report",
            "internal_obstacle_cell_count",
            "cleanup_report",
        ):
            with self.subTest(key=key):
                self.assertIn(f'cache_entry.pop("{key}", None)', restore_source)


def _runner_source() -> str:
    return RUNNER_SOURCE.read_text(encoding="utf-8")


def _fsi_loop_body(source: str) -> str:
    trial_start = source.index("    def evaluate_trial(")
    trial_end = source.index("    def commit_step(", trial_start)
    return source[trial_start:trial_end]


def _history_append_body(source: str) -> str:
    commit_start = source.index("    def commit_step(")
    commit_end = source.index("    class RectangularMarkerVelocityRuntime", commit_start)
    commit_body = source[commit_start:commit_end]
    append_start = commit_body.index("history.append(")
    append_end = commit_body.index("\n        )", append_start) + len("\n        )")
    return commit_body[append_start:append_end]


def _flow_advance_body(source: str) -> str:
    start = source.index("def _flow_advance_current_step(")
    end = source.index("\ndef _effective_flow_driver_mode", start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
