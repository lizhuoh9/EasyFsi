from __future__ import annotations

import ast
import unittest
from pathlib import Path


RUNNER_SOURCE = Path("benchmarks") / "official" / "solid_mpm_fsi_runner.py"


class AnsysVerticalFlapRunnerLoopContractTests(unittest.TestCase):
    def test_runner_projects_flow_inside_fsi_loop_before_stress_sampling(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())
        project_index = loop_body.index("_flow_advance_current_step(")
        stress_index = loop_body.index("_sample_stress_to_marker_forces(")

        self.assertLess(project_index, stress_index)
        self.assertIn("_project_current_flow(", _flow_advance_body(_runner_source()))

    def test_closed_loop_solver_must_report_fluid_recompute_count(self) -> None:
        source = _runner_source()

        self.assertIn("fluid_projection_count = 0", source)
        self.assertIn("fluid_projection_after_feedback_count = 0", source)
        self.assertIn("feedback_available_for_projection = False", source)
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

    def test_iqn_runner_maps_generic_threshold_audit_histories(self) -> None:
        source = _runner_source()
        mapping_start = source.index(
            "coupling_step_report = {",
            source.index("generic_run = solve_fsi_runtime("),
        )
        mapping_end = source.index("\n        step_trial_work_summary", mapping_start)
        mapping_body = source[mapping_start:mapping_end]

        expected_history_fields = (
            (
                "fsi_coupling_relative_residual_history",
                "hibm_fsi_coupling_relative_residual_history",
            ),
            (
                "fsi_coupling_absolute_residual_history_mps",
                "hibm_fsi_coupling_absolute_residual_history_mps",
            ),
            (
                "fsi_coupling_candidate_velocity_rms_history_mps",
                "hibm_fsi_coupling_candidate_velocity_rms_history_mps",
            ),
            (
                "fsi_coupling_max_marker_residual_history_mps",
                "hibm_fsi_coupling_max_marker_residual_history_mps",
            ),
            (
                "fsi_coupling_relative_tolerance_equivalent_history_mps",
                "hibm_fsi_coupling_relative_tolerance_equivalent_history_mps",
            ),
            (
                "fsi_coupling_effective_tolerance_history_mps",
                "hibm_fsi_coupling_effective_tolerance_history_mps",
            ),
            (
                "fsi_coupling_residual_to_effective_tolerance_history",
                "hibm_fsi_coupling_residual_to_effective_tolerance_history",
            ),
            (
                "fsi_iqn_fallback_reasons",
                "hibm_fsi_coupling_iqn_fallback_reasons",
            ),
            (
                "fsi_iqn_update_limited_history",
                "hibm_fsi_coupling_iqn_update_limited_history",
            ),
        )
        for generic_field, runner_field in expected_history_fields:
            self.assertIn(generic_field, mapping_body)
            self.assertIn(runner_field, mapping_body)

    def test_closed_loop_solver_must_project_fluid_inside_fsi_loop(self) -> None:
        loop_body = _fsi_loop_body(_runner_source())
        stress_index = loop_body.index("_sample_stress_to_marker_forces(")

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
            "FSI loop must project or recompute fluid before stress sampling",
        )
        self.assertLess(min(recompute_indices), stress_index)

    def test_research_probe_rejects_iqn_history_reuse(self) -> None:
        source = _runner_source()
        config_start = source.index(
            "def _iqn_kalman_oracle_interpolation_config("
        )
        config_end = source.index("\ndef _mixed_iqn_kalman_oracle_guess(", config_start)
        config_body = source[config_start:config_end]

        self.assertIn(
            'getattr(config, "iqn_reuse_previous_step_history", False)',
            config_body,
        )
        self.assertIn(
            "Kalman-Oracle interpolation probe isolates initial-guess ",
            config_body,
        )
        self.assertIn(
            "interpolation and requires iqn_reuse_previous_step_history=False",
            config_body,
        )

    def test_one_partitioned_trial_is_extracted_with_acceptance_outside(self) -> None:
        source = _runner_source()
        tree = ast.parse(source)
        run_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_hibm_mpm_fsi"
        )
        step_loop = next(
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "step_index"
        )
        trial_functions = [
            node
            for node in step_loop.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_hibm_mpm_coupling_trial"
        ]
        self.assertEqual(len(trial_functions), 1)
        trial_body = ast.get_source_segment(source, trial_functions[0])
        self.assertIsNotNone(trial_body)
        ordered_tokens = (
            "_flow_advance_current_step(",
            "_sample_stress_to_marker_forces(",
            "scatter_marker_forces_to_mpm_particles(",
            "_select_and_advance_solid_macro_step(",
            "update_surface_feedback_from_mpm_surface_particles(",
            "_apply_hibm_sharp_marker_boundary_to_fluid(",
        )
        indices = [trial_body.index(token) for token in ordered_tokens]
        self.assertEqual(indices, sorted(indices))
        for accepted_only_token in (
            "refresh_runtime_pressure_pair_anchors()",
            "kalman_controller.commit_step()",
            "history.append(",
        ):
            self.assertNotIn(accepted_only_token, trial_body)
        trial_calls = [
            node
            for node in ast.walk(step_loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_hibm_mpm_coupling_trial"
        ]
        self.assertEqual(
            len(trial_calls),
            2,
            "one direct call and one generic-runtime callback must share the "
            "same extracted physical trial",
        )
        loop_body = ast.get_source_segment(source, step_loop)
        self.assertIsNotNone(loop_body)
        self.assertIn("HibmMpmMarkerVelocityRuntime(", loop_body)
        self.assertIn("solve_fsi_runtime(", loop_body)
        self.assertIn("solve_fsi_step(", loop_body)
        probe_index = loop_body.index("research_probe_terminal")
        self.assertLess(loop_body.index("solve_fsi_step("), probe_index)
        self.assertNotIn("iqn_runtime.commit_step(", loop_body)
        self.assertNotIn("run_strong_coupling_iterations(", loop_body)
        self.assertNotIn("aitken", loop_body.lower())
        self.assertNotIn("solid_step_execution_reports.append", trial_body)

    def test_research_probe_terminal_satisfies_official_report_contract(self) -> None:
        source = _runner_source()
        tree = ast.parse(source)
        run_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_hibm_mpm_fsi"
        )
        terminal_returns = []
        for node in ast.walk(run_function):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "research_probe_terminal" in keys:
                terminal_returns.append(keys)

        self.assertEqual(len(terminal_returns), 1)
        self.assertIn("computed_result_sources", terminal_returns[0])
        self.assertIn("taichi_runtime_identity", terminal_returns[0])
        self.assertIn("profile_wall_time_enabled", terminal_returns[0])
        terminal_return = next(
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "research_probe_terminal"
                for key in node.value.keys
            )
        )
        self.assertTrue(
            any(
                key is None
                and isinstance(value, ast.Name)
                and value.id == "preflow_report"
                for key, value in zip(
                    terminal_return.value.keys,
                    terminal_return.value.values,
                )
            )
        )

    def test_research_probe_recaptures_and_compares_after_each_rollback(self) -> None:
        source = _runner_source()
        probe_start = source.index(
            "if (\n                research_probe_config is not None"
        )
        probe_end = source.index("\n            generic_run = solve_fsi_runtime(", probe_start)
        probe = source[probe_start:probe_end]

        rollback = probe.index("probe_runtime.rollback_step(context)")
        recapture = probe.index(
            "restored_probe_state = capture_iqn_step_state()",
            rollback,
        )
        compare = probe.index(
            "_host_macro_step_state_mismatch_fields(",
            recapture,
        )
        append = probe.index("probe_rows.append(", compare)
        self.assertLess(rollback, recapture)
        self.assertLess(recapture, compare)
        self.assertLess(compare, append)
        self.assertIn(
            "research probe rollback changed accepted HostMacroStepState",
            probe,
        )

    def test_iqn_trial_reseals_pressure_pair_anchors_without_accepting_feedback(self) -> None:
        source = _runner_source()
        tree = ast.parse(source)
        apply_guess = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_iqn_marker_velocity_guess"
        )
        body = ast.get_source_segment(source, apply_guess)
        self.assertIsNotNone(body)

        restore_index = body.index("restore_marker_interface_state(")
        refresh_index = body.index("refresh_runtime_pressure_pair_anchors()")
        invalidate_index = body.index(
            "_invalidate_hibm_sharp_boundary_derived_cache("
        )
        self.assertLess(restore_index, refresh_index)
        self.assertLess(refresh_index, invalidate_index)
        self.assertNotIn(
            "feedback_available_for_projection = True",
            body,
            "a same-time IQN guess is not accepted prior-step feedback",
        )

    def test_direct_runner_initializes_strict_cuda_before_fluid_build(self) -> None:
        tree = ast.parse(_runner_source())
        run_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_hibm_mpm_fsi"
        )
        runtime_calls = [
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TaichiRuntimeConfig"
        ]
        self.assertEqual(len(runtime_calls), 1)
        runtime_call = runtime_calls[0]
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in runtime_call.keywords
        }
        self.assertEqual(
            keywords,
            {
                "arch": "cuda",
                "default_fp": "f32",
                "default_ip": "i32",
                "random_seed": 0,
                "cfg_optimization": False,
                "opt_level": 1,
                "advanced_optimization": True,
                "fast_math": True,
                "debug": False,
                "strict_arch": True,
            },
        )
        fluid_build = next(
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_fluid"
        )
        self.assertLess(runtime_call.lineno, fluid_build.lineno)


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


def _flow_advance_body(source: str) -> str:
    start = source.index("def _flow_advance_current_step(")
    end = source.index("\ndef _effective_flow_driver_mode", start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
