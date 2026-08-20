from __future__ import annotations

import ast
import importlib
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path


class SquidPackageExportTests(unittest.TestCase):
    def test_single_sharp_solver_has_no_mode_selection_api(self) -> None:
        for module_name in (
            "simulation_core",
            "simulation_core.coupling",
            "simulation_core.coupling.hibm_mpm",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(
                    module.FSI_COUPLING_MODE_HIBM_MPM_SHARP,
                    "hibm_mpm_sharp",
                )
                for removed_name in (
                    "FSI_COUPLING_MODE_CHOICES",
                    "FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED",
                    "fsi_coupling_mode_report",
                    "require_implemented_fsi_coupling_mode",
                ):
                    self.assertFalse(hasattr(module, removed_name))

    def test_removed_legacy_coupling_module_and_symbols_are_absent(self) -> None:
        legacy_module = Path("cases/squid_soft_robot/coupling_legacy.py")
        self.assertFalse(legacy_module.exists())
        self.assertIsNone(
            importlib.util.find_spec("cases.squid_soft_robot.coupling_legacy")
        )

        source = "\n".join(
            path.read_text(encoding="utf-8")
            for root in (Path("cases/squid_soft_robot"), Path("simulation_core"))
            for path in sorted(root.rglob("*.py"))
        )
        self.assertNotIn("legacy_projected_reduced", source)

    def test_cli_has_no_solver_mode_option(self) -> None:
        cli = importlib.import_module("cases.squid_soft_robot.cli")

        args = cli.parse_args([])
        self.assertFalse(hasattr(args, "fsi_coupling_mode"))
        for removed_value in ("legacy_projected_reduced", "hibm_mpm_sharp"):
            with self.subTest(removed_value=removed_value):
                with self.assertRaises(SystemExit):
                    cli.parse_args(["--fsi-coupling-mode", removed_value])

    def test_cli_rejects_removed_noop_coupling_options(self) -> None:
        cli = importlib.import_module("cases.squid_soft_robot.cli")
        removed_options = (
            ("--constraint-force-scale", "1.0"),
            ("--fsi-constraint-force-solid-mobility-ratio", "0.0"),
            ("--fsi-solid-response-mobility-coupling",),
            ("--fsi-solid-response-velocity-mobility-coupling",),
            ("--fsi-velocity-target-solid-mobility-ratio", "0.0"),
            ("--fsi-velocity-constraint-blend", "0.0"),
            ("--fsi-velocity-constraint-solid-mobility-ratio", "0.0"),
            ("--interface-reaction-passivity-limit",),
            ("--interface-reaction-aitken-lower-bound", "0.01"),
            ("--interface-reaction-aitken-upper-bound", "1.0"),
            ("--interface-reaction-robin-impedance-ns-m", "0.0"),
            ("--interface-reaction-robin-matrix-impedance-ns-m", "0.0"),
            ("--interface-reaction-robin-target-mode", "stabilized"),
            ("--ibm-correction-iterations", "2"),
            ("--fsi-coupling-adaptive-iterations-max", "2"),
            ("--fsi-coupling-adaptive-iterations-residual-threshold-n", "1.0"),
            ("--fsi-coupling-adaptive-iterations-cfl-threshold", "0.1"),
            ("--fsi-coupling-same-step-rerun-iterations-max", "2"),
            ("--fsi-coupling-same-step-rerun-residual-threshold-n", "1.0"),
            ("--fsi-coupling-same-step-rerun-fluid-substep-factor", "2.0"),
            ("--fsi-coupling-residual-continuation-iterations-max", "2"),
            ("--fsi-coupling-residual-continuation-threshold-n", "1.0"),
            ("--fsi-coupling-residual-continuation-rebound-secant-from-best",),
            ("--fsi-coupling-residual-continuation-rebound-secant-factor", "2.0"),
            (
                "--fsi-coupling-residual-continuation-rebound-secant-evaluation-extensions-max",
                "2",
            ),
            ("--fsi-coupling-trial-interior-divergence-tolerance", "1.0"),
            ("--fsi-stabilization-preset", "off"),
            ("--fsi-coupling-solver", "aitken"),
            ("--fsi-coupling-tolerance-n", "1.0"),
            ("--fsi-coupling-target-map-relaxation", "1.0"),
            ("--fsi-coupling-rejected-trial-backtrack", "1.0"),
            ("--fsi-coupling-residual-growth-rejection-factor", "2.0"),
            ("--fsi-coupling-max-accepted-residual-n", "1.0"),
            ("--fsi-coupling-trust-region-force-increment-n", "1.0"),
            ("--fsi-coupling-trust-region-adaptive",),
            ("--fsi-coupling-trust-region-shrink-factor", "0.5"),
            ("--fsi-coupling-trust-region-growth-factor", "1.5"),
            ("--fsi-coupling-trust-region-rebound-factor", "2.0"),
            ("--fsi-coupling-trust-region-rebound-backtrack", "0.5"),
            ("--fsi-coupling-trust-region-rebound-stop-factor", "2.0"),
            ("--fsi-coupling-trust-region-rebound-stop-max-residual-n", "1.0"),
            ("--reuse-accepted-fsi-trial-state",),
        )

        defaults = cli.parse_args([])
        self.assertTrue(hasattr(defaults, "interface_reaction_aitken"))
        self.assertTrue(hasattr(defaults, "interface_reaction_relaxation"))
        for option in removed_options:
            dest = option[0][2:].replace("-", "_")
            with self.subTest(option=option[0]):
                self.assertFalse(hasattr(defaults, dest))
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        cli.parse_args(list(option))

    def test_noop_coupling_plumbing_is_absent_from_case_source(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("cases/squid_soft_robot").rglob("*.py"))
        )

        for symbol in (
            "FSI_STABILIZATION_PRESET_CHOICES",
            "FSI_STABILIZATION_PRESET_CONFLICT_POLICY",
            "resolve_fsi_stabilization_preset_parameters",
            "fsi_stabilization_effective_parameters_from_args",
            "sharp_fsi_iteration_budget",
            "fsi_coupling_adaptive_iterations_max",
            "fsi_coupling_same_step_rerun_iterations_max",
            "fsi_coupling_residual_continuation_iterations_max",
            "fsi_coupling_trial_interior_divergence_tolerance",
            "fsi_coupling_solver",
            "fsi_coupling_tolerance_n",
            "fsi_coupling_target_map_relaxation",
            "fsi_coupling_rejected_trial_backtrack",
            "fsi_coupling_residual_growth_rejection_factor",
            "fsi_coupling_max_accepted_residual_n",
            "fsi_coupling_trust_region_force_increment_n",
            "reuse_accepted_fsi_trial_state",
            "constraint_force_scale",
            "fsi_constraint_force_solid_mobility_ratio",
            "fsi_solid_response_mobility_coupling",
            "fsi_solid_response_velocity_mobility_coupling",
            "fsi_velocity_target_solid_mobility_ratio",
            "fsi_velocity_constraint_blend",
            "fsi_velocity_constraint_solid_mobility_ratio",
            "interface_reaction_passivity_limit",
            "interface_reaction_aitken_lower_bound",
            "interface_reaction_aitken_upper_bound",
            "interface_reaction_robin_impedance_ns_m",
            "interface_reaction_robin_matrix_impedance_ns_m",
            "interface_reaction_robin_target_mode",
            "INTERFACE_REACTION_ROBIN_TARGET_CHOICES",
            "raise_for_unsupported_hibm_mpm_sharp_robin_options",
            "_raw_cli_option_present",
            "ibm_correction_iterations",
            "accepted_fsi_trial_state_reused",
            "fsi_trial_pressure_projection_cg_project_calls",
            "fsi_coupling_iqn_ils_least_squares_update_count",
        ):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)

        self.assertIn("fsi_coupling_aitken_update_count", source)

    def test_runtime_state_has_no_zero_only_interface_reaction_state(self) -> None:
        runtime_source = Path("cases/squid_soft_robot/runtime_state.py").read_text(
            encoding="utf-8"
        )
        runtime_module = ast.parse(runtime_source)
        runtime_class = next(
            node
            for node in runtime_module.body
            if isinstance(node, ast.ClassDef) and node.name == "ReducedSquidFSI"
        )
        self_attributes = {
            node.attr
            for node in ast.walk(runtime_class)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        method_names = {
            node.name
            for node in runtime_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for removed_attribute in (
            "primary_interface_reaction_force_n",
            "secondary_interface_reaction_force_n",
            "saved_primary_interface_reaction_force_n",
            "saved_secondary_interface_reaction_force_n",
            "sample_report_host_snapshot",
        ):
            with self.subTest(removed_attribute=removed_attribute):
                self.assertNotIn(removed_attribute, self_attributes)
        for removed_method in (
            "set_interface_reaction",
            "set_interface_reaction_kernel",
        ):
            with self.subTest(removed_method=removed_method):
                self.assertNotIn(removed_method, method_names)

        checkpoint_source = Path(
            "cases/squid_soft_robot/checkpointing.py"
        ).read_text(encoding="utf-8")
        for removed_checkpoint_symbol in (
            "primary_interface_reaction_force_n",
            "secondary_interface_reaction_force_n",
            "_read_vector_field",
            "_write_vector_field",
        ):
            with self.subTest(removed_checkpoint_symbol=removed_checkpoint_symbol):
                self.assertNotIn(removed_checkpoint_symbol, checkpoint_source)

        # Historical output columns remain stable even though this obsolete state
        # is no longer allocated or checkpointed by the Squid case.
        self.assertIn('"main_interface_reaction_z_n"', runtime_source)
        self.assertIn('"tail_interface_reaction_z_n"', runtime_source)

    def test_step_loop_has_explicit_result_without_dead_interface_state(self) -> None:
        case_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("cases/squid_soft_robot").rglob("*.py"))
        )
        for removed_symbol in (
            "interface_reaction_state",
            "InterfaceReactionRelaxationState",
            "_checkpoint_interface_state_dict",
            "_interface_state_from_checkpoint",
            "return dict(locals())",
        ):
            with self.subTest(removed_symbol=removed_symbol):
                self.assertNotIn(removed_symbol, case_source)

        step_loop_source = Path("cases/squid_soft_robot/step_loop.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(step_loop_source)
        run_function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_squid_step_loop"
        )
        result = run_function.body[-1]
        self.assertIsInstance(result, ast.Return)
        self.assertIsInstance(result.value, ast.Dict)
        self.assertEqual(
            [key.value for key in result.value.keys],
            ["rows", "partial_run_stopped", "partial_run_reason"],
        )

    def test_step_loop_context_contains_only_required_live_inputs(self) -> None:
        source = Path("cases/squid_soft_robot/step_loop.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        run_function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_squid_step_loop"
        )
        first_step_loop_index = next(
            index
            for index, node in enumerate(run_function.body)
            if isinstance(node, ast.For)
        )
        optional_context_names = []
        for node in run_function.body[:first_step_loop_index]:
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            if (
                len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.In)
                and isinstance(node.test.left, ast.Constant)
                and node.test.comparators
                and isinstance(node.test.comparators[0], ast.Name)
                and node.test.comparators[0].id == "context"
            ):
                optional_context_names.append(node.test.left.value)
        self.assertEqual(optional_context_names, [])

        required_context_names = {
            target.id: statement.value.args[1].value
            for statement in run_function.body[:first_step_loop_index]
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance((target := statement.targets[0]), ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "_required_context_value"
            and len(statement.value.args) == 2
            and isinstance(statement.value.args[1], ast.Constant)
        }
        for required_name in (
            "partial_run_reason",
            "partial_run_stopped",
            "previous_step_cfl",
            "previous_step_fluid_substeps",
        ):
            with self.subTest(required_name=required_name):
                self.assertEqual(required_context_names[required_name], required_name)
        self.assertNotIn("Mapping", required_context_names)
        self.assertNotIn("math", required_context_names)
        self.assertNotIn("np", required_context_names)
        self.assertNotIn("solid_mpm_report = None", source)
        self.assertNotIn("\n        continue\n\n    return {", source)
        runner_source = Path("cases/squid_soft_robot/runner.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("**step_loop_result", runner_source)

    def test_final_squid_internal_dead_code_is_absent(self) -> None:
        runner_source = Path("cases/squid_soft_robot/runner.py").read_text(
            encoding="utf-8"
        )
        for dead_runner_name in (
            "solid_response_dt_s",
            "fsi_solid_response_dt_s",
            "total_fsi_face_area_m2",
            "primary_fsi_face_area_m2",
            "secondary_fsi_face_area_m2",
            "**step_loop_result",
        ):
            with self.subTest(dead_runner_name=dead_runner_name):
                self.assertNotIn(dead_runner_name, runner_source)

        summary_module = ast.parse(
            Path("cases/squid_soft_robot/summary.py").read_text(encoding="utf-8")
        )
        summary_functions = {
            node.name for node in summary_module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("runtime_budget_report", summary_functions)

        setup_module = ast.parse(
            Path("cases/squid_soft_robot/setup.py").read_text(encoding="utf-8")
        )
        setup_functions = {
            node.name for node in setup_module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("_z_min_connected_active_mask", setup_functions)
        self.assertNotIn("reduced_active_water_connectivity", setup_functions)

        outputs_source = Path("cases/squid_soft_robot/outputs.py").read_text(
            encoding="utf-8"
        )
        spec_source = Path("cases/squid_soft_robot/spec.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_final_row_int", outputs_source)
        self.assertNotIn("_row_bool", outputs_source)
        self.assertNotIn("from collections.abc import Mapping", spec_source)

        snapshots_module = ast.parse(
            Path("cases/squid_soft_robot/snapshots.py").read_text(encoding="utf-8")
        )
        snapshot_writer = next(
            node
            for node in snapshots_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_write_fluid_snapshot_npz"
        )
        assigned_names = {
            node.id
            for node in ast.walk(snapshot_writer)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.assertNotIn("nz", assigned_names)

        cli_source = Path("cases/squid_soft_robot/cli.py").read_text(encoding="utf-8")
        self.assertNotIn("accepted-step interface-reaction residual updates", cli_source)
        self.assertIn(
            '"Use Aitken Delta^2 adaptation for step-internal interface-reaction "',
            cli_source,
        )
        self.assertIn(
            '"fixed-point updates. "',
            cli_source,
        )

    def test_root_docs_describe_current_sharp_architecture(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        for stale_claim in (
            "功能等价重构副本",
            '以"行为不变"为第一原则',
            "未做结构性大改",
            "公开 API 完全不变",
            "top-level legacy modules are compatibility shims",
        ):
            with self.subTest(stale_claim=stale_claim):
                self.assertNotIn(stale_claim, readme)
        for current_entry in (
            "benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi",
            "cases.ansys_vertical_flap_fsi.run_ansys_vertical_flap_benchmark",
            "Squid",
            "Turek-Hron",
            "COMSOL",
        ):
            with self.subTest(current_entry=current_entry):
                self.assertIn(current_entry, readme)

        historical_goal = Path(
            "docs/refactoring/SQUID_2S_SIMULATION_GOAL_2026-06-17.md"
        ).read_text(encoding="utf-8")
        banner = historical_goal[:1200]
        self.assertIn("已归档", banner)
        self.assertIn("legacy_projected_reduced", banner)
        self.assertIn("HIBM_MPM_SHARP", banner)
        self.assertIn("docs/MODULE_MAP.md", banner)

    def test_case_package_exposes_main_for_generic_entrypoint(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        self.assertTrue(callable(module.main))

    def test_case_package_allows_runner_submodule_import(self) -> None:
        runner = importlib.import_module("cases.squid_soft_robot.runner")
        self.assertTrue(callable(runner.main))

    def test_new_code_can_import_explicit_submodules(self) -> None:
        for module_name in (
            "cases.squid_soft_robot.cli",
            "cases.squid_soft_robot.spec",
            "cases.squid_soft_robot.source_config",
            "cases.squid_soft_robot.schedules",
            "cases.squid_soft_robot.history",
            "cases.squid_soft_robot.checkpointing",
            "cases.squid_soft_robot.diagnostics",
            "cases.squid_soft_robot.snapshots",
            "cases.squid_soft_robot.runtime_state",
            "cases.squid_soft_robot.summary",
            "cases.squid_soft_robot.rows",
            "cases.squid_soft_robot.setup",
            "cases.squid_soft_robot.step_loop",
            "cases.squid_soft_robot.solid_step",
            "cases.squid_soft_robot.coupling_common",
            "cases.squid_soft_robot.coupling_sharp",
        ):
            module = importlib.import_module(module_name)
            self.assertIsNotNone(module)

    def test_removed_legacy_support_modules_are_absent(self) -> None:
        for module_name in ("step_context", "trial_replay", "fluid_step"):
            with self.subTest(module_name=module_name):
                path = Path("cases/squid_soft_robot") / f"{module_name}.py"
                self.assertFalse(path.exists())
                self.assertIsNone(
                    importlib.util.find_spec(f"cases.squid_soft_robot.{module_name}")
                )

    def test_removed_case_legacy_helpers_are_absent(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("cases/squid_soft_robot").rglob("*.py"))
        )

        for symbol in (
            "build_projected_ibm_region_pair_step_config",
            "accepted_trial_replay_reports",
            "StepTiming",
            "required_projected_ibm_force_report",
            "fsi_trial_acceptance_passes",
            "fsi_same_step_rerun_triggered",
            "force_decomposition_report",
            "FINITE_REQUIRED_ROW_FIELDS",
        ):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)

    def test_case_package_has_only_explicit_main_export(self) -> None:
        module = importlib.import_module("cases.squid_soft_robot")
        source = Path("cases/squid_soft_robot/__init__.py").read_text(encoding="utf-8")

        self.assertEqual(module.__all__, ("main",))
        self.assertFalse(hasattr(module, "ReducedSquidFSI"))
        self.assertFalse(hasattr(module, "_cell_indices_for_points"))
        self.assertNotIn("_EXPORT_MODULES", source)
        self.assertNotIn("def __getattr__", source)


if __name__ == "__main__":
    unittest.main()
