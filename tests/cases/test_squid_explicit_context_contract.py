from __future__ import annotations

import ast
import unittest
from dataclasses import fields
from pathlib import Path

from cases.squid_soft_robot.step_context import (
    StepLoopCallbacks,
    StepLoopContext,
    StepLoopMutableState,
    StepLoopResources,
    StepLoopResult,
    StepLoopSettings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SQUID_CASE_ROOT = REPO_ROOT / "cases" / "squid_soft_robot"


def _parsed_module(filename: str) -> ast.Module:
    return ast.parse(
        (SQUID_CASE_ROOT / filename).read_text(encoding="utf-8"),
        filename=filename,
    )


def _module_bound_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names
            )
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _context_lookup_names(module: ast.Module, helper_name: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            continue
        names.add(node.args[1].value)
    return names


def _single_constructor_call(module: ast.Module, class_name: str) -> ast.Call:
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == class_name
    ]
    if len(calls) != 1:
        raise AssertionError(f"expected one {class_name} call, found {len(calls)}")
    return calls[0]


class SquidExplicitContextContractTests(unittest.TestCase):
    def test_runner_does_not_inject_module_globals_into_split_modules(self) -> None:
        runner = _parsed_module("runner.py")
        globals_calls = [
            node
            for node in ast.walk(runner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "globals"
        ]
        self.assertEqual(globals_calls, [])

    def test_step_loop_imports_module_dependencies_instead_of_context_lookup(
        self,
    ) -> None:
        runner = _parsed_module("runner.py")
        step_loop = _parsed_module("step_loop.py")
        hidden_module_dependencies = _context_lookup_names(
            step_loop,
            "_required_context_value",
        ) & _module_bound_names(runner)
        self.assertEqual(hidden_module_dependencies, set())
        self.assertEqual(
            _context_lookup_names(step_loop, "_required_context_value"), set()
        )

    def test_step_loop_context_is_split_by_runtime_role(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StepLoopContext)),
            ("settings", "resources", "callbacks", "state"),
        )
        self.assertEqual(len(fields(StepLoopSettings)), 32)
        self.assertEqual(
            tuple(field.name for field in fields(StepLoopResources)),
            (
                "args",
                "fluid_substep_controller",
                "history_path",
                "material",
                "output_dir",
                "process_path",
                "run_checkpoint_path",
                "frozen_run_fingerprint",
                "run_started_at_perf",
                "simulator",
                "solid_mpm",
                "spec",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(StepLoopCallbacks)),
            ("publish_solid_report_to_reduced_state",),
        )
        self.assertEqual(
            tuple(field.name for field in fields(StepLoopMutableState)),
            (
                "first_step",
                "rows",
                "sharp_coupling_state",
                "partial_run_reason",
                "partial_run_stopped",
                "previous_step_cfl",
                "previous_step_fluid_substeps",
            ),
        )
        grouped_names = [
            field.name
            for group in (
                StepLoopSettings,
                StepLoopResources,
                StepLoopCallbacks,
                StepLoopMutableState,
            )
            for field in fields(group)
        ]
        self.assertEqual(len(grouped_names), len(set(grouped_names)))

    def test_runner_builds_typed_step_loop_context_without_locals(self) -> None:
        runner_source = (SQUID_CASE_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("run_squid_step_loop(locals())", runner_source)
        self.assertIn("step_loop_context = StepLoopContext(", runner_source)
        self.assertIn("run_squid_step_loop(step_loop_context)", runner_source)
        runner = _parsed_module("runner.py")
        for context_group in (
            StepLoopSettings,
            StepLoopResources,
            StepLoopCallbacks,
            StepLoopMutableState,
        ):
            call = _single_constructor_call(runner, context_group.__name__)
            expected_names = {field.name for field in fields(context_group)}
            actual_names = {keyword.arg for keyword in call.keywords}
            self.assertEqual(actual_names, expected_names)
            if context_group is StepLoopCallbacks:
                callback_values = {
                    keyword.arg: keyword.value.id
                    for keyword in call.keywords
                    if keyword.arg is not None
                    and isinstance(keyword.value, ast.Name)
                }
                self.assertEqual(
                    callback_values,
                    {
                        "publish_solid_report_to_reduced_state": (
                            "publish_solid_report_to_reduced_state"
                        ),
                    },
                )
                continue
            self.assertTrue(
                all(
                    keyword.arg is not None
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == keyword.arg
                    for keyword in call.keywords
                )
            )

    def test_step_loop_consumes_typed_context_without_namespace_hydration(self) -> None:
        step_loop_source = (SQUID_CASE_ROOT / "step_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "def run_squid_step_loop(context: StepLoopContext)", step_loop_source
        )
        self.assertNotIn("_required_context_value", step_loop_source)
        self.assertNotIn(" in context:", step_loop_source)

    def test_step_loop_does_not_alias_single_use_immutable_context_fields(self) -> None:
        step_loop = _parsed_module("step_loop.py")
        run_loop = next(
            node
            for node in step_loop.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_squid_step_loop"
        )
        immutable_aliases: set[str] = set()
        for statement in run_loop.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            value = statement.value
            if not (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in {"settings", "resources", "callbacks"}
            ):
                continue
            immutable_aliases.add(target.id)
        loaded_names = [
            node.id
            for node in ast.walk(run_loop)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        ]
        single_use_aliases = {
            name for name in immutable_aliases if loaded_names.count(name) == 1
        }
        self.assertEqual(single_use_aliases, set())

    def test_summary_imports_module_dependencies_instead_of_context_lookup(
        self,
    ) -> None:
        runner = _parsed_module("runner.py")
        summary = _parsed_module("summary.py")
        hidden_module_dependencies = _context_lookup_names(
            summary,
            "_context_value",
        ) & _module_bound_names(runner)
        self.assertEqual(hidden_module_dependencies, set())
        self.assertTrue(_context_lookup_names(summary, "_context_value"))

    def test_step_loop_result_has_only_cross_boundary_state(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StepLoopResult)),
            (
                "rows",
                "sharp_coupling_state",
                "partial_run_stopped",
                "partial_run_reason",
            ),
        )

    def test_step_loop_does_not_leak_its_local_namespace(self) -> None:
        step_loop_source = (SQUID_CASE_ROOT / "step_loop.py").read_text(
            encoding="utf-8"
        )
        runner_source = (SQUID_CASE_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("dict(locals())", step_loop_source)
        self.assertNotIn("**step_loop_result", runner_source)


if __name__ == "__main__":
    unittest.main()
