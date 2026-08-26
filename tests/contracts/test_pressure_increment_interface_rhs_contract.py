from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOLVER_PATH = ROOT / "simulation_core" / "fluids" / "solver.py"
RUNNER_PATH = ROOT / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
CORE_PATH = ROOT / "simulation_core" / "coupling" / "hibm_mpm" / "core.py"
MODE_ARGUMENT = "homogenize_pressure_interface_rhs_for_increment"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    for statement in _module(path).body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"missing function {name!r} in {path}")


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for statement in _module(path).body:
        if not isinstance(statement, ast.ClassDef) or statement.name != class_name:
            continue
        for member in statement.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"missing method {class_name}.{method_name}")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and (
            (isinstance(candidate.func, ast.Name) and candidate.func.id == name)
            or (
                isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr == name
            )
        )
    ]


def _keywords(call: ast.Call) -> dict[str, ast.AST]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _default(function: ast.FunctionDef, argument_name: str) -> ast.AST:
    positional = [*function.args.posonlyargs, *function.args.args]
    offset = len(positional) - len(function.args.defaults)
    for index, argument in enumerate(positional):
        if argument.arg == argument_name:
            if index < offset:
                raise AssertionError(f"argument {argument_name!r} has no default")
            return function.args.defaults[index - offset]
    for argument, default in zip(
        function.args.kwonlyargs,
        function.args.kw_defaults,
    ):
        if argument.arg == argument_name:
            if default is None:
                raise AssertionError(f"argument {argument_name!r} has no default")
            return default
    raise AssertionError(f"missing argument {argument_name!r}")


class PressureIncrementInterfaceRhsContracts(unittest.TestCase):
    def test_solver_mode_is_explicit_default_off_and_reported(self) -> None:
        project = _class_method(
            SOLVER_PATH,
            "CartesianFluidSolver",
            "project",
        )

        self.assertIs(ast.literal_eval(_default(project, MODE_ARGUMENT)), False)
        self.assertIn(
            MODE_ARGUMENT,
            {
                node.value
                for node in ast.walk(project)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            },
        )

    def test_runner_enables_homogeneous_rhs_only_for_joint_consistency_p(self) -> None:
        wrapper = _top_level_function(RUNNER_PATH, "_project_current_flow")
        self.assertIs(ast.literal_eval(_default(wrapper, MODE_ARGUMENT)), False)
        wrapper_project_calls = _calls(wrapper, "project")
        self.assertEqual(len(wrapper_project_calls), 1)
        self.assertIn(MODE_ARGUMENT, _keywords(wrapper_project_calls[0]))

        advance = _top_level_function(RUNNER_PATH, "_flow_advance_current_step_trial")
        calls = _calls(advance, "_project_current_flow")
        self.assertEqual(len(calls), 2)
        consistency_calls = [
            call
            for call in calls
            if ast.literal_eval(
                _keywords(call).get(
                    "accumulate_pressure_into_previous",
                    ast.Constant(value=False),
                )
            )
            is True
        ]
        self.assertEqual(len(consistency_calls), 1)
        consistency_keywords = _keywords(consistency_calls[0])
        self.assertIs(
            ast.literal_eval(consistency_keywords[MODE_ARGUMENT]),
            True,
        )
        main_call = next(call for call in calls if call is not consistency_calls[0])
        self.assertNotIn(MODE_ARGUMENT, _keywords(main_call))

    def test_generic_core_enables_only_consistency_not_post_solid_projection(
        self,
    ) -> None:
        assembly = _top_level_function(
            CORE_PATH,
            "assemble_hibm_mpm_sharp_fluid_to_mpm_loads",
        )
        incremental_assembly_calls = [
            call
            for call in _calls(assembly, "project")
            if "accumulate_pressure_into_previous" in _keywords(call)
        ]
        self.assertEqual(len(incremental_assembly_calls), 1)
        self.assertIn(MODE_ARGUMENT, _keywords(incremental_assembly_calls[0]))

        post_solid = _top_level_function(
            CORE_PATH,
            "advance_hibm_mpm_sharp_mpm_step",
        )
        incremental_post_solid_calls = [
            call
            for call in _calls(post_solid, "project")
            if "accumulate_pressure_into_previous" in _keywords(call)
        ]
        self.assertEqual(len(incremental_post_solid_calls), 1)
        self.assertNotIn(MODE_ARGUMENT, _keywords(incremental_post_solid_calls[0]))


if __name__ == "__main__":
    unittest.main()
