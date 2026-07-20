"""Source contracts for marker-MAC PCG device work elision.

These tests intentionally parse the production source without importing Taichi.
They are RED design contracts, not numerical-runtime evidence: a later GREEN run
must still exercise the real kernels on the supported Taichi backend.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


OPERATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation_core"
    / "coupling"
    / "hibm_mpm"
    / "marker_mac_constraint.py"
)
OPERATOR_CLASS = "HibmMpmMarkerMacConstraintOperator"


def _module() -> ast.Module:
    return ast.parse(
        OPERATOR_PATH.read_text(encoding="utf-8"),
        filename=str(OPERATOR_PATH),
    )


def _method(name: str) -> ast.FunctionDef:
    for statement in _module().body:
        if not isinstance(statement, ast.ClassDef):
            continue
        if statement.name != OPERATOR_CLASS:
            continue
        for member in statement.body:
            if isinstance(member, ast.FunctionDef) and member.name == name:
                return member
    raise AssertionError(f"missing method {OPERATOR_CLASS}.{name}")


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _argument_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _keyword_value(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"call at line {call.lineno} is missing keyword {name!r}")


def _is_bool_literal(node: ast.AST, expected: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is expected


def _self_field_subscripts(node: ast.AST, field_name: str) -> list[ast.Subscript]:
    matches: list[ast.Subscript] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript):
            continue
        value = candidate.value
        if not isinstance(value, ast.Attribute) or value.attr != field_name:
            continue
        if not isinstance(value.value, ast.Name) or value.value.id != "self":
            continue
        matches.append(candidate)
    return matches


def _top_level_loop(function: ast.FunctionDef) -> ast.For:
    for statement in function.body:
        if isinstance(statement, ast.For):
            return statement
    raise AssertionError(
        f"{function.name} must keep its Taichi struct_for at kernel top level"
    )


def _per_iteration_guard(function: ast.FunctionDef, loop: ast.For) -> ast.If:
    for statement in loop.body:
        if isinstance(statement, ast.If):
            return statement
    raise AssertionError(
        f"{function.name} must guard work inside its top-level struct_for body"
    )


def _attribute_names(node: ast.AST) -> set[str]:
    return {
        candidate.attr
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Attribute)
    }


def _name_ids(node: ast.AST) -> set[str]:
    return {
        candidate.id
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name)
    }


def _failure_codes_written(function: ast.FunctionDef) -> set[int]:
    codes: set[int] = set()
    for candidate in ast.walk(function):
        if isinstance(candidate, ast.Assign):
            targets = candidate.targets
            value = candidate.value
        elif isinstance(candidate, ast.AnnAssign):
            targets = [candidate.target]
            value = candidate.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        if any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "_failure_code"
            for target in targets
        ):
            codes.add(int(value.value))

    for call in _calls(function, "atomic_max"):
        if len(call.args) < 2:
            continue
        target, value = call.args[:2]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "_failure_code"
        ):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            codes.add(int(value.value))
    return codes


def _raise_messages(function: ast.FunctionDef) -> list[tuple[int, str]]:
    messages: list[tuple[int, str]] = []
    for candidate in ast.walk(function):
        if not isinstance(candidate, ast.Raise) or candidate.exc is None:
            continue
        if not isinstance(candidate.exc, ast.Call) or not candidate.exc.args:
            continue
        message = candidate.exc.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            messages.append((candidate.lineno, message.value))
    return messages


class MarkerMacPcgWorkElisionContracts(unittest.TestCase):
    def test_full_grid_matrix_work_is_device_gated_after_convergence(self) -> None:
        """Every O(grid) kernel must skip work on converged/failed iterations."""

        kernel_names = (
            "_clear_grid_scratch_kernel",
            "_scatter_rows_to_grid_kernel",
            "_gather_grid_to_rows_kernel",
        )
        for kernel_name in kernel_names:
            with self.subTest(kernel=kernel_name):
                kernel = _method(kernel_name)
                self.assertIn(
                    "force_run",
                    _argument_names(kernel),
                    "the same kernel must support ordinary elision and the final "
                    "forced lambda apply",
                )
                loop = _top_level_loop(kernel)
                guard = _per_iteration_guard(kernel, loop)
                condition_attributes = _attribute_names(guard.test)
                condition_names = _name_ids(guard.test)
                self.assertIn("force_run", condition_names)
                self.assertIn("_device_converged", condition_attributes)
                self.assertIn("_failure_code", condition_attributes)
                self.assertIsInstance(
                    guard.test,
                    ast.BoolOp,
                    "the device guard must admit force_run OR healthy-unconverged",
                )
                self.assertIsInstance(guard.test.op, ast.Or)

    def test_iterative_apply_elides_but_final_lambda_apply_force_runs(self) -> None:
        """The final lambda-to-correction transfer must survive convergence."""

        solve = _method("solve_device")
        iteration_loop = next(
            (
                statement
                for statement in solve.body
                if isinstance(statement, ast.For)
                and _calls(statement, "_apply_matrix")
            ),
            None,
        )
        self.assertIsNotNone(iteration_loop, "solve_device lost its fixed host loop")
        assert iteration_loop is not None
        iterative_calls = _calls(iteration_loop, "_apply_matrix")
        self.assertEqual(len(iterative_calls), 1)
        self.assertTrue(
            _is_bool_literal(
                _keyword_value(iterative_calls[0], "force_run"),
                False,
            ),
            "ordinary PCG matrix applies must remain device-elidable",
        )

        final_calls = []
        for call in _calls(solve, "_apply_matrix"):
            if call in iterative_calls or not call.args:
                continue
            first_argument = call.args[0]
            if (
                isinstance(first_argument, ast.Attribute)
                and first_argument.attr == "_lambda"
            ):
                final_calls.append(call)
        self.assertEqual(
            len(final_calls),
            1,
            "solve_device must have exactly one final lambda matrix apply",
        )
        final_call = final_calls[0]
        self.assertGreater(final_call.lineno, iteration_loop.end_lineno)
        self.assertTrue(
            _is_bool_literal(_keyword_value(final_call, "force_run"), True),
            "lambda-to-correction apply must execute even after device convergence",
        )

    def test_fixed_host_loop_never_polls_device_convergence(self) -> None:
        """Elision stays device-resident instead of synchronizing every iteration."""

        solve = _method("solve_device")
        iteration_loop = next(
            statement
            for statement in solve.body
            if isinstance(statement, ast.For)
            and _calls(statement, "_apply_matrix")
        )
        self.assertEqual(
            _self_field_subscripts(iteration_loop, "_device_converged"),
            [],
            "the fixed host loop must not read device convergence",
        )
        apply_matrix = _method("_apply_matrix")
        self.assertEqual(
            _self_field_subscripts(apply_matrix, "_device_converged"),
            [],
            "the per-iteration Python wrapper must pass a flag, not poll a field",
        )
        post_loop_reads = _self_field_subscripts(solve, "_device_converged")
        self.assertEqual(
            len(post_loop_reads),
            1,
            "device convergence may be synchronized exactly once after the host loop",
        )
        self.assertGreater(post_loop_reads[0].lineno, iteration_loop.end_lineno)

    def test_zero_mobility_and_breakdown_remain_fail_closed(self) -> None:
        """Work elision must not turn unsatisfiable systems into false success."""

        initialize = _method("_initialize_pcg_kernel")
        pcg_step = _method("_pcg_step_device_kernel")
        finish_direction = _method("_pcg_finish_direction_device_kernel")
        self.assertIn(3, _failure_codes_written(initialize))
        self.assertIn(4, _failure_codes_written(pcg_step))
        self.assertIn(5, _failure_codes_written(finish_direction))

        solve = _method("solve_device")
        iteration_loop = next(
            statement
            for statement in solve.body
            if isinstance(statement, ast.For)
            and _calls(statement, "_apply_matrix")
        )
        messages = _raise_messages(solve)
        zero_mobility_raises = [
            line
            for line, message in messages
            if "no free MAC support" in message
        ]
        breakdown_raises = [
            line
            for line, message in messages
            if "PCG breakdown" in message
        ]
        self.assertEqual(len(zero_mobility_raises), 1)
        self.assertLess(zero_mobility_raises[0], iteration_loop.lineno)
        self.assertEqual(len(breakdown_raises), 1)
        self.assertGreater(breakdown_raises[0], iteration_loop.end_lineno)


if __name__ == "__main__":
    unittest.main()
