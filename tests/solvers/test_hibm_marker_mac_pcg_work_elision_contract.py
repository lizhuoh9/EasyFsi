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


def _solver_iteration_loop(function: ast.FunctionDef) -> ast.While:
    for statement in function.body:
        if isinstance(statement, ast.While) and _calls(statement, "_apply_matrix"):
            return statement
    raise AssertionError(
        f"{function.name} must keep its actual-device-budgeted host loop"
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

    def test_iterative_apply_elides_and_exact_confirmation_reuses_candidate(
        self,
    ) -> None:
        """The exact candidate must be materialized once and reused after acceptance."""

        solve = _method("solve_device")
        iteration_loop = _solver_iteration_loop(solve)
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
            final_calls,
            [],
            "solve_device must not rematerialize lambda after exact acceptance",
        )

        confirmation = _method("_confirm_exact_pcg_residual")
        confirmation_calls = _calls(confirmation, "_apply_matrix")
        self.assertEqual(len(confirmation_calls), 1)
        self.assertTrue(
            _is_bool_literal(
                _keyword_value(confirmation_calls[0], "force_run"),
                True,
            ),
            "exact confirmation must force materialization after recursive convergence",
        )
        self.assertEqual(
            len(_calls(confirmation, "_copy_grid_scratch_to_correction_kernel")),
            1,
            "the candidate being confirmed must be the pending correction",
        )
        self.assertEqual(
            len(_calls(solve, "_compute_true_candidate_residual_kernel")),
            1,
            "final validation must remeasure the accepted candidate without rebuilding it",
        )

    def test_host_loop_polls_terminal_state_in_bounded_batches(self) -> None:
        """A converged solve must not dispatch the full configured iteration budget."""

        solve = _method("solve_device")
        iteration_loop = _solver_iteration_loop(solve)
        convergence_reads = _self_field_subscripts(
            iteration_loop.test,
            "_device_converged",
        )
        failure_reads = _self_field_subscripts(
            iteration_loop.test,
            "_failure_code",
        )
        self.assertGreaterEqual(
            len(convergence_reads),
            1,
            "the host loop must periodically observe device convergence",
        )
        self.assertGreaterEqual(
            len(failure_reads),
            1,
            "the host loop must stop promptly after a device-side breakdown",
        )
        actual_iteration_reads = _self_field_subscripts(
            iteration_loop,
            "_device_iterations",
        )
        self.assertGreaterEqual(
            len(actual_iteration_reads),
            3,
            "batch dispatch and remaining budget must use actual device iterations",
        )
        exact_confirmation_branches = [
            candidate
            for candidate in ast.walk(iteration_loop)
            if isinstance(candidate, ast.If)
            and _calls(candidate, "_confirm_exact_pcg_residual")
        ]
        self.assertEqual(
            len(exact_confirmation_branches),
            1,
            "recursive convergence or budget exhaustion must trigger one exact check",
        )

    def test_recursive_convergence_is_replaced_by_exact_residual_restart(self) -> None:
        """A recursive stop is provisional until ``rhs - A lambda`` is rebuilt."""

        replacement = _method("_replace_pcg_residual_from_exact_candidate_kernel")
        replacement_attributes = _attribute_names(replacement)
        for required in (
            "_rhs",
            "_correction",
            "_stencil_free",
            "_stencil_index",
            "_stencil_weight",
            "_residual",
            "_preconditioned",
            "_direction",
            "_rz_old",
            "_device_converged",
        ):
            with self.subTest(required=required):
                self.assertIn(required, replacement_attributes)

        confirmation = _method("_confirm_exact_pcg_residual")
        matrix_calls = _calls(confirmation, "_apply_matrix")
        self.assertEqual(len(matrix_calls), 1)
        self.assertTrue(
            _is_bool_literal(_keyword_value(matrix_calls[0], "force_run"), True)
        )
        self.assertEqual(
            len(
                _calls(
                    confirmation,
                    "_replace_pcg_residual_from_exact_candidate_kernel",
                )
            ),
            1,
        )

    def test_zero_mobility_and_breakdown_remain_fail_closed(self) -> None:
        """Work elision must not turn unsatisfiable systems into false success."""

        initialize = _method("_initialize_pcg_kernel")
        pcg_step = _method("_pcg_step_device_kernel")
        finish_direction = _method("_pcg_finish_direction_device_kernel")
        self.assertIn(3, _failure_codes_written(initialize))
        self.assertIn(4, _failure_codes_written(pcg_step))
        self.assertIn(5, _failure_codes_written(finish_direction))

        solve = _method("solve_device")
        iteration_loop = _solver_iteration_loop(solve)
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
