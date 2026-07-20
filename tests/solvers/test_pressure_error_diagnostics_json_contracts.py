from __future__ import annotations

import inspect
import json
import re
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

import simulation_core.fluids.solver as fluid_solver_module
from simulation_core.fluids.constants import HIBM_PRESSURE_COMPONENT_CAPACITY
from simulation_core.fluids.solver import (
    CartesianFluidSolver,
    PressureSolveConvergenceError,
    _run_pre_projection_velocity_transaction,
)


class _HostScalar:
    def __init__(self, value: int | float = 0) -> None:
        self.value = value

    def __getitem__(self, _index: object) -> int | float:
        return self.value

    def __setitem__(self, _index: object, value: int | float) -> None:
        self.value = value


class _OpaqueDiagnostic:
    pass


class _StringTrapMappingKey:
    """Hashable mapping key whose string conversion must never be executed."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.string_call_count = 0

    def __str__(self) -> str:
        self.string_call_count += 1
        raise AssertionError("diagnostic mapping keys must not execute __str__")


class PressureErrorJsonContracts(unittest.TestCase):
    @staticmethod
    def _solver_with_graph_metadata(
        *,
        valid: bool,
        context: str = "stale-context",
        source_count: int = 4,
        solve_count: int = 3,
    ) -> CartesianFluidSolver:
        solver = object.__new__(CartesianFluidSolver)
        solver._pressure_outlet_nullspace_graph_valid = bool(valid)
        solver._pressure_outlet_nullspace_graph_context = str(context)
        solver._pressure_outlet_nullspace_source_component_count = int(source_count)
        solver._pressure_outlet_nullspace_component_count = int(solve_count)
        solver._hibm_pressure_unreached_count = 0
        solver._hibm_pressure_unreached_component_count = 0
        solver.last_hibm_pressure_component_labels_converged = True
        solver.last_hibm_pressure_unreached_component_overflow = False
        solver.last_hibm_reachability_valid = True
        solver.hibm_reachability_revision = 2
        solver.last_hibm_reachability_revision = 2
        solver._hibm_reachability_checksum = ("published",)
        return solver

    def _assert_missing_or_stale_graph_failure(self, call) -> None:
        with self.assertRaises(PressureSolveConvergenceError) as caught:
            call()
        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["stage"], "pressure_nullspace_component_graph")
        self.assertEqual(diagnostics["reason"], "operator_graph_missing_or_stale")
        for key in (
            "count",
            "capacity",
            "labels_converged",
            "component_overflow",
        ):
            self.assertIn(key, diagnostics)
        self.assertIsInstance(diagnostics["count"], int)
        self.assertEqual(diagnostics["capacity"], HIBM_PRESSURE_COMPONENT_CAPACITY)
        self.assertIsInstance(diagnostics["labels_converged"], bool)
        self.assertIsInstance(diagnostics["component_overflow"], bool)
        json.dumps(diagnostics, allow_nan=False)

    def test_pressure_error_diagnostics_are_deeply_json_safe_without_mutating_input(
        self,
    ) -> None:
        array = np.asarray([np.nan, np.inf, -np.inf, 2.5], dtype=np.float64)
        values = [np.float64(np.nan), np.int64(7), _OpaqueDiagnostic()]
        nested = {
            "array": array,
            "values": values,
            "mapping": {"positive_inf": float("inf")},
        }
        source = {"nested": nested, "negative_inf": float("-inf")}
        array_before = array.copy()
        value_identities = tuple(id(value) for value in values)

        error = PressureSolveConvergenceError(
            "synthetic diagnostics",
            diagnostics=source,
        )

        json.dumps(error.diagnostics, allow_nan=False)
        self.assertIsNot(error.diagnostics, source)
        self.assertIsNot(error.diagnostics["nested"], nested)
        self.assertIsNot(error.diagnostics["nested"]["values"], values)
        self.assertIs(source["nested"], nested)
        self.assertIs(nested["array"], array)
        self.assertIs(nested["values"], values)
        self.assertTrue(np.array_equal(array, array_before, equal_nan=True))
        self.assertEqual(tuple(id(value) for value in values), value_identities)
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 7)
        self.assertIsInstance(values[2], _OpaqueDiagnostic)
        self.assertTrue(np.isinf(source["negative_inf"]))

    def test_json_safe_mapping_keys_neither_execute_user_code_nor_collide(
        self,
    ) -> None:
        first_trap = _StringTrapMappingKey("first")
        second_trap = _StringTrapMappingKey("second")
        source_mapping = {
            "1": "string-key",
            1: "integer-key",
            first_trap: "first-trap",
            second_trap: "second-trap",
        }

        error = PressureSolveConvergenceError(
            "mapping-key safety",
            diagnostics={"mapping": source_mapping},
        )

        safe_mapping = error.diagnostics["mapping"]
        self.assertIsInstance(safe_mapping, dict)
        self.assertEqual(len(safe_mapping), len(source_mapping))
        self.assertEqual(safe_mapping["1"], "string-key")
        self.assertCountEqual(
            safe_mapping.values(),
            ("string-key", "integer-key", "first-trap", "second-trap"),
        )
        self.assertTrue(all(isinstance(key, str) for key in safe_mapping))
        self.assertEqual(first_trap.string_call_count, 0)
        self.assertEqual(second_trap.string_call_count, 0)
        json.dumps(error.diagnostics, allow_nan=False)

    @staticmethod
    def _failure_diagnostics_stub() -> SimpleNamespace:
        return SimpleNamespace(
            pressure_interface_matrix_terms_report=lambda: {
                "row_diagonal_integral": 1.0,
                "row_diagonal_integral_abs_mismatch": 0.0,
                "row_count": 0,
                "row_active_count": 0,
                "row_invalid_count": 0,
                "row_overflow_count": 0,
            },
            last_cg_preconditioner_requested="fv_multigrid",
            last_cg_preconditioner_effective="fv_multigrid",
            last_cg_multigrid_pre_smooth_iterations=1,
            last_cg_multigrid_coarse_smooth_iterations=1,
            last_cg_multigrid_post_smooth_iterations=1,
            _mg_shapes=(),
            last_cg_multigrid_apply_count=0,
            last_cg_multigrid_to_jacobi_fallback_count=0,
            last_cg_iterations=1,
            last_cg_pcg_iterations=1,
            last_cg_bicgstab_iterations=0,
            last_cg_recursive_relative_residual=1.0,
            last_cg_exact_relative_residual=1.0,
            last_cg_exact_residual_confirmation_count=1,
            last_cg_breakdown="unit-test",
            last_cg_restart_count=0,
        )

    def test_nonfinite_or_opaque_step_indices_use_json_safe_integer_defaults(
        self,
    ) -> None:
        cases = (
            (float("nan"), 4, -1, 4),
            (5, float("inf"), 5, -1),
            (_OpaqueDiagnostic(), _OpaqueDiagnostic(), -1, -1),
        )
        for local, global_, expected_local, expected_global in cases:
            with self.subTest(local=type(local), global_=type(global_)):
                raw_diagnostics = (
                    CartesianFluidSolver._pressure_solve_failure_diagnostics(
                        self._failure_diagnostics_stub(),
                        tolerance=1.0e-8,
                        pressure_interface_matrix_active=False,
                        context={
                            "phase": "unit-test",
                            "step_index_local": local,
                            "step_index_global": global_,
                        },
                    )
                )
                error = PressureSolveConvergenceError(
                    "unsafe step-index regression",
                    diagnostics=raw_diagnostics,
                )
                self.assertEqual(
                    error.diagnostics["step_index_local"],
                    expected_local,
                )
                self.assertEqual(
                    error.diagnostics["step_index_global"],
                    expected_global,
                )
                self.assertIsInstance(error.diagnostics["step_index_local"], int)
                self.assertIsInstance(error.diagnostics["step_index_global"], int)
                json.dumps(error.diagnostics, allow_nan=False)

    def test_opaque_phase_uses_stable_fallback_without_string_conversion(
        self,
    ) -> None:
        phase = _StringTrapMappingKey("phase")

        raw_diagnostics = CartesianFluidSolver._pressure_solve_failure_diagnostics(
            self._failure_diagnostics_stub(),
            tolerance=1.0e-8,
            pressure_interface_matrix_active=False,
            context={"phase": phase},
        )

        self.assertEqual(raw_diagnostics["phase"], "unknown")
        self.assertEqual(phase.string_call_count, 0)

    def test_pressure_solve_context_keys_are_validated_without_stringifying(
        self,
    ) -> None:
        copy_context = getattr(
            fluid_solver_module,
            "_validated_pressure_solve_context",
        )
        key = _StringTrapMappingKey("context-key")
        source = {"phase": "unit-test"}

        copied = copy_context(source)
        self.assertEqual(copied, source)
        self.assertIsNot(copied, source)
        self.assertEqual(copy_context(None), {})
        with self.assertRaisesRegex(TypeError, "keys must be strings"):
            copy_context({key: "unsafe"})
        self.assertEqual(key.string_call_count, 0)

    def test_project_context_and_phase_have_no_raw_string_conversion(self) -> None:
        source = inspect.getsource(CartesianFluidSolver.project)
        self.assertNotIn("str(key): value", source)
        unsafe_phase = re.compile(
            r"(?<![\w])str\s*\(\s*pressure_solve_context_payload\.get\s*\(\s*"
            r"['\"]phase['\"]"
        )
        self.assertEqual(unsafe_phase.findall(source), [])

    def test_failure_diagnostics_use_stable_phase_without_stringifying_objects(
        self,
    ) -> None:
        phase_trap = _StringTrapMappingKey("phase")
        cases = (
            ("assembly", "assembly"),
            (None, "unknown"),
            (17, "unknown"),
            (phase_trap, "unknown"),
        )
        for phase, expected in cases:
            with self.subTest(phase_type=type(phase)):
                diagnostics = CartesianFluidSolver._pressure_solve_failure_diagnostics(
                    self._failure_diagnostics_stub(),
                    tolerance=1.0e-8,
                    pressure_interface_matrix_active=False,
                    context={"phase": phase},
                )
                self.assertEqual(diagnostics["phase"], expected)
        self.assertEqual(phase_trap.string_call_count, 0)

    def test_project_rejects_non_string_context_keys_without_stringifying_them(
        self,
    ) -> None:
        key_trap = _StringTrapMappingKey("project-context-key")
        solver = object.__new__(CartesianFluidSolver)
        solver._require_velocity_dirichlet_component_ledger_sealed = MethodType(
            lambda _self: None,
            solver,
        )

        with self.assertRaises(TypeError):
            solver.project(
                iterations=1,
                pressure_solve_context={key_trap: "value"},
            )

        self.assertEqual(key_trap.string_call_count, 0)

    def test_project_context_copy_and_all_phase_paths_avoid_raw_stringification(
        self,
    ) -> None:
        project_source = inspect.getsource(CartesianFluidSolver.project)
        failure_source = inspect.getsource(
            CartesianFluidSolver._pressure_solve_failure_diagnostics
        )
        self.assertNotRegex(project_source, r"str\s*\(\s*key\s*\)")
        unsafe_project_phase = re.compile(
            r"(?<![\w])str\s*\(\s*pressure_solve_context_payload\.get\s*\(\s*"
            r"['\"]phase['\"]"
        )
        unsafe_failure_phase = re.compile(
            r"(?<![\w])str\s*\(\s*context\.get\s*\(\s*['\"]phase['\"]"
        )
        self.assertEqual(unsafe_project_phase.findall(project_source), [])
        self.assertEqual(unsafe_failure_phase.findall(failure_source), [])

    def test_project_preflight_diagnostics_have_no_raw_step_index_int_casts(
        self,
    ) -> None:
        source = inspect.getsource(CartesianFluidSolver.project)
        unsafe_cast = re.compile(
            r"(?<!\w)int\s*\(\s*pressure_solve_context_payload\.get\s*\(\s*"
            r"['\"]step_index_(?:local|global)['\"]"
        )
        self.assertEqual(
            unsafe_cast.findall(source),
            [],
            "all label/overflow/failure diagnostics paths must use the same "
            "non-throwing integer conversion contract",
        )

    def test_direct_operator_graph_consumers_fail_with_one_structured_contract(
        self,
    ) -> None:
        consumers = (
            lambda solver: solver._subtract_pressure_outlet_nullspace_component_means_device(
                object()
            ),
            lambda solver: solver._measure_pressure_outlet_nullspace_rhs_incompatibility(
                object()
            ),
            lambda solver: solver._confirm_pressure_poisson_fv_cg_exact_residual(
                pressure_outlet_zmin=False,
                tolerance=1.0e-8,
                remove_nullspace_mean=True,
                pressure_nullspace_component_count=1,
                pressure_components_use_operator_graph=True,
            ),
        )
        for consumer in consumers:
            with self.subTest(consumer=consumer.__code__.co_firstlineno):
                solver = self._solver_with_graph_metadata(valid=False)
                self._assert_missing_or_stale_graph_failure(
                    lambda consumer=consumer, solver=solver: consumer(solver)
                )

    def test_closed_neumann_solver_rejects_missing_and_stale_operator_graph_structurally(
        self,
    ) -> None:
        for valid, context in ((False, ""), (True, "wrong-context")):
            with self.subTest(valid=valid, context=context):
                solver = self._solver_with_graph_metadata(
                    valid=valid,
                    context=context,
                    source_count=2,
                    solve_count=2,
                )
                solver.cg_breakdown_code = _HostScalar(0)
                solver.cg_breakdown_dAd = _HostScalar(0.0)
                solver._hibm_pressure_unreached_component_count = 0
                solver._hibm_pressure_unreached_count = 0
                solver._preflight_pressure_interface_operator_storage = MethodType(
                    lambda _self, **_kwargs: None,
                    solver,
                )

                self._assert_missing_or_stale_graph_failure(
                    lambda: solver._solve_pressure_poisson_fv_cg(
                        iterations=1,
                        rhs_scale=1.0,
                        pressure_outlet_zmin=False,
                        tolerance=1.0e-8,
                        pressure_nullspace_component_count=2,
                        pressure_interface_matrix_active=True,
                        pressure_components_use_operator_graph=True,
                        report_requested=False,
                    )
                )

    def test_exact_residual_revalidates_closed_neumann_graph_semantics(
        self,
    ) -> None:
        cases = (
            ("outlet-disconnected", 2, 2, 2),
            ("closed-Neumann", 2, 1, 2),
            ("closed-Neumann", 0, 2, 2),
            ("closed-Neumann", 1, 2, 2),
        )
        for context, source_count, solve_count, requested_count in cases:
            with self.subTest(
                context=context,
                source_count=source_count,
                solve_count=solve_count,
                requested_count=requested_count,
            ):
                solver = self._solver_with_graph_metadata(
                    valid=True,
                    context=context,
                    source_count=source_count,
                    solve_count=solve_count,
                )
                self._assert_missing_or_stale_graph_failure(
                    lambda solver=solver, requested_count=requested_count: (
                        solver._confirm_pressure_poisson_fv_cg_exact_residual(
                            pressure_outlet_zmin=False,
                            tolerance=1.0e-8,
                            remove_nullspace_mean=True,
                            pressure_nullspace_component_count=requested_count,
                            pressure_components_use_operator_graph=True,
                        )
                    )
                )

    def test_exact_residual_revalidates_outlet_disconnected_graph_semantics(
        self,
    ) -> None:
        cases = (
            ("closed-Neumann", 2),
            ("outlet-disconnected", 0),
            ("outlet-disconnected", 3),
        )
        for context, published_source_count in cases:
            with self.subTest(
                context=context,
                published_source_count=published_source_count,
            ):
                solver = self._solver_with_graph_metadata(
                    valid=True,
                    context=context,
                    source_count=published_source_count,
                    solve_count=1,
                )
                solver._hibm_pressure_unreached_count = 8
                solver._hibm_pressure_unreached_component_count = 2
                self._assert_missing_or_stale_graph_failure(
                    lambda solver=solver: (
                        solver._confirm_pressure_poisson_fv_cg_exact_residual(
                            pressure_outlet_zmin=True,
                            tolerance=1.0e-8,
                            remove_nullspace_mean=True,
                            pressure_nullspace_component_count=0,
                            pressure_components_use_operator_graph=True,
                        )
                    )
                )

    def test_pre_projection_topology_commit_cannot_leave_old_graph_consumable(
        self,
    ) -> None:
        solver = self._solver_with_graph_metadata(
            valid=True,
            context="closed-Neumann",
            source_count=1,
            solve_count=1,
        )
        solver._mark_sphere_kernel = lambda *_args: None

        class _TopologyMutatingProjector:
            def prepare_projection_transaction(self, **_kwargs: object) -> None:
                return None

            def solve_projection_transaction(self) -> None:
                return None

            def commit_projection_transaction(self):
                solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.1)
                return {"prepared": True, "converged": True, "committed": True}

        report = _run_pre_projection_velocity_transaction(
            pre_projection_velocity_projector=_TopologyMutatingProjector(),
            fluid=solver,
            pressure_solve_context={"phase": "generic-callback"},
        )
        self.assertTrue(report["pre_projection_velocity_projector_committed"])
        self._assert_missing_or_stale_graph_failure(
            lambda: solver._confirm_pressure_poisson_fv_cg_exact_residual(
                pressure_outlet_zmin=False,
                tolerance=1.0e-8,
                remove_nullspace_mean=True,
                pressure_nullspace_component_count=1,
                pressure_components_use_operator_graph=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
