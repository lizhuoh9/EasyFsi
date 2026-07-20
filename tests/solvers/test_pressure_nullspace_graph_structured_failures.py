from __future__ import annotations

import json
from types import MethodType
import unittest

from simulation_core.fluids.constants import HIBM_PRESSURE_COMPONENT_CAPACITY
from simulation_core.fluids.solver import (
    CartesianFluidSolver,
    PressureSolveConvergenceError,
)


class _HostScalar:
    """Minimal scalar-field double for graph host-control-flow contracts."""

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def __getitem__(self, _index: object) -> int:
        return self.value

    def __setitem__(self, _index: object, value: int) -> None:
        self.value = int(value)


class PressureNullspaceGraphStructuredFailureContracts(unittest.TestCase):
    @staticmethod
    def _bare_solver() -> CartesianFluidSolver:
        # These contracts exercise only host graph control flow.  Avoiding a
        # Taichi runtime keeps structural exception tests deterministic and fast.
        solver = object.__new__(CartesianFluidSolver)
        solver._pressure_outlet_nullspace_graph_valid = True
        solver._pressure_outlet_nullspace_source_component_count = 19
        solver._pressure_outlet_nullspace_component_count = 17
        solver._pressure_outlet_nullspace_graph_context = "stale"
        return solver

    def _assert_structured_graph_failure(
        self,
        *,
        solver: CartesianFluidSolver,
        physical_component_count: int,
        labels_converged: bool,
        component_overflow: bool,
        context: str,
        expected_reason: str,
    ) -> None:
        with self.assertRaises(PressureSolveConvergenceError) as caught:
            solver._prepare_pressure_nullspace_component_graph(
                physical_component_count=physical_component_count,
                labels_converged=labels_converged,
                component_overflow=component_overflow,
                context=context,
            )

        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["stage"], "pressure_nullspace_component_graph")
        self.assertEqual(diagnostics["context"], context)
        self.assertEqual(diagnostics["reason"], expected_reason)
        self.assertEqual(diagnostics["count"], physical_component_count)
        self.assertEqual(
            diagnostics["capacity"],
            HIBM_PRESSURE_COMPONENT_CAPACITY,
        )
        self.assertEqual(diagnostics["labels_converged"], labels_converged)
        self.assertEqual(diagnostics["component_overflow"], component_overflow)
        json.dumps(diagnostics, allow_nan=False)
        self.assertFalse(solver._pressure_outlet_nullspace_graph_valid)
        self.assertEqual(solver._pressure_outlet_nullspace_source_component_count, 0)
        self.assertEqual(solver._pressure_outlet_nullspace_component_count, 0)
        self.assertEqual(solver._pressure_outlet_nullspace_graph_context, "")

    def test_graph_labels_failure_precedes_simultaneous_component_overflow(
        self,
    ) -> None:
        self._assert_structured_graph_failure(
            solver=self._bare_solver(),
            physical_component_count=0,
            labels_converged=False,
            component_overflow=True,
            context="closed-neumann-label-priority",
            expected_reason="component_labels_not_converged",
        )

    def test_graph_component_overflow_is_structured_after_converged_labels(
        self,
    ) -> None:
        self._assert_structured_graph_failure(
            solver=self._bare_solver(),
            physical_component_count=HIBM_PRESSURE_COMPONENT_CAPACITY,
            labels_converged=True,
            component_overflow=True,
            context="closed-neumann-overflow",
            expected_reason="component_capacity_overflow",
        )

    def test_graph_out_of_range_component_count_is_structured(self) -> None:
        for component_count in (-1, HIBM_PRESSURE_COMPONENT_CAPACITY + 1):
            with self.subTest(component_count=component_count):
                self._assert_structured_graph_failure(
                    solver=self._bare_solver(),
                    physical_component_count=component_count,
                    labels_converged=True,
                    component_overflow=False,
                    context="closed-neumann-count-range",
                    expected_reason="component_count_out_of_range",
                )

    def _solver_at_operator_graph_stage(
        self,
        *,
        failure_stage: str,
    ) -> CartesianFluidSolver:
        solver = self._bare_solver()
        solver.pressure_solve_component_invalid = _HostScalar(0)
        solver.pressure_solve_component_count = _HostScalar(0)
        solver.reduction_count = _HostScalar(0)
        solver._preflight_pressure_interface_operator_storage = MethodType(
            lambda _self, **_kwargs: None,
            solver,
        )
        solver._init_pressure_outlet_nullspace_component_graph_kernel = MethodType(
            lambda _self, _count: None,
            solver,
        )
        solver._accumulate_pressure_outlet_component_diagonal_kernel = MethodType(
            lambda _self, _count: None,
            solver,
        )

        def accumulate_edges(_self: CartesianFluidSolver, _count: int) -> None:
            if failure_stage == "interface_graph_invalid":
                _self.pressure_solve_component_invalid[None] = 1

        solver._accumulate_pressure_outlet_interface_edges_kernel = MethodType(
            accumulate_edges,
            solver,
        )

        def relax_union(_self: CartesianFluidSolver, _count: int) -> None:
            _self.reduction_count[None] = int(
                failure_stage == "component_union_not_converged"
            )

        solver._relax_pressure_outlet_component_union_kernel = MethodType(
            relax_union,
            solver,
        )

        def finalize_roots(_self: CartesianFluidSolver, _count: int) -> None:
            if failure_stage == "diagonal_provenance_mismatch":
                _self.pressure_solve_component_invalid[None] = 1

        solver._finalize_pressure_outlet_component_roots_kernel = MethodType(
            finalize_roots,
            solver,
        )
        return solver

    def test_operator_graph_failures_use_one_structured_exception_contract(
        self,
    ) -> None:
        for reason in (
            "interface_graph_invalid",
            "component_union_not_converged",
            "diagonal_provenance_mismatch",
        ):
            with self.subTest(reason=reason):
                self._assert_structured_graph_failure(
                    solver=self._solver_at_operator_graph_stage(failure_stage=reason),
                    physical_component_count=1,
                    labels_converged=True,
                    component_overflow=False,
                    context="closed-neumann-operator-graph",
                    expected_reason=reason,
                )

    def test_outlet_exact_graph_capacity_gate_uses_final_unanchored_roots(
        self,
    ) -> None:
        solver = self._bare_solver()
        solver.nx = 1
        solver.ny = 1
        solver.nz = 1
        solver._hibm_pressure_unreached_component_count = (
            HIBM_PRESSURE_COMPONENT_CAPACITY
        )
        solver.last_hibm_pressure_unreached_component_raw_count = (
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1
        )
        solver.last_hibm_pressure_component_labels_converged = True
        solver.last_hibm_pressure_unreached_component_overflow = True
        solver.pressure_solve_component_invalid = _HostScalar(0)
        solver.pressure_solve_component_count = _HostScalar(
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1
        )
        solver.pressure_outlet_operator_raw_component_count = _HostScalar(
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1
        )
        solver._preflight_pressure_interface_operator_storage = MethodType(
            lambda _self, **_kwargs: None,
            solver,
        )
        solver._init_pressure_outlet_operator_component_labels_kernel = MethodType(
            lambda _self: None,
            solver,
        )
        solver._propagate_pressure_outlet_operator_labels_batch = MethodType(
            lambda _self: 0,
            solver,
        )
        solver._accumulate_pressure_outlet_operator_raw_components_kernel = (
            MethodType(lambda _self: None, solver)
        )
        solver._subtract_pressure_outlet_operator_interface_edges_kernel = (
            MethodType(lambda _self: None, solver)
        )
        solver._finalize_pressure_outlet_operator_raw_roots_kernel = MethodType(
            lambda _self: None,
            solver,
        )
        solver._publish_pressure_outlet_operator_compact_labels_kernel = MethodType(
            lambda _self, _count: None,
            solver,
        )

        with self.assertRaises(PressureSolveConvergenceError) as caught:
            solver._prepare_pressure_outlet_nullspace_component_graph()

        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["stage"], "pressure_nullspace_component_graph")
        self.assertEqual(diagnostics["context"], "outlet-disconnected")
        self.assertEqual(diagnostics["reason"], "component_capacity_overflow")
        self.assertEqual(
            diagnostics["count"],
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1,
        )
        self.assertEqual(
            diagnostics["source_physical_component_count"],
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1,
        )
        self.assertFalse(solver._pressure_outlet_nullspace_graph_valid)


if __name__ == "__main__":
    unittest.main()
