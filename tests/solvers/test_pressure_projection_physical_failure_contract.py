from __future__ import annotations

import inspect
import json
from types import MethodType
import unittest

from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.fluids.constants import HIBM_PRESSURE_COMPONENT_CAPACITY
from simulation_core.fluids.solver import PressureSolveConvergenceError
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig


_STAGE_TIMING = {
    "canonical_ledger_build": 1.0,
    "canonical_prepare_seal": 2.0,
    "pressure_reachability_flood": 3.0,
    "pressure_neumann_assembly": 4.0,
}


class PressureProjectionPhysicalFailureContracts(unittest.TestCase):
    def _solver_with_invalid_component_labels(
        self,
        *,
        labels_converged: bool,
        component_overflow: bool,
    ) -> tuple[CartesianFluidSolver, list[str]]:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.last_hibm_pressure_unreached_cell_count = 37
        solver.last_hibm_pressure_unreached_raw_cell_count = 41
        solver.last_hibm_pressure_unreached_component_count = (
            HIBM_PRESSURE_COMPONENT_CAPACITY
        )
        solver.last_hibm_pressure_unreached_component_raw_count = (
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1
        )
        solver.last_hibm_pressure_unreached_component_overflow = bool(
            component_overflow
        )
        solver.last_hibm_pressure_component_labels_converged = bool(
            labels_converged
        )
        solver.last_hibm_pressure_reachability_converged = True
        solver.last_hibm_pressure_reachability_frontier_overflow = False
        solver.last_hibm_pressure_reachability_sweeps = 13
        solver.last_hibm_unreached_incompatible_component_count = 99
        solver.last_hibm_unreached_component_rhs_mean_max_abs = 123.0
        solver.last_hibm_unreached_component_rhs_integral_max_abs = 456.0
        solver._hibm_pressure_unreached_count = 37
        solver._hibm_pressure_unreached_component_count = (
            HIBM_PRESSURE_COMPONENT_CAPACITY
        )

        solver._require_prepared_hibm_pressure_reachability_current = MethodType(
            lambda _self: None,
            solver,
        )
        solver.convert_hibm_row_cloud_orphan_components = MethodType(
            lambda _self, **_kwargs: 0,
            solver,
        )
        pressure_solve_calls: list[str] = []

        def forbidden_pressure_solve(_self, **_kwargs) -> None:
            pressure_solve_calls.append("pressure_solve")
            raise AssertionError(
                "pressure solve ran after structural physical preflight failed"
            )

        solver._solve_pressure_poisson_with_solver = MethodType(
            forbidden_pressure_solve,
            solver,
        )
        return solver, pressure_solve_calls

    def _assert_structured_preflight_failure(
        self,
        *,
        labels_converged: bool,
        expected_reason: str,
    ) -> None:
        solver, pressure_solve_calls = self._solver_with_invalid_component_labels(
            labels_converged=labels_converged,
            component_overflow=True,
        )

        with self.assertRaises(PressureSolveConvergenceError) as caught:
            solver.project(
                iterations=8,
                pressure_outlet_zmin=True,
                pressure_solver="fv_cg",
                pressure_solve_failure_policy="raise",
                hibm_pressure_reachability_prepared=True,
                pressure_solve_context={
                    "phase": "preflow",
                    "step_index_local": 0,
                    "step_index_global": 0,
                    "hibm_sharp_marker_boundary_stage_wall_time_s": dict(
                        _STAGE_TIMING
                    ),
                },
            )

        diagnostics = caught.exception.diagnostics
        self.assertEqual(pressure_solve_calls, [])
        self.assertEqual(diagnostics["stage"], "pressure_projection_physical_gate")
        self.assertEqual(diagnostics["reason"], expected_reason)
        self.assertEqual(diagnostics["count"], HIBM_PRESSURE_COMPONENT_CAPACITY)
        self.assertEqual(diagnostics["capacity"], HIBM_PRESSURE_COMPONENT_CAPACITY)
        self.assertEqual(
            diagnostics["failure_kind"],
            "pressure_projection_physical_gate",
        )
        self.assertEqual(
            diagnostics["pressure_projection_physical_failure_reason"],
            expected_reason,
        )
        self.assertFalse(diagnostics["correction_applied"])
        self.assertEqual(diagnostics["hibm_pressure_unreached_cell_count"], 37)
        self.assertEqual(
            diagnostics["hibm_pressure_unreached_raw_cell_count"],
            41,
        )
        self.assertEqual(
            diagnostics["cg_unreached_component_count"],
            HIBM_PRESSURE_COMPONENT_CAPACITY,
        )
        self.assertEqual(
            diagnostics["cg_unreached_component_raw_count"],
            HIBM_PRESSURE_COMPONENT_CAPACITY + 1,
        )
        self.assertEqual(
            diagnostics["cg_unreached_component_capacity"],
            HIBM_PRESSURE_COMPONENT_CAPACITY,
        )
        self.assertTrue(diagnostics["cg_unreached_component_overflow"])
        self.assertEqual(
            diagnostics["hibm_pressure_component_labels_converged"],
            labels_converged,
        )
        self.assertTrue(
            diagnostics["hibm_pressure_reachability_converged"]
        )
        self.assertFalse(
            diagnostics["hibm_pressure_reachability_frontier_overflow"]
        )
        self.assertEqual(
            diagnostics["hibm_pressure_reachability_frontier_levels"],
            13,
        )
        self.assertFalse(
            diagnostics["hibm_unreached_rhs_compatibility_measured"]
        )
        self.assertEqual(
            diagnostics["hibm_unreached_incompatible_component_count"],
            0,
        )
        self.assertEqual(
            diagnostics["hibm_unreached_component_rhs_mean_max_abs"],
            0.0,
        )
        self.assertEqual(
            diagnostics["hibm_unreached_component_rhs_integral_max_abs"],
            0.0,
        )
        self.assertEqual(
            diagnostics["context"][
                "hibm_sharp_marker_boundary_stage_wall_time_s"
            ],
            _STAGE_TIMING,
        )
        json.dumps(diagnostics, allow_nan=False)

    def test_labels_not_converged_precedes_overflow_and_pressure_solve(self) -> None:
        self._assert_structured_preflight_failure(
            labels_converged=False,
            expected_reason="unreached_component_labels_not_converged",
        )

    def test_component_overflow_precedes_pressure_solve_with_raw_count(self) -> None:
        self._assert_structured_preflight_failure(
            labels_converged=True,
            expected_reason="unreached_component_label_overflow",
        )

    def test_active_exact_operator_reclassifies_physical_overflow_before_structural_gate(
        self,
    ) -> None:
        expected_policy = {
            0: "pressure_outlet_dirichlet_operator_anchored",
            1: "outlet_disconnected_fv_cg_operator_componentwise_zero_mean",
        }
        for final_component_count in (0, 1):
            with self.subTest(final_component_count=final_component_count):
                solver, _ = self._solver_with_invalid_component_labels(
                    labels_converged=True,
                    component_overflow=True,
                )
                solver.pressure_interface_matrix_diagonal[1, 1, 1] = 1.0
                operator_graph_calls: list[str] = []
                pressure_solve_calls: list[dict[str, object]] = []

                def prepare_operator_graph(_self) -> tuple[int, int]:
                    operator_graph_calls.append("prepare_operator_graph")
                    source_count = HIBM_PRESSURE_COMPONENT_CAPACITY + 1
                    _self._pressure_outlet_nullspace_source_component_count = (
                        source_count
                    )
                    _self._pressure_outlet_operator_component_count = (
                        final_component_count
                    )
                    _self._pressure_outlet_nullspace_component_count = (
                        final_component_count
                    )
                    _self._pressure_outlet_nullspace_graph_context = (
                        "outlet-disconnected"
                    )
                    _self._pressure_outlet_nullspace_graph_valid = True
                    return source_count, final_component_count

                def record_pressure_solve(_self, **kwargs) -> None:
                    pressure_solve_calls.append(dict(kwargs))
                    _self.last_cg_converged = True
                    _self.last_hibm_unreached_incompatible_component_count = 0
                    _self.last_hibm_unreached_component_rhs_mean_max_abs = 0.0
                    _self.last_hibm_unreached_component_rhs_integral_max_abs = 0.0

                solver._prepare_pressure_outlet_nullspace_component_graph = (
                    MethodType(prepare_operator_graph, solver)
                )
                solver._solve_pressure_poisson_with_solver = MethodType(
                    record_pressure_solve,
                    solver,
                )

                report = solver.project(
                    iterations=8,
                    pressure_outlet_zmin=True,
                    pressure_solver="fv_cg",
                    pressure_solve_failure_policy="raise",
                    hibm_pressure_reachability_prepared=True,
                    pressure_solve_context={
                        "phase": "preflow",
                        "step_index_local": 0,
                        "step_index_global": 0,
                        "hibm_sharp_marker_boundary_stage_wall_time_s": dict(
                            _STAGE_TIMING
                        ),
                    },
                )

                self.assertEqual(operator_graph_calls, ["prepare_operator_graph"])
                self.assertEqual(len(pressure_solve_calls), 1)
                solve_kwargs = pressure_solve_calls[0]
                self.assertIs(solve_kwargs["pressure_interface_matrix_active"], True)
                self.assertIs(
                    solve_kwargs["pressure_components_use_operator_graph"],
                    True,
                )
                self.assertEqual(
                    solve_kwargs["pressure_nullspace_component_count"],
                    final_component_count,
                )
                self.assertIs(solve_kwargs["remove_nullspace_mean"], False)
                self.assertEqual(
                    report["pressure_nullspace_policy"],
                    expected_policy[final_component_count],
                )
                self.assertEqual(
                    report["pressure_nullspace_component_count"],
                    final_component_count,
                )
                self.assertIs(
                    report["pressure_nullspace_componentwise_projection_applied"],
                    final_component_count == 1,
                )

    def test_physical_preflight_follows_final_topology_and_precedes_cg(self) -> None:
        source = inspect.getsource(CartesianFluidSolver.project)
        pressure_solve = source.index("self._solve_pressure_poisson_with_solver(")
        final_topology_refresh = source.rindex(
            "self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()",
            0,
            pressure_solve,
        )
        physical_gate = source.index(
            "handle_projection_physical_failure()",
            final_topology_refresh,
        )

        self.assertLess(final_topology_refresh, physical_gate)
        self.assertLess(physical_gate, pressure_solve)


if __name__ == "__main__":
    unittest.main()
