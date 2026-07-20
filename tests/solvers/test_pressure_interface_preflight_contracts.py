from __future__ import annotations

import unittest
from types import MethodType

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids.solver import PressureSolveConvergenceError


class _PressureSolveEntered(RuntimeError):
    """Sentinel proving an invalid operator reached the pressure solver."""


class PressureInterfacePreflightContracts(unittest.TestCase):
    """Fail-closed contracts for interface data outside any nullspace pocket."""

    @staticmethod
    def _solver_with_active_interface() -> CartesianFluidSolver:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        # Keep the interface policy active independently of the invalid value
        # under test.  The z-min outlet remains connected to every fluid cell,
        # so no unreached-component/nullspace graph will be built.
        solver.pressure_interface_matrix_diagonal[1, 1, 1] = 1.0
        return solver

    @staticmethod
    def _install_invalid_case(
        solver: CartesianFluidSolver,
        case: str,
    ) -> None:
        if case == "negative_row_count":
            solver.pressure_interface_row_count[None] = -1
        elif case == "over_capacity_row_count":
            solver.pressure_interface_row_count[None] = (
                int(solver.pressure_interface_row_capacity) + 1
            )
        elif case == "nan_diagonal":
            solver.pressure_interface_matrix_diagonal[2, 2, 2] = float("nan")
        elif case == "inf_diagonal":
            solver.pressure_interface_matrix_diagonal[2, 2, 2] = float("inf")
        elif case == "negative_diagonal":
            solver.pressure_interface_matrix_diagonal[2, 2, 2] = -1.0
        elif case == "nan_rhs":
            solver.pressure_interface_matrix_rhs[2, 2, 2] = float("nan")
        elif case == "inf_rhs":
            solver.pressure_interface_matrix_rhs[2, 2, 2] = float("inf")
        elif case == "invalid_row_endpoint":
            solver.pressure_interface_row_count[None] = 1
            solver.pressure_interface_row_owner[0] = (-1, 1, 1)
            solver.pressure_interface_row_neighbor[0] = (2, 2, 2)
            solver.pressure_interface_row_transmissibility[0] = 1.0
        elif case == "self_row_endpoint":
            solver.pressure_interface_row_count[None] = 1
            solver.pressure_interface_row_owner[0] = (2, 2, 2)
            solver.pressure_interface_row_neighbor[0] = (2, 2, 2)
            solver.pressure_interface_row_transmissibility[0] = 1.0
        elif case in {
            "nan_row_transmissibility",
            "inf_row_transmissibility",
            "zero_row_transmissibility",
            "negative_row_transmissibility",
        }:
            solver.pressure_interface_row_count[None] = 1
            solver.pressure_interface_row_owner[0] = (1, 1, 1)
            solver.pressure_interface_row_neighbor[0] = (2, 2, 2)
            values = {
                "nan_row_transmissibility": float("nan"),
                "inf_row_transmissibility": float("inf"),
                "zero_row_transmissibility": 0.0,
                "negative_row_transmissibility": -1.0,
            }
            solver.pressure_interface_row_transmissibility[0] = values[case]
        elif case in {"negative_legacy_active", "over_capacity_legacy_active"}:
            slot_capacity = 1 + int(
                solver.pressure_interface_coupling_extra_coefficient.shape[-1]
            )
            solver.pressure_interface_coupling_active[2, 2, 2] = (
                -1
                if case == "negative_legacy_active"
                else slot_capacity + 1
            )
        else:  # pragma: no cover - protects the table itself
            raise AssertionError(f"unknown invalid interface case: {case}")

    def test_all_reachable_outlet_preflights_every_active_interface_input_before_solve(
        self,
    ) -> None:
        cases = (
            "negative_row_count",
            "over_capacity_row_count",
            "nan_diagonal",
            "inf_diagonal",
            "negative_diagonal",
            "nan_rhs",
            "inf_rhs",
            "invalid_row_endpoint",
            "self_row_endpoint",
            "nan_row_transmissibility",
            "inf_row_transmissibility",
            "zero_row_transmissibility",
            "negative_row_transmissibility",
            "negative_legacy_active",
            "over_capacity_legacy_active",
        )

        for case in cases:
            with self.subTest(case=case):
                solver = self._solver_with_active_interface()
                self._install_invalid_case(solver, case)
                self.assertEqual(
                    solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                        pressure_outlet_zmin=True,
                    ),
                    0,
                    "the contract must exercise the no-nullspace-graph path",
                )

                def forbidden_pressure_solve(_self, **_kwargs) -> None:
                    raise _PressureSolveEntered(
                        "invalid interface data reached pressure matvec/CG"
                    )

                solver._solve_pressure_poisson_with_solver = MethodType(
                    forbidden_pressure_solve,
                    solver,
                )
                try:
                    solver.project(
                        iterations=2,
                        pressure_outlet_zmin=True,
                        reset_pressure=True,
                        pressure_solver="fv_cg",
                        cg_preconditioner="jacobi",
                        pressure_solve_failure_policy="report",
                        read_report=False,
                    )
                except _PressureSolveEntered as exc:
                    self.fail(str(exc))
                except (RuntimeError, ValueError):
                    pass
                else:
                    self.fail(
                        f"{case} was not rejected before pressure matvec/CG"
                    )

    def test_all_reachable_empty_graph_preflight_rejects_locally_inconsistent_row(
        self,
    ) -> None:
        """A component-total balance may not hide an invalid cell-local row sum."""

        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        owner = (1, 1, 1)
        neighbor = (2, 1, 1)
        transmissibility = 1.0
        cell_volume_m3 = float(solver.dx * solver.dy * solver.dz)
        diagonal = solver.pressure_interface_matrix_diagonal.to_numpy()
        # Aggregate diagonal integral is 2T, exactly matching the two endpoint
        # incidences.  Locally the endpoint excesses are -T and +T, so the
        # matrix cannot be accepted as a conservative row.
        diagonal[owner] = 0.0
        diagonal[neighbor] = 2.0 * transmissibility / cell_volume_m3
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        solver.pressure_interface_row_count[None] = 1
        solver.pressure_interface_row_owner[0] = owner
        solver.pressure_interface_row_neighbor[0] = neighbor
        solver.pressure_interface_row_transmissibility[0] = transmissibility
        self.assertAlmostEqual(
            float((diagonal[owner] + diagonal[neighbor]) * cell_volume_m3),
            2.0 * transmissibility,
            delta=1.0e-14,
            msg="the fixture must be component-total balanced",
        )
        self.assertEqual(
            solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            0,
            "the fixture must exercise the all-outlet-reachable empty graph path",
        )

        context = "all-reachable-local-provenance"
        with self.assertRaises(PressureSolveConvergenceError) as caught:
            solver._prepare_pressure_nullspace_component_graph(
                physical_component_count=0,
                labels_converged=True,
                component_overflow=False,
                context=context,
            )

        diagnostics = caught.exception.diagnostics
        self.assertEqual(
            diagnostics.get("stage"),
            "pressure_interface_operator_preflight",
            "the generic storage preflight, not graph-root finalization, must reject it",
        )
        self.assertEqual(diagnostics.get("context"), f"{context}-graph")
        invalid_counts = diagnostics.get("invalid_counts", {})
        self.assertGreater(
            int(invalid_counts.get("local_diagonal_provenance", 0)),
            0,
        )
        self.assertFalse(bool(solver._pressure_outlet_nullspace_graph_valid))


if __name__ == "__main__":
    unittest.main()
