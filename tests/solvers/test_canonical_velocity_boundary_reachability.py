from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryReachabilityContracts(unittest.TestCase):
    """Host/static contracts for the component-face reachability consumer."""

    def test_reachability_has_its_own_opaque_capability_only(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertIn("reachability", capabilities)
        self.assertIsNotNone(capabilities["reachability"])
        self.assertEqual(
            len({id(value) for value in capabilities.values()}),
            len(capabilities),
        )
        self.assertIn("projection", capabilities)
        self.assertIn("multigrid", capabilities)
        self.assertIsNot(capabilities["reachability"], capabilities["projection"])
        self.assertIsNot(capabilities["multigrid"], capabilities["projection"])
        self.assertIsNot(capabilities["reachability"], capabilities["multigrid"])

    def test_prepare_refreshes_then_invalidates_then_registers_current_generation(
        self,
    ) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 17
        solver.velocity_dirichlet_component_ledger_sealed = False
        solver._velocity_dirichlet_component_ledger_consumer_generations = {}
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {}
        solver.last_hibm_reachability_valid = True
        solver.last_hibm_reachability_revision = 4
        solver.hibm_reachability_revision = 4
        solver._hibm_reachability_checksum = (1, 2, 3, 4)
        events: list[str] = []

        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask = (
            lambda: events.append("refresh") or (1, 1)
        )
        invalidate = (
            CartesianFluidSolver._invalidate_hibm_pressure_reachability.__get__(
                solver,
                CartesianFluidSolver,
            )
        )
        register = CartesianFluidSolver._register_velocity_dirichlet_component_ledger_consumer_generation.__get__(
            solver,
            CartesianFluidSolver,
        )

        def invalidate_with_trace() -> None:
            events.append("invalidate")
            invalidate()

        def register_with_trace(consumer: str, *, capability: object) -> None:
            events.append("register")
            register(consumer, capability=capability)

        solver._invalidate_hibm_pressure_reachability = invalidate_with_trace
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            register_with_trace
        )

        CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_reachability(
            solver
        )

        self.assertEqual(events, ["refresh", "invalidate", "register"])
        self.assertFalse(solver.last_hibm_reachability_valid)
        self.assertIsNone(solver._hibm_reachability_checksum)
        self.assertEqual(solver.hibm_reachability_revision, 5)
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_generations[
                "reachability"
            ],
            17,
        )
        self.assertIs(
            solver._velocity_dirichlet_component_ledger_consumer_capabilities[
                "reachability"
            ],
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES[
                "reachability"
            ],
        )

    def test_prepare_is_a_strict_noop_for_legacy_authority(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "legacy"

        def unexpected_call(*_args, **_kwargs):
            raise AssertionError("legacy prepare touched canonical reachability state")

        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask = (
            unexpected_call
        )
        solver._invalidate_hibm_pressure_reachability = unexpected_call
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            unexpected_call
        )

        result = (
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_reachability(
                solver
            )
        )

        self.assertIsNone(result)

    def test_unsealed_canonical_state_fails_before_barrier_checksum_or_flood(
        self,
    ) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.last_hibm_reachability_valid = True
        solver._hibm_reachability_checksum = (11, 12, 13, 14)

        def reject_unsealed() -> None:
            raise RuntimeError("expected sealed reachability guard")

        def unexpected_call(*_args, **_kwargs):
            raise AssertionError("canonical reachability touched a kernel before seal")

        solver._require_velocity_dirichlet_component_ledger_sealed = reject_unsealed
        solver._prepare_hibm_pressure_reachability_barrier = unexpected_call
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask = (
            unexpected_call
        )
        solver._hibm_reachability_pattern_checksum = unexpected_call
        solver._init_hibm_pressure_outlet_reachable_kernel = unexpected_call

        with self.assertRaisesRegex(
            RuntimeError,
            "expected sealed reachability guard",
        ):
            CartesianFluidSolver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                solver,
                pressure_outlet_zmin=True,
            )

    def test_guard_precedes_every_reachability_kernel_in_host_orchestration(
        self,
    ) -> None:
        source = inspect.getsource(
            CartesianFluidSolver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells
        )
        guard_index = source.index(
            "_require_velocity_dirichlet_component_ledger_sealed"
        )
        for call_name in (
            "_prepare_hibm_pressure_reachability_barrier",
            "_hibm_reachability_pattern_checksum",
            "_init_hibm_pressure_outlet_reachable_kernel",
            "_expand_and_commit_hibm_pressure_outlet_reachable_kernel",
            "_count_hibm_pressure_outlet_unreached_cells_kernel",
        ):
            with self.subTest(call=call_name):
                self.assertLess(guard_index, source.index(call_name))

    def test_single_z_claim_only_blocks_z_connectivity(self) -> None:
        connected_source = inspect.getsource(
            CartesianFluidSolver._pressure_cells_connected
        )
        cleanup_source = inspect.getsource(
            CartesianFluidSolver._velocity_dirichlet_pressure_barrier
        )

        for axis_bit in ("& 1", "& 2", "& 4"):
            with self.subTest(axis_bit=axis_bit):
                self.assertIn(axis_bit, connected_source)
        self.assertNotIn(
            "_velocity_dirichlet_pressure_barrier",
            connected_source,
        )
        self.assertIn("Scalar protection predicate for cleanup", cleanup_source)
        self.assertIn("_pressure_cells_connected", cleanup_source)


if __name__ == "__main__":
    unittest.main()
