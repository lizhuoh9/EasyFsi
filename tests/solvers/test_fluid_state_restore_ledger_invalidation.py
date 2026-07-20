from __future__ import annotations

import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


def _host_only_restorable_solver() -> tuple[CartesianFluidSolver, list[str]]:
    """Build a restore fixture without allocating fields or launching Taichi."""

    solver = object.__new__(CartesianFluidSolver)
    events: list[str] = []

    solver.hibm_external_obstacle_topology_revision = 12
    solver._restore_state_kernel = lambda: events.append(
        "restore_kernel:"
        + str(solver.hibm_external_obstacle_topology_revision)
    )
    solver._saved_hibm_dynamic_solid_volume_enabled = True
    solver._reset_hibm_pressure_unreached_component_distribution_stats = (
        lambda: events.append("reset_component_stats")
    )
    solver._pressure_outlet_nullspace_graph_valid = True
    solver._pressure_outlet_nullspace_source_component_count = 4
    solver._pressure_outlet_operator_component_count = 3
    solver._pressure_outlet_nullspace_component_count = 2
    solver._pressure_outlet_nullspace_graph_context = "stale-restore-graph"
    solver.last_hibm_reachability_valid = True
    solver.hibm_reachability_revision = 9
    solver.last_hibm_reachability_revision = 9
    solver._hibm_reachability_checksum = ("stale-restore",)

    invalidate_reachability = CartesianFluidSolver._invalidate_hibm_pressure_reachability

    def record_reachability_invalidation() -> None:
        events.append("invalidate_reachability")
        invalidate_reachability(solver)

    solver._invalidate_hibm_pressure_reachability = record_reachability_invalidation

    solver.velocity_dirichlet_boundary_authority = "canonical"
    solver.velocity_dirichlet_component_ledger_generation = 37
    solver.velocity_dirichlet_component_ledger_sealed = True
    solver._velocity_dirichlet_component_ledger_consumer_generations = {
        "apply": 37,
        "projection": 37,
    }
    solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
        "apply": object(),
        "projection": object(),
    }

    invalidate_ledger = (
        CartesianFluidSolver._invalidate_velocity_dirichlet_component_ledger
    )

    def record_ledger_invalidation() -> int:
        events.append("invalidate_ledger")
        return invalidate_ledger(solver)

    solver._invalidate_velocity_dirichlet_component_ledger = (
        record_ledger_invalidation
    )
    return solver, events


class FluidStateRestoreLedgerInvalidationContracts(unittest.TestCase):
    """Host-only contracts for restored canonical-ledger identity."""

    def test_restore_invalidates_stale_canonical_ledger_identity(self) -> None:
        solver, events = _host_only_restorable_solver()
        old_consumer_generations = (
            solver._velocity_dirichlet_component_ledger_consumer_generations
        )
        old_consumer_capabilities = (
            solver._velocity_dirichlet_component_ledger_consumer_capabilities
        )

        solver.restore_state()

        self.assertEqual(
            events[:3],
            ["invalidate_reachability", "invalidate_ledger", "restore_kernel:13"],
        )
        self.assertEqual(solver.hibm_external_obstacle_topology_revision, 13)
        self.assertEqual(solver.velocity_dirichlet_boundary_authority, "canonical")
        self.assertEqual(solver.velocity_dirichlet_component_ledger_generation, 38)
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_generations,
            {},
        )
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_capabilities,
            {},
        )
        self.assertIsNot(
            solver._velocity_dirichlet_component_ledger_consumer_generations,
            old_consumer_generations,
        )
        self.assertIsNot(
            solver._velocity_dirichlet_component_ledger_consumer_capabilities,
            old_consumer_capabilities,
        )
        self.assertFalse(solver.last_hibm_reachability_valid)
        self.assertIsNone(solver._hibm_reachability_checksum)

    def test_restore_invalidates_all_caches_before_failed_device_write(self) -> None:
        solver, events = _host_only_restorable_solver()
        observed_at_restore: list[tuple[int, bool, bool]] = []

        def fail_restore() -> None:
            observed_at_restore.append(
                (
                    int(solver.velocity_dirichlet_component_ledger_generation),
                    bool(solver.velocity_dirichlet_component_ledger_sealed),
                    bool(solver.last_hibm_reachability_valid),
                )
            )
            raise RuntimeError("injected restore kernel failure")

        solver._restore_state_kernel = fail_restore

        with self.assertRaisesRegex(RuntimeError, "injected restore kernel failure"):
            solver.restore_state()

        self.assertEqual(
            events[:2],
            ["invalidate_reachability", "invalidate_ledger"],
        )
        self.assertEqual(observed_at_restore, [(38, False, False)])
        self.assertFalse(solver._pressure_outlet_nullspace_graph_valid)
        self.assertFalse(solver.last_hibm_reachability_valid)
        self.assertEqual(
            solver._velocity_dirichlet_component_ledger_consumer_generations,
            {},
        )

    def test_directed_plane_writer_invalidates_before_first_device_write(
        self,
    ) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_component_ledger_generation = 19
        solver.velocity_dirichlet_component_ledger_sealed = True
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            "apply": 19
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            "apply": object()
        }
        solver._invalidate_hibm_pressure_reachability = lambda: None
        observed_at_first_write: list[tuple[int, bool]] = []

        def fail_first_device_write(*_args: object) -> None:
            observed_at_first_write.append(
                (
                    int(solver.velocity_dirichlet_component_ledger_generation),
                    bool(solver.velocity_dirichlet_component_ledger_sealed),
                )
            )
            raise RuntimeError("injected directed-plane device write failure")

        solver._refresh_external_velocity_boundary_x_face_uniform_kernel = (
            fail_first_device_write
        )
        solver._refresh_external_velocity_boundary_y_face_uniform_kernel = lambda *_: None
        solver._refresh_external_velocity_boundary_z_face_uniform_kernel = lambda *_: None

        with self.assertRaisesRegex(RuntimeError, "injected directed-plane"):
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=0,
                side_index=1,
                target_velocity_mps=(1.0, 2.0, 3.0),
                active_component_mask=0b111,
            )

        self.assertEqual(
            observed_at_first_write,
            [(20, False)],
            msg=(
                "the generic directed-plane transaction must invalidate its "
                "consumer generation before the first device write, so a "
                "partial kernel failure cannot leave an old seal reusable"
            ),
        )


if __name__ == "__main__":
    unittest.main()
