from __future__ import annotations

import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryProjectionContracts(unittest.TestCase):
    """Host contracts for projection as the last solver-consumer umbrella."""

    def test_projection_capability_is_opaque_and_distinct(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertIn("projection", capabilities)
        self.assertIsNotNone(capabilities["projection"])
        self.assertEqual(
            len({id(value) for value in capabilities.values()}),
            len(capabilities),
        )

    def test_projection_prepare_requires_every_current_generation_dependency(
        self,
    ) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 41
        solver._velocity_dirichlet_component_ledger_consumer_generations = {}
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {}
        registered: list[str] = []
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            lambda consumer, *, capability: registered.append(consumer)
        )

        with self.assertRaisesRegex(RuntimeError, "projection dependencies"):
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_projection(
                solver
            )

        self.assertEqual(registered, [])

    def test_projection_registers_only_after_all_dependencies_match(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 43
        dependencies = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_PROJECTION_DEPENDENCIES
        )
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            name: 43 for name in dependencies
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            name: capabilities[name] for name in dependencies
        }
        registered: list[tuple[str, object]] = []
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            lambda consumer, *, capability: registered.append((consumer, capability))
        )

        CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_projection(
            solver
        )

        self.assertEqual(
            registered,
            [("projection", capabilities["projection"])],
        )

    def test_projection_prepare_is_strict_noop_for_legacy_authority(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "legacy"

        def unexpected(*_args, **_kwargs):
            raise AssertionError("legacy projection prepare touched canonical state")

        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            unexpected
        )
        result = (
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_projection(
                solver
            )
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
