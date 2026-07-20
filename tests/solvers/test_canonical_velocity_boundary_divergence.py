from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryDivergenceContracts(unittest.TestCase):
    """Host/static contracts for the axis-local divergence consumer."""

    def test_divergence_has_a_distinct_opaque_capability(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertIn("divergence", capabilities)
        self.assertIsNotNone(capabilities["divergence"])
        for other in ("apply", "reference", "snapshot", "fv_operator", "gradient"):
            with self.subTest(other=other):
                self.assertIsNot(capabilities["divergence"], capabilities[other])

    def test_prepare_path_registers_only_the_canonical_current_generation(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_divergence
        )

        self.assertIn('velocity_dirichlet_boundary_authority != "canonical"', source)
        self.assertIn(
            "_register_velocity_dirichlet_component_ledger_consumer_generation",
            source,
        )
        self.assertIn('"divergence"', source)
        self.assertIn(
            "_VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES",
            source,
        )

    def test_external_exact_face_predicate_is_axis_local_under_canonical_authority(
        self,
    ) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._velocity_dirichlet_external_exact_face_at
        )
        canonical_branch, legacy_branch = source.split("else:", 1)

        self.assertIn("canonical_authority", source)
        self.assertIn("1 << component_axis", source)
        self.assertIn(
            "velocity_dirichlet_boundary_active_component_mask", canonical_branch
        )
        self.assertIn(
            "velocity_dirichlet_boundary_external_exact_component_mask",
            canonical_branch,
        )
        self.assertIn(
            "velocity_dirichlet_boundary_owned_component_mask", canonical_branch
        )
        self.assertNotIn("velocity_dirichlet_boundary_active[", canonical_branch)
        self.assertNotIn("velocity_dirichlet_boundary_owned_row[", canonical_branch)
        self.assertIn("velocity_dirichlet_boundary_active[", legacy_branch)
        self.assertIn("velocity_dirichlet_boundary_owned_row[", legacy_branch)

    def test_zmax_counter_and_topology_resolver_take_explicit_authority(self) -> None:
        counter_source = inspect.getsource(
            CartesianFluidSolver._count_external_exact_zmax_normal_rows_kernel
        )
        resolver_source = inspect.getsource(
            CartesianFluidSolver._resolve_velocity_inlet_zmax_topology_mode
        )

        self.assertIn("canonical_authority", counter_source)
        self.assertIn("_velocity_dirichlet_external_exact_face_at", counter_source)
        self.assertIn(", 2, canonical_authority", " ".join(counter_source.split()))
        self.assertIn("canonical_authority", resolver_source)
        self.assertIn(
            "_count_external_exact_zmax_normal_rows_kernel(canonical_authority)",
            " ".join(resolver_source.split()),
        )

    def test_divergence_kernel_uses_the_predicate_for_each_face_axis(self) -> None:
        source = inspect.getsource(CartesianFluidSolver._compute_divergence_kernel)
        compact = " ".join(source.split())

        self.assertIn("canonical_authority", source)
        self.assertGreaterEqual(
            source.count("_velocity_dirichlet_external_exact_face_at"), 3
        )
        for axis in range(3):
            with self.subTest(axis=axis):
                self.assertIn(f", {axis}, canonical_authority", compact)

    def test_public_and_project_dispatch_pass_explicit_authority(self) -> None:
        public_source = inspect.getsource(CartesianFluidSolver.compute_divergence)
        dispatch_source = inspect.getsource(
            CartesianFluidSolver._compute_divergence_with_topology_mode
        )
        project_source = inspect.getsource(CartesianFluidSolver.project)

        self.assertIn("_velocity_dirichlet_boundary_authority_code", public_source)
        self.assertIn("canonical_authority=canonical_authority", public_source)
        self.assertIn("canonical_authority", dispatch_source)
        self.assertIn("int(canonical_authority)", dispatch_source)
        self.assertIn("_velocity_dirichlet_boundary_authority_code", project_source)
        self.assertIn("canonical_authority=canonical_authority", project_source)

    def test_projection_umbrella_is_distinct_from_divergence_and_multigrid(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertIn("projection", capabilities)
        self.assertIn("multigrid", capabilities)
        self.assertIsNot(capabilities["divergence"], capabilities["projection"])
        self.assertIsNot(capabilities["multigrid"], capabilities["projection"])
        self.assertIsNot(capabilities["divergence"], capabilities["multigrid"])


if __name__ == "__main__":
    unittest.main()
