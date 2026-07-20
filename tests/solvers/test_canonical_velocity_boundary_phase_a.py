from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryPhaseAContracts(unittest.TestCase):
    """Static/host RED contracts for the first canonical-consumer migration."""

    def test_phase_a_consumers_have_distinct_opaque_capabilities(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )
        phase_a = ("apply", "reference", "no_slip")

        for consumer in phase_a:
            with self.subTest(consumer=consumer):
                self.assertIn(consumer, capabilities)
                self.assertIsNotNone(capabilities[consumer])
        self.assertEqual(
            len({id(capabilities[consumer]) for consumer in phase_a}),
            len(phase_a),
        )

    def test_prepare_paths_register_only_their_own_capability(self) -> None:
        prepare_methods = {
            "apply": "prepare_velocity_dirichlet_component_ledger_apply",
            "reference": "prepare_velocity_dirichlet_component_ledger_reference",
            "no_slip": "prepare_hibm_no_slip_component_face_valid_mask",
        }

        for consumer, method_name in prepare_methods.items():
            with self.subTest(consumer=consumer):
                source = inspect.getsource(getattr(CartesianFluidSolver, method_name))
                self.assertIn(
                    "_register_velocity_dirichlet_component_ledger_consumer_generation",
                    source,
                )
                self.assertIn(f'"{consumer}"', source)
                self.assertIn(
                    "_VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES",
                    source,
                )

    def test_canonical_apply_is_componentwise_and_does_not_broadcast_active_row(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._apply_canonical_velocity_dirichlet_boundary_rows_kernel
        )

        self.assertIn("velocity_dirichlet_boundary_active_component_mask", source)
        self.assertIn(
            "velocity_dirichlet_boundary_component_enforcement_weight",
            source,
        )
        self.assertIn("velocity_dirichlet_boundary_hard_fixed_component_mask", source)
        self.assertIn("for axis in ti.static(range(3))", source)
        self.assertIn("active_mask & (1 << axis)", source)
        self.assertNotIn("velocity_dirichlet_boundary_active[", source)
        self.assertNotIn("velocity_dirichlet_boundary_projection_weight[", source)

    def test_public_and_project_apply_share_the_authority_dispatch(self) -> None:
        dispatch_name = "_apply_velocity_dirichlet_boundary_rows_dispatch"
        public_source = inspect.getsource(
            CartesianFluidSolver.apply_velocity_dirichlet_boundary_rows
        )
        project_source = inspect.getsource(CartesianFluidSolver.project)

        self.assertIn(dispatch_name, public_source)
        self.assertIn(dispatch_name, project_source)
        self.assertNotIn(
            "self._apply_velocity_dirichlet_boundary_rows_kernel(",
            public_source,
        )

    def test_canonical_reference_mirrors_all_eight_fields_and_identity(self) -> None:
        canonical_fields = (
            "active_component_mask",
            "value_mps",
            "pressure_mobility",
            "component_enforcement_weight",
            "component_region_id",
            "hard_fixed_component_mask",
            "external_exact_component_mask",
            "owned_component_mask",
        )
        init_source = inspect.getsource(CartesianFluidSolver.__init__)
        capture_source = inspect.getsource(
            CartesianFluidSolver._capture_canonical_velocity_dirichlet_boundary_ledger_reference_kernel
        )
        mismatch_source = inspect.getsource(
            CartesianFluidSolver._count_canonical_velocity_dirichlet_boundary_ledger_mismatch_rows_kernel
        )
        host_capture_source = inspect.getsource(
            CartesianFluidSolver.capture_velocity_dirichlet_boundary_ledger_reference
        )
        host_mismatch_source = inspect.getsource(
            CartesianFluidSolver.velocity_dirichlet_boundary_ledger_mismatch_rows
        )
        host_comparison_source = inspect.getsource(
            CartesianFluidSolver.velocity_dirichlet_boundary_ledger_comparison
        )

        for suffix in canonical_fields:
            with self.subTest(field=suffix):
                reference_name = (
                    "_velocity_dirichlet_component_ledger_reference_" + suffix
                )
                self.assertIn(reference_name, init_source)
                self.assertIn(reference_name, capture_source)
                self.assertIn(reference_name, mismatch_source)
        self.assertIn("_velocity_dirichlet_ledger_reference_authority", host_capture_source)
        self.assertIn(
            "_velocity_dirichlet_ledger_reference_component_generation",
            host_capture_source,
        )
        self.assertIn(
            "velocity_dirichlet_boundary_ledger_comparison",
            host_mismatch_source,
        )
        self.assertIn(
            "_velocity_dirichlet_ledger_reference_authority",
            host_comparison_source,
        )
        self.assertIn(
            "_velocity_dirichlet_ledger_reference_component_generation",
            host_comparison_source,
        )

    def test_no_slip_prepare_builds_component_view_but_read_requires_seal(self) -> None:
        prepare_source = inspect.getsource(
            CartesianFluidSolver.prepare_hibm_no_slip_component_face_valid_mask
        )
        read_source = inspect.getsource(
            CartesianFluidSolver.build_hibm_no_slip_component_face_valid_mask
        )
        obstacle_source = inspect.getsource(
            CartesianFluidSolver._build_canonical_hibm_no_slip_sampling_obstacle_kernel
        )

        self.assertIn(
            "_build_hibm_no_slip_component_face_valid_mask_kernel",
            prepare_source,
        )
        self.assertIn(
            "_register_velocity_dirichlet_component_ledger_consumer_generation",
            prepare_source,
        )
        self.assertIn(
            "_require_velocity_dirichlet_component_ledger_sealed",
            read_source,
        )
        self.assertIn("velocity_dirichlet_boundary_active_component_mask", obstacle_source)
        self.assertNotIn("velocity_dirichlet_boundary_active[", obstacle_source)

    def test_phase_a_registrations_cannot_pretend_global_seal_is_complete(self) -> None:
        required = CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertTrue(
            {
                "projection",
                "reachability",
                "fv_operator",
                "gradient",
                "multigrid",
            }.issubset(required)
        )
        self.assertEqual(set(required), set(capabilities))

        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 7
        solver.velocity_dirichlet_component_ledger_sealed = False
        solver.velocity_dirichlet_face_symmetric = 0
        phase_a = {"apply", "reference", "no_slip"}
        solver._velocity_dirichlet_component_ledger_consumer_generations = {
            consumer: 7 for consumer in phase_a
        }
        solver._velocity_dirichlet_component_ledger_consumer_capabilities = {
            consumer: capabilities[consumer] for consumer in phase_a
        }

        with self.assertRaisesRegex(RuntimeError, "consumer|canonical"):
            solver.seal_velocity_dirichlet_component_ledger()
        self.assertFalse(solver.velocity_dirichlet_component_ledger_sealed)


if __name__ == "__main__":
    unittest.main()
