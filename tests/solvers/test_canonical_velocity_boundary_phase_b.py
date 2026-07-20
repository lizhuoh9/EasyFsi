from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryPhaseBContracts(unittest.TestCase):
    """Static contracts for fine-grid per-component pressure mobility."""

    @staticmethod
    def _canonical_weight(
        *, active_mask: int, hard_mask: int, mobility: tuple[float, float, float], axis: int
    ) -> float:
        bit = 1 << axis
        if hard_mask & bit:
            return 0.0
        if not (active_mask & bit):
            return 1.0
        return min(max(float(mobility[axis]), 0.0), 1.0)

    def test_single_z_claim_leaves_x_and_y_pressure_faces_open(self) -> None:
        weights = tuple(
            self._canonical_weight(
                active_mask=0b100,
                hard_mask=0,
                mobility=(0.0, 0.0, 0.25),
                axis=axis,
            )
            for axis in range(3)
        )
        self.assertEqual(weights, (1.0, 1.0, 0.25))

    def test_phase_b_consumers_have_distinct_opaque_capabilities(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )
        for consumer in ("fv_operator", "gradient"):
            with self.subTest(consumer=consumer):
                self.assertIn(consumer, capabilities)
                self.assertIsNotNone(capabilities[consumer])
        self.assertIsNot(capabilities["fv_operator"], capabilities["gradient"])

    def test_fv_and_gradient_use_the_same_component_weight_helper(self) -> None:
        helper_name = "_velocity_dirichlet_pressure_face_weight"
        fv_source = inspect.getsource(
            CartesianFluidSolver._fv_pressure_velocity_face_weight
        )
        gradient_source = inspect.getsource(
            CartesianFluidSolver._subtract_pressure_gradient_kernel
        )
        self.assertIn(helper_name, fv_source)
        self.assertIn(helper_name, gradient_source)

    def test_component_weight_contract_is_axis_local_and_legacy_branch_is_retained(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._velocity_dirichlet_pressure_face_weight
        )
        self.assertIn("canonical_active_component_mask", source)
        self.assertIn("canonical_pressure_mobility", source)
        self.assertIn("canonical_hard_fixed_component_mask", source)
        self.assertIn("1 << component_axis", source)
        self.assertIn("canonical_authority", source)
        self.assertIn("velocity_dirichlet_boundary_active", source)
        self.assertIn("velocity_dirichlet_boundary_projection_weight", source)

    def test_canonical_hard_refresh_does_not_read_scalar_active_row(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._velocity_dirichlet_pressure_effective_component_mask_at
        )
        canonical_branch = source.split("else:", 1)[0]
        self.assertIn("velocity_dirichlet_boundary_active_component_mask", canonical_branch)
        self.assertNotIn("velocity_dirichlet_boundary_active[", canonical_branch)

    def test_projection_umbrella_is_distinct_from_fv_operator_and_multigrid(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )
        self.assertIn("projection", capabilities)
        self.assertIn("multigrid", capabilities)
        self.assertIsNot(capabilities["fv_operator"], capabilities["projection"])
        self.assertIsNot(capabilities["multigrid"], capabilities["projection"])
        self.assertIsNot(capabilities["fv_operator"], capabilities["multigrid"])

    def test_fv_jacobi_dispatch_matches_kernel_signature(self) -> None:
        source = textwrap.dedent(
            inspect.getsource(CartesianFluidSolver._smooth_fv_pressure_fields)
        )
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_pressure_fv_jacobi_kernel"
        ]
        self.assertEqual(len(calls), 1)
        kernel_parameter_count = len(
            inspect.signature(
                CartesianFluidSolver._pressure_fv_jacobi_kernel
            ).parameters
        )
        self.assertEqual(len(calls[0].args), kernel_parameter_count - 1)


if __name__ == "__main__":
    unittest.main()
