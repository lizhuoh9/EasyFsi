from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class CanonicalVelocityBoundaryMultigridContracts(unittest.TestCase):
    """Host/static contracts for the component-face multigrid consumer."""

    @staticmethod
    def _coarsen(
        *subfaces: tuple[float, bool, float, bool, bool, bool],
    ) -> tuple[bool, float, bool, bool, bool]:
        return CartesianFluidSolver._coarsen_canonical_velocity_boundary_face(
            tuple(subfaces)
        )

    def test_area_weighted_exact_face_weight_preserves_mixed_hard_open_face(
        self,
    ) -> None:
        # (area, active, mobility, hard, owned, external)
        result = self._coarsen(
            (1.0, True, 0.75, True, False, False),
            (3.0, False, 0.0, False, False, False),
        )

        # The hard quarter contributes weight zero and the open three quarters
        # contribute weight one.  A mixed face is soft with mobility 0.75; it
        # must not be closed by OR-ing the hard bit.
        self.assertEqual(result, (True, 0.75, False, False, False))

    def test_mobility_is_clamped_before_transverse_area_weighting(self) -> None:
        result = self._coarsen(
            (1.0, True, -3.0, False, False, False),
            (3.0, True, 2.0, False, False, False),
        )

        self.assertEqual(result, (True, 0.75, False, False, False))

    def test_hard_is_promoted_only_when_every_valid_subface_is_hard(self) -> None:
        all_hard = self._coarsen(
            (1.0, True, 1.0, True, False, False),
            (2.0, True, 1.0, True, False, False),
        )
        mixed = self._coarsen(
            (1.0, True, 1.0, True, False, False),
            (2.0, True, 0.5, False, False, False),
        )

        self.assertEqual(all_hard, (True, 0.0, True, False, False))
        self.assertEqual(mixed, (True, 1.0 / 3.0, False, False, False))

    def test_owned_requires_complete_active_owned_face_coverage(self) -> None:
        complete = self._coarsen(
            (1.0, True, 0.2, False, True, False),
            (1.0, True, 0.4, False, True, False),
        )
        incomplete = self._coarsen(
            (1.0, True, 0.2, False, True, False),
            (1.0, False, 0.0, False, False, False),
        )

        self.assertEqual(complete[0], True)
        self.assertAlmostEqual(complete[1], 0.3)
        self.assertEqual(complete[2:], (False, True, False))
        self.assertEqual(incomplete, (True, 0.6, False, False, False))

    def test_external_exact_requires_consistent_complete_provenance(self) -> None:
        complete = self._coarsen(
            (1.0, True, 1.0, True, False, True),
            (1.0, True, 1.0, True, False, True),
        )

        self.assertEqual(complete, (True, 0.0, True, False, True))
        with self.assertRaisesRegex(ValueError, "external exact provenance"):
            self._coarsen(
                (1.0, True, 1.0, True, False, True),
                (1.0, True, 1.0, True, False, False),
            )
        with self.assertRaisesRegex(ValueError, "owned and external"):
            self._coarsen(
                (1.0, True, 1.0, True, True, True),
                (1.0, True, 1.0, True, True, True),
            )

    def test_invalid_subface_contract_fails_closed(self) -> None:
        invalid_cases = (
            (),
            ((0.0, True, 0.5, False, False, False),),
            ((1.0, False, 0.5, True, False, False),),
            ((1.0, False, 0.5, False, True, False),),
            ((1.0, False, 0.5, False, False, True),),
        )
        for subfaces in invalid_cases:
            with self.subTest(subfaces=subfaces):
                with self.assertRaises(ValueError):
                    CartesianFluidSolver._coarsen_canonical_velocity_boundary_face(
                        subfaces
                    )

    def test_multigrid_and_projection_have_distinct_opaque_capabilities(self) -> None:
        capabilities = (
            CartesianFluidSolver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
        )

        self.assertIn("multigrid", capabilities)
        self.assertIsNotNone(capabilities["multigrid"])
        self.assertEqual(
            len({id(value) for value in capabilities.values()}),
            len(capabilities),
        )
        self.assertIn("projection", capabilities)
        self.assertIsNot(capabilities["multigrid"], capabilities["projection"])

    def test_prepare_builds_every_level_then_validates_every_level_before_register(
        self,
    ) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver.velocity_dirichlet_component_ledger_generation = 29
        solver._mg_shapes = ((8, 8, 8), (4, 4, 4), (2, 2, 2))
        events: list[str] = []

        solver._build_canonical_velocity_dirichlet_multigrid_level = (
            lambda fine_level, coarse_level: events.append(
                f"build:{fine_level}->{coarse_level}"
            )
        )
        solver._validate_canonical_velocity_dirichlet_multigrid_level = (
            lambda level: events.append(f"validate:{level}") or 0
        )
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            lambda consumer, *, capability: events.append(f"register:{consumer}")
        )

        CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_multigrid(
            solver
        )

        self.assertEqual(
            events,
            [
                "build:0->1",
                "build:1->2",
                "validate:0",
                "validate:1",
                "validate:2",
                "register:multigrid",
            ],
        )

    def test_prepare_rejects_invalid_level_before_register(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "canonical"
        solver._mg_shapes = ((4, 4, 4), (2, 2, 2))
        registered: list[str] = []
        solver._build_canonical_velocity_dirichlet_multigrid_level = (
            lambda _fine, _coarse: None
        )
        solver._validate_canonical_velocity_dirichlet_multigrid_level = (
            lambda level: 1 if level == 1 else 0
        )
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            lambda consumer, *, capability: registered.append(consumer)
        )

        with self.assertRaisesRegex(RuntimeError, "level 1"):
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_multigrid(
                solver
            )

        self.assertEqual(registered, [])

    def test_prepare_is_strict_noop_for_legacy_authority(self) -> None:
        solver = object.__new__(CartesianFluidSolver)
        solver.velocity_dirichlet_boundary_authority = "legacy"

        def unexpected(*_args, **_kwargs):
            raise AssertionError("legacy prepare touched canonical multigrid state")

        solver._build_canonical_velocity_dirichlet_multigrid_level = unexpected
        solver._validate_canonical_velocity_dirichlet_multigrid_level = unexpected
        solver._register_velocity_dirichlet_component_ledger_consumer_generation = (
            unexpected
        )

        result = (
            CartesianFluidSolver.prepare_velocity_dirichlet_component_ledger_multigrid(
                solver
            )
        )

        self.assertIsNone(result)

    def test_level_zero_aliases_canonical_fields_and_coarse_levels_are_independent(
        self,
    ) -> None:
        source = inspect.getsource(CartesianFluidSolver.__init__)
        required_aliases = (
            "_mg_velocity_dirichlet_boundary_active_component_mask",
            "_mg_velocity_dirichlet_boundary_pressure_mobility",
            "_mg_velocity_dirichlet_boundary_hard_fixed_component_mask",
            "_mg_velocity_dirichlet_boundary_owned_component_mask",
            "_mg_velocity_dirichlet_boundary_external_exact_component_mask",
        )
        for name in required_aliases:
            with self.subTest(name=name):
                self.assertIn(name, source)
        self.assertIn("self.velocity_dirichlet_boundary_active_component_mask", source)
        self.assertIn("self.velocity_dirichlet_boundary_pressure_mobility", source)
        self.assertIn("self.velocity_dirichlet_boundary_owned_component_mask", source)
        self.assertIn(
            "self.velocity_dirichlet_boundary_external_exact_component_mask", source
        )

    def test_device_builder_uses_only_geometrically_coplanar_fine_subfaces(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._canonical_multigrid_fine_face_index
        )
        self.assertIn("2 * coarse_i", source)
        self.assertIn("2 * coarse_j", source)
        self.assertIn("2 * coarse_k", source)
        self.assertIn("transverse_offset_a", source)
        self.assertIn("transverse_offset_b", source)
        self.assertNotIn("ti.ndrange(2, 2, 2)", source)

    def test_fv_neighbor_helper_reads_level_templates_not_fine_grid_self_fields(
        self,
    ) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._velocity_dirichlet_pressure_face_weight
        )
        self.assertIn("canonical_active_component_mask", source)
        self.assertIn("canonical_pressure_mobility", source)
        self.assertIn("canonical_hard_fixed_component_mask", source)
        canonical_branch = source.split("elif", 1)[0]
        self.assertNotIn("self.velocity_dirichlet_boundary_active_component_mask", canonical_branch)
        self.assertNotIn("self.velocity_dirichlet_boundary_pressure_mobility", canonical_branch)

    def test_canonical_restriction_does_not_use_legacy_eight_child_scalar_or_max(
        self,
    ) -> None:
        source = inspect.getsource(CartesianFluidSolver._mg_restrict_residual_kernel)
        self.assertIn("canonical_authority", source)
        self.assertIn("ti.static(canonical_authority == 0)", source)
        self.assertIn("soft_boundary_active", source)
        self.assertIn("soft_boundary_projection_weight", source)


if __name__ == "__main__":
    unittest.main()
