from __future__ import annotations

import unittest

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    TaichiRuntimeConfig,
)


def _seal_current_generation(solver: CartesianFluidSolver) -> None:
    capabilities = (
        solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES
    )
    for consumer in sorted(
        solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMERS
    ):
        solver._register_velocity_dirichlet_component_ledger_consumer_generation(
            consumer,
            capability=capabilities[consumer],
        )
    solver.seal_velocity_dirichlet_component_ledger()


def _canonical_solver() -> CartesianFluidSolver:
    solver = CartesianFluidSolver(
        FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )
    solver.set_velocity_dirichlet_boundary_authority("canonical")
    solver.velocity_dirichlet_boundary_active_component_mask.fill(0)
    solver.velocity_dirichlet_boundary_value_mps.fill(0.0)
    solver.velocity_dirichlet_boundary_pressure_mobility.fill(0.0)
    solver.velocity_dirichlet_boundary_component_enforcement_weight.fill(0.0)
    solver.velocity_dirichlet_boundary_component_region_id.fill(-1)
    solver.velocity_dirichlet_boundary_hard_fixed_component_mask.fill(0)
    solver.velocity_dirichlet_boundary_external_exact_component_mask.fill(0)
    solver.velocity_dirichlet_boundary_owned_component_mask.fill(0)
    solver.obstacle.fill(0)
    solver.external_velocity_boundary_x_face_active_component_mask.fill(0)
    solver.external_velocity_boundary_x_face_value_mps.fill(0.0)
    solver.external_velocity_boundary_y_face_active_component_mask.fill(0)
    solver.external_velocity_boundary_y_face_value_mps.fill(0.0)
    solver.external_velocity_boundary_z_face_active_component_mask.fill(0)
    solver.external_velocity_boundary_z_face_value_mps.fill(0.0)
    _seal_current_generation(solver)
    return solver


class VelocityDirichletLedgerContentEquivalenceTests(unittest.TestCase):
    def test_reference_generation_rejects_bool_and_fractional_tokens(self) -> None:
        solver = _canonical_solver()
        solver.capture_velocity_dirichlet_boundary_ledger_reference()

        for invalid in (True, np.bool_(False), 1.0, np.float64(1.0)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "exact integer"):
                    solver.velocity_dirichlet_boundary_ledger_comparison(
                        expected_generation=invalid,
                    )
                with self.assertRaisesRegex(TypeError, "exact integer"):
                    solver.velocity_dirichlet_boundary_ledger_mismatch_rows(
                        expected_generation=invalid,
                    )

    def test_identical_rebuild_changes_identity_but_not_content(self) -> None:
        solver = _canonical_solver()
        reference_generation = (
            solver.capture_velocity_dirichlet_boundary_ledger_reference()
        )
        reference_component_generation = int(
            solver.velocity_dirichlet_component_ledger_generation
        )

        solver._invalidate_velocity_dirichlet_component_ledger()
        _seal_current_generation(solver)

        comparison = solver.velocity_dirichlet_boundary_ledger_comparison(
            expected_generation=reference_generation,
        )

        self.assertEqual(comparison["device_content_mismatch_rows"], 0)
        self.assertEqual(comparison["content_equivalence_mismatch_rows"], 0)
        self.assertEqual(comparison["identity_mismatch_rows"], 1)
        self.assertTrue(comparison["component_generation_changed"])
        self.assertEqual(
            comparison["reference_component_generation"],
            reference_component_generation,
        )
        self.assertEqual(
            comparison["current_component_generation"],
            reference_component_generation + 1,
        )
        self.assertIsNone(comparison["first_content_mismatch_field"])
        self.assertEqual(
            comparison["first_identity_mismatch_field"],
            "component_ledger_generation",
        )
        self.assertEqual(
            solver.velocity_dirichlet_boundary_ledger_mismatch_rows(
                expected_generation=reference_generation,
            ),
            1,
        )

    def test_content_mode_keeps_every_projection_field_exact(self) -> None:
        solver = _canonical_solver()
        solver.velocity_dirichlet_boundary_value_mps[1, 1, 1] = (
            0.0,
            1.0,
            0.0,
        )
        reference_generation = (
            solver.capture_velocity_dirichlet_boundary_ledger_reference()
        )
        solver._invalidate_velocity_dirichlet_component_ledger()
        _seal_current_generation(solver)

        one_ulp = np.nextafter(
            np.float32(1.0),
            np.float32(2.0),
            dtype=np.float32,
        )
        row = (1, 1, 1)
        face = (0, 1, 1)
        cases = (
            (
                "active_component_mask",
                lambda: solver.velocity_dirichlet_boundary_active_component_mask.__setitem__(row, 1),
                lambda: solver.velocity_dirichlet_boundary_active_component_mask.__setitem__(row, 0),
            ),
            (
                "value_mps_one_ulp",
                lambda: solver.velocity_dirichlet_boundary_value_mps.__setitem__(row, (0.0, float(one_ulp), 0.0)),
                lambda: solver.velocity_dirichlet_boundary_value_mps.__setitem__(row, (0.0, 1.0, 0.0)),
            ),
            (
                "pressure_mobility",
                lambda: solver.velocity_dirichlet_boundary_pressure_mobility.__setitem__(row, (0.0, 0.5, 0.0)),
                lambda: solver.velocity_dirichlet_boundary_pressure_mobility.__setitem__(row, (0.0, 0.0, 0.0)),
            ),
            (
                "component_enforcement_weight",
                lambda: solver.velocity_dirichlet_boundary_component_enforcement_weight.__setitem__(row, (0.0, 0.75, 0.0)),
                lambda: solver.velocity_dirichlet_boundary_component_enforcement_weight.__setitem__(row, (0.0, 0.0, 0.0)),
            ),
            (
                "component_region_id",
                lambda: solver.velocity_dirichlet_boundary_component_region_id.__setitem__(row, (-1, 202, -1)),
                lambda: solver.velocity_dirichlet_boundary_component_region_id.__setitem__(row, (-1, -1, -1)),
            ),
            (
                "hard_fixed_component_mask",
                lambda: solver.velocity_dirichlet_boundary_hard_fixed_component_mask.__setitem__(row, 2),
                lambda: solver.velocity_dirichlet_boundary_hard_fixed_component_mask.__setitem__(row, 0),
            ),
            (
                "external_exact_component_mask",
                lambda: solver.velocity_dirichlet_boundary_external_exact_component_mask.__setitem__(row, 2),
                lambda: solver.velocity_dirichlet_boundary_external_exact_component_mask.__setitem__(row, 0),
            ),
            (
                "owned_component_mask",
                lambda: solver.velocity_dirichlet_boundary_owned_component_mask.__setitem__(row, 2),
                lambda: solver.velocity_dirichlet_boundary_owned_component_mask.__setitem__(row, 0),
            ),
            (
                "obstacle",
                lambda: solver.obstacle.__setitem__(row, 1),
                lambda: solver.obstacle.__setitem__(row, 0),
            ),
            (
                "external_x_active_mask",
                lambda: solver.external_velocity_boundary_x_face_active_component_mask.__setitem__(face, 1),
                lambda: solver.external_velocity_boundary_x_face_active_component_mask.__setitem__(face, 0),
            ),
            (
                "external_x_value",
                lambda: solver.external_velocity_boundary_x_face_value_mps.__setitem__(face, (1.0, 0.0, 0.0)),
                lambda: solver.external_velocity_boundary_x_face_value_mps.__setitem__(face, (0.0, 0.0, 0.0)),
            ),
            (
                "external_y_active_mask",
                lambda: solver.external_velocity_boundary_y_face_active_component_mask.__setitem__(face, 2),
                lambda: solver.external_velocity_boundary_y_face_active_component_mask.__setitem__(face, 0),
            ),
            (
                "external_y_value",
                lambda: solver.external_velocity_boundary_y_face_value_mps.__setitem__(face, (0.0, 1.0, 0.0)),
                lambda: solver.external_velocity_boundary_y_face_value_mps.__setitem__(face, (0.0, 0.0, 0.0)),
            ),
            (
                "external_z_active_mask",
                lambda: solver.external_velocity_boundary_z_face_active_component_mask.__setitem__(face, 4),
                lambda: solver.external_velocity_boundary_z_face_active_component_mask.__setitem__(face, 0),
            ),
            (
                "external_z_value",
                lambda: solver.external_velocity_boundary_z_face_value_mps.__setitem__(face, (0.0, 0.0, 1.0)),
                lambda: solver.external_velocity_boundary_z_face_value_mps.__setitem__(face, (0.0, 0.0, 0.0)),
            ),
        )

        for label, mutate, restore in cases:
            with self.subTest(field=label):
                mutate()
                comparison = solver.velocity_dirichlet_boundary_ledger_comparison(
                    expected_generation=reference_generation,
                )
                self.assertEqual(
                    comparison["device_content_mismatch_rows"],
                    1,
                )
                self.assertEqual(
                    comparison["content_equivalence_mismatch_rows"],
                    1,
                )
                self.assertEqual(
                    comparison["first_content_mismatch_field"],
                    "device_content",
                )
                restore()
                restored = solver.velocity_dirichlet_boundary_ledger_comparison(
                    expected_generation=reference_generation,
                )
                self.assertEqual(
                    restored["content_equivalence_mismatch_rows"],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
