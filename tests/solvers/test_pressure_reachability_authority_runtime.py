from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class PressureReachabilityAuthorityRuntimeContracts(unittest.TestCase):
    """Runtime contracts for authority-local pressure-face reachability."""

    @staticmethod
    def _corridor_solver(*, authority: str) -> CartesianFluidSolver:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, :] = 0
        solver.obstacle.from_numpy(obstacle)
        solver.set_velocity_dirichlet_boundary_authority(authority)
        solver.hibm_pressure_reachability_barrier.fill(0)
        return solver

    def test_canonical_zmin_seed_rejects_raw_hard_face_even_when_derived_mask_is_open(
        self,
    ) -> None:
        solver = self._corridor_solver(authority="canonical")
        outlet = (1, 1, 0)
        # A mixed external-exact x claim makes the derived pressure-hard z bit
        # open, but the canonical raw hard z clamp still has zero pressure
        # mobility in the real composite operator.
        solver.velocity_dirichlet_boundary_active_component_mask[outlet] = 5
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[outlet] = 4
        solver.velocity_dirichlet_boundary_external_exact_component_mask[outlet] = 1
        solver.velocity_dirichlet_boundary_owned_component_mask[outlet] = 0
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        self.assertEqual(
            int(solver.velocity_dirichlet_pressure_hard_fixed_component_mask[outlet]),
            0,
            "the test requires pressure-effective provenance to remain open",
        )

        solver._flood_hibm_pressure_outlet_reachable_frontier()

        self.assertEqual(int(solver.hibm_pressure_outlet_reachable[outlet]), 0)
        self.assertTrue(
            all(
                int(solver.hibm_pressure_outlet_reachable[1, 1, k]) == 0
                for k in range(4)
            )
        )

    def test_canonical_zmin_seed_rejects_zero_mobility_face_with_open_provenance(
        self,
    ) -> None:
        solver = self._corridor_solver(authority="canonical")
        outlet = (1, 1, 0)
        solver.velocity_dirichlet_boundary_active_component_mask[outlet] = 4
        solver.velocity_dirichlet_boundary_pressure_mobility[outlet] = (
            1.0,
            1.0,
            0.0,
        )
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        self.assertEqual(
            int(solver.velocity_dirichlet_pressure_hard_fixed_component_mask[outlet]),
            0,
        )

        solver._flood_hibm_pressure_outlet_reachable_frontier()

        self.assertEqual(int(solver.hibm_pressure_outlet_reachable[outlet]), 0)
        self.assertTrue(
            all(
                int(solver.hibm_pressure_outlet_reachable[1, 1, k]) == 0
                for k in range(4)
            )
        )

    def test_legacy_reachability_ignores_residual_canonical_shadow_faces(
        self,
    ) -> None:
        solver = self._corridor_solver(authority="legacy")
        shadow = (1, 1, 1)
        # No legacy scalar row exists.  Residual canonical fields therefore
        # have no authority to split the legacy pressure graph.
        solver.velocity_dirichlet_boundary_active_component_mask[shadow] = 4
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[shadow] = 4
        solver.velocity_dirichlet_boundary_pressure_mobility[shadow] = (
            1.0,
            1.0,
            0.0,
        )

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 0)
        self.assertTrue(
            all(
                int(solver.hibm_pressure_outlet_reachable[1, 1, k]) == 1
                for k in range(4)
            )
        )

    @staticmethod
    def _record_prepared_identity(
        solver: CartesianFluidSolver,
    ) -> None:
        solver._prepare_hibm_pressure_reachability_barrier(
            None,
            barrier_node_code=0,
        )
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        solver._record_hibm_pressure_reachability_identity(None)
        solver._mark_hibm_pressure_reachability_valid(
            fresh_flood=True,
            valid=True,
        )

    def test_canonical_prepared_identity_covers_raw_hard_mobility_and_authority(
        self,
    ) -> None:
        cases = ("raw_hard", "mobility", "authority")
        cell = (1, 1, 0)

        for case in cases:
            with self.subTest(case=case):
                solver = self._corridor_solver(authority="canonical")
                self._record_prepared_identity(solver)
                if case == "raw_hard":
                    solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        cell
                    ] = 4
                elif case == "mobility":
                    solver.velocity_dirichlet_boundary_pressure_mobility[cell] = (
                        1.0,
                        1.0,
                        0.0,
                    )
                else:
                    # Bypass the public invalidation intentionally: this is an
                    # identity checksum contract, not a generation-counter test.
                    solver.velocity_dirichlet_boundary_authority = "legacy"

                with self.assertRaisesRegex(
                    RuntimeError,
                    "reachability|topology|identity|authority",
                ):
                    solver._require_prepared_hibm_pressure_reachability_current()
                self.assertFalse(solver.last_hibm_reachability_valid)

    def test_legacy_prepared_identity_ignores_canonical_shadow_mutation(
        self,
    ) -> None:
        solver = self._corridor_solver(authority="legacy")
        self._record_prepared_identity(solver)
        shadow = (1, 1, 1)
        solver.velocity_dirichlet_boundary_active_component_mask[shadow] = 4
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[shadow] = 4
        solver.velocity_dirichlet_boundary_pressure_mobility[shadow] = (
            1.0,
            1.0,
            0.0,
        )

        solver._require_prepared_hibm_pressure_reachability_current()

        self.assertTrue(solver.last_hibm_reachability_valid)


if __name__ == "__main__":
    unittest.main()
