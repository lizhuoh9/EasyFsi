from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class HibmPreparedReachabilityIdentityContracts(unittest.TestCase):
    """Host-only contracts for prepared pressure-reachability reuse."""

    @staticmethod
    def _prepared_solver(*, checksum: tuple[int, int, int, int]):
        solver = object.__new__(CartesianFluidSolver)
        solver.last_hibm_reachability_valid = True
        solver.hibm_reachability_revision = 8
        solver.last_hibm_reachability_revision = 8
        solver._hibm_reachability_checksum = checksum
        return solver

    def test_fresh_flood_identity_is_recorded_from_device_checksum(self) -> None:
        solver = self._prepared_solver(checksum=(0, 0, 0, 0))
        solver._hibm_reachability_checksum = None
        solver._hibm_reachability_pattern_checksum = lambda: (11, 12, 13, 14)

        recorded = (
            CartesianFluidSolver._record_hibm_pressure_reachability_identity(
                solver,
                None,
            )
        )

        self.assertEqual(recorded, (11, 12, 13, 14))
        self.assertEqual(solver._hibm_reachability_checksum, recorded)

    def test_prepared_identity_accepts_the_exact_current_device_pattern(self) -> None:
        solver = self._prepared_solver(checksum=(21, 22, 23, 24))
        solver._hibm_reachability_pattern_checksum = lambda: (21, 22, 23, 24)

        CartesianFluidSolver._require_prepared_hibm_pressure_reachability_current(
            solver
        )

        self.assertTrue(solver.last_hibm_reachability_valid)
        self.assertEqual(solver.hibm_reachability_revision, 8)

    def test_prepared_identity_rejects_direct_device_topology_change(self) -> None:
        solver = self._prepared_solver(checksum=(31, 32, 33, 34))
        solver._hibm_reachability_pattern_checksum = lambda: (31, 32, 99, 34)
        solver._invalidate_hibm_pressure_reachability = (
            CartesianFluidSolver._invalidate_hibm_pressure_reachability.__get__(
                solver,
                CartesianFluidSolver,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "prepared HIBM pressure reachability topology identity mismatch",
        ):
            CartesianFluidSolver._require_prepared_hibm_pressure_reachability_current(
                solver
            )

        self.assertFalse(solver.last_hibm_reachability_valid)
        self.assertIsNone(solver._hibm_reachability_checksum)
        self.assertEqual(solver.hibm_reachability_revision, 9)

    def test_prepared_identity_rejects_revision_mismatch_before_device_read(self) -> None:
        solver = self._prepared_solver(checksum=(41, 42, 43, 44))
        solver.last_hibm_reachability_revision = 7
        solver._invalidate_hibm_pressure_reachability = (
            CartesianFluidSolver._invalidate_hibm_pressure_reachability.__get__(
                solver,
                CartesianFluidSolver,
            )
        )

        def unexpected_checksum_read() -> tuple[int, int, int, int]:
            raise AssertionError("stale host revision read the device pattern")

        solver._hibm_reachability_pattern_checksum = unexpected_checksum_read

        with self.assertRaisesRegex(
            RuntimeError,
            "prepared HIBM pressure reachability revision mismatch",
        ):
            CartesianFluidSolver._require_prepared_hibm_pressure_reachability_current(
                solver
            )

        self.assertFalse(solver.last_hibm_reachability_valid)
        self.assertIsNone(solver._hibm_reachability_checksum)

    def test_project_validates_prepared_identity_after_hard_mask_refresh(self) -> None:
        source = inspect.getsource(CartesianFluidSolver.project)
        refresh_index = source.index(
            "_refresh_velocity_dirichlet_pressure_hard_fixed_component_mask"
        )
        validate_index = source.index(
            "_require_prepared_hibm_pressure_reachability_current"
        )
        prepared_use_index = source.index(
            "converted_overflow_singletons",
            validate_index,
        )

        self.assertLess(refresh_index, validate_index)
        self.assertLess(validate_index, prepared_use_index)


if __name__ == "__main__":
    unittest.main()
