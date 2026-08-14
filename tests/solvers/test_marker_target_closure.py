import unittest

import numpy as np

from simulation_core.coupling.hibm_mpm.marker_target_closure import (
    MarkerTargetClosureIncompatibleError,
    solve_weighted_marker_target_closure,
)


class MarkerTargetClosureTests(unittest.TestCase):
    def test_rank_deficient_consistent_rows_are_solved_to_tolerance(self) -> None:
        matrix = np.asarray(
            (
                (0.5, 0.5, 0.0),
                (0.5, 0.5, 0.0),
                (0.0, 0.25, 0.75),
            ),
            dtype=np.float64,
        )
        residual = np.asarray((1.0, 1.0, -0.5), dtype=np.float64)
        inverse_mass = np.asarray((1.0, 4.0, 2.0), dtype=np.float64)

        result = solve_weighted_marker_target_closure(
            matrix,
            residual,
            inverse_mass,
            absolute_tolerance_mps=1.0e-12,
        )

        np.testing.assert_allclose(
            matrix @ result.correction_mps,
            residual,
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(result.rank, 2)
        self.assertLessEqual(result.max_residual_mps, 1.0e-12)
        self.assertEqual(result.constraint_count, 3)
        self.assertEqual(result.adjustable_dof_count, 3)

    def test_inconsistent_duplicate_rows_fail_with_residual_diagnostics(self) -> None:
        matrix = np.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=np.float64)
        residual = np.asarray((0.0, 1.0e-3), dtype=np.float64)
        inverse_mass = np.asarray((1.0, 1.0), dtype=np.float64)

        with self.assertRaisesRegex(
            MarkerTargetClosureIncompatibleError,
            r"least_squares_max_residual_mps=0\.0005.*rank=1",
        ):
            solve_weighted_marker_target_closure(
                matrix,
                residual,
                inverse_mass,
                absolute_tolerance_mps=1.0e-6,
            )

    def test_inverse_mass_weights_select_the_expected_minimum_norm_solution(
        self,
    ) -> None:
        result = solve_weighted_marker_target_closure(
            np.asarray(((1.0, 1.0),), dtype=np.float64),
            np.asarray((1.0,), dtype=np.float64),
            np.asarray((1.0, 4.0), dtype=np.float64),
            absolute_tolerance_mps=1.0e-12,
        )

        np.testing.assert_allclose(
            result.correction_mps,
            (0.2, 0.8),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_f32_near_duplicate_rows_are_rank_collapsed_and_fail_closed(
        self,
    ) -> None:
        matrix = np.asarray(
            ((1.0, 1.0), (1.0, 1.0 + 5.0e-7)),
            dtype=np.float64,
        )

        with self.assertRaisesRegex(
            MarkerTargetClosureIncompatibleError,
            r"rank=1",
        ):
            solve_weighted_marker_target_closure(
                matrix,
                np.asarray((1.0, 1.0 + 1.0e-3), dtype=np.float64),
                np.asarray((1.0, 1.0), dtype=np.float64),
                absolute_tolerance_mps=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
