"""CPU-only contracts for the scalable sparse marker rank helper."""

from __future__ import annotations

import unittest

import numpy as np

from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    _sparse_metric_columns,
    _sparse_normalized_pivoted_cholesky,
)


class HibmMarkerSparseRankReductionTests(unittest.TestCase):
    @staticmethod
    def _rank_deficient_rows(marker_capacity: int):
        rows = []
        for row in range(3 * marker_capacity):
            axis = row % 3
            pattern = (row // 3) % 16
            rows.append(
                {
                    (axis, pattern, offset, 0): 0.25 * (offset + 1)
                    for offset in range(8)
                }
            )
        return rows

    def test_l1_and_l2_capacity_keep_only_m_by_actual_rank_work(self) -> None:
        for marker_capacity in (223, 334):
            rows = self._rank_deficient_rows(marker_capacity)
            selected, factor, norms = _sparse_normalized_pivoted_cholesky(
                rows,
                relative_pivot_tolerance=1.0e-12,
            )
            self.assertEqual(factor.shape[0], 3 * marker_capacity)
            self.assertLess(selected.size, 3 * marker_capacity)
            self.assertEqual(factor.shape[1], selected.size)
            self.assertTrue(np.all(norms > 0.0))

            columns = _sparse_metric_columns(rows, selected)
            target = columns @ np.linspace(0.1, 1.0, selected.size)
            solution, _, _, _ = np.linalg.lstsq(columns, target, rcond=1.0e-12)
            np.testing.assert_allclose(columns @ solution, target, atol=1.0e-11)

    def test_zero_energy_row_is_not_rank_selected(self) -> None:
        rows = [
            {(0, 0, 0, 0): 1.0},
            {(0, 0, 0, 0): 1.0},
            {},
        ]
        selected, factor, norms = _sparse_normalized_pivoted_cholesky(
            rows,
            relative_pivot_tolerance=1.0e-12,
        )
        np.testing.assert_array_equal(selected, np.array([0]))
        self.assertEqual(factor.shape, (3, 1))
        self.assertEqual(norms[2], 0.0)


if __name__ == "__main__":
    unittest.main()
