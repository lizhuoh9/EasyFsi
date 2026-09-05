"""Strict-CUDA coverage of collective closure beyond the old dense cap."""

import unittest

import numpy as np

from simulation_core import TaichiRuntimeConfig, init_taichi
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)
from tests.solvers import test_hibm_component_face_geometry as geometry


class LargeCollectiveProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_taichi(TaichiRuntimeConfig(arch="cuda", strict_arch=True, default_fp="f32"))

    def test_large_f_minimax_and_weighted_fh_keep_exact_row_contracts(self):
        seed = geometry.HibmComponentFaceGeometryTests._seed_isolated_collective_rows
        for capacity in (223, 334):
            with self.subTest(marker_capacity=capacity):
                operator = HibmMpmMarkerMacConstraintOperator(
                    grid_nodes=(4, 4, 4), marker_capacity=capacity
                )
                # Every marker contributes an active row. Repeated supports
                # retain an inconsistent, minimax-feasible all-row system.
                rows = [
                    (3 * i, 3.0e-4 if i == capacity - 1 else 0.0,
                     (((0, 0, 0), 1.0, True, 1.0),))
                    for i in range(capacity)
                ]
                seed(operator, rows)
                self.assertTrue(operator._collective_isolated_f_only_feasible(1.6e-4))
                np.testing.assert_array_equal(
                    operator._collective_delta_free.to_numpy(), 0.0
                )

                # Structural rank must not discard a distinct tiny-mobility
                # DOF. The first row has the exact minimum-energy split 1:4.
                rows = [
                    (3 * i, 1.0, (((0, 0, 0), 1.0, True, 1.0),
                                  ((0, 0, 1), 1.0, False, 4.0)))
                    for i in range(capacity - 1)
                ]
                rows.append((3 * (capacity - 1), 1.0e-8,
                             (((1, 0, 0), 1.0, True, 1.0e-32),)))
                seed(operator, rows, adjustable_supports=[
                    (3 * i, 1) for i in range(capacity - 1)
                ])
                operator._collective_row_certificate[0] = 1
                report = operator._collective_isolated_fh_repair(
                    closure_tolerance=1.0e-7, absolute_tolerance=1.0e-7
                )
                self.assertIsNotNone(report)
                free = operator._collective_delta_free.to_numpy()
                hard = operator._collective_delta_hard.to_numpy()
                np.testing.assert_allclose(
                    (free[0, 0, 0, 0], hard[0, 0, 1, 0]), (0.2, 0.8),
                    rtol=0.0, atol=3.0e-8,
                )
                self.assertAlmostEqual(float(free[1, 0, 0, 0]), 1.0e-8, delta=1.0e-15)

                # Disconnected rows without a certificate cannot acquire H
                # authority and must make an absolute-infeasible repair fail.
                rows.extend([
                    (1, 0.0, (((2, 0, 0), 1.0, True, 1.0),)),
                    (4, 4.0e-4, (((2, 0, 0), 1.0, True, 1.0),)),
                ])
                seed(operator, rows, adjustable_supports=[
                    (3 * i, 1) for i in range(capacity - 1)
                ])
                operator._collective_row_certificate[0] = 1
                self.assertIsNone(operator._collective_isolated_fh_repair(
                    closure_tolerance=1.0e-6, absolute_tolerance=1.0e-4
                ))
                np.testing.assert_array_equal(operator._collective_delta_free.to_numpy(), 0.0)
                np.testing.assert_array_equal(operator._collective_delta_hard.to_numpy(), 0.0)
                np.testing.assert_array_equal(operator._collective_row_repair_active.to_numpy(), 0)


if __name__ == "__main__":
    unittest.main()
