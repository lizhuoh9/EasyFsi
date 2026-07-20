from __future__ import annotations

import unittest

import numpy as np

from tools.diagnose_velocity_dirichlet_ledger_mismatch import (
    _array_difference_summary,
)


class VelocityDirichletLedgerDiagnosticTests(unittest.TestCase):
    def test_vector_difference_reports_exact_row_component_and_float_bits(self) -> None:
        reference = np.zeros((2, 2, 1, 3), dtype=np.float32)
        current = reference.copy()
        current[1, 0, 0, 2] = np.float32(1.25)

        summary = _array_difference_summary(
            current,
            reference,
            component_axis=True,
        )

        self.assertEqual(summary["mismatch_row_count"], 1)
        self.assertEqual(summary["mismatch_element_count"], 1)
        self.assertEqual(summary["examples"][0]["index"], [1, 0, 0, 2])
        self.assertEqual(summary["examples"][0]["current"], 1.25)
        self.assertEqual(summary["examples"][0]["reference"], 0.0)
        self.assertEqual(summary["examples"][0]["current_bits_hex"], "0x3fa00000")
        self.assertEqual(summary["examples"][0]["reference_bits_hex"], "0x00000000")

    def test_scalar_difference_counts_each_grid_row_once(self) -> None:
        reference = np.zeros((2, 2, 1), dtype=np.int32)
        current = reference.copy()
        current[0, 1, 0] = 7

        summary = _array_difference_summary(
            current,
            reference,
            component_axis=False,
        )

        self.assertEqual(summary["mismatch_row_count"], 1)
        self.assertEqual(summary["mismatch_element_count"], 1)
        self.assertEqual(summary["examples"], [
            {"index": [0, 1, 0], "current": 7, "reference": 0}
        ])


if __name__ == "__main__":
    unittest.main()
