from __future__ import annotations

import unittest

import numpy as np

from tools.diagnostics.inspect_visit_stats import _safe_percentiles_and_max
from tools.diagnostics.summarize_preflight_log import _projected_residual_mps
from tools.rendering.squid_jet_render import _mask_nonphysical_and_floor_values


class PostprocessDiagnosticsTests(unittest.TestCase):
    def test_visit_stats_empty_arrays_report_zero_instead_of_reducing_empty(self) -> None:
        percentiles, maximum = _safe_percentiles_and_max(np.asarray([], dtype=np.float64))

        self.assertEqual(percentiles, [0.0, 0.0, 0.0])
        self.assertEqual(maximum, 0.0)

    def test_render_threshold_keeps_nan_mask_and_floors_only_finite_physical_values(self) -> None:
        values = np.asarray([np.nan, 0.001, 0.02, 0.03], dtype=np.float64)
        physical = np.asarray([True, True, True, False], dtype=bool)

        rendered = _mask_nonphysical_and_floor_values(values, physical=physical, vmin_mps=0.005)

        self.assertTrue(np.isnan(rendered[0]))
        self.assertEqual(rendered[1], 0.0)
        self.assertEqual(rendered[2], 0.02)
        self.assertTrue(np.isnan(rendered[3]))

    def test_preflight_summary_ignores_old_projected_residual_keys(self) -> None:
        residual = _projected_residual_mps(
            {
                "fsi_projected_ibm_marker_residual_max_mps": 0.0,
                "fsi_velocity_residual_mps": 0.123,
            }
        )

        self.assertIsNone(residual)

    def test_preflight_summary_reads_current_projected_ibm_residual_key(self) -> None:
        residual = _projected_residual_mps(
            {
                "projected_ibm_residual_mps": 0.041,
            }
        )

        self.assertEqual(residual, 0.041)


if __name__ == "__main__":
    unittest.main()
