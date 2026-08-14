from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from cases.turek_hron_fsi import (
    TurekHronFsiConfig,
    _build_fluid,
    _write_channel_external_velocity_faces,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig


class TurekCanonicalBoundaryRuntimeTests(unittest.TestCase):
    def test_channel_faces_build_and_seal_canonical_ledger(self) -> None:
        config = replace(
            TurekHronFsiConfig(),
            grid_nodes=(4, 6, 8),
            solid_particle_counts=(1, 2, 4),
        )
        fluid = _build_fluid(
            config,
            TaichiRuntimeConfig(arch="cuda", offline_cache=False),
        )

        _write_channel_external_velocity_faces(fluid, config, t_s=2.0)

        self.assertEqual(fluid.velocity_dirichlet_boundary_authority, "canonical")
        self.assertTrue(fluid.velocity_dirichlet_component_ledger_sealed)
        y_masks = (
            fluid.external_velocity_boundary_y_face_active_component_mask.to_numpy()
        )
        z_masks = (
            fluid.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        z_values = fluid.external_velocity_boundary_z_face_value_mps.to_numpy()
        np.testing.assert_array_equal(y_masks, np.full_like(y_masks, 7))
        np.testing.assert_array_equal(z_masks[0], np.zeros_like(z_masks[0]))
        np.testing.assert_array_equal(z_masks[1], np.full_like(z_masks[1], 7))
        np.testing.assert_array_equal(z_values[1, :, 0], 0.0)
        np.testing.assert_array_equal(z_values[1, :, -1], 0.0)
        self.assertTrue(np.all(z_values[1, :, 1:-1, 2] < 0.0))


if __name__ == "__main__":
    unittest.main()
