from __future__ import annotations

import unittest

import numpy as np

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.geometry_tools import SurfaceMesh
from simulation_core.solids.mooney_shell import TriMooneyShellMpmState


class MooneyShellPostStepOutOfBoundsTests(unittest.TestCase):
    def test_particle_leaving_grid_during_step_is_reported_immediately(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=0.25,
            primary_region_id=1,
            secondary_region_id=2,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        positions = state.x.to_numpy()
        velocities = state.v.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] - 0.25 * state.dx[0])
        velocities[0, 0] = 2.0
        state.x.from_numpy(positions.astype(np.float32))
        state.v.from_numpy(velocities.astype(np.float32))

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.step(
                dt_s=float(state.dx[0]),
                pressure_pa=0.0,
                velocity_damping=1.0,
                flip_blend=1.0,
                read_report=False,
            )


if __name__ == "__main__":
    unittest.main()
