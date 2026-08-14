from __future__ import annotations

import inspect
import unittest

from simulation_core.fluids.solver import CartesianFluidSolver


class SstObstacleTransportContracts(unittest.TestCase):
    def test_obstacle_transport_does_not_seed_omega_from_inlet(self) -> None:
        source = inspect.getsource(
            CartesianFluidSolver._advance_sst_transport_kernel
        )
        obstacle_start = source.index("if self.obstacle[i, j, k] != 0:")
        obstacle_end = source.index("            else:", obstacle_start)
        obstacle_branch = source[obstacle_start:obstacle_end]

        self.assertNotIn(
            "self.sst_specific_dissipation_rate_next[i, j, k] = inlet_omega_s",
            source,
        )
        self.assertIn(
            "wall_distance = ti.max(self.sst_wall_distance_m[i, j, k], 1.0e-12)",
            obstacle_branch,
        )
        self.assertIn(
            "self.sst_specific_dissipation_rate_next[i, j, k] = (",
            obstacle_branch,
        )
        self.assertIn("self._sst_wall_omega_target(", obstacle_branch)
        self.assertIn("molecular_nu_m2_s,", obstacle_branch)
        self.assertIn("wall_distance,", obstacle_branch)

    def test_accepted_state_applies_wall_omega_after_transport_commit(self) -> None:
        advance_source = inspect.getsource(CartesianFluidSolver.advance_sst_transport)
        wall_state_source = inspect.getsource(
            CartesianFluidSolver._apply_sst_wall_state_kernel
        )

        commit = advance_source.index("self._commit_sst_transport_kernel()")
        lod_x = advance_source.index("self._sst_lod_backward_euler_axis_kernel(")
        lod_y = advance_source.index(
            "self._sst_lod_backward_euler_axis_kernel(", lod_x + 1
        )
        lod_z = advance_source.index(
            "self._sst_lod_backward_euler_axis_kernel(", lod_y + 1
        )
        wall_state = advance_source.index("self._apply_sst_wall_state_kernel(")
        state_commit_after = advance_source.index(
            'observe_initial_transport_stage("state_commit_after")'
        )

        self.assertLess(commit, lod_x)
        self.assertLess(commit, state_commit_after)
        self.assertLess(state_commit_after, lod_x)
        self.assertLess(lod_x, lod_y)
        self.assertLess(lod_y, lod_z)
        self.assertLess(lod_z, wall_state)
        self.assertIn("self._sst_wall_omega_target(", wall_state_source)


if __name__ == "__main__":
    unittest.main()
