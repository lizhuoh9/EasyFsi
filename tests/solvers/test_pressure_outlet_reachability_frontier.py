from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class PressureOutletReachabilityFrontierContracts(unittest.TestCase):
    @staticmethod
    def _solver(grid_nodes: tuple[int, int, int]) -> CartesianFluidSolver:
        return CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    def test_pressure_outlet_reachability_uses_sparse_device_frontiers(self) -> None:
        source = Path("simulation_core/fluids/solver.py").read_text(encoding="utf-8")
        start = source.index(
            "def mark_hibm_pressure_outlet_disconnected_nonprojectable_cells"
        )
        end = source.index("    @ti.kernel", start)
        flood_source = source[start:end]
        frontier_start = source.index(
            "def _expand_hibm_pressure_outlet_reachable_frontier_kernel"
        )
        frontier_end = source.index("    @ti.kernel", frontier_start)
        frontier_source = source[frontier_start:frontier_end]

        self.assertIn("hibm_pressure_reachability_frontier_a", source)
        self.assertIn("hibm_pressure_reachability_frontier_b", source)
        self.assertIn("_expand_hibm_pressure_outlet_reachable_frontier_kernel", source)
        self.assertIn("ti.atomic_or", frontier_source)
        self.assertIn("_pressure_cells_connected", frontier_source)
        self.assertNotIn(
            "_expand_and_commit_hibm_pressure_outlet_reachable_kernel()",
            flood_source,
        )

    def test_sparse_frontier_matches_maze_obstacle_and_barrier_semantics(self) -> None:
        grid_nodes = (7, 4, 7)
        solver = self._solver(grid_nodes)
        expected_capacity = int(np.prod(grid_nodes))
        self.assertEqual(solver.hibm_pressure_reachability_frontier_capacity, expected_capacity)
        self.assertEqual(
            tuple(solver.hibm_pressure_reachability_frontier_a.shape),
            (expected_capacity,),
        )
        self.assertEqual(
            tuple(solver.hibm_pressure_reachability_frontier_b.shape),
            (expected_capacity,),
        )
        obstacle = np.ones(grid_nodes, dtype=np.int32)
        path = (
            [(1, 1, 0)]
            + [(i, 1, 1) for i in range(1, 6)]
            + [(5, 1, 2)]
            + [(i, 1, 3) for i in range(5, 0, -1)]
            + [(1, 1, 4)]
            + [(i, 1, 5) for i in range(1, 6)]
            + [(5, 1, 6)]
        )
        disconnected_cavity = ((3, 0, 6), (4, 0, 6))
        for cell in (*path, *disconnected_cavity):
            obstacle[cell] = 0
        solver.obstacle.from_numpy(obstacle)

        def unexpected_full_grid_expand() -> None:
            raise AssertionError("sparse flood called the legacy full-grid expand kernel")

        solver._expand_and_commit_hibm_pressure_outlet_reachable_kernel = (
            unexpected_full_grid_expand
        )
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, len(disconnected_cavity))
        self.assertTrue(solver.last_hibm_pressure_reachability_converged)
        self.assertEqual(solver.last_hibm_pressure_reachability_sweeps, len(path))
        self.assertTrue(
            all(int(solver.hibm_pressure_outlet_reachable[cell]) == 1 for cell in path)
        )
        self.assertTrue(
            all(
                int(solver.hibm_pressure_outlet_reachable[cell]) == 0
                for cell in disconnected_cavity
            )
        )

        barrier_index = 7
        barrier_cell = path[barrier_index]
        solver.hibm_pressure_reachability_barrier.fill(0)
        solver.hibm_pressure_reachability_barrier[barrier_cell] = 1
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
            use_existing_reachability_barrier=True,
        )

        self.assertEqual(
            unreached,
            len(path) - barrier_index - 1 + len(disconnected_cavity),
        )
        self.assertEqual(solver.last_hibm_pressure_reachability_sweeps, barrier_index)
        self.assertEqual(int(solver.hibm_pressure_outlet_reachable[barrier_cell]), 0)
        self.assertTrue(
            all(
                int(solver.hibm_pressure_outlet_reachable[cell]) == 0
                for cell in path[barrier_index + 1 :]
            )
        )

    def test_sparse_frontier_preserves_directional_hard_face_connectivity(self) -> None:
        grid_nodes = (5, 5, 5)
        solver = self._solver(grid_nodes)
        corridor = tuple((2, 2, k) for k in range(grid_nodes[2]))
        obstacle = np.ones(grid_nodes, dtype=np.int32)
        for cell in corridor:
            obstacle[cell] = 0
        solver.obstacle.from_numpy(obstacle)
        hard_face_owner = corridor[1]
        solver.velocity_dirichlet_boundary_active[hard_face_owner] = 1
        solver.velocity_dirichlet_face_symmetric = 0

        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
            hard_face_owner
        ] = 4
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, len(corridor) - 1)
        self.assertEqual(solver.last_hibm_pressure_reachability_sweeps, 1)

        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
            hard_face_owner
        ] = 1
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 0)
        self.assertEqual(solver.last_hibm_pressure_reachability_sweeps, len(corridor))

    def test_sparse_frontier_overflow_fails_closed(self) -> None:
        solver = self._solver((4, 4, 4))
        solver.hibm_pressure_reachability_frontier_capacity = 0

        with self.assertRaisesRegex(RuntimeError, "reachability frontier overflow"):
            solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )

        self.assertFalse(solver.last_hibm_pressure_reachability_converged)
        self.assertFalse(solver.last_hibm_reachability_valid)

    def test_disabled_pressure_outlet_clears_stale_frontier_overflow(self) -> None:
        solver = self._solver((4, 4, 4))
        solver.last_hibm_pressure_reachability_frontier_overflow = True

        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=False,
        )

        self.assertEqual(unreached, 0)
        self.assertFalse(solver.last_hibm_pressure_reachability_frontier_overflow)


if __name__ == "__main__":
    unittest.main()
