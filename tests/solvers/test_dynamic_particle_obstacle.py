from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class DynamicParticleObstacleTests(unittest.TestCase):
    @staticmethod
    def _solver(
        grid_nodes: tuple[int, int, int] = (8, 8, 8),
    ) -> CartesianFluidSolver:
        return CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    @staticmethod
    def _particle_positions(capacity: int, values: np.ndarray):
        field = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        field.from_numpy(np.asarray(values, dtype=np.float32))
        return field

    @staticmethod
    def _particle_deformation_gradients(capacity: int, values: np.ndarray):
        field = ti.Matrix.field(3, 3, dtype=ti.f32, shape=capacity)
        field.from_numpy(np.asarray(values, dtype=np.float32))
        return field

    @staticmethod
    def _six_neighbor_mean(field: np.ndarray, cell: tuple[int, int, int]) -> np.ndarray:
        i, j, k = cell
        neighbors = (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        )
        return np.mean(np.asarray([field[index] for index in neighbors]), axis=0)

    @staticmethod
    def _occupied_extents(mask: np.ndarray) -> tuple[int, int, int]:
        occupied = np.argwhere(np.asarray(mask) != 0)
        if occupied.size == 0:
            return (0, 0, 0)
        return tuple(
            int(occupied[:, axis].max() - occupied[:, axis].min() + 1)
            for axis in range(3)
        )

    def test_static_base_obstacle_survives_repeated_particle_updates(self) -> None:
        solver = self._solver()
        static_base = np.zeros((8, 8, 8), dtype=np.int32)
        static_cells = ((0, 0, 0), (7, 7, 7))
        for cell in static_cells:
            static_base[cell] = 1
        solver.obstacle.from_numpy(static_base)
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.3125, 0.4375, 0.5625]], dtype=np.float32),
        )
        support_size_m = (0.08, 0.08, 0.08)

        first_report = solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        first_composite = solver.obstacle.to_numpy()
        for cell in static_cells:
            self.assertEqual(int(first_composite[cell]), 1)
        particle_position_m.from_numpy(
            np.asarray([[0.6875, 0.5625, 0.4375]], dtype=np.float32)
        )
        second_report = solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        second_composite = solver.obstacle.to_numpy()
        for cell in static_cells:
            self.assertEqual(int(second_composite[cell]), 1)
        final_report = solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=0,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )

        self.assertGreater(first_report["fluid_dynamic_obstacle_cell_count"], 0)
        self.assertGreater(second_report["fluid_dynamic_obstacle_cell_count"], 0)
        self.assertEqual(final_report["fluid_dynamic_obstacle_cell_count"], 0)
        composite = solver.obstacle.to_numpy()
        base_snapshot = solver.hibm_base_obstacle.to_numpy()
        for cell in static_cells:
            self.assertEqual(int(composite[cell]), 1)
            self.assertEqual(int(base_snapshot[cell]), 1)
        np.testing.assert_array_equal(composite, static_base)

    def test_dynamic_obstacle_revision_changes_only_when_mask_changes(self) -> None:
        solver = self._solver()
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.3125, 0.4375, 0.5625]], dtype=np.float32),
        )
        support_size_m = (0.08, 0.08, 0.08)
        initial_revision = solver.hibm_external_obstacle_topology_revision

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        installed_revision = solver.hibm_external_obstacle_topology_revision
        self.assertGreater(installed_revision, initial_revision)

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        self.assertEqual(
            solver.hibm_external_obstacle_topology_revision,
            installed_revision,
        )

        particle_position_m.from_numpy(
            np.asarray([[0.6875, 0.5625, 0.4375]], dtype=np.float32)
        )
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        self.assertGreater(
            solver.hibm_external_obstacle_topology_revision,
            installed_revision,
        )

    def test_dynamic_commit_reopens_cleanup_cell_and_advances_revision(self) -> None:
        solver = self._solver()
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.3125, 0.4375, 0.5625]], dtype=np.float32),
        )
        support_size_m = (0.08, 0.08, 0.08)
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        obstacle = solver.obstacle.to_numpy()
        cleanup_cell = (7, 7, 7)
        self.assertEqual(int(obstacle[cleanup_cell]), 0)
        obstacle[cleanup_cell] = 1
        solver.obstacle.from_numpy(obstacle)
        revision_before_recompose = (
            solver.hibm_external_obstacle_topology_revision
        )

        report = solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )

        self.assertEqual(report["fluid_dynamic_obstacle_added_cell_count"], 0)
        self.assertEqual(report["fluid_dynamic_obstacle_removed_cell_count"], 0)
        self.assertEqual(int(solver.obstacle[cleanup_cell]), 0)
        self.assertGreater(
            solver.hibm_external_obstacle_topology_revision,
            revision_before_recompose,
        )

    def test_dynamic_commit_invalidates_epoch_before_failing_reconstruction(self) -> None:
        solver = self._solver()
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.3125, 0.4375, 0.5625]], dtype=np.float32),
        )
        revision_before_write = solver.hibm_external_obstacle_topology_revision

        def fail_reconstruction() -> None:
            raise RuntimeError("synthetic reconstruction failure")

        solver._reconstruct_fresh_fluid_cells = fail_reconstruction
        with self.assertRaisesRegex(RuntimeError, "synthetic reconstruction"):
            solver.update_dynamic_solid_obstacle_from_particles(
                particle_position_m,
                particle_count=1,
                particle_support_size_m=(0.08, 0.08, 0.08),
                store_as_hibm_dynamic_solid_volume=True,
            )

        self.assertGreater(
            solver.hibm_external_obstacle_topology_revision,
            revision_before_write,
        )

    def test_repeated_identical_sphere_does_not_advance_revision(self) -> None:
        solver = self._solver()
        solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.2)
        installed_revision = solver.hibm_external_obstacle_topology_revision

        solver.mark_sphere_obstacle((0.5, 0.5, 0.5), 0.2)

        self.assertEqual(
            solver.hibm_external_obstacle_topology_revision,
            installed_revision,
        )

    def test_vacated_particle_cell_reconstructs_current_and_previous_velocity(
        self,
    ) -> None:
        solver = self._solver()
        grid_nodes = (8, 8, 8)
        index = np.indices(grid_nodes, dtype=np.float32)
        velocity = np.empty((*grid_nodes, 3), dtype=np.float32)
        velocity_prev = np.empty_like(velocity)
        velocity[..., 0] = 1.0 + 0.10 * index[0] + 0.01 * index[1]
        velocity[..., 1] = 2.0 + 0.20 * index[1] + 0.02 * index[2]
        velocity[..., 2] = 3.0 + 0.30 * index[2] + 0.03 * index[0]
        velocity_prev[..., 0] = 4.0 + 0.40 * index[0] + 0.04 * index[2]
        velocity_prev[..., 1] = 5.0 + 0.50 * index[1] + 0.05 * index[0]
        velocity_prev[..., 2] = 6.0 + 0.60 * index[2] + 0.06 * index[1]
        solver.velocity.from_numpy(velocity)
        solver.velocity_prev.from_numpy(velocity_prev)
        cell_a = (3, 3, 3)
        cell_b = (6, 3, 3)
        position_a = np.asarray([[(axis + 0.5) / 8.0 for axis in cell_a]], dtype=np.float32)
        position_b = np.asarray([[(axis + 0.5) / 8.0 for axis in cell_b]], dtype=np.float32)
        particle_position_m = self._particle_positions(1, position_a)
        support_size_m = (0.08, 0.08, 0.08)
        expected_velocity = self._six_neighbor_mean(velocity, cell_a)
        expected_velocity_prev = self._six_neighbor_mean(velocity_prev, cell_a)

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        particle_position_m.from_numpy(position_b)
        report = solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )

        obstacle = solver.obstacle.to_numpy()
        reconstructed_velocity = solver.velocity.to_numpy()[cell_a]
        reconstructed_velocity_prev = solver.velocity_prev.to_numpy()[cell_a]
        self.assertEqual(int(obstacle[cell_a]), 0)
        self.assertEqual(int(obstacle[cell_b]), 1)
        self.assertEqual(report["fluid_dynamic_obstacle_removed_cell_count"], 1)
        self.assertEqual(int(solver.hibm_fresh_fluid_cell[cell_a]), 1)
        self.assertTrue(np.all(np.isfinite(reconstructed_velocity)))
        self.assertTrue(np.all(np.isfinite(reconstructed_velocity_prev)))
        self.assertGreater(float(np.linalg.norm(reconstructed_velocity)), 0.0)
        self.assertGreater(float(np.linalg.norm(reconstructed_velocity_prev)), 0.0)
        np.testing.assert_allclose(
            reconstructed_velocity,
            expected_velocity,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            reconstructed_velocity_prev,
            expected_velocity_prev,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_optional_deformation_gradient_rotates_and_stretches_current_support(
        self,
    ) -> None:
        solver = self._solver(grid_nodes=(12, 12, 8))
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        )
        particle_deformation_gradient = self._particle_deformation_gradients(
            1,
            np.eye(3, dtype=np.float32)[None, ...],
        )
        rest_support_size_m = (0.10, 0.30, 0.10)

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=rest_support_size_m,
            store_as_hibm_dynamic_solid_volume=True,
        )
        undeformed_mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=rest_support_size_m,
            particle_deformation_gradient=particle_deformation_gradient,
            store_as_hibm_dynamic_solid_volume=True,
        )
        identity_mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        np.testing.assert_array_equal(identity_mask, undeformed_mask)

        quarter_turn_z = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        particle_deformation_gradient.from_numpy(quarter_turn_z[None, ...])
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=rest_support_size_m,
            particle_deformation_gradient=particle_deformation_gradient,
            store_as_hibm_dynamic_solid_volume=True,
        )
        rotated_mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        identity_extents = self._occupied_extents(identity_mask)
        rotated_extents = self._occupied_extents(rotated_mask)
        self.assertGreater(identity_extents[1], identity_extents[0])
        self.assertGreater(rotated_extents[0], rotated_extents[1])
        self.assertFalse(np.array_equal(rotated_mask, identity_mask))

        stretch_x = np.diag(np.asarray([2.0, 1.0, 1.0], dtype=np.float32))
        particle_deformation_gradient.from_numpy(stretch_x[None, ...])
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=rest_support_size_m,
            particle_deformation_gradient=particle_deformation_gradient,
            store_as_hibm_dynamic_solid_volume=True,
        )
        stretched_mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        self.assertGreater(
            self._occupied_extents(stretched_mask)[0],
            identity_extents[0],
        )
        self.assertGreater(
            int(np.count_nonzero(stretched_mask)),
            int(np.count_nonzero(identity_mask)),
        )

    def test_rotated_current_support_does_not_fill_its_axis_aligned_bounds(
        self,
    ) -> None:
        solver = self._solver(grid_nodes=(20, 20, 8))
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        )
        angle = np.pi / 4.0
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        particle_deformation_gradient = self._particle_deformation_gradients(
            1,
            rotation[None, ...],
        )

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=(0.40, 0.10, 0.10),
            particle_deformation_gradient=particle_deformation_gradient,
            store_as_hibm_dynamic_solid_volume=True,
        )

        mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        # The diagonal cell lies on the deformed long axis.  The cross-corner
        # lies inside the support's broad-phase AABB but outside the rotated
        # parallelepiped, so marking it would recreate an artificial halo.
        self.assertEqual(int(mask[12, 12, 4]), 1)
        self.assertEqual(int(mask[13, 6, 4]), 0)

    def test_identity_deformation_keeps_exact_face_touch_cells_fluid(self) -> None:
        solver = self._solver(grid_nodes=(10, 10, 10))
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        )
        identity = self._particle_deformation_gradients(
            1,
            np.eye(3, dtype=np.float32)[None, ...],
        )
        support = (0.2, 0.2, 0.2)
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support,
            store_as_hibm_dynamic_solid_volume=True,
        )
        axis_aligned = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support,
            particle_deformation_gradient=identity,
            store_as_hibm_dynamic_solid_volume=True,
        )
        deformed_identity = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()

        np.testing.assert_array_equal(deformed_identity, axis_aligned)
        self.assertEqual(int(deformed_identity[3, 4, 4]), 0)
        self.assertEqual(int(deformed_identity[6, 4, 4]), 0)
        self.assertEqual(self._occupied_extents(deformed_identity), (2, 2, 2))

    def test_deformed_support_intersection_is_scale_consistent_at_microns(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0e-5, 1.0e-5, 1.0e-5),
                grid_nodes=(10, 10, 10),
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-7,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        particle_position_m = self._particle_positions(
            1,
            np.asarray([[5.0e-6, 5.0e-6, 5.0e-6]], dtype=np.float32),
        )
        identity = self._particle_deformation_gradients(
            1,
            np.eye(3, dtype=np.float32)[None, ...],
        )
        support = (2.0e-6, 2.0e-6, 2.0e-6)

        solver.update_dynamic_solid_obstacle_from_particles(
            particle_position_m,
            particle_count=1,
            particle_support_size_m=support,
            particle_deformation_gradient=identity,
            store_as_hibm_dynamic_solid_volume=True,
        )

        mask = solver.hibm_dynamic_solid_volume_obstacle.to_numpy()
        self.assertEqual(int(np.count_nonzero(mask)), 8)
        self.assertEqual(self._occupied_extents(mask), (2, 2, 2))

    def test_particle_and_deformation_gradient_capacity_fail_fast(self) -> None:
        solver = self._solver()
        particle_position_m = self._particle_positions(
            2,
            np.asarray(
                [[0.3, 0.4, 0.5], [0.6, 0.5, 0.4]],
                dtype=np.float32,
            ),
        )
        one_gradient = self._particle_deformation_gradients(
            1,
            np.eye(3, dtype=np.float32)[None, ...],
        )
        obstacle_before = solver.obstacle.to_numpy()

        with self.assertRaisesRegex(ValueError, "particle_count.*capacity"):
            solver.update_dynamic_solid_obstacle_from_particles(
                particle_position_m,
                particle_count=3,
                particle_support_size_m=(0.1, 0.1, 0.1),
            )
        np.testing.assert_array_equal(solver.obstacle.to_numpy(), obstacle_before)
        with self.assertRaisesRegex(ValueError, "deformation.*capacity"):
            solver.update_dynamic_solid_obstacle_from_particles(
                particle_position_m,
                particle_count=2,
                particle_support_size_m=(0.1, 0.1, 0.1),
                particle_deformation_gradient=one_gradient,
            )
        np.testing.assert_array_equal(solver.obstacle.to_numpy(), obstacle_before)


if __name__ == "__main__":
    unittest.main()
