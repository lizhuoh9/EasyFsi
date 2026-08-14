from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from simulation_core import (
    HibmMpmSurfaceMarkers,
    NeoHookeanMpmState,
    TaichiRuntimeConfig,
)


RUNTIME = TaichiRuntimeConfig(arch="cuda")
CORE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "simulation_core"
    / "coupling"
    / "hibm_mpm"
    / "core.py"
).read_text(encoding="utf-8")


def _solid_with_particles(positions: np.ndarray) -> NeoHookeanMpmState:
    particle_count = int(positions.shape[0])
    solid = NeoHookeanMpmState(
        particle_capacity=particle_count,
        bounds_min_m=(-100.0, -100.0, -100.0),
        bounds_max_m=(100.0, 100.0, 100.0),
        grid_nodes=(4, 4, 4),
        runtime=RUNTIME,
    )
    solid.particle_count = particle_count
    solid.x.from_numpy(np.asarray(positions, dtype=np.float32))
    return solid


def _weights(marker: np.ndarray, particles: np.ndarray, radius: float) -> np.ndarray:
    relative = np.abs(particles - marker[None, :])
    within = np.all(relative < radius, axis=1)
    weights = np.prod(np.maximum(1.0 - relative / radius, 0.0), axis=1)
    return np.where(within, weights, 0.0)


class HibmMpmParticleBinTests(unittest.TestCase):
    def test_particle_bins_and_force_scatter_use_stable_gpu_order(self) -> None:
        fill_kernel = CORE_SOURCE.split(
            "def _fill_mpm_particle_bin_members_kernel(", 1
        )[1].split("@ti.kernel", 1)[0]
        scatter_kernel = CORE_SOURCE.split(
            "def _scatter_marker_forces_to_mpm_particles_kernel(", 1
        )[1].split("@ti.kernel", 1)[0]

        self.assertIn(
            "ti.loop_config(serialize=True)\n"
            "        for particle in range(particle_count):",
            fill_kernel,
        )
        self.assertIn(
            "ti.loop_config(serialize=True)\n"
            "        for source_marker in range(marker_count + tip_cap_marker_count):",
            scatter_kernel,
        )

    def test_scatter_matches_bruteforce_and_conserves_each_marker_force(self) -> None:
        marker_positions = np.asarray(
            [[-0.15, 0.0, 0.0], [0.55, 0.05, 0.0]], dtype=np.float32
        )
        marker_forces = np.asarray(
            [[2.0, -1.0, 0.5], [-0.25, 0.75, 1.5]], dtype=np.float32
        )
        particle_positions = np.asarray(
            [
                [-0.30, 0.0, 0.0],
                [-0.05, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [0.45, 0.05, 0.0],
                [0.70, 0.05, 0.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        radius = 0.4
        markers = HibmMpmSurfaceMarkers(marker_capacity=2, runtime=RUNTIME)
        markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=np.zeros((2, 3), dtype=np.float32),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(1.0, 1.0),
            region_ids=(7, 8),
        )
        markers.set_marker_tractions_pa(marker_forces)
        markers.compute_marker_forces()
        solid = _solid_with_particles(particle_positions)

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=radius,
        )

        expected = np.zeros_like(particle_positions)
        expected_pairs = 0
        for marker, force in zip(marker_positions, marker_forces, strict=True):
            weights = _weights(marker, particle_positions, radius)
            expected_pairs += int(np.count_nonzero(weights > 0.0))
            expected += weights[:, None] / np.sum(weights) * force[None, :]
        np.testing.assert_allclose(
            solid.external_force_n.to_numpy(), expected, rtol=2.0e-6, atol=2.0e-6
        )
        self.assertEqual(report.active_pair_count, expected_pairs)
        self.assertLessEqual(report.action_reaction_residual_n, 3.0e-6)

    def test_sparse_scatter_candidate_count_tracks_unique_pairs(self) -> None:
        marker_count = 32
        particle_count = 2048
        marker_positions = np.zeros((marker_count, 3), dtype=np.float32)
        marker_positions[:, 0] = np.arange(marker_count, dtype=np.float32) * 2.0
        particle_positions = np.full((particle_count, 3), 80.0, dtype=np.float32)
        particle_positions[:marker_count] = marker_positions
        markers = HibmMpmSurfaceMarkers(marker_capacity=marker_count, runtime=RUNTIME)
        markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=np.zeros((marker_count, 3), dtype=np.float32),
            normals=((1.0, 0.0, 0.0),) * marker_count,
            areas_m2=(1.0,) * marker_count,
            region_ids=(7,) * marker_count,
        )
        markers.set_marker_tractions_pa(((1.0, 0.0, 0.0),) * marker_count)
        markers.compute_marker_forces()
        solid = _solid_with_particles(particle_positions)

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=particle_count,
            support_radius_m=0.25,
        )

        candidate_tests = int(markers.report_mpm_scatter_candidate_pair_count[None])
        self.assertEqual(report.candidate_pair_count, marker_count)
        self.assertEqual(candidate_tests, report.candidate_pair_count)
        self.assertLess(candidate_tests, marker_count * particle_count // 16)

    def test_surface_feedback_matches_bruteforce_velocity_normal_and_area(self) -> None:
        marker_position = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        particle_positions = np.asarray(
            [[-0.2, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        particle_velocities = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [9.0, 9.0, 9.0]],
            dtype=np.float32,
        )
        particle_normals = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        particle_areas = np.asarray([2.0, 4.0, 8.0], dtype=np.float32)
        radius = 0.5
        dt = 0.01
        markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
        markers.load_markers(
            positions_m=marker_position,
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        solid = _solid_with_particles(particle_positions)
        solid.v.from_numpy(particle_velocities)
        solid.surface_normal.from_numpy(particle_normals)
        solid.area_weight_m2.from_numpy(particle_areas)

        report = markers.update_surface_feedback_from_mpm_surface_particles(
            solid.x,
            solid.v,
            solid.surface_normal,
            solid.area_weight_m2,
            particle_count=solid.particle_count,
            support_radius_m=radius,
            dt_s=dt,
        )

        weights = _weights(marker_position[0], particle_positions, radius)
        weight_sum = float(np.sum(weights))
        expected_velocity = np.sum(weights[:, None] * particle_velocities, axis=0) / weight_sum
        expected_normal_sum = np.sum(weights[:, None] * particle_normals, axis=0)
        expected_normal = expected_normal_sum / np.linalg.norm(expected_normal_sum)
        expected_area = float(np.sum(weights * particle_areas) / weight_sum)
        np.testing.assert_allclose(markers.v_gamma_mps.to_numpy()[0], expected_velocity, atol=2e-6)
        np.testing.assert_allclose(
            markers.x_gamma_m.to_numpy()[0], marker_position[0] + dt * expected_velocity, atol=2e-6
        )
        np.testing.assert_allclose(markers.n_gamma.to_numpy()[0], expected_normal, atol=2e-6)
        self.assertAlmostEqual(float(markers.A_gamma_m2[0]), expected_area, delta=2e-6)
        self.assertEqual(report.updated_marker_count, 1)
        self.assertEqual(report.geometry_updated_marker_count, 1)

    def test_particle_count_exceeding_field_capacity_fails_before_kernel(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1, runtime=RUNTIME)
        markers.load_markers(
            positions_m=((0.0, 0.0, 0.0),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        solid = _solid_with_particles(np.zeros((1, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "particle_count.*capacity"):
            markers.scatter_marker_forces_to_mpm_particles(
                solid.external_force_n,
                solid.x,
                particle_count=2,
                support_radius_m=0.5,
            )


if __name__ == "__main__":
    unittest.main()
