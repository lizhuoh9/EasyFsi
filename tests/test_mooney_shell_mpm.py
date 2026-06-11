from __future__ import annotations

import inspect
import unittest

import numpy as np

from simulation_core.geometry import SurfaceMesh, UvSphereResolution, make_uv_sphere
from simulation_core.mooney_shell_mpm import TriMooneyShellMpmState, UvMooneyShellMpmState


def _triangle_mooney_membrane_energy(
    rest_vertices: np.ndarray,
    current_vertices: np.ndarray,
    *,
    thickness_m: float,
    c1_pa: float,
    c2_pa: float,
) -> float:
    rest_a, rest_b, rest_c = rest_vertices
    current_a, current_b, current_c = current_vertices
    rest_area_vec = np.cross(rest_b - rest_a, rest_c - rest_a)
    rest_area = 0.5 * float(np.linalg.norm(rest_area_vec))
    rest_normal = rest_area_vec / np.linalg.norm(rest_area_vec)
    rest_edge0 = rest_b - rest_a
    rest_t0 = rest_edge0 / np.linalg.norm(rest_edge0)
    rest_t1 = np.cross(rest_normal, rest_t0)
    rest_ac = rest_c - rest_a
    rest_xc = float(np.dot(rest_ac, rest_t0))
    rest_yc = float(np.dot(rest_ac, rest_t1))
    inv_dm = np.array(
        [
            [1.0 / np.linalg.norm(rest_edge0), -rest_xc / (np.linalg.norm(rest_edge0) * rest_yc)],
            [0.0, 1.0 / rest_yc],
        ],
        dtype=np.float64,
    )
    ds = np.column_stack((current_b - current_a, current_c - current_a))
    deformation_gradient = ds @ inv_dm
    right_cauchy_green = deformation_gradient.T @ deformation_gradient
    det_c = float(np.linalg.det(right_cauchy_green))
    trace_c = float(np.trace(right_cauchy_green))
    invariant_1 = trace_c + 1.0 / det_c
    invariant_2 = det_c + trace_c / det_c
    return thickness_m * rest_area * (c1_pa * (invariant_1 - 3.0) + c2_pa * (invariant_2 - 3.0))


def _finite_difference_mooney_force(
    rest_vertices: np.ndarray,
    current_vertices: np.ndarray,
    *,
    thickness_m: float,
    c1_pa: float,
    c2_pa: float,
    eps: float = 1.0e-5,
) -> np.ndarray:
    force = np.zeros_like(current_vertices, dtype=np.float64)
    for vertex_id in range(current_vertices.shape[0]):
        for axis in range(3):
            plus = current_vertices.copy()
            minus = current_vertices.copy()
            plus[vertex_id, axis] += eps
            minus[vertex_id, axis] -= eps
            force[vertex_id, axis] = -(
                _triangle_mooney_membrane_energy(
                    rest_vertices,
                    plus,
                    thickness_m=thickness_m,
                    c1_pa=c1_pa,
                    c2_pa=c2_pa,
                )
                - _triangle_mooney_membrane_energy(
                    rest_vertices,
                    minus,
                    thickness_m=thickness_m,
                    c1_pa=c1_pa,
                    c2_pa=c2_pa,
                )
            ) / (2.0 * eps)
    return force


def _uv_sphere_faces(resolution: UvSphereResolution) -> np.ndarray:
    def ring_index(ring: int, segment: int) -> int:
        return 1 + (ring - 1) * resolution.longitude_segments + segment % resolution.longitude_segments

    faces: list[tuple[int, int, int]] = []
    top_index = 0
    bottom_index = resolution.vertex_count - 1
    last_ring = resolution.latitude_bands - 1
    for segment in range(resolution.longitude_segments):
        faces.append((top_index, ring_index(1, segment), ring_index(1, segment + 1)))
        faces.append(
            (
                bottom_index,
                ring_index(last_ring, segment + 1),
                ring_index(last_ring, segment),
            )
        )
    for ring in range(resolution.latitude_bands - 2):
        current_ring = ring + 1
        next_ring = current_ring + 1
        for segment in range(resolution.longitude_segments):
            a = ring_index(current_ring, segment)
            b = ring_index(current_ring, segment + 1)
            c = ring_index(next_ring, segment + 1)
            d = ring_index(next_ring, segment)
            faces.append((a, d, c))
            faces.append((a, c, b))
    return np.asarray(faces, dtype=np.int32)


def _uv_mooney_membrane_energy(
    rest_vertices: np.ndarray,
    current_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    thickness_m: float,
    c1_pa: float,
    c2_pa: float,
) -> float:
    return float(
        sum(
            _triangle_mooney_membrane_energy(
                rest_vertices[face],
                current_vertices[face],
                thickness_m=thickness_m,
                c1_pa=c1_pa,
                c2_pa=c2_pa,
            )
            for face in faces
        )
    )


def _finite_difference_uv_mooney_force(
    rest_vertices: np.ndarray,
    current_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    thickness_m: float,
    c1_pa: float,
    c2_pa: float,
    eps: float = 1.0e-5,
) -> np.ndarray:
    force = np.zeros_like(current_vertices, dtype=np.float64)
    for vertex_id in range(current_vertices.shape[0]):
        for axis in range(3):
            plus = current_vertices.copy()
            minus = current_vertices.copy()
            plus[vertex_id, axis] += eps
            minus[vertex_id, axis] -= eps
            force[vertex_id, axis] = -(
                _uv_mooney_membrane_energy(
                    rest_vertices,
                    plus,
                    faces,
                    thickness_m=thickness_m,
                    c1_pa=c1_pa,
                    c2_pa=c2_pa,
                )
                - _uv_mooney_membrane_energy(
                    rest_vertices,
                    minus,
                    faces,
                    thickness_m=thickness_m,
                    c1_pa=c1_pa,
                    c2_pa=c2_pa,
                )
            ) / (2.0 * eps)
    return force


class UvMooneyShellMpmStateTests(unittest.TestCase):
    def test_uv_shell_exposes_unit_surface_normals_for_hibm_markers(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )

        top = tuple(float(state.surface_normal[0][axis]) for axis in range(3))
        bottom = tuple(
            float(state.surface_normal[state.particle_count - 1][axis])
            for axis in range(3)
        )

        self.assertGreater(top[2], 0.8)
        self.assertLess(bottom[2], -0.8)
        for particle in (0, state.particle_count - 1):
            norm = float(state.surface_normal[particle].norm())
            self.assertAlmostEqual(norm, 1.0, delta=1.0e-5)

    def test_rest_sphere_zero_pressure_stays_at_unit_radius(self) -> None:
        resolution = UvSphereResolution(latitude_bands=8, longitude_segments=25)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(24, 24, 24),
        )

        report = state.step(dt_s=1.0e-3, pressure_pa=0.0, velocity_damping=1.0)

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertEqual(report.particle_count, resolution.vertex_count)
        self.assertEqual(report.edge_count, 3 * resolution.longitude_segments * (resolution.latitude_bands - 1))
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_edge_strain, 0.0, delta=1.0e-6)
        self.assertAlmostEqual(report.internal_force_rms_n, 0.0, delta=1.0e-5)

    def test_uv_step_applies_body_acceleration_to_particle_velocity(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=2.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )

        state.step(
            dt_s=2.0e-3,
            pressure_pa=0.0,
            velocity_damping=1.0,
            flip_blend=1.0,
            body_acceleration_mps2=(0.0, 0.0, -3.0),
        )

        velocities = state.v.to_numpy()
        self.assertAlmostEqual(float(np.max(np.abs(velocities[:, 0]))), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(float(np.max(np.abs(velocities[:, 1]))), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(float(np.mean(velocities[:, 2])), -6.0e-3, delta=2.0e-5)

    def test_uv_report_reads_packed_host_snapshot(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )
        state.step(dt_s=1.0e-3, pressure_pa=0.0, velocity_damping=1.0)

        state.report_host_snapshot[8] = 123.0
        report = state.report()

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertAlmostEqual(report.total_mass_kg, 123.0)

    def test_internal_pressure_expands_mooney_shell(self) -> None:
        resolution = UvSphereResolution(latitude_bands=8, longitude_segments=25)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(36, 36, 36),
        )

        report = state.step(dt_s=1.0e-3, pressure_pa=2.0, velocity_damping=1.0)

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertGreater(report.mean_radial_stretch, 1.0)
        self.assertGreater(report.max_speed_mps, 0.0)
        self.assertGreater(report.active_grid_nodes, 0)

    def test_radial_stretch_diagnostic_is_translation_invariant(self) -> None:
        resolution = UvSphereResolution(latitude_bands=8, longitude_segments=25)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(36, 36, 36),
        )
        rigid_translation = np.array([0.2, -0.15, 0.08], dtype=np.float32)
        translated = state.rest_x.to_numpy() + rigid_translation
        state.x.from_numpy(translated.astype(np.float32))

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=2.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=2.0e-6)

    def test_uv_radial_stretch_diagnostic_ignores_out_of_bounds_particle(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )
        translated = state.rest_x.to_numpy() + np.array([0.2, -0.15, 0.08], dtype=np.float32)
        translated[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(translated.astype(np.float32))

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertTrue(np.isfinite(report.mean_radial_stretch))
        self.assertTrue(np.isfinite(report.max_radial_stretch_error))
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=2.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=2.0e-6)

    def test_velocity_damping_does_not_pollute_transfer_diagnostic(self) -> None:
        resolution = UvSphereResolution(latitude_bands=8, longitude_segments=25)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(36, 36, 36),
        )
        initial_velocity = np.zeros((resolution.vertex_count, 3), dtype=np.float32)
        initial_velocity[:, 0] = 0.05
        state.v.from_numpy(initial_velocity)

        report = state.step(dt_s=1.0e-3, pressure_pa=0.0, velocity_damping=0.95)

        self.assertGreater(report.max_speed_mps, 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_uv_out_of_bounds_particles_do_not_deposit_to_clamped_boundary_nodes(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )
        positions = state.x.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))
        particle_mass = state.mass_kg.to_numpy()

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        deposited_mass = float(np.sum(state.grid_mass_kg.to_numpy()))

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertAlmostEqual(
            deposited_mass,
            float(np.sum(particle_mass[1:])),
            delta=max(float(np.sum(particle_mass)) * 1.0e-6, 1.0e-12),
        )

    def test_uv_out_of_bounds_particle_with_inward_velocity_can_reenter_grid(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
        )
        dt_s = 0.1
        positions = state.x.to_numpy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        positions[0, 0] = float(state.bounds_max[0] + 0.5 * state.dx[0])
        velocities[0, 0] = float(-2.0 * state.dx[0] / dt_s)
        state.x.from_numpy(positions.astype(np.float32))
        state.v.from_numpy(velocities)

        report = state.step(dt_s=dt_s, pressure_pa=0.0, velocity_damping=1.0)
        recovered_position = state.x.to_numpy()[0]

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertLessEqual(float(recovered_position[0]), float(state.bounds_max[0]))

    def test_anisotropic_uv_sphere_force_matches_mooney_energy_gradient(self) -> None:
        resolution = UvSphereResolution(latitude_bands=4, longitude_segments=8)
        thickness_m = 0.05
        c1_pa = 20.0
        c2_pa = 10.0
        state = UvMooneyShellMpmState(
            resolution,
            radius_m=1.0,
            thickness_m=thickness_m,
            density_kgm3=1.0,
            c1_pa=c1_pa,
            c2_pa=c2_pa,
            grid_nodes=(24, 24, 24),
        )
        rest_vertices = state.rest_x.to_numpy().astype(np.float64)
        faces = _uv_sphere_faces(resolution)
        current_vertices = rest_vertices.copy()
        deformation = np.array(
            [
                [1.20, 0.18, 0.00],
                [0.00, 0.85, 0.10],
                [0.00, 0.00, 1.00],
            ],
            dtype=np.float64,
        )
        current_vertices = current_vertices @ deformation.T
        state.x.from_numpy(current_vertices.astype(np.float32))

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        internal_force = state.internal_force_n.to_numpy().astype(np.float64)
        expected_force = _finite_difference_uv_mooney_force(
            rest_vertices,
            current_vertices,
            faces,
            thickness_m=thickness_m,
            c1_pa=c1_pa,
            c2_pa=c2_pa,
        )

        self.assertGreater(float(np.linalg.norm(expected_force)), 1.0e-2)
        np.testing.assert_allclose(internal_force, expected_force, rtol=2.0e-3, atol=4.0e-5)
        np.testing.assert_allclose(np.sum(internal_force, axis=0), np.zeros(3), atol=2.0e-5)


class TriMooneyShellMpmStateTests(unittest.TestCase):
    def test_tri_shell_exposes_unit_surface_normals_for_hibm_markers(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )

        for particle in range(state.particle_count):
            normal = tuple(float(state.surface_normal[particle][axis]) for axis in range(3))
            self.assertAlmostEqual(normal[0], 0.0, delta=1.0e-6)
            self.assertAlmostEqual(normal[1], 0.0, delta=1.0e-6)
            self.assertAlmostEqual(normal[2], 1.0, delta=1.0e-6)

    def test_tri_shell_updates_surface_normals_after_current_shape_changes(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        tilted = mesh.vertices.copy()
        tilted[2] = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        state.x.from_numpy(tilted.astype(np.float32))

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        expected = np.cross(tilted[1] - tilted[0], tilted[2] - tilted[0])
        expected = expected / np.linalg.norm(expected)
        for particle in range(state.particle_count):
            actual = np.array(
                [float(state.surface_normal[particle][axis]) for axis in range(3)]
            )
            np.testing.assert_allclose(actual, expected, atol=1.0e-5)

    def test_tri_shell_updates_surface_area_weights_after_current_shape_changes(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        rest_share = float(state.area_weight_m2[0])
        stretched = mesh.vertices.copy()
        stretched[2] = np.array([0.0, 2.0, 0.0], dtype=np.float64)
        state.x.from_numpy(stretched.astype(np.float32))

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        expected_share = 1.0 / 3.0
        self.assertAlmostEqual(rest_share, 0.5 / 3.0, delta=1.0e-6)
        for particle in range(state.particle_count):
            self.assertAlmostEqual(
                float(state.area_weight_m2[particle]),
                expected_share,
                delta=1.0e-5,
            )

    def test_equal_biaxial_triangle_force_matches_mooney_area_gradient(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        thickness_m = 0.05
        c1_pa = 20.0
        c2_pa = 10.0
        stretch = 1.2
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=thickness_m,
            density_kgm3=1.0,
            c1_pa=c1_pa,
            c2_pa=c2_pa,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        state.x.from_numpy((mesh.vertices * stretch).astype(np.float32))

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        internal_force = state.internal_force_n.to_numpy()

        expected_tension_npm = (
            2.0
            * thickness_m
            * (1.0 - stretch**-6)
            * (c1_pa + stretch * stretch * c2_pa)
        )
        expected_force_a = np.array(
            [0.5 * stretch * expected_tension_npm, 0.5 * stretch * expected_tension_npm, 0.0],
            dtype=np.float32,
        )
        np.testing.assert_allclose(internal_force[0], expected_force_a, rtol=5.0e-4, atol=1.0e-6)

    def test_area_preserving_in_plane_stretch_matches_mooney_energy_gradient(self) -> None:
        rest_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        mesh = SurfaceMesh(
            vertices=rest_vertices,
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        thickness_m = 0.05
        c1_pa = 20.0
        c2_pa = 10.0
        stretch = 1.35
        current_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [stretch, 0.0, 0.0],
                [0.0, 1.0 / stretch, 0.0],
            ],
            dtype=np.float64,
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=thickness_m,
            density_kgm3=1.0,
            c1_pa=c1_pa,
            c2_pa=c2_pa,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        state.x.from_numpy(current_vertices.astype(np.float32))

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        internal_force = state.internal_force_n.to_numpy()
        expected_force = _finite_difference_mooney_force(
            rest_vertices,
            current_vertices,
            thickness_m=thickness_m,
            c1_pa=c1_pa,
            c2_pa=c2_pa,
        )

        self.assertAlmostEqual(float(np.linalg.det(current_vertices[1:, :2])), 1.0, delta=1.0e-12)
        self.assertGreater(float(np.linalg.norm(expected_force)), 1.0e-2)
        np.testing.assert_allclose(internal_force, expected_force, rtol=2.0e-3, atol=2.0e-5)
        np.testing.assert_allclose(np.sum(internal_force, axis=0), np.zeros(3), atol=1.0e-6)

    def test_degenerate_current_triangle_keeps_finite_mooney_restoring_force(
        self,
    ) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=0.0,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=1.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        collapsed = mesh.vertices.copy()
        collapsed[1] = collapsed[0]
        state.x.from_numpy(collapsed.astype(np.float32))

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        internal_force = state.internal_force_n.to_numpy()

        self.assertTrue(np.all(np.isfinite(internal_force)))
        self.assertGreater(report.internal_force_rms_n, 1.0e-6)
        np.testing.assert_allclose(np.sum(internal_force, axis=0), np.zeros(3), atol=1.0e-6)

    def test_rest_triangle_sphere_zero_pressure_stays_at_unit_radius(self) -> None:
        mesh = make_uv_sphere(UvSphereResolution(latitude_bands=8, longitude_segments=25), 1.0)
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(24, 24, 24),
            primary_region_id=1,
            secondary_region_id=2,
        )

        report = state.step(dt_s=1.0e-3, pressure_pa=0.0, velocity_damping=1.0)

        self.assertEqual(report.particle_count, mesh.vertex_count)
        self.assertEqual(report.face_count, mesh.face_count)
        self.assertGreater(report.edge_count, 0)
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_edge_strain, 0.0, delta=1.0e-6)
        self.assertAlmostEqual(report.internal_force_rms_n, 0.0, delta=1.0e-5)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_radial_stretch_diagnostic_is_translation_invariant(self) -> None:
        rest_vertices = np.array(
            [
                [10.0, -4.0, 2.0],
                [11.0, -4.0, 2.0],
                [11.0, -3.0, 2.0],
                [10.0, -3.0, 2.0],
            ],
            dtype=np.float64,
        )
        mesh = SurfaceMesh(
            vertices=rest_vertices,
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=0.5,
            primary_region_id=1,
            secondary_region_id=2,
        )
        rigid_translation = np.array([0.2, -0.1, 0.3], dtype=np.float32)
        translated_vertices = rest_vertices.astype(np.float32) + rigid_translation
        state.x.from_numpy(translated_vertices)
        state.u.from_numpy(np.broadcast_to(rigid_translation, translated_vertices.shape).copy())

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=2.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=2.0e-6)

    def test_tri_radial_stretch_diagnostic_ignores_out_of_bounds_particle(self) -> None:
        rest_vertices = np.array(
            [
                [10.0, -4.0, 2.0],
                [11.0, -4.0, 2.0],
                [11.0, -3.0, 2.0],
                [10.0, -3.0, 2.0],
            ],
            dtype=np.float64,
        )
        mesh = SurfaceMesh(
            vertices=rest_vertices,
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=0.5,
            primary_region_id=1,
            secondary_region_id=2,
        )
        rigid_translation = np.array([0.2, -0.1, 0.3], dtype=np.float32)
        translated_vertices = rest_vertices.astype(np.float32) + rigid_translation
        translated_vertices[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(translated_vertices)
        state.u.from_numpy(np.broadcast_to(rigid_translation, translated_vertices.shape).copy())

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertTrue(np.isfinite(report.mean_radial_stretch))
        self.assertTrue(np.isfinite(report.max_radial_stretch_error))
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=2.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=2.0e-6)

    def test_tri_region_mean_excludes_out_of_bounds_particle(self) -> None:
        rest_vertices = np.array(
            [
                [-0.004, -0.004, 0.0],
                [0.004, -0.004, 0.0],
                [-0.004, 0.004, 0.0],
                [0.010, -0.004, 0.0],
                [0.018, -0.004, 0.0],
                [0.010, 0.004, 0.0],
            ],
            dtype=np.float64,
        )
        mesh = SurfaceMesh(
            vertices=rest_vertices,
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7, 8], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )
        current = rest_vertices.astype(np.float32)
        displacement = np.zeros_like(current)
        displacement[0] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        displacement[1] = np.array([0.001, 0.0, 0.0], dtype=np.float32)
        displacement[2] = np.array([0.003, 0.0, 0.0], dtype=np.float32)
        current += displacement
        current[0] = np.array([state.bounds_max[0] + 2.0 * state.dx[0], 0.0, 0.0], dtype=np.float32)
        velocity = np.zeros_like(current)
        velocity[0] = np.array([7.0, 0.0, 0.0], dtype=np.float32)
        velocity[1] = np.array([0.02, 0.0, 0.0], dtype=np.float32)
        velocity[2] = np.array([0.04, 0.0, 0.0], dtype=np.float32)
        state.x.from_numpy(current)
        state.u.from_numpy(displacement)
        state.v.from_numpy(velocity)

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        displacement_after = state.u.to_numpy()
        velocity_after = state.v.to_numpy()
        np.testing.assert_allclose(
            report.primary_mean_displacement_m,
            displacement_after[[1, 2]].mean(axis=0),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            report.primary_mean_velocity_mps,
            velocity_after[[1, 2]].mean(axis=0),
            atol=1.0e-7,
        )
        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertLess(abs(report.primary_mean_velocity_mps[0]), 1.0)

    def test_out_of_bounds_particle_mapping_is_reported(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=0.25,
            primary_region_id=1,
            secondary_region_id=2,
        )
        positions = state.x.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)

    def test_out_of_bounds_particles_do_not_deposit_to_clamped_boundary_nodes(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=0.25,
            primary_region_id=1,
            secondary_region_id=2,
        )
        positions = state.x.to_numpy()
        positions[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))
        particle_mass = state.mass_kg.to_numpy()

        report = state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        deposited_mass = float(np.sum(state.grid_mass_kg.to_numpy()))

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertAlmostEqual(
            deposited_mass,
            float(np.sum(particle_mass[1:])),
            delta=max(float(np.sum(particle_mass)) * 1.0e-6, 1.0e-12),
        )

    def test_out_of_bounds_particle_with_inward_velocity_can_reenter_grid(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(12, 12, 12),
            bounds_padding_fraction=0.25,
            primary_region_id=1,
            secondary_region_id=2,
        )
        dt_s = 0.1
        positions = state.x.to_numpy()
        velocities = np.zeros_like(positions, dtype=np.float32)
        positions[0, 0] = float(state.bounds_max[0] + 0.5 * state.dx[0])
        velocities[0, 0] = float(-2.0 * state.dx[0] / dt_s)
        state.x.from_numpy(positions.astype(np.float32))
        state.v.from_numpy(velocities)

        report = state.step(dt_s=dt_s, pressure_pa=0.0, velocity_damping=1.0)
        recovered_position = state.x.to_numpy()[0]

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertLessEqual(float(recovered_position[0]), float(state.bounds_max[0]))

    def test_internal_pressure_expands_generic_triangle_mooney_shell(self) -> None:
        mesh = make_uv_sphere(UvSphereResolution(latitude_bands=8, longitude_segments=25), 1.0)
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.05,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(36, 36, 36),
            primary_region_id=1,
            secondary_region_id=2,
        )

        report = state.step(dt_s=1.0e-3, pressure_pa=2.0, velocity_damping=1.0)

        self.assertGreater(report.mean_radial_stretch, 1.0)
        self.assertGreater(report.max_speed_mps, 0.0)
        self.assertGreater(report.active_grid_nodes, 0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_step_body_acceleration_reports_mass_weighted_force(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.008, 0.0, 0.0],
                    [0.0, 0.008, 0.0],
                ],
                dtype=np.float32,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            grid_nodes=(16, 16, 16),
        )

        report = state.step(
            dt_s=1.0e-4,
            pressure_pa=0.0,
            velocity_damping=1.0,
            flip_blend=1.0,
            body_acceleration_mps2=(0.0, -2.0, -9.81),
        )

        self.assertAlmostEqual(report.total_force_n[0], 0.0, delta=1.0e-10)
        self.assertAlmostEqual(
            report.total_force_n[1],
            -2.0 * report.total_mass_kg,
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            report.total_force_n[2],
            -9.81 * report.total_mass_kg,
            delta=1.0e-9,
        )
        self.assertLess(report.primary_mean_velocity_mps[2], 0.0)

    def test_velocity_damping_does_not_pollute_transfer_diagnostic(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        report = state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            velocity_damping=0.95,
        )

        self.assertGreater(report.max_speed_mps, 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_region_load_step_can_skip_host_report_read(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        skipped = state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            read_report=False,
        )

        self.assertIsNone(skipped)
        self.assertEqual(state.last_report_host_reads, 0)
        report = state.report()
        self.assertEqual(state.last_report_host_reads, 1)
        self.assertGreater(report.max_speed_mps, 0.0)

    def test_tri_report_reads_packed_host_snapshot(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.002, 0.0, 0.0],
                    [0.0, 0.002, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )
        state.step(dt_s=1.0e-4, pressure_pa=0.0, velocity_damping=1.0)

        state.report_host_snapshot[4] = 0.0125
        report = state.report()

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertAlmostEqual(report.total_area_m2, 0.0125)

    def test_fixed_particles_do_not_deposit_stationary_mass_into_free_particle_grid_nodes(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.00001],
                    [0.001, 0.0, 0.00001],
                    [0.0, 0.001, 0.00001],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7, 5], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=10.0,
        )
        velocities = np.zeros((6, 3), dtype=np.float32)
        velocities[:3, 0] = 1.0
        state.v.from_numpy(velocities)

        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0, flip_blend=0.0)
        updated = state.v.to_numpy()

        np.testing.assert_allclose(updated[:3, 0], np.ones(3), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(updated[3:], np.zeros((3, 3)), atol=1.0e-7)

    def test_region_zero_vertices_are_not_treated_as_unassigned(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [-0.001, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([0, 7], dtype=np.int32),
            primary_region_id=0,
            secondary_region_id=7,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=10.0,
        )

        vertex_regions = state.vertex_region_id.to_numpy()
        self.assertEqual(int(vertex_regions[0]), 0)
        self.assertEqual(int(vertex_regions[1]), 0)
        self.assertEqual(int(vertex_regions[2]), 0)
        self.assertEqual(int(vertex_regions[3]), 7)

    def test_region_load_api_has_no_old_z_only_alias(self) -> None:
        parameters = inspect.signature(TriMooneyShellMpmState.advance_region_loads).parameters

        self.assertIn("primary_area_load_npm2", parameters)
        self.assertIn("primary_interface_reaction_n", parameters)
        self.assertIn("secondary_interface_reaction_n", parameters)
        self.assertNotIn("primary_fluid_feedback_n", parameters)
        self.assertNotIn("secondary_fluid_feedback_n", parameters)
        self.assertNotIn("pressure_pa", parameters)
        self.assertFalse(hasattr(TriMooneyShellMpmState, "advance_regions_z_loads"))

    def test_tri_mooney_constructor_has_no_radial_origin_override(self) -> None:
        parameters = inspect.signature(TriMooneyShellMpmState).parameters

        self.assertNotIn("radial_origin_m", parameters)

    def test_mooney_force_scale_is_named_for_membrane_not_edge_springs(self) -> None:
        tri_parameters = inspect.signature(TriMooneyShellMpmState).parameters
        uv_parameters = inspect.signature(UvMooneyShellMpmState).parameters

        self.assertIn("membrane_force_scale", tri_parameters)
        self.assertIn("membrane_force_scale", uv_parameters)
        self.assertNotIn("edge_force_scale", tri_parameters)
        self.assertNotIn("edge_force_scale", uv_parameters)

    def test_region_load_requires_explicit_area_load(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        with self.assertRaises(TypeError):
            state.advance_region_loads(
                dt_s=1.0e-4,
                primary_region_id=7,
                secondary_region_id=8,
                primary_interface_reaction_n=(0.0, 0.0, 0.0),
                secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            )

    def test_flip_blend_preserves_particle_velocity_without_grid_force(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.001, -0.001, 0.0],
                    [0.001, -0.001, 0.0],
                    [-0.001, 0.001, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=10.0,
            primary_region_id=1,
            secondary_region_id=2,
        )
        initial_velocity = np.array(
            [
                [0.012, -0.004, 0.001],
                [-0.006, 0.008, -0.003],
                [0.002, 0.005, 0.009],
            ],
            dtype=np.float32,
        )
        state.v.from_numpy(initial_velocity)

        state.step(
            dt_s=0.0,
            pressure_pa=0.0,
            velocity_damping=1.0,
            flip_blend=1.0,
        )

        np.testing.assert_allclose(state.v.to_numpy(), initial_velocity, rtol=1.0e-6, atol=1.0e-8)

    def test_flip_blend_rejects_values_outside_unit_interval(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.001, -0.001, 0.0],
                    [0.001, -0.001, 0.0],
                    [-0.001, 0.001, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=10.0,
            primary_region_id=1,
            secondary_region_id=2,
        )

        with self.assertRaises(ValueError):
            state.step(
                dt_s=0.0,
                pressure_pa=0.0,
                velocity_damping=1.0,
                flip_blend=1.1,
            )

    def test_device_state_snapshot_restores_positions_displacements_and_velocities(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )
        initial_x = state.x.to_numpy()
        initial_u = state.u.to_numpy()
        initial_v = state.v.to_numpy()
        initial_surface_normal = state.surface_normal.to_numpy()
        state.save_state()

        state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            velocity_damping=1.0,
        )
        self.assertGreater(float(np.linalg.norm(state.x.to_numpy() - initial_x)), 0.0)
        tilted = initial_x.copy()
        tilted[2, 2] += 0.001
        state.x.from_numpy(tilted.astype(np.float32))
        state.step(dt_s=0.0, pressure_pa=0.0, velocity_damping=1.0)
        self.assertGreater(
            float(np.linalg.norm(state.surface_normal.to_numpy() - initial_surface_normal)),
            0.0,
        )
        state.restore_state()

        np.testing.assert_allclose(state.x.to_numpy(), initial_x, atol=1.0e-8)
        np.testing.assert_allclose(state.u.to_numpy(), initial_u, atol=1.0e-8)
        np.testing.assert_allclose(state.v.to_numpy(), initial_v, atol=1.0e-8)
        np.testing.assert_allclose(
            state.surface_normal.to_numpy(),
            initial_surface_normal,
            atol=1.0e-8,
        )

    def test_region_area_load_moves_only_primary_region(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                    [0.016, -0.004, 0.0],
                    [0.024, -0.004, 0.0],
                    [0.016, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7, 8], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        report = state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            velocity_damping=1.0,
        )

        self.assertLess(report.primary_mean_velocity_mps[2], 0.0)
        self.assertLess(report.primary_mean_displacement_m[2], 0.0)
        self.assertAlmostEqual(report.secondary_mean_velocity_mps[2], 0.0, delta=1.0e-10)
        self.assertLess(report.total_force_n[2], 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_external_force_step_preserves_hibm_marker_traction_load(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )
        state.external_force_n[0] = (2.0e-3, 0.0, 0.0)
        state.add_region_area_load(
            region_id=7,
            area_load_npm2=(0.0, 0.0, -1000.0),
        )

        report = state.advance_with_external_forces(
            dt_s=0.0,
            primary_region_id=7,
            secondary_region_id=8,
            velocity_damping=1.0,
            flip_blend=1.0,
        )

        face_area_m2 = 0.5 * 0.008 * 0.008
        self.assertAlmostEqual(report.total_force_n[0], 2.0e-3, delta=2.0e-8)
        self.assertAlmostEqual(
            report.total_force_n[2],
            -1000.0 * face_area_m2,
            delta=2.0e-7,
        )

    def test_region_interface_reaction_accepts_full_3d_forces(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [-0.004, 0.004, 0.0],
                    [0.016, -0.004, 0.0],
                    [0.024, -0.004, 0.0],
                    [0.016, 0.004, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7, 8], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        report = state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, 0.0),
            primary_interface_reaction_n=(2.0e-3, -3.0e-3, 0.0),
            secondary_interface_reaction_n=(-1.0e-3, 4.0e-3, 0.0),
            velocity_damping=1.0,
        )

        self.assertAlmostEqual(report.total_force_n[0], 1.0e-3, delta=2.0e-7)
        self.assertAlmostEqual(report.total_force_n[1], 1.0e-3, delta=2.0e-7)
        self.assertAlmostEqual(report.total_force_n[2], 0.0, delta=2.0e-7)
        self.assertGreater(report.grid_momentum_kg_mps[0], 0.0)
        self.assertGreater(report.grid_momentum_kg_mps[1], 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_fixed_region_vertices_remain_at_rest_under_main_pressure(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [-0.004, -0.004, 0.0],
                    [0.004, -0.004, 0.0],
                    [0.0, 0.004, 0.0],
                    [0.0, -0.010, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32),
        )
        state = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.003,
            density_kgm3=1040.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([7, 5], dtype=np.int32),
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
            grid_nodes=(16, 16, 16),
            bounds_padding_fraction=1.0,
        )

        report = state.advance_region_loads(
            dt_s=1.0e-4,
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
            velocity_damping=1.0,
        )
        current = state.x.to_numpy()

        np.testing.assert_allclose(current[[0, 1, 3]], mesh.vertices[[0, 1, 3]], atol=1.0e-8)
        self.assertLess(current[2, 2], 0.0)
        self.assertLess(report.primary_mean_displacement_m[2], 0.0)
        self.assertLess(report.transfer_relative_error, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
