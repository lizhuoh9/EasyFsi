from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import numpy as np

from simulation_core.geometry import SurfaceMesh
from simulation_core.geometry import UvSphereResolution, make_uv_sphere
from simulation_core.hyperelastic import ecoflex_0010_material
from simulation_core.neo_hookean_mpm import NeoHookeanMpmState
from simulation_core.tri_surface import TriSurfaceRegionDiagnostics


def _tri_surface_from_mesh(mesh, region_id: int = 1) -> TriSurfaceRegionDiagnostics:
    vertices = mesh.vertices
    faces = mesh.faces
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    centroids = (a + b + c) / 3.0
    area_normals = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(area_normals, axis=1)
    normals = area_normals / np.maximum(np.linalg.norm(area_normals, axis=1, keepdims=True), 1.0e-12)
    tri_surface = TriSurfaceRegionDiagnostics(face_capacity=mesh.face_count)
    tri_surface.load_faces(
        centroid_m=centroids.astype(np.float32),
        normal=normals.astype(np.float32),
        area_m2=areas.astype(np.float32),
        region_id=np.full(mesh.face_count, region_id, dtype=np.int32),
    )
    return tri_surface


class NeoHookeanMpmStateTests(unittest.TestCase):
    def test_layered_surface_updates_area_and_normal_from_deformation_gradient(
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
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-1.0, -1.0, -1.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(12, 12, 12),
        )
        tri_surface = _tri_surface_from_mesh(mesh, region_id=1)
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=1,
            primary_region_id=1,
            secondary_region_id=2,
            density_kgm3=1000.0,
            primary_thickness_m=0.02,
            secondary_thickness_m=0.02,
        )
        rest_area = float(state.area_weight_m2[0])
        deformation = state.F.to_numpy()
        deformation[0] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        state.F.from_numpy(deformation)

        state.step(
            dt_s=0.0,
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=1,
            secondary_region_id=2,
        )

        expected_area_scale = math.sqrt(1.25)
        expected_normal = np.array([-0.5, 0.0, 1.0], dtype=np.float32)
        expected_normal /= np.linalg.norm(expected_normal)
        actual_normal = np.array(
            [float(state.surface_normal[0][axis]) for axis in range(3)]
        )
        self.assertAlmostEqual(rest_area, 0.5, delta=1.0e-6)
        self.assertAlmostEqual(
            float(state.area_weight_m2[0]),
            rest_area * expected_area_scale,
            delta=1.0e-5,
        )
        np.testing.assert_allclose(actual_normal, expected_normal, atol=1.0e-5)

    def test_uniform_velocity_transfer_is_conservative(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=64,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(4, 4, 4),
            box_min_m=(-0.005, -0.005, -0.005),
            box_max_m=(0.005, 0.005, 0.005),
            density_kgm3=material.density_kgm3,
        )
        state.set_uniform_velocity((0.02, -0.01, 0.03))

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertEqual(report.particle_count, 64)
        self.assertGreater(report.active_grid_nodes, 0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)
        self.assertGreater(report.total_mass_kg, 0.0)
        self.assertAlmostEqual(report.max_abs_j, 1.0, places=4)

    def test_small_substep_position_increments_accumulate(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(0.1, 0.1, 0.1),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.049, 0.049, 0.049),
            box_max_m=(0.051, 0.051, 0.051),
            density_kgm3=1000.0,
        )
        state.set_uniform_velocity((0.0, 0.0, 1.0e-3))
        initial = state.x.to_numpy().copy()

        for _ in range(2000):
            state.step(
                dt_s=1.0e-7,
                mu_pa=0.0,
                lambda_pa=0.0,
                primary_region_id=0,
                secondary_region_id=-1,
                read_report=False,
            )

        displacement_z = float(state.x.to_numpy()[0, 2] - initial[0, 2])
        self.assertAlmostEqual(displacement_z, 2.0e-7, delta=5.0e-8)

    def test_non_cubic_grid_uses_axis_spacing_for_particle_mapping(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.01, -0.01, -0.04),
            bounds_max_m=(0.01, 0.01, 0.04),
            grid_nodes=(12, 12, 12),
        )
        self.assertFalse(hasattr(state, "h"))
        self.assertNotAlmostEqual(state.dx[0], state.dx[2])
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.002, -0.002, 0.024),
            box_max_m=(0.002, 0.002, 0.030),
            density_kgm3=material.density_kgm3,
        )
        state.set_uniform_velocity((0.0, 0.0, 0.01))

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertGreater(report.active_grid_nodes, 0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)
        self.assertAlmostEqual(report.max_abs_j, 1.0, places=4)

    def test_external_force_changes_grid_momentum_consistently(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )
        state.set_uniform_external_force((0.0, 0.0, -1.0e-4))

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertLess(report.external_force_n[2], 0.0)
        self.assertLess(report.grid_momentum_kg_mps[2], 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_rest_z_plane_constraint_removes_out_of_plane_state(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        x = state.x.to_numpy()
        v = state.v.to_numpy()
        c = state.C.to_numpy()
        f = state.F.to_numpy()
        x[0, 2] += 0.01
        v[0, 2] = 2.0
        c[0, 0, 2] = 3.0
        c[0, 2, 1] = -4.0
        f[0, 0, 2] = 0.25
        f[0, 2, 1] = -0.5
        f[0, 2, 2] = 1.2
        state.x.from_numpy(x)
        state.v.from_numpy(v)
        state.C.from_numpy(c)
        state.F.from_numpy(f)

        state.enforce_rest_z_plane()

        actual_x = state.x.to_numpy()
        actual_v = state.v.to_numpy()
        actual_c = state.C.to_numpy()
        actual_f = state.F.to_numpy()
        self.assertAlmostEqual(actual_x[0, 2], state.rest_x.to_numpy()[0, 2])
        self.assertEqual(actual_v[0, 2], 0.0)
        self.assertEqual(actual_c[0, 0, 2], 0.0)
        self.assertEqual(actual_c[0, 2, 1], 0.0)
        self.assertEqual(actual_f[0, 0, 2], 0.0)
        self.assertEqual(actual_f[0, 2, 1], 0.0)
        self.assertEqual(actual_f[0, 2, 2], 1.0)

    def test_rest_x_plane_constraint_removes_out_of_plane_state(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        x = state.x.to_numpy()
        v = state.v.to_numpy()
        c = state.C.to_numpy()
        f = state.F.to_numpy()
        x[0, 0] += 0.01
        v[0, 0] = 2.0
        c[0, 0, 1] = 3.0
        c[0, 2, 0] = -4.0
        f[0, 0, 1] = 0.25
        f[0, 2, 0] = -0.5
        f[0, 0, 0] = 1.2
        state.x.from_numpy(x)
        state.v.from_numpy(v)
        state.C.from_numpy(c)
        state.F.from_numpy(f)

        state.enforce_rest_x_plane()

        actual_x = state.x.to_numpy()
        actual_v = state.v.to_numpy()
        actual_c = state.C.to_numpy()
        actual_f = state.F.to_numpy()
        self.assertAlmostEqual(actual_x[0, 0], state.rest_x.to_numpy()[0, 0])
        self.assertEqual(actual_v[0, 0], 0.0)
        self.assertEqual(actual_c[0, 0, 1], 0.0)
        self.assertEqual(actual_c[0, 2, 0], 0.0)
        self.assertEqual(actual_f[0, 0, 1], 0.0)
        self.assertEqual(actual_f[0, 2, 0], 0.0)
        self.assertEqual(actual_f[0, 0, 0], 1.0)

    def test_velocity_damping_does_not_pollute_transfer_diagnostic(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )
        state.set_uniform_velocity((0.05, 0.0, 0.0))

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            velocity_damping=0.9,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertGreater(report.max_speed_mps, 0.0)
        self.assertGreater(report.particle_momentum_kg_mps[0], 0.0)
        self.assertAlmostEqual(
            report.grid_momentum_kg_mps[0],
            0.9 * report.particle_momentum_kg_mps[0],
            delta=abs(report.particle_momentum_kg_mps[0]) * 5.0e-5,
        )
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_device_state_snapshot_restores_apic_and_deformation_state(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )
        initial_x = state.x.to_numpy()
        initial_v = state.v.to_numpy()
        initial_c = state.C.to_numpy()
        initial_f = state.F.to_numpy()
        state.save_state()

        state.x.from_numpy(initial_x + np.array([0.001, -0.002, 0.003], dtype=np.float32))
        state.v.from_numpy(np.ones_like(initial_v, dtype=np.float32))
        state.C.from_numpy(np.ones_like(initial_c, dtype=np.float32))
        state.F.from_numpy(np.full_like(initial_f, 2.0, dtype=np.float32))
        state.external_force_n.from_numpy(np.ones_like(state.external_force_n.to_numpy(), dtype=np.float32))
        state.restore_state()

        np.testing.assert_allclose(state.x.to_numpy(), initial_x, atol=1.0e-8)
        np.testing.assert_allclose(state.v.to_numpy(), initial_v, atol=1.0e-8)
        np.testing.assert_allclose(state.C.to_numpy(), initial_c, atol=1.0e-8)
        np.testing.assert_allclose(state.F.to_numpy(), initial_f, atol=1.0e-8)
        np.testing.assert_allclose(
            state.external_force_n.to_numpy(),
            np.zeros_like(state.external_force_n.to_numpy()),
            atol=1.0e-8,
        )

    def test_radial_stretch_diagnostic_is_translation_invariant(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-1.0, -1.0, 0.5),
            bounds_max_m=(1.0, 1.0, 2.0),
            grid_nodes=(16, 16, 16),
            out_of_bounds_particle_tolerance=1,
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.05, -0.05, 1.0),
            box_max_m=(0.05, 0.05, 1.1),
            density_kgm3=material.density_kgm3,
        )
        translated = state.rest_x.to_numpy() + np.array([0.25, -0.1, 0.05], dtype=np.float32)
        state.x.from_numpy(translated.astype(np.float32))

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=1.0e-6)

    def test_radial_stretch_diagnostic_ignores_out_of_bounds_particle(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-1.0, -1.0, 0.5),
            bounds_max_m=(1.0, 1.0, 2.0),
            grid_nodes=(16, 16, 16),
            out_of_bounds_particle_tolerance=1,
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.05, -0.05, 1.0),
            box_max_m=(0.05, 0.05, 1.1),
            density_kgm3=material.density_kgm3,
        )
        translated = state.rest_x.to_numpy() + np.array([0.25, -0.1, 0.05], dtype=np.float32)
        translated[0, 0] = float(state.bounds_max[0] + state.dx[0])
        state.x.from_numpy(translated.astype(np.float32))
        particle_mass = state.mass_kg.to_numpy()

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )
        deposited_mass = float(np.sum(state.grid_mass_kg.to_numpy()))

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertAlmostEqual(deposited_mass, float(np.sum(particle_mass[1:])), delta=1.0e-8)
        self.assertTrue(np.isfinite(report.mean_radial_stretch))
        self.assertTrue(np.isfinite(report.max_radial_stretch_error))
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=5.0e-6)

    def test_near_boundary_particle_does_not_deposit_partial_quadratic_stencil(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
            out_of_bounds_particle_tolerance=1,
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )
        positions = state.x.to_numpy()
        positions[0, 0] = state.bounds_min[0] + 0.25 * state.dx[0]
        state.x.from_numpy(positions.astype(np.float32))
        particle_mass = state.mass_kg.to_numpy()

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )
        deposited_mass = float(np.sum(state.grid_mass_kg.to_numpy()))

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertAlmostEqual(deposited_mass, float(np.sum(particle_mass[1:])), delta=1.0e-8)
        self.assertTrue(np.isfinite(report.mean_radial_stretch))
        self.assertTrue(np.isfinite(report.max_radial_stretch_error))
        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=5.0e-6)

    def test_region_mean_excludes_out_of_bounds_particles(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=3,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
            out_of_bounds_particle_tolerance=1,
        )
        state.initialize_box(
            particle_counts=(3, 1, 1),
            box_min_m=(-0.006, -0.001, -0.001),
            box_max_m=(0.006, 0.001, 0.001),
            density_kgm3=material.density_kgm3,
        )
        rest = np.array(
            [
                [-0.006, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.006, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        current = rest.copy()
        current[0] += np.array([0.001, 0.0, 0.0], dtype=np.float32)
        current[1] = np.array([state.bounds_max[0] + 2.0 * state.dx[0], 0.0, 0.0], dtype=np.float32)
        velocities = np.array(
            [
                [0.02, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        state.rest_x.from_numpy(rest)
        state.x.from_numpy(current)
        state.v.from_numpy(velocities)
        state.region_id.from_numpy(np.array([7, 7, 8], dtype=np.int32))

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=7,
            secondary_region_id=8,
        )

        velocity_after = state.v.to_numpy()
        np.testing.assert_allclose(
            report.primary_mean_displacement_m,
            current[0] - rest[0],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            report.primary_mean_velocity_mps,
            velocity_after[0],
            atol=1.0e-7,
        )
        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertLess(abs(report.primary_mean_velocity_mps[0]), 1.0)

    def test_radial_stretch_mean_excludes_zero_rest_radius_from_denominator(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=3,
            bounds_min_m=(-0.03, -0.01, -0.01),
            bounds_max_m=(0.03, 0.01, 0.01),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_box(
            particle_counts=(3, 1, 1),
            box_min_m=(-0.009, -0.001, -0.001),
            box_max_m=(0.009, 0.001, 0.001),
            density_kgm3=material.density_kgm3,
        )
        rest_positions = np.array(
            [
                [-0.006, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.006, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        state.rest_x.from_numpy(rest_positions)
        state.x.from_numpy(rest_positions)
        state.rest_center_m[None] = (0.0, 0.0, 0.0)

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertAlmostEqual(report.mean_radial_stretch, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.max_radial_stretch_error, 0.0, delta=1.0e-6)

    def test_inverted_deformation_gradient_triggers_svd_clamp(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.01, -0.01, -0.01),
            bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=material.density_kgm3,
        )
        inverted_f = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ],
            dtype=np.float32,
        )
        state.F.from_numpy(inverted_f)

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertEqual(report.deformation_clamp_count, 1)
        self.assertGreater(report.max_abs_j, 0.0)
        self.assertLess(abs(report.max_abs_j - 1.0), 1.0e-5)

    def test_svd_clamp_persists_corrected_deformation_gradient(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.01, -0.01, -0.01),
            bounds_max_m=(0.01, 0.01, 0.01),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=material.density_kgm3,
        )
        inverted_stretch = np.array(
            [
                [
                    [-2.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                ]
            ],
            dtype=np.float32,
        )
        state.F.from_numpy(inverted_stretch)

        report = state.step(
            dt_s=0.0,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )
        corrected_f = state.F.to_numpy()[0]

        self.assertEqual(report.deformation_clamp_count, 1)
        self.assertGreater(float(np.linalg.det(corrected_f)), 0.0)
        singular_values = np.linalg.svd(corrected_f, compute_uv=False)
        self.assertGreaterEqual(float(np.min(singular_values)), 1.0e-2)
        self.assertLessEqual(float(np.max(singular_values)), 1.0e2)

    def test_stress_jacobian_uses_corrected_deformation_gradient_without_unilateral_clamp(self) -> None:
        source = Path("simulation_core/neo_hookean_mpm.py").read_text(encoding="utf-8")

        self.assertNotIn("ti.max(Fp.determinant(), 1.0e-12)", source)
        self.assertIn("J = Fp.determinant()", source)

    def test_layered_tri_surface_area_load_moves_primary_region(self) -> None:
        material = ecoflex_0010_material()
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-6, 8.0e-6], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=4,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=2,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )
        state.set_layered_region_loads(
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, -1000.0),
            primary_interface_reaction_n=(0.0, 0.0, 0.0),
            secondary_interface_reaction_n=(0.0, 0.0, 0.0),
        )

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=7,
            secondary_region_id=8,
        )

        self.assertEqual(report.particle_count, 4)
        self.assertLess(report.external_force_n[2], 0.0)
        self.assertLess(report.primary_mean_velocity_mps[2], 0.0)
        self.assertLess(report.primary_mean_displacement_m[2], 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_layered_tri_surface_rejects_unmodeled_fixed_region_faces(self) -> None:
        material = ecoflex_0010_material()
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-6, 8.0e-6], dtype=np.float32),
            region_id=np.array([7, 5], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=4,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )

        with self.assertRaisesRegex(ValueError, "unsupported.*region"):
            state.initialize_layered_tri_surface(
                tri_surface,
                layer_count=2,
                primary_region_id=7,
                secondary_region_id=8,
                density_kgm3=material.density_kgm3,
                primary_thickness_m=0.003,
                secondary_thickness_m=0.0025,
            )

    def test_layered_region_reaction_accepts_full_3d_forces(self) -> None:
        material = ecoflex_0010_material()
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-6, 8.0e-6], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=4,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=2,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )
        state.set_layered_region_loads(
            primary_region_id=7,
            secondary_region_id=8,
            primary_area_load_npm2=(0.0, 0.0, 0.0),
            primary_interface_reaction_n=(2.0e-3, -3.0e-3, 0.0),
            secondary_interface_reaction_n=(-1.0e-3, 4.0e-3, 0.0),
        )

        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=7,
            secondary_region_id=8,
        )

        self.assertAlmostEqual(report.external_force_n[0], 1.0e-3, delta=2.0e-7)
        self.assertAlmostEqual(report.external_force_n[1], 1.0e-3, delta=2.0e-7)
        self.assertAlmostEqual(report.external_force_n[2], 0.0, delta=2.0e-7)
        self.assertGreater(report.grid_momentum_kg_mps[0], 0.0)
        self.assertGreater(report.grid_momentum_kg_mps[1], 0.0)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_step_can_skip_host_report_read(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(10, 10, 10),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )

        skipped = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
            read_report=False,
        )

        self.assertIsNone(skipped)
        self.assertEqual(state.last_report_host_reads, 0)
        report = state.report()
        self.assertEqual(state.last_report_host_reads, 1)
        self.assertEqual(report.particle_count, 8)

    def test_report_reads_packed_host_snapshot(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(10, 10, 10),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.004, -0.004, -0.004),
            box_max_m=(0.004, 0.004, 0.004),
            density_kgm3=material.density_kgm3,
        )
        state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        state.report_host_snapshot[1] = 0.0042
        report = state.report()

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertAlmostEqual(report.total_volume_m3, 0.0042)

    def test_layered_region_load_api_has_no_pressure_z_alias(self) -> None:
        parameters = inspect.signature(NeoHookeanMpmState.set_layered_region_loads).parameters

        self.assertIn("primary_interface_reaction_n", parameters)
        self.assertIn("secondary_interface_reaction_n", parameters)
        self.assertNotIn("primary_fluid_feedback_n", parameters)
        self.assertNotIn("secondary_fluid_feedback_n", parameters)
        self.assertFalse(hasattr(NeoHookeanMpmState, "set_layered_surface_loads"))

    def test_sphere_normal_pressure_compresses_radially_like_fsi_traction(self) -> None:
        material = ecoflex_0010_material()
        mesh = make_uv_sphere(UvSphereResolution(latitude_bands=8, longitude_segments=25), 1.0)
        tri_surface = _tri_surface_from_mesh(mesh, region_id=1)
        state = NeoHookeanMpmState(
            particle_capacity=tri_surface.face_count,
            bounds_min_m=(-1.7, -1.7, -1.7),
            bounds_max_m=(1.7, 1.7, 1.7),
            grid_nodes=(32, 32, 32),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=1,
            primary_region_id=1,
            secondary_region_id=2,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.05,
            secondary_thickness_m=0.05,
        )
        state.set_region_normal_pressure(region_id=1, pressure_pa=5000.0)

        report = state.step(
            dt_s=2.0e-4,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertGreater(report.external_force_n[0] ** 2 + report.external_force_n[1] ** 2 + report.external_force_n[2] ** 2, 0.0)
        self.assertLess(report.mean_radial_stretch, 1.0)
        self.assertGreater(report.max_speed_mps, 0.0)

    def test_add_region_normal_pressure_preserves_existing_marker_external_force(
        self,
    ) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=material.density_kgm3,
        )
        state.region_id[0] = 1
        state.area_weight_m2[0] = 0.02
        state.surface_normal[0] = (0.0, 0.0, 1.0)
        state.set_uniform_external_force((0.25, 0.0, 0.5))

        state.add_region_normal_pressure(region_id=1, pressure_pa=50.0)
        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertAlmostEqual(report.external_force_n[0], 0.25, delta=1.0e-6)
        self.assertAlmostEqual(report.external_force_n[1], 0.0, delta=1.0e-6)
        self.assertAlmostEqual(report.external_force_n[2], -0.5, delta=1.0e-6)

    def test_add_region_area_load_preserves_marker_traction_and_uses_vector_direction(
        self,
    ) -> None:
        material = ecoflex_0010_material()
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-4, 8.0e-4], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )
        state.set_uniform_external_force((0.1, 0.0, 0.0))

        state.add_region_area_load(region_id=7, area_load_npm2=(0.0, 0.0, -50.0))
        report = state.step(
            dt_s=1.0e-5,
            mu_pa=material.shear_modulus_pa,
            lambda_pa=material.lame_lambda_pa,
            primary_region_id=7,
            secondary_region_id=8,
        )

        self.assertAlmostEqual(report.external_force_n[0], 0.2, delta=1.0e-6)
        self.assertAlmostEqual(report.external_force_n[1], 0.0, delta=1.0e-6)
        self.assertAlmostEqual(report.external_force_n[2], -0.01, delta=1.0e-6)
        self.assertLess(report.primary_mean_velocity_mps[2], 0.0)


class NeoHookeanFullOutOfBoundsGuardTests(unittest.TestCase):
    def test_full_out_of_bounds_particle_set_raises_instead_of_zombie_zeroing(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-1.0, -1.0, 0.5),
            bounds_max_m=(1.0, 1.0, 2.0),
            grid_nodes=(16, 16, 16),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.05, -0.05, 1.0),
            box_max_m=(0.05, 0.05, 1.1),
            density_kgm3=material.density_kgm3,
        )
        translated = state.x.to_numpy()
        translated[:, 0] = float(state.bounds_max[0] + 2.0 * state.dx[0])
        state.x.from_numpy(translated.astype(np.float32))

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.step(
                dt_s=1.0e-5,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=1,
                secondary_region_id=2,
            )

    def test_neo_hookean_partial_out_of_bounds_raises(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=8,
            bounds_min_m=(-1.0, -1.0, 0.5),
            bounds_max_m=(1.0, 1.0, 2.0),
            grid_nodes=(16, 16, 16),
        )
        state.initialize_box(
            particle_counts=(2, 2, 2),
            box_min_m=(-0.05, -0.05, 1.0),
            box_max_m=(0.05, 0.05, 1.1),
            density_kgm3=material.density_kgm3,
        )
        translated = state.x.to_numpy()
        translated[0, 0] = float(state.bounds_max[0] + 2.0 * state.dx[0])
        state.x.from_numpy(translated.astype(np.float32))

        with self.assertRaisesRegex(RuntimeError, "outside the background grid"):
            state.step(
                dt_s=1.0e-5,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=1,
                secondary_region_id=2,
            )

    def test_neo_hookean_shell_region_counts_must_be_nonzero(self) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.20, 0.20, 0.20],
                    [0.40, 0.20, 0.20],
                    [0.20, 0.40, 0.20],
                ],
                dtype=np.float32,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        tri_surface = _tri_surface_from_mesh(mesh, region_id=7)
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )

        with self.assertRaisesRegex(RuntimeError, "secondary shell region"):
            state.step(
                dt_s=0.0,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=7,
                secondary_region_id=8,
            )


class NeoHookeanFixedRegionConstraintTests(unittest.TestCase):
    """S2-A11: the 2s production run died because the neo layered path has no
    fixed-region concept: the case's Fixed Support rim (region 5) was honored
    by the tri_mooney_shell_mpm path (fixed_particle machinery) but silently
    dropped by NeoHookeanMpmState, leaving the main membrane an untethered
    rigid disc that drifted laterally (zero-stiffness neutral mode), tunneled
    into the chamber wall, and bled closure coverage (deficit vs wall-overlap
    Pearson r = 0.9999). These tests pin the generic solver capability:
    initialize_layered_tri_surface(fixed_region_id=...) marks the particles
    of that region, and every substep enforces v = 0, frozen x, rest-identity
    F in-kernel while the fixed mass still anchors the grid in P2G.
    """

    @staticmethod
    def _two_face_surface(
        *,
        free_centroid_m: tuple[float, float, float],
        fixed_centroid_m: tuple[float, float, float],
        free_area_m2: float,
        fixed_area_m2: float,
    ) -> TriSurfaceRegionDiagnostics:
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [free_centroid_m, fixed_centroid_m], dtype=np.float32
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([free_area_m2, fixed_area_m2], dtype=np.float32),
            region_id=np.array([7, 5], dtype=np.int32),
        )
        return tri_surface

    def test_layered_init_marks_exactly_fixed_region_particles(self) -> None:
        """(a) Membership: fixed_region_id marks all particles generated from
        faces of that region and nothing else, fixed faces carry real anchor
        mass (primary thickness fallback, mirroring the mooney shell path),
        and the default fixed_region_id=-1 marks no particle."""
        material = ecoflex_0010_material()
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=3)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.008, 0.0, 0.0],
                    [-0.008, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-6, 2.0e-6, 8.0e-6], dtype=np.float32),
            region_id=np.array([7, 5, 8], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=6,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=2,
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )
        fixed = state.fixed_particle.to_numpy()[: state.particle_count]
        regions = state.region_id.to_numpy()[: state.particle_count]
        masses = state.mass_kg.to_numpy()[: state.particle_count]

        self.assertEqual(state.particle_count, 6)
        np.testing.assert_array_equal(fixed, (regions == 5).astype(np.int32))
        self.assertEqual(int(np.sum(fixed)), 2)
        # Fixed faces get the primary thickness (mooney shell mirror), so the
        # anchors have real mass: rho * (area / layer_count) * thickness.
        expected_fixed_mass = material.density_kgm3 * (2.0e-6 / 2.0) * 0.003
        for mass in masses[fixed == 1]:
            self.assertAlmostEqual(float(mass), expected_fixed_mass, delta=1.0e-9)

        default_state = NeoHookeanMpmState(
            particle_capacity=4,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        default_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        default_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.008, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normal=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.array([2.0e-6, 8.0e-6], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        default_state.initialize_layered_tri_surface(
            default_surface,
            layer_count=2,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.0025,
        )

        np.testing.assert_array_equal(
            default_state.fixed_particle.to_numpy()[: default_state.particle_count],
            np.zeros(default_state.particle_count, dtype=np.int32),
        )

    def test_fixed_particle_ignores_external_force_while_free_particle_moves(
        self,
    ) -> None:
        """(b) Paper-walk (single isolated particle, quadratic B-spline APIC):
        each substep the particle's stencil nodes receive momentum w*(m*v) and
        force w*f, so every touched node carries velocity v + dt*f/m and G2P
        returns exactly v_k = k*dt*f/m with zero affine C (uniform field, the
        first weight moment vanishes). After n substeps the free particle has
        moved sum_k v_k*dt = n*(n+1)/2 * dt^2 * f/m. Here m = rho*A*h
        ~= 1043 * 1e-4 * 0.003 ~= 3.1e-4 kg, f = 0.05 N, dt = 1e-4 s, n = 5:
        ~2.4e-5 m, i.e. order 1e-5 m. The fixed particle sees the same
        external force but must stay bitwise frozen with v = 0, and the
        momentum-transfer diagnostic must stay consistent because the inert
        force is excluded from both the grid spread and the report."""
        material = ecoflex_0010_material()
        # 12 mm separation = 3.6 cells of dx = 40/12 mm: quadratic stencils
        # (reach 1.5 cells) cannot overlap, so the two particles are isolated.
        tri_surface = self._two_face_surface(
            free_centroid_m=(-0.006, 0.0, 0.0),
            fixed_centroid_m=(0.006, 0.0, 0.0),
            free_area_m2=1.0e-4,
            fixed_area_m2=1.0e-4,
        )
        state = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        state.initialize_layered_tri_surface(
            tri_surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.003,
        )
        initial_x = state.x.to_numpy()[:2].copy()
        force_n = 0.05
        dt_s = 1.0e-4
        substeps = 5
        state.set_uniform_external_force((force_n, 0.0, 0.0))

        report = None
        for _ in range(substeps):
            report = state.step(
                dt_s=dt_s,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=7,
                secondary_region_id=8,
            )
        final_x = state.x.to_numpy()[:2]
        final_v = state.v.to_numpy()[:2]
        free_mass = float(state.mass_kg.to_numpy()[0])
        expected_free_displacement_m = (
            0.5 * substeps * (substeps + 1) * dt_s * dt_s * force_n / free_mass
        )

        self.assertAlmostEqual(
            float(final_x[0, 0] - initial_x[0, 0]),
            expected_free_displacement_m,
            delta=0.02 * expected_free_displacement_m,
        )
        # The fixed particle is never advected: bitwise frozen, v exactly 0.
        np.testing.assert_array_equal(final_x[1], initial_x[1])
        np.testing.assert_array_equal(final_v[1], np.zeros(3, dtype=np.float32))
        # Only the free particle's force changes momentum, and the report
        # bookkeeping must agree with what was actually spread to the grid.
        self.assertAlmostEqual(report.external_force_n[0], force_n, delta=1.0e-6)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

    def test_fixed_anchor_mass_resists_grid_coupled_free_particle(self) -> None:
        """(c) The anchor (fixed_particle=1) contributes its mass with ZERO
        momentum in P2G. Grid nodes shared by a free particle and the anchor
        divide the free particle's momentum by the combined node mass, so the
        gathered G2P velocity of the coupled particle is strictly smaller than
        that of an identical particle with no anchor in stencil range. Here a
        4x-mass anchor sits one cell away sharing 2/3 of the stencil; direct
        weight arithmetic on the first substep gives a velocity ratio of
        ~0.58, so the displacement must land well below the 0.9 gate. Unlike
        the isolated fixtures (uniform field, C = 0, F = I exactly), the
        coupled state has a nonuniform grid velocity field, so F evolves and
        real near-incompressible Ecoflex stress acts: dt must sit inside the
        material's explicit elastic CFL (~3.0e-5 s at this spacing), hence
        dt = 1e-5 s. This grid-mediated mass coupling is the mechanism by
        which a fixed rim region clamps neighboring free membrane material in
        the layered neo path."""
        material = ecoflex_0010_material()
        bounds_min = (-0.02, -0.02, -0.02)
        bounds_max = (0.02, 0.02, 0.02)
        grid_nodes = (12, 12, 12)
        dx = (bounds_max[0] - bounds_min[0]) / grid_nodes[0]
        coupled_surface = self._two_face_surface(
            free_centroid_m=(0.0, 0.0, 0.0),
            fixed_centroid_m=(dx, 0.0, 0.0),
            free_area_m2=1.0e-4,
            fixed_area_m2=4.0e-4,
        )
        coupled = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=grid_nodes,
        )
        coupled.initialize_layered_tri_surface(
            coupled_surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            fixed_region_id=5,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.003,
        )
        reference_surface = TriSurfaceRegionDiagnostics(face_capacity=1)
        reference_surface.load_faces(
            centroid_m=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            normal=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.array([1.0e-4], dtype=np.float32),
            region_id=np.array([7], dtype=np.int32),
        )
        reference = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            grid_nodes=grid_nodes,
        )
        reference.initialize_layered_tri_surface(
            reference_surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.003,
        )
        # The same area load on the same region-7 face area applies the same
        # external force to both free particles; the anchor receives nothing.
        for state in (coupled, reference):
            state.add_region_area_load(
                region_id=7,
                area_load_npm2=(500.0, 0.0, 0.0),
            )
            for _ in range(30):
                state.step(
                    dt_s=1.0e-5,
                    mu_pa=material.shear_modulus_pa,
                    lambda_pa=material.lame_lambda_pa,
                    primary_region_id=7,
                    secondary_region_id=8,
                    read_report=False,
                )
        coupled_displacement_m = float(
            coupled.x.to_numpy()[0, 0] - coupled.rest_x.to_numpy()[0, 0]
        )
        reference_displacement_m = float(
            reference.x.to_numpy()[0, 0] - reference.rest_x.to_numpy()[0, 0]
        )
        anchor_displacement_m = float(
            np.abs(coupled.x.to_numpy()[1] - coupled.rest_x.to_numpy()[1]).max()
        )

        self.assertGreater(reference_displacement_m, 0.0)
        self.assertGreater(coupled_displacement_m, 0.0)
        self.assertLess(coupled_displacement_m, 0.9 * reference_displacement_m)
        self.assertEqual(anchor_displacement_m, 0.0)

    def test_save_restore_and_reinit_keep_constraint_active(self) -> None:
        """(d) The constraint is derived state: save_state/restore_state never
        touch fixed_particle, and the checkpoint-resume order (initialize, the
        marks are rebuilt from region ids, then x/v/C/F overwritten from the
        payload) keeps the constraint active in-kernel even when the loaded
        payload carries a stale nonzero velocity or deformation on a fixed
        particle (e.g. a checkpoint written by a pre-constraint build)."""
        material = ecoflex_0010_material()
        tri_surface = self._two_face_surface(
            free_centroid_m=(-0.006, 0.0, 0.0),
            fixed_centroid_m=(0.006, 0.0, 0.0),
            free_area_m2=1.0e-4,
            fixed_area_m2=1.0e-4,
        )
        state = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )

        def initialize() -> None:
            state.initialize_layered_tri_surface(
                tri_surface,
                layer_count=1,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_region_id=5,
                density_kgm3=material.density_kgm3,
                primary_thickness_m=0.003,
                secondary_thickness_m=0.003,
            )

        def step_n(count: int) -> None:
            for _ in range(count):
                state.step(
                    dt_s=1.0e-4,
                    mu_pa=material.shear_modulus_pa,
                    lambda_pa=material.lame_lambda_pa,
                    primary_region_id=7,
                    secondary_region_id=8,
                )

        initialize()
        base_x = state.x.to_numpy()[:2].copy()
        state.save_state()
        state.set_uniform_external_force((0.05, 0.0, 0.0))
        step_n(3)
        moved_x = state.x.to_numpy()[:2].copy()
        self.assertGreater(float(moved_x[0, 0] - base_x[0, 0]), 0.0)
        np.testing.assert_array_equal(moved_x[1], base_x[1])

        state.restore_state()
        np.testing.assert_array_equal(state.x.to_numpy()[:2], base_x)
        np.testing.assert_array_equal(
            state.fixed_particle.to_numpy()[:2],
            np.array([0, 1], dtype=np.int32),
        )
        state.set_uniform_external_force((0.05, 0.0, 0.0))
        step_n(3)
        after_restore_x = state.x.to_numpy()[:2]
        after_restore_v = state.v.to_numpy()[:2]
        self.assertGreater(float(after_restore_x[0, 0] - base_x[0, 0]), 0.0)
        np.testing.assert_array_equal(after_restore_x[1], base_x[1])
        np.testing.assert_array_equal(
            after_restore_v[1], np.zeros(3, dtype=np.float32)
        )

        # Checkpoint-resume shape: re-init rebuilds the marks from region ids,
        # then the solid payload (x/v/C/F) is loaded over the fields. A stale
        # payload from a pre-constraint build may carry nonzero velocity and
        # deformation on the fixed particle; the in-kernel enforcement must
        # zero v, freeze x at the loaded position, and reset F to identity.
        initialize()
        np.testing.assert_array_equal(
            state.fixed_particle.to_numpy()[:2],
            np.array([0, 1], dtype=np.int32),
        )
        loaded_x = base_x.copy()
        loaded_x[1] += np.array([1.0e-3, -5.0e-4, 2.0e-4], dtype=np.float32)
        loaded_v = np.zeros((2, 3), dtype=np.float32)
        loaded_v[1] = np.array([0.2, -0.1, 0.05], dtype=np.float32)
        loaded_f = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        loaded_f[1, 0, 1] = 0.25
        state.x.from_numpy(loaded_x)
        state.v.from_numpy(loaded_v)
        state.C.from_numpy(np.zeros((2, 3, 3), dtype=np.float32))
        state.F.from_numpy(loaded_f)

        step_n(1)

        resumed_x = state.x.to_numpy()[:2]
        resumed_v = state.v.to_numpy()[:2]
        resumed_f = state.F.to_numpy()[:2]
        np.testing.assert_array_equal(resumed_x[1], loaded_x[1])
        np.testing.assert_array_equal(resumed_v[1], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            resumed_f[1], np.eye(3, dtype=np.float32), atol=0.0
        )


class NeoHookeanVacuousFixedRegionGuardTests(unittest.TestCase):
    """S2-A11c: the first A11 wiring passed fixed_region_id=5 while the neo
    solid mesh subset contained regions (7, 8) only - the constraint matched
    ZERO faces and was silently vacuous (the membrane stayed an untethered
    free disc; only a source review caught it, every synthetic-fixture test
    passed). A requested fixed region that matches no faces of the supplied
    mesh must fail loudly at init time instead of pretending to constrain.
    """

    def test_layered_init_rejects_fixed_region_matching_no_faces(self) -> None:
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [[0.0, 0.0, 0.0], [0.008, 0.0, 0.0]], dtype=np.float32
            ),
            normal=np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
            area_m2=np.array([2.0e-6, 2.0e-6], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        state = NeoHookeanMpmState(
            particle_capacity=4,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )

        with self.assertRaisesRegex(ValueError, "fixed_region_id=5"):
            state.initialize_layered_tri_surface(
                tri_surface,
                layer_count=2,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_region_id=5,
                density_kgm3=1000.0,
                primary_thickness_m=1.5e-3,
                secondary_thickness_m=1.0e-3,
            )


if __name__ == "__main__":
    unittest.main()
