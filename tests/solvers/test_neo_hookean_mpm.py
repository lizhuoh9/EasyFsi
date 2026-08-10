from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import numpy as np
import taichi as ti

from simulation_core.geometry_tools import SurfaceMesh
from simulation_core.geometry_tools import UvSphereResolution, make_uv_sphere
from simulation_core.materials.hyperelastic import ecoflex_0010_material
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmState
from simulation_core.coupling.tri_surface import TriSurfaceRegionDiagnostics


NEO_HOOKEAN_MPM_SOURCE = Path("simulation_core/solids/neo_hookean_mpm.py")


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


def _single_particle_constitutive_stress_map(
    deformation_gradient: np.ndarray,
    *,
    constitutive_model: str,
    mu_pa: float,
    lambda_pa: float,
) -> np.ndarray:
    """Recover the constitutive ``P @ F.T`` map from one MPM P2G step.

    The particle is placed at the center of a uniform grid.  At the three
    positive axial stencil nodes, the normalized grid velocity is

        v_i = -(4 * dt / (rho * dx_i)) * (P @ F.T)[:, i].

    This keeps the test on the public ``step`` behavior while making the
    expected Saint-Venant--Kirchhoff response independently computable from
    the Green--Lagrange strain.
    """

    density_kgm3 = 1000.0
    dt_s = 1.0e-6
    state = NeoHookeanMpmState(
        particle_capacity=1,
        bounds_min_m=(-0.02, -0.02, -0.02),
        bounds_max_m=(0.02, 0.02, 0.02),
        grid_nodes=(8, 8, 8),
    )
    state.initialize_box(
        particle_counts=(1, 1, 1),
        box_min_m=(-0.001, -0.001, -0.001),
        box_max_m=(0.001, 0.001, 0.001),
        density_kgm3=density_kgm3,
    )
    state.F.from_numpy(
        np.asarray(deformation_gradient, dtype=np.float32).reshape(1, 3, 3)
    )

    state.step(
        dt_s=dt_s,
        mu_pa=float(mu_pa),
        lambda_pa=float(lambda_pa),
        primary_region_id=0,
        secondary_region_id=-1,
        constitutive_model=constitutive_model,
    )

    grid_velocity = state.grid_velocity_mps.to_numpy()
    stress_map = np.empty((3, 3), dtype=np.float64)
    for axis, node in enumerate(((5, 4, 4), (4, 5, 4), (4, 4, 5))):
        stress_map[:, axis] = (
            -np.asarray(grid_velocity[node], dtype=np.float64)
            * density_kgm3
            * float(state.dx[axis])
            / (4.0 * dt_s)
        )
    return stress_map


class NeoHookeanMpmStateTests(unittest.TestCase):
    def test_svk_dispatch_does_not_precompute_neo_inverse_or_log(self) -> None:
        source = NEO_HOOKEAN_MPM_SOURCE.read_text(encoding="utf-8")
        kernel_start = source.index("    def _step_kernel(")
        dispatch_start = source.index(
            "if constitutive_model == CONSTITUTIVE_SAINT_VENANT_KIRCHHOFF",
            kernel_start,
        )
        pre_dispatch = source[kernel_start:dispatch_start]
        dispatch_end = source.index("                inv_dx2 = ti.Matrix(", dispatch_start)
        dispatch = source[dispatch_start:dispatch_end]

        self.assertNotIn("Fp.inverse().transpose()", pre_dispatch)
        self.assertNotIn("ti.log(J)", pre_dispatch)
        self.assertIn("elif constitutive_model == CONSTITUTIVE_LINEAR_ELASTIC", dispatch)
        self.assertIn("else:", dispatch)
        self.assertEqual(dispatch.count("Fp.inverse().transpose()"), 1)
        self.assertEqual(dispatch.count("ti.log(J)"), 1)

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

    def test_position_accumulator_does_not_create_multi_ulp_jumps(self) -> None:
        """Sub-ULP motion must be compensated, not released in 5-8 ULP bursts."""

        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 1.9),
            bounds_max_m=(0.1, 0.1, 2.1),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.049, 0.049, 1.999),
            box_max_m=(0.051, 0.051, 2.001),
            density_kgm3=1000.0,
        )
        velocity_mps = 1.0e-2
        dt_s = 1.0e-5
        state.set_uniform_velocity((0.0, 0.0, velocity_mps))
        initial_z = np.float32(state.x.to_numpy()[0, 2])
        ulp_m = float(np.spacing(initial_z))
        visible_positions_m = [float(initial_z)]

        for _ in range(12):
            state.step(
                dt_s=dt_s,
                mu_pa=0.0,
                lambda_pa=0.0,
                primary_region_id=0,
                secondary_region_id=-1,
                read_report=False,
            )
            visible_positions_m.append(float(state.x.to_numpy()[0, 2]))

        expected_six_step_z_m = float(initial_z) + 6.0 * dt_s * velocity_mps
        self.assertLessEqual(
            abs(visible_positions_m[6] - expected_six_step_z_m),
            0.55 * ulp_m,
        )
        self.assertLessEqual(
            max(abs(value) for value in np.diff(visible_positions_m)),
            1.01 * ulp_m,
        )
        final_residual_m = abs(
            float(state.position_increment_residual_m.to_numpy()[0, 2])
        )
        self.assertLessEqual(final_residual_m, 0.55 * ulp_m)

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

    def test_linear_elastic_constitutive_model_is_explicitly_selectable(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )

        report = state.step(
            dt_s=1.0e-6,
            mu_pa=1000.0,
            lambda_pa=1000.0,
            primary_region_id=0,
            secondary_region_id=-1,
            constitutive_model="plane_stress_linear_elastic",
            velocity_transfer_flip_blend=0.5,
        )

        self.assertEqual(report.particle_count, 1)
        with self.assertRaisesRegex(ValueError, "velocity_transfer_flip_blend"):
            state.step(
                dt_s=1.0e-6,
                mu_pa=1000.0,
                lambda_pa=1000.0,
                primary_region_id=0,
                secondary_region_id=-1,
                velocity_transfer_flip_blend=1.1,
            )
        with self.assertRaisesRegex(ValueError, "constitutive_model"):
            state.step(
                dt_s=1.0e-6,
                mu_pa=1000.0,
                lambda_pa=1000.0,
                primary_region_id=0,
                secondary_region_id=-1,
                constitutive_model="unknown_model",
            )

    def test_plane_stress_linear_elastic_zeroes_out_of_plane_stress(self) -> None:
        """The vertical-flap 2-D plane is yz, so x is stress-free."""

        deformation_gradient = np.array(
            [
                [1.08, 0.03, -0.04],
                [0.02, 1.05, 0.07],
                [-0.01, 0.03, 0.96],
            ],
            dtype=np.float64,
        )
        mu_pa = 3.5e5
        plane_stress_lambda_pa = 5.0e5
        displacement_gradient = deformation_gradient - np.eye(3, dtype=np.float64)
        strain = 0.5 * (displacement_gradient + displacement_gradient.T)
        in_plane_trace = float(strain[1, 1] + strain[2, 2])
        expected_stress_map = np.zeros((3, 3), dtype=np.float64)
        expected_stress_map[1:, 1:] = 2.0 * mu_pa * strain[1:, 1:]
        expected_stress_map[1, 1] += plane_stress_lambda_pa * in_plane_trace
        expected_stress_map[2, 2] += plane_stress_lambda_pa * in_plane_trace

        actual_stress_map = _single_particle_constitutive_stress_map(
            deformation_gradient,
            constitutive_model="plane_stress_linear_elastic",
            mu_pa=mu_pa,
            lambda_pa=plane_stress_lambda_pa,
        )

        np.testing.assert_allclose(
            actual_stress_map,
            expected_stress_map,
            rtol=5.0e-4,
            atol=5.0,
        )

    def test_saint_venant_kirchhoff_constitutive_model_is_explicitly_selectable(
        self,
    ) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )

        report = state.step(
            dt_s=1.0e-6,
            mu_pa=5.0e5,
            lambda_pa=2.0e6,
            primary_region_id=0,
            secondary_region_id=-1,
            constitutive_model="saint_venant_kirchhoff",
        )

        self.assertEqual(report.particle_count, 1)

    def test_saint_venant_kirchhoff_non_small_strain_matches_green_lagrange(
        self,
    ) -> None:
        deformation_gradient = np.diag([1.4, 0.8, 1.1]).astype(np.float64)
        mu_pa = 5.0e5
        lambda_pa = 2.0e6
        identity = np.eye(3, dtype=np.float64)
        green_lagrange = 0.5 * (
            deformation_gradient.T @ deformation_gradient - identity
        )
        second_piola = (
            lambda_pa * np.trace(green_lagrange) * identity
            + 2.0 * mu_pa * green_lagrange
        )
        expected_svk_stress_map = (
            deformation_gradient @ second_piola @ deformation_gradient.T
        )

        neo_hookean_stress_map = _single_particle_constitutive_stress_map(
            deformation_gradient,
            constitutive_model="neo_hookean",
            mu_pa=mu_pa,
            lambda_pa=lambda_pa,
        )
        expected_neo_hookean_stress_map = (
            mu_pa
            * (deformation_gradient @ deformation_gradient.T - identity)
            + lambda_pa
            * math.log(float(np.linalg.det(deformation_gradient)))
            * identity
        )
        np.testing.assert_allclose(
            neo_hookean_stress_map,
            expected_neo_hookean_stress_map,
            rtol=5.0e-4,
            atol=5.0,
        )

        actual_svk_stress_map = _single_particle_constitutive_stress_map(
            deformation_gradient,
            constitutive_model="saint_venant_kirchhoff",
            mu_pa=mu_pa,
            lambda_pa=lambda_pa,
        )

        np.testing.assert_allclose(
            actual_svk_stress_map,
            expected_svk_stress_map,
            rtol=5.0e-4,
            atol=5.0,
        )
        self.assertGreater(
            float(np.linalg.norm(actual_svk_stress_map - neo_hookean_stress_map)),
            0.05 * float(np.linalg.norm(expected_svk_stress_map)),
        )

    def test_flip_baseline_excludes_elastic_stress_impulse(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        deformation = state.F.to_numpy()
        deformation[0] = np.array(
            [
                [1.0, 0.03, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        state.F.from_numpy(deformation)

        state.step(
            dt_s=1.0e-4,
            mu_pa=1000.0,
            lambda_pa=1000.0,
            primary_region_id=0,
            secondary_region_id=-1,
            constitutive_model="linear_elastic",
            velocity_transfer_flip_blend=1.0,
        )

        active = state.grid_mass_kg.to_numpy() > 0.0
        before = state.grid_velocity_before_update_mps.to_numpy()[active]
        after = state.grid_velocity_mps.to_numpy()[active]
        self.assertGreater(float(np.linalg.norm(after, axis=1).max()), 1.0e-8)
        self.assertLess(float(np.linalg.norm(before, axis=1).max()), 1.0e-8)

    def test_linear_elastic_pure_shear_does_not_create_normal_affine_rate(self) -> None:
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(8, 8, 8),
        )
        state.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(-0.001, -0.001, -0.001),
            box_max_m=(0.001, 0.001, 0.001),
            density_kgm3=1000.0,
        )
        deformation = state.F.to_numpy()
        deformation[0] = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        state.F.from_numpy(deformation)

        state.step(
            dt_s=1.0e-5,
            mu_pa=1000.0,
            lambda_pa=0.0,
            primary_region_id=0,
            secondary_region_id=-1,
            constitutive_model="linear_elastic",
        )

        affine = state.C.to_numpy()[0]
        self.assertGreater(abs(float(affine[0, 1])), 1.0e-8)
        self.assertGreater(abs(float(affine[1, 0])), 1.0e-8)
        self.assertLess(abs(float(affine[0, 0])), 1.0e-5)

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
        initial_position_residual = np.linspace(
            -1.0e-8,
            1.0e-8,
            num=state.position_increment_residual_m.to_numpy().size,
            dtype=np.float32,
        ).reshape(state.position_increment_residual_m.to_numpy().shape)
        state.position_increment_residual_m.from_numpy(
            initial_position_residual
        )
        initial_v = state.v.to_numpy()
        initial_c = state.C.to_numpy()
        initial_f = state.F.to_numpy()
        state.save_state()

        state.x.from_numpy(initial_x + np.array([0.001, -0.002, 0.003], dtype=np.float32))
        state.position_increment_residual_m.from_numpy(
            np.zeros_like(initial_position_residual)
        )
        state.v.from_numpy(np.ones_like(initial_v, dtype=np.float32))
        state.C.from_numpy(np.ones_like(initial_c, dtype=np.float32))
        state.F.from_numpy(np.full_like(initial_f, 2.0, dtype=np.float32))
        state.external_force_n.from_numpy(np.ones_like(state.external_force_n.to_numpy(), dtype=np.float32))
        state.restore_state()

        np.testing.assert_allclose(state.x.to_numpy(), initial_x, atol=1.0e-8)
        np.testing.assert_array_equal(
            state.position_increment_residual_m.to_numpy(),
            initial_position_residual,
        )
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
        source = NEO_HOOKEAN_MPM_SOURCE.read_text(encoding="utf-8")

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

        self.assertEqual(state.report_host_snapshot.dtype, ti.f64)
        state.report_host_snapshot[1] = 0.0042
        large_count = 2**24 + 3
        state.report_host_snapshot[28] = float(large_count)
        report = state.report()

        self.assertEqual(state.last_report_host_reads, 1)
        self.assertAlmostEqual(report.total_volume_m3, 0.0042)
        self.assertEqual(report.active_grid_nodes, large_count)

    def test_transfer_error_ignores_particles_skipped_by_oob_stencil(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
            out_of_bounds_particle_tolerance=1,
        )
        state.initialize_box(
            particle_counts=(2, 1, 1),
            box_min_m=(-0.002, -0.001, -0.001),
            box_max_m=(0.002, 0.001, 0.001),
            density_kgm3=material.density_kgm3,
        )
        positions = state.x.to_numpy()
        positions[1] = (0.2, 0.0, 0.0)
        state.x.from_numpy(positions)
        state.set_uniform_velocity((0.1, 0.0, 0.0))

        report = state.step(
            dt_s=1.0e-6,
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=1,
            secondary_region_id=2,
        )

        self.assertEqual(report.grid_out_of_bounds_particle_count, 1)
        self.assertLess(report.transfer_relative_error, 2.0e-5)

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
        # A mesh containing ONLY primary-region faces must not be forced
        # through the secondary-region guard (see
        # test_layered_surface_updates_area_and_normal_from_deformation_gradient
        # and test_sphere_normal_pressure_compresses_radially_like_fsi_traction,
        # which are primary-only meshes that must NOT raise). To pin the
        # guard's "genuinely required but empty" behavior instead, this
        # fixture tags a real secondary-region face in the mesh (so
        # require_nonempty_secondary_region_count is legitimately True) and
        # then pushes that secondary particle out of the background grid so
        # its counted secondary_particle_count comes back 0. The tolerance
        # is raised to 1 so the out-of-bounds guard doesn't preempt the
        # region-count guard with a different error message.
        tri_surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        tri_surface.load_faces(
            centroid_m=np.array(
                [
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
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
            area_m2=np.array([0.02, 0.02], dtype=np.float32),
            region_id=np.array([7, 8], dtype=np.int32),
        )
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
            out_of_bounds_particle_tolerance=1,
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
        positions = state.x.to_numpy()
        positions[1, 0] = float(state.bounds_max[0] + 2.0 * state.dx[0])
        state.x.from_numpy(positions.astype(np.float32))

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

    def test_pure_fixed_mass_policy_does_not_lock_mixed_free_fixed_nodes(self) -> None:
        material = ecoflex_0010_material()
        bounds_min = (-0.02, -0.02, -0.02)
        bounds_max = (0.02, 0.02, 0.02)
        grid_nodes = (12, 12, 12)
        dx = (bounds_max[0] - bounds_min[0]) / grid_nodes[0]

        def make_state() -> NeoHookeanMpmState:
            surface = self._two_face_surface(
                free_centroid_m=(0.0, 0.0, 0.0),
                fixed_centroid_m=(dx, 0.0, 0.0),
                free_area_m2=1.0e-4,
                fixed_area_m2=4.0e-4,
            )
            state = NeoHookeanMpmState(
                particle_capacity=2,
                bounds_min_m=bounds_min,
                bounds_max_m=bounds_max,
                grid_nodes=grid_nodes,
            )
            state.initialize_layered_tri_surface(
                surface,
                layer_count=1,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_region_id=5,
                density_kgm3=material.density_kgm3,
                primary_thickness_m=0.003,
                secondary_thickness_m=0.003,
            )
            state.add_region_area_load(
                region_id=7,
                area_load_npm2=(500.0, 0.0, 0.0),
            )
            return state

        legacy = make_state()
        pure_mass = make_state()
        for _ in range(20):
            legacy.step(
                dt_s=1.0e-5,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_node_lock_policy="any_fixed_particle",
                read_report=False,
            )
            pure_mass.step(
                dt_s=1.0e-5,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_node_lock_policy="pure_fixed_mass",
                read_report=False,
            )

        legacy_free_displacement_m = float(
            legacy.x.to_numpy()[0, 0] - legacy.rest_x.to_numpy()[0, 0]
        )
        pure_free_displacement_m = float(
            pure_mass.x.to_numpy()[0, 0] - pure_mass.rest_x.to_numpy()[0, 0]
        )
        pure_anchor_displacement_m = float(
            np.abs(pure_mass.x.to_numpy()[1] - pure_mass.rest_x.to_numpy()[1]).max()
        )

        self.assertGreater(pure_free_displacement_m, legacy_free_displacement_m)
        self.assertGreater(pure_free_displacement_m, 0.0)
        self.assertEqual(pure_anchor_displacement_m, 0.0)

    def test_fixed_root_cantilever_statics_require_any_fixed_particle_lock(
        self,
    ) -> None:
        """Root-clamp integrity regression (ANSYS vertical-flap overshoot,
        2026-07-03 audit).

        A box cantilever (0.003 x 0.010 x 0.003 m, E=1 MPa, nu=0.47,
        rho=1600 kg/m3, plane-stress linear elastic) with its bottom particle
        row fixed and a constant distributed tip-normal load F_z = -1.157e-2 N
        has an Euler-Bernoulli static tip deflection of
        q*L^4/(8*E*I) = 2.14e-4 m (q = F/L, I = b*t^3/12 = 6.75e-12 m4).

        With fixed_node_lock_policy="any_fixed_particle" the MPM solution
        settles at that static value. With "pure_fixed_mass" the mixed
        fixed/free grid nodes stay mobile while fixed particles contribute
        mass but zero stress, so the root clamp has no elastic restoring
        path and the flap creeps monotonically PAST static equilibrium
        (measured 4.0x at t=0.025 s and still rising; on refined grids the
        clamp fails entirely and the flap free-falls). This test pins both
        behaviors so the flap-case default cannot silently regress.
        """
        span_m = 0.003
        height_m = 0.010
        thickness_m = 0.003
        young_pa = 1.0e6
        nu = 0.47
        rho_kgm3 = 1600.0
        mu_pa = young_pa / (2.0 * (1.0 + nu))
        lambda_plane_stress_pa = young_pa * nu / (1.0 - nu * nu)
        total_force_z_n = -1.157e-2
        second_moment_m4 = span_m * thickness_m**3 / 12.0
        eb_static_tip_m = (
            (abs(total_force_z_n) / height_m)
            * height_m**4
            / (8.0 * young_pa * second_moment_m4)
        )
        # Same discretization family as the vertical-flap coarse run:
        # fluid-domain-sized background grid, 1x12x4 particles, root row fixed.
        grid_nodes = (4, 32, 64)
        base_dy = 0.02 / grid_nodes[1]
        bounds_min = (0.0, -3.0 * base_dy, 0.0)
        bounds_max = (span_m, 0.02, 0.10)
        flap_z_min = 0.10 - 0.053
        particle_counts = (1, 12, 4)
        # Elastic wave speed sqrt((lambda+2mu)/rho) = 28.3 m/s; dt = 2e-6 s
        # keeps the sound CFL at 0.08 on dy = 6.8e-4 m (dt >= 4e-6 s / CFL
        # >= 0.17 slowly injects energy with this explicit quadratic-B-spline
        # transfer, and dt = 1e-5 s / CFL 0.41 is outright unstable).
        dt_sub_s = 2.0e-6
        substep_count = 10000  # t_end = 0.020 s
        sample_interval = 100
        damping_per_substep = 0.995 ** (dt_sub_s / 5.0e-4)

        def run_case(lock_policy: str) -> list[float]:
            state = NeoHookeanMpmState(
                particle_capacity=int(np.prod(particle_counts)),
                bounds_min_m=bounds_min,
                bounds_max_m=bounds_max,
                grid_nodes=grid_nodes,
            )
            state.initialize_box(
                particle_counts=particle_counts,
                box_min_m=(0.0, 0.0, flap_z_min),
                box_max_m=(span_m, height_m, flap_z_min + thickness_m),
                density_kgm3=rho_kgm3,
            )
            rest = state.x.to_numpy()[: state.particle_count]
            root_limit = 1.01 * height_m / particle_counts[1]
            fixed = np.zeros((state.particle_capacity,), dtype=np.int32)
            fixed[: state.particle_count] = (
                rest[:, 1] <= root_limit
            ).astype(np.int32)
            state.fixed_particle.from_numpy(fixed)
            free_count = int(state.particle_count - fixed.sum())
            forces = np.zeros((state.particle_capacity, 3), dtype=np.float32)
            forces[: state.particle_count, 2] = np.where(
                fixed[: state.particle_count] == 0,
                total_force_z_n / free_count,
                0.0,
            )
            state.external_force_n.from_numpy(forces)
            tip_mask = rest[:, 1] >= rest[:, 1].max() - 1.0e-9

            def tip_dz_m() -> float:
                positions = state.x.to_numpy()[: state.particle_count]
                return float((positions - rest)[tip_mask, 2].mean())

            samples_m: list[float] = []
            for substep in range(substep_count):
                state.step(
                    dt_s=dt_sub_s,
                    mu_pa=mu_pa,
                    lambda_pa=lambda_plane_stress_pa,
                    primary_region_id=0,
                    secondary_region_id=-1,
                    velocity_damping=damping_per_substep,
                    fixed_node_lock_policy=lock_policy,
                    constitutive_model="plane_stress_linear_elastic",
                    velocity_transfer_flip_blend=0.0,
                    read_report=False,
                )
                if (substep + 1) % sample_interval == 0:
                    samples_m.append(tip_dz_m())
            root_positions = state.x.to_numpy()[: state.particle_count]
            root_mask = fixed[: state.particle_count] != 0
            self.assertEqual(
                float(
                    np.abs(root_positions[root_mask] - rest[root_mask]).max()
                ),
                0.0,
            )
            return samples_m

        locked_samples_m = run_case("any_fixed_particle")
        creep_samples_m = run_case("pure_fixed_mass")
        half = len(locked_samples_m) // 2
        # The locked solution rings about the static deflection (lightly
        # damped), so instantaneous values are phase-sensitive; compare the
        # second-half window mean against Euler-Bernoulli instead.
        locked_window_mean_m = abs(
            float(np.mean(locked_samples_m[half:]))
        )
        locked_peak_m = float(np.max(np.abs(locked_samples_m)))
        creep_window_mean_m = abs(float(np.mean(creep_samples_m[half:])))
        creep_final_m = abs(creep_samples_m[-1])
        creep_mid_m = abs(creep_samples_m[half])

        # Locked clamp: ringing about the Euler-Bernoulli static deflection.
        self.assertLess(
            abs(locked_window_mean_m - eb_static_tip_m),
            0.5 * eb_static_tip_m,
        )
        # Bounded response: the elastic ringing peak stays below the dynamic
        # overshoot ceiling (~2x static); creep/instability would exceed it.
        self.assertLess(locked_peak_m, 3.0 * eb_static_tip_m)
        # Unlocked clamp: creeping past static equilibrium and still rising.
        self.assertGreater(creep_window_mean_m, 2.0 * locked_window_mean_m)
        self.assertGreater(
            creep_final_m,
            1.2 * creep_mid_m,
            msg="pure_fixed_mass root clamp is expected to keep creeping",
        )

    def test_step_rejects_unknown_fixed_node_lock_policy(self) -> None:
        material = ecoflex_0010_material()
        state = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(-0.02, -0.02, -0.02),
            bounds_max_m=(0.02, 0.02, 0.02),
            grid_nodes=(12, 12, 12),
        )
        surface = TriSurfaceRegionDiagnostics(face_capacity=1)
        surface.load_faces(
            centroid_m=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            normal=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.array([1.0e-4], dtype=np.float32),
            region_id=np.array([7], dtype=np.int32),
        )
        state.initialize_layered_tri_surface(
            surface,
            layer_count=1,
            primary_region_id=7,
            secondary_region_id=8,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=0.003,
            secondary_thickness_m=0.003,
        )

        with self.assertRaisesRegex(ValueError, "fixed_node_lock_policy"):
            state.step(
                dt_s=1.0e-5,
                mu_pa=material.shear_modulus_pa,
                lambda_pa=material.lame_lambda_pa,
                primary_region_id=7,
                secondary_region_id=8,
                fixed_node_lock_policy="unknown",
            )

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
