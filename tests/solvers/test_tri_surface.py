from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    CartesianGrid,
    FluidDomainSpec,
    TaichiRuntimeConfig,
    TriSurfaceRegionDiagnostics,
)


class TriSurfaceRegionDiagnosticsTests(unittest.TestCase):
    def test_public_fsi_api_requires_generic_3d_velocity_names(self) -> None:
        for method_name in (
            "spread_fsi_forces",
            "spread_fsi_velocity_constraints",
            "diagnose_from_fields",
        ):
            parameters = inspect.signature(
                getattr(TriSurfaceRegionDiagnostics, method_name)
            ).parameters
            self.assertIn("primary_velocity_mps", parameters)
            self.assertIn("secondary_velocity_mps", parameters)
            self.assertNotIn("main_velocity_z_mps", parameters)
            self.assertNotIn("tail_velocity_z_mps", parameters)
            self.assertNotIn("main_velocity_mps", parameters)
            self.assertNotIn("tail_velocity_mps", parameters)
            self.assertIn("grid_fields", parameters)
        self.assertFalse(hasattr(TriSurfaceRegionDiagnostics, "diagnose"))

    def test_kernels_accept_vector_target_velocities_without_component_aliases(self) -> None:
        source = Path("simulation_core/tri_surface.py").read_text(encoding="utf-8")

        self.assertIn("primary_target_velocity_mps: ti.types.vector(3, ti.f32)", source)
        self.assertIn("secondary_target_velocity_mps: ti.types.vector(3, ti.f32)", source)
        for token in (
            "primary_target_vx_mps",
            "primary_target_vy_mps",
            "primary_target_vz_mps",
            "secondary_target_vx_mps",
            "secondary_target_vy_mps",
            "secondary_target_vz_mps",
        ):
            self.assertNotIn(token, source, msg=token)

    def test_surface_stress_diagnostics_require_explicit_viscosity(self) -> None:
        parameter = inspect.signature(
            TriSurfaceRegionDiagnostics.diagnose_from_fields
        ).parameters["viscosity_pa_s"]

        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_pressure_traction_and_zero_residual_on_static_triangle(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.pressure.from_numpy(np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32))
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertEqual(diagnostics.last_report_host_reads, 3)
        self.assertEqual(report.pressure_traction_face_count, 1)
        self.assertAlmostEqual(report.pressure_traction_abs_force_n, 20.0, delta=1.0e-5)
        self.assertAlmostEqual(report.pressure_traction_force_n[2], -20.0, delta=1.0e-5)
        self.assertEqual(report.projected_ibm_sample_count, 1)
        self.assertEqual(report.invalid_probe_count, 0)
        self.assertAlmostEqual(report.projected_ibm_residual_mps, 0.0, delta=1.0e-7)

    def test_load_faces_normalizes_normals_for_pressure_traction(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.pressure.from_numpy(np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32))
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertEqual(report.pressure_traction_face_count, 1)
        np.testing.assert_allclose(report.pressure_traction_force_n, (-20.0, 0.0, 0.0), rtol=1.0e-5)

    def test_nonzero_pressure_traction_spreads_equal_opposite_fluid_stress_action(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.pressure.from_numpy(np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32))
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=1.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        np.testing.assert_allclose(
            report.primary_pressure_traction_force_n,
            (0.0, 0.0, -20.0),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(report.primary_constraint_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)
        np.testing.assert_allclose(report.primary_fluid_stress_traction_force_n, (0.0, 0.0, -20.0), atol=1.0e-5)
        np.testing.assert_allclose(report.primary_fluid_force_n, (0.0, 0.0, 20.0), atol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, (0.0, 0.0, 20.0), atol=1.0e-5)

    def test_boundary_normal_velocity_spreads_volume_source_to_fluid_grid(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.2),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        grid_source_m3s = float(fluid.volume_source_s.to_numpy().sum() * fluid.spec.cell_volume_m3)
        self.assertAlmostEqual(report.volume_source_m3s, 0.4, delta=1.0e-5)
        self.assertAlmostEqual(report.primary_volume_source_m3s, 0.4, delta=1.0e-5)
        self.assertAlmostEqual(report.secondary_volume_source_m3s, 0.0, delta=1.0e-7)
        self.assertAlmostEqual(grid_source_m3s, 0.4, delta=1.0e-5)

    def test_fsi_force_spreading_conserves_force_near_domain_boundary(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.01, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.002, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        self.assertEqual(report.projected_ibm_sample_count, 1)
        self.assertEqual(report.invalid_probe_count, 0)
        self.assertGreater(report.active_force_cells, 0)
        expected_force_n = (1000.0 * 2.0 * 0.01 * 0.002 / 1.0e-3 * 0.5, 0.0, 0.0)
        np.testing.assert_allclose(report.primary_pressure_traction_force_n, (0.0, 0.0, 0.0), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.primary_fluid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.secondary_fluid_force_n, (0.0, 0.0, 0.0), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)

    def test_fsi_constraint_force_solid_mobility_ratio_reduces_constraint_impulse(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.01, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.002, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            constraint_force_solid_mobility_ratio=3.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        baseline_force_x_n = 1000.0 * 2.0 * 0.01 * 0.002 / 1.0e-3 * 0.5
        expected_force_n = (baseline_force_x_n / 4.0, 0.0, 0.0)
        np.testing.assert_allclose(report.primary_constraint_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.primary_fluid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)

    def test_region_solid_mobility_ratios_scale_primary_and_secondary_forces_independently(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray(
                [
                    [0.01, 0.4, 0.5],
                    [0.01, 0.6, 0.5],
                ],
                dtype=np.float32,
            ),
            normal=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.asarray([2.0, 4.0], dtype=np.float32),
            region_id=np.asarray([7, 8], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.002, 0.0, 0.0),
            secondary_velocity_mps=(0.004, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            constraint_force_solid_mobility_ratio=0.0,
            primary_constraint_force_solid_mobility_ratio=1.0,
            secondary_constraint_force_solid_mobility_ratio=3.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        primary_baseline_force_x_n = 1000.0 * 2.0 * 0.01 * 0.002 / 1.0e-3 * 0.5
        secondary_baseline_force_x_n = 1000.0 * 4.0 * 0.01 * 0.004 / 1.0e-3 * 0.5
        np.testing.assert_allclose(
            report.primary_constraint_force_n,
            (primary_baseline_force_x_n / 2.0, 0.0, 0.0),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            report.secondary_constraint_force_n,
            (secondary_baseline_force_x_n / 4.0, 0.0, 0.0),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_fsi_constraint_force_rejects_invalid_solid_mobility_ratio(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.01, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        for value in (-0.01, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "constraint_force_solid_mobility_ratio"):
                    diagnostics.spread_fsi_forces(
                        fluid.velocity,
                        fluid.pressure,
                        fluid.force,
                        fluid.volume_source_s,
                        fluid.obstacle,
                        grid_fields=fluid,
                        primary_region_id=7,
                        secondary_region_id=8,
                        primary_velocity_mps=(0.002, 0.0, 0.0),
                        secondary_velocity_mps=(0.0, 0.0, 0.0),
                        probe_distance_m=0.01,
                        density_kgm3=1000.0,
                        viscosity_pa_s=fluid.spec.viscosity_pa_s,
                        dt_s=1.0e-3,
                        constraint_force_scale=0.5,
                        constraint_force_solid_mobility_ratio=value,
                        bounds_min_m=(0.0, 0.0, 0.0),
                        bounds_max_m=(1.0, 1.0, 1.0),
                        spacing_m=fluid.spec.spacing_m,
                        grid_nodes=fluid.spec.grid_nodes,
                    )

    def test_fsi_force_spreading_can_return_lightweight_reaction_forces(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            read_full_report=False,
        )

        expected_force_n = (1000.0 * 2.0 * 0.002 * 0.2 / 1.0e-3 * 0.5, 0.0, 0.0)
        self.assertEqual(diagnostics.last_report_host_reads, 1)
        self.assertEqual(report.force_sample_count, 1)
        self.assertEqual(report.force_invalid_probe_count, 0)
        self.assertEqual(report.force_valid_probe_count, 1)
        self.assertAlmostEqual(report.force_valid_probe_fraction, 1.0)
        self.assertEqual(report.invalid_probe_count, 0)
        self.assertAlmostEqual(report.valid_probe_fraction, 1.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 0.0)
        self.assertAlmostEqual(report.invalid_probe_volume_source_m3s, 0.0)
        np.testing.assert_allclose(report.primary_fluid_force_n, expected_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.secondary_fluid_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)

    def test_lightweight_fsi_force_spread_can_be_diagnosed_without_reapplying_force(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        full_fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        lightweight_fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        velocity = np.zeros(full_fluid.spec.grid_nodes + (3,), dtype=np.float32)
        velocity[..., 0] = 0.03
        velocity[..., 1] = -0.01
        pressure = np.full(full_fluid.spec.grid_nodes, 4.0, dtype=np.float32)
        for fluid in (full_fluid, lightweight_fluid):
            fluid.velocity.from_numpy(velocity)
            fluid.pressure.from_numpy(pressure)
        full_diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        lightweight_diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        centroids = np.asarray(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5625],
            ],
            dtype=np.float32,
        )
        normals = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        areas = np.asarray([1.5, 0.75], dtype=np.float32)
        regions = np.asarray([7, 8], dtype=np.int32)
        for diagnostics in (full_diagnostics, lightweight_diagnostics):
            diagnostics.load_faces(
                centroid_m=centroids,
                normal=normals,
                area_m2=areas,
                region_id=regions,
            )

        kwargs = dict(
            grid_fields=full_fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, -0.15),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=full_fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=full_fluid.spec.spacing_m,
            grid_nodes=full_fluid.spec.grid_nodes,
        )
        full_report = full_diagnostics.spread_fsi_forces(
            full_fluid.velocity,
            full_fluid.pressure,
            full_fluid.force,
            full_fluid.volume_source_s,
            full_fluid.obstacle,
            **kwargs,
        )
        lightweight_kwargs = dict(kwargs)
        lightweight_kwargs["grid_fields"] = lightweight_fluid
        lightweight_diagnostics.spread_fsi_forces(
            lightweight_fluid.velocity,
            lightweight_fluid.pressure,
            lightweight_fluid.force,
            lightweight_fluid.volume_source_s,
            lightweight_fluid.obstacle,
            read_full_report=False,
            **lightweight_kwargs,
        )
        lightweight_fluid.clear_force()
        lightweight_fluid.clear_volume_source()
        force_before = lightweight_fluid.force.to_numpy()
        volume_before = lightweight_fluid.volume_source_s.to_numpy()

        report = lightweight_diagnostics.diagnose_fsi_forces_from_fields(
            lightweight_fluid.velocity,
            lightweight_fluid.pressure,
            lightweight_fluid.force,
            lightweight_fluid.volume_source_s,
            lightweight_fluid.obstacle,
            **lightweight_kwargs,
        )

        np.testing.assert_allclose(lightweight_fluid.force.to_numpy(), force_before, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            lightweight_fluid.volume_source_s.to_numpy(),
            volume_before,
            rtol=0.0,
            atol=0.0,
        )
        for name in (
            "pressure_traction_force_n",
            "viscous_traction_force_n",
            "fluid_stress_traction_force_n",
            "grid_force_n",
            "primary_fluid_force_n",
            "secondary_fluid_force_n",
            "constraint_force_n",
            "primary_constraint_force_n",
            "secondary_constraint_force_n",
        ):
            np.testing.assert_allclose(
                getattr(report, name),
                getattr(full_report, name),
                rtol=1.0e-6,
                atol=1.0e-7,
                err_msg=name,
            )
        self.assertEqual(report.projected_ibm_sample_count, full_report.projected_ibm_sample_count)
        self.assertEqual(report.force_valid_probe_count, full_report.force_valid_probe_count)
        self.assertAlmostEqual(report.volume_source_m3s, full_report.volume_source_m3s, delta=1.0e-9)
        self.assertEqual(report.active_force_cells, full_report.active_force_cells)

    def test_report_only_active_force_cells_match_final_cancelled_force_field(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        full_fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        lightweight_fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        full_diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        lightweight_diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        centroids = np.asarray(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        normals = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        areas = np.asarray([1.0, 1.0], dtype=np.float32)
        regions = np.asarray([7, 8], dtype=np.int32)
        for diagnostics in (full_diagnostics, lightweight_diagnostics):
            diagnostics.load_faces(
                centroid_m=centroids,
                normal=normals,
                area_m2=areas,
                region_id=regions,
            )

        kwargs = dict(
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=full_fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            primary_interface_impedance_force_n=(1.0, 0.0, 0.0),
            secondary_interface_impedance_force_n=(-1.0, 0.0, 0.0),
            primary_interface_area_m2=1.0,
            secondary_interface_area_m2=1.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=full_fluid.spec.spacing_m,
            grid_nodes=full_fluid.spec.grid_nodes,
        )
        full_report = full_diagnostics.spread_fsi_forces(
            full_fluid.velocity,
            full_fluid.pressure,
            full_fluid.force,
            full_fluid.volume_source_s,
            full_fluid.obstacle,
            grid_fields=full_fluid,
            **kwargs,
        )
        lightweight_fluid.clear_force()
        lightweight_fluid.clear_volume_source()
        diagnostic_report = lightweight_diagnostics.diagnose_fsi_forces_from_fields(
            lightweight_fluid.velocity,
            lightweight_fluid.pressure,
            lightweight_fluid.force,
            lightweight_fluid.volume_source_s,
            lightweight_fluid.obstacle,
            grid_fields=lightweight_fluid,
            **kwargs,
        )

        self.assertEqual(full_report.active_force_cells, 0)
        self.assertEqual(diagnostic_report.active_force_cells, 0)
        self.assertEqual(diagnostic_report.active_force_cells, full_report.active_force_cells)
        np.testing.assert_allclose(
            full_fluid.force.to_numpy(),
            np.zeros((*full_fluid.spec.grid_nodes, 3), dtype=np.float32),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(diagnostic_report.grid_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)

    def test_fsi_force_spreading_can_accumulate_impulse_without_force_pair_read(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        diagnostics.reset_force_impulse_accumulator()
        sentinel_snapshot = np.linspace(10.0, 19.0, 10, dtype=np.float32)
        diagnostics.report_force_pair_snapshot.from_numpy(sentinel_snapshot)
        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            read_full_report=False,
            read_force_pair_report=False,
        )
        diagnostics.accumulate_force_impulse(1.0e-3)
        primary_impulse, secondary_impulse = diagnostics.force_impulse_report()
        snapshot_after_force_spread = diagnostics.report_force_pair_snapshot.to_numpy()

        expected_force_n = (1000.0 * 2.0 * 0.002 * 0.2 / 1.0e-3 * 0.5, 0.0, 0.0)
        self.assertIsNone(report)
        self.assertEqual(diagnostics.last_report_host_reads, 0)
        self.assertEqual(diagnostics.last_force_impulse_host_reads, 1)
        np.testing.assert_allclose(
            snapshot_after_force_spread,
            sentinel_snapshot,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            primary_impulse,
            np.asarray(expected_force_n) * 1.0e-3,
            rtol=1.0e-5,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(secondary_impulse, (0.0, 0.0, 0.0), atol=1.0e-9)

    def test_fsi_force_spreading_applies_interface_impedance_to_fluid(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.0,
            primary_interface_impedance_force_n=(0.0, 0.0, -2.0),
            primary_interface_area_m2=2.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        np.testing.assert_allclose(report.primary_fluid_force_n, (0.0, 0.0, 2.0), atol=1.0e-6)
        np.testing.assert_allclose(report.grid_force_n, (0.0, 0.0, 2.0), atol=1.0e-6)
        np.testing.assert_allclose(report.constraint_force_n, (0.0, 0.0, 0.0), atol=1.0e-9)
        fluid.clear_force()
        fluid.clear_volume_source()
        force_before = fluid.force.to_numpy()
        volume_before = fluid.volume_source_s.to_numpy()

        diagnostic_report = diagnostics.diagnose_fsi_forces_from_fields(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            primary_interface_impedance_force_n=(0.0, 0.0, -2.0),
            primary_interface_area_m2=2.0,
        )

        np.testing.assert_allclose(diagnostic_report.primary_fluid_force_n, report.primary_fluid_force_n, atol=1.0e-6)
        np.testing.assert_allclose(diagnostic_report.grid_force_n, report.grid_force_n, atol=1.0e-6)
        np.testing.assert_allclose(fluid.force.to_numpy(), force_before, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(fluid.volume_source_s.to_numpy(), volume_before, rtol=0.0, atol=0.0)

    def test_pressure_robin_matrix_terms_scatter_from_interface_markers(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        diagnostics.spread_pressure_interface_matrix_terms(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_pressure_robin_impedance_ns_m=4.0,
            secondary_pressure_robin_impedance_ns_m=0.0,
            primary_pressure_robin_reference_pa=10.0,
            secondary_pressure_robin_reference_pa=0.0,
            primary_interface_area_m2=2.0,
            secondary_interface_area_m2=0.0,
            density_kgm3=1000.0,
            dt_s=1.0e-3,
            probe_distance_m=0.002,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        cell_volume_m3 = fluid.spec.cell_volume_m3
        diagonal_integral = float(
            np.sum(fluid.pressure_interface_matrix_diagonal.to_numpy()) * cell_volume_m3
        )
        rhs_integral = float(
            np.sum(fluid.pressure_interface_matrix_rhs.to_numpy()) * cell_volume_m3
        )
        expected_diagonal_integral = 1000.0 / 1.0e-3 * (2.0 * 2.0 / 4.0)
        self.assertAlmostEqual(
            diagonal_integral,
            expected_diagonal_integral,
            delta=expected_diagonal_integral * 1.0e-5,
        )
        self.assertAlmostEqual(
            rhs_integral,
            expected_diagonal_integral * 10.0,
            delta=expected_diagonal_integral * 10.0 * 1.0e-5,
        )

    def test_fsi_velocity_mobility_ratio_blends_boundary_velocity_and_volume_source(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        velocity = np.zeros((*fluid.spec.grid_nodes, 3), dtype=np.float32)
        velocity[..., 0] = 0.1
        fluid.velocity.from_numpy(velocity)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.3, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.002,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            primary_velocity_target_solid_mobility_ratio=1.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        expected_effective_velocity_x = 0.1 + (0.3 - 0.1) / 2.0
        expected_force_x = (
            1000.0
            * 2.0
            * 0.002
            * (expected_effective_velocity_x - 0.1)
            / 1.0e-3
            * 0.5
        )
        self.assertAlmostEqual(report.projected_ibm_residual_mps, 0.1, delta=1.0e-6)
        np.testing.assert_allclose(
            report.constraint_force_n,
            (expected_force_x, 0.0, 0.0),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        self.assertAlmostEqual(
            report.volume_source_m3s,
            expected_effective_velocity_x * 2.0,
            delta=1.0e-6,
        )

    def test_fsi_force_spreading_renormalizes_around_obstacle_cells(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        obstacle = np.zeros(fluid.spec.grid_nodes, dtype=np.int32)
        obstacle[5, 5, 5] = 1
        fluid.obstacle.from_numpy(obstacle)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.49, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.002, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )
        impulse = fluid.apply_body_force(dt_s=1.0e-3)

        expected_force_n = (1000.0 * 2.0 * 0.01 * 0.002 / 1.0e-3 * 0.5, 0.0, 0.0)
        np.testing.assert_allclose(report.primary_fluid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, expected_force_n, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            impulse.momentum_delta_n_s,
            (expected_force_n[0] * 1.0e-3, 0.0, 0.0),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_fsi_force_spread_reports_force_probe_valid_fraction(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.995, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.02,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=1.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        self.assertEqual(report.force_sample_count, 0)
        self.assertEqual(report.force_invalid_probe_count, 1)
        self.assertEqual(report.force_valid_probe_count, 0)
        self.assertAlmostEqual(report.force_valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 2.0, delta=1.0e-7)
        self.assertAlmostEqual(report.invalid_probe_volume_source_m3s, 0.4, delta=1.0e-7)
        self.assertEqual(report.active_force_cells, 0)
        np.testing.assert_allclose(report.grid_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)
        self.assertAlmostEqual(report.volume_source_m3s, 0.0, delta=1.0e-7)
        grid_source_m3s = float(fluid.volume_source_s.to_numpy().sum() * fluid.spec.cell_volume_m3)
        self.assertAlmostEqual(grid_source_m3s, 0.0, delta=1.0e-7)

    def test_lightweight_fsi_force_report_keeps_invalid_probe_diagnostics(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.995, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.02,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=1.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            read_full_report=False,
        )

        self.assertEqual(diagnostics.last_report_host_reads, 1)
        self.assertEqual(report.force_sample_count, 0)
        self.assertEqual(report.force_invalid_probe_count, 1)
        self.assertEqual(report.force_valid_probe_count, 0)
        self.assertAlmostEqual(report.force_valid_probe_fraction, 0.0)
        self.assertEqual(report.invalid_probe_count, 1)
        self.assertAlmostEqual(report.valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 2.0, delta=1.0e-7)
        self.assertAlmostEqual(report.invalid_probe_volume_source_m3s, 0.4, delta=1.0e-7)
        np.testing.assert_allclose(report.primary_fluid_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)
        np.testing.assert_allclose(report.secondary_fluid_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)
        self.assertAlmostEqual(float(fluid.force.to_numpy().sum()), 0.0, delta=1.0e-7)

    def test_fsi_velocity_constraint_reports_invalid_probe_extent(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.995, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        fluid.clear_velocity_constraints()
        report = diagnostics.spread_fsi_velocity_constraints(
            fluid.velocity_constraint_sum,
            fluid.velocity_constraint_weight,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.02,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        self.assertEqual(report.projected_ibm_sample_count, 0)
        self.assertEqual(report.invalid_probe_count, 1)
        self.assertAlmostEqual(report.valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 2.0, delta=1.0e-7)
        self.assertAlmostEqual(report.invalid_probe_volume_source_m3s, 0.4, delta=1.0e-7)
        self.assertAlmostEqual(float(fluid.velocity_constraint_weight.to_numpy().sum()), 0.0)
        np.testing.assert_allclose(
            fluid.velocity_constraint_sum.to_numpy().sum(axis=(0, 1, 2)),
            (0.0, 0.0, 0.0),
            atol=1.0e-6,
        )
        self.assertTrue(all(math.isnan(value) for value in report.pressure_traction_force_n))
        self.assertTrue(math.isnan(report.pressure_traction_abs_force_n))
        self.assertTrue(math.isnan(report.projected_ibm_residual_mps))
        self.assertTrue(all(math.isnan(value) for value in report.grid_force_n))
        self.assertTrue(all(math.isnan(value) for value in report.primary_fluid_force_n))
        self.assertTrue(math.isnan(report.volume_source_m3s))
        self.assertIsNone(report.force_sample_count)
        self.assertIsNone(report.active_force_cells)

    def test_fsi_velocity_constraint_can_skip_report_without_skipping_scatter(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        fluid.clear_velocity_constraints()
        report = diagnostics.spread_fsi_velocity_constraints(
            fluid.velocity_constraint_sum,
            fluid.velocity_constraint_weight,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, -0.25),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            read_full_report=False,
        )

        self.assertIsNone(report)
        self.assertEqual(diagnostics.last_report_host_reads, 0)
        self.assertGreater(float(fluid.velocity_constraint_weight.to_numpy().sum()), 0.0)

    def test_diagnose_from_fields_reports_out_of_bounds_probe_without_clamped_traction(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.pressure.from_numpy(np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32))
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.995, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.02,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertEqual(report.projected_ibm_sample_count, 0)
        self.assertEqual(report.invalid_probe_count, 1)
        self.assertAlmostEqual(report.valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 2.0, delta=1.0e-7)
        self.assertAlmostEqual(report.invalid_probe_volume_source_m3s, 0.4, delta=1.0e-7)
        self.assertEqual(report.pressure_traction_face_count, 0)
        np.testing.assert_allclose(report.pressure_traction_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)
        np.testing.assert_allclose(report.fluid_stress_traction_force_n, (0.0, 0.0, 0.0), atol=1.0e-7)

    def test_viscous_gradient_ignores_obstacle_cell_velocity(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        obstacle = np.zeros(fluid.spec.grid_nodes, dtype=np.int32)
        obstacle[2, 5, 5] = 1
        velocity = np.zeros(fluid.spec.grid_nodes + (3,), dtype=np.float32)
        velocity[2, 5, 5, 0] = 120.0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity.from_numpy(velocity)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.125, 0.4583333333, 0.4583333333]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=1.0e-9,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=1.0,
        )

        self.assertEqual(report.projected_ibm_sample_count, 1)
        np.testing.assert_allclose(report.viscous_traction_force_n, (0.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(report.fluid_stress_traction_force_n, (0.0, 0.0, 0.0), atol=1.0e-6)

    def test_velocity_sampling_ignores_obstacle_cell_velocity(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        obstacle = np.zeros(fluid.spec.grid_nodes, dtype=np.int32)
        obstacle[2, 5, 5] = 1
        velocity = np.zeros(fluid.spec.grid_nodes + (3,), dtype=np.float32)
        velocity[2, 5, 5, 0] = 120.0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity.from_numpy(velocity)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[1.0 / 6.0, 0.4583333333, 0.4583333333]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=1.0e-9,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=1.0,
        )

        self.assertEqual(report.projected_ibm_sample_count, 1)
        self.assertAlmostEqual(report.projected_ibm_residual_mps, 0.0, delta=1.0e-6)
        self.assertAlmostEqual(report.projected_ibm_residual_l2_mps, 0.0, delta=1.0e-6)

    def test_pressure_sampling_ignores_obstacle_cell_pressure(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        obstacle = np.zeros(fluid.spec.grid_nodes, dtype=np.int32)
        obstacle[2, 5, 5] = 1
        pressure = np.full(fluid.spec.grid_nodes, 10.0, dtype=np.float32)
        pressure[2, 5, 5] = 0.0
        fluid.obstacle.from_numpy(obstacle)
        fluid.pressure.from_numpy(pressure)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[1.0 / 6.0, 0.4583333333, 0.4583333333]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=1.0e-9,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=1.0,
        )

        self.assertEqual(report.pressure_traction_face_count, 1)
        np.testing.assert_allclose(report.pressure_traction_force_n, (-20.0, 0.0, 0.0), atol=1.0e-5)

    def test_velocity_sampling_reports_probe_invalid_when_all_support_cells_are_obstacles(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.obstacle.from_numpy(np.ones(fluid.spec.grid_nodes, dtype=np.int32))
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.0,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=1.0,
        )

        self.assertEqual(report.projected_ibm_sample_count, 0)
        self.assertEqual(report.invalid_probe_count, 1)
        self.assertAlmostEqual(report.valid_probe_fraction, 0.0)
        self.assertAlmostEqual(report.invalid_probe_area_m2, 2.0, delta=1.0e-6)

    def test_fsi_constraint_force_report_is_split_by_region(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=2, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray(
                [
                    [0.4, 0.5, 0.5],
                    [0.6, 0.5, 0.5],
                ],
                dtype=np.float32,
            ),
            normal=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            area_m2=np.asarray([2.0, 3.0], dtype=np.float32),
            region_id=np.asarray([7, 8], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.2),
            secondary_velocity_mps=(0.0, 0.0, -0.1),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        local_width_m = 1.0 / 16.0
        expected_primary_force_n = (0.0, 0.0, 1000.0 * 2.0 * 0.01 * 0.2 / 1.0e-3 * 0.5)
        expected_secondary_force_n = (0.0, 0.0, 1000.0 * 3.0 * 0.01 * -0.1 / 1.0e-3 * 0.5)
        expected_total_force_n = (
            expected_primary_force_n[0] + expected_secondary_force_n[0],
            expected_primary_force_n[1] + expected_secondary_force_n[1],
            expected_primary_force_n[2] + expected_secondary_force_n[2],
        )
        np.testing.assert_allclose(report.primary_constraint_force_n, expected_primary_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.secondary_constraint_force_n, expected_secondary_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.primary_fluid_force_n, report.primary_constraint_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.secondary_fluid_force_n, report.secondary_constraint_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.constraint_force_n, expected_total_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, report.constraint_force_n, rtol=1.0e-5)

    def test_fsi_constraint_force_uses_full_3d_target_velocity(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.2, -0.1, 0.05),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )

        expected_force_n = tuple(
            1000.0 * 2.0 * 0.01 * component / 1.0e-3 * 0.5
            for component in (0.2, -0.1, 0.05)
        )
        np.testing.assert_allclose(report.primary_constraint_force_n, expected_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.constraint_force_n, report.primary_constraint_force_n, rtol=1.0e-5)
        np.testing.assert_allclose(report.grid_force_n, report.constraint_force_n, rtol=1.0e-5)

    def test_nonuniform_fsi_constraint_force_respects_explicit_probe_distance(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.05, 0.10, 0.30, 0.55),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=grid.grid_nodes,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.25]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.2),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.05,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            spacing_m=(0.05, 0.05, 0.05),
            grid_nodes=grid.grid_nodes,
        )

        expected_force_z_n = 1000.0 * 2.0 * 0.05 * 0.2 / 1.0e-3 * 0.5
        local_grid_thickness_force_z_n = 1000.0 * 2.0 * 0.30 * 0.2 / 1.0e-3 * 0.5
        self.assertAlmostEqual(
            report.primary_constraint_force_n[2],
            expected_force_z_n,
            delta=expected_force_z_n * 1.0e-5,
        )
        self.assertEqual(report.force_invalid_probe_count, 0)
        self.assertEqual(report.force_valid_probe_count, 1)
        self.assertAlmostEqual(report.force_valid_probe_fraction, 1.0)
        self.assertLess(
            report.primary_constraint_force_n[2],
            local_grid_thickness_force_z_n * 0.25,
        )

    def test_nonuniform_fsi_constraint_force_falls_back_to_local_grid_thickness_without_probe_distance(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.05, 0.10, 0.30, 0.55),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=grid.grid_nodes,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.25]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.spread_fsi_forces(
            fluid.velocity,
            fluid.pressure,
            fluid.force,
            fluid.volume_source_s,
            fluid.obstacle,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.2),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.0,
            density_kgm3=1000.0,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
            dt_s=1.0e-3,
            constraint_force_scale=0.5,
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            spacing_m=(0.05, 0.05, 0.05),
            grid_nodes=grid.grid_nodes,
        )

        expected_force_z_n = 1000.0 * 2.0 * 0.25 * 0.2 / 1.0e-3 * 0.5
        self.assertAlmostEqual(
            report.primary_constraint_force_n[2],
            expected_force_z_n,
            delta=expected_force_z_n * 1.0e-5,
        )

    def test_region_offset_moves_pressure_sampling_point(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(32, 32, 32), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.set_vertical_pressure_gradient(
            reference_height_m=0.0,
            gradient_z_pa_per_m=10.0,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.4]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        before = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )
        diagnostics.update_region_offsets(
            primary_region_id=7,
            secondary_region_id=8,
            primary_offset_m=(0.0, 0.0, 0.2),
            secondary_offset_m=(0.0, 0.0, 0.0),
        )
        after = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertLess(after.pressure_traction_force_n[2], before.pressure_traction_force_n[2] - 1.5)

    def test_diagnose_from_fields_reports_viscous_stress_traction(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(32, 32, 32), dt_s=1.0e-3),
            runtime=runtime,
        )
        fluid.set_simple_shear_velocity(shear_rate_s=4.0, center_y_m=0.5)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([2.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=0.25,
        )

        np.testing.assert_allclose(report.pressure_traction_force_n, (0.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(report.viscous_traction_force_n, (2.0, 0.0, 0.0), rtol=1.0e-4)
        np.testing.assert_allclose(report.fluid_stress_traction_force_n, (2.0, 0.0, 0.0), rtol=1.0e-4)

    def test_nonuniform_diagnose_viscous_stress_uses_physical_center_span(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.10, 0.20, 0.40, 0.30),
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=grid.grid_nodes,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=runtime,
        )
        fluid.set_simple_shear_velocity(shear_rate_s=4.0, center_y_m=0.0)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.35, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.0,
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            spacing_m=(0.25, 0.25, 0.25),
            grid_nodes=grid.grid_nodes,
            viscosity_pa_s=0.25,
        )

        np.testing.assert_allclose(report.viscous_traction_force_n, (1.0, 0.0, 0.0), rtol=1.0e-4)
        np.testing.assert_allclose(report.fluid_stress_traction_force_n, (1.0, 0.0, 0.0), rtol=1.0e-4)

    def test_nonuniform_diagnose_viscous_stress_samples_adjacent_centers(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.10, 0.20, 0.40, 0.30),
            cell_widths_z_m=(0.25, 0.25, 0.25, 0.25),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=grid.grid_nodes,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=runtime,
        )
        velocity = np.zeros((*grid.grid_nodes, 3), dtype=np.float32)
        for index, center_y_m in enumerate(grid.cell_centers_y_m):
            velocity[:, index, :, 0] = float(center_y_m) ** 2
        fluid.velocity.from_numpy(velocity)
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.35, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, 0.0),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=1.0e-9,
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            spacing_m=(0.25, 0.25, 0.25),
            grid_nodes=grid.grid_nodes,
            viscosity_pa_s=1.0,
        )

        expected_du_dy = (0.5**2 - 0.2**2) / (0.5 - 0.2)
        np.testing.assert_allclose(
            report.viscous_traction_force_n,
            (expected_du_dy, 0.0, 0.0),
            rtol=1.0e-4,
        )

    def test_fsi_velocity_constraint_enforces_probe_no_slip(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        fluid.clear_velocity_constraints()
        spread_report = diagnostics.spread_fsi_velocity_constraints(
            fluid.velocity_constraint_sum,
            fluid.velocity_constraint_weight,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, -0.25),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )
        constraint_report = fluid.apply_velocity_constraints(blend=1.0)
        residual_report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, -0.25),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertEqual(spread_report.projected_ibm_sample_count, 1)
        self.assertGreater(constraint_report.active_cells, 0)
        np.testing.assert_allclose(
            constraint_report.primary_momentum_delta_n_s,
            constraint_report.momentum_delta_n_s,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            constraint_report.secondary_momentum_delta_n_s,
            (0.0, 0.0, 0.0),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(residual_report.projected_ibm_residual_mps, 0.0, delta=1.0e-6)

    def test_fsi_velocity_constraint_can_survive_pressure_projection(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda")
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=runtime,
        )
        diagnostics = TriSurfaceRegionDiagnostics(face_capacity=1, runtime=runtime)
        diagnostics.load_faces(
            centroid_m=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
            normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            area_m2=np.asarray([1.0], dtype=np.float32),
            region_id=np.asarray([7], dtype=np.int32),
        )

        fluid.clear_velocity_constraints()
        diagnostics.spread_fsi_velocity_constraints(
            fluid.velocity_constraint_sum,
            fluid.velocity_constraint_weight,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, -0.25),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
        )
        fluid.apply_velocity_constraints(blend=1.0)
        fluid.set_vertical_pressure_gradient(
            reference_height_m=0.0,
            gradient_z_pa_per_m=100.0,
        )

        fluid.project(
            iterations=1,
            preserve_velocity_constraints=True,
            velocity_constraint_blend=1.0,
        )
        residual_report = diagnostics.diagnose_from_fields(
            fluid.velocity,
            fluid.pressure,
            grid_fields=fluid,
            primary_region_id=7,
            secondary_region_id=8,
            primary_velocity_mps=(0.0, 0.0, -0.25),
            secondary_velocity_mps=(0.0, 0.0, 0.0),
            probe_distance_m=0.01,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            spacing_m=fluid.spec.spacing_m,
            grid_nodes=fluid.spec.grid_nodes,
            viscosity_pa_s=fluid.spec.viscosity_pa_s,
        )

        self.assertAlmostEqual(residual_report.projected_ibm_residual_mps, 0.0, delta=1.0e-6)


if __name__ == "__main__":
    unittest.main()
