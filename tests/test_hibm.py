from __future__ import annotations

import math
import unittest

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    CartesianGrid,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    NeoHookeanMpmState,
    SurfaceMesh,
    TaichiRuntimeConfig,
    TriMooneyShellMpmState,
    TriSurfaceRegionDiagnostics,
    HibmMpmSharpCouplingState,
    HibmMpmSharpNeoHookeanStepReport,
    advance_hibm_mpm_sharp_mpm_step,
    advance_hibm_mpm_sharp_neo_hookean_step,
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
)
from simulation_core.hibm import (
    build_hibm_ib_node_boundary_conditions,
    classify_hibm_near_boundary_nodes,
    compute_hibm_surface_tractions,
)


class HibmMpmSurfaceMarkerTests(unittest.TestCase):
    def test_marker_fields_compute_per_marker_force_from_traction_and_area(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            velocities_mps=((0.0, 0.0, 0.1), (0.0, 0.0, -0.2)),
            normals=((0.0, 0.0, 2.0), (0.0, 2.0, 0.0)),
            areas_m2=(0.5, 0.25),
            region_ids=(101, 202),
        )

        markers.set_marker_tractions_pa(
            ((2.0, 0.0, -4.0), (0.0, 8.0, 0.0))
        )
        markers.compute_marker_forces()

        self.assertEqual(markers.marker_count, 2)
        self.assertEqual(markers.marker_force_n(0), (1.0, 0.0, -2.0))
        self.assertEqual(markers.marker_force_n(1), (0.0, 2.0, 0.0))
        self.assertEqual(markers.marker_normal(0), (0.0, 0.0, 1.0))
        self.assertEqual(markers.marker_normal(1), (0.0, 1.0, 0.0))

    def test_marker_aggregation_preserves_action_reaction(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=3)
        markers.load_markers(
            positions_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            velocities_mps=((0.0, 0.0, 0.0),) * 3,
            normals=((1.0, 0.0, 0.0),) * 3,
            areas_m2=(1.0, 2.0, 4.0),
            region_ids=(101, 101, 202),
        )
        markers.set_marker_tractions_pa(
            ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, -3.0))
        )
        markers.compute_marker_forces()

        report = markers.aggregate_region_forces(
            primary_region_id=101,
            secondary_region_id=202,
        )

        self.assertEqual(report.primary_marker_force_n, (1.0, 4.0, 0.0))
        self.assertEqual(report.secondary_marker_force_n, (0.0, 0.0, -12.0))
        self.assertEqual(report.total_marker_force_n, (1.0, 4.0, -12.0))
        self.assertEqual(report.primary_marker_count, 2)
        self.assertEqual(report.secondary_marker_count, 1)
        self.assertEqual(report.total_marker_count, 3)
        self.assertEqual(report.fluid_reaction_force_n, (-1.0, -4.0, 12.0))
        self.assertTrue(math.isclose(report.action_reaction_residual_n, 0.0))

    def test_marker_loader_rejects_invalid_geometry(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)

        with self.assertRaisesRegex(ValueError, "areas_m2"):
            markers.load_markers(
                positions_m=((0.0, 0.0, 0.0),),
                velocities_mps=((0.0, 0.0, 0.0),),
                normals=((1.0, 0.0, 0.0),),
                areas_m2=(-1.0,),
                region_ids=(1,),
            )

        with self.assertRaisesRegex(ValueError, "normals"):
            markers.load_markers(
                positions_m=((0.0, 0.0, 0.0),),
                velocities_mps=((0.0, 0.0, 0.0),),
                normals=((0.0, 0.0, 0.0),),
                areas_m2=(1.0,),
                region_ids=(1,),
            )

    def test_marker_loader_uses_taichi_surface_fields_without_host_roundtrip(self) -> None:
        surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        surface.load_faces(
            centroid_m=np.array(
                ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
                dtype=np.float32,
            ),
            normal=np.array(
                ((0.0, 0.0, 2.0), (0.0, 3.0, 0.0)),
                dtype=np.float32,
            ),
            area_m2=np.array((0.25, 0.75), dtype=np.float32),
            region_id=np.array((101, 202), dtype=np.int32),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)

        markers.load_markers_from_surface_fields(
            surface.centroid_m,
            surface.normal,
            surface.area_m2,
            surface.region_id,
            marker_count=surface.face_count,
            initial_velocity_mps=(1.0, -2.0, 3.0),
        )

        self.assertEqual(markers.marker_count, 2)
        self.assertEqual(markers.marker_normal(0), (0.0, 0.0, 1.0))
        self.assertEqual(markers.marker_normal(1), (0.0, 1.0, 0.0))
        self.assertEqual(markers.marker_region_id(0), 101)
        self.assertEqual(markers.marker_region_id(1), 202)
        self.assertEqual(markers.marker_traction_pa(0), (0.0, 0.0, 0.0))
        self.assertEqual(markers.marker_force_n(1), (0.0, 0.0, 0.0))
        for axis, expected in enumerate((0.1, 0.2, 0.3)):
            self.assertAlmostEqual(
                float(markers.x_gamma_m[0][axis]),
                expected,
                delta=1.0e-6,
            )
        self.assertAlmostEqual(float(markers.A_gamma_m2[1]), 0.75, delta=1.0e-6)
        self.assertEqual(markers.marker_velocity_mps(1), (1.0, -2.0, 3.0))

    def test_marker_loader_can_take_surface_velocity_taichi_field_for_no_slip(
        self,
    ) -> None:
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.45, 0.55, 0.65)
        solid.v[0] = (0.25, -0.5, 0.75)
        solid.surface_normal[0] = (0.0, 2.0, 0.0)
        solid.area_weight_m2[0] = 0.125
        solid.region_id[0] = 202
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)

        markers.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            surface_velocity_mps=solid.v,
            marker_count=solid.particle_count,
        )

        self.assertEqual(markers.marker_velocity_mps(0), (0.25, -0.5, 0.75))
        self.assertEqual(markers.marker_normal(0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(float(markers.A_gamma_m2[0]), 0.125, delta=1.0e-6)

    def test_tail_surface_field_marker_participates_via_marker_force_aggregation(
        self,
    ) -> None:
        surface = TriSurfaceRegionDiagnostics(face_capacity=2)
        surface.load_faces(
            centroid_m=np.array(
                ((0.2, 0.2, 0.2), (0.7, 0.7, 0.7)),
                dtype=np.float32,
            ),
            normal=np.array(
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            ),
            area_m2=np.array((0.5, 2.0), dtype=np.float32),
            region_id=np.array((101, 202), dtype=np.int32),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers_from_surface_fields(
            surface.centroid_m,
            surface.normal,
            surface.area_m2,
            surface.region_id,
            marker_count=surface.face_count,
        )

        markers.set_marker_tractions_pa(((4.0, 0.0, 0.0), (0.0, 0.0, -3.0)))
        markers.compute_marker_forces()
        report = markers.aggregate_region_forces(
            primary_region_id=101,
            secondary_region_id=202,
        )

        self.assertEqual(report.primary_marker_force_n, (2.0, 0.0, 0.0))
        self.assertEqual(report.secondary_marker_force_n, (0.0, 0.0, -6.0))
        self.assertEqual(report.total_marker_force_n, (2.0, 0.0, -6.0))
        self.assertEqual(report.primary_marker_count, 1)
        self.assertEqual(report.secondary_marker_count, 1)
        self.assertEqual(report.total_marker_count, 2)
        self.assertTrue(math.isclose(report.action_reaction_residual_n, 0.0))

    def test_uniform_pressure_samples_full_stress_traction_to_markers(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.pressure.fill(7.0)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=fluid.mu,
        )
        markers.compute_marker_forces()

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(markers.marker_traction_pa(0), (-7.0, 0.0, 0.0))
        self.assertEqual(markers.marker_force_n(0), (-3.5, 0.0, 0.0))

    def test_two_sided_thin_wall_pressure_samples_jump_not_center_average(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :4] = 2.0
        pressure[:, :, 4:] = 10.0
        fluid.pressure.from_numpy(pressure)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=fluid.mu,
            two_sided_pressure=True,
        )
        markers.compute_marker_forces()

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 1)
        self.assertEqual(markers.marker_traction_pa(0), (0.0, 0.0, -8.0))
        self.assertEqual(markers.marker_force_n(0), (0.0, 0.0, -8.0))

    def test_simple_shear_samples_viscous_stress_traction_to_markers(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 1.0, 0.0),),
            areas_m2=(0.25,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                viscosity_pa_s=2.0,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.set_simple_shear_velocity(shear_rate_s=3.0, center_y_m=0.0)

        markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=fluid.mu,
        )
        markers.compute_marker_forces()

        traction = markers.marker_traction_pa(0)
        force = markers.marker_force_n(0)
        self.assertAlmostEqual(traction[0], 6.0, delta=1.0e-5)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-5)
        self.assertAlmostEqual(traction[2], 0.0, delta=1.0e-5)
        self.assertAlmostEqual(force[0], 1.5, delta=1.0e-5)
        self.assertAlmostEqual(force[1], 0.0, delta=1.0e-5)
        self.assertAlmostEqual(force[2], 0.0, delta=1.0e-5)

    def test_viscous_stress_marks_marker_invalid_when_gradient_neighbor_is_missing(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 1.0, 0.0),),
            areas_m2=(0.25,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                viscosity_pa_s=2.0,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.set_simple_shear_velocity(shear_rate_s=3.0, center_y_m=0.0)
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[:, 4, :] = 1
        fluid.obstacle.from_numpy(obstacle)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=fluid.mu,
        )
        markers.compute_marker_forces()

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.viscous_gradient_invalid_marker_count, 1)
        self.assertEqual(markers.marker_traction_pa(0), (0.0, 0.0, 0.0))
        self.assertEqual(markers.marker_force_n(0), (0.0, 0.0, 0.0))

    def test_two_sided_stress_sampling_walks_past_masked_wall_band_cells(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(101,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                viscosity_pa_s=2.0,
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.set_simple_shear_velocity(shear_rate_s=3.0, center_y_m=0.0)
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[:, :, :4] = 2.0
        pressure[:, :, 4:] = 10.0
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[:, :, 3:5] = 1
        fluid.obstacle.from_numpy(obstacle)

        report = markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=fluid.mu,
            two_sided_pressure=True,
        )
        markers.compute_marker_forces()

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.viscous_gradient_invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_no_slip_residual_samples_fluid_velocity_against_marker_velocity(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.625),),
            velocities_mps=((0.1, -0.1, 0.2),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.25,),
            region_ids=(202,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity[2, 2, 2] = (0.3, -0.1, 0.0)

        report = markers.sample_no_slip_residual(
            fluid.velocity,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.grid.grid_nodes,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertAlmostEqual(
            report.max_no_slip_residual_mps,
            math.sqrt(0.08),
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.l2_no_slip_residual_mps,
            math.sqrt(0.08),
            delta=1.0e-6,
        )

    def test_marker_forces_scatter_to_mpm_external_force_particles(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        markers.set_marker_tractions_pa(((2.0, -4.0, 6.0),))
        markers.compute_marker_forces()
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=0.25,
        )

        self.assertEqual(report.active_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.active_particle_count, 1)
        self.assertEqual(solid.external_force_n[0], (1.0, -2.0, 3.0))
        self.assertEqual(report.total_marker_force_n, (1.0, -2.0, 3.0))
        self.assertEqual(report.total_mpm_external_force_n, (1.0, -2.0, 3.0))
        self.assertAlmostEqual(report.action_reaction_residual_n, 0.0)

    def test_marker_force_scatter_partitions_force_across_supported_particles(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(202,),
        )
        markers.set_marker_tractions_pa(((0.0, 0.0, 4.0),))
        markers.compute_marker_forces()
        solid = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(2, 1, 1),
            box_min_m=(0.25, 0.0, 0.0),
            box_max_m=(0.75, 1.0, 1.0),
            density_kgm3=1000.0,
        )

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=0.4,
        )

        self.assertEqual(report.active_marker_count, 1)
        self.assertEqual(report.active_particle_count, 2)
        self.assertAlmostEqual(solid.external_force_n[0][2], 2.0, delta=1.0e-5)
        self.assertAlmostEqual(solid.external_force_n[1][2], 2.0, delta=1.0e-5)
        self.assertAlmostEqual(report.total_mpm_external_force_n[2], 4.0, delta=1.0e-5)
        self.assertAlmostEqual(report.action_reaction_residual_n, 0.0, delta=1.0e-5)

    def test_marker_force_scatter_clears_existing_mpm_external_force_first(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        markers.set_marker_tractions_pa(((2.0, -4.0, 6.0),))
        markers.compute_marker_forces()
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.external_force_n[0] = (99.0, 0.0, 0.0)

        clear_report = markers.clear_mpm_external_forces(
            solid.external_force_n,
            particle_count=solid.particle_count,
        )
        scatter_report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=0.25,
        )

        self.assertEqual(clear_report.cleared_particle_count, 1)
        self.assertAlmostEqual(clear_report.max_abs_external_force_before_n, 99.0)
        self.assertEqual(solid.external_force_n[0], (1.0, -2.0, 3.0))
        self.assertEqual(scatter_report.total_mpm_external_force_n, (1.0, -2.0, 3.0))

    def test_surface_feedback_updates_marker_position_and_velocity_from_mpm_particles(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(202,),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.6, 0.45, 0.7)
        solid.v[0] = (1.0, -2.0, 3.0)

        report = markers.update_surface_feedback_from_mpm_particles(
            solid.x,
            solid.v,
            particle_count=solid.particle_count,
            support_radius_m=0.5,
            dt_s=0.1,
        )

        self.assertEqual(report.updated_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        marker_position = tuple(float(markers.x_gamma_m[0][axis]) for axis in range(3))
        marker_velocity = tuple(float(markers.v_gamma_mps[0][axis]) for axis in range(3))
        for actual, expected in zip(marker_position, (0.6, 0.3, 0.8), strict=True):
            self.assertAlmostEqual(actual, expected, delta=1.0e-6)
        for actual, expected in zip(marker_velocity, (1.0, -2.0, 3.0), strict=True):
            self.assertAlmostEqual(actual, expected, delta=1.0e-6)

    def test_surface_feedback_updates_marker_normal_and_area_from_mpm_surface_fields(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.25,),
            region_ids=(202,),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.6, 0.45, 0.7)
        solid.v[0] = (1.0, -2.0, 3.0)
        solid.surface_normal[0] = (0.0, 2.0, 0.0)
        solid.area_weight_m2[0] = 0.75

        report = markers.update_surface_feedback_from_mpm_surface_particles(
            solid.x,
            solid.v,
            solid.surface_normal,
            solid.area_weight_m2,
            particle_count=solid.particle_count,
            support_radius_m=0.5,
            dt_s=0.1,
        )

        self.assertEqual(report.updated_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.geometry_updated_marker_count, 1)
        self.assertEqual(markers.marker_normal(0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(float(markers.A_gamma_m2[0]), 0.75, delta=1.0e-6)
        self.assertAlmostEqual(report.max_marker_area_change_m2, 0.5, delta=1.0e-6)
        self.assertGreater(report.max_marker_normal_change, 0.0)

    def test_surface_feedback_advances_marker_by_velocity_not_particle_centroid(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.25,),
            region_ids=(202,),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.9, 0.5, 0.5)
        solid.v[0] = (0.1, 0.0, 0.0)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.25

        report = markers.update_surface_feedback_from_mpm_surface_particles(
            solid.x,
            solid.v,
            solid.surface_normal,
            solid.area_weight_m2,
            particle_count=solid.particle_count,
            support_radius_m=0.5,
            dt_s=0.1,
        )

        marker_position = tuple(float(markers.x_gamma_m[0][axis]) for axis in range(3))
        marker_velocity = tuple(float(markers.v_gamma_mps[0][axis]) for axis in range(3))
        self.assertEqual(report.updated_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        np.testing.assert_allclose(marker_position, (0.51, 0.5, 0.5), atol=1.0e-6)
        np.testing.assert_allclose(marker_velocity, (0.1, 0.0, 0.0), atol=1.0e-6)
        self.assertAlmostEqual(report.max_marker_displacement_m, 0.01, delta=1.0e-6)

    def test_surface_feedback_records_invalid_marker_without_particle_support(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 1.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(303,),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(0.2, 0.2, 0.2),
            density_kgm3=1000.0,
        )

        report = markers.update_surface_feedback_from_mpm_particles(
            solid.x,
            solid.v,
            particle_count=solid.particle_count,
            support_radius_m=0.1,
            dt_s=0.1,
        )

        self.assertEqual(report.updated_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        marker_position = tuple(float(markers.x_gamma_m[0][axis]) for axis in range(3))
        marker_velocity = tuple(float(markers.v_gamma_mps[0][axis]) for axis in range(3))
        self.assertEqual(marker_position, (0.5, 0.5, 0.5))
        self.assertEqual(marker_velocity, (0.0, 0.0, 0.0))


class HibmMpmIbNodeSearchTests(unittest.TestCase):
    def test_plane_search_classifies_external_and_internal_nodes_on_taichi_fields(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.26,
            interior_probe_distance_m=0.125,
        )

        self.assertEqual(report.near_boundary_node_count, 2)
        self.assertEqual(report.external_ib_node_count, 1)
        self.assertEqual(report.internal_node_count, 1)
        self.assertEqual(report.invalid_projection_count, 0)
        self.assertEqual(search.node_kind((0, 0, 2)), "external_ib")
        self.assertEqual(search.node_kind((0, 0, 1)), "internal")
        self.assertEqual(search.nearest_marker_index((0, 0, 2)), 0)
        self.assertEqual(search.nearest_marker_index((0, 0, 1)), 0)
        self.assertEqual(search.boundary_point_m((0, 0, 2)), (0.5, 0.5, 0.5))
        self.assertEqual(search.interior_fluid_point_m((0, 0, 2)), (0.5, 0.5, 0.75))
        self.assertGreater(search.signed_distance_m((0, 0, 2)), 0.0)
        self.assertLess(search.signed_distance_m((0, 0, 1)), 0.0)

    def test_triangle_projection_finds_closest_surface_point_between_markers(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=3,
            projection_triangle_capacity=1,
        )
        markers.load_markers(
            positions_m=(
                (0.25, 0.25, 0.5),
                (0.75, 0.25, 0.5),
                (0.25, 0.75, 0.5),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 3,
            normals=((0.0, 0.0, 1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(7,) * 3,
        )
        markers.set_projection_triangles(((0, 1, 2),))
        search = HibmMpmIbNodeSearch(
            grid_nodes=(8, 8, 8),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=3,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.08,
            interior_probe_distance_m=0.05,
        )

        self.assertGreaterEqual(report.near_boundary_node_count, 1)
        self.assertGreaterEqual(report.external_ib_node_count, 1)
        self.assertEqual(search.node_kind((2, 2, 4)), "external_ib")
        boundary_point = search.boundary_point_m((2, 2, 4))
        np.testing.assert_allclose(
            boundary_point,
            (0.3125, 0.3125, 0.5),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(search.signed_distance_m((2, 2, 4)), 0.0625)
        np.testing.assert_allclose(
            search.interior_fluid_point_m((2, 2, 4)),
            (0.3125, 0.3125, 0.6125),
            atol=1.0e-6,
        )

    def test_closed_surface_search_classifies_deep_internal_nodes(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=6)
        markers.load_markers(
            positions_m=(
                (0.25, 0.5, 0.5),
                (0.75, 0.5, 0.5),
                (0.5, 0.25, 0.5),
                (0.5, 0.75, 0.5),
                (0.5, 0.5, 0.25),
                (0.5, 0.5, 0.75),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 6,
            normals=(
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
            ),
            areas_m2=(1.0,) * 6,
            region_ids=(7,) * 6,
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=6,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.1,
            interior_probe_distance_m=0.125,
            classify_far_internal_nodes=True,
        )

        self.assertEqual(report.near_boundary_node_count, 0)
        self.assertEqual(report.external_ib_node_count, 0)
        self.assertEqual(report.internal_node_count, 1)
        self.assertEqual(search.node_kind((0, 0, 0)), "internal")
        self.assertGreaterEqual(search.nearest_marker_index((0, 0, 0)), 0)

    def test_closed_surface_grid_field_search_classifies_deep_internal_nodes(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=6)
        markers.load_markers(
            positions_m=(
                (0.25, 0.5, 0.5),
                (0.75, 0.5, 0.5),
                (0.5, 0.25, 0.5),
                (0.5, 0.75, 0.5),
                (0.5, 0.5, 0.25),
                (0.5, 0.5, 0.75),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 6,
            normals=(
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
            ),
            areas_m2=(1.0,) * 6,
            region_ids=(7,) * 6,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=6,
        )

        report = search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            search_radius_m=0.08,
            interior_probe_distance_m=0.125,
            classify_far_internal_nodes=True,
        )

        self.assertEqual(report.near_boundary_node_count, 0)
        self.assertEqual(report.internal_node_count, 8)
        self.assertEqual(search.node_kind((1, 1, 1)), "internal")
        self.assertEqual(search.node_kind((2, 2, 2)), "internal")
        self.assertEqual(search.node_kind((0, 0, 0)), "none")
        masked_count = fluid.apply_hibm_internal_obstacles(
            search.node_kind_code,
            internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        )
        self.assertEqual(masked_count, 8)
        self.assertEqual(int(fluid.obstacle[1, 1, 1]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 2]), 1)
        self.assertEqual(int(fluid.obstacle[0, 0, 0]), 0)

    def test_default_sign_tolerance_flags_f32_scale_ambiguous_nodes(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5 - 5.0e-8),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.01,
            interior_probe_distance_m=0.125,
        )

        self.assertEqual(report.near_boundary_node_count, 1)
        self.assertEqual(report.invalid_projection_count, 1)
        self.assertEqual(search.node_kind((0, 0, 0)), "internal")
        self.assertGreater(search.signed_distance_m((0, 0, 0)), 0.0)

    def test_search_uses_nonuniform_fluid_cell_center_fields(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.1, 0.1, 0.2, 0.6),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.28),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=1,
        )

        report = search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            search_radius_m=0.04,
            interior_probe_distance_m=0.05,
        )

        self.assertEqual(report.near_boundary_node_count, 1)
        self.assertEqual(report.external_ib_node_count, 1)
        self.assertEqual(report.internal_node_count, 0)
        self.assertEqual(search.node_kind((2, 2, 2)), "external_ib")
        self.assertEqual(search.node_kind((2, 2, 1)), "none")
        self.assertAlmostEqual(search.signed_distance_m((2, 2, 2)), 0.02, delta=1.0e-6)
        boundary_point = search.boundary_point_m((2, 2, 2))
        self.assertAlmostEqual(boundary_point[0], 0.625, delta=1.0e-6)
        self.assertAlmostEqual(boundary_point[1], 0.625, delta=1.0e-6)
        self.assertAlmostEqual(boundary_point[2], 0.28, delta=1.0e-6)
        self.assertAlmostEqual(
            search.interior_fluid_point_m((2, 2, 2))[2],
            0.35,
            delta=1.0e-6,
        )

    def test_search_treats_node_as_external_if_any_local_normal_is_positive(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.55), (0.5, 0.5, 0.45)),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(7, 8),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.08,
            interior_probe_distance_m=0.05,
        )

        self.assertEqual(report.near_boundary_node_count, 1)
        self.assertEqual(report.external_ib_node_count, 1)
        self.assertEqual(report.internal_node_count, 0)
        self.assertEqual(search.node_kind((0, 0, 0)), "external_ib")

    def test_external_search_stores_external_projection_candidate(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.49), (0.5, 0.5, 0.46)),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(7, 8),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.05,
            interior_probe_distance_m=0.05,
            sign_tolerance_m=1.0e-8,
        )

        self.assertEqual(report.near_boundary_node_count, 1)
        self.assertEqual(report.external_ib_node_count, 1)
        self.assertEqual(report.internal_node_count, 0)
        self.assertEqual(search.node_kind((0, 0, 0)), "external_ib")
        self.assertEqual(search.nearest_marker_index((0, 0, 0)), 1)
        self.assertGreater(search.signed_distance_m((0, 0, 0)), 0.0)
        self.assertAlmostEqual(
            search.boundary_point_m((0, 0, 0))[2],
            0.46,
            delta=1.0e-6,
        )
        self.assertGreater(
            search.interior_fluid_point_m((0, 0, 0))[2],
            0.5,
        )

    def test_sphere_local_radial_marker_classifies_inside_and_outside_nodes(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.75, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
        )

        self.assertEqual(report.near_boundary_node_count, 2)
        self.assertEqual(report.external_ib_node_count, 1)
        self.assertEqual(report.internal_node_count, 1)
        self.assertEqual(search.node_kind((3, 0, 0)), "external_ib")
        self.assertEqual(search.node_kind((2, 0, 0)), "internal")
        self.assertGreater(search.signed_distance_m((3, 0, 0)), 0.0)
        self.assertLess(search.signed_distance_m((2, 0, 0)), 0.0)

    def test_tangent_projection_fallback_records_invalid_count(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )

        report = search.search_and_classify(
            markers,
            search_radius_m=0.1,
            interior_probe_distance_m=0.125,
            sign_tolerance_m=1.0e-9,
        )

        self.assertEqual(report.near_boundary_node_count, 1)
        self.assertEqual(report.invalid_projection_count, 1)
        self.assertEqual(search.nearest_marker_index((0, 0, 0)), 0)
        self.assertEqual(search.boundary_point_m((0, 0, 0)), (0.5, 0.5, 0.5))
        self.assertEqual(search.interior_fluid_point_m((0, 0, 0)), (0.5, 0.5, 0.625))


class HibmMpmIbBoundaryConditionTests(unittest.TestCase):
    def test_builds_external_ib_no_slip_and_pressure_neumann_targets(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((1.25, -0.5, 0.75),),
            normals=((0.0, 0.0, 2.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.26,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(1, 1, 4),
            marker_capacity=1,
        )

        report = boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(42.0,),
        )

        self.assertEqual(report.no_slip_dirichlet_count, 1)
        self.assertEqual(report.pressure_neumann_count, 1)
        self.assertEqual(report.inactive_internal_node_count, 1)
        self.assertTrue(boundary.is_active((0, 0, 2)))
        self.assertFalse(boundary.is_active((0, 0, 1)))
        self.assertEqual(boundary.velocity_dirichlet_mps((0, 0, 2)), (1.25, -0.5, 0.75))
        self.assertEqual(boundary.pressure_neumann_normal((0, 0, 2)), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(
            boundary.pressure_neumann_gradient_pa_per_m((0, 0, 2)),
            42.0,
        )

    def test_triangle_projection_interpolates_curved_surface_boundary_targets(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=3,
            projection_triangle_capacity=1,
        )
        markers.load_markers(
            positions_m=(
                (0.25, 0.25, 0.5),
                (0.75, 0.25, 0.5),
                (0.25, 0.75, 0.5),
            ),
            velocities_mps=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            normals=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(7,) * 3,
        )
        markers.set_projection_triangles(((0, 1, 2),))
        search = HibmMpmIbNodeSearch(
            grid_nodes=(8, 8, 8),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=3,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.08,
            interior_probe_distance_m=0.05,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(8, 8, 8),
            marker_capacity=3,
        )

        report = boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(100.0, 200.0, 300.0),
        )

        expected_weights = np.array((0.75, 0.125, 0.125), dtype=np.float64)
        expected_normal = expected_weights / np.linalg.norm(expected_weights)
        self.assertEqual(report.no_slip_dirichlet_count, report.pressure_neumann_count)
        self.assertTrue(boundary.is_active((2, 2, 4)))
        np.testing.assert_allclose(
            boundary.velocity_dirichlet_mps((2, 2, 4)),
            tuple(expected_weights),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            boundary.pressure_neumann_normal((2, 2, 4)),
            tuple(expected_normal),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(
            boundary.pressure_neumann_gradient_pa_per_m((2, 2, 4)),
            137.5,
            delta=1.0e-5,
        )

    def test_builds_boundary_targets_from_taichi_neumann_field(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 2.0, 0.0),),
            normals=((0.0, 1.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 4, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.26,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(1, 4, 1),
            marker_capacity=1,
        )
        boundary.marker_pressure_neumann_gradient_field[0] = -12.5

        report = boundary.build_from_search_device_fields(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
        )

        self.assertEqual(report.no_slip_dirichlet_count, 1)
        self.assertTrue(boundary.is_active((0, 2, 0)))
        self.assertEqual(boundary.velocity_dirichlet_mps((0, 2, 0)), (0.0, 2.0, 0.0))
        self.assertAlmostEqual(
            boundary.pressure_neumann_gradient_pa_per_m((0, 2, 0)),
            -12.5,
        )

    def test_fluid_predictor_normal_mismatch_updates_pressure_neumann_gradient_field(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(1, 1, 1),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 0.2))

        report = markers.update_pressure_neumann_gradient_from_fluid_predictor(
            boundary.marker_pressure_neumann_gradient_field,
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=1000.0,
            dt_s=1.0e-2,
            probe_distance_m=0.125,
        )

        self.assertEqual(report.active_marker_count, 1)
        self.assertAlmostEqual(report.max_abs_gradient_pa_per_m, 20000.0, delta=1.0e-3)
        self.assertAlmostEqual(
            float(boundary.marker_pressure_neumann_gradient_field[0]),
            20000.0,
            delta=1.0e-3,
        )

    def test_pressure_neumann_reconstruction_enters_fv_cg_matrix_row(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(25.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()
        fluid._prepare_fv_multigrid_rhs(rhs_scale=1.0)
        fluid._cg_build_positive_rhs_kernel(fluid._mg_rhs[0], fluid.cg_z, 0.0)

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertAlmostEqual(
            report.min_reconstruction_gap_m,
            0.25,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.max_reconstruction_gap_m,
            0.25,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.max_transmissibility_m,
            0.25,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.max_diagonal_per_m2,
            16.0,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(matrix_report["diagonal_integral"], 0.5, delta=1.0e-5)
        self.assertAlmostEqual(report.rhs_integral, 0.0, delta=1.0e-5)
        self.assertAlmostEqual(matrix_report["rhs_integral"], 0.0, delta=1.0e-5)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[2, 2, 2]), 1)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[2, 2, 2][axis])
                for axis in range(3)
            ),
            (2, 2, 3),
        )
        self.assertLess(fluid.cg_z[2, 2, 2], 0.0)
        self.assertGreater(fluid.cg_z[2, 2, 3], 0.0)

    def test_pressure_neumann_matrix_skips_velocity_dirichlet_row(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(25.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity_dirichlet_boundary_active[2, 2, 2] = 1

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()

        self.assertEqual(report.active_pressure_neumann_rows, 0)
        self.assertEqual(report.skipped_velocity_dirichlet_row_count, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[2, 2, 2]), 0)
        self.assertEqual(matrix_report["active_cells"], 0)
        self.assertAlmostEqual(matrix_report["diagonal_integral"], 0.0)
        self.assertAlmostEqual(matrix_report["rhs_integral"], 0.0)

    def test_pressure_neumann_matrix_counts_skipped_obstacle_owner_rows(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(25.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.obstacle[2, 2, 2] = 1

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_pressure_neumann_rows, 0)
        self.assertEqual(report.skipped_velocity_dirichlet_row_count, 0)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[2, 2, 2]), 0)

    def test_pressure_neumann_reconstruction_uses_fv_spacing_not_gap_squared(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        normal = (math.sqrt(0.99), 0.0, 0.1)
        node_position = (0.625, 0.625, 0.625)
        boundary_point = tuple(
            node_position[axis] - 0.01 * normal[axis] for axis in range(3)
        )
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = normal
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = boundary_point
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.875)

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()
        expected_normal_spacing_m = 0.25 / (abs(normal[0]) + abs(normal[2]))
        expected_coefficient = 1.0 / (expected_normal_spacing_m * 0.025)
        expected_transmissibility = expected_coefficient * (0.25**3)
        expected_diagonal_integral = 2.0 * expected_coefficient * (0.25**3)

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertAlmostEqual(
            float(fluid.pressure_interface_coupling_coefficient[node]),
            expected_transmissibility,
            delta=1.0e-4,
        )
        self.assertAlmostEqual(
            matrix_report["max_abs_diagonal"],
            expected_coefficient,
            delta=1.0e-4,
        )
        self.assertAlmostEqual(
            matrix_report["diagonal_integral"],
            expected_diagonal_integral,
            delta=1.0e-5,
        )

    def test_pressure_neumann_transmissibility_is_capped_and_reported(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        normal_z = 1.0e-2
        normal = (math.sqrt(1.0 - normal_z * normal_z), 0.0, normal_z)
        node_position = (0.625, 0.625, 0.625)
        boundary_point = tuple(
            node_position[axis] - 0.01 * normal[axis] for axis in range(3)
        )
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = normal
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = boundary_point
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.875)

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()
        expected_normal_spacing_m = 0.25 / (abs(normal[0]) + abs(normal[2]))
        expected_interface_area_m2 = (0.25**3) / expected_normal_spacing_m
        expected_cap_m = 20.0 * expected_interface_area_m2 / expected_normal_spacing_m
        expected_raw_m = expected_interface_area_m2 / (0.25 * abs(normal[2]))
        expected_diagonal = expected_cap_m / (0.25**3)

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.transmissibility_capped_row_count, 1)
        self.assertAlmostEqual(
            report.max_raw_transmissibility_m,
            expected_raw_m,
            delta=1.0e-2,
        )
        self.assertAlmostEqual(
            report.max_transmissibility_limit_m,
            expected_cap_m,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            report.max_transmissibility_m,
            expected_cap_m,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            float(fluid.pressure_interface_coupling_coefficient[node]),
            expected_cap_m,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            matrix_report["max_abs_diagonal"],
            expected_diagonal,
            delta=1.0e-4,
        )

    def test_pressure_neumann_falls_back_for_degenerate_normal_reconstruction_row(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.625),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        normal = (math.sqrt(1.0 - 1.0e-10), 0.0, 1.0e-5)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = normal
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.625, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.875)

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertGreater(report.min_reconstruction_gap_m, 0.0)
        self.assertGreater(report.max_reconstruction_gap_m, 0.0)
        self.assertGreater(report.max_transmissibility_m, 0.0)
        self.assertGreater(report.max_diagonal_per_m2, 0.0)
        self.assertEqual(matrix_report["active_cells"], 2)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 1)
        self.assertEqual(
            tuple(int(fluid.pressure_interface_coupling_neighbor[node][axis]) for axis in range(3)),
            (3, 2, 2),
        )

    def test_pressure_neumann_falls_back_for_node_outside_normal_reconstruction_segment(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 3)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.625, 0.625, 0.5)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.625)

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertGreater(report.min_reconstruction_gap_m, 0.0)
        self.assertEqual(matrix_report["active_cells"], 2)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 1)

    def test_pressure_neumann_rows_are_self_adjoint_on_nonuniform_fv_grid(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.1, 0.2, 0.35, 0.35),
            cell_widths_y_m=(0.15, 0.2, 0.3, 0.35),
            cell_widths_z_m=(0.18, 0.22, 0.27, 0.33),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.15, 0.5, 0.535),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=grid.grid_nodes,
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=grid.grid_nodes,
            marker_capacity=1,
        )
        node = (1, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.15, 0.5, 0.535)
        search.node_interior_fluid_point_m[node] = (0.475, 0.5, 0.535)

        boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        rng = np.random.default_rng(43)
        x = rng.normal(size=grid.grid_nodes).astype(np.float32)
        y = rng.normal(size=grid.grid_nodes).astype(np.float32)
        fluid.pressure_tmp.from_numpy(x)
        fluid.cg_d.from_numpy(y)

        fluid._fv_laplacian_apply_kernel(fluid.pressure_tmp, fluid.cg_r, 0)
        fluid._fv_laplacian_apply_kernel(fluid.cg_d, fluid.cg_Ad, 0)

        lhs = float(fluid._weighted_dot_kernel(fluid.cg_r, fluid.cg_d))
        rhs = float(fluid._weighted_dot_kernel(fluid.pressure_tmp, fluid.cg_Ad))
        relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)
        self.assertLess(relative_error, 1.0e-5)

    def test_no_slip_dirichlet_targets_assemble_into_fluid_boundary_rows(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.25, -0.5, 0.75),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(0.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = boundary.assemble_velocity_dirichlet_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.obstacle,
        )
        apply_report = fluid.apply_velocity_dirichlet_boundary_rows(read_report=True)

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(apply_report.active_cells, 1)
        self.assertEqual(fluid.velocity_dirichlet_boundary_active[2, 2, 2], 1)
        self.assertEqual(
            tuple(float(fluid.velocity[2, 2, 2][axis]) for axis in range(3)),
            (0.25, -0.5, 0.75),
        )
        self.assertEqual(fluid.velocity_constraint_weight[2, 2, 2], 0.0)

    def test_no_slip_dirichlet_rows_reconstruct_along_surface_normal(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.1),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(0.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 0.4))

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertAlmostEqual(report.min_projection_weight, 0.5, delta=1.0e-6)
        self.assertAlmostEqual(report.max_projection_weight, 0.5, delta=1.0e-6)
        self.assertEqual(fluid.velocity_dirichlet_boundary_active[2, 2, 2], 1)
        reconstructed_z = float(
            fluid.velocity_dirichlet_boundary_value_mps[2, 2, 2][2]
        )
        self.assertAlmostEqual(reconstructed_z, 0.25, delta=1.0e-6)
        self.assertAlmostEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[2, 2, 2]),
            0.5,
            delta=1.0e-6,
        )

    def test_no_slip_reconstruction_falls_back_to_nearest_marker_velocity(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.625),),
            velocities_mps=((0.1, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.velocity_dirichlet_mps_field[node] = (0.1, 0.0, 0.0)
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[node] = (0.625, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.625)

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 1)
        self.assertEqual(report.invalid_no_fluid_sample_row_count, 0)
        self.assertEqual(report.invalid_nonpositive_gap_row_count, 1)
        self.assertEqual(report.invalid_node_behind_boundary_row_count, 0)
        self.assertEqual(report.invalid_node_beyond_interior_row_count, 0)
        self.assertEqual(report.min_projection_weight, 0.0)
        self.assertEqual(report.max_projection_weight, 0.0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[node]), 1)
        np.testing.assert_allclose(
            tuple(float(fluid.velocity_dirichlet_boundary_value_mps[node][axis]) for axis in range(3)),
            (0.1, 0.0, 0.0),
            atol=1.0e-7,
        )
        self.assertEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[node]),
            0.0,
        )

    def test_velocity_dirichlet_row_relocates_to_first_fluid_cell_when_node_masked(
        self,
    ) -> None:
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.velocity_dirichlet_mps_field[node] = (0.2, 0.0, 0.0)
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[node] = (0.5, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.875, 0.625, 0.625)
        fluid.obstacle[2, 2, 2] = 1

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.relocated_row_count, 1)
        self.assertEqual(report.relocation_blocked_row_count, 0)
        self.assertEqual(report.inactive_obstacle_rows, 0)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[2, 2, 2]), 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[3, 2, 2]), 1)
        np.testing.assert_allclose(
            tuple(
                float(fluid.velocity_dirichlet_boundary_value_mps[3, 2, 2][axis])
                for axis in range(3)
            ),
            (0.08, 0.0, 0.0),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[3, 2, 2]),
            0.6,
            delta=1.0e-6,
        )

    def test_no_slip_reconstruction_classifies_wall_limited_gap_as_narrow_gap_row(
        self,
    ) -> None:
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.velocity_dirichlet_mps_field[node] = (0.3, 0.0, 0.0)
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[node] = (0.5, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.875, 0.625, 0.625)
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[3, :, :] = 1
        fluid.obstacle.from_numpy(obstacle)

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.narrow_gap_boundary_velocity_row_count, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_no_fluid_sample_row_count, 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[node]), 1)
        np.testing.assert_allclose(
            tuple(
                float(fluid.velocity_dirichlet_boundary_value_mps[node][axis])
                for axis in range(3)
            ),
            (0.3, 0.0, 0.0),
            atol=1.0e-7,
        )
        self.assertEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[node]),
            0.0,
        )

    def test_no_slip_reconstruction_reports_node_beyond_interior_segment(
        self,
    ) -> None:
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (3, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.velocity_dirichlet_mps_field[node] = (0.2, 0.0, 0.0)
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[node] = (0.5, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.625)

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 1)
        self.assertEqual(report.invalid_no_fluid_sample_row_count, 0)
        self.assertEqual(report.invalid_nonpositive_gap_row_count, 0)
        self.assertEqual(report.invalid_node_behind_boundary_row_count, 0)
        self.assertEqual(report.invalid_node_beyond_interior_row_count, 1)
        self.assertEqual(float(fluid.velocity_dirichlet_boundary_projection_weight[node]), 0.0)

    def test_project_consumes_hibm_no_slip_dirichlet_rows_without_legacy_constraints(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.1, 0.2, -0.3),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(0.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        boundary.assemble_velocity_dirichlet_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.obstacle,
        )

        fluid.project(
            iterations=2,
            pressure_solver="fv_jacobi",
            preserve_velocity_constraints=False,
            reset_pressure=True,
            read_report=False,
        )

        velocity = tuple(float(fluid.velocity[2, 2, 2][axis]) for axis in range(3))
        for actual, expected in zip(velocity, (0.1, 0.2, -0.3), strict=True):
            self.assertAlmostEqual(actual, expected, delta=1.0e-6)
        self.assertEqual(fluid.velocity_constraint_weight[2, 2, 2], 0.0)

    def test_pressure_neumann_reconstruction_activates_multiple_ib_rows(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.28,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(25.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = boundary.assemble_pressure_neumann_matrix_rows(
            fluid.pressure_interface_matrix_diagonal,
            fluid.pressure_interface_matrix_rhs,
            fluid.pressure_interface_coupling_active,
            fluid.pressure_interface_coupling_neighbor,
            fluid.pressure_interface_coupling_coefficient,
            fluid.obstacle,
            fluid.velocity_dirichlet_boundary_active,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            search,
            markers,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()

        self.assertEqual(report.active_pressure_neumann_rows, 4)
        self.assertEqual(report.active_pressure_neumann_marker_count, 1)
        self.assertEqual(report.max_pressure_neumann_rows_per_marker, 4)
        self.assertGreater(matrix_report["diagonal_integral"], 0.0)
        self.assertGreater(report.max_abs_rhs, 0.0)

    def test_rejects_pressure_neumann_marker_count_mismatch(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(1, 1, 1),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(1, 1, 1),
            marker_capacity=1,
        )

        with self.assertRaisesRegex(ValueError, "marker_pressure_neumann_gradient"):
            boundary.build_from_search(
                search,
                markers,
                marker_pressure_neumann_gradient_pa_per_m=(),
            )


class HibmMpmSharpAssemblyTests(unittest.TestCase):
    def test_sharp_fluid_to_mpm_load_assembly_uses_taichi_fields_not_legacy_constraints(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.25, -0.5, 0.75),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.marker_pressure_neumann_gradient_field[0] = 25.0
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.625, 0.625, 0.5)
        solid.external_force_n[0] = (99.0, 0.0, 0.0)

        report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_count=solid.particle_count,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=64,
            divergence_cleanup_iterations=1,
            divergence_cleanup_relaxation=0.25,
            pressure_solver="fv_cg",
        )

        self.assertEqual(report.ib_node_search.external_ib_node_count, 1)
        self.assertEqual(report.velocity_dirichlet.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.pressure_neumann.active_pressure_neumann_rows, 0)
        self.assertEqual(
            report.pressure_neumann.skipped_velocity_dirichlet_row_count,
            1,
        )
        self.assertEqual(report.no_slip_residual.valid_marker_count, 1)
        stale_row_residual = 0.5 * math.sqrt(0.25 * 0.25 + 0.5 * 0.5 + 0.75 * 0.75)
        self.assertLess(report.no_slip_residual.max_no_slip_residual_mps, stale_row_residual)
        self.assertEqual(report.fluid_stress.valid_marker_count, 0)
        self.assertEqual(report.fluid_stress.invalid_marker_count, 1)
        self.assertEqual(report.fluid_stress.two_sided_pressure_marker_count, 1)
        self.assertEqual(report.fluid_stress.viscous_gradient_invalid_marker_count, 1)
        self.assertEqual(report.marker_forces.primary_marker_force_n, (0.0, 0.0, 0.0))
        self.assertEqual(
            report.marker_forces.secondary_marker_force_n,
            report.marker_forces.total_marker_force_n,
        )
        self.assertEqual(report.mpm_external_force_clear.cleared_particle_count, 1)
        self.assertEqual(report.mpm_force_scatter.active_marker_count, 1)
        self.assertAlmostEqual(
            report.mpm_force_scatter.action_reaction_residual_n,
            0.0,
            delta=1.0e-5,
        )
        self.assertEqual(fluid.velocity_constraint_weight[2, 2, 2], 0.0)
        self.assertEqual(fluid.velocity_dirichlet_boundary_active[2, 2, 2], 1)
        row_velocity = tuple(float(fluid.velocity[2, 2, 2][axis]) for axis in range(3))
        row_target = tuple(
            float(fluid.velocity_dirichlet_boundary_value_mps[2, 2, 2][axis])
            for axis in range(3)
        )
        np.testing.assert_allclose(
            row_velocity,
            row_target,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_sharp_assembly_masks_internal_ib_nodes_for_no_slip_sampling(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 1.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)

        report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_count=solid.particle_count,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=64,
            run_fluid_predictor=False,
            pressure_solver="fv_cg",
        )

        self.assertEqual(search.node_kind((2, 2, 1)), "internal")
        self.assertEqual(search.node_kind((2, 2, 2)), "external_ib")
        self.assertEqual(int(fluid.obstacle[2, 2, 1]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 2]), 0)
        self.assertEqual(report.no_slip_residual.valid_marker_count, 1)
        self.assertLess(report.no_slip_residual.max_no_slip_residual_mps, 0.5)

    def test_hibm_internal_obstacle_mask_preserves_static_obstacles(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.obstacle[0, 0, 0] = 1
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
        )

        masked_count = fluid.apply_hibm_internal_obstacles(
            search.node_kind_code,
            internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        )
        self.assertGreater(masked_count, 0)
        self.assertEqual(int(fluid.obstacle[0, 0, 0]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 1]), 1)

        markers.load_markers(
            positions_m=((0.95, 0.95, 0.95),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search.search_and_classify_grid_fields(
            markers,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            search_radius_m=0.01,
            interior_probe_distance_m=0.125,
        )
        masked_count = fluid.apply_hibm_internal_obstacles(
            search.node_kind_code,
            internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        )

        self.assertEqual(masked_count, 0)
        self.assertEqual(int(fluid.obstacle[0, 0, 0]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 1]), 0)

    def test_sharp_fluid_solve_runs_predictor_before_projection(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.25, -0.5, 0.75),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        calls: list[str] = []
        velocity_row_apply_contexts: list[str] = []
        original_apply_velocity_rows = fluid.apply_velocity_dirichlet_boundary_rows
        inside_project = False

        def counted_apply_velocity_rows(*args, **kwargs):
            velocity_row_apply_contexts.append(
                "project" if inside_project else "outside_project"
            )
            return original_apply_velocity_rows(*args, **kwargs)

        fluid.apply_velocity_dirichlet_boundary_rows = counted_apply_velocity_rows

        def fake_predict(dt_s=None, *, advection_scheme="euler") -> None:
            substep = sum(1 for call in calls if call.startswith("predict"))
            self.assertEqual(fluid.velocity_dirichlet_boundary_active[2, 2, 2], 1)
            self.assertEqual(dt_s, 2.5e-4)
            self.assertEqual(advection_scheme, "euler")
            calls.append(f"predict{substep}")

        def fake_project(**kwargs):
            substep = sum(1 for call in calls if call.startswith("project"))
            if substep < 2:
                self.assertEqual(
                    calls,
                    [
                        item
                        for i in range(substep)
                        for item in (f"predict{i}", f"project{i}")
                    ]
                    + [f"predict{substep}"],
                )
            else:
                self.assertEqual(
                    calls,
                    ["predict0", "project0", "predict1", "project1"],
                )
            self.assertEqual(kwargs["dt_s"], 2.5e-4)
            calls.append(f"project{substep}")
            nonlocal inside_project
            inside_project = True
            try:
                fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
            finally:
                inside_project = False
            return {
                "projection_l2": 0.0,
                "cg_project_calls": 1,
                "cg_iterations_total": 3,
                "cg_iterations_max": 3,
                "cg_host_residual_checks": 2,
                "cg_mean_host_reads": 1,
                "cg_mean_projection_count": 4 + substep,
                "cg_initial_relative_residual_max": 0.25,
                "cg_relative_residual_max": 0.01,
                "cg_converged_all": True,
                "cg_breakdown_count": 0,
                "cg_breakdown": "",
            }

        fluid.predict = fake_predict
        fluid.project = fake_project

        report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_count=solid.particle_count,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            dt_s=5.0e-4,
            fluid_substeps=2,
            projection_iterations=8,
            pressure_solver="fv_cg",
        )

        self.assertEqual(
            calls,
            ["predict0", "project0", "predict1", "project1", "project2"],
        )
        self.assertEqual(velocity_row_apply_contexts, ["project", "project", "project"])
        self.assertTrue(report.fluid_predictor_applied)
        self.assertEqual(report.fluid_projection["fluid_substeps"], 2)
        self.assertEqual(report.fluid_projection["cg_project_calls"], 3)
        self.assertEqual(report.fluid_projection["cg_iterations_total"], 9)
        self.assertEqual(report.fluid_projection["cg_mean_projection_count"], 15)

    def test_sharp_skipped_neumann_rows_do_not_force_fv_cg_projection_solver(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.marker_pressure_neumann_gradient_field[0] = 25.0
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        captured_project_kwargs: list[dict[str, object]] = []

        def fake_project(**kwargs):
            captured_project_kwargs.append(dict(kwargs))
            failure_policy = str(kwargs["pressure_solve_failure_policy"])
            return {
                "projection_l2": 0.0,
                "pressure_solve_failure_policy": failure_policy,
                "pressure_solve_failed": False,
                "pressure_solve_failure_action": "none",
                "cg_project_calls": 1,
                "cg_iterations_total": 3,
                "cg_iterations_max": 3,
                "cg_host_residual_checks": 1,
                "cg_mean_host_reads": 1,
                "cg_initial_relative_residual_max": 1.0,
                "cg_relative_residual_max": 1.0e-7,
                "cg_converged_all": True,
                "cg_breakdown_count": 0,
                "cg_breakdown": "",
            }

        fluid.project = fake_project

        report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_count=solid.particle_count,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.14,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=8,
            run_fluid_predictor=False,
            pressure_solver="fv_multigrid",
            pressure_solve_failure_policy="report",
        )

        self.assertEqual(report.pressure_neumann.active_pressure_neumann_rows, 0)
        self.assertEqual(
            report.pressure_neumann.skipped_velocity_dirichlet_row_count,
            1,
        )
        self.assertEqual(
            [str(kwargs["pressure_solver"]) for kwargs in captured_project_kwargs],
            ["fv_multigrid", "fv_multigrid"],
        )
        self.assertEqual(
            [
                str(kwargs["pressure_solve_failure_policy"])
                for kwargs in captured_project_kwargs
            ],
            ["report", "report"],
        )
        self.assertEqual(report.fluid_projection["pressure_solver_requested"], "fv_multigrid")
        self.assertEqual(report.fluid_projection["pressure_solver"], "fv_multigrid")
        self.assertEqual(report.fluid_projection["pressure_solve_failure_policy"], "report")
        self.assertFalse(report.fluid_projection["pressure_solver_forced_to_fv_cg"])
        self.assertEqual(
            report.fluid_projection["pressure_solver_force_reason"],
            "",
        )

    def test_sharp_assembly_search_uses_nonuniform_fluid_cell_centers(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_y_m=(0.25, 0.25, 0.25, 0.25),
            cell_widths_z_m=(0.1, 0.1, 0.2, 0.6),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(1.0, 1.0, 1.0),
                grid_nodes=(4, 4, 4),
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.28),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=fluid.grid.grid_nodes,
            marker_capacity=1,
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.625, 0.625, 0.28)

        report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_count=solid.particle_count,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.04,
            interior_probe_distance_m=0.05,
            mpm_support_radius_m=0.2,
            projection_iterations=8,
            run_fluid_predictor=False,
            pressure_solver="fv_cg",
        )

        self.assertEqual(report.ib_node_search.near_boundary_node_count, 1)
        self.assertEqual(report.ib_node_search.external_ib_node_count, 1)
        self.assertEqual(report.velocity_dirichlet.active_velocity_dirichlet_rows, 1)

    def test_sharp_neo_hookean_step_advances_mpm_and_rebuilds_next_surface_boundary(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(8, 8, 8),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(8, 8, 8),
            marker_capacity=1,
        )
        boundary.marker_pressure_neumann_gradient_field[0] = 2500.0
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(8, 8, 8), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.pressure.fill(7.0)
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(8, 8, 8),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.625, 0.625, 0.5)
        solid.v[0] = (0.1, 0.0, 0.0)
        solid.surface_normal[0] = (1.0, 0.0, 0.0)
        solid.area_weight_m2[0] = 0.06

        report = advance_hibm_mpm_sharp_neo_hookean_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            solid=solid,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            fluid_dt_s=1.0e-3,
            solid_dt_s=1.0e-3,
            projection_iterations=64,
            pressure_solver="fv_cg",
            mu_pa=0.0,
            lambda_pa=0.0,
            primary_region_id=101,
            secondary_region_id=202,
        )

        self.assertEqual(
            report.fluid_to_mpm_loads.mpm_force_scatter.active_particle_count,
            1,
        )
        self.assertAlmostEqual(
            report.fluid_to_mpm_loads.mpm_force_scatter.action_reaction_residual_n,
            0.0,
            delta=1.0e-7,
        )
        self.assertIsNotNone(report.mpm)
        self.assertEqual(report.surface_feedback.updated_marker_count, 1)
        self.assertGreater(report.next_ib_node_search.external_ib_node_count, 0)
        self.assertGreater(report.next_velocity_dirichlet.active_velocity_dirichlet_rows, 0)
        self.assertGreater(
            report.next_pressure_neumann.active_pressure_neumann_rows
            + report.next_pressure_neumann.skipped_velocity_dirichlet_row_count,
            0,
        )
        self.assertEqual(report.surface_feedback.geometry_updated_marker_count, 1)
        self.assertEqual(markers.marker_normal(0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(float(markers.A_gamma_m2[0]), 0.06, delta=1.0e-6)
        marker_speed = math.sqrt(
            sum(float(markers.v_gamma_mps[0][axis]) ** 2 for axis in range(3))
        )
        self.assertGreater(marker_speed, 0.0)

    def test_generic_sharp_mpm_step_uses_explicit_taichi_surface_fields(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=1,
        )
        boundary.marker_pressure_neumann_gradient_field[0] = 2500.0
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.surface_normal[0] = (0.0, 1.0, 0.0)
        solid.area_weight_m2[0] = 0.06

        report = advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=lambda: solid.step(
                dt_s=1.0e-3,
                mu_pa=0.0,
                lambda_pa=0.0,
                primary_region_id=101,
                secondary_region_id=202,
                velocity_damping=1.0,
                read_report=True,
            ),
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            fluid_dt_s=1.0e-3,
            projection_iterations=16,
            pressure_solver="fv_cg",
        )

        self.assertIsNotNone(report.mpm)
        self.assertEqual(report.surface_feedback.updated_marker_count, 1)
        self.assertEqual(report.surface_feedback.geometry_updated_marker_count, 1)
        self.assertGreater(report.next_ib_node_search.external_ib_node_count, 0)
        self.assertEqual(markers.marker_normal(0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(float(markers.A_gamma_m2[0]), 0.06, delta=1.0e-6)

    def test_generic_sharp_mpm_step_accepts_tri_mooney_shell_surface_fields(
        self,
    ) -> None:
        mesh = SurfaceMesh(
            vertices=np.array(
                [
                    [0.45, 0.45, 0.5],
                    [0.55, 0.45, 0.5],
                    [0.45, 0.55, 0.5],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        solid = TriMooneyShellMpmState(
            mesh,
            thickness_m=0.01,
            density_kgm3=1.0,
            c1_pa=20.0,
            c2_pa=10.0,
            face_region_id=np.array([202], dtype=np.int32),
            primary_region_id=101,
            secondary_region_id=202,
            grid_nodes=(8, 8, 8),
            bounds_padding_fraction=2.0,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers = HibmMpmSurfaceMarkers(marker_capacity=solid.particle_count)
        markers.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.vertex_region_id,
            marker_count=solid.particle_count,
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=solid.particle_count,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=solid.particle_count,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        report = advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=markers,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=lambda: solid.step(
                dt_s=0.0,
                pressure_pa=0.0,
                velocity_damping=1.0,
                read_report=True,
            ),
            marker_pressure_neumann_gradient_pa_per_m_field=(
                boundary.marker_pressure_neumann_gradient_field
            ),
            search_radius_m=0.2,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.12,
            fluid_dt_s=1.0e-3,
            projection_iterations=8,
            pressure_solver="fv_cg",
        )

        self.assertEqual(report.surface_feedback.updated_marker_count, 3)
        self.assertEqual(report.surface_feedback.geometry_updated_marker_count, 3)
        self.assertGreater(report.next_ib_node_search.external_ib_node_count, 0)
        self.assertGreater(report.next_velocity_dirichlet.active_velocity_dirichlet_rows, 0)
        self.assertEqual(markers.marker_region_id(0), 202)
        self.assertEqual(markers.marker_normal(0), (0.0, 0.0, 1.0))

    def test_sharp_coupling_state_owns_marker_search_and_boundary_fields(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 202
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=solid.particle_count,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        coupling.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            marker_count=solid.particle_count,
        )
        coupling.marker_pressure_neumann_gradient_pa_per_m[0] = 2500.0

        report = coupling.advance_mpm_step(
            fluid=fluid,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=lambda: solid.step(
                dt_s=1.0e-4,
                mu_pa=10.0,
                lambda_pa=10.0,
                primary_region_id=101,
                secondary_region_id=202,
                velocity_damping=1.0,
                read_report=True,
            ),
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=64,
            pressure_solver="fv_cg",
        )

        self.assertEqual(report.surface_feedback.updated_marker_count, 1)
        self.assertGreater(report.next_ib_node_search.external_ib_node_count, 0)
        self.assertEqual(coupling.markers.marker_region_id(0), 202)

    def test_sharp_coupling_state_advances_neo_hookean_without_case_owned_fields(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 202
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=solid.particle_count,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        coupling.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            marker_count=solid.particle_count,
            surface_velocity_mps=solid.v,
        )

        report = coupling.advance_neo_hookean_step(
            fluid=fluid,
            solid=solid,
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            solid_dt_s=1.0e-4,
            mu_pa=10.0,
            lambda_pa=10.0,
            primary_region_id=101,
            secondary_region_id=202,
            fluid_dt_s=1.0e-3,
            projection_iterations=8,
            pressure_solver="fv_cg",
            solid_external_loads=lambda: solid.add_region_normal_pressure(
                region_id=202,
                pressure_pa=25.0,
            ),
        )

        self.assertIsInstance(report, HibmMpmSharpNeoHookeanStepReport)
        self.assertEqual(report.surface_feedback.updated_marker_count, 1)
        self.assertGreater(report.next_ib_node_search.external_ib_node_count, 0)
        self.assertLess(report.mpm.external_force_n[2], 0.0)
        self.assertEqual(coupling.markers.marker_region_id(0), 202)

    def test_sharp_step_summary_flattens_core_diagnostics_for_case_reports(
        self,
    ) -> None:
        from simulation_core import hibm_mpm_sharp_step_summary

        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 202
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=solid.particle_count,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        coupling.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            marker_count=solid.particle_count,
        )

        report = coupling.advance_mpm_step(
            fluid=fluid,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=lambda: solid.step(
                dt_s=0.0,
                mu_pa=0.0,
                lambda_pa=0.0,
                primary_region_id=101,
                secondary_region_id=202,
                velocity_damping=1.0,
                read_report=True,
            ),
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=8,
            pressure_solver="fv_cg",
        )

        summary = hibm_mpm_sharp_step_summary(report)

        expected_keys = {
            "hibm_ib_node_count",
            "hibm_ib_invalid_projection_count",
            "hibm_internal_obstacle_cell_count",
            "hibm_no_slip_residual_max_mps",
            "hibm_no_slip_residual_l2_mps",
            "hibm_velocity_dirichlet_invalid_reconstruction_count",
            "hibm_velocity_dirichlet_invalid_no_fluid_sample_count",
            "hibm_velocity_dirichlet_invalid_nonpositive_gap_count",
            "hibm_velocity_dirichlet_invalid_node_behind_boundary_count",
            "hibm_velocity_dirichlet_invalid_node_beyond_interior_count",
            "hibm_velocity_dirichlet_min_projection_weight",
            "hibm_velocity_dirichlet_max_projection_weight",
            "hibm_pressure_neumann_active_rows",
            "hibm_pressure_neumann_invalid_reconstruction_count",
            "hibm_pressure_neumann_min_reconstruction_gap_m",
            "hibm_pressure_neumann_max_reconstruction_gap_m",
            "hibm_pressure_neumann_max_transmissibility_m",
            "hibm_pressure_neumann_max_raw_transmissibility_m",
            "hibm_pressure_neumann_max_transmissibility_limit_m",
            "hibm_pressure_neumann_transmissibility_capped_row_count",
            "hibm_pressure_neumann_max_diagonal_per_m2",
            "hibm_pressure_neumann_active_marker_count",
            "hibm_pressure_neumann_max_rows_per_marker",
            "hibm_full_stress_viscous_gradient_invalid_marker_count",
            "hibm_marker_primary_count",
            "hibm_marker_secondary_count",
            "hibm_marker_total_count",
            "hibm_marker_total_force_n",
            "hibm_marker_action_reaction_residual_n",
            "hibm_mpm_scatter_action_reaction_residual_n",
            "hibm_surface_updated_marker_count",
            "hibm_next_ib_node_count",
            "hibm_next_internal_obstacle_cell_count",
            "hibm_next_boundary_no_slip_count",
            "hibm_next_velocity_dirichlet_invalid_reconstruction_count",
            "hibm_next_pressure_neumann_invalid_reconstruction_count",
            "hibm_coupling_scheme",
            "hibm_added_mass_stability_status",
            "hibm_added_mass_stability_measured",
            "hibm_added_mass_stabilization",
            "hibm_semi_implicit_coupling_enabled",
            "hibm_semi_implicit_coupling_matrix_active",
            "hibm_pressure_correctable_divergence_l2",
            "hibm_pressure_correctable_divergence_max_abs",
            "hibm_pressure_correctable_divergence_cell_count",
            "hibm_pressure_fixed_divergence_l2",
            "hibm_pressure_fixed_divergence_max_abs",
            "hibm_pressure_fixed_divergence_cell_count",
            "hibm_interior_pressure_correctable_divergence_l2",
            "hibm_interior_pressure_correctable_divergence_max_abs",
            "hibm_interior_pressure_correctable_divergence_cell_count",
            "hibm_interior_pressure_fixed_divergence_l2",
            "hibm_interior_pressure_fixed_divergence_max_abs",
            "hibm_interior_pressure_fixed_divergence_cell_count",
        }
        self.assertTrue(expected_keys <= set(summary))
        self.assertEqual(summary["hibm_coupling_scheme"], "explicit_loose")
        self.assertEqual(summary["hibm_added_mass_stability_status"], "unmeasured")
        self.assertFalse(summary["hibm_added_mass_stability_measured"])
        self.assertEqual(summary["hibm_added_mass_stabilization"], "none")
        self.assertFalse(summary["hibm_semi_implicit_coupling_enabled"])
        self.assertFalse(summary["hibm_semi_implicit_coupling_matrix_active"])
        self.assertEqual(
            summary["hibm_ib_node_count"],
            report.fluid_to_mpm_loads.ib_node_search.near_boundary_node_count,
        )
        self.assertEqual(
            summary["hibm_no_slip_residual_max_mps"],
            report.fluid_to_mpm_loads.no_slip_residual.max_no_slip_residual_mps,
        )
        self.assertEqual(
            summary["hibm_marker_total_force_n"],
            report.fluid_to_mpm_loads.marker_forces.total_marker_force_n,
        )
        self.assertEqual(
            summary["hibm_marker_secondary_count"],
            report.fluid_to_mpm_loads.marker_forces.secondary_marker_count,
        )
        self.assertEqual(
            summary["hibm_full_stress_viscous_gradient_invalid_marker_count"],
            report.fluid_to_mpm_loads.fluid_stress.viscous_gradient_invalid_marker_count,
        )
        projection = report.fluid_to_mpm_loads.fluid_projection
        self.assertEqual(
            summary["hibm_pressure_correctable_divergence_l2"],
            projection["pressure_correctable_l2"],
        )
        self.assertEqual(
            summary["hibm_pressure_correctable_divergence_cell_count"],
            projection["pressure_correctable_cell_count"],
        )
        self.assertEqual(
            summary["hibm_pressure_fixed_divergence_l2"],
            projection["pressure_fixed_l2"],
        )
        self.assertEqual(
            summary["hibm_pressure_fixed_divergence_cell_count"],
            projection["pressure_fixed_cell_count"],
        )
        self.assertEqual(
            summary["hibm_interior_pressure_correctable_divergence_l2"],
            projection["interior_pressure_correctable_l2"],
        )
        self.assertEqual(
            summary["hibm_interior_pressure_correctable_divergence_cell_count"],
            projection["interior_pressure_correctable_cell_count"],
        )
        self.assertEqual(
            summary["hibm_interior_pressure_fixed_divergence_l2"],
            projection["interior_pressure_fixed_l2"],
        )
        self.assertEqual(
            summary["hibm_interior_pressure_fixed_divergence_cell_count"],
            projection["interior_pressure_fixed_cell_count"],
        )

    def test_sharp_coupling_state_loads_marker_velocity_from_surface_field(
        self,
    ) -> None:
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.v[0] = (0.125, -0.25, 0.5)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 202
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=solid.particle_count,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        coupling.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            surface_velocity_mps=solid.v,
            marker_count=solid.particle_count,
        )

        self.assertEqual(
            coupling.markers.marker_velocity_mps(0),
            (0.125, -0.25, 0.5),
        )

    def test_sharp_coupling_state_updates_neumann_gradient_from_fluid_predictor(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 0.2))
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(1, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1.0,
        )
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.v[0] = (0.0, 0.0, 0.0)
        solid.surface_normal[0] = (0.0, 0.0, 1.0)
        solid.area_weight_m2[0] = 0.04
        solid.region_id[0] = 202
        coupling = HibmMpmSharpCouplingState(
            grid_nodes=fluid.grid.grid_nodes,
            bounds_min_m=fluid.grid.bounds_min_m,
            bounds_max_m=fluid.grid.bounds_max_m,
            marker_capacity=solid.particle_count,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        coupling.load_markers_from_surface_fields(
            solid.x,
            solid.surface_normal,
            solid.area_weight_m2,
            solid.region_id,
            marker_count=solid.particle_count,
        )

        def move_surface_velocity() -> dict[str, bool]:
            return {"velocity_updated": True}

        report = coupling.advance_mpm_step(
            fluid=fluid,
            mpm_external_force_n=solid.external_force_n,
            mpm_particle_position_m=solid.x,
            mpm_particle_velocity_mps=solid.v,
            mpm_particle_normal=solid.surface_normal,
            mpm_particle_area_m2=solid.area_weight_m2,
            mpm_particle_count=solid.particle_count,
            solid_step=move_surface_velocity,
            search_radius_m=0.24,
            interior_probe_distance_m=0.125,
            mpm_support_radius_m=0.5,
            primary_region_id=101,
            secondary_region_id=202,
            projection_iterations=64,
            run_fluid_predictor=False,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-4,
            pressure_neumann_density_kgm3=1000.0,
            pressure_neumann_dt_s=1.0e-2,
        )

        gradient = report.fluid_to_mpm_loads.pressure_neumann_gradient
        self.assertIsNotNone(gradient)
        self.assertEqual(
            gradient.active_marker_count,
            report.fluid_to_mpm_loads.pressure_neumann.active_pressure_neumann_rows
            + report.fluid_to_mpm_loads.pressure_neumann.skipped_velocity_dirichlet_row_count,
        )
        self.assertEqual(report.fluid_to_mpm_loads.pressure_neumann.active_pressure_neumann_rows, 0)
        self.assertGreater(gradient.active_marker_count, 1)
        self.assertAlmostEqual(
            gradient.max_abs_gradient_pa_per_m,
            20000.0,
            delta=1.0e-2,
        )
        self.assertAlmostEqual(
            float(coupling.marker_pressure_neumann_gradient_pa_per_m[0]),
            20000.0,
            delta=1.0e-3,
        )
        self.assertIsNotNone(report.next_pressure_neumann_gradient)
        self.assertEqual(
            report.next_pressure_neumann_gradient.active_marker_count,
            report.next_pressure_neumann.active_pressure_neumann_rows
            + report.next_pressure_neumann.skipped_velocity_dirichlet_row_count,
        )
        self.assertEqual(report.next_pressure_neumann.active_pressure_neumann_rows, 0)
        self.assertGreater(report.next_pressure_neumann_gradient.active_marker_count, 1)
        self.assertGreater(
            report.next_pressure_neumann_gradient.max_abs_gradient_pa_per_m,
            0.0,
        )


class HibmMpmSharpPathFailFastTests(unittest.TestCase):
    def test_unimplemented_hibm_solver_primitives_fail_fast_until_taichi_resident(self) -> None:
        for primitive in (
            classify_hibm_near_boundary_nodes,
            build_hibm_ib_node_boundary_conditions,
            compute_hibm_surface_tractions,
        ):
            with self.subTest(primitive=primitive.__name__):
                with self.assertRaisesRegex(NotImplementedError, "Taichi-resident"):
                    primitive()


class HibmMpmFarSidePressureClosureTests(unittest.TestCase):
    """S2-A red tests: known far-side pressure closure for water-air markers.

    The squid main membrane has water on its inside (-n) and pressurized air
    on its outside (+n); the air side is outside the fluid domain by
    construction, so the strict both-sides rule can never validate those
    markers at any grid resolution. The closure must be explicit opt-in per
    region: assuming p_far on an unresolved thin WATER gap would inject
    O(waveform) spurious force.
    """

    def _water_below_air_above_fixture(self):
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((8, 8, 8), 2.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[:, :, 4:] = 1
        fluid.obstacle.from_numpy(obstacle)
        return markers, fluid

    def _sample(self, markers, fluid, **far_pressure_kwargs):
        return markers.sample_fluid_stress_to_marker_tractions(
            fluid.velocity,
            fluid.pressure,
            fluid.obstacle,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            fluid.grid.grid_nodes,
            viscosity_pa_s=0.0,
            two_sided_pressure=True,
            **far_pressure_kwargs,
        )

    def test_far_pressure_closes_water_air_marker_with_known_outside_pressure(self) -> None:
        markers, fluid = self._water_below_air_above_fixture()

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # traction = (p_inside_water - p_far_air) * n = (2 - 10) * +z
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_far_pressure_closure_is_opt_in_per_region_not_automatic(self) -> None:
        markers, fluid = self._water_below_air_above_fixture()

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=-1,
            far_pressure_pa=0.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)

    def test_far_pressure_does_not_relax_other_regions(self) -> None:
        markers, fluid = self._water_below_air_above_fixture()

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=8,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)

    def test_far_pressure_closure_is_orientation_agnostic(self) -> None:
        # Mirrored fixture: water ABOVE the marker plane, dry side BELOW.
        # The marker normal still points +z (into the water), so the dry side
        # is now the INSIDE (-n) walk. The covariant two-sided formula
        # (p_inside - p_outside) * n must close with p_inside := p_far and
        # produce traction (p_far - p_water) * n, independent of which side
        # of the CAD winding the fluid happens to be on.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((8, 8, 8), 2.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[:, :, :4] = 1
        fluid.obstacle.from_numpy(obstacle)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # traction = (p_far_air - p_outside_water) * n = (10 - 2) * +z
        self.assertAlmostEqual(traction[2], 8.0, delta=1.0e-4)


if __name__ == "__main__":
    unittest.main()

