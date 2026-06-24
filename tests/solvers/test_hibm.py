from __future__ import annotations

import math
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import taichi as ti

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
    hibm_mpm_sharp_step_summary,
)
from simulation_core.hibm import (
    build_hibm_ib_node_boundary_conditions,
    classify_hibm_near_boundary_nodes,
    compute_hibm_surface_tractions,
)
from simulation_core.hibm_mpm import (
    HibmMpmExternalForceClearReport,
    HibmMpmMpmForceScatterReport,
    HibmMpmPressureNeumannMatrixReport,
    hibm_mpm_external_force_fresh_for_solid_step,
)
from simulation_core.pressure_interface import PRESSURE_INTERFACE_COUPLING_SLOT_COUNT


HIBM_MPM_CORE_SOURCE = Path("simulation_core/coupling/hibm_mpm/core.py")


class HibmMpmSurfaceMarkerTests(unittest.TestCase):
    def test_pressure_sampling_accumulates_pressure_in_f64(self) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count("value = ti.cast(0.0, ti.f64)"), 2)
        self.assertGreaterEqual(source.count("fluid_weight = ti.cast(0.0, ti.f64)"), 2)
        self.assertIn("pressure = ti.cast(0.0, ti.f64)", source)
        self.assertIn("outside_pressure = ti.cast(0.0, ti.f64)", source)

    def test_marker_pressure_traction_fields_are_f64(self) -> None:
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=4,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

        self.assertEqual(markers.t_gamma_pa.dtype, ti.f64)
        self.assertEqual(markers.F_gamma_n.dtype, ti.f64)
        self.assertEqual(markers.report_stress_max_abs_traction_pa.dtype, ti.f64)
        self.assertEqual(markers.report_mpm_scatter_marker_force_n.dtype, ti.f64)
        self.assertEqual(markers.report_mpm_scatter_external_force_n.dtype, ti.f64)

    def test_pressure_neumann_rows_force_fv_multigrid_to_fv_cg(self) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            'effective_pressure_solver in {"jacobi", "compact_jacobi", "fv_multigrid"}',
            source,
        )

    def test_hibm_sharp_steps_warm_start_pressure_by_default(self) -> None:
        mpm_default = inspect.signature(
            advance_hibm_mpm_sharp_mpm_step
        ).parameters["reset_pressure"].default
        neo_hookean_default = inspect.signature(
            advance_hibm_mpm_sharp_neo_hookean_step
        ).parameters["reset_pressure"].default

        self.assertIs(mpm_default, False)
        self.assertIs(neo_hookean_default, False)

    def test_sharp_mpm_step_can_preserve_explicit_marker_geometry(self) -> None:
        signature = inspect.signature(advance_hibm_mpm_sharp_mpm_step)
        self.assertIn("update_surface_geometry_from_mpm", signature.parameters)
        self.assertIs(
            signature.parameters["update_surface_geometry_from_mpm"].default,
            True,
        )

        source = inspect.getsource(advance_hibm_mpm_sharp_mpm_step)
        self.assertIn("update_surface_feedback_from_mpm_surface_particles", source)
        self.assertIn("update_surface_feedback_from_mpm_particles", source)

    def test_hibm_sharp_internal_obstacle_conversion_is_on_by_default(self) -> None:
        defaulted_symbols = (
            assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
            advance_hibm_mpm_sharp_mpm_step,
            advance_hibm_mpm_sharp_neo_hookean_step,
            HibmMpmSharpCouplingState.advance_mpm_step,
            HibmMpmSharpCouplingState.advance_neo_hookean_step,
        )

        for symbol in defaulted_symbols:
            with self.subTest(symbol=symbol.__name__):
                default_value = inspect.signature(symbol).parameters[
                    "convert_internal_nodes_to_obstacles"
                ].default
                self.assertIs(default_value, True)
                sign_default = inspect.signature(symbol).parameters[
                    "far_pressure_air_backed_probe_normal_sign"
                ].default
                self.assertEqual(sign_default, 0.0)

    def test_internal_obstacle_stress_sampling_keeps_plain_two_sided_walk_real_fluid_only(
        self,
    ) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")
        branch_start = source.index("elif bool(convert_internal_nodes_to_obstacles):")
        branch_end = source.index(
            '_debug_stage_progress("sample_fluid_stress_to_marker_tractions:start")',
            branch_start,
        )
        branch = source[branch_start:branch_end]

        self.assertNotIn("fluid.fill_hibm_converted_cell_pressures()", branch)
        self.assertNotIn("fluid.build_hibm_sampling_obstacle(", branch)
        self.assertNotIn("stress_sampling_obstacle_field = fluid.sampling_obstacle", branch)
        self.assertNotIn(
            "stress_sampling_obstacle_field = fluid.hibm_base_obstacle",
            branch,
        )

    def test_far_pressure_stress_sampling_keeps_dedicated_sampling_view(
        self,
    ) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")
        branch_start = source.index("if int(far_pressure_region_id) != -1:")
        branch_end = source.index(
            "elif bool(convert_internal_nodes_to_obstacles):",
            branch_start,
        )
        branch = source[branch_start:branch_end]

        self.assertIn("fluid.fill_hibm_converted_cell_pressures()", branch)
        self.assertIn("fluid.build_hibm_sampling_obstacle(", branch)
        self.assertIn("stress_sampling_obstacle_field = fluid.sampling_obstacle", branch)

    def test_two_sided_viscous_stress_sampling_uses_split_pass(self) -> None:
        source = inspect.getsource(
            HibmMpmSurfaceMarkers.sample_fluid_stress_to_marker_tractions
        )

        self.assertIn("split_viscous_path", source)
        pressure_call = source.index("_sample_fluid_stress_to_marker_tractions_kernel")
        viscous_call = source.index("_add_split_viscous_mode_marker_tractions_kernel")
        self.assertLess(pressure_call, viscous_call)

    def test_two_sided_sampling_view_walks_through_own_hibm_wall_band(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        base = np.zeros((4, 4, 16), dtype=np.int32)
        fluid.obstacle.from_numpy(base)
        fluid.snapshot_hibm_base_obstacle()
        obstacle = base.copy()
        obstacle[:, :, 6:11] = 1
        fluid.obstacle.from_numpy(obstacle)
        pressure = np.zeros((4, 4, 16), dtype=np.float32)
        pressure[:, :, :6] = 5.0
        pressure[:, :, 11:] = 1.0
        fluid.pressure.from_numpy(pressure)
        node_kind_code = ti.field(dtype=ti.i32, shape=(4, 4, 16))
        node_kind = np.zeros((4, 4, 16), dtype=np.int32)
        node_kind[:, :, 6:11] = 1
        node_kind_code.from_numpy(node_kind)
        fluid.build_hibm_sampling_obstacle(
            node_kind_code,
            unclassified_node_code=0,
        )
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=1,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        markers.load_markers(
            positions_m=((0.5, 0.5, 11.0 / 16.0),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(0,),
        )

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
            viscosity_pa_s=1.0e-3,
            two_sided_pressure=True,
            two_sided_probe_max_multiplier=8.0,
            sampling_obstacle_field=fluid.sampling_obstacle,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.two_sided_pressure_marker_count, 1)
        self.assertEqual(report.two_sided_extended_marker_count, 1)
        self.assertAlmostEqual(float(markers.t_gamma_pa[0].z), 4.0, delta=1.0e-5)

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

    def test_marker_aggregation_reports_region_scoped_stress_validity(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=4)
        markers.load_markers(
            positions_m=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 4,
            normals=((1.0, 0.0, 0.0),) * 4,
            areas_m2=(1.0, 1.0, 1.0, 1.0),
            region_ids=(101, 101, 202, 303),
        )
        markers.set_marker_tractions_pa(
            (
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            )
        )
        markers._stress_pressure_valid.from_numpy(
            np.array((1, 0, 1, 0), dtype=np.int32)
        )
        markers.compute_marker_forces()

        report = markers.aggregate_region_forces(
            primary_region_id=101,
            secondary_region_id=202,
        )

        self.assertEqual(report.primary_marker_count, 2)
        self.assertEqual(report.secondary_marker_count, 1)
        self.assertEqual(report.total_marker_count, 4)
        self.assertEqual(report.primary_stress_valid_marker_count, 1)
        self.assertEqual(report.primary_stress_invalid_marker_count, 1)
        self.assertEqual(report.secondary_stress_valid_marker_count, 1)
        self.assertEqual(report.secondary_stress_invalid_marker_count, 0)

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
            primary_region_id=202,
            secondary_region_id=303,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.primary_region_valid_marker_count, 1)
        self.assertEqual(report.primary_region_invalid_marker_count, 0)
        self.assertEqual(report.secondary_region_valid_marker_count, 0)
        self.assertEqual(report.other_region_valid_marker_count, 0)
        self.assertEqual(report.direct_sample_marker_count, 1)
        self.assertEqual(report.normal_walk_sample_marker_count, 0)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
        self.assertEqual(report.no_fluid_sample_marker_count, 0)
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

    def test_no_slip_residual_walks_from_dry_marker_cell_to_nearest_fluid(
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
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 1.0))
        fluid.obstacle[2, 2, 2] = 1

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
        self.assertEqual(report.direct_sample_marker_count, 0)
        self.assertEqual(report.normal_walk_sample_marker_count, 1)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
        self.assertEqual(report.no_fluid_sample_marker_count, 0)
        self.assertAlmostEqual(report.max_no_slip_residual_mps, 1.0, delta=1.0e-6)
        self.assertAlmostEqual(report.l2_no_slip_residual_mps, 1.0, delta=1.0e-6)

    def test_no_slip_residual_falls_back_to_nearest_fluid_when_normal_is_sealed(
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
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 0.0))
        fluid.velocity[3, 2, 2] = (2.0, 0.0, 0.0)
        fluid.obstacle.fill(1)
        fluid.obstacle[3, 2, 2] = 0

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
        self.assertEqual(report.direct_sample_marker_count, 0)
        self.assertEqual(report.normal_walk_sample_marker_count, 0)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 1)
        self.assertEqual(report.no_fluid_sample_marker_count, 0)
        self.assertAlmostEqual(report.max_no_slip_residual_mps, 2.0, delta=1.0e-6)
        self.assertAlmostEqual(report.l2_no_slip_residual_mps, 2.0, delta=1.0e-6)

    def test_no_slip_residual_reports_no_fluid_sample_reason(self) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.obstacle.fill(1)

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
            primary_region_id=7,
            secondary_region_id=8,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.primary_region_valid_marker_count, 0)
        self.assertEqual(report.primary_region_invalid_marker_count, 1)
        self.assertEqual(report.secondary_region_invalid_marker_count, 0)
        self.assertEqual(report.other_region_invalid_marker_count, 0)
        self.assertEqual(report.direct_sample_marker_count, 0)
        self.assertEqual(report.normal_walk_sample_marker_count, 0)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
        self.assertEqual(report.zero_normal_marker_count, 0)
        self.assertEqual(report.no_fluid_sample_marker_count, 1)

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
        self.assertAlmostEqual(report.max_abs_velocity_mps, 1.25, delta=1.0e-6)
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

    def test_ib_node_gradient_update_clears_stale_marker_gradient_without_fluid_sample(
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
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.velocity_dirichlet_mps_field[node] = (0.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.875)
        fluid.obstacle.fill(1)

        report = boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            search=search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=1000.0,
            dt_s=1.0e-3,
        )

        self.assertEqual(report.active_marker_count, 0)
        self.assertAlmostEqual(
            boundary.pressure_neumann_gradient_pa_per_m(node),
            0.0,
            delta=1.0e-6,
        )

    def test_ib_node_gradient_update_reports_raw_spike(self) -> None:
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
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.velocity_dirichlet_mps_field[node] = (0.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.875)
        fluid.velocity.fill((0.0, 0.0, 20.0))

        report = boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
            velocity_field=fluid.velocity,
            obstacle_field=fluid.obstacle,
            search=search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            density_kgm3=1000.0,
            dt_s=1.0e-3,
        )

        self.assertEqual(report.active_marker_count, 1)
        self.assertEqual(report.limited_gradient_count, 0)
        self.assertAlmostEqual(
            report.max_raw_abs_gradient_pa_per_m,
            2.0e7,
            delta=2.0e3,
        )
        self.assertAlmostEqual(
            boundary.pressure_neumann_gradient_pa_per_m(node),
            2.0e7,
            delta=2.0e3,
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
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
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
        expected_transmissibility_m = 0.04 / 0.25
        self.assertAlmostEqual(
            report.max_transmissibility_m,
            expected_transmissibility_m,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.max_diagonal_per_m2,
            expected_transmissibility_m / (0.25**3),
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            matrix_report["diagonal_integral"],
            2.0 * expected_transmissibility_m,
            delta=1.0e-5,
        )
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

    def test_pressure_neumann_rowlist_compacts_duplicate_owner_neighbor_pairs(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5), (0.625, 0.625, 0.5)),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.04, 0.04),
            region_ids=(7, 7),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(4, 4, 4),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(4, 4, 4),
            marker_capacity=2,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        for marker_index, node in enumerate(((0, 0, 0), (0, 0, 1))):
            fluid.obstacle[node] = 1
            boundary.active_ib_node[node] = 1
            boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
            boundary.pressure_neumann_gradient_field[node] = 25.0
            search.nearest_marker[node] = marker_index
            search.node_anchor_cell[node] = (2, 2, 2)
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
            pressure_coupling_extra_neighbor=(
                fluid.pressure_interface_coupling_extra_neighbor
            ),
            pressure_coupling_extra_coefficient=(
                fluid.pressure_interface_coupling_extra_coefficient
            ),
            pressure_interface_row_count=fluid.pressure_interface_row_count,
            pressure_interface_row_owner=fluid.pressure_interface_row_owner,
            pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
            pressure_interface_row_transmissibility=(
                fluid.pressure_interface_row_transmissibility
            ),
            pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        matrix_report = fluid.pressure_interface_matrix_terms_report()

        self.assertEqual(report.duplicate_owner_row_count, 1)
        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.pressure_interface_row_list_count, 1)
        self.assertEqual(int(fluid.pressure_interface_row_count[None]), 1)
        owner = tuple(
            int(fluid.pressure_interface_row_owner[0][axis]) for axis in range(3)
        )
        neighbor = tuple(
            int(fluid.pressure_interface_row_neighbor[0][axis]) for axis in range(3)
        )
        self.assertEqual(owner, (2, 2, 2))
        self.assertEqual(neighbor, (2, 2, 3))
        expected_single_transmissibility_m = 0.04 / 0.25
        expected_compacted_transmissibility_m = 2.0 * expected_single_transmissibility_m
        self.assertAlmostEqual(
            float(fluid.pressure_interface_row_transmissibility[0]),
            expected_compacted_transmissibility_m,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            matrix_report["diagonal_integral"],
            2.0 * expected_compacted_transmissibility_m,
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            matrix_report["row_diagonal_integral"],
            matrix_report["diagonal_integral"],
            delta=1.0e-5,
        )

    def test_pressure_neumann_matrix_rejects_row_adjacent_to_velocity_dirichlet_face(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5833333333, 0.5833333333, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(6, 6, 6),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(6, 6, 6),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(6, 6, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (0, 0, 0)
        owner = (3, 3, 3)
        neighbor = (3, 3, 4)
        seam = (3, 3, 5)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_anchor_cell[node] = owner
        search.node_boundary_point_m[node] = (0.5833333333, 0.5833333333, 0.5)
        search.node_interior_fluid_point_m[node] = (
            0.5833333333,
            0.5833333333,
            0.75,
        )
        fluid.obstacle[node] = 1
        fluid.velocity_dirichlet_boundary_active[seam] = 1

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
            pressure_interface_row_count=fluid.pressure_interface_row_count,
            pressure_interface_row_owner=fluid.pressure_interface_row_owner,
            pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
            pressure_interface_row_transmissibility=(
                fluid.pressure_interface_row_transmissibility
            ),
            pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
        )

        self.assertEqual(report.active_pressure_neumann_rows, 0)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.skipped_pressure_boundary_adjacent_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_row_count[None]), 0)
        self.assertEqual(float(fluid.pressure_interface_matrix_diagonal[owner]), 0.0)
        self.assertEqual(float(fluid.pressure_interface_matrix_diagonal[neighbor]), 0.0)

    def test_pressure_neumann_matrix_relocates_velocity_dirichlet_owner(self) -> None:
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

        active_rows = report.active_pressure_neumann_rows
        skipped_rows = report.skipped_velocity_dirichlet_row_count
        seam_skipped_rows = report.skipped_pressure_boundary_adjacent_row_count
        self.assertEqual(active_rows + skipped_rows + seam_skipped_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[2, 2, 2]), 0)
        if active_rows > 0:
            self.assertGreater(matrix_report["active_cells"], 0)
            self.assertGreater(matrix_report["diagonal_integral"], 0.0)
        self.assertAlmostEqual(matrix_report["rhs_integral"], 0.0)

    def test_pressure_neumann_matrix_counts_obstacle_owner_without_fluid_owner(
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
        fluid.obstacle.fill(1)

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

    def test_pressure_neumann_matrix_walks_obstacle_owner_to_first_fluid_cell(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.4),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        owner = (2, 2, 3)
        neighbor = (2, 2, 4)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.4)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.7)
        fluid.obstacle[node] = 1

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(report.relocated_obstacle_owner_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[owner]), 1)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[owner][axis])
                for axis in range(3)
            ),
            neighbor,
        )

    def test_pressure_neumann_matrix_walks_opposite_side_when_owner_side_blocked(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.6),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 3)
        owner = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.6)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.8)
        fluid.obstacle[node] = 1
        fluid.obstacle[2, 2, 4] = 1

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(report.relocated_obstacle_owner_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[owner]), 1)
        self.assertEqual(
            tuple(
                int(markers.marker_pressure_anchor_cell[0][axis])
                for axis in range(3)
            ),
            owner,
        )

    def test_pressure_neumann_matrix_walks_beyond_thick_air_backed_obstacle_band(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(12, 12, 12),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(12, 12, 12),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(12, 12, 12), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (6, 6, 6)
        owner = (6, 6, 1)
        neighbor = (6, 6, 0)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.5)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.04)
        fluid.obstacle.fill(1)
        fluid.obstacle[owner] = 0
        fluid.obstacle[neighbor] = 0

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(report.relocated_obstacle_owner_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[owner]), 1)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[owner][axis])
                for axis in range(3)
            ),
            neighbor,
        )

    def test_pressure_neumann_matrix_relocates_obstacle_owner_to_node_anchor(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.4),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        anchor = (2, 2, 3)
        neighbor = (2, 2, 4)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.4)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.7)
        search.node_anchor_cell[node] = anchor
        fluid.obstacle[node] = 1

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[anchor]), 1)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[anchor][axis])
                for axis in range(3)
            ),
            neighbor,
        )
        self.assertEqual(
            tuple(
                int(markers.marker_pressure_anchor_cell[0][axis])
                for axis in range(3)
            ),
            anchor,
        )
        self.assertGreater(float(fluid.pressure_interface_matrix_diagonal[anchor]), 0.0)
        self.assertGreater(float(fluid.pressure_interface_matrix_diagonal[neighbor]), 0.0)

    def test_pressure_neumann_matrix_coexists_with_relocated_velocity_dirichlet_anchor(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.4),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        anchor = (2, 2, 3)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.4)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.7)
        search.node_anchor_cell[node] = anchor
        fluid.obstacle[node] = 1
        fluid.velocity_dirichlet_boundary_active[anchor] = 1

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.skipped_velocity_dirichlet_row_count, 0)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(report.relocated_obstacle_owner_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[anchor]), 1)

    def test_pressure_neumann_duplicate_owner_rows_use_distinct_coupling_slots(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.6), (0.5, 0.5, 0.6)),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.04, 0.04),
            region_ids=(7, 7),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=2,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        owner = (2, 2, 3)
        neighbor = (2, 2, 4)
        active_nodes = ((2, 2, 2), (2, 3, 2))
        for marker, node in enumerate(active_nodes):
            boundary.active_ib_node[node] = 1
            boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
            boundary.pressure_neumann_gradient_field[node] = 25.0 + marker
            search.nearest_marker[node] = marker
            search.node_boundary_point_m[node] = (0.5, 0.5, 0.6)
            search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.9)
            search.node_anchor_cell[node] = owner
            fluid.obstacle[node] = 1

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
            pressure_coupling_extra_neighbor=(
                fluid.pressure_interface_coupling_extra_neighbor
            ),
            pressure_coupling_extra_coefficient=(
                fluid.pressure_interface_coupling_extra_coefficient
            ),
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        coupling0 = float(fluid.pressure_interface_coupling_coefficient[owner])
        coupling1 = float(
            fluid.pressure_interface_coupling_extra_coefficient[
                owner[0],
                owner[1],
                owner[2],
                0,
            ]
        )
        expected_diagonal = (coupling0 + coupling1) / (0.2**3)

        self.assertEqual(report.active_pressure_neumann_rows, 2)
        self.assertEqual(report.duplicate_owner_row_count, 1)
        self.assertEqual(report.overflow_owner_row_count, 0)
        self.assertEqual(report.max_owner_slot_count, 2)
        self.assertEqual(report.relocated_obstacle_owner_row_count, 2)
        self.assertEqual(report.skipped_obstacle_owner_row_count, 0)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[owner]), 2)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[owner][axis])
                for axis in range(3)
            ),
            neighbor,
        )
        self.assertEqual(
            tuple(
                int(
                    fluid.pressure_interface_coupling_extra_neighbor[
                        owner[0],
                        owner[1],
                        owner[2],
                        0,
                    ][axis]
                )
                for axis in range(3)
            ),
            neighbor,
        )
        self.assertAlmostEqual(
            float(fluid.pressure_interface_matrix_diagonal[owner]),
            expected_diagonal,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            float(fluid.pressure_interface_matrix_diagonal[neighbor]),
            expected_diagonal,
            delta=5.0e-5,
        )

    def test_pressure_neumann_owner_slot_overflow_merges_duplicate_neighbor(
        self,
    ) -> None:
        marker_count = PRESSURE_INTERFACE_COUPLING_SLOT_COUNT + 1
        markers = HibmMpmSurfaceMarkers(marker_capacity=marker_count)
        markers.load_markers(
            positions_m=tuple((0.5, 0.5, 0.6) for _ in range(marker_count)),
            velocities_mps=tuple((0.0, 0.0, 0.0) for _ in range(marker_count)),
            normals=tuple((0.0, 0.0, 1.0) for _ in range(marker_count)),
            areas_m2=tuple(0.04 for _ in range(marker_count)),
            region_ids=tuple(7 for _ in range(marker_count)),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=marker_count,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=marker_count,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        owner = (2, 2, 3)
        neighbor = (2, 2, 4)
        active_nodes = tuple(
            node
            for node in (
                (i, j, k)
                for i in range(5)
                for j in range(5)
                for k in range(5)
            )
            if node not in (owner, neighbor)
        )[:marker_count]
        self.assertEqual(len(active_nodes), marker_count)
        for marker, node in enumerate(active_nodes):
            boundary.active_ib_node[node] = 1
            boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
            boundary.pressure_neumann_gradient_field[node] = 25.0
            search.nearest_marker[node] = marker
            search.node_boundary_point_m[node] = (0.5, 0.5, 0.6)
            search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.9)
            search.node_anchor_cell[node] = owner
            fluid.obstacle[node] = 1

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
            pressure_coupling_extra_neighbor=(
                fluid.pressure_interface_coupling_extra_neighbor
            ),
            pressure_coupling_extra_coefficient=(
                fluid.pressure_interface_coupling_extra_coefficient
            ),
            pressure_interface_row_count=fluid.pressure_interface_row_count,
            pressure_interface_row_owner=fluid.pressure_interface_row_owner,
            pressure_interface_row_neighbor=fluid.pressure_interface_row_neighbor,
            pressure_interface_row_transmissibility=(
                fluid.pressure_interface_row_transmissibility
            ),
            pressure_interface_row_capacity=fluid.pressure_interface_row_capacity,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        accepted_coupling = 0.0
        for row in range(marker_count):
            self.assertEqual(
                tuple(
                    int(fluid.pressure_interface_row_owner[row][axis])
                    for axis in range(3)
                ),
                owner,
            )
            self.assertEqual(
                tuple(
                    int(fluid.pressure_interface_row_neighbor[row][axis])
                    for axis in range(3)
                ),
                neighbor,
            )
            accepted_coupling += float(
                fluid.pressure_interface_row_transmissibility[row]
            )
        expected_diagonal = accepted_coupling / (0.2**3)

        self.assertEqual(report.active_pressure_neumann_rows, marker_count)
        self.assertEqual(report.overflow_owner_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_row_count[None]), marker_count)
        self.assertEqual(
            report.duplicate_owner_row_count,
            PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
        )
        self.assertEqual(
            report.max_owner_slot_count,
            PRESSURE_INTERFACE_COUPLING_SLOT_COUNT + 1,
        )
        self.assertEqual(
            int(fluid.pressure_interface_coupling_active[owner]),
            PRESSURE_INTERFACE_COUPLING_SLOT_COUNT + 1,
        )
        self.assertAlmostEqual(
            float(fluid.pressure_interface_matrix_diagonal[owner]),
            expected_diagonal,
            delta=5.0e-5,
        )
        self.assertAlmostEqual(
            float(fluid.pressure_interface_matrix_diagonal[neighbor]),
            expected_diagonal,
            delta=5.0e-5,
        )
        fluid.pressure.fill(0.0)
        fluid.cg_z.fill(0.0)
        fluid.pressure[neighbor] = 1.0
        fluid.pressure_interface_row_count[None] = 0
        fluid.pressure_interface_coupling_active[owner] = 0
        fluid._fv_laplacian_apply_kernel(fluid.pressure, fluid.cg_z, 0)
        owner_without_rows = float(fluid.cg_z[owner])
        fluid.cg_z.fill(0.0)
        fluid.pressure_interface_row_count[None] = marker_count
        fluid._fv_laplacian_apply_kernel(fluid.pressure, fluid.cg_z, 0)
        owner_with_rows = float(fluid.cg_z[owner])
        self.assertAlmostEqual(
            owner_with_rows - owner_without_rows,
            -expected_diagonal,
            delta=5.0e-5,
        )

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
        expected_cell_area_m2 = (0.25**3) / expected_normal_spacing_m
        expected_interface_area_m2 = min(0.04, expected_cell_area_m2)
        expected_transmissibility = expected_interface_area_m2 / 0.025
        expected_coefficient = expected_transmissibility / (0.25**3)
        expected_diagonal_integral = 2.0 * expected_transmissibility

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
        expected_cell_area_m2 = (0.25**3) / expected_normal_spacing_m
        expected_interface_area_m2 = min(0.04, expected_cell_area_m2)
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

    def test_pressure_neumann_rejects_when_no_fallback_neighbor_available(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.45, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.45, 0.5, 0.5)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.5)
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[node] = 0
        fluid.obstacle.from_numpy(obstacle)

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
        self.assertEqual(report.invalid_reconstruction_row_count, 1)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 0)
        self.assertEqual(float(fluid.pressure_interface_matrix_diagonal[node]), 0.0)

        diagnostic_rows = boundary.pressure_neumann_invalid_diagnostic_rows(
            search=search,
            markers=markers,
            fluid=fluid,
        )
        self.assertEqual(len(diagnostic_rows), 1)
        diagnostic = diagnostic_rows[0]
        self.assertEqual(diagnostic["reason"], "unreconstructable")
        self.assertEqual(
            (diagnostic["node_i"], diagnostic["node_j"], diagnostic["node_k"]),
            node,
        )
        self.assertEqual(
            (diagnostic["owner_i"], diagnostic["owner_j"], diagnostic["owner_k"]),
            node,
        )
        self.assertEqual(diagnostic["marker_index"], 0)
        self.assertEqual(diagnostic["marker_region_id"], 7)

    def test_pressure_neumann_zero_gradient_unreconstructable_row_is_natural_skip(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.45, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 0.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.45, 0.5, 0.5)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.5)
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[node] = 0
        fluid.obstacle.from_numpy(obstacle)

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
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)
        self.assertEqual(int(fluid.pressure_interface_coupling_active[node]), 0)
        self.assertEqual(float(fluid.pressure_interface_matrix_diagonal[node]), 0.0)

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

    def test_pressure_neumann_falls_back_for_owner_behind_boundary_plane(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.75),),
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
        node = (2, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.625, 0.625, 0.75)
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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)
        self.assertGreater(report.min_reconstruction_gap_m, 0.0)

    def test_pressure_neumann_fallback_scans_non_dominant_neighbor(
        self,
    ) -> None:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.625),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((inv_sqrt2, inv_sqrt2, 0.0),),
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
        boundary.pressure_neumann_normal_field[node] = (inv_sqrt2, inv_sqrt2, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.625)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.625)
        fluid.obstacle[3, 2, 2] = 1
        fluid.obstacle[1, 2, 2] = 1

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)

    def test_pressure_neumann_walks_normal_line_to_pressure_correctable_neighbor(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.43, 0.4166666667, 0.4166666667),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(6, 6, 6),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(6, 6, 6),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(6, 6, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        blocked_neighbor = (3, 2, 2)
        normal_line_neighbor = (4, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.43, 0.4166666667, 0.4166666667)
        search.node_interior_fluid_point_m[node] = (0.58, 0.4166666667, 0.4166666667)

        obstacle = np.ones((6, 6, 6), dtype=np.int32)
        obstacle[node] = 0
        obstacle[blocked_neighbor] = 1
        obstacle[normal_line_neighbor] = 0
        fluid.obstacle.from_numpy(obstacle)

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[node][axis])
                for axis in range(3)
            ),
            normal_line_neighbor,
        )

    def test_pressure_neumann_falls_back_to_nearby_pressure_correctable_cell_when_normal_band_blocked(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.43, 0.4166666667, 0.4166666667),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(6, 6, 6),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(6, 6, 6),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(6, 6, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        nearby_pressure_cell = (4, 3, 2)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.43, 0.4166666667, 0.4166666667)
        search.node_interior_fluid_point_m[node] = (
            0.58,
            0.4166666667,
            0.4166666667,
        )

        obstacle = np.ones((6, 6, 6), dtype=np.int32)
        obstacle[node] = 0
        obstacle[nearby_pressure_cell] = 0
        fluid.obstacle.from_numpy(obstacle)

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[node][axis])
                for axis in range(3)
            ),
            nearby_pressure_cell,
        )

    def test_pressure_neumann_uses_node_anchor_when_fluid_owner_has_no_local_neighbor(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.45),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=(5, 5, 5),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(5, 5, 5),
            marker_capacity=1,
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        node = (2, 2, 2)
        anchor = (2, 2, 4)
        boundary.active_ib_node[node] = 1
        boundary.pressure_neumann_normal_field[node] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_gradient_field[node] = 25.0
        search.nearest_marker[node] = 0
        search.node_boundary_point_m[node] = (0.5, 0.5, 0.45)
        search.node_interior_fluid_point_m[node] = (0.5, 0.5, 0.7)
        search.node_anchor_cell[node] = anchor

        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[node] = 0
        obstacle[anchor] = 0
        fluid.obstacle.from_numpy(obstacle)

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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertEqual(report.invalid_unreconstructable_row_count, 0)
        self.assertEqual(
            tuple(
                int(fluid.pressure_interface_coupling_neighbor[node][axis])
                for axis in range(3)
            ),
            anchor,
        )

    def test_pressure_neumann_invalid_reconstruction_reports_subtypes(self) -> None:
        report = HibmMpmPressureNeumannMatrixReport(
            active_pressure_neumann_rows=0,
            rhs_integral=0.0,
            max_abs_rhs=0.0,
            invalid_reconstruction_row_count=3,
            invalid_unreconstructable_row_count=1,
            invalid_bad_marker_row_count=1,
            invalid_nonpositive_volume_row_count=1,
        )

        self.assertEqual(report.invalid_reconstruction_row_count, 3)
        self.assertEqual(report.invalid_unreconstructable_row_count, 1)
        self.assertEqual(report.invalid_bad_marker_row_count, 1)
        self.assertEqual(report.invalid_nonpositive_volume_row_count, 1)

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
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.invalid_reconstruction_row_count, 0)
        self.assertAlmostEqual(report.min_projection_weight, 0.5, delta=1.0e-6)
        self.assertAlmostEqual(report.max_projection_weight, 0.5, delta=1.0e-6)
        self.assertEqual(fluid.velocity_dirichlet_boundary_active[2, 2, 2], 1)
        self.assertEqual(
            int(fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 2]),
            7,
        )
        reconstructed_z = float(
            fluid.velocity_dirichlet_boundary_value_mps[2, 2, 2][2]
        )
        self.assertAlmostEqual(reconstructed_z, 0.25, delta=1.0e-6)
        self.assertAlmostEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[2, 2, 2]),
            0.5,
            delta=1.0e-6,
        )

    def test_no_slip_wall_velocity_mode_does_not_reinject_polluted_interior_velocity(
        self,
    ) -> None:
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
        fluid.velocity.fill((0.0, 0.0, 1000.0))

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
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
            interpolate_interior_velocity=False,
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(report.boundary_velocity_only_row_count, 1)
        self.assertAlmostEqual(
            report.raw_reconstructed_max_abs_velocity_mps,
            500.05,
            delta=1.0e-4,
        )
        self.assertAlmostEqual(report.max_abs_velocity_mps, 0.1, delta=1.0e-6)
        self.assertAlmostEqual(
            float(fluid.velocity_dirichlet_boundary_value_mps[2, 2, 2][2]),
            0.1,
            delta=1.0e-6,
        )
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

    def test_velocity_dirichlet_row_relocates_to_opposite_fluid_side_when_normal_side_is_air(
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
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[2:, :, :] = 1
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
        self.assertEqual(report.relocated_row_count, 1)
        self.assertEqual(report.relocation_blocked_row_count, 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[2, 2, 2]), 0)
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[1, 2, 2]), 1)
        np.testing.assert_allclose(
            tuple(
                float(fluid.velocity_dirichlet_boundary_value_mps[1, 2, 2][axis])
                for axis in range(3)
            ),
            (0.13333334, 0.0, 0.0),
            atol=1.0e-7,
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

    def test_no_slip_reconstruction_falls_back_for_node_beyond_interior_segment(
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
        search.node_boundary_point_m[node] = (0.25, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.375, 0.625, 0.625)
        fluid.velocity.fill((0.4, 0.0, 0.0))

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
        self.assertEqual(report.invalid_no_fluid_sample_row_count, 0)
        self.assertEqual(report.invalid_nonpositive_gap_row_count, 0)
        self.assertEqual(report.invalid_node_behind_boundary_row_count, 0)
        self.assertEqual(report.invalid_node_beyond_interior_row_count, 0)
        self.assertGreater(
            float(fluid.velocity_dirichlet_boundary_projection_weight[node]),
            0.0,
        )
        np.testing.assert_allclose(
            tuple(
                float(fluid.velocity_dirichlet_boundary_value_mps[node][axis])
                for axis in range(3)
            ),
            (0.35, 0.0, 0.0),
            atol=1.0e-6,
        )

    def test_no_slip_reconstruction_falls_back_for_node_behind_boundary_plane(
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
        node = (1, 2, 2)
        boundary.active_ib_node[node] = 1
        boundary.velocity_dirichlet_mps_field[node] = (0.2, 0.0, 0.0)
        boundary.pressure_neumann_normal_field[node] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[node] = (0.5, 0.625, 0.625)
        search.node_interior_fluid_point_m[node] = (0.625, 0.625, 0.625)
        fluid.velocity.fill((0.4, 0.0, 0.0))

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
        self.assertEqual(report.invalid_node_behind_boundary_row_count, 0)
        self.assertEqual(report.invalid_node_beyond_interior_row_count, 0)
        self.assertAlmostEqual(
            float(fluid.velocity_dirichlet_boundary_projection_weight[node]),
            0.5,
            delta=1.0e-6,
        )
        np.testing.assert_allclose(
            tuple(
                float(fluid.velocity_dirichlet_boundary_value_mps[node][axis])
                for axis in range(3)
            ),
            (0.3, 0.0, 0.0),
            atol=1.0e-6,
        )

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
        self.assertLessEqual(
            matrix_report["diagonal_integral"],
            2.0 * 0.04 / max(report.min_reconstruction_gap_m, 1.0e-12) + 1.0e-5,
        )
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
    def test_reported_pressure_solve_failure_stops_before_stress_sampling(
        self,
    ) -> None:
        source = inspect.getsource(assemble_hibm_mpm_sharp_fluid_to_mpm_loads)

        guard = source.index('projection_report.get("pressure_solve_failed"')
        no_slip = source.index('sample_no_slip_residual:start')
        stress = source.index('sample_fluid_stress_to_marker_tractions:start')
        scatter = source.index('marker_force_scatter:start')

        self.assertLess(guard, no_slip)
        self.assertLess(guard, stress)
        self.assertLess(guard, scatter)
        self.assertIn("raise RuntimeError", source[guard:no_slip])
        self.assertIn("pressure solve failed", source[guard:no_slip])

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
            runtime=TaichiRuntimeConfig(arch="cpu"),
        )
        solid = NeoHookeanMpmState(
            particle_capacity=1,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cpu"),
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

    def test_mpm_force_scatter_rejects_nonfinite_marker_force(self) -> None:
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
        solid.x[0] = (0.5, 0.5, 0.5)
        solid.external_force_n[0] = (1.0, 2.0, 3.0)
        markers.F_gamma_n[0] = (float("nan"), 0.0, 0.0)

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=0.5,
        )

        self.assertEqual(report.active_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.total_marker_force_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.total_mpm_external_force_n, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(
            tuple(float(solid.external_force_n[0][axis]) for axis in range(3)),
            (1.0, 2.0, 3.0),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_mpm_external_force_clear_counts_all_particles_with_zero_stale_force(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        solid = NeoHookeanMpmState(
            particle_capacity=2,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(4, 4, 4),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solid.initialize_box(
            particle_counts=(2, 1, 1),
            box_min_m=(0.0, 0.0, 0.0),
            box_max_m=(1.0, 1.0, 1.0),
            density_kgm3=1000.0,
        )

        report = markers.clear_mpm_external_forces(
            solid.external_force_n,
            particle_count=solid.particle_count,
        )

        self.assertEqual(report.cleared_particle_count, 2)
        self.assertEqual(report.max_abs_external_force_before_n, 0.0)

    def test_mpm_force_scatter_reports_applied_mpm_force_after_adapter_cast(
        self,
    ) -> None:
        source = inspect.getsource(
            HibmMpmSurfaceMarkers._scatter_marker_forces_to_mpm_particles_kernel
        )
        self.assertIn("force_contribution_for_external", source)
        self.assertIn("ti.cast(force_contribution.x, ti.f32)", source)
        self.assertIn(
            "external_force_n[particle] += force_contribution_for_external",
            source,
        )

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
        solid.x[0] = (0.5, 0.5, 0.5)
        markers.F_gamma_n[0] = (1.0 / 3.0, -2.0 / 3.0, 7.0 / 9.0)

        report = markers.scatter_marker_forces_to_mpm_particles(
            solid.external_force_n,
            solid.x,
            particle_count=solid.particle_count,
            support_radius_m=0.5,
        )

        applied = tuple(float(solid.external_force_n[0][axis]) for axis in range(3))
        self.assertEqual(solid.external_force_n.dtype, ti.f32)
        np.testing.assert_allclose(
            report.total_mpm_external_force_n,
            applied,
            rtol=0.0,
            atol=0.0,
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
        pressure_row_assembly_project_counts: list[int] = []
        original_assemble_pressure_rows = boundary.assemble_pressure_neumann_matrix_rows
        inside_project = False

        def counted_apply_velocity_rows(*args, **kwargs):
            velocity_row_apply_contexts.append(
                "project" if inside_project else "outside_project"
            )
            return original_apply_velocity_rows(*args, **kwargs)

        fluid.apply_velocity_dirichlet_boundary_rows = counted_apply_velocity_rows

        def counted_assemble_pressure_rows(*args, **kwargs):
            pressure_row_assembly_project_counts.append(
                sum(1 for call in calls if call.startswith("project"))
            )
            return original_assemble_pressure_rows(*args, **kwargs)

        boundary.assemble_pressure_neumann_matrix_rows = counted_assemble_pressure_rows

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
                    ["predict0", "project0", "predict1", "project1"]
                    + [f"project{i}" for i in range(2, substep)],
                )
            self.assertEqual(kwargs["dt_s"], 2.5e-4)
            self.assertEqual(len(pressure_row_assembly_project_counts), substep + 1)
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
            [
                "predict0",
                "project0",
                "predict1",
                "project1",
                "project2",
                "project3",
                "project4",
            ],
        )
        self.assertEqual(
            velocity_row_apply_contexts,
            ["project", "project", "project", "project", "project"],
        )
        self.assertEqual(pressure_row_assembly_project_counts, [0, 1, 2, 3, 4])
        self.assertTrue(report.fluid_predictor_applied)
        self.assertEqual(report.fluid_projection["fluid_substeps"], 2)
        self.assertEqual(report.fluid_projection["cg_project_calls"], 5)
        self.assertEqual(report.fluid_projection["cg_iterations_total"], 15)
        self.assertEqual(report.fluid_projection["cg_mean_projection_count"], 30)

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
        self.assertTrue(report.post_solid_kinematic_projection_applied)
        self.assertIsNotNone(report.post_solid_no_slip_residual)
        summary = hibm_mpm_sharp_step_summary(report)
        self.assertTrue(summary["hibm_post_solid_kinematic_projection_applied"])
        self.assertIn("hibm_post_solid_no_slip_residual_max_mps", summary)
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
            "hibm_pressure_disconnected_nonprojectable_cell_count",
            "hibm_pressure_disconnected_component_count",
            "hibm_pressure_disconnected_component_raw_count",
            "hibm_pressure_disconnected_largest_component_cell_count",
            "hibm_pressure_disconnected_singleton_component_count",
            "hibm_pressure_disconnected_small_component_threshold_cells",
            "hibm_pressure_disconnected_small_component_count",
            "hibm_pressure_disconnected_small_component_cell_count",
            "hibm_pressure_disconnected_component_overflow",
            "hibm_no_slip_residual_max_mps",
            "hibm_no_slip_residual_l2_mps",
            "hibm_no_slip_residual_direct_sample_marker_count",
            "hibm_no_slip_residual_normal_walk_sample_marker_count",
            "hibm_no_slip_residual_nearest_fluid_sample_marker_count",
            "hibm_no_slip_residual_zero_normal_marker_count",
            "hibm_no_slip_residual_no_fluid_sample_marker_count",
            "hibm_no_slip_residual_primary_region_valid_marker_count",
            "hibm_no_slip_residual_primary_region_invalid_marker_count",
            "hibm_no_slip_residual_secondary_region_valid_marker_count",
            "hibm_no_slip_residual_secondary_region_invalid_marker_count",
            "hibm_no_slip_residual_other_region_valid_marker_count",
            "hibm_no_slip_residual_other_region_invalid_marker_count",
            "hibm_boundary_max_abs_velocity_mps",
            "hibm_velocity_dirichlet_max_abs_velocity_mps",
            "hibm_velocity_dirichlet_invalid_reconstruction_count",
            "hibm_velocity_dirichlet_invalid_no_fluid_sample_count",
            "hibm_velocity_dirichlet_invalid_nonpositive_gap_count",
            "hibm_velocity_dirichlet_invalid_node_behind_boundary_count",
            "hibm_velocity_dirichlet_invalid_node_beyond_interior_count",
            "hibm_velocity_dirichlet_primary_region_active_rows",
            "hibm_velocity_dirichlet_secondary_region_active_rows",
            "hibm_velocity_dirichlet_other_region_active_rows",
            "hibm_velocity_dirichlet_unassigned_region_active_rows",
            "hibm_velocity_dirichlet_min_projection_weight",
            "hibm_velocity_dirichlet_max_projection_weight",
            "hibm_velocity_dirichlet_apply_calls",
            "hibm_velocity_dirichlet_applied_active_cells_total",
            "hibm_velocity_dirichlet_applied_active_cells_max",
            "hibm_velocity_dirichlet_applied_max_delta_mps",
            "hibm_velocity_dirichlet_applied_mean_delta_mps",
            "hibm_pressure_neumann_active_rows",
            "hibm_pressure_neumann_skipped_pressure_boundary_adjacent_count",
            "hibm_pressure_neumann_invalid_reconstruction_count",
            "hibm_pressure_neumann_invalid_unreconstructable_count",
            "hibm_pressure_neumann_invalid_bad_marker_count",
            "hibm_pressure_neumann_invalid_nonpositive_volume_count",
            "hibm_pressure_neumann_min_reconstruction_gap_m",
            "hibm_pressure_neumann_max_reconstruction_gap_m",
            "hibm_pressure_neumann_max_transmissibility_m",
            "hibm_pressure_neumann_max_raw_transmissibility_m",
            "hibm_pressure_neumann_max_transmissibility_limit_m",
            "hibm_pressure_neumann_transmissibility_capped_row_count",
            "hibm_pressure_neumann_max_diagonal_per_m2",
            "hibm_pressure_neumann_active_marker_count",
            "hibm_pressure_neumann_max_rows_per_marker",
            "hibm_pressure_neumann_gradient_raw_max_abs_pa_per_m",
            "hibm_pressure_neumann_gradient_limited_count",
            "hibm_full_stress_viscous_gradient_invalid_marker_count",
            "hibm_marker_primary_count",
            "hibm_marker_secondary_count",
            "hibm_marker_total_count",
            "hibm_marker_primary_stress_valid_count",
            "hibm_marker_primary_stress_invalid_count",
            "hibm_marker_secondary_stress_valid_count",
            "hibm_marker_secondary_stress_invalid_count",
            "hibm_marker_total_force_n",
            "hibm_marker_action_reaction_residual_n",
            "hibm_mpm_external_force_clear_particle_count",
            "hibm_mpm_external_force_clear_max_abs_before_n",
            "hibm_mpm_external_force_fresh_for_solid_step",
            "hibm_mpm_scatter_action_reaction_residual_n",
            "hibm_surface_updated_marker_count",
            "hibm_next_ib_node_count",
            "hibm_next_internal_obstacle_cell_count",
            "hibm_next_pressure_disconnected_nonprojectable_cell_count",
            "hibm_next_pressure_disconnected_component_count",
            "hibm_next_pressure_disconnected_component_raw_count",
            "hibm_next_pressure_disconnected_largest_component_cell_count",
            "hibm_next_pressure_disconnected_singleton_component_count",
            "hibm_next_pressure_disconnected_small_component_threshold_cells",
            "hibm_next_pressure_disconnected_small_component_count",
            "hibm_next_pressure_disconnected_small_component_cell_count",
            "hibm_next_pressure_disconnected_component_overflow",
            "hibm_next_boundary_no_slip_count",
            "hibm_next_boundary_max_abs_velocity_mps",
            "hibm_next_velocity_dirichlet_active_rows",
            "hibm_next_velocity_dirichlet_max_abs_velocity_mps",
            "hibm_next_velocity_dirichlet_invalid_reconstruction_count",
            "hibm_next_velocity_dirichlet_primary_region_active_rows",
            "hibm_next_velocity_dirichlet_secondary_region_active_rows",
            "hibm_next_velocity_dirichlet_other_region_active_rows",
            "hibm_next_velocity_dirichlet_unassigned_region_active_rows",
            "hibm_next_pressure_neumann_skipped_pressure_boundary_adjacent_count",
            "hibm_next_pressure_neumann_invalid_reconstruction_count",
            "hibm_next_pressure_neumann_invalid_unreconstructable_count",
            "hibm_next_pressure_neumann_invalid_bad_marker_count",
            "hibm_next_pressure_neumann_invalid_nonpositive_volume_count",
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
            "hibm_post_solid_divergence_l2",
            "hibm_post_solid_divergence_max_abs",
            "hibm_post_solid_interior_divergence_l2",
            "hibm_post_solid_interior_divergence_max_abs",
            "hibm_post_solid_projection_divergence_l2",
            "hibm_post_solid_projection_divergence_max_abs",
            "hibm_post_solid_post_boundary_divergence_l2",
            "hibm_post_solid_post_boundary_divergence_max_abs",
            "hibm_post_solid_post_constraint_divergence_l2",
            "hibm_post_solid_post_constraint_divergence_max_abs",
            "hibm_post_solid_no_slip_residual_direct_sample_marker_count",
            "hibm_post_solid_no_slip_residual_normal_walk_sample_marker_count",
            "hibm_post_solid_no_slip_residual_nearest_fluid_sample_marker_count",
            "hibm_post_solid_no_slip_residual_zero_normal_marker_count",
            "hibm_post_solid_no_slip_residual_no_fluid_sample_marker_count",
            "hibm_post_solid_no_slip_residual_primary_region_valid_marker_count",
            "hibm_post_solid_no_slip_residual_primary_region_invalid_marker_count",
            "hibm_post_solid_no_slip_residual_secondary_region_valid_marker_count",
            "hibm_post_solid_no_slip_residual_secondary_region_invalid_marker_count",
            "hibm_post_solid_no_slip_residual_other_region_valid_marker_count",
            "hibm_post_solid_no_slip_residual_other_region_invalid_marker_count",
        }
        self.assertTrue(expected_keys <= set(summary))
        self.assertEqual(summary["hibm_coupling_scheme"], "explicit_loose")
        self.assertEqual(
            summary["hibm_added_mass_stability_status"],
            "unmeasured_single_pass",
        )
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
            summary["hibm_marker_primary_stress_invalid_count"],
            report.fluid_to_mpm_loads.marker_forces.primary_stress_invalid_marker_count,
        )
        self.assertEqual(
            summary["hibm_marker_secondary_stress_invalid_count"],
            report.fluid_to_mpm_loads.marker_forces.secondary_stress_invalid_marker_count,
        )
        self.assertEqual(
            summary["hibm_mpm_external_force_clear_particle_count"],
            report.fluid_to_mpm_loads.mpm_external_force_clear.cleared_particle_count,
        )
        self.assertEqual(
            summary["hibm_mpm_external_force_clear_max_abs_before_n"],
            report.fluid_to_mpm_loads.mpm_external_force_clear.max_abs_external_force_before_n,
        )
        self.assertTrue(summary["hibm_mpm_external_force_fresh_for_solid_step"])
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

    def test_sharp_mpm_step_requires_fresh_external_force_before_solid_advance(
        self,
    ) -> None:
        def load_report(
            *,
            cleared_particle_count: int,
            active_marker_count: int,
            active_particle_count: int,
            residual_n: float,
        ):
            return SimpleNamespace(
                mpm_external_force_clear=HibmMpmExternalForceClearReport(
                    cleared_particle_count=cleared_particle_count,
                    max_abs_external_force_before_n=3.0,
                ),
                mpm_force_scatter=HibmMpmMpmForceScatterReport(
                    active_marker_count=active_marker_count,
                    invalid_marker_count=0,
                    active_particle_count=active_particle_count,
                    total_marker_force_n=(0.0, 0.0, 0.0),
                    total_mpm_external_force_n=(0.0, 0.0, 0.0),
                    action_reaction_residual_n=residual_n,
                ),
            )

        self.assertTrue(
            hibm_mpm_external_force_fresh_for_solid_step(
                load_report(
                    cleared_particle_count=1,
                    active_marker_count=1,
                    active_particle_count=1,
                    residual_n=0.0,
                )
            )
        )
        self.assertFalse(
            hibm_mpm_external_force_fresh_for_solid_step(
                load_report(
                    cleared_particle_count=1,
                    active_marker_count=1,
                    active_particle_count=0,
                    residual_n=0.0,
                )
            )
        )
        self.assertFalse(
            hibm_mpm_external_force_fresh_for_solid_step(
                load_report(
                    cleared_particle_count=1,
                    active_marker_count=1,
                    active_particle_count=1,
                    residual_n=float("nan"),
                )
            )
        )

        source = inspect.getsource(advance_hibm_mpm_sharp_mpm_step)
        guard_index = source.index(
            "if not hibm_mpm_external_force_fresh_for_solid_step(load_report):"
        )
        solid_step_index = source.index("mpm_report = solid_step()")
        self.assertLess(guard_index, solid_step_index)


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

    def test_far_pressure_side_sign_overrides_ambiguous_two_side_sampling(
        self,
    ) -> None:
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
        fluid.pressure.from_numpy(np.full((8, 8, 8), 2.0, dtype=np.float32))
        fluid.obstacle.from_numpy(np.zeros((8, 8, 8), dtype=np.int32))

        inside_far = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_side_normal_sign=-1.0,
        )

        self.assertEqual(inside_far.valid_marker_count, 1)
        self.assertEqual(inside_far.far_pressure_closed_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[2], 8.0, delta=1.0e-4)

        outside_far = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_side_normal_sign=1.0,
        )

        self.assertEqual(outside_far.valid_marker_count, 1)
        self.assertEqual(outside_far.far_pressure_closed_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_extended_inside_walk_closes_marker_behind_thick_band(self) -> None:
        # 16^3 unit box: cell width 1/16 = 0.0625, cell center z_c(k) = (k + 0.5) / 16.
        # Marker x = y = 0.53125 is exactly cell-center index 8, so trilinear
        # sampling reduces to the single x = y = 8 column; z = 0.5 is the face
        # between cells 7 and 8 (grid coordinate 7.5, k_near = 8), so the
        # normal-aligned probe distance is exactly one cell width (0.0625).
        #
        # z layout (cell indices: centers -> role):
        #   8..15: 0.53125..0.96875 -> air side, obstacle = 1
        #   4..7 : 0.28125..0.46875 -> solid band + membrane thickness, obstacle = 1
        #   0..3 : 0.03125..0.21875 -> water, obstacle = 0, pressure = 2.0
        #
        # Standard inside (-n) candidates probe z = 0.5 - {1.0, 1.5, 2.0, 2.5, 3.0}
        # * 0.0625 = {0.4375, 0.40625, 0.375, 0.34375, 0.3125}; their trilinear
        # supports {6,7}, {6}, {5,6}, {5}, {4,5} are all obstacle cells, so the
        # standard ladder can never reach water. The first extended candidate at
        # 3.6 * 0.0625 = 0.225 probes z = 0.275 (grid coordinate 3.9, support
        # {3: 0.1, 4: 0.9}) and reaches water cell 3, i.e. the first water
        # contact needs > 3x and <= 6x the normal grid spacing.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 2.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 8:] = 1
        obstacle[:, :, 4:8] = 1
        fluid.obstacle.from_numpy(obstacle)

        default_report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(default_report.valid_marker_count, 0)
        self.assertEqual(default_report.invalid_marker_count, 1)
        self.assertEqual(default_report.far_pressure_closed_marker_count, 0)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_inside_probe_max_multiplier=6.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # traction = (p_inside_water - p_far_air) * n = (2 - 10) * +z
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_extended_walk_does_not_change_standard_reach_markers(self) -> None:
        markers, fluid = self._water_below_air_above_fixture()

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_inside_probe_max_multiplier=6.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # Identical to test_far_pressure_closes_water_air_marker_with_known_
        # outside_pressure: (2 - 10) * +z, untouched by the opt-in extended
        # multiplier because the standard 1x candidate already reaches water.
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_extended_walk_must_not_bypass_mirrored_closure_through_thin_solid(
        self,
    ) -> None:
        # Mirrored orientation (water OUTSIDE on +n, structurally dry side on
        # the INSIDE walk) with deep unrelated water beyond the dry band.
        # 16^3 unit box, spacing 0.0625; marker at z=0.5, normal +z.
        #   cells z 8..15  : water above (outside walk finds it at 1.0x)
        #   cells z 4..7   : dry band (standard inside ladder, supports
        #                    {6,7}/{6}/{5,6}/{5}/{4,5}, never reaches water)
        #   cells z 0..3   : deep unrelated water, first reachable by the
        #                    extended candidate 3.6x (z=0.275, support
        #                    {3: 0.1, 4: 0.9})
        # Without the outside_found == 0 gate the extended walk "tunnels" to
        # the deep water and silently migrates the marker from the mirrored
        # closure (p_far - p_water) * n to a spurious two-sided sample
        # (p_deep - p_water) * n, dropping the drive pressure entirely. The
        # marker must stay on the mirrored closure branch instead.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 2.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 4:8] = 1
        fluid.obstacle.from_numpy(obstacle)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_inside_probe_max_multiplier=6.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # Mirrored closure: (p_far_air - p_outside_water) * n = (10 - 2) * +z.
        # The buggy tunneling path would instead give (p_deep - p_outside) * n
        # = (2 - 2) * +z = 0.
        self.assertAlmostEqual(traction[2], 8.0, delta=1.0e-4)

    def test_closure_survives_incomplete_viscous_gradient_stencil(self) -> None:
        # S2-A4 red test. Production viscosity is nonzero, and the squid
        # band obstacle slabs leave only 1-3 cell wide water gaps behind the
        # membranes at the 2.5 mm grid: trilinear pressure is samplable
        # inside such a gap, but _sample_velocity_gradient requires complete
        # fluid neighbor pairs on all three axes, which a 1-cell gap can
        # never provide along the gap normal. The merged per-candidate gate
        # (weight AND gradient_valid) therefore rejects every candidate and
        # silently drops the O(1e3 Pa) pressure drive because the O(0.1 Pa)
        # viscous term is unsamplable - 4 orders of magnitude of signal lost
        # to its own gate. In far-pressure closure regions the pressure
        # "found" decision must survive an incomplete gradient stencil.
        #
        # 16^3 unit box: cell width 1/16 = 0.0625, centers z_c(k) = (k+0.5)/16.
        # x = y = 0.53125 is exactly cell-center column 8 (single-column
        # trilinear support, mirroring the extended-walk test). Water exists
        # only in the 1-cell slab z index 7 (obstacle below: 0..6, above:
        # 8..15). The marker sits one cell above the slab center, at
        # z = 0.53125 (center of obstacle cell 8), normal +z (air side up):
        #   outside (+n) candidates z = 0.59375..0.71875 are all obstacle ->
        #     never found on either semantics;
        #   inside (-n) 1.0x candidate z = 0.46875 = center of water cell 7 ->
        #     pressure weight 1.0 (samplable), but the gradient z-axis pair
        #     samples grid coordinates 7.0 and 8.0 and the 8.0 plane is fully
        #     obstacle -> stencil incomplete -> the merged gate rejects it;
        #   deeper candidates (1.5x..3x: z = 0.4375, 0.40625, 0.375, 0.34375)
        #     sit half or fully inside the lower obstacle block (pressure
        #     weight 0.5 then 0) and never complete a stencil either.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.53125),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 3.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, :7] = 1
        obstacle[:, :, 8:] = 1
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
            viscosity_pa_s=2.0,
            two_sided_pressure=True,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        self.assertEqual(report.closure_gradient_missing_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        # Pure pressure closure: (p_gap_water - p_far_air) * n = (3 - 10) * +z.
        # The viscous term contributes exactly zero because no complete
        # gradient stencil exists anywhere in the 1-cell gap (the unfound
        # sides keep zero gradient matrices).
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -7.0, delta=1.0e-4)

    def test_closure_sampling_view_with_viscosity_preserves_pressure_drive(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.53125),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 3.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, :7] = 1
        obstacle[:, :, 8:] = 1
        fluid.obstacle.from_numpy(obstacle)
        sampling_view = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        sampling_view.from_numpy(obstacle)

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
            viscosity_pa_s=2.0,
            two_sided_pressure=True,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        self.assertEqual(report.closure_gradient_missing_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -7.0, delta=1.0e-4)

    def test_closure_gradient_missing_is_zero_when_stencil_complete(self) -> None:
        # Guard for the S2-A4 decoupling: deep water must NOT report a
        # missing gradient. Same geometry as the basic closure test but with
        # nonzero viscosity (the _sample helper hardcodes viscosity 0.0, so
        # the call is written out in full). Paper check of the 8^3 fixture
        # (cell width 0.125, water cells z 0..3 i.e. 4 cells deep, marker at
        # z = 0.5, normal +z): the inside 1.0x candidate probes z = 0.375,
        # grid coordinate (4.5, 4.5, 2.5); the gradient axis pairs sample at
        # x in {4.0, 5.0}, y in {4.0, 5.0}, z in {2.0, 3.0}, and every
        # trilinear support cell of those six samples lies inside water
        # z 0..3 -> the stencil is complete, so pressure and gradient are
        # both found at the first candidate and nothing is "missing".
        markers, fluid = self._water_below_air_above_fixture()

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
            viscosity_pa_s=2.0,
            two_sided_pressure=True,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.closure_gradient_missing_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        # The velocity field is identically zero, so the (complete) gradient
        # is the zero matrix and the traction equals the inviscid closure
        # value (p_water - p_far_air) * n = (2 - 10) * +z.
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -8.0, delta=1.0e-4)

    def test_anchor_fallback_closes_fully_sealed_marker(self) -> None:
        # S2-A6 red test: a closure-region marker whose +/-n normal walks
        # are FULLY sealed (no standard candidate ever samples even a
        # pressure weight) must still close when the pressure-Neumann row
        # assembly has anchored it to a fluid cell that participates in
        # the pressure solve.
        #
        # 16^3 unit box, spacing 0.0625. Marker at x = y = 0.53125 (the
        # cell-center column i = j = 8) and z = 0.5 (face between cells 7
        # and 8), normal +z, so the normal-aligned probe distance is
        # exactly one cell width. Every cell is obstacle except the single
        # water cell (8, 8, 2), deliberately OFF the marker's normal line:
        #   outside (+n) candidates probe z grid coords {8.5, 9, 9.5, 10,
        #   10.5} -> supports within cells 8..11, all obstacle;
        #   inside (-n) candidates probe z grid coords {6.5, 6, 5.5, 5,
        #   4.5} -> supports within cells 4..7, all obstacle.
        # Both closure branches therefore miss and, without the anchor
        # fallback, the marker is invalid (asserted at the end with the
        # default flag). With use_pressure_anchor_fallback=True the
        # sampler must read the anchor cell-center pressure directly (no
        # interpolation), decide the water side from the anchor center's
        # normal projection (z_c(2) = 0.15625 < 0.5 -> water on -n), and
        # close with traction = (p_anchor - p_far) * n.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 5.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.ones((16, 16, 16), dtype=np.int32)
        obstacle[8, 8, 2] = 0
        fluid.obstacle.from_numpy(obstacle)
        # Pin the anchor directly (the assembly-capture path has its own
        # unit test below): marker 0 is anchored to the lone water cell.
        markers.marker_pressure_anchor_cell.from_numpy(
            np.array([[8, 8, 2]], dtype=np.int32)
        )

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            use_pressure_anchor_fallback=True,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_anchor_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        # traction = (p_anchor_water - p_far_air) * n = (5 - 10) * +z
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -5.0, delta=1.0e-4)

        # The anchor must never fire on its own: with the flag left at its
        # default (False) the same sealed marker stays invalid even though
        # the anchor field is populated.
        default_report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(default_report.valid_marker_count, 0)
        self.assertEqual(default_report.invalid_marker_count, 1)
        self.assertEqual(default_report.far_pressure_closed_marker_count, 0)
        self.assertEqual(
            default_report.far_pressure_anchor_closed_marker_count,
            0,
        )

    def test_pressure_neumann_row_assembly_records_marker_anchor_cell(self) -> None:
        # S2-A6 red test (capture side): the pressure-Neumann row assembly
        # must publish, for every marker that received at least one matrix
        # row, the (i, j, k) of the row-owning fluid cell, and the anchor
        # field must read (-1, -1, -1) before any assembly ran. Same
        # minimal 4^3 fixture as
        # test_pressure_neumann_reconstruction_enters_fv_cg_matrix_row:
        # exactly one row, owned by fluid cell (2, 2, 2).
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(7,),
        )
        anchor_before = tuple(
            int(markers.marker_pressure_anchor_cell[0][axis])
            for axis in range(3)
        )
        self.assertEqual(anchor_before, (-1, -1, -1))
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

        self.assertEqual(report.active_pressure_neumann_rows, 1)
        anchor = tuple(
            int(markers.marker_pressure_anchor_cell[0][axis])
            for axis in range(3)
        )
        self.assertGreaterEqual(anchor[0], 0)
        # The anchor is the row-owning fluid cell: the same cell whose
        # pressure_interface_coupling_active flag the assembly raised.
        self.assertEqual(anchor, (2, 2, 2))
        self.assertEqual(
            int(fluid.pressure_interface_coupling_active[2, 2, 2]),
            1,
        )

    def test_node_anchor_fallback_closes_sealed_marker_without_marker_anchor(
        self,
    ) -> None:
        # S2-A7 red test: the S2-A6 marker-level anchor is sourced from the
        # pressure-Neumann row assembly, but the 25-step production probe
        # measured pressure_neumann_active_rows == 0 (every near-boundary
        # fluid cell carries a velocity-Dirichlet row instead, which the
        # Neumann assembly skips), so that anchor source is structurally
        # EMPTY in the squid geometry. The node-level anchor field
        # node_anchor_cell on HibmMpmIbNodeSearch - populated by the
        # velocity-Dirichlet row assembly (row write-out + relocation
        # success paths) and by the interior-fluid-point prefill - must
        # close the marker instead.
        #
        # Same fully sealed 16^3 fixture as
        # test_anchor_fallback_closes_fully_sealed_marker: marker at the
        # cell-center column i = j = 8, z = 0.5 (face between cells 7 and
        # 8), normal +z; every cell obstacle except the lone water cell
        # (8, 8, 2) OFF the normal line, so both +/-n walks miss at every
        # reach. The marker-level anchor is left at its (-1, -1, -1)
        # sentinel. The sampler must take the 8 corner nodes of the
        # marker's cell base floor(grid_coord) = (8, 8, 7) (z-fastest
        # traversal, indices clamped to the node grid), find the manually
        # pinned node anchor (8, 8, 7) -> water cell (8, 8, 2), read that
        # cell-center pressure directly, orient the covariant formula from
        # the anchor center's normal projection (z_c(2) = 0.15625 < 0.5 ->
        # water on -n) and close with traction = (p_anchor - p_far) * n.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        # Precondition: the marker-level Neumann anchor stays unset, so
        # the S2-A6 first-stage fallback cannot fire and its counter must
        # stay 0 below.
        marker_anchor = tuple(
            int(markers.marker_pressure_anchor_cell[0][axis])
            for axis in range(3)
        )
        self.assertEqual(marker_anchor, (-1, -1, -1))
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 5.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.ones((16, 16, 16), dtype=np.int32)
        obstacle[8, 8, 2] = 0
        fluid.obstacle.from_numpy(obstacle)
        search = HibmMpmIbNodeSearch(
            grid_nodes=(16, 16, 16),
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        # Pin the node-level anchor directly (the assembly-capture path
        # has its own unit test below): only the corner node (8, 8, 7) of
        # the marker's cell points at the lone water cell. (8, 8, 7) is
        # reached from the nominal corner base (8, 8, 7) at offset
        # (0, 0, 0) and from a one-ulp-low base (7, 7, 7) at offset
        # (1, 1, 0), so the test is robust to f32 floor jitter.
        node_anchor = np.full((16, 16, 16, 3), -1, dtype=np.int32)
        node_anchor[8, 8, 7] = (8, 8, 2)
        search.node_anchor_cell.from_numpy(node_anchor)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            use_pressure_anchor_fallback=True,
            node_anchor_cell=search.node_anchor_cell,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.far_pressure_node_anchor_closed_marker_count, 1)
        # First-stage (marker-level) anchor never fired: its source is
        # empty here, exactly the S2-A7 production gap.
        self.assertEqual(report.far_pressure_anchor_closed_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        # traction = (p_anchor_water - p_far_air) * n = (5 - 10) * +z
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -5.0, delta=1.0e-4)

        # The node anchor must never fire on its own: with the runtime
        # switch left at its default (False) the same sealed marker stays
        # invalid even though the node anchor field is populated and
        # passed in.
        default_report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            node_anchor_cell=search.node_anchor_cell,
        )

        self.assertEqual(default_report.valid_marker_count, 0)
        self.assertEqual(default_report.invalid_marker_count, 1)
        self.assertEqual(default_report.far_pressure_closed_marker_count, 0)
        self.assertEqual(
            default_report.far_pressure_node_anchor_closed_marker_count,
            0,
        )
        self.assertEqual(
            default_report.far_pressure_anchor_closed_marker_count,
            0,
        )

    def test_velocity_dirichlet_row_assembly_records_node_anchor_cell(self) -> None:
        # S2-A7 red test (capture side): the velocity-Dirichlet
        # reconstructed row assembly must publish, for the IB node that
        # received a row, an interior-fluid anchor cell in
        # search.node_anchor_cell, and the field must read (-1, -1, -1)
        # before any assembly ran. Same minimal 4^3 fixture as
        # test_no_slip_dirichlet_rows_reconstruct_along_surface_normal:
        # exactly one reconstructed row at node (2, 2, 2).
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
        anchor_before = tuple(
            int(search.node_anchor_cell[2, 2, 2][axis]) for axis in range(3)
        )
        self.assertEqual(anchor_before, (-1, -1, -1))
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
        anchor = tuple(
            int(search.node_anchor_cell[2, 2, 2][axis]) for axis in range(3)
        )
        # The anchor is the containing cell of the row's interior fluid
        # sample: a non-obstacle cell that participates in the pressure
        # solve. (Exact cell index is not pinned: the sample sits on the
        # face between cells 2 and 3 along z, so f32 rounding may resolve
        # the containing cell to either side - both are fluid.)
        self.assertGreaterEqual(anchor[0], 0)
        self.assertGreaterEqual(anchor[1], 0)
        self.assertGreaterEqual(anchor[2], 0)
        self.assertLess(anchor[0], 4)
        self.assertLess(anchor[1], 4)
        self.assertLess(anchor[2], 4)
        self.assertEqual(int(fluid.obstacle[anchor[0], anchor[1], anchor[2]]), 0)
        # A node that never was an IB node keeps the sentinel after the
        # assembly-time reset + prefill + capture sequence.
        untouched = tuple(
            int(search.node_anchor_cell[0, 0, 0][axis]) for axis in range(3)
        )
        self.assertEqual(untouched, (-1, -1, -1))

    def test_velocity_dirichlet_row_assembly_preserves_external_boundary_rows(
        self,
    ) -> None:
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

        active = np.zeros((4, 4, 4), dtype=np.int32)
        values = np.zeros((4, 4, 4, 3), dtype=np.float32)
        weights = np.zeros((4, 4, 4), dtype=np.float32)
        marker_regions = np.full((4, 4, 4), -1, dtype=np.int32)
        active[0, 0, 3] = 1
        values[0, 0, 3] = (0.0, 0.0, -10.0)
        weights[0, 0, 3] = 1.0
        active[0, 0, 2] = 1
        values[0, 0, 2] = (9.0, 9.0, 9.0)
        weights[0, 0, 2] = 0.5
        marker_regions[0, 0, 2] = 7
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        fluid.velocity_dirichlet_boundary_marker_region_id.from_numpy(marker_regions)

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
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            marker_region_id=markers.region_id,
            primary_region_id=7,
        )

        active_after = fluid.velocity_dirichlet_boundary_active.to_numpy()
        values_after = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
        weights_after = fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        marker_regions_after = (
            fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
        )

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        self.assertEqual(int(active_after[0, 0, 3]), 1)
        np.testing.assert_allclose(values_after[0, 0, 3], (0.0, 0.0, -10.0))
        self.assertAlmostEqual(float(weights_after[0, 0, 3]), 1.0)
        self.assertEqual(int(marker_regions_after[0, 0, 3]), -1)
        self.assertEqual(int(active_after[0, 0, 2]), 0)
        np.testing.assert_allclose(values_after[0, 0, 2], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(float(weights_after[0, 0, 2]), 0.0)
        self.assertEqual(int(marker_regions_after[0, 0, 2]), -1)

    def test_sampling_view_lets_closure_fire_through_converted_sealed_water(
        self,
    ) -> None:
        # S2-A8'' red test (corrected A8 deadlock fixture). The full band
        # conversion is the CORRECT projection behavior (zero-correctable
        # cells are zero matrix rows; A8' interior-only measured a CG
        # residual floor of 0.518) - but the stress sampler shares the
        # projection's obstacle view, so the converted sealed water under
        # the membrane is invisible and the closure starves. The fix is a
        # dedicated sampling view: base geometry and the classified row
        # cloud envelope stay DRY (the A8 experiment proved opening the
        # envelope makes every marker sample zero-pressure dead water on
        # both sides, Delta-p = 0, drive dead), while the NONE-classified
        # converted cells become samplable water carrying their
        # back-filled pressure.
        #
        # 16^3 unit box, spacing 0.0625; marker at the cell-center column
        # i = j = 8 (x = y = 0.53125 -> single-column trilinear support),
        # z = 0.5 (the face between cells 7 and 8), normal +z, so the
        # normal-aligned probe distance is exactly one cell width.
        #
        # REAL obstacle view (the projection's view): every cell is
        # obstacle - membrane envelope and the sealed water below it were
        # all band-converted; the air side z >= 8 is base obstacle. The
        # control call without the sampling view must therefore walk
        # completely dry on both sides and stay invalid (the A8 deadlock).
        #
        # SAMPLING view (artificial field, passed directly; the
        # fluid-side builder has its own unit test in test_core_fluid):
        #   z <= 6 : 0 - converted sealed water, NONE-classified
        #   z == 7 : 1 - 1-cell membrane envelope below the marker
        #            (classified by the IB node search -> dry)
        #   z >= 8 : 1 - envelope above the marker + base air obstacle
        #
        # The sealed water carries the back-filled pressure 3.0. The
        # envelope cell z = 7 carries the poison value 100.0: if the view
        # wrongly admitted it, the first inside candidate (grid z = 6.5,
        # support {6: 0.5, 7: 0.5}) would read (0.5*3 + 0.5*100) / 1.0 =
        # 51.5 and the traction would flip to (51.5 - 10) * n = +41.5;
        # with the correct view the masked trilinear mean over water-only
        # corners is exactly 3.0.
        import taichi as ti

        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 3.0, dtype=np.float32)
        pressure[:, :, 7] = 100.0
        fluid.pressure.from_numpy(pressure)
        obstacle = np.ones((16, 16, 16), dtype=np.int32)
        fluid.obstacle.from_numpy(obstacle)
        sampling_view = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        view = np.ones((16, 16, 16), dtype=np.int32)
        view[:, :, :7] = 0
        sampling_view.from_numpy(view)

        # Control (the A8 deadlock): without the sampling view the walks
        # share the projection obstacle view and find nothing on either
        # side -> invalid, no closure.
        control_report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )
        self.assertEqual(control_report.valid_marker_count, 0)
        self.assertEqual(control_report.invalid_marker_count, 1)
        self.assertEqual(control_report.far_pressure_closed_marker_count, 0)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        self.assertEqual(report.far_pressure_anchor_closed_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        # Closure fires through the back-filled sealed water:
        # traction = (p_filled_water - p_far_air) * n = (3 - 10) * +z.
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -7.0, delta=1.0e-4)

    def test_sampling_view_default_is_bitwise_status_quo(self) -> None:
        # S2-A8'' default contract: omitting the new kwarg and passing
        # sampling_obstacle_field=None must be the same call - both bind
        # the never-indexed stand-in with the runtime gate off, so every
        # sample reads the projection obstacle view exactly as before.
        markers, fluid = self._water_below_air_above_fixture()

        baseline = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )
        baseline_traction = markers.marker_traction_pa(0)

        explicit_none = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=None,
        )
        explicit_none_traction = markers.marker_traction_pa(0)

        self.assertEqual(baseline, explicit_none)
        self.assertEqual(tuple(baseline_traction), tuple(explicit_none_traction))

    def test_assemble_orders_fill_and_view_between_projection_and_sampling(
        self,
    ) -> None:
        # S2-A8'' wiring contract on the assemble chain: the converted-
        # cell pressure fill and the sampling-view build run strictly
        # after the LAST fluid.project(...) (including the post-Dirichlet
        # consistency projection) and strictly before the stress
        # sampling, and only when the far-pressure closure is enabled
        # (far_pressure_region_id != -1). The default path must stay
        # bitwise-unchanged: no closure, no fill, no view.
        import inspect

        import simulation_core.hibm_mpm as hibm_mpm_module

        source = inspect.getsource(
            hibm_mpm_module.assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        )
        self.assertIn("fluid.fill_hibm_converted_cell_pressures(", source)
        self.assertIn("fluid.build_hibm_sampling_obstacle(", source)

        last_project = source.rindex("fluid.project(")
        fill_call = source.index("fluid.fill_hibm_converted_cell_pressures(")
        build_call = source.index("fluid.build_hibm_sampling_obstacle(")
        stress_call = source.index("sample_fluid_stress_to_marker_tractions(")
        self.assertLess(last_project, fill_call)
        self.assertLess(fill_call, build_call)
        self.assertLess(build_call, stress_call)

        # The fill block is gated on the closure opt-in.
        gate_window = source[max(0, fill_call - 600):fill_call]
        self.assertIn("far_pressure_region_id) != -1", gate_window)
        # The view build consumes the same frozen classification the band
        # conversion consumed.
        build_window = source[build_call:build_call + 400]
        self.assertIn("ib_search.node_kind_code", build_window)
        self.assertIn(
            "unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE",
            build_window,
        )
        # The stress sampling call receives the view (None when the
        # closure is disabled - the default bitwise path).
        stress_window = source[stress_call:stress_call + 2400]
        self.assertIn("sampling_obstacle_field=", stress_window)

    def test_closure_region_suppresses_spurious_outside_water(self) -> None:
        # S2-A9 red test. Production probe at the 2 s campaign: the main
        # membrane sinks ~17 mm as a whole and the vacated space above it
        # is WATER in the carve model (there is no air phase), so the
        # region-7 markers' outside (+n) walk samples spurious water
        # there. The genuine two-sided branch then takes over with
        # Delta-p = (real water - spurious water) ~ 0 and the closed
        # count collapses 7782 -> 0 intermittently, redirecting the
        # drive. The declared semantics of a far-pressure region is an
        # AIR-BACKED interface: the outside is ALWAYS the known far
        # pressure, regardless of what the geometry happens to sample
        # there, so for closure-region markers the closure branch must
        # take priority over the two-sided branch.
        #
        # Fixture: _water_below_air_above_fixture variant with water on
        # BOTH sides (whole domain fluid, no obstacle). 8^3 unit box,
        # cell width 0.125, centers z_c(k) = (k + 0.5) / 8: pressure 3.0
        # in the lower half (cells z 0..3) and 9.0 in the upper half
        # (cells z 4..7). Marker at z = 0.5 (the face between cells 3
        # and 4), normal +z, region 7, far pressure 10. Both walks find
        # water at the first 1.0x candidate: inside probe z = 0.375 ->
        # support cells z {2, 3} -> 3.0; outside probe z = 0.625 ->
        # support cells z {4, 5} -> 9.0.
        #
        # Status quo (two-sided wins): traction = (3 - 9) * +z = -6.
        # Declared closure semantics: outside := p_far = 10, traction =
        # (3 - 10) * +z = -7, the marker still counts as closed, and the
        # suppressed spurious outside water is directly observable in
        # the new far_pressure_outside_suppressed_marker_count.
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
        pressure = np.full((8, 8, 8), 3.0, dtype=np.float32)
        pressure[:, :, 4:] = 9.0
        fluid.pressure.from_numpy(pressure)
        # The whole domain is fluid: the obstacle field stays all zero.

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 1)
        self.assertEqual(
            report.far_pressure_outside_suppressed_marker_count,
            1,
        )
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        self.assertEqual(report.closure_gradient_missing_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # Declared closure: (p_inside_water - p_far_air) * n =
        # (3 - 10) * +z = -7, NOT the spurious two-sided
        # (3 - 9) * +z = -6.
        self.assertAlmostEqual(traction[2], -7.0, delta=1.0e-4)

        # Contrast leg: the same both-sides-water fixture with the
        # marker in a NON-closure region (marker region 8, far region 7)
        # must keep the genuine two-sided branch bit for bit:
        # traction = (3 - 9) * +z = -6, no closure, no suppression.
        contrast_markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        contrast_markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(8,),
        )

        contrast_report = self._sample(
            contrast_markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
        )

        self.assertEqual(contrast_report.valid_marker_count, 1)
        self.assertEqual(contrast_report.invalid_marker_count, 0)
        self.assertEqual(contrast_report.two_sided_pressure_marker_count, 1)
        self.assertEqual(contrast_report.far_pressure_closed_marker_count, 0)
        self.assertEqual(
            contrast_report.far_pressure_outside_suppressed_marker_count,
            0,
        )
        contrast_traction = contrast_markers.marker_traction_pa(0)
        self.assertAlmostEqual(contrast_traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(contrast_traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(contrast_traction[2], -6.0, delta=1.0e-4)

    def test_suppression_keeps_mirrored_closure_for_inside_dry(self) -> None:
        # S2-A9 guard: the declared closure priority must NOT over-fire
        # on mirrored-orientation markers. Mirrored fixture (water ONLY
        # above the marker plane, structurally dry below): 8^3 unit box,
        # pressure 9.0 everywhere, obstacle in cells z 0..3. Marker at
        # z = 0.5, normal +z, region 7, far pressure 10. The inside (-n)
        # candidates (probe z = 0.375, 0.3125, 0.25, 0.1875, 0.125 ->
        # supports within cells 0..3) are all obstacle and never find
        # water; the outside (+n) walk finds the REAL water at the first
        # 1.0x candidate (probe z = 0.625 -> support cells z {4, 5},
        # both fluid at 9.0).
        #
        # The closure-priority branch keys on inside_pressure_found == 1
        # and must not swallow this marker; the mirrored closure stays
        # in charge and substitutes the far pressure on the dry INSIDE.
        # Paper check of the mirrored body:
        #   traction = (p_inside - p_outside) * n
        #            = (p_far_air - p_water_above) * n
        #            = (10 - 9) * +z = +1.
        # The outside water is the genuine water side of the interface
        # here, not spurious, so the suppressed counter must stay 0.
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
        pressure = np.full((8, 8, 8), 9.0, dtype=np.float32)
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
        self.assertEqual(
            report.far_pressure_outside_suppressed_marker_count,
            0,
        )
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # Mirrored closure: (10 - 9) * +z = +1.
        self.assertAlmostEqual(traction[2], 1.0, delta=1.0e-4)


class HibmSolidBandPopulationSplitContractTests(unittest.TestCase):
    """S2-A8' contracts on the hibm_mpm side of the band split.

    The assemble chain must feed the IB node-search classification to
    every solid-band call so the env-gated interior-only mode can split
    membrane-interior slivers (classified cells, convert) from enclosed
    water (unclassified cells, stay active for the per-component
    anchoring chain), and the reports must expose both populations.
    """

    def test_sharp_reports_carry_band_population_split_fields(self) -> None:
        import dataclasses

        from simulation_core.hibm_mpm import (
            HibmMpmSharpFluidToMpmLoadReport,
            HibmMpmSharpMpmStepReport,
        )

        load_fields = {
            field.name: field
            for field in dataclasses.fields(HibmMpmSharpFluidToMpmLoadReport)
        }
        self.assertIn("solid_band_interior_cell_count", load_fields)
        self.assertIn("solid_band_enclosed_water_cell_count", load_fields)
        self.assertIn(
            "solid_band_velocity_dirichlet_protected_cell_count",
            load_fields,
        )
        self.assertIn("solid_band_mask_protected_cell_count", load_fields)
        self.assertIn("row_cloud_orphan_cell_count", load_fields)
        self.assertIn("overflow_singleton_cleanup_cell_count", load_fields)
        self.assertIn("overflow_singleton_cleanup_component_count", load_fields)
        self.assertIn("air_backed_reachability_barrier_cell_count", load_fields)
        # -1 means "the band ran without a classification split"; the
        # default keeps legacy constructions honest instead of reporting
        # a misleading zero population.
        self.assertEqual(load_fields["solid_band_interior_cell_count"].default, -1)
        self.assertEqual(
            load_fields["solid_band_enclosed_water_cell_count"].default, -1
        )
        self.assertEqual(
            load_fields[
                "solid_band_velocity_dirichlet_protected_cell_count"
            ].default,
            -1,
        )
        self.assertEqual(
            load_fields["solid_band_mask_protected_cell_count"].default,
            -1,
        )
        self.assertEqual(
            load_fields["air_backed_reachability_barrier_cell_count"].default,
            -1,
        )

        step_fields = {
            field.name: field
            for field in dataclasses.fields(HibmMpmSharpMpmStepReport)
        }
        self.assertIn("next_solid_band_interior_cell_count", step_fields)
        self.assertIn("next_solid_band_enclosed_water_cell_count", step_fields)
        self.assertIn(
            "next_solid_band_velocity_dirichlet_protected_cell_count",
            step_fields,
        )
        self.assertIn("next_solid_band_mask_protected_cell_count", step_fields)
        self.assertIn("next_row_cloud_orphan_cell_count", step_fields)
        self.assertIn("next_overflow_singleton_cleanup_cell_count", step_fields)
        self.assertIn(
            "next_overflow_singleton_cleanup_component_count",
            step_fields,
        )
        self.assertIn("post_solid_kinematic_projection_applied", step_fields)
        self.assertIn("post_solid_fluid_projection", step_fields)
        self.assertIn("post_solid_no_slip_residual", step_fields)
        self.assertEqual(
            step_fields["next_solid_band_interior_cell_count"].default, -1
        )
        self.assertEqual(
            step_fields["next_solid_band_enclosed_water_cell_count"].default, -1
        )
        self.assertEqual(
            step_fields[
                "next_solid_band_velocity_dirichlet_protected_cell_count"
            ].default,
            -1,
        )
        self.assertEqual(
            step_fields["next_solid_band_mask_protected_cell_count"].default,
            -1,
        )
        self.assertFalse(
            step_fields["post_solid_kinematic_projection_applied"].default
        )
        self.assertIsNone(step_fields["post_solid_fluid_projection"].default)
        self.assertIsNone(step_fields["post_solid_no_slip_residual"].default)

    def test_pressure_disconnected_reports_carry_component_distribution(self) -> None:
        import dataclasses

        from simulation_core.hibm_mpm import HibmMpmPressureDisconnectedRegionReport

        fields = {
            field.name: field
            for field in dataclasses.fields(HibmMpmPressureDisconnectedRegionReport)
        }

        self.assertIn("component_raw_count", fields)
        self.assertIn("largest_component_cell_count", fields)
        self.assertIn("singleton_component_count", fields)
        self.assertIn("small_component_threshold_cells", fields)
        self.assertIn("small_component_count", fields)
        self.assertIn("small_component_cell_count", fields)
        self.assertEqual(fields["component_raw_count"].default, 0)
        self.assertEqual(fields["largest_component_cell_count"].default, 0)
        self.assertEqual(fields["singleton_component_count"].default, 0)
        self.assertEqual(fields["small_component_threshold_cells"].default, 128)
        self.assertEqual(fields["small_component_count"].default, 0)
        self.assertEqual(fields["small_component_cell_count"].default, 0)

    def test_every_band_call_site_passes_node_classification(self) -> None:
        import inspect

        import simulation_core.hibm_mpm as hibm_mpm_module

        source = inspect.getsource(hibm_mpm_module)
        fragments = source.split(
            "fluid.mark_hibm_solid_band_nonprojectable_cells("
        )[1:]
        # The assemble band loop plus the two post-step band calls.
        self.assertGreaterEqual(len(fragments), 3)
        for fragment in fragments:
            window = fragment[:700]
            self.assertIn("node_kind_code=ib_search.node_kind_code", window)
            self.assertIn(
                "unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE",
                window,
            )
            self.assertIn(
                "protect_solid_band_mask=True",
                window,
            )
            self.assertIn(
                "protect_velocity_dirichlet_radius_cells=0",
                window,
            )
            self.assertIn(
                "protect_unstamped_velocity_dirichlet_rows=True",
                window,
            )

    def test_step_summary_exposes_band_population_split_counts(self) -> None:
        import inspect

        import simulation_core.hibm_mpm as hibm_mpm_module

        source = inspect.getsource(hibm_mpm_module)
        self.assertIn('"hibm_solid_band_interior_cell_count"', source)
        self.assertIn('"hibm_solid_band_enclosed_water_cell_count"', source)
        self.assertIn(
            '"hibm_solid_band_velocity_dirichlet_protected_cell_count"',
            source,
        )
        self.assertIn('"hibm_solid_band_mask_protected_cell_count"', source)
        self.assertIn('"hibm_next_solid_band_interior_cell_count"', source)
        self.assertIn('"hibm_next_solid_band_enclosed_water_cell_count"', source)
        self.assertIn(
            '"hibm_next_solid_band_velocity_dirichlet_protected_cell_count"',
            source,
        )
        self.assertIn('"hibm_next_solid_band_mask_protected_cell_count"', source)
        self.assertIn('"hibm_post_solid_kinematic_projection_applied"', source)
        self.assertIn('"hibm_post_solid_interior_divergence_l2"', source)
        self.assertIn('"hibm_post_solid_no_slip_residual_l2_mps"', source)
        self.assertIn(
            '"hibm_projection_stage": "post_solid_kinematic_consistency"',
            source,
        )


class HibmMpmTwoSidedExtendedWalkTests(unittest.TestCase):
    """S2-A10 red tests: closure-style extended walk for two-sided markers.

    The S2-A8'' dedicated sampling view (base geometry UNION row-cloud
    envelope) starves genuinely thin features: a thin tail fin sits entirely
    inside its own row-cloud envelope, so BOTH standard two-sided walks
    (max 3.0x local spacing) run dry and the marker silently drops to zero
    traction (per-step two-sided valid population 171-1017 avg 210 before
    A8'' -> ~0 after; the case check tail_marker_participates flipped
    True -> False at exactly that change). The fix is opt-in: a new
    two_sided_probe_max_multiplier (default 3.0 = bitwise status quo,
    activation gate strictly > 3.0 mirroring the closure multiplier) lets
    NON-closure markers whose standard ladder found nothing on EITHER side
    re-walk each still-missing side out to the requested multiplier with the same
    5-candidate ladder spacing, the same sampling view, and a per-side
    H1-type crossing guard (never tunnel through projection-view solid into
    another compartment's water).
    """

    def _thin_slab_fixture(self, *, wall_inside: bool):
        # 16^3 unit box: cell width 1/16 = 0.0625, centers z_c(k) = (k+0.5)/16.
        # Marker x = y = 0.53125 is exactly cell-center column 8 (single-column
        # trilinear support); z = 0.5 is the face between cells 7 and 8 (grid
        # z-coordinate 7.5, k_near = 8), normal +z, so the normal-aligned
        # probe distance is exactly one cell width (0.0625). The marker's
        # region is 3 while the run arms far_pressure_region_id=7, proving
        # the new walk keys on the closure_region_marker conjunct being
        # FALSE, not merely on the closure being disabled.
        #
        # Sampling view (the dedicated A8'' view, hand-built): z 4..11 dry -
        # the thin feature's own row-cloud envelope, covering the standard
        # ladder's full reach on both sides plus margin:
        #   outside (+n) standard candidates z = 0.5 + {1.0, 1.5, 2.0, 2.5,
        #   3.0} * 0.0625 -> grid z {8.5, 9.0, 9.5, 10.0, 10.5} -> supports
        #   {8,9}, {9}, {9,10}, {10}, {10,11};
        #   inside (-n) standard candidates -> grid z {6.5, 6.0, 5.5, 5.0,
        #   4.5} -> supports {6,7}, {6}, {5,6}, {5}, {4,5};
        # every support cell lies in the dry band 4..11, so the standard walk
        # finds NOTHING on either side and the marker is invalid today.
        #
        # Genuine water sits just beyond the envelope on both sides:
        #   z >= 12 at pressure 5.0 (outside) and z <= 3 at pressure 2.0
        #   (inside) - first reachable at 4.8x (the first extended rung of
        #   the 5-candidate ladder 3 + (12-3)*(r+1)/5 = {4.8, 6.6, 8.4,
        #   10.2, 12.0}):
        #   outside rung 0: z = 0.5 + 4.8*0.0625 = 0.8 -> grid z 12.3 ->
        #     support {12: 0.7, 13: 0.3}, all water -> pressure 5.0;
        #   inside rung 0: z = 0.5 - 0.3 = 0.2 -> grid z 2.7 -> support
        #     {2: 0.3, 3: 0.7}, all water -> pressure 2.0.
        #
        # wall_inside=True adds the H1 mirror geometry: a thin BASE-solid
        # wall in cells z {2, 3} (mirrored into the sampling view, since the
        # view is base UNION envelope) with different-pressure water behind
        # it in cells z {0, 1} at 7.0. The inside extension's rung 0 nearest
        # cell is floor(2.7 + 0.5) = 3 = wall -> the per-side crossing guard
        # latches; rung 1 (z = 0.5 - 6.6*0.0625 = 0.0875 -> grid z 0.9 ->
        # support {0: 0.1, 1: 0.9}, weight 1.0, pressure 7.0) WOULD close
        # the marker with the tunneled spurious traction (7 - 5) * n =
        # (0, 0, +2) if the guard were missing.
        import taichi as ti

        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(3,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 2.0, dtype=np.float32)
        pressure[:, :, 12:] = 5.0
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        view = np.zeros((16, 16, 16), dtype=np.int32)
        view[:, :, 4:12] = 1
        if wall_inside:
            pressure[:, :, :2] = 7.0
            obstacle[:, :, 2:4] = 1
            view[:, :, 2:4] = 1
        fluid.pressure.from_numpy(pressure)
        fluid.obstacle.from_numpy(obstacle)
        sampling_view = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        sampling_view.from_numpy(view)
        return markers, fluid, sampling_view

    def _one_standard_side_one_extended_side_fixture(self):
        import taichi as ti

        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(3,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 2.0, dtype=np.float32)
        pressure[:, :, 13:] = 5.0
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 8:13] = 1
        view = np.zeros((16, 16, 16), dtype=np.int32)
        view[:, :, 8:13] = 1
        fluid.pressure.from_numpy(pressure)
        fluid.obstacle.from_numpy(obstacle)
        sampling_view = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        sampling_view.from_numpy(view)
        return markers, fluid, sampling_view

    def _sample(self, markers, fluid, **kwargs):
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
            **kwargs,
        )

    def test_thin_slab_marker_closes_two_sided_via_extended_walk(self) -> None:
        # Red fixture (a). Both standard walks are fully inside the sampling
        # envelope (see _thin_slab_fixture paper walk); genuine water sits at
        # 4.8x on both sides at different pressures. With the extension armed
        # at 12.0 the marker must become a VALID two-sided sample through the
        # EXISTING covariant branch:
        #   traction = (p_inside - p_outside) * n = (2.0 - 5.0) * (0, 0, 1)
        #            = (0, 0, -3.0),
        # and the new counter must report exactly this one marker (it is a
        # subset of two_sided_pressure_marker_count, the family convention of
        # far_pressure_closed_extended ⊂ far_pressure_closed).
        markers, fluid, sampling_view = self._thin_slab_fixture(wall_inside=False)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
            two_sided_probe_max_multiplier=12.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 1)
        self.assertEqual(report.two_sided_extended_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_extended_marker_count, 0)
        self.assertEqual(report.far_pressure_outside_suppressed_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # traction = (p_inside - p_outside) * n = (2 - 5) * +z
        self.assertAlmostEqual(traction[2], -3.0, delta=1.0e-4)

    def test_extended_walk_closes_when_only_one_side_is_missing(self) -> None:
        markers, fluid, sampling_view = (
            self._one_standard_side_one_extended_side_fixture()
        )

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
            two_sided_probe_max_multiplier=12.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 1)
        self.assertEqual(report.two_sided_extended_marker_count, 1)
        diagnostic = report.marker_diagnostics[0]
        self.assertTrue(diagnostic["inside_pressure_found"])
        self.assertTrue(diagnostic["outside_pressure_found"])
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -3.0, delta=1.0e-4)

    def test_extended_walk_must_not_tunnel_through_thin_solid_wall(self) -> None:
        # Red fixture (b), the H1 mirror. Same thin-slab geometry, but the
        # inside (-n) far water (7.0 in cells z {0, 1}) is only reachable by
        # crossing the BASE-solid wall in cells z {2, 3}: the inside
        # extension's rung 0 (grid z 2.7) has nearest cell 3 = wall, latching
        # the per-side crossing guard, so the genuinely wet rung 1 (grid z
        # 0.9, weight 1.0, pressure 7.0) must NOT be accepted - without the
        # guard the marker would close with the spurious tunneled traction
        # (7 - 5) * n = (0, 0, +2). The outside extension still finds its
        # genuine 5.0 water, so the marker ends ONE-sided and must stay
        # invalid with zero traction, exactly today's behavior.
        markers, fluid, sampling_view = self._thin_slab_fixture(wall_inside=True)

        report = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
            two_sided_probe_max_multiplier=12.0,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.two_sided_extended_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)
        self.assertEqual(report.max_abs_traction_pa, 0.0)
        traction = markers.marker_traction_pa(0)
        self.assertEqual(tuple(traction), (0.0, 0.0, 0.0))

    def test_default_multiplier_keeps_thin_slab_marker_invalid_as_today(
        self,
    ) -> None:
        # Red fixture (c), backward compatibility. Same thin-slab fixture,
        # multiplier left at its default (omitted) and then passed explicitly
        # as 3.0: the activation gate is strictly > 3.0 (mirroring the
        # closure multiplier), so the extension must never run, the marker
        # stays invalid exactly as today (both standard walks dry inside the
        # envelope), the new counter stays 0, and the two calls produce
        # bitwise-identical reports.
        markers, fluid, sampling_view = self._thin_slab_fixture(wall_inside=False)

        omitted = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
        )

        self.assertEqual(omitted.valid_marker_count, 0)
        self.assertEqual(omitted.invalid_marker_count, 1)
        self.assertEqual(omitted.two_sided_pressure_marker_count, 0)
        self.assertEqual(omitted.two_sided_extended_marker_count, 0)
        self.assertEqual(omitted.max_abs_traction_pa, 0.0)
        diagnostic = omitted.marker_diagnostics[0]
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(
            diagnostic["invalid_reason"],
            "two_sided_pressure_missing",
        )
        self.assertFalse(diagnostic["base_pressure_found"])
        self.assertFalse(diagnostic["inside_pressure_found"])
        self.assertFalse(diagnostic["outside_pressure_found"])
        self.assertEqual(diagnostic["normal"], [0.0, 0.0, 1.0])
        omitted_traction = markers.marker_traction_pa(0)
        self.assertEqual(tuple(omitted_traction), (0.0, 0.0, 0.0))

        explicit_default = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            sampling_obstacle_field=sampling_view,
            two_sided_probe_max_multiplier=3.0,
        )
        explicit_traction = markers.marker_traction_pa(0)

        self.assertEqual(omitted, explicit_default)
        self.assertEqual(tuple(omitted_traction), tuple(explicit_traction))

    def test_closure_region_marker_is_unaffected_by_two_sided_multiplier(
        self,
    ) -> None:
        # Red fixture (d), non-interference. The thick-band closure-extension
        # fixture of test_extended_inside_walk_closes_marker_behind_thick_band
        # verbatim (16^3, spacing 0.0625; marker at the cell-center column
        # i = j = 8, z = 0.5 face, normal +z, REGION 7 = the closure region;
        # obstacle z 8..15 = air side, z 4..7 = band; water z 0..3 at 2.0;
        # far pressure 10.0). With the closure multiplier at 6.0 the closure
        # extension closes the marker at its first extended rung 3.6x
        # (z = 0.275, grid z 3.9, support {3: 0.1, 4: 0.9}, masked fluid
        # weight 0.1 -> pressure 2.0) with
        #   traction = (p_inside_water - p_far_air) * n = (2 - 10) * +z
        #            = (0, 0, -8).
        # Arming two_sided_probe_max_multiplier=12.0 on top must change
        # NOTHING: the A10 walk requires closure_region_marker == 0, so the
        # closure branch keeps absolute priority - the two reports must be
        # equal field for field (closure counters unchanged) and the new
        # counter must stay 0 in both.
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(16, 16, 16),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((16, 16, 16), 2.0, dtype=np.float32)
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 8:] = 1
        obstacle[:, :, 4:8] = 1
        fluid.obstacle.from_numpy(obstacle)

        baseline = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_inside_probe_max_multiplier=6.0,
        )
        baseline_traction = markers.marker_traction_pa(0)

        extended = self._sample(
            markers,
            fluid,
            far_pressure_region_id=7,
            far_pressure_pa=10.0,
            far_pressure_inside_probe_max_multiplier=6.0,
            two_sided_probe_max_multiplier=12.0,
        )
        extended_traction = markers.marker_traction_pa(0)

        self.assertEqual(baseline, extended)
        self.assertEqual(tuple(baseline_traction), tuple(extended_traction))
        self.assertEqual(extended.valid_marker_count, 1)
        self.assertEqual(extended.far_pressure_closed_marker_count, 1)
        self.assertEqual(extended.far_pressure_closed_extended_marker_count, 1)
        self.assertEqual(extended.two_sided_pressure_marker_count, 0)
        self.assertEqual(extended.two_sided_extended_marker_count, 0)
        self.assertAlmostEqual(extended_traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(extended_traction[1], 0.0, delta=1.0e-4)
        # Mirrors the closure-extension test: (2 - 10) * +z, untouched by
        # the two-sided multiplier.
        self.assertAlmostEqual(extended_traction[2], -8.0, delta=1.0e-4)


class HibmMpmOneSidedPressureInterfaceTests(unittest.TestCase):
    """Generic dry-side reference traction for thin FSI interfaces.

    This is the squid tail failure in solver terms: the marker is not in the
    declared main-membrane closure region, one probe side reaches water, and
    the other side is structurally dry. The result must be a computed fluid
    traction, not a case-layer force assignment.
    """

    def _one_sided_outside_fixture(self):
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(8,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((8, 8, 8), 0.0, dtype=np.float32)
        pressure[:, :, 4:] = 5.0
        fluid.pressure.from_numpy(pressure)
        obstacle = np.zeros((8, 8, 8), dtype=np.int32)
        obstacle[:, :, :4] = 1
        fluid.obstacle.from_numpy(obstacle)
        return markers, fluid

    def _one_sided_inside_with_spurious_outside_fixture(self):
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(8,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.full((8, 8, 8), 1.0, dtype=np.float32)
        pressure[:, :, :4] = 5.0
        fluid.pressure.from_numpy(pressure)
        return markers, fluid

    def _sample(self, markers, fluid, viscosity_pa_s=0.0, **kwargs):
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
            viscosity_pa_s=viscosity_pa_s,
            two_sided_pressure=True,
            **kwargs,
        )

    def test_one_sided_policy_closes_nonclosure_marker_with_computed_fluid_traction(
        self,
    ) -> None:
        markers, fluid = self._one_sided_outside_fixture()

        report = self._sample(
            markers,
            fluid,
            one_sided_pressure_region_id=8,
            one_sided_reference_pressure_pa=0.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.one_sided_pressure_marker_count, 1)
        self.assertEqual(report.one_sided_extended_marker_count, 0)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        # Water is on +n at 5 Pa and the dry side is 0 Pa:
        # traction = (p_dry_inside - p_outside_water) * n = -5 * +z.
        self.assertAlmostEqual(traction[2], -5.0, delta=1.0e-4)

        baseline = self._sample(
            markers,
            fluid,
        )

        self.assertEqual(baseline.valid_marker_count, 0)
        self.assertEqual(baseline.invalid_marker_count, 1)
        self.assertEqual(markers.marker_traction_pa(0), (0.0, 0.0, 0.0))

    def test_one_sided_policy_closes_viscous_split_path_marker(self) -> None:
        markers, fluid = self._one_sided_outside_fixture()

        report = self._sample(
            markers,
            fluid,
            viscosity_pa_s=2.0,
            one_sided_pressure_region_id=8,
            one_sided_reference_pressure_pa=0.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.one_sided_pressure_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -5.0, delta=1.0e-4)

    def test_one_sided_policy_suppresses_spurious_water_on_declared_dry_side(
        self,
    ) -> None:
        markers, fluid = self._one_sided_inside_with_spurious_outside_fixture()

        report = self._sample(
            markers,
            fluid,
            one_sided_pressure_region_id=8,
            one_sided_reference_pressure_pa=0.0,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.two_sided_pressure_marker_count, 0)
        self.assertEqual(report.one_sided_pressure_marker_count, 1)
        traction = markers.marker_traction_pa(0)
        # The declared one-sided region treats the +n side as structurally dry
        # even if the sampling view sees spurious water there:
        # traction = (p_inside_water - p_dry_outside) * n = +5 * +z.
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], 5.0, delta=1.0e-4)

    def test_one_sided_policy_uses_pressure_anchor_when_sampling_walks_miss(
        self,
    ) -> None:
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(8,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(8, 8, 8),
                dt_s=1.0e-3,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        pressure = np.zeros((8, 8, 8), dtype=np.float32)
        pressure[4, 4, 4] = 7.0
        fluid.pressure.from_numpy(pressure)
        sampling_obstacle = ti.field(dtype=ti.i32, shape=(8, 8, 8))
        sampling_obstacle.from_numpy(np.ones((8, 8, 8), dtype=np.int32))
        markers.marker_pressure_anchor_cell[0] = (4, 4, 4)

        report = self._sample(
            markers,
            fluid,
            one_sided_pressure_region_id=8,
            one_sided_reference_pressure_pa=0.0,
            use_pressure_anchor_fallback=True,
            sampling_obstacle_field=sampling_obstacle,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.one_sided_pressure_marker_count, 1)
        self.assertEqual(report.far_pressure_closed_marker_count, 0)
        traction = markers.marker_traction_pa(0)
        self.assertAlmostEqual(traction[0], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[1], 0.0, delta=1.0e-4)
        self.assertAlmostEqual(traction[2], -7.0, delta=1.0e-4)


class HibmMpmFarPressureAirBackedTests(unittest.TestCase):
    """S2-A12: fluid-side air zone for declared air-backed closure regions.

    Forensic anchor (run_2s_20260613b): the carve model's air chamber above
    the main membrane is fake incompressible water, flood-disconnected from
    the outlet from step 1 (unreach_n 7654); the anchored projection parks
    the membrane's vacated-volume debt there (unreached_divergence_l2 0.04
    -> 0.7 over 1000 steps, excluded from the interior guard), and the
    step-1015 band-flicker reconnection dumped it into the correctable
    interior (interior_divergence_l2 9.4e-7 -> 2.8e-2 in 3 steps, guard
    kill). A12 classifies the enclosed far-side pocket per step (closure
    markers seed flood-unreached components on their +n side) and converts
    it to obstacle-like air cells stamped with p_far.

    Shared fixture: 16^3 unit box (spacing h = 1/16 = 0.0625), full
    obstacle slab at z = 8, pressure outlet at z-min. Below the slab
    (z 0..7) is outlet-connected water; above it (z 9..15) is a sealed
    pocket of 16*16*7 = 1792 cells with exactly one flood component and
    volume 1792 * 0.0625^3 = 0.4375 m^3 (binary-exact). The closure marker
    (region 7) sits at the cell-center column i = j = 8 (x = y = 0.53125)
    on the slab's upper face z = 9/16 = 0.5625 with normal +z, so its
    far-side ladder rung 0 (1.0x, probe distance = h) lands at z = 0.625 =
    grid coordinate 9.5 -> nearest cell (8, 8, 10): active, unreached,
    labeled -> seed.
    """

    def _solver(self):
        return CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(16, 16, 16), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    def test_post_step_next_rebuild_applies_air_backed_conversion(self) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")
        next_rebuild = source.split("next_internal_obstacle_cell_count =", 1)[1]
        next_rebuild = next_rebuild.split("next_pressure_report =", 1)[0]

        self.assertIn("use_next_air_backed_reachability_barrier", next_rebuild)
        self.assertIn("write_region_pressure_reachability_barrier", next_rebuild)
        self.assertIn(
            "use_existing_reachability_barrier=use_next_air_backed_reachability_barrier",
            next_rebuild,
        )
        self.assertIn("mark_far_pressure_air_backed_seed_components", next_rebuild)
        self.assertIn("convert_hibm_air_backed_cells", next_rebuild)
        convert_call = next_rebuild.index(
            "next_air_backed_cell_count = fluid.convert_hibm_air_backed_cells()"
        )
        region_report_call = next_rebuild.index(
            "next_pressure_disconnected_region ="
        )
        post_air_conversion = next_rebuild[convert_call:region_report_call]
        self.assertIn(
            "fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells",
            post_air_conversion,
        )
        self.assertIn(
            "convert_next_projection_topology_cleanup_until_saturated()",
            post_air_conversion,
        )
        self.assertNotIn(
            "Refresh the post-step rows and reachability after air-backed conversion",
            post_air_conversion,
        )

    def test_projection_topology_cleanup_starts_with_row_cloud_sweep(self) -> None:
        source = HIBM_MPM_CORE_SOURCE.read_text(encoding="utf-8")
        fluid_cleanup = source.split(
            "def convert_projection_topology_cleanup_until_saturated() -> None:",
            1,
        )[1].split('_debug_stage_progress("fluid_substeps:start")', 1)[0]
        post_solid_cleanup = source.split(
            "def convert_next_projection_topology_cleanup_until_saturated() -> None:",
            1,
        )[1].split(
            "if (\n        int(next_solid_band_nonprojectable_cell_count) > 0",
            1,
        )[0]

        self.assertLess(
            fluid_cleanup.index("convert_row_cloud_orphans_until_saturated()"),
            fluid_cleanup.index("convert_overflow_singletons_without_row_reload()"),
        )
        self.assertLess(
            post_solid_cleanup.index(
                "convert_next_row_cloud_orphans_until_saturated()"
            ),
            post_solid_cleanup.index(
                "convert_next_overflow_singletons_without_row_reload()"
            ),
        )

    def _sealed_chamber_fixture(self):
        fluid = self._solver()
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 8] = 1
        fluid.obstacle.from_numpy(obstacle)
        # The seed walk's crossing guard reads the base snapshot; in
        # production apply_hibm_internal_obstacles takes it before any
        # band/air conversion - unit fixtures must do it explicitly.
        fluid.snapshot_hibm_base_obstacle()
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5625),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        return fluid, markers

    def _seed(
        self,
        markers,
        fluid,
        *,
        multiplier=12.0,
        region_id=7,
        probe_normal_sign=0.0,
        fallback_to_bidirectional_if_all_missed=False,
    ):
        return markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=region_id,
            far_pressure_inside_probe_max_multiplier=multiplier,
            far_pressure_air_backed_probe_normal_sign=probe_normal_sign,
            fallback_to_bidirectional_if_all_missed=(
                fallback_to_bidirectional_if_all_missed
            ),
        )

    def test_air_seed_selects_far_side_unreached_component_and_converts(
        self,
    ) -> None:
        # (a) The happy path, paper-walked: flood finds the 1792-cell
        # pocket as ONE component; the marker's rung-0 far probe lands in
        # cell (8, 8, 10) and selects it; conversion turns exactly the
        # 1792 pocket cells into obstacle-like air cells (band write set +
        # air tag) with volume 0.4375 m^3; the p_far stamp then writes the
        # declared chamber pressure for the sampling view; a re-flood sees
        # the pocket as wall -> unreached drops to ZERO (the debt channel
        # is structurally gone). Outlet-side water is untouched.
        fluid, markers = self._sealed_chamber_fixture()
        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 1792)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_count, 1)

        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (1, 0))

        converted = fluid.convert_hibm_air_backed_cells()

        self.assertEqual(converted, 1792)
        self.assertEqual(fluid.last_hibm_air_backed_cell_count, 1792)
        self.assertEqual(fluid.last_hibm_air_backed_component_count, 1)
        self.assertAlmostEqual(
            fluid.last_hibm_air_backed_cell_volume_m3,
            0.4375,
            delta=1.0e-9,
        )
        self.assertEqual(int(fluid.obstacle[8, 8, 12]), 1)
        self.assertEqual(int(fluid.hibm_air_cell[8, 8, 12]), 1)
        self.assertEqual(int(fluid.hibm_air_cell[8, 8, 4]), 0)
        self.assertEqual(int(fluid.obstacle[8, 8, 4]), 0)
        self.assertEqual(
            tuple(float(fluid.velocity[8, 8, 12][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )

        fluid.write_hibm_air_backed_cell_pressures(123.0)
        self.assertAlmostEqual(float(fluid.pressure[8, 8, 12]), 123.0, delta=1.0e-6)
        self.assertAlmostEqual(float(fluid.pressure[8, 8, 4]), 0.0, delta=1.0e-6)

        unreached_after = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )
        self.assertEqual(unreached_after, 0)

    def test_air_seed_can_be_restricted_to_configured_normal_side(self) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(
            self._seed(markers, fluid, probe_normal_sign=1.0),
            (1, 0),
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(
            self._seed(markers, fluid, probe_normal_sign=-1.0),
            (0, 1),
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 0)

    def test_air_seed_falls_back_to_bidirectional_when_pinned_side_all_misses(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(
            self._seed(
                markers,
                fluid,
                probe_normal_sign=-1.0,
                fallback_to_bidirectional_if_all_missed=True,
            ),
            (1, 0),
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_falls_back_to_region_adjacent_unreached_component(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 12] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 12] = 0

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=-1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
        )

        self.assertEqual((seeded, missed), (0, 1))
        self.assertGreater(
            int(markers.report_air_backed_seed_fallback_cell_count[None]),
            0,
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_region_fallback_runs_when_only_some_markers_miss(
        self,
    ) -> None:
        fluid = self._solver()
        obstacle = np.zeros((16, 16, 16), dtype=np.int32)
        obstacle[:, :, 8] = 1
        fluid.obstacle.from_numpy(obstacle)
        fluid.snapshot_hibm_base_obstacle()
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=(
                (0.53125, 0.53125, 0.5625),
                (0.53125, 0.53125, 0.5625),
            ),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(7, 7),
        )
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 12] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 12] = 0

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
        )

        self.assertEqual((seeded, missed), (1, 1))
        self.assertGreater(
            int(markers.report_air_backed_seed_fallback_cell_count[None]),
            0,
        )

    def test_air_seed_region_fallback_uses_boundary_node_adjacency(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 8] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 8] = 0

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=-1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
        )

        self.assertEqual((seeded, missed), (0, 1))
        self.assertGreater(
            int(markers.report_air_backed_seed_fallback_cell_count[None]),
            0,
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_region_fallback_uses_boundary_anchor_cell(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 7] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 7] = 0
        node_anchor_cell = ti.Vector.field(3, dtype=ti.i32, shape=(16, 16, 16))
        node_anchor_cell.fill((-1, -1, -1))
        node_anchor_cell[8, 8, 7] = (8, 8, 12)

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=-1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
            node_anchor_cell=node_anchor_cell,
        )

        self.assertEqual((seeded, missed), (0, 1))
        self.assertGreater(
            int(markers.report_air_backed_seed_fallback_cell_count[None]),
            0,
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_region_fallback_uses_dirichlet_row_region_map(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        row_region = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        row_region.fill(-1)
        row_region[8, 8, 12] = 7

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=-1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
            velocity_dirichlet_marker_region_id=row_region,
        )

        self.assertEqual((seeded, missed), (0, 1))
        self.assertGreater(
            int(markers.report_air_backed_seed_fallback_cell_count[None]),
            0,
        )
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_region_fallback_selects_component_beyond_old_32_slot_cap(
        self,
    ) -> None:
        fluid = self._solver()
        obstacle = np.ones((16, 16, 16), dtype=np.int32)
        component_cells: list[tuple[int, int, int]] = []
        for k in range(2, 16, 2):
            for j in range(2, 16, 2):
                for i in range(2, 16, 2):
                    component_cells.append((i, j, k))
                    if len(component_cells) == 40:
                        break
                if len(component_cells) == 40:
                    break
            if len(component_cells) == 40:
                break
        for cell in component_cells:
            obstacle[cell] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.snapshot_hibm_base_obstacle()
        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 40)
        self.assertGreater(fluid.hibm_air_component_selected.shape[0], 32)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_count, 40)
        self.assertFalse(fluid.last_hibm_pressure_unreached_component_overflow)

        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.03125, 0.03125, 0.03125),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        row_region = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        row_region.fill(-1)
        target_cell = component_cells[39]
        row_region[target_cell] = 7

        seeded, missed = markers.mark_far_pressure_air_backed_seed_components(
            fluid.obstacle,
            fluid.hibm_base_obstacle,
            fluid.hibm_pressure_outlet_reachable,
            fluid.hibm_pressure_unreached_component_label,
            fluid.hibm_air_component_selected,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.cell_width_x_m,
            fluid.cell_width_y_m,
            fluid.cell_width_z_m,
            (16, 16, 16),
            far_pressure_region_id=7,
            far_pressure_inside_probe_max_multiplier=12.0,
            far_pressure_air_backed_probe_normal_sign=-1.0,
            fallback_to_region_adjacency_if_all_missed=True,
            node_kind_code=node_kind,
            nearest_marker=nearest_marker,
            velocity_dirichlet_marker_region_id=row_region,
        )

        self.assertEqual((seeded, missed), (0, 1))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1)
        self.assertEqual(int(fluid.hibm_air_cell[target_cell]), 1)

    def test_solid_band_split_preserves_protected_dirichlet_row_region_cells(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[2, 2, 3] = 0
        fluid.obstacle.from_numpy(obstacle)
        node_kind = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        fluid.velocity_dirichlet_boundary_active[2, 2, 3] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 3] = 8

        marked = fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=True,
            node_kind_code=node_kind,
            unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
            protect_velocity_dirichlet_radius_cells=0,
            protect_velocity_dirichlet_marker_region_id=8,
        )

        self.assertEqual(marked, 0)
        self.assertEqual(fluid.last_hibm_solid_band_interior_cells, 0)
        self.assertEqual(fluid.last_hibm_solid_band_enclosed_water_cells, 0)
        self.assertEqual(
            fluid.last_hibm_solid_band_velocity_dirichlet_protected_cells,
            1,
        )
        self.assertEqual(int(fluid.obstacle[2, 2, 3]), 0)

    def test_solid_band_split_preserves_unstamped_external_dirichlet_rows(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[2, 2, 3] = 0
        fluid.obstacle.from_numpy(obstacle)
        node_kind = ti.field(dtype=ti.i32, shape=(5, 5, 5))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        fluid.velocity_dirichlet_boundary_active[2, 2, 3] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)

        marked = fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=True,
            node_kind_code=node_kind,
            unclassified_node_code=HibmMpmIbNodeSearch._NODE_NONE,
            protect_velocity_dirichlet_radius_cells=0,
            protect_unstamped_velocity_dirichlet_rows=True,
        )

        self.assertEqual(marked, 0)
        self.assertEqual(
            fluid.last_hibm_solid_band_velocity_dirichlet_protected_cells,
            1,
        )
        self.assertEqual(int(fluid.obstacle[2, 2, 3]), 0)

    def test_pressure_reachability_allows_unstamped_external_velocity_inlet(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, :] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.velocity_dirichlet_boundary_active[1, 1, 3] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 0)
        self.assertEqual(int(fluid.hibm_pressure_outlet_reachable[1, 1, 3]), 1)

        fluid.velocity_dirichlet_boundary_marker_region_id[1, 1, 3] = 7
        unreached_with_hibm_row = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )

        self.assertEqual(unreached_with_hibm_row, 0)
        self.assertEqual(int(fluid.hibm_pressure_outlet_reachable[1, 1, 3]), 0)

    def test_row_cloud_orphan_conversion_preserves_unstamped_components(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(7, 5, 7), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((7, 5, 7), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 4] = 0
        obstacle[2, 2, 5] = 0
        obstacle[5, 2, 4] = 0
        obstacle[5, 2, 5] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 4] = 8
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=8,
        )

        self.assertEqual(converted, 2)
        self.assertEqual(fluid.last_hibm_row_cloud_orphan_component_count, 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 4]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 5]), 1)
        self.assertEqual(int(fluid.obstacle[5, 2, 4]), 0)
        self.assertEqual(int(fluid.obstacle[5, 2, 5]), 0)

    def test_row_cloud_orphan_conversion_uses_pressure_component_barriers(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 6), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 6), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 3] = 0
        obstacle[2, 2, 4] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.velocity_dirichlet_boundary_active[2, 2, 4] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 4] = 8

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 2)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_raw_count, 2)

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
        )

        self.assertEqual(converted, 2)
        self.assertEqual(fluid.last_hibm_row_cloud_orphan_component_count, 2)
        self.assertEqual(int(fluid.obstacle[2, 2, 3]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 4]), 1)

    def test_row_cloud_orphan_conversion_cleans_overflow_singletons_without_row_stamp(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(29, 29, 29), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((29, 29, 29), dtype=np.int32)
        obstacle[:, :, 0] = 0
        singleton_cells: list[tuple[int, int, int]] = []
        for i in range(1, 24, 2):
            for j in range(1, 24, 2):
                for k in range(2, 24, 2):
                    cell = (i, j, k)
                    singleton_cells.append(cell)
                    obstacle[cell] = 0
        large_cells: list[tuple[int, int, int]] = []
        for i in range(25, 29):
            for j in range(25, 29):
                for k in range(25, 29):
                    cell = (i, j, k)
                    large_cells.append(cell)
                    obstacle[cell] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, len(singleton_cells) + len(large_cells))
        self.assertTrue(fluid.last_hibm_pressure_unreached_component_overflow)
        self.assertEqual(
            fluid.last_hibm_pressure_unreached_component_raw_count,
            len(singleton_cells) + 1,
        )
        self.assertEqual(
            fluid.last_hibm_pressure_unreached_component_largest_cell_count,
            len(large_cells),
        )
        self.assertEqual(
            fluid.last_hibm_pressure_unreached_component_singleton_count,
            len(singleton_cells),
        )

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=128,
        )

        self.assertEqual(converted, len(singleton_cells))
        self.assertEqual(
            fluid.last_hibm_row_cloud_orphan_component_count,
            len(singleton_cells),
        )
        obstacle_after = fluid.obstacle.to_numpy()
        self.assertEqual(int(obstacle_after[singleton_cells[0]]), 1)
        self.assertEqual(int(obstacle_after[large_cells[0]]), 0)
        unreached_after = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )
        self.assertEqual(unreached_after, len(large_cells))
        self.assertFalse(fluid.last_hibm_pressure_unreached_component_overflow)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_raw_count, 1)
        self.assertEqual(
            fluid.last_hibm_pressure_unreached_component_largest_cell_count,
            len(large_cells),
        )

    def test_overflow_singleton_cleanup_respects_velocity_dirichlet_face_barriers(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 3] = 0
        obstacle[2, 2, 4] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.velocity_dirichlet_boundary_active[2, 2, 4] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 4] = 8

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 2)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_raw_count, 2)
        self.assertEqual(
            fluid.last_hibm_pressure_unreached_component_singleton_count,
            2,
        )
        fluid.last_hibm_pressure_unreached_component_overflow = True

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
            overflow_singletons_only=True,
        )

        self.assertEqual(converted, 2)
        self.assertEqual(fluid.last_hibm_row_cloud_orphan_component_count, 2)
        self.assertEqual(int(fluid.obstacle[2, 2, 3]), 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 4]), 1)

    def test_overflow_singleton_cleanup_preserves_no_slip_row_neighborhood(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(7, 7, 7), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((7, 7, 7), dtype=np.int32)
        obstacle[:, :, 0] = 0
        protected_singleton = (3, 3, 4)
        remote_singleton = (6, 6, 6)
        obstacle[protected_singleton] = 0
        obstacle[remote_singleton] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.velocity_dirichlet_boundary_active[3, 3, 2] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 2)
        self.assertEqual(fluid.last_hibm_pressure_unreached_component_raw_count, 2)
        fluid.last_hibm_pressure_unreached_component_overflow = True

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
            overflow_singletons_only=True,
            protect_velocity_dirichlet_radius_cells=2,
        )

        self.assertEqual(converted, 1)
        self.assertEqual(int(fluid.obstacle[protected_singleton]), 0)
        self.assertEqual(int(fluid.obstacle[remote_singleton]), 1)

    def test_row_cloud_orphan_cleanup_zero_radius_protects_exact_dirichlet_row(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(7, 7, 7), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((7, 7, 7), dtype=np.int32)
        obstacle[:, :, 0] = 0
        protected_singleton = (3, 3, 4)
        remote_singleton = (6, 6, 6)
        obstacle[protected_singleton] = 0
        obstacle[remote_singleton] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity_dirichlet_boundary_active.fill(0)
        fluid.velocity_dirichlet_boundary_active[protected_singleton] = 1
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )

        self.assertEqual(unreached, 2)
        fluid.last_hibm_pressure_unreached_component_overflow = True

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=1,
            overflow_singletons_only=True,
            protect_velocity_dirichlet_radius_cells=0,
        )

        self.assertEqual(converted, 1)
        self.assertEqual(int(fluid.obstacle[protected_singleton]), 0)
        self.assertEqual(int(fluid.obstacle[remote_singleton]), 1)

    def test_row_cloud_orphan_conversion_handles_uncompacted_positive_labels(
        self,
    ) -> None:
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        obstacle = np.ones((5, 5, 5), dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[2, 2, 3] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.velocity[2, 2, 3] = (1.0, 2.0, 3.0)
        fluid.velocity_prev[2, 2, 3] = (1.0, 2.0, 3.0)
        labels = np.zeros((5, 5, 5), dtype=np.int32)
        labels[2, 2, 3] = 12345
        fluid.hibm_pressure_unreached_component_label.from_numpy(labels)
        fluid.hibm_pressure_outlet_reachable.fill(0)
        fluid.hibm_pressure_reachability_barrier.fill(0)
        fluid.velocity_dirichlet_boundary_marker_region_id.fill(-1)
        fluid.velocity_dirichlet_boundary_marker_region_id[2, 2, 3] = 8

        converted = fluid.convert_hibm_row_cloud_orphan_components(
            max_component_cells=8,
        )

        self.assertEqual(converted, 1)
        self.assertEqual(fluid.last_hibm_row_cloud_orphan_component_count, 1)
        self.assertEqual(int(fluid.obstacle[2, 2, 3]), 1)
        self.assertEqual(tuple(float(v) for v in fluid.velocity[2, 2, 3]), (0.0, 0.0, 0.0))
        self.assertEqual(
            tuple(float(v) for v in fluid.velocity_prev[2, 2, 3]),
            (0.0, 0.0, 0.0),
        )

    def test_air_seed_uses_closure_region_external_nodes_as_reachability_barrier(
        self,
    ) -> None:
        # A real sharp membrane is a topology barrier even when its row is not
        # a base CAD obstacle. This fixture opens a one-cell numerical leak in
        # the slab; only the declared closure region's HIBM external node at
        # that leak may seal the pressure reachability flood. Unrelated
        # external IB nodes remain ordinary near-wall water for this topology
        # classification.
        fluid, markers = self._sealed_chamber_fixture()
        obstacle = fluid.obstacle.to_numpy()
        obstacle[8, 8, 8] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.snapshot_hibm_base_obstacle()

        unreached_without_barrier = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )
        self.assertEqual(unreached_without_barrier, 0)
        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (0, 1))

        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 8] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 8] = 0

        nonclosure_barrier_count = markers.write_region_pressure_reachability_barrier(
            fluid.hibm_pressure_reachability_barrier,
            node_kind,
            nearest_marker,
            barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            barrier_region_id=8,
        )
        self.assertEqual(nonclosure_barrier_count, 0)
        unreached_nonclosure = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
                use_existing_reachability_barrier=True,
            )
        )
        self.assertEqual(unreached_nonclosure, 0)

        closure_barrier_count = markers.write_region_pressure_reachability_barrier(
            fluid.hibm_pressure_reachability_barrier,
            node_kind,
            nearest_marker,
            barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            barrier_region_id=7,
        )
        self.assertEqual(closure_barrier_count, 1)
        unreached_with_barrier = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
                use_existing_reachability_barrier=True,
            )
        )
        self.assertEqual(unreached_with_barrier, 1792)

        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (1, 0))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)
        self.assertEqual(int(fluid.hibm_air_cell[8, 8, 12]), 1)
        self.assertEqual(int(fluid.hibm_air_cell[8, 8, 4]), 0)

    def test_air_seed_reachability_barrier_accepts_third_interface_region(
        self,
    ) -> None:
        fluid, _ = self._sealed_chamber_fixture()
        obstacle = fluid.obstacle.to_numpy()
        obstacle[8, 8, 8] = 0
        fluid.obstacle.from_numpy(obstacle)
        fluid.snapshot_hibm_base_obstacle()
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=(
                (0.53125, 0.53125, 0.5625),
                (0.53125, 0.53125, 0.5),
            ),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(7, 8),
        )

        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        node_kind[8, 8, 8] = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
        nearest_marker = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        nearest_marker.fill(-1)
        nearest_marker[8, 8, 8] = 1

        without_third = markers.write_region_pressure_reachability_barrier(
            fluid.hibm_pressure_reachability_barrier,
            node_kind,
            nearest_marker,
            barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            barrier_region_id=7,
            secondary_barrier_region_id=5,
            include_all_classified_region_nodes=True,
        )
        self.assertEqual(without_third, 0)
        self.assertEqual(
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
                use_existing_reachability_barrier=True,
            ),
            0,
        )

        with_third = markers.write_region_pressure_reachability_barrier(
            fluid.hibm_pressure_reachability_barrier,
            node_kind,
            nearest_marker,
            barrier_node_code=HibmMpmIbNodeSearch._NODE_EXTERNAL_IB,
            barrier_region_id=7,
            secondary_barrier_region_id=5,
            tertiary_barrier_region_id=8,
            include_all_classified_region_nodes=True,
        )
        self.assertEqual(with_third, 1)
        self.assertEqual(
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
                use_existing_reachability_barrier=True,
            ),
            1792,
        )

    def test_air_seed_selects_mirrored_far_side_unreached_component(
        self,
    ) -> None:
        fluid, markers = self._sealed_chamber_fixture()
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5625),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, -1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 1792)

        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (1, 0))

        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_air_seed_never_selects_outlet_reachable_water_or_crosses_base(
        self,
    ) -> None:
        # (b) The structural veto, two ways. (b1) A mis-oriented marker
        # whose far walk points INTO the outlet-connected water (lower
        # slab face, n = -z) finds only reachable cells on every rung ->
        # no seed, counted missed, zero conversion: outlet-reachable water
        # is NEVER classified, by construction (air is a subset of the
        # flood-unreached set). (b2) The H1-style crossing guard: a marker
        # BELOW the slab pointing +z hits the base-obstacle slab at rung 0
        # (z = 0.5 -> grid 7.5 -> nearest cell 8) and the walk ENDS - the
        # pocket beyond the wall is not seeded through base geometry even
        # though rungs 2+ would land in it geometrically.
        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.5),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, -1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (0, 1))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 0)
        self.assertEqual(int(fluid.obstacle[8, 8, 4]), 0)
        self.assertEqual(int(fluid.obstacle[8, 8, 12]), 0)

        markers.load_markers(
            positions_m=((0.53125, 0.53125, 0.4375),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(1.0,),
            region_ids=(7,),
        )
        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (0, 1))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 0)
        self.assertEqual(int(fluid.obstacle[8, 8, 12]), 0)

    def test_air_backed_state_defaults_inert_with_anchoring_status_quo(
        self,
    ) -> None:
        # (c) Default-off contract at the fluid level: when nothing ever
        # invokes the mechanism, the air state is the -1/0 sentinel set and
        # the sealed pocket keeps today's per-component zero-mean anchoring
        # path through the projection (cg_unreached_set_mean_projection_
        # count >= 1, the established fixture assertion family). The
        # assemble-level gate (`if bool(far_pressure_air_backed) and ...`)
        # is pinned by the wiring test below; the legacy noise-floor smoke
        # at apply time carries the bitwise claim for the production path.
        fluid, _ = self._sealed_chamber_fixture()
        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        self.assertEqual(unreached, 1792)

        report = fluid.project(
            iterations=200,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )

        self.assertTrue(report["cg_converged_all"])
        self.assertGreaterEqual(
            int(report.get("cg_unreached_set_mean_projection_count", 0)),
            1,
        )
        self.assertEqual(fluid.last_hibm_air_backed_cell_count, -1)
        self.assertEqual(fluid.last_hibm_air_backed_component_count, -1)
        self.assertEqual(fluid.last_hibm_air_backed_cell_volume_m3, -1.0)
        self.assertEqual(int(fluid.hibm_air_cell[8, 8, 12]), 0)
        self.assertEqual(int(fluid.hibm_air_component_selected[0]), 0)

    def test_air_backed_clears_sealed_pocket_divergence_debt(self) -> None:
        # (d) THE MONEY TEST, paper-walked. A volume source S = 4.0 1/s in
        # pocket cell (8, 8, 12) stands in for the membrane's vacated-zone
        # demand (the production pocket has no water supply, so continuity
        # is unsatisfiable there by exactly the source magnitude).
        #
        # WITHOUT A12 (today): the anchored projection can satisfy
        # everything EXCEPT the component mean - sum(div * V) over a sealed
        # component is invariant (every boundary face is blocked), so the
        # post-projection residual (div - vol_src) settles at the constant
        #   -S * V_cell / V_pocket = -(4.0 * 0.0625^3) / 0.4375
        #                          = -4.0 / 1792 = -2.2321e-3 1/s
        # per pocket cell; the RMS-style unreached_l2 equals |that| =
        # 2.2321e-3 and STAYS there on every further projection (the debt
        # is never absorbed - it is the per-step accrual run #2 integrated
        # for 1000 steps before the step-1015 reconnection dumped it).
        # The pocket also carries nonzero projected velocity - the debt
        # lives in the velocity field; the interior guard never sees it.
        #
        # WITH A12: the seed/convert pass removes the pocket AND its
        # forcing from the solve (vol_src is cleared by the band-cell
        # write set), the re-flood reports ZERO unreached cells, the
        # unreached divergence channel reads exactly 0 with count 0, the
        # pocket velocity is exactly zero (obstacle), and punching the
        # step-1015-style hole in the slab afterwards reconnects NOTHING
        # (air cells are walls): unreached stays 0 and the post-hole
        # projection's interior residual stays clean. The reconnection
        # bomb's precondition - fake water carrying inherited divergence -
        # is structurally absent. (The 1000-step growth curve itself is
        # gate-probe evidence, not unit-fixture material.)
        expected_residual = 4.0 / 1792.0

        fluid_without, _ = self._sealed_chamber_fixture()
        unreached = (
            fluid_without.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )
        self.assertEqual(unreached, 1792)
        fluid_without.volume_source_s[8, 8, 12] = 4.0
        for _ in range(3):
            report = fluid_without.project(
                iterations=400,
                pressure_outlet_zmin=True,
                pressure_solver="fv_cg",
                cg_tolerance=1.0e-6,
            )
            self.assertTrue(report["cg_converged_all"])
            fluid_without.compute_divergence(pressure_outlet_zmin=True)
            fluid_without.final_divergence_report_stats(pressure_outlet_zmin=True)
            stats = fluid_without.last_unreached_divergence_stats
            self.assertEqual(int(stats["count"]), 1792)
            self.assertGreater(float(stats["l2"]), 1.0e-3)
            self.assertAlmostEqual(
                float(stats["l2"]),
                expected_residual,
                delta=0.02 * expected_residual,
            )
            self.assertAlmostEqual(
                float(stats["max_abs"]),
                expected_residual,
                delta=0.10 * expected_residual,
            )
        velocity_without = fluid_without.velocity.to_numpy()
        self.assertGreater(float(np.abs(velocity_without[:, :, 9:]).max()), 0.0)

        fluid_with, markers = self._sealed_chamber_fixture()
        unreached = (
            fluid_with.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            )
        )
        self.assertEqual(unreached, 1792)
        fluid_with.volume_source_s[8, 8, 12] = 4.0
        seeded, missed = self._seed(markers, fluid_with)
        self.assertEqual((seeded, missed), (1, 0))
        self.assertEqual(fluid_with.convert_hibm_air_backed_cells(), 1792)
        self.assertAlmostEqual(
            float(fluid_with.volume_source_s[8, 8, 12]), 0.0, delta=0.0
        )
        self.assertEqual(
            fluid_with.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            0,
        )
        report = fluid_with.project(
            iterations=400,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )
        self.assertTrue(report["cg_converged_all"])
        fluid_with.compute_divergence(pressure_outlet_zmin=True)
        fluid_with.final_divergence_report_stats(pressure_outlet_zmin=True)
        stats = fluid_with.last_unreached_divergence_stats
        self.assertEqual(int(stats["count"]), 0)
        self.assertEqual(float(stats["l2"]), 0.0)
        velocity_with = fluid_with.velocity.to_numpy()
        self.assertEqual(float(np.abs(velocity_with[:, :, 9:]).max()), 0.0)

        # The step-1015 scenario: punch a 1-cell hole in the slab. The
        # flood enters the hole cell and stops at the air walls - there is
        # no fake-water pocket left to reconnect.
        fluid_with.obstacle[8, 8, 8] = 0
        self.assertEqual(
            fluid_with.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            0,
        )
        report = fluid_with.project(
            iterations=400,
            pressure_outlet_zmin=True,
            pressure_solver="fv_cg",
            cg_tolerance=1.0e-6,
        )
        self.assertTrue(report["cg_converged_all"])
        fluid_with.compute_divergence(pressure_outlet_zmin=True)
        fluid_with.final_divergence_report_stats(pressure_outlet_zmin=True)
        self.assertLess(
            float(fluid_with.last_unreached_divergence_stats["l2"]), 1.0e-12
        )

    def test_air_backed_cells_rewet_via_fresh_fluid_reconstruction(
        self,
    ) -> None:
        # (e) Statelessness + re-wetting, the per-step cycle: conversion at
        # step N; step N+1's apply_hibm_internal_obstacles resets obstacle
        # to base, detects every ex-air cell as fresh fluid (old obstacle,
        # base fluid) and reconstructs velocities - the pocket interior is
        # an all-fresh neighborhood, so the reconstruction finds no donor
        # and keeps the conversion's exact zeros (the right initial state
        # for water entering a region that physically held air; on the
        # production rebound the band/rows hand it real boundary data).
        # The re-flood then re-finds the 1792-cell pocket and the
        # classification reconverts it identically - the steady cycle the
        # checkpoint/resume path replays from any step.
        import taichi as ti

        fluid, markers = self._sealed_chamber_fixture()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (1, 0))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)
        self.assertEqual(int(fluid.obstacle[8, 8, 12]), 1)

        node_kind = ti.field(dtype=ti.i32, shape=(16, 16, 16))
        node_kind.fill(HibmMpmIbNodeSearch._NODE_NONE)
        internal_count = fluid.apply_hibm_internal_obstacles(
            node_kind,
            internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
        )

        self.assertEqual(internal_count, 0)
        self.assertEqual(int(fluid.report_hibm_fresh_fluid_cells[None]), 1792)
        self.assertEqual(int(fluid.obstacle[8, 8, 12]), 0)
        self.assertEqual(int(fluid.obstacle[8, 8, 8]), 1)
        self.assertEqual(
            tuple(float(fluid.velocity[8, 8, 12][axis]) for axis in range(3)),
            (0.0, 0.0, 0.0),
        )

        self.assertEqual(
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=True,
            ),
            1792,
        )
        seeded, missed = self._seed(markers, fluid)
        self.assertEqual((seeded, missed), (1, 0))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 1792)

    def test_assemble_wires_air_backed_classification_order_and_gate(
        self,
    ) -> None:
        # (f) Wiring contract on the assemble chain (the A8'' fill/view
        # ordering-test pattern): the classification consumes THIS step's
        # flood + component labels (strictly after the first disconnected-
        # cells call, strictly before the substep loop whose rows+flood
        # rebuilds see air as obstacle); any conversion re-assembles rows
        # and reruns the band fixed point (the S2-A8' zero-row lesson);
        # the p_far stamp sits strictly between the A8'' fill and the
        # sampling-view build; everything rides the per-closure-region
        # opt-in gate, default False = the block is dead code.
        import inspect

        import simulation_core.hibm_mpm as hibm_mpm_module

        source = inspect.getsource(
            hibm_mpm_module.assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        )
        self.assertIn("far_pressure_air_backed: bool = False,", source)
        self.assertIn(
            "if bool(far_pressure_air_backed) and int(far_pressure_region_id) != -1:",
            source,
        )
        self.assertIn("markers.mark_far_pressure_air_backed_seed_components(", source)
        self.assertIn("fluid.convert_hibm_air_backed_cells()", source)
        self.assertIn("for _air_band_pass in range(8):", source)
        self.assertIn("fluid.write_hibm_air_backed_cell_pressures(", source)

        first_flood = source.index(
            "fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells("
        )
        seed_call = source.index(
            "markers.mark_far_pressure_air_backed_seed_components("
        )
        convert_call = source.index("fluid.convert_hibm_air_backed_cells()")
        air_band_loop = source.index("for _air_band_pass in range(8):")
        substep_loop = source.index("for _ in range(substeps):")
        fill_call = source.index("fluid.fill_hibm_converted_cell_pressures(")
        stamp_call = source.index("fluid.write_hibm_air_backed_cell_pressures(")
        view_call = source.index("fluid.build_hibm_sampling_obstacle(")
        stress_call = source.index("sample_fluid_stress_to_marker_tractions(")
        self.assertLess(first_flood, seed_call)
        self.assertLess(seed_call, convert_call)
        self.assertLess(convert_call, air_band_loop)
        self.assertLess(air_band_loop, substep_loop)
        self.assertLess(fill_call, stamp_call)
        self.assertLess(stamp_call, view_call)
        self.assertLess(view_call, stress_call)

        # The stamp is double-gated: armed AND converted (no kernel launch
        # on inert steps - flicker/partial-enclosure steps stay cheap).
        stamp_gate_window = source[max(0, stamp_call - 600):stamp_call]
        self.assertIn(
            "bool(far_pressure_air_backed) and int(hibm_air_backed_cell_count) > 0",
            stamp_gate_window,
        )


if __name__ == "__main__":
    unittest.main()
