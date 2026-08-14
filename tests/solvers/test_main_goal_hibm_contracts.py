import unittest
from types import SimpleNamespace

import numpy as np
import taichi as ti

from simulation_core import (
    HibmMpmIbBoundaryConditions,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from simulation_core.coupling.hibm_mpm.core import (
    hibm_mpm_external_force_parts_fresh_for_solid_step,
    hibm_mpm_external_force_fresh_for_solid_step,
)
from simulation_core.coupling.hibm_mpm.reports import (
    HibmMpmMpmForceScatterReport,
)


class _ScalarField:
    def __init__(self, value=0):
        self.value = value

    def __getitem__(self, _index):
        return self.value

    def __setitem__(self, _index, value):
        self.value = value


class _CountedField:
    def __init__(self, capacity: int):
        self.shape = (capacity,)


@ti.kernel
def _select_storage_probe(
    boundary: ti.template(),
    obstacle: ti.template(),
    cell_face_x_m: ti.template(),
    cell_face_y_m: ti.template(),
    cell_face_z_m: ti.template(),
    cell_center_x_m: ti.template(),
    cell_center_y_m: ti.template(),
    cell_center_z_m: ti.template(),
    result_i32: ti.template(),
    result_f32: ti.template(),
):
    valid, storage, alpha, error_code = (
        boundary._select_canonical_component_face_storage_device(
            ti.Vector([0, 1, 1]),
            1,
            ti.Vector([0.25, 0.50, 0.60]),
            ti.Vector([0.25, 0.501, 0.10]),
            obstacle,
            0,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            2,
            3,
            2,
        )
    )
    result_i32[0] = valid
    result_i32[1] = storage.x
    result_i32[2] = storage.y
    result_i32[3] = storage.z
    result_i32[4] = error_code
    result_f32[0] = alpha


class HibmMainGoalContracts(unittest.TestCase):
    @staticmethod
    def _three_marker_surface() -> HibmMpmSurfaceMarkers:
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=3,
            runtime=TaichiRuntimeConfig(arch="cuda", default_fp="f32"),
        )
        markers.load_markers(
            positions_m=((0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1)),
            velocities_mps=((0.0, 0.0, 0.0),) * 3,
            normals=((0.0, 0.0, 1.0),) * 3,
            areas_m2=(0.5, 0.5, 0.5),
            region_ids=(1, 1, 1),
        )
        markers.set_projection_triangles(((0, 1, 2),))
        return markers

    @staticmethod
    def _valid_gate_parts() -> dict[str, object]:
        marker_count = 2
        return {
            "clear": {
                "cleared_particle_count": 1,
                "max_abs_external_force_before_n": 0.0,
            },
            "scatter": {
                "active_marker_count": marker_count,
                "invalid_marker_count": 0,
                "active_pair_count": marker_count,
                "total_marker_force_n": (1.0, 0.0, 0.0),
                "total_mpm_external_force_n": (1.0, 0.0, 0.0),
                "action_reaction_residual_n": 0.0,
                "invalid_external_force_particle_count": 0,
                "max_abs_external_force_component_n": 1.0,
            },
            "marker_forces": {
                "total_marker_count": marker_count,
                "total_marker_force_n": (1.0, 0.0, 0.0),
                "fluid_reaction_force_n": (-1.0, 0.0, 0.0),
                "action_reaction_residual_n": 0.0,
                "primary_stress_invalid_marker_count": 0,
                "secondary_stress_invalid_marker_count": 0,
            },
            "stress": {
                "valid_marker_count": marker_count,
                "invalid_marker_count": 0,
                "viscous_gradient_invalid_marker_count": 0,
                "max_abs_traction_pa": 1.0,
            },
            "no_slip": {
                "valid_marker_count": marker_count,
                "invalid_marker_count": 0,
                "max_no_slip_residual_mps": 0.0,
                "l2_no_slip_residual_mps": 0.0,
            },
            "projection": {
                "cg_converged_all": True,
                "cg_breakdown_count": 0,
                "cg_relative_residual_max": 0.0,
                "pressure_solve_failed": False,
                "pressure_projection_physical_failure": False,
            },
            "pressure_component_overflow": False,
            "pressure_component_labels_converged": True,
        }

    def test_counted_triangle_loader_rejects_fractional_count_before_mutation(
        self,
    ) -> None:
        markers = self._three_marker_surface()
        triangles = ti.Vector.field(3, dtype=ti.i32, shape=1)
        triangles[0] = (0, 1, 2)
        old_revision = markers.marker_geometry_revision

        with self.assertRaises(TypeError):
            markers.load_projection_triangles_from_field(
                triangles,
                triangle_count=1.5,
            )

        self.assertEqual(markers.marker_geometry_revision, old_revision)
        self.assertEqual(markers.projection_triangle_count, 1)

    def test_counted_triangle_loader_rejects_float_nan_before_mutation(self) -> None:
        markers = self._three_marker_surface()
        triangles = ti.Vector.field(3, dtype=ti.f32, shape=1)
        triangles[0] = (float("nan"), 1.0, 2.0)
        old_revision = markers.marker_geometry_revision

        with self.assertRaises((TypeError, ValueError)):
            markers.load_projection_triangles_from_field(
                triangles,
                triangle_count=1,
            )

        self.assertEqual(markers.marker_geometry_revision, old_revision)
        self.assertEqual(markers.projection_triangle_count, 1)

    def test_counted_surface_loader_rejects_nonintegral_region_before_mutation(
        self,
    ) -> None:
        markers = self._three_marker_surface()
        positions = ti.Vector.field(3, dtype=ti.f32, shape=1)
        normals = ti.Vector.field(3, dtype=ti.f32, shape=1)
        areas = ti.field(dtype=ti.f32, shape=1)
        regions = ti.field(dtype=ti.f32, shape=1)
        positions[0] = (0.4, 0.4, 0.4)
        normals[0] = (0.0, 1.0, 0.0)
        areas[0] = 1.0
        regions[0] = float("nan")
        old_revision = markers.marker_geometry_revision

        with self.assertRaises((TypeError, ValueError)):
            markers.load_markers_from_surface_fields(
                positions,
                normals,
                areas,
                regions,
                marker_count=1,
            )

        self.assertEqual(markers.marker_geometry_revision, old_revision)
        self.assertEqual(markers.marker_count, 3)
        self.assertEqual(markers.projection_triangle_count, 1)

    def test_storage_selector_prefers_face_on_probe_ray_over_earlier_projection(
        self,
    ) -> None:
        """Tangential consistency owns the MAC face; progress breaks true ties."""

        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=(2, 3, 2),
            marker_capacity=1,
            runtime=runtime,
        )
        obstacle = ti.field(dtype=ti.i32, shape=(2, 3, 2))
        cell_face_x_m = ti.field(dtype=ti.f32, shape=3)
        cell_face_y_m = ti.field(dtype=ti.f32, shape=4)
        cell_face_z_m = ti.field(dtype=ti.f32, shape=3)
        cell_center_x_m = ti.field(dtype=ti.f32, shape=2)
        cell_center_y_m = ti.field(dtype=ti.f32, shape=3)
        cell_center_z_m = ti.field(dtype=ti.f32, shape=2)
        result_i32 = ti.field(dtype=ti.i32, shape=5)
        result_f32 = ti.field(dtype=ti.f32, shape=1)

        cell_face_x_m.from_numpy(np.asarray((0.0, 0.5, 1.0), dtype=np.float32))
        cell_face_y_m.from_numpy(
            np.asarray((0.0, 0.25, 0.50, 0.75), dtype=np.float32)
        )
        cell_face_z_m.from_numpy(np.asarray((0.0, 0.5, 1.0), dtype=np.float32))
        cell_center_x_m.from_numpy(np.asarray((0.25, 0.75), dtype=np.float32))
        cell_center_y_m.from_numpy(
            np.asarray((0.125, 0.375, 0.625), dtype=np.float32)
        )
        cell_center_z_m.from_numpy(np.asarray((0.25, 0.50), dtype=np.float32))

        _select_storage_probe(
            boundary,
            obstacle,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            result_i32,
            result_f32,
        )

        result = result_i32.to_numpy()
        self.assertEqual(int(result[0]), 1)
        self.assertEqual(tuple(int(value) for value in result[1:4]), (0, 2, 1))
        self.assertEqual(int(result[4]), 0)
        self.assertGreater(float(result_f32[0]), 0.0)

    def test_scatter_census_rejects_device_force_overflow(self) -> None:
        runtime = TaichiRuntimeConfig(arch="cuda", default_fp="f32")
        markers = HibmMpmSurfaceMarkers(marker_capacity=2, runtime=runtime)
        markers.load_markers(
            positions_m=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(1.0, 1.0),
            region_ids=(1, 1),
        )
        markers.F_gamma_n[0] = (2.0e38, 0.0, 0.0)
        markers.F_gamma_n[1] = (2.0e38, 0.0, 0.0)
        particle_position_m = ti.Vector.field(3, dtype=ti.f32, shape=1)
        external_force_n = ti.Vector.field(3, dtype=ti.f32, shape=1)
        particle_position_m[0] = (0.5, 0.5, 0.5)

        report = markers.scatter_marker_forces_to_mpm_particles(
            external_force_n,
            particle_position_m,
            particle_count=1,
            support_radius_m=1.0,
            particle_position_generation=1,
        )

        self.assertTrue(np.isinf(external_force_n.to_numpy()[0, 0]))
        self.assertEqual(report.invalid_external_force_particle_count, 1)

    def test_load_gate_rejects_nonfinite_device_force_census(self) -> None:
        marker_count = 2
        load_report = SimpleNamespace(
            mpm_external_force_clear=SimpleNamespace(
                cleared_particle_count=1,
                max_abs_external_force_before_n=0.0,
            ),
            mpm_force_scatter=SimpleNamespace(
                active_marker_count=marker_count,
                invalid_marker_count=0,
                active_pair_count=marker_count,
                total_marker_force_n=(4.0e38, 0.0, 0.0),
                total_mpm_external_force_n=(4.0e38, 0.0, 0.0),
                action_reaction_residual_n=0.0,
                invalid_external_force_particle_count=1,
                max_abs_external_force_component_n=float("inf"),
            ),
            marker_forces=SimpleNamespace(
                total_marker_count=marker_count,
                total_marker_force_n=(4.0e38, 0.0, 0.0),
                fluid_reaction_force_n=(-4.0e38, 0.0, 0.0),
                action_reaction_residual_n=0.0,
                primary_stress_invalid_marker_count=0,
                secondary_stress_invalid_marker_count=0,
            ),
            fluid_stress=SimpleNamespace(
                valid_marker_count=marker_count,
                invalid_marker_count=0,
                viscous_gradient_invalid_marker_count=0,
                max_abs_traction_pa=1.0,
            ),
            no_slip_residual=SimpleNamespace(
                valid_marker_count=marker_count,
                invalid_marker_count=0,
                max_no_slip_residual_mps=0.0,
                l2_no_slip_residual_mps=0.0,
            ),
            fluid_projection={
                "cg_converged_all": True,
                "cg_breakdown_count": 0,
                "cg_relative_residual_max": 0.0,
                "pressure_solve_failed": False,
                "pressure_projection_physical_failure": False,
            },
            pressure_disconnected_region=SimpleNamespace(
                component_overflow=False,
                component_labels_converged=True,
            ),
        )

        self.assertFalse(
            hibm_mpm_external_force_fresh_for_solid_step(load_report)
        )

    def test_parts_gate_accepts_runner_mapping_and_pressure_health_scalars(
        self,
    ) -> None:
        self.assertTrue(
            hibm_mpm_external_force_parts_fresh_for_solid_step(
                **self._valid_gate_parts(),
            )
        )

    def test_parts_gate_rejects_coerced_counts_and_boolean_strings(self) -> None:
        invalid_parts = (
            ("scatter", "active_marker_count", 2.0),
            ("marker_forces", "total_marker_count", True),
            ("projection", "cg_breakdown_count", 0.5),
            ("projection", "cg_converged_all", "true"),
            ("projection", "pressure_solve_failed", 0),
        )
        for section, field_name, invalid_value in invalid_parts:
            with self.subTest(section=section, field_name=field_name):
                parts = self._valid_gate_parts()
                parts[section] = {
                    **parts[section],
                    field_name: invalid_value,
                }
                self.assertFalse(
                    hibm_mpm_external_force_parts_fresh_for_solid_step(**parts)
                )

        parts = {
            **self._valid_gate_parts(),
            "pressure_component_overflow": "false",
        }
        self.assertFalse(
            hibm_mpm_external_force_parts_fresh_for_solid_step(**parts)
        )

    def test_parts_gate_rejects_scatter_report_without_device_census(self) -> None:
        parts = self._valid_gate_parts()
        scatter = parts["scatter"]
        parts["scatter"] = HibmMpmMpmForceScatterReport(
            active_marker_count=scatter["active_marker_count"],
            invalid_marker_count=scatter["invalid_marker_count"],
            active_pair_count=scatter["active_pair_count"],
            total_marker_force_n=scatter["total_marker_force_n"],
            total_mpm_external_force_n=scatter["total_mpm_external_force_n"],
            action_reaction_residual_n=scatter["action_reaction_residual_n"],
        )

        self.assertFalse(
            hibm_mpm_external_force_parts_fresh_for_solid_step(**parts)
        )

    def test_partial_surface_feedback_retires_geometry_before_return(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_count = 2
        markers.projection_vertex_count = 2
        markers.projection_triangle_count = 1
        markers.projection_segment_count = 1
        markers.marker_geometry_revision = 4
        markers._open_ribbon_tip_cap_binding = ("active",)
        markers._current_no_slip_sampling_identity = object()
        markers._input_validation_failure_count = _ScalarField()
        markers.report_surface_feedback_updated_marker_count = _ScalarField()
        markers.report_surface_feedback_invalid_marker_count = _ScalarField()
        markers.report_surface_feedback_geometry_updated_marker_count = _ScalarField()
        markers.report_surface_feedback_geometry_invalid_marker_count = _ScalarField()
        markers.report_surface_feedback_max_displacement_m = _ScalarField()
        markers.report_surface_feedback_max_speed_mps = _ScalarField()
        markers.report_surface_feedback_max_normal_change = _ScalarField()
        markers.report_surface_feedback_max_area_change_m2 = _ScalarField()
        markers.report_surface_feedback_candidate_pair_count = _ScalarField()
        markers._mpm_bin_counts = object()
        markers._mpm_bin_offsets = object()
        markers._mpm_bin_members = object()
        markers._mpm_marker_neighbor_slots = object()
        markers._mpm_marker_neighbor_slot_counts = object()
        markers._validate_mpm_surface_feedback_fields_kernel = lambda *_args: None
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: None
        markers._begin_marker_geometry_write = lambda: None
        markers._refresh_open_ribbon_tip_cap_projection_vertices = lambda: None

        def partial_feedback(*_args):
            markers.report_surface_feedback_updated_marker_count[None] = 1
            markers.report_surface_feedback_invalid_marker_count[None] = 1
            markers.report_surface_feedback_geometry_updated_marker_count[None] = 1
            markers.report_surface_feedback_geometry_invalid_marker_count[None] = 1

        markers._update_surface_feedback_from_mpm_surface_particles_kernel = (
            partial_feedback
        )
        field = _CountedField(2)

        with self.assertRaisesRegex(RuntimeError, "incomplete.*surface feedback"):
            markers.update_surface_feedback_from_mpm_surface_particles(
                field,
                field,
                field,
                field,
                particle_count=2,
                support_radius_m=1.0,
                dt_s=1.0,
            )

        self.assertEqual(markers.marker_count, 0)
        self.assertEqual(markers.projection_vertex_count, 0)
        self.assertEqual(markers.projection_triangle_count, 0)
        self.assertEqual(markers.projection_segment_count, 0)


if __name__ == "__main__":
    unittest.main()
