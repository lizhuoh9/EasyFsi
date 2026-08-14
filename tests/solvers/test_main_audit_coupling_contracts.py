import math
import unittest
from types import SimpleNamespace

import numpy as np
import taichi as ti

from simulation_core.coupling.hibm_mpm.core import (
    HibmMpmSurfaceMarkers,
    _normalize_vector3,
    _scalar_for_f32_field,
    hibm_mpm_external_force_fresh_for_solid_step,
)
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
    _uses_marker_constraint_hash,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi


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
        self.dtype = ti.i32


def _namespace_with(value, **changes):
    return SimpleNamespace(**(vars(value) | changes))


def _valid_load_report(marker_count: int = 10):
    scatter = SimpleNamespace(
        active_marker_count=marker_count,
        invalid_marker_count=0,
        active_pair_count=marker_count,
        total_marker_force_n=(1.0, 0.0, 0.0),
        total_mpm_external_force_n=(1.0, 0.0, 0.0),
        action_reaction_residual_n=0.0,
        invalid_external_force_particle_count=0,
        max_abs_external_force_component_n=1.0,
    )
    return SimpleNamespace(
        mpm_external_force_clear=SimpleNamespace(
            cleared_particle_count=marker_count,
            max_abs_external_force_before_n=0.0,
        ),
        mpm_force_scatter=scatter,
        marker_forces=SimpleNamespace(
            total_marker_count=marker_count,
            total_marker_force_n=(1.0, 0.0, 0.0),
            fluid_reaction_force_n=(-1.0, 0.0, 0.0),
            action_reaction_residual_n=0.0,
            primary_stress_invalid_marker_count=0,
            secondary_stress_invalid_marker_count=0,
        ),
        fluid_stress=SimpleNamespace(
            valid_marker_count=marker_count,
            invalid_marker_count=0,
            max_abs_traction_pa=1.0,
            viscous_gradient_invalid_marker_count=0,
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


class HibmMpmAdvanceGateContracts(unittest.TestCase):
    def test_gate_requires_one_complete_finite_load_transaction(self) -> None:
        report = _valid_load_report()
        self.assertTrue(hibm_mpm_external_force_fresh_for_solid_step(report))

        rejected = {
            "invalid scatter marker": _namespace_with(
                report,
                mpm_force_scatter=_namespace_with(
                    report.mpm_force_scatter,
                    invalid_marker_count=1,
                ),
            ),
            "partial active marker census": _namespace_with(
                report,
                mpm_force_scatter=_namespace_with(
                    report.mpm_force_scatter,
                    active_marker_count=9,
                ),
            ),
            "failed stress sample": _namespace_with(
                report,
                fluid_stress=_namespace_with(
                    report.fluid_stress,
                    valid_marker_count=9,
                    invalid_marker_count=1,
                ),
            ),
            "failed no-slip sample": _namespace_with(
                report,
                no_slip_residual=_namespace_with(
                    report.no_slip_residual,
                    valid_marker_count=9,
                    invalid_marker_count=1,
                ),
            ),
            "excess force residual": _namespace_with(
                report,
                mpm_force_scatter=_namespace_with(
                    report.mpm_force_scatter,
                    action_reaction_residual_n=1.0e-3,
                ),
            ),
            "failed pressure solve": _namespace_with(
                report,
                fluid_projection=report.fluid_projection
                | {"pressure_solve_failed": True},
            ),
        }
        for reason, rejected_report in rejected.items():
            with self.subTest(reason=reason):
                self.assertFalse(
                    hibm_mpm_external_force_fresh_for_solid_step(rejected_report)
                )

    def test_gate_rejects_an_incomplete_report(self) -> None:
        report = _valid_load_report()
        del report.fluid_stress
        self.assertFalse(hibm_mpm_external_force_fresh_for_solid_step(report))

    def test_gate_rejects_finite_vectors_with_overflowing_norms(self) -> None:
        report = _valid_load_report()
        huge_force = (1.7e308, 1.7e308, 0.0)
        scatter = _namespace_with(
            report.mpm_force_scatter,
            total_marker_force_n=huge_force,
            total_mpm_external_force_n=huge_force,
        )
        marker_forces = _namespace_with(
            report.marker_forces,
            total_marker_force_n=(0.0, 0.0, 0.0),
        )

        self.assertFalse(
            hibm_mpm_external_force_fresh_for_solid_step(
                _namespace_with(
                    report,
                    mpm_force_scatter=scatter,
                    marker_forces=marker_forces,
                )
            )
        )

    def test_gate_rejects_missing_projection_fields_and_malformed_vectors(self) -> None:
        report = _valid_load_report()
        for missing_field in (
            "cg_converged_all",
            "cg_breakdown_count",
            "cg_relative_residual_max",
            "pressure_solve_failed",
            "pressure_projection_physical_failure",
        ):
            with self.subTest(missing_field=missing_field):
                projection = dict(report.fluid_projection)
                projection.pop(missing_field)
                self.assertFalse(
                    hibm_mpm_external_force_fresh_for_solid_step(
                        _namespace_with(report, fluid_projection=projection)
                    )
                )

        malformed_scatter = _namespace_with(
            report.mpm_force_scatter,
            total_mpm_external_force_n=(1.0, 0.0),
        )
        self.assertFalse(
            hibm_mpm_external_force_fresh_for_solid_step(
                _namespace_with(report, mpm_force_scatter=malformed_scatter)
            )
        )


class MarkerMacTransactionContracts(unittest.TestCase):
    def test_failed_reprepare_retires_the_previous_commit(self) -> None:
        operator = object.__new__(HibmMpmMarkerMacConstraintOperator)
        operator.marker_capacity = 1
        operator._phase = "committed"
        operator._prepared = True
        operator._converged = True
        operator._committed = True
        operator._clear_pressure_nullspace_lifecycle = lambda: None

        with self.assertRaisesRegex(ValueError, "marker_count"):
            operator.prepare(
                markers=SimpleNamespace(marker_count=2),
                fluid=SimpleNamespace(),
                component_face_valid_mask=SimpleNamespace(),
                primary_region_id=7,
                secondary_region_id=8,
            )

        self.assertEqual(operator._phase, "failed")
        self.assertFalse(operator._prepared)
        self.assertFalse(operator._converged)
        self.assertFalse(operator._committed)

    def test_stale_transaction_retires_pressure_nullspace_state(self) -> None:
        operator = object.__new__(HibmMpmMarkerMacConstraintOperator)
        operator._phase = "committed"
        operator._prepared = True
        operator._converged = True
        operator._committed = True
        operator._pressure_nullspace_prepared = True

        def clear_pressure_lifecycle():
            operator._pressure_nullspace_prepared = False

        operator._clear_pressure_nullspace_lifecycle = clear_pressure_lifecycle

        with self.assertRaisesRegex(RuntimeError, "stale"):
            operator._invalidate_stale_transaction("synthetic drift")

        self.assertFalse(operator._committed)
        self.assertFalse(operator._pressure_nullspace_prepared)

    def test_marker_constraint_hash_routes_only_large_populations(self) -> None:
        self.assertFalse(_uses_marker_constraint_hash(64))
        self.assertTrue(_uses_marker_constraint_hash(65))

    def test_large_marker_hash_preserves_exact_duplicate_contract(self) -> None:
        init_taichi(TaichiRuntimeConfig(arch="cuda"))
        marker_count = 65
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=(2, 2, 2),
            marker_capacity=marker_count,
        )
        positions = ti.Vector.field(3, dtype=ti.f32, shape=marker_count)
        targets = ti.Vector.field(3, dtype=ti.f32, shape=marker_count)
        regions = ti.field(dtype=ti.i32, shape=marker_count)
        position_values = np.zeros((marker_count, 3), dtype=np.float32)
        position_values[:, 0] = np.arange(marker_count, dtype=np.float32)
        position_values[-2] = (0.0, 0.25, 0.5)
        position_values[-1] = (-0.0, 0.25, 0.5)
        target_values = np.zeros((marker_count, 3), dtype=np.float32)
        target_values[-2:, 0] = np.float32(1.0)
        region_values = np.zeros(marker_count, dtype=np.int32)
        region_values[-2:] = 7
        positions.from_numpy(position_values)
        targets.from_numpy(target_values)
        regions.from_numpy(region_values)

        operator._failure_code[None] = 0
        operator._reset_marker_constraint_hash_kernel()
        operator._canonicalize_marker_constraints_kernel(
            positions,
            targets,
            regions,
            marker_count,
            7,
            8,
        )
        owners = operator._marker_constraint_owner.to_numpy()
        self.assertEqual(int(owners[-2]), marker_count - 2)
        self.assertEqual(int(owners[-1]), marker_count - 2)
        self.assertEqual(int(operator._failure_code[None]), 0)

        target_values[-1, 0] = np.nextafter(
            np.float32(1.0),
            np.float32(2.0),
        )
        targets.from_numpy(target_values)
        operator._failure_code[None] = 0
        operator._reset_marker_constraint_hash_kernel()
        operator._canonicalize_marker_constraints_kernel(
            positions,
            targets,
            regions,
            marker_count,
            7,
            8,
        )
        self.assertEqual(int(operator._failure_code[None]), 2)


class MarkerInputContracts(unittest.TestCase):
    def test_host_normalization_is_overflow_safe(self) -> None:
        normal = _normalize_vector3((1.0e308, 1.0e308, 0.0), name="normal")
        self.assertTrue(all(math.isfinite(component) for component in normal))
        self.assertAlmostEqual(normal[0], math.sqrt(0.5))
        self.assertAlmostEqual(normal[1], math.sqrt(0.5))

    def test_host_marker_load_rejects_values_not_representable_as_f32(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_capacity = 1
        markers._begin_marker_geometry_write = lambda: self.fail(
            "a rejected host load started a geometry transaction"
        )

        with self.assertRaisesRegex(ValueError, "f32"):
            markers.load_markers(
                positions_m=((1.0e100, 0.0, 0.0),),
                velocities_mps=((0.0, 0.0, 0.0),),
                normals=((1.0, 0.0, 0.0),),
                areas_m2=(1.0,),
                region_ids=(7,),
            )

    def test_scatter_rejects_support_radius_not_representable_as_f32(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: self.fail(
            "invalid support radius reached bin preparation"
        )
        field = _CountedField(1)

        with self.assertRaisesRegex(ValueError, "f32"):
            markers.scatter_marker_forces_to_mpm_particles(
                field,
                field,
                particle_count=1,
                support_radius_m=1.0e100,
            )

    def test_nonzero_scalar_that_rounds_to_f32_zero_is_rejected(self) -> None:
        for name in ("support_radius_m", "dt_s"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "f32"):
                _scalar_for_f32_field(1.0e-50, name=name)

    def test_scatter_rejects_invalid_particle_position_before_device_mutation(
        self,
    ) -> None:
        init_taichi(TaichiRuntimeConfig(arch="cuda"))
        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
        external_force = ti.Vector.field(3, dtype=ti.f32, shape=1)
        positions[0] = (1.0e-50, 0.0, 0.0)
        external_force[0] = (3.0, 4.0, 5.0)
        markers._scatter_marker_forces_to_mpm_particles_kernel = (
            lambda *_args: self.fail("invalid positions reached force scatter")
        )

        with self.assertRaisesRegex(ValueError, "particle_position_m"):
            markers.scatter_marker_forces_to_mpm_particles(
                external_force,
                positions,
                particle_count=1,
                support_radius_m=1.0,
                particle_position_generation=4,
            )

        np.testing.assert_array_equal(
            external_force.to_numpy(),
            np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32),
        )
        self.assertEqual(markers._mpm_bin_particle_capacity, 0)
        self.assertIsNone(markers._mpm_particle_bin_cache_source)
        self.assertIsNone(markers._mpm_particle_bin_cache_key)

    def test_device_loader_checks_source_capacity_before_geometry_write(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_capacity = 2
        markers._begin_marker_geometry_write = lambda: self.fail(
            "an undersized source field started a geometry transaction"
        )
        short_field = _CountedField(1)

        with self.assertRaisesRegex(ValueError, "surface_position_m.*capacity"):
            markers.load_markers_from_surface_fields(
                short_field,
                _CountedField(2),
                _CountedField(2),
                _CountedField(2),
                marker_count=2,
            )

    def test_invalid_device_loader_is_atomic(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_capacity = 2
        markers.marker_count = 1
        markers.projection_vertex_count = 1
        markers.projection_triangle_count = 1
        markers.projection_segment_count = 1
        markers.marker_geometry_revision = 4
        markers.report_surface_field_load_invalid_marker_count = _ScalarField()
        markers.report_surface_field_load_marker_count = _ScalarField()
        markers._open_ribbon_tip_cap_binding = ("old",)

        def begin_write():
            markers.marker_geometry_revision += 1

        def reject_fields(*_args):
            markers.report_surface_field_load_invalid_marker_count[None] = 1

        markers._begin_marker_geometry_write = begin_write
        markers._validate_surface_fields_kernel = reject_fields
        markers._load_markers_from_surface_fields_kernel = reject_fields
        markers.projection_vertex_pressure_owner_index = SimpleNamespace(
            from_numpy=lambda _values: None
        )
        markers.reset_stress_diagnostics = lambda _count: None
        field = _CountedField(2)

        with self.assertRaisesRegex(ValueError, "surface fields"):
            markers.load_markers_from_surface_fields(
                field,
                field,
                field,
                field,
                marker_count=2,
            )

        self.assertEqual(markers.marker_count, 1)
        self.assertEqual(markers.projection_vertex_count, 1)
        self.assertEqual(markers.projection_triangle_count, 1)
        self.assertEqual(markers.projection_segment_count, 1)
        self.assertEqual(markers.marker_geometry_revision, 4)
        self.assertEqual(markers._open_ribbon_tip_cap_binding, ("old",))

    def test_device_loader_commit_failure_is_fail_closed(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_capacity = 2
        markers.marker_count = 1
        markers.projection_vertex_count = 1
        markers.projection_triangle_count = 1
        markers.projection_segment_count = 1
        markers.report_surface_field_load_invalid_marker_count = _ScalarField()
        markers.report_surface_field_load_marker_count = _ScalarField()
        markers._open_ribbon_tip_cap_binding = ("old",)
        markers._begin_marker_geometry_write = lambda: None
        markers._validate_surface_fields_kernel = lambda *_args: None

        def fail_commit(*_args):
            raise RuntimeError("synthetic device write failure")

        markers._load_markers_from_surface_fields_kernel = fail_commit
        field = _CountedField(2)

        with self.assertRaisesRegex(RuntimeError, "device write"):
            markers.load_markers_from_surface_fields(
                field,
                field,
                field,
                field,
                marker_count=2,
            )

        self.assertEqual(markers.marker_count, 0)
        self.assertEqual(markers.projection_vertex_count, 0)
        self.assertEqual(markers.projection_triangle_count, 0)
        self.assertEqual(markers.projection_segment_count, 0)
        self.assertIsNone(markers._open_ribbon_tip_cap_binding)

    def test_nonfinite_feedback_is_rejected_before_geometry_write(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_count = 1
        markers.marker_geometry_revision = 2
        markers._input_validation_failure_count = _ScalarField()
        markers.report_surface_feedback_updated_marker_count = _ScalarField()
        markers.report_surface_feedback_invalid_marker_count = _ScalarField()
        markers.report_surface_feedback_max_displacement_m = _ScalarField()
        markers.report_surface_feedback_max_speed_mps = _ScalarField()
        markers.report_surface_feedback_candidate_pair_count = _ScalarField()

        def reject_feedback(*_args):
            markers._input_validation_failure_count[None] = 1

        def begin_write():
            markers.marker_geometry_revision += 1

        markers._validate_mpm_feedback_fields_kernel = reject_feedback
        markers._begin_marker_geometry_write = begin_write
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: None
        markers._update_surface_feedback_from_mpm_particles_kernel = (
            lambda *_args: None
        )
        markers._refresh_open_ribbon_tip_cap_projection_vertices = lambda: None
        field = _CountedField(1)

        with self.assertRaisesRegex(ValueError, "finite"):
            markers.update_surface_feedback_from_mpm_particles(
                field,
                field,
                particle_count=1,
                support_radius_m=1.0,
                dt_s=1.0,
            )
        self.assertEqual(markers.marker_geometry_revision, 2)

    def test_feedback_device_failure_retires_active_geometry(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_count = 1
        markers.projection_vertex_count = 1
        markers.projection_triangle_count = 1
        markers.projection_segment_count = 1
        markers.marker_geometry_revision = 2
        markers._open_ribbon_tip_cap_binding = ("old",)
        markers._input_validation_failure_count = _ScalarField()
        markers._mpm_bin_counts = object()
        markers._mpm_bin_offsets = object()
        markers._mpm_bin_members = object()
        markers._mpm_marker_neighbor_slots = object()
        markers._mpm_marker_neighbor_slot_counts = object()
        markers._validate_mpm_feedback_fields_kernel = lambda *_args: None
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: None
        markers._begin_marker_geometry_write = lambda: None

        def fail_feedback(*_args):
            raise RuntimeError("synthetic feedback device failure")

        markers._update_surface_feedback_from_mpm_particles_kernel = fail_feedback
        markers._refresh_open_ribbon_tip_cap_projection_vertices = lambda: None
        field = _CountedField(1)

        with self.assertRaisesRegex(RuntimeError, "feedback device"):
            markers.update_surface_feedback_from_mpm_particles(
                field,
                field,
                particle_count=1,
                support_radius_m=1.0,
                dt_s=1.0,
            )

        self.assertEqual(markers.marker_count, 0)
        self.assertEqual(markers.projection_vertex_count, 0)
        self.assertEqual(markers.projection_triangle_count, 0)
        self.assertEqual(markers.projection_segment_count, 0)
        self.assertIsNone(markers._open_ribbon_tip_cap_binding)


class ParticleBinCacheContracts(unittest.TestCase):
    def test_particle_bins_reuse_only_an_explicit_matching_generation(self) -> None:
        markers = object.__new__(HibmMpmSurfaceMarkers)
        markers.marker_count = 2
        markers.marker_geometry_revision = 7
        markers._mpm_bin_hash_capacity = 32
        markers._mpm_bin_counts = object()
        markers._mpm_bin_offsets = object()
        markers._mpm_bin_write_cursors = object()
        markers._mpm_bin_members = object()
        markers._mpm_marker_neighbor_slots = object()
        markers._mpm_marker_neighbor_slot_counts = object()
        markers._mpm_particle_bin_cache_source = None
        markers._mpm_particle_bin_cache_key = None
        markers._mpm_marker_neighbor_cache_key = None
        markers._input_validation_failure_count = _ScalarField()
        markers._ensure_mpm_particle_bin_workspace = lambda _count: None

        calls = []
        for name in (
            "validate",
            "reset",
            "count",
            "scan",
            "fill",
            "neighbors",
        ):
            setattr(
                markers,
                {
                    "validate": "_validate_mpm_particle_positions_kernel",
                    "reset": "_reset_mpm_particle_bins_kernel",
                    "count": "_count_mpm_particles_per_bin_kernel",
                    "scan": "_scan_mpm_particle_bin_offsets_kernel",
                    "fill": "_fill_mpm_particle_bin_members_kernel",
                    "neighbors": "_build_mpm_marker_neighbor_slots_kernel",
                }[name],
                lambda *_args, name=name: calls.append(name),
            )

        first_positions = object()
        prepare = HibmMpmSurfaceMarkers._prepare_mpm_particle_bins
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
            particle_position_generation=11,
        )
        self.assertEqual(
            calls,
            ["validate", "reset", "count", "scan", "fill", "neighbors"],
        )

        calls.clear()
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
            particle_position_generation=11,
        )
        self.assertEqual(calls, [])

        calls.clear()
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
            particle_position_generation=12,
        )
        self.assertEqual(calls, ["validate", "reset", "count", "scan", "fill"])

        calls.clear()
        markers.marker_geometry_revision += 1
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
            particle_position_generation=12,
        )
        self.assertEqual(calls, ["neighbors"])

        calls.clear()
        prepare(
            markers,
            object(),
            particle_count=3,
            support_radius_m=0.5,
            particle_position_generation=12,
        )
        self.assertEqual(calls, ["validate", "reset", "count", "scan", "fill"])

        calls.clear()
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
        )
        prepare(
            markers,
            first_positions,
            particle_count=3,
            support_radius_m=0.5,
        )
        self.assertEqual(
            calls,
            ["validate", "reset", "count", "scan", "fill"] * 2,
        )


if __name__ == "__main__":
    unittest.main()
