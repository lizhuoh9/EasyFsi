from __future__ import annotations

import unittest

from simulation_core.coupling.hibm_mpm.core import HibmMpmSurfaceMarkers


class _RecordingArrayField:
    def __init__(self, owner, events, *, fail=False):
        self._owner = owner
        self._events = events
        self._fail = bool(fail)

    def from_numpy(self, _values):
        self._events.append(
            (
                "field-write",
                int(self._owner.marker_geometry_revision),
                self._owner._current_no_slip_sampling_identity,
            )
        )
        if self._fail:
            raise RuntimeError("synthetic field write failure")


class _RecordingTriangleField:
    def __init__(self, owner, events, *, fail=False):
        self._owner = owner
        self._events = events
        self._fail = bool(fail)

    def __setitem__(self, index, value):
        self._events.append(
            (
                "triangle-write",
                int(index),
                tuple(int(component) for component in value),
                int(self._owner.marker_geometry_revision),
                self._owner._current_no_slip_sampling_identity,
            )
        )
        if self._fail:
            raise RuntimeError("synthetic triangle write failure")


def _bare_markers(*, revision: int = 7):
    markers = object.__new__(HibmMpmSurfaceMarkers)
    markers.marker_capacity = 2
    markers.marker_count = 1
    markers.projection_vertex_count = 1
    markers.projection_triangle_count = 1
    markers.projection_segment_count = 0
    markers.projection_triangle_capacity = 2
    markers.marker_geometry_revision = int(revision)
    markers._current_no_slip_sampling_identity = object()
    return markers


class HibmMpmMarkerGeometryRevisionTests(unittest.TestCase):
    def test_pure_prevalidation_failures_preserve_revision_and_sampling_identity(self):
        cases = []

        load_markers = _bare_markers()
        cases.append(
            (
                load_markers,
                lambda: load_markers.load_markers(
                    positions_m=((0.0, 0.0, 0.0),),
                    velocities_mps=(),
                    normals=((0.0, 0.0, 1.0),),
                    areas_m2=(1.0,),
                    region_ids=(1,),
                ),
            )
        )

        surface_fields = _bare_markers()
        cases.append(
            (
                surface_fields,
                lambda: surface_fields.load_markers_from_surface_fields(
                    object(),
                    object(),
                    object(),
                    object(),
                    marker_count=3,
                ),
            )
        )

        particle_feedback = _bare_markers()
        cases.append(
            (
                particle_feedback,
                lambda: particle_feedback.update_surface_feedback_from_mpm_particles(
                    object(),
                    object(),
                    particle_count=0,
                    support_radius_m=1.0,
                    dt_s=1.0,
                ),
            )
        )

        surface_feedback = _bare_markers()
        cases.append(
            (
                surface_feedback,
                lambda: surface_feedback.update_surface_feedback_from_mpm_surface_particles(
                    object(),
                    object(),
                    object(),
                    object(),
                    particle_count=0,
                    support_radius_m=1.0,
                    dt_s=1.0,
                ),
            )
        )

        projection_field = _bare_markers()
        cases.append(
            (
                projection_field,
                lambda: projection_field.load_projection_triangles_from_field(
                    object(),
                    triangle_count=3,
                ),
            )
        )

        for markers, invoke in cases:
            with self.subTest(entry_point=invoke):
                identity = markers._current_no_slip_sampling_identity
                with self.assertRaises(ValueError):
                    invoke()
                self.assertEqual(markers.marker_geometry_revision, 7)
                self.assertIs(markers._current_no_slip_sampling_identity, identity)

    def test_projection_triangle_host_validation_precedes_revision_and_field_writes(self):
        markers = _bare_markers()
        markers.marker_count = 3
        markers.projection_vertex_count = 3
        events = []
        markers.projection_triangle_indices = _RecordingTriangleField(
            markers,
            events,
        )
        identity = markers._current_no_slip_sampling_identity

        with self.assertRaisesRegex(ValueError, "projection triangle index"):
            markers.set_projection_triangles(((0, 1, 2), (0, 1, 3)))

        self.assertEqual(events, [])
        self.assertEqual(markers.marker_geometry_revision, 7)
        self.assertIs(markers._current_no_slip_sampling_identity, identity)

    def test_projection_triangle_loader_retires_identity_before_first_write(self):
        markers = _bare_markers()
        markers.marker_count = 3
        markers.projection_vertex_count = 3
        events = []
        markers.projection_triangle_indices = _RecordingTriangleField(
            markers,
            events,
            fail=True,
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic triangle write failure"):
            markers.set_projection_triangles(((0, 1, 2),))

        self.assertEqual(
            events,
            [("triangle-write", 0, (0, 1, 2), 8, None)],
        )
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)

    def test_projection_triangle_field_loader_retires_identity_before_kernel_write(self):
        markers = _bare_markers()
        markers.marker_count = 3
        events = []

        def fail_kernel(*_args):
            events.append(
                (
                    "triangle-field-write",
                    int(markers.marker_geometry_revision),
                    markers._current_no_slip_sampling_identity,
                )
            )
            raise RuntimeError("synthetic triangle field failure")

        markers._load_projection_triangles_from_field_kernel = fail_kernel

        with self.assertRaisesRegex(RuntimeError, "synthetic triangle field failure"):
            markers.load_projection_triangles_from_field(
                object(),
                triangle_count=1,
            )

        self.assertEqual(events, [("triangle-field-write", 8, None)])
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)

    def test_load_markers_retires_identity_before_first_write_and_keeps_revision_on_failure(self):
        markers = _bare_markers()
        events = []
        markers.x_gamma_m = _RecordingArrayField(markers, events, fail=True)
        for name in (
            "pressure_probe_origin_m",
            "pressure_probe_origin_explicit",
            "v_gamma_mps",
            "n_gamma",
            "A_gamma_m2",
            "region_id",
            "t_gamma_pa",
            "F_gamma_n",
        ):
            setattr(markers, name, _RecordingArrayField(markers, events))
        markers.reset_pressure_pair_anchor_cells = lambda: None
        markers.reset_stress_diagnostics = lambda _count=None: None

        with self.assertRaisesRegex(RuntimeError, "synthetic field write failure"):
            markers.load_markers(
                positions_m=((0.0, 0.0, 0.0),),
                velocities_mps=((0.0, 0.0, 0.0),),
                normals=((0.0, 0.0, 1.0),),
                areas_m2=(1.0,),
                region_ids=(1,),
            )

        self.assertEqual(events, [("field-write", 8, None)])
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)

    def test_surface_field_load_retires_identity_before_kernel_write(self):
        markers = _bare_markers()
        events = []

        def fail_kernel(*_args):
            events.append(
                (
                    "surface-load",
                    int(markers.marker_geometry_revision),
                    markers._current_no_slip_sampling_identity,
                )
            )
            raise RuntimeError("synthetic surface load failure")

        markers._load_markers_from_surface_fields_kernel = fail_kernel
        markers.reset_stress_diagnostics = lambda _count=None: None

        with self.assertRaisesRegex(RuntimeError, "synthetic surface load failure"):
            markers.load_markers_from_surface_fields(
                object(),
                object(),
                object(),
                object(),
                marker_count=1,
            )

        self.assertEqual(events, [("surface-load", 8, None)])
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)

    def test_particle_feedback_retires_identity_before_geometry_kernel(self):
        markers = _bare_markers()
        events = []
        markers._require_particle_field_capacity = lambda *_args: None
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: events.append(
            ("prepare", markers.marker_geometry_revision)
        )
        for name in (
            "_mpm_bin_counts",
            "_mpm_bin_offsets",
            "_mpm_bin_members",
            "_mpm_marker_neighbor_slots",
            "_mpm_marker_neighbor_slot_counts",
        ):
            setattr(markers, name, object())

        def fail_kernel(*_args):
            events.append(
                (
                    "feedback-write",
                    int(markers.marker_geometry_revision),
                    markers._current_no_slip_sampling_identity,
                )
            )
            raise RuntimeError("synthetic feedback failure")

        markers._update_surface_feedback_from_mpm_particles_kernel = fail_kernel

        with self.assertRaisesRegex(RuntimeError, "synthetic feedback failure"):
            markers.update_surface_feedback_from_mpm_particles(
                object(),
                object(),
                particle_count=1,
                support_radius_m=1.0,
                dt_s=1.0,
            )

        self.assertEqual(events, [("prepare", 8), ("feedback-write", 8, None)])
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)

    def test_surface_particle_feedback_retires_identity_before_geometry_kernel(self):
        markers = _bare_markers()
        events = []
        markers._require_particle_field_capacity = lambda *_args: None
        markers._prepare_mpm_particle_bins = lambda *_args, **_kwargs: events.append(
            ("prepare", markers.marker_geometry_revision)
        )
        for name in (
            "_mpm_bin_counts",
            "_mpm_bin_offsets",
            "_mpm_bin_members",
            "_mpm_marker_neighbor_slots",
            "_mpm_marker_neighbor_slot_counts",
        ):
            setattr(markers, name, object())

        def fail_kernel(*_args):
            events.append(
                (
                    "surface-feedback-write",
                    int(markers.marker_geometry_revision),
                    markers._current_no_slip_sampling_identity,
                )
            )
            raise RuntimeError("synthetic surface feedback failure")

        markers._update_surface_feedback_from_mpm_surface_particles_kernel = fail_kernel

        with self.assertRaisesRegex(RuntimeError, "synthetic surface feedback failure"):
            markers.update_surface_feedback_from_mpm_surface_particles(
                object(),
                object(),
                object(),
                object(),
                particle_count=1,
                support_radius_m=1.0,
                dt_s=1.0,
            )

        self.assertEqual(
            events,
            [("prepare", 8), ("surface-feedback-write", 8, None)],
        )
        self.assertEqual(markers.marker_geometry_revision, 8)
        self.assertIsNone(markers._current_no_slip_sampling_identity)


if __name__ == "__main__":
    unittest.main()
