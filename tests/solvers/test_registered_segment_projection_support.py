"""Strict-CUDA contracts for bounded face-global geometric aggregation."""

import math

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm.component_face_segment_audit import RegisteredSegmentAudit
from tests.solvers.test_registered_segment_path_audit import _configure_case, _cyclic, path_case


@ti.data_oriented
class _SupportProbe(RegisteredSegmentAudit):
    @ti.kernel
    def aggregation(
        self, point: ti.types.vector(3, ti.f64), center: ti.types.vector(3, ti.f64),
        inactive_axis: ti.i32, anisotropic: ti.i32, radius: ti.types.vector(3, ti.f64),
    ) -> ti.i32:
        return self._audit_in_aggregation_support(point, center, inactive_axis, anisotropic, radius)

    @ti.kernel
    def strict(
        self, point: ti.types.vector(3, ti.f64), center: ti.types.vector(3, ti.f64),
        inactive_axis: ti.i32, anisotropic: ti.i32, radius: ti.types.vector(3, ti.f64),
    ) -> ti.i32:
        return self._audit_in_support(point, center, inactive_axis, anisotropic, radius)


@pytest.fixture(scope="module")
def probe(path_case):
    # Reuse the existing strict-CUDA runtime, never reset it during collection.
    assert path_case["assembler"] is not None
    return _SupportProbe()


def _v(pair):
    return ti.Vector([pair[0], pair[1], 0.0], dt=ti.f64)


def _aggregation(probe, point, *, radius=(1.0, 1.0), anisotropic=1):
    return int(probe.aggregation(
        _v(point), _v((0.0, 0.0)), 2, anisotropic,
        ti.Vector([*radius, 1.0], dt=ti.f64),
    ))


@pytest.mark.parametrize(("point", "expected"), (
    ((0.0, 0.0), 1),
    ((0.5, -0.75), 1),
    ((1.0, 0.0), 1),     # old source-box boundary is not the geometry boundary
    ((-1.0, 0.0), 1),
    ((1.125, 0.4375), 1),
    ((1.125, 0.5), 1),   # the superseded projection hull facet is now interior
    ((1.125, 0.625), 1), # smooth connectors need not lie in that hull
    ((1.25, 0.0), 1),
    ((1.0, 1.0), 0),     # strict circumscribed-disk boundary
    ((1.5, 0.0), 0),
))
def test_geometry_uses_smallest_strict_disk_containing_source_box(probe, point, expected):
    assert _aggregation(probe, point) == expected


@pytest.mark.parametrize(("point", "expected"), (
    ((4.875, 0.0), 1), ((0.0, 4.875), 1),
    ((3.0, 4.0), 0), ((5.0, 0.0), 0),
    ((0.0, -5.0), 0), ((5.25, 0.0), 0),
))
def test_exact_dyadic_disk_boundary_has_no_tolerance_expansion(probe, point, expected):
    assert _aggregation(probe, point, radius=(3.0, 4.0)) == expected


@pytest.mark.parametrize("inactive_axis", (0, 1, 2))
@pytest.mark.parametrize(("sx", "sy"), ((1, 1), (1, -1), (-1, 1), (-1, -1)))
def test_aggregation_disk_axis_reflection_and_inactive_coordinate_symmetry(probe, inactive_axis, sx, sy):
    def embed(pair, inactive_value=0.0):
        return ti.Vector(_cyclic((pair[0], pair[1], inactive_value), inactive_axis), dt=ti.f64)

    center = embed((0.0, 0.0), 4.0)
    radius = embed((3.0, 4.0), 100.0)
    assert int(probe.aggregation(embed((4.0 * sx, 2.0 * sy), -4.0), center, inactive_axis, 1, radius)) == 1
    assert int(probe.aggregation(embed((3.0 * sx, 4.0 * sy)), center, inactive_axis, 1, radius)) == 0
    assert int(probe.aggregation(embed((4.0 * sx, 4.0 * sy)), center, inactive_axis, 1, radius)) == 0


@pytest.mark.parametrize("field_index", (0, 1))
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_nonfinite_point_or_center_rejects_even_inactive_axis(probe, field_index, axis, bad):
    values = [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
    values[field_index][axis] = bad
    assert int(probe.aggregation(
        *(ti.Vector(value, dt=ti.f64) for value in values), 2, 1,
        ti.Vector([1.0, 1.0, 1.0], dt=ti.f64),
    )) == 0


@pytest.mark.parametrize("inactive_axis", (0, 1, 2))
@pytest.mark.parametrize("point", ((0.5, 0.5), (0.9, 0.9), (1.0, 0.0)))
def test_scalar_disk_is_exactly_the_original_support(probe, inactive_axis, point):
    radius = ti.Vector([1.0, 7.0, 9.0], dt=ti.f64)
    center = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    query = ti.Vector(_cyclic((*point, 0.0), inactive_axis), dt=ti.f64)
    assert int(probe.aggregation(query, center, inactive_axis, 0, radius)) == int(
        probe.strict(query, center, inactive_axis, 0, radius)
    )


# Exact F32 values from the 53-array reproduce32c capture, vertices 14/15/16.
_POSITIONS = np.asarray((
    (0.001500000013038516, 0.0020830794237554073, 0.04697826877236366),
    (0.001500000013038516, 0.002937779063358903, 0.04687279090285301),
    (0.001500000013038516, 0.003763167653232813, 0.046743545681238174),
), dtype=np.float32)
_NORMALS = np.asarray((
    (3.7927523521830153e-7, -0.045641738921403885, -0.9989578723907471),
    (7.338978207371838e-7, -0.13521146774291992, -0.9908167719841003),
    (9.918193200064707e-7, -0.1527099311351776, -0.9882710576057434),
), dtype=np.float32)
_VELOCITIES = np.asarray((
    (3.341068932627422e-8, 0.0020245122723281384, -0.00444862199947238),
    (5.081584220079094e-8, 0.009820814244449139, -0.025358954444527626),
    (6.42504858205939e-8, 0.010942310094833374, -0.05005955696105957),
), dtype=np.float32)
_FACE = (0.000375000003259629, 0.0024999999441206455, 0.04453124850988388)
_SOURCE = (0.000375000003259629, 0.0021875000093132257, 0.04453124850988388)
_ANCHOR = (0.000375000003259629, 0.002937779063358903, 0.04687279090285301)
_SAMPLE = (0.000375000003259629, 0.002199359703809023, 0.04215708374977112)
_RAW_NORMAL = (0.0, -0.1547020673751831, -0.9879611730575562)
_RADIUS = (0.0012000000000000001, 0.003125, 0.0023437500000000003)


def _capture_case(case, inactive_axis, reverse_storage, split_owner):
    def rotate(value):
        return _cyclic((value[1], value[2], value[0]), inactive_axis)

    points, normals, velocities = _POSITIONS, _NORMALS, _VELOCITIES
    if split_owner:
        # Split the actual owner between P and vertex 15: this new connector
        # is outside D but inside its circumscribed disk.
        parameter = (1.0 + 0.828502700046922) * 0.5
        def split(values):
            added = np.asarray((1.0 - parameter) * values[0].astype(np.float64)
                               + parameter * values[1].astype(np.float64), np.float32)
            return np.asarray((values[0], added, values[1], values[2]), np.float32)
        points, normals, velocities = (split(values) for values in (points, normals, velocities))
    count = len(points)
    raw_first = count - 2
    component_axis = (inactive_axis + 1) % 3
    edges = [(index, index + 1) for index in range(count - 1)]
    if reverse_storage:
        edges = [(second, first) for first, second in edges]
    scan = _configure_case(
        case, points=[rotate(row) for row in points] + [(0.0, 0.0, 0.0)] * (6 - count),
        normals=[rotate(row) for row in normals] + [(0.0, 0.0, 0.0)] * (6 - count),
        segments=edges, inactive_axis=inactive_axis, component_axis=component_axis,
        owner_face=rotate(_FACE), source_center=rotate(_SOURCE),
        source_anchor=rotate(_ANCHOR), source_sample=rotate(_SAMPLE),
    )
    case["velocities"].from_numpy(np.asarray(
        [rotate(row) for row in velocities] + [(0.0, 0.0, 0.0)] * (6 - count), np.float32,
    ))
    case["regions"].fill(202)
    assembler = case["assembler"]
    key = (0, 0, 0, component_axis)
    assembler.raw_route_primitive[key] = ((raw_first + 1, raw_first, -1)
                                          if reverse_storage else (raw_first, raw_first + 1, -1))
    assembler.raw_route_weights[key] = (0.0, 1.0, 0.0) if reverse_storage else (1.0, 0.0, 0.0)
    assembler.raw_route_region[key] = 202
    assembler.raw_route_boundary_target_mps[key] = float(_VELOCITIES[1, 1])
    assembler.raw_route_normal[key] = rotate(_RAW_NORMAL)
    assembler.scan_registered_active_faces_device(**scan)
    return scan, key, rotate(_RADIUS)


@pytest.mark.parametrize("inactive_axis", (0, 1, 2))
@pytest.mark.parametrize("reverse_storage", (False, True))
@pytest.mark.parametrize("split_owner", (False, True))
def test_captured_global_owner_and_connected_path_are_projection_closed(
    path_case, probe, inactive_axis, reverse_storage, split_owner,
):
    scan, key, radius = _capture_case(path_case, inactive_axis, reverse_storage, split_owner)
    assembler = path_case["assembler"]
    assert int(assembler.owner_segment_index[key]) == 0
    assert tuple(assembler.owner_segment[key]) == (0, 1)
    face = ti.Vector(_cyclic((_FACE[1], _FACE[2], _FACE[0]), inactive_axis), dt=ti.f64)
    owner = assembler.owner_point_m[key]
    assert int(probe.strict(owner, face, inactive_axis, 1, radius)) == 0
    if split_owner:
        assert int(probe.strict(path_case["positions"][1], face, inactive_axis, 1, radius)) == 0
    assembler.certify_active_raw_routes_device(
        expected_generation=31, support_available=1, support_anisotropic=1,
        strict_support_radius_xyz_m=radius, marker_normal_m=path_case["normals"],
        marker_role=path_case["roles"], **scan,
    )
    assert int(assembler._audited_owner_failure[key]) == 0
    assert int(assembler.raw_route_audit_failure[key]) == 0
    assert int(assembler.audit_valid[key]) == 1
    assert int(assembler.audit_raw_count[key]) == 1
    assert int(assembler.audit_rejection_count[None]) == 0


@pytest.mark.parametrize(("face_y", "source_y", "anchor_y"), (
    (0.25, 0.0, 1.0),  # anchor on original source-box boundary
    (0.0, 1.0, 0.5),   # source center on original face-box boundary
    (0.0, 0.5, 1.0),   # anchor on original face-box boundary
))
def test_all_three_raw_open_box_boundaries_remain_fail_closed(path_case, face_y, source_y, anchor_y):
    scan = _configure_case(
        path_case,
        points=[(-2.0, anchor_y, 0.5), (2.0, anchor_y, 0.5)] + [(0.0, 0.0, 0.0)] * 4,
        normals=[(0.0, -1.0, 0.0)] * 6, segments=[(0, 1)],
        inactive_axis=2, component_axis=1,
        owner_face=(0.0, face_y, 0.5), source_center=(0.0, source_y, 0.5),
        source_anchor=(0.0, anchor_y, 0.5), source_sample=(0.0, anchor_y - 0.1, 0.5),
    )
    assembler = path_case["assembler"]
    assembler.raw_route_normal[0, 0, 0, 1] = (0.0, -1.0, 0.0)
    assembler.certify_active_raw_routes_device(
        expected_generation=31, support_available=1, support_anisotropic=1,
        strict_support_radius_xyz_m=(1.0, 1.0, 1.0),
        marker_normal_m=path_case["normals"], marker_role=path_case["roles"], **scan,
    )
    assert int(assembler.raw_route_audit_failure[0, 0, 0, 1]) == 4
    assert int(assembler.audit_valid[0, 0, 0, 1]) == 0
    assert int(assembler.audit_rejection_count[None]) > 0
