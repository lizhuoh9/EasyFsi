"""Strict-CUDA certificates for aggregation locality around a curved bridge."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.solvers.test_registered_segment_path_audit import (
    _configure_case,
    _cyclic,
    path_case,
)


def _clockwise_normal(
    first: tuple[float, float, float], second: tuple[float, float, float],
) -> tuple[float, float, float]:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    assert length > 0.0
    return dy / length, -dx / length, 0.0


def _curve_normals(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    edges = [
        _clockwise_normal(points[index], points[index + 1])
        for index in range(4)
    ]
    return [
        edges[0],
        tuple(edges[0][axis] + edges[1][axis] for axis in range(3)),
        tuple(edges[1][axis] + edges[2][axis] for axis in range(3)),
        tuple(edges[2][axis] + edges[3][axis] for axis in range(3)),
        edges[3],
        (0.0, 1.0, 0.0),
    ]


def _certify_curve(
    case: dict[str, object],
    *,
    inactive_axis: int,
    radii_xy: tuple[float, float],
    height: float,
    reverse_storage: bool,
) -> tuple[object, int]:
    """Build the captured curved local bridge with raw author 3/4.

    Its source checks remain in the original anisotropic boxes.  The selected
    B owner is the globally nearest shared corner on edge 0; only the
    aggregation path leaves the old box through the legal curved connector.
    """
    component_axis = (inactive_axis + 2) % 3
    points = [
        (-0.6, 0.5, 0.5),
        (-0.4, 0.5, 0.5),
        (0.0, height, 0.5),
        (0.5, 0.75, 0.5),
        (0.7, 0.7, 0.5),
        (0.0, 0.0, 0.0),
    ]
    normals = _curve_normals(points)
    reference_segments = [(0, 1), (1, 2), (2, 3), (3, 4)]
    segments = [
        (second, first) if reverse_storage else (first, second)
        for first, second in reference_segments
    ]
    face = (0.0, 0.0, 0.5)
    source_center = (0.0, 0.125, 0.5)
    raw_anchor = points[3]
    edge3_normal = _clockwise_normal(points[3], points[4])
    raw_sample = tuple(
        raw_anchor[axis] + 0.125 * edge3_normal[axis]
        for axis in range(3)
    )
    scan = _configure_case(
        case,
        points=[_cyclic(point, inactive_axis) for point in points],
        normals=[_cyclic(normal, inactive_axis) for normal in normals],
        segments=segments,
        inactive_axis=inactive_axis,
        component_axis=component_axis,
        owner_face=_cyclic(face, inactive_axis),
        source_center=_cyclic(source_center, inactive_axis),
        source_anchor=_cyclic(raw_anchor, inactive_axis),
        source_sample=_cyclic(raw_sample, inactive_axis),
    )
    assembler = case["assembler"]
    key = (0, 0, 0, component_axis)
    if reverse_storage:
        assembler.raw_route_primitive[key] = (4, 3, -1)
        assembler.raw_route_weights[key] = (0.0, 1.0, 0.0)
    else:
        assembler.raw_route_primitive[key] = (3, 4, -1)
        assembler.raw_route_weights[key] = (1.0, 0.0, 0.0)
    assembler.raw_route_anchor_m[key] = _cyclic(raw_anchor, inactive_axis)
    assembler.raw_route_nominal_sample_m[key] = _cyclic(raw_sample, inactive_axis)
    assembler.raw_route_actual_sample_m[key] = _cyclic(raw_sample, inactive_axis)
    assembler.raw_route_normal[key] = _cyclic(edge3_normal, inactive_axis)
    radius = _cyclic((radii_xy[0], radii_xy[1], 1.0), inactive_axis)
    assembler.certify_active_raw_routes_device(
        expected_generation=31,
        support_available=1,
        support_anisotropic=1,
        strict_support_radius_xyz_m=radius,
        marker_normal_m=case["normals"],
        marker_role=case["roles"],
        **scan,
    )
    return assembler, component_axis


@pytest.mark.parametrize("inactive_axis", (0, 1, 2))
@pytest.mark.parametrize("reverse_storage", (False, True))
@pytest.mark.parametrize(
    ("radii_xy", "height", "accepted"),
    (
        ((1.0, 1.0), 1.25, True),
        ((1.0, 1.0), 2.0, False),
        ((3.0, 4.0), 4.875, True),
        ((3.0, 4.0), 5.0, False),
        ((3.0, 4.0), 5.25, False),
    ),
)
def test_registered_segment_aggregation_locality_uses_bounding_disk_for_curved_path(
    path_case: dict[str, object],
    inactive_axis: int,
    reverse_storage: bool,
    radii_xy: tuple[float, float],
    height: float,
    accepted: bool,
) -> None:
    assembler, component_axis = _certify_curve(
        path_case,
        inactive_axis=inactive_axis,
        radii_xy=radii_xy,
        height=height,
        reverse_storage=reverse_storage,
    )
    key = (0, 0, 0, component_axis)

    assert tuple(int(value) for value in assembler.owner_segment[key]) == (0, 1)
    assert int(assembler.owner_vertex[key]) == 1
    assert int(assembler._audited_owner_failure[key]) == 0
    assert int(assembler.raw_route_audit_failure[key]) == (0 if accepted else 7)
    assert int(assembler.audit_valid[key]) == int(accepted)


def test_curved_bridge_fixture_keeps_raw_anchor_in_original_source_box() -> None:
    """The positive change is only aggregation locality, never raw support."""
    anchor = np.asarray((0.5, 0.75), dtype=np.float64)
    source_center = np.asarray((0.0, 0.125), dtype=np.float64)
    radius = np.asarray((1.0, 1.0), dtype=np.float64)
    assert np.all(np.abs(anchor - source_center) < radius)
