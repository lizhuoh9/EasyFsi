"""Finite registered-segment geometry shared by the MAC face assembler.

Runtime geometry intentionally stays in Taichi functions.  The small host
topology validator is only used while installing integer connectivity.
"""

from dataclasses import dataclass
from typing import Sequence

import taichi as ti


@dataclass(frozen=True)
class RegisteredSegmentTopology:
    """Validated integer topology for the restricted 2-D registered route."""

    segments: tuple[tuple[int, int], ...]
    degree: tuple[int, ...]
    adjacency: tuple[tuple[int, int], ...]


def build_registered_segment_topology(
    segment_indices: Sequence[Sequence[int]],
    *,
    vertex_count: int,
) -> RegisteredSegmentTopology:
    """Validate closed finite-edge registration without inspecting geometry."""

    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    canonical: list[tuple[int, int]] = []
    degree = [0] * vertex_count
    for segment in segment_indices:
        if len(segment) != 2:
            raise ValueError("registered segment must have two endpoints")
        first, second = int(segment[0]), int(segment[1])
        if first == second or first < 0 or second < 0:
            raise ValueError("registered segment endpoints must be distinct")
        if first >= vertex_count or second >= vertex_count:
            raise ValueError("registered segment endpoint is out of range")
        edge = (min(first, second), max(first, second))
        canonical.append(edge)
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    if len(set(canonical)) != len(canonical):
        raise ValueError("registered segments must be unique")
    if any(item > 2 for item in degree):
        raise ValueError("registered segments contain a non-manifold branch")
    adjacency = [[-1, -1] for _ in range(vertex_count)]
    for segment_index, (first, second) in enumerate(canonical):
        for vertex in (first, second):
            if adjacency[vertex][0] == -1:
                adjacency[vertex][0] = segment_index
            else:
                adjacency[vertex][1] = segment_index
    return RegisteredSegmentTopology(
        tuple(canonical),
        tuple(degree),
        tuple((items[0], items[1]) for items in adjacency),
    )


@ti.func
def finite_segment_projection_2d(
    face_center,
    position_a,
    position_b,
    inactive_axis: ti.i32,
):
    """Return a clamped F64 finite-segment projection in the active plane."""

    valid = 1
    a = ti.Vector([
        ti.cast(position_a.x, ti.f64),
        ti.cast(position_a.y, ti.f64),
        ti.cast(position_a.z, ti.f64),
    ])
    b = ti.Vector([
        ti.cast(position_b.x, ti.f64),
        ti.cast(position_b.y, ti.f64),
        ti.cast(position_b.z, ti.f64),
    ])
    point = ti.Vector([
        ti.cast(face_center.x, ti.f64),
        ti.cast(face_center.y, ti.f64),
        ti.cast(face_center.z, ti.f64),
    ])
    chord = b - a
    offset = point - a
    chord[inactive_axis] = 0.0
    offset[inactive_axis] = 0.0
    length_squared = chord.dot(chord)
    if (
        ti.math.isnan(length_squared)
        or ti.math.isinf(length_squared)
        or length_squared <= 1.0e-24
    ):
        valid = 0
    parameter = ti.cast(0.0, ti.f64)
    closest = a
    distance_squared = ti.cast(1.0e30, ti.f64)
    if valid != 0:
        parameter = ti.min(ti.max(offset.dot(chord) / length_squared, 0.0), 1.0)
        closest = a + parameter * (b - a)
        closest[inactive_axis] = point[inactive_axis]
        residual = point - closest
        residual[inactive_axis] = 0.0
        distance_squared = ti.max(residual.dot(residual), 0.0)
        if ti.math.isnan(distance_squared) or ti.math.isinf(distance_squared):
            valid = 0
    return valid, parameter, closest, distance_squared


@ti.func
def registered_endpoint_in_strict_face_support_2d(
    face_center,
    endpoint,
    inactive_axis: ti.i32,
    strict_support_radius_m: ti.f64,
):
    """Test finite endpoint support in the active plane without host copies."""

    point = ti.Vector([
        ti.cast(endpoint.x, ti.f64),
        ti.cast(endpoint.y, ti.f64),
        ti.cast(endpoint.z, ti.f64),
    ])
    center = ti.Vector([
        ti.cast(face_center.x, ti.f64),
        ti.cast(face_center.y, ti.f64),
        ti.cast(face_center.z, ti.f64),
    ])
    valid = ti.cast(strict_support_radius_m > 0.0, ti.i32)
    for axis in ti.static(range(3)):
        if ti.math.isnan(point[axis]) or ti.math.isinf(point[axis]):
            valid = 0
    delta = point - center
    delta[inactive_axis] = 0.0
    if delta.dot(delta) >= strict_support_radius_m * strict_support_radius_m:
        valid = 0
    return valid


@ti.func
def registered_normal_in_active_plane_2d(normal, inactive_axis: ti.i32):
    """Return an F64 finite active-plane normal and its validity flag."""

    projected = ti.Vector([
        ti.cast(normal.x, ti.f64),
        ti.cast(normal.y, ti.f64),
        ti.cast(normal.z, ti.f64),
    ])
    projected[inactive_axis] = 0.0
    squared = projected.dot(projected)
    valid = ti.cast(squared > 1.0e-24, ti.i32)
    for axis in ti.static(range(3)):
        if ti.math.isnan(projected[axis]) or ti.math.isinf(projected[axis]):
            valid = 0
    if valid != 0:
        projected = projected / ti.sqrt(squared)
    return valid, projected
