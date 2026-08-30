"""Focused device-contract tests for registered finite-segment assembly."""

import math

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm.component_face_segment_assembly import (
    RegisteredComponentFaceSegmentAssembler,
)

def _runtime() -> object:
    return None


def test_registered_segment_assembler_exposes_source_axis_capacity() -> None:
    """The route must retain one record per source and component axis."""

    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(3, 4, 2),
        marker_capacity=8,
        runtime=_runtime(),
    )

    assert assembler.source_axis_record_capacity == 3 * 4 * 2 * 3
    assert tuple(assembler.raw_route_valid.shape) == (3, 4, 2, 3)


def test_registered_segment_assembler_gpu_owner_scan_rejects_nearer_invalid() -> None:
    """A nearer unsupported segment blocks, rather than promoting a farther one."""

    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(2, 2, 1),
        marker_capacity=4,
        runtime=_runtime(),
    )
    marker_position = ti.Vector.field(3, dtype=ti.f32, shape=4)
    marker_velocity = ti.Vector.field(3, dtype=ti.f32, shape=4)
    marker_region = ti.field(dtype=ti.i32, shape=4)
    segments = ti.Vector.field(3, dtype=ti.i32, shape=2)
    marker_position.from_numpy(
        __import__("numpy").array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
            dtype="float32",
        )
    )
    marker_velocity.fill(0.0)
    marker_region.fill(0)
    segments.from_numpy(__import__("numpy").array([[0, 1, -1], [2, 3, -1]], dtype="int32"))

    assembler.clear_device_transaction()
    assembler.scan_registered_owner_device(
        face=(1, 1, 0),
        component_axis=0,
        inactive_axis=2,
        face_center=(0.5, 0.01, 0.0),
        projection_segment_indices=segments,
        projection_segment_count=2,
        marker_position_m=marker_position,
        marker_velocity_mps=marker_velocity,
        marker_region_id=marker_region,
        projection_vertex_count=4,
    )

    assert int(assembler.owner_valid[1, 1, 0, 0]) == 1
    assert tuple(int(value) for value in assembler.owner_segment[1, 1, 0, 0]) == (0, 1)


def test_registered_segment_gpu_scan_blocks_nearer_invalid_owner() -> None:
    """A malformed nearer owner must not make a farther edge eligible."""

    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(2, 2, 1),
        marker_capacity=4,
        runtime=_runtime(),
    )
    marker_position = ti.Vector.field(3, dtype=ti.f32, shape=4)
    marker_velocity = ti.Vector.field(3, dtype=ti.f32, shape=4)
    marker_region = ti.field(dtype=ti.i32, shape=4)
    segments = ti.Vector.field(3, dtype=ti.i32, shape=2)
    marker_position.from_numpy(
        __import__("numpy").array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
            dtype="float32",
        )
    )
    marker_velocity.fill(0.0)
    marker_region.from_numpy(
        __import__("numpy").array([0, 1, 0, 0], dtype="int32")
    )
    segments.from_numpy(
        __import__("numpy").array([[0, 1, -1], [2, 3, -1]], dtype="int32")
    )

    assembler.clear_device_transaction()
    assembler.scan_registered_owner_device(
        face=(1, 1, 0), component_axis=0, inactive_axis=2,
        face_center=(0.5, 0.01, 0.0), projection_segment_indices=segments,
        projection_segment_count=2, marker_position_m=marker_position,
        marker_velocity_mps=marker_velocity, marker_region_id=marker_region,
        projection_vertex_count=4,
    )

    assert int(assembler.owner_valid[1, 1, 0, 0]) == 0
    assert int(assembler.owner_blocked[1, 1, 0, 0]) == 1
    assert tuple(int(value) for value in assembler.owner_segment[1, 1, 0, 0]) == (0, 1)


def _certify_local_path(*, segments, points, normals, source_segment, owner_segment, support_radius=8.0, aliases=()):
    assembler = RegisteredComponentFaceSegmentAssembler(grid_nodes=(1, 1, 1), marker_capacity=len(points), runtime=_runtime())
    marker_position = ti.Vector.field(3, dtype=ti.f32, shape=len(points))
    registered_segments = ti.Vector.field(3, dtype=ti.i32, shape=len(segments))
    segment_normals = ti.Vector.field(3, dtype=ti.f32, shape=len(segments))
    marker_velocity = ti.Vector.field(3, dtype=ti.f32, shape=len(points))
    marker_role = ti.field(dtype=ti.i32, shape=len(points))
    marker_position.from_numpy(np.asarray(points, dtype=np.float32))
    registered_segments.from_numpy(np.asarray([[first, second, -1] for first, second in segments], dtype=np.int32))
    segment_normals.from_numpy(np.asarray(normals, dtype=np.float32))
    marker_velocity.fill(0.0)
    marker_role.fill(0)
    assembler.install_registered_topology(segments, vertex_count=len(points))
    assembler.install_explicit_endpoint_aliases(aliases)
    assembler.clear_device_transaction()
    assembler.certify_registered_local_path_device(face=(0, 0, 0), component_axis=0, inactive_axis=2, source_segment_index=source_segment, owner_segment_index=owner_segment, face_center=(0.5, 0.0, 0.0), strict_support_radius_m=support_radius, projection_segment_indices=registered_segments, projection_segment_normals=segment_normals, projection_segment_count=len(segments), marker_position_m=marker_position, marker_velocity_mps=marker_velocity, marker_role=marker_role, projection_vertex_count=len(points))
    return assembler


def test_registered_local_path_certificate_accepts_five_edge_chain() -> None:
    """A local chain is not limited to the former one-hop proof."""
    assembler = _certify_local_path(segments=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], points=[(float(index), 0.0, 0.0) for index in range(6)], normals=[(0.0, 1.0, 0.0)] * 5, source_segment=0, owner_segment=4)
    assert int(assembler.certificate_valid[0, 0, 0, 0]) == 1
    assert int(assembler.certificate_path_edge_count[0, 0, 0, 0]) == 5


def test_registered_local_path_certificate_is_storage_reverse_and_mirror_symmetric() -> None:
    """Endpoint storage order and a reflected local embedding keep the proof."""
    assembler = _certify_local_path(segments=[(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)], points=[(-float(index), 0.0, 0.0) for index in range(6)], normals=[(0.0, -1.0, 0.0)] * 5, source_segment=0, owner_segment=4)
    assert int(assembler.certificate_valid[0, 0, 0, 0]) == 1
    assert int(assembler.certificate_path_edge_count[0, 0, 0, 0]) == 5


def test_registered_local_path_certificate_rejects_connected_path_outside_face_support() -> None:
    """Global connectedness cannot authorize an edge outside this face's support."""
    assembler = _certify_local_path(segments=[(0, 1), (1, 2), (2, 3)], points=[(float(index), 0.0, 0.0) for index in range(4)], normals=[(0.0, 1.0, 0.0)] * 3, source_segment=0, owner_segment=2, support_radius=1.1)
    assert int(assembler.certificate_valid[0, 0, 0, 0]) == 0
    assert int(assembler.certificate_blocked[0, 0, 0, 0]) == 1


def test_registered_local_path_certificate_rejects_non_manifold_degree() -> None:
    """A branch is ambiguous before any runtime path may be selected."""
    with pytest.raises(ValueError, match="non-manifold branch"):
        _certify_local_path(segments=[(0, 1), (1, 2), (1, 3)], points=[(float(index), 0.0, 0.0) for index in range(4)], normals=[(0.0, 1.0, 0.0)] * 3, source_segment=0, owner_segment=2)


def test_registered_local_path_certificate_rejects_closed_or_wider_normal_fan() -> None:
    """Neighboring normals may agree while the full path spans a closed half-plane."""
    angles = [0.0, 45.0, 90.0, 135.0, 180.0]
    assembler = _certify_local_path(segments=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], points=[(float(index), 0.0, 0.0) for index in range(6)], normals=[(math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0) for angle in angles], source_segment=0, owner_segment=4)
    assert int(assembler.certificate_valid[0, 0, 0, 0]) == 0
    assert int(assembler.certificate_blocked[0, 0, 0, 0]) == 1


def test_registered_local_path_certificate_uses_only_validated_explicit_cap_alias() -> None:
    """A declared and equal cap endpoint may bridge two registered local edges."""
    assembler = _certify_local_path(segments=[(0, 1), (2, 3)], points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], normals=[(0.0, 1.0, 0.0)] * 2, source_segment=0, owner_segment=1, aliases=[(1, 2)])
    assert int(assembler.certificate_valid[0, 0, 0, 0]) == 1
    assert int(assembler.certificate_path_edge_count[0, 0, 0, 0]) == 2
