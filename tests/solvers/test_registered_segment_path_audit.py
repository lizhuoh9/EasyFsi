"""End-to-end scratch, owner-scan, and raw-route path-audit contracts."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm.component_face_segment_assembly import (
    RegisteredComponentFaceSegmentAssembler,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig


@pytest.fixture(scope="module")
def path_case() -> dict[str, object]:
    """One field bundle exercises Pass A scratch, B, and C without re-JITs."""

    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(1, 1, 1),
        marker_capacity=6,
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )
    return {
        "assembler": assembler,
        "positions": ti.Vector.field(3, dtype=ti.f32, shape=6),
        "velocities": ti.Vector.field(3, dtype=ti.f32, shape=6),
        "normals": ti.Vector.field(3, dtype=ti.f32, shape=6),
        "regions": ti.field(dtype=ti.i32, shape=6),
        "roles": ti.field(dtype=ti.i32, shape=6),
        "segments": ti.Vector.field(3, dtype=ti.i32, shape=5),
        "face_x": ti.field(dtype=ti.f32, shape=2),
        "face_y": ti.field(dtype=ti.f32, shape=2),
        "face_z": ti.field(dtype=ti.f32, shape=2),
        "center_x": ti.field(dtype=ti.f32, shape=1),
        "center_y": ti.field(dtype=ti.f32, shape=1),
        "center_z": ti.field(dtype=ti.f32, shape=1),
    }


def _cyclic(vector: tuple[float, float, float], inactive_axis: int) -> tuple[float, float, float]:
    """Map the z-inactive reference plane onto every declared 2-D plane."""

    x, y, z = vector
    if inactive_axis == 2:
        return x, y, z
    if inactive_axis == 0:
        return z, x, y
    if inactive_axis == 1:
        return y, z, x
    raise AssertionError(f"invalid inactive axis: {inactive_axis}")


def _configure_case(
    case: dict[str, object],
    *,
    points: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    segments: list[tuple[int, int]],
    inactive_axis: int,
    component_axis: int,
    owner_face: tuple[float, float, float],
    source_center: tuple[float, float, float],
    source_anchor: tuple[float, float, float],
    source_sample: tuple[float, float, float],
) -> dict[str, object]:
    """Write one full-A raw record, then execute B's all-active-face scan."""

    assembler = case["assembler"]
    positions = case["positions"]
    velocities = case["velocities"]
    marker_normals = case["normals"]
    regions = case["regions"]
    roles = case["roles"]
    segment_field = case["segments"]
    assembler.clear_device_transaction()
    positions.from_numpy(np.asarray(points, dtype=np.float32))
    velocities.fill(0.0)
    marker_normals.from_numpy(np.asarray(normals, dtype=np.float32))
    regions.fill(0)
    roles.fill(0)
    stored_segments = [(first, second, -1) for first, second in segments]
    stored_segments.extend([(-1, -1, -1)] * (5 - len(stored_segments)))
    segment_field.from_numpy(np.asarray(stored_segments, dtype=np.int32))
    assembler.install_registered_topology(segments, vertex_count=6)
    assembler.install_explicit_endpoint_aliases(())

    for axis, name in enumerate(("face_x", "face_y", "face_z")):
        case[name].from_numpy(np.asarray((owner_face[axis], owner_face[axis]), np.float32))
    for axis, name in enumerate(("center_x", "center_y", "center_z")):
        case[name][0] = source_center[axis]

    key = (0, 0, 0, component_axis)
    assembler.raw_route_valid[key] = 1
    assembler.raw_route_kind[key] = 0
    assembler.raw_route_target[key] = (0, 0, 0)
    assembler.raw_route_region[key] = 0
    assembler.raw_route_primitive[key] = (0, 1, -1)
    assembler.raw_route_weights[key] = (0.5, 0.5, 0.0)
    assembler.raw_route_boundary_target_mps[key] = 0.0
    assembler.raw_route_anchor_m[key] = source_anchor
    assembler.raw_route_nominal_sample_m[key] = source_sample
    assembler.raw_route_actual_sample_m[key] = source_sample
    assembler.raw_route_normal[key] = _cyclic((0.0, 1.0, 0.0), inactive_axis)
    assembler.raw_route_sample_valid[key] = 1
    assembler.raw_route_generation[key] = 31
    counts = [0, 0, 0]
    counts[component_axis] = 1
    assembler.face_raw_count[0, 0, 0] = tuple(counts)

    scan = {
        "inactive_axis": inactive_axis,
        "cell_face_x_m": case["face_x"],
        "cell_face_y_m": case["face_y"],
        "cell_face_z_m": case["face_z"],
        "cell_center_x_m": case["center_x"],
        "cell_center_y_m": case["center_y"],
        "cell_center_z_m": case["center_z"],
        "projection_segment_indices": segment_field,
        "projection_segment_count": len(segments),
        "marker_position_m": positions,
        "marker_velocity_mps": velocities,
        "marker_region_id": regions,
        "projection_vertex_count": 6,
    }
    assembler.scan_registered_active_faces_device(**scan)
    return scan


def _audit(case: dict[str, object], scan: dict[str, object]) -> None:
    case["assembler"].certify_active_raw_routes_device(
        expected_generation=31,
        support_available=1,
        support_anisotropic=0,
        strict_support_radius_xyz_m=(6.0, 6.0, 6.0),
        marker_normal_m=case["normals"],
        marker_role=case["roles"],
        **scan,
    )


@pytest.mark.parametrize(("inactive_axis", "component_axis"), ((2, 1), (0, 2), (1, 0)))
@pytest.mark.parametrize(("mirrored", "reverse_storage"), ((False, False), (True, True)))
def test_full_a_b_c_path_audit_accepts_five_edge_runtime_chain(
    path_case: dict[str, object],
    inactive_axis: int,
    component_axis: int,
    mirrored: bool,
    reverse_storage: bool,
) -> None:
    """A source-to-owner path beyond one hop remains certified in every plane."""

    x_values = [float(index) for index in range(6)]
    if mirrored:
        x_values = [5.0 - value for value in x_values]
    points = [_cyclic((x, 0.0, 0.5), inactive_axis) for x in x_values]
    normal = _cyclic((0.0, 1.0, 0.0), inactive_axis)
    segments = [(index, index + 1) for index in range(5)]
    if reverse_storage:
        segments = [(second, first) for first, second in segments]
    owner_x = 0.5 if mirrored else 4.5
    source_x = 4.5 if mirrored else 0.5
    scan = _configure_case(
        path_case,
        points=points,
        normals=[normal] * 6,
        segments=segments,
        inactive_axis=inactive_axis,
        component_axis=component_axis,
        owner_face=_cyclic((owner_x, 0.1, 0.5), inactive_axis),
        source_center=_cyclic((owner_x, 0.15, 0.5), inactive_axis),
        source_anchor=_cyclic((source_x, 0.0, 0.5), inactive_axis),
        source_sample=_cyclic((source_x, 0.1, 0.5), inactive_axis),
    )
    assembler = path_case["assembler"]
    assert int(assembler.owner_segment_index[0, 0, 0, component_axis]) == 4
    _audit(path_case, scan)
    assert int(assembler.audit_valid[0, 0, 0, component_axis]) == 1
    assert int(assembler.audit_raw_count[0, 0, 0, component_axis]) == 1
    assert int(assembler.audit_rejection_count[None]) == 0


def test_full_a_b_c_path_audit_rejects_disconnected_owner_path(
    path_case: dict[str, object],
) -> None:
    """B may identify a nearest owner, but C rejects a disconnected raw author."""

    scan = _configure_case(
        path_case,
        points=[(float(index), 0.0, 0.5) for index in range(6)],
        normals=[(0.0, 1.0, 0.0)] * 6,
        segments=[(0, 1), (1, 2), (3, 4), (4, 5)],
        inactive_axis=2,
        component_axis=1,
        owner_face=(4.5, 0.1, 0.5),
        source_center=(4.5, 0.15, 0.5),
        source_anchor=(0.5, 0.0, 0.5),
        source_sample=(0.5, 0.1, 0.5),
    )
    assembler = path_case["assembler"]
    assert int(assembler.owner_segment_index[0, 0, 0, 1]) == 3
    _audit(path_case, scan)
    assert int(assembler.audit_valid[0, 0, 0, 1]) == 0
    assert int(assembler.audit_rejection_count[None]) > 0


def test_full_a_b_c_path_audit_rejects_folded_normal_fan(
    path_case: dict[str, object],
) -> None:
    """A connected path whose normal fan closes a half-plane is not admissible."""

    scan = _configure_case(
        path_case,
        points=[
            (0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5),
            (0.0, 1.0, 0.5), (0.0, 2.0, 0.5), (1.0, 2.0, 0.5),
        ],
        normals=[
            (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        ],
        segments=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        inactive_axis=2,
        component_axis=1,
        owner_face=(0.5, 2.1, 0.5),
        source_center=(0.5, 2.15, 0.5),
        source_anchor=(0.5, 0.0, 0.5),
        source_sample=(0.5, 0.1, 0.5),
    )
    assembler = path_case["assembler"]
    assert int(assembler.owner_segment_index[0, 0, 0, 1]) == 4
    _audit(path_case, scan)
    assert int(assembler.audit_valid[0, 0, 0, 1]) == 0
    assert int(assembler.audit_rejection_count[None]) > 0
