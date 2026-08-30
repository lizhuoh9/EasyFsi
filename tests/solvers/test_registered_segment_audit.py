"""Focused device contracts for the full-source registered segment audit."""

from __future__ import annotations

import pytest
import numpy as np
import taichi as ti

from simulation_core.coupling.hibm_mpm.component_face_segment_assembly import (
    RegisteredComponentFaceSegmentAssembler,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig


@pytest.fixture(scope="module")
def audit_case() -> dict[str, object]:
    """Allocate one strict-CUDA field bundle, reused by all focused cases."""

    assembler = RegisteredComponentFaceSegmentAssembler(
        grid_nodes=(5, 1, 1),
        marker_capacity=4,
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )
    fields: dict[str, object] = {
        "assembler": assembler,
        "positions": ti.Vector.field(3, dtype=ti.f32, shape=4),
        "velocities": ti.Vector.field(3, dtype=ti.f32, shape=4),
        "normals": ti.Vector.field(3, dtype=ti.f32, shape=4),
        "regions": ti.field(dtype=ti.i32, shape=4),
        "roles": ti.field(dtype=ti.i32, shape=4),
        "segments": ti.Vector.field(3, dtype=ti.i32, shape=2),
        "face_x": ti.field(dtype=ti.f32, shape=6),
        "center_x": ti.field(dtype=ti.f32, shape=5),
        "face_y": ti.field(dtype=ti.f32, shape=2),
        "center_y": ti.field(dtype=ti.f32, shape=1),
        "face_z": ti.field(dtype=ti.f32, shape=2),
        "center_z": ti.field(dtype=ti.f32, shape=1),
    }
    fields["face_x"].from_numpy(np.asarray((0, .2, .4, .6, .8, 1), np.float32))
    fields["center_x"].from_numpy(np.asarray((.1, .3, .5, .7, .9), np.float32))
    fields["face_y"].from_numpy(np.asarray((.1, .2), np.float32))
    fields["center_y"][0] = .15
    fields["face_z"].from_numpy(np.asarray((0, 1), np.float32))
    fields["center_z"][0] = .5
    return fields


def _prepare(
    case: dict[str, object],
    *,
    reverse_storage: bool = False,
    mirrored: bool = False,
    mutation: str | None = None,
) -> dict[str, object]:
    """Make five legal y-axis raw authors, then run Pass B before Pass C."""

    assembler = case["assembler"]
    positions = case["positions"]
    velocities = case["velocities"]
    normals = case["normals"]
    regions = case["regions"]
    roles = case["roles"]
    segments = case["segments"]
    face_x = case["face_x"]
    center_x = case["center_x"]
    face_y = case["face_y"]
    center_y = case["center_y"]
    face_z = case["face_z"]
    center_z = case["center_z"]

    left, right = (
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if mirrored
        else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    positions.from_numpy(
        np.asarray((left, right, (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)), np.float32)
    )
    velocities.fill(0.0)
    normals.from_numpy(
        np.asarray(((0.0, 1.0, 0.0),) * 4, dtype=np.float32)
    )
    regions.fill(0)
    roles.fill(0)
    segments.from_numpy(
        np.asarray(
            ((1, 0, -1) if reverse_storage else (0, 1, -1), (2, 3, -1)),
            np.int32,
        )
    )
    assembler.install_registered_topology(((0, 1), (2, 3)), vertex_count=4)
    assembler.install_explicit_endpoint_aliases(())
    assembler.clear_device_transaction()

    for source_i in range(5):
        key = (source_i, 0, 0, 1)
        assembler.raw_route_valid[key] = 1
        assembler.raw_route_kind[key] = 0
        assembler.raw_route_target[key] = (2, 0, 0)
        assembler.raw_route_region[key] = 0
        assembler.raw_route_primitive[key] = (0, 1, -1)
        assembler.raw_route_weights[key] = (0.5, 0.5, 0.0)
        assembler.raw_route_boundary_target_mps[key] = 0.0
        assembler.raw_route_anchor_m[key] = (0.5, 0.0, 0.0)
        assembler.raw_route_normal[key] = (0.0, 1.0, 0.0)
        assembler.raw_route_nominal_sample_m[key] = (0.5, 0.1, 0.0)
        assembler.raw_route_actual_sample_m[key] = (0.5, 0.1, 0.0)
        assembler.raw_route_sample_valid[key] = 1
        assembler.raw_route_generation[key] = 19
    assembler.face_raw_count[2, 0, 0] = (0, 5, 0)

    scan = dict(
        inactive_axis=2,
        cell_face_x_m=face_x,
        cell_face_y_m=face_y,
        cell_face_z_m=face_z,
        cell_center_x_m=center_x,
        cell_center_y_m=center_y,
        cell_center_z_m=center_z,
        projection_segment_indices=segments,
        projection_segment_count=2,
        marker_position_m=positions,
        marker_velocity_mps=velocities,
        marker_region_id=regions,
        projection_vertex_count=4,
    )
    assembler.scan_registered_active_faces_device(**scan)

    if mutation == "stale_generation":
        assembler.raw_route_generation[0, 0, 0, 1] = 18
    elif mutation == "target_out_of_bounds":
        assembler.raw_route_target[0, 0, 0, 1] = (99, 0, 0)
    elif mutation == "anchor_outside_source_support":
        assembler.raw_route_anchor_m[0, 0, 0, 1] = (3.0, 0.0, 0.0)
    elif mutation == "unregistered_primitive":
        assembler.raw_route_primitive[0, 0, 0, 1] = (0, 2, -1)
    elif mutation == "corrupted_boundary_target":
        assembler.raw_route_boundary_target_mps[0, 0, 0, 1] = 1.0
    elif mutation == "owner_target_mismatch":
        assembler.owner_target_mps[2, 0, 0, 1] = 0.1
    elif mutation == "owner_point_nonfinite":
        assembler.owner_point_m[2, 0, 0, 1] = (9.0, 9.0, 9.0)
    elif mutation == "raw_normal_not_ray_aligned":
        assembler.raw_route_normal[0, 0, 0, 1] = (.6, .8, 0.0)
    elif mutation == "nominal_sample_nonfinite":
        assembler.raw_route_nominal_sample_m[0, 0, 0, 1] = (float("nan"), .1, 0.0)
    elif mutation == "actual_sample_nonfinite":
        assembler.raw_route_actual_sample_m[0, 0, 0, 1] = (.5, float("nan"), 0.0)
    elif mutation is not None:
        raise AssertionError(f"unknown mutation: {mutation}")

    return scan


def _audit(case: dict[str, object], scan: dict[str, object]) -> None:
    assembler = case["assembler"]
    assembler.certify_active_raw_routes_device(
        expected_generation=19,
        support_available=1,
        support_anisotropic=0,
        strict_support_radius_xyz_m=(2.0, 2.0, 2.0),
        marker_normal_m=case["normals"],
        marker_role=case["roles"],
        **scan,
    )


@pytest.mark.parametrize(
    ("reverse_storage", "mirrored"),
    ((False, False), (True, False), (True, True)),
)
def test_registered_segment_audit_accepts_five_authors_storage_order_and_mirror(
    audit_case: dict[str, object],
    reverse_storage: bool,
    mirrored: bool,
) -> None:
    """Pass C keeps all raw authors and is invariant to storage order/mirroring."""

    _audit(
        audit_case,
        _prepare(
            audit_case,
            reverse_storage=reverse_storage,
            mirrored=mirrored,
        ),
    )
    assembler = audit_case["assembler"]
    assert assembler.audit_valid[2, 0, 0, 1] == 1
    assert assembler.audit_raw_count[2, 0, 0, 1] == 5
    assert assembler.audit_failure[2, 0, 0, 1] == 0
    assert assembler.audit_rejection_count[None] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "stale_generation",
        "target_out_of_bounds",
        "anchor_outside_source_support",
        "unregistered_primitive",
        "corrupted_boundary_target",
        "owner_target_mismatch",
        "owner_point_nonfinite",
        "raw_normal_not_ray_aligned",
        "nominal_sample_nonfinite",
        "actual_sample_nonfinite",
    ),
)
def test_registered_segment_audit_rejects_corrupted_raw_provenance(
    audit_case: dict[str, object],
    mutation: str,
) -> None:
    """Each raw record is revalidated after B, not trusted from Pass A."""

    _audit(audit_case, _prepare(audit_case, mutation=mutation))
    assembler = audit_case["assembler"]
    assert assembler.audit_rejection_count[None] > 0
    assert assembler.audit_valid[2, 0, 0, 1] == 0


@pytest.mark.parametrize("concave", (False, True))
def test_registered_segment_audit_shared_vertex_corner_cone(
    audit_case: dict[str, object], concave: bool
) -> None:
    """Only the convex shared-vertex normal cone can certify a nearest tie."""

    assembler = audit_case["assembler"]
    positions = audit_case["positions"]
    normals = audit_case["normals"]
    velocities = audit_case["velocities"]
    regions = audit_case["regions"]
    roles = audit_case["roles"]
    segments = audit_case["segments"]
    assembler.clear_device_transaction()
    positions.from_numpy(np.asarray(((0, 0, 0), (-1, 0, 0), (0, -1, 0), (4, 0, 0)), np.float32))
    normals.from_numpy(np.asarray(((-1 if concave else 1, 1, 0), (0, 1, 0), (-1 if concave else 1, 0, 0), (0, 1, 0)), np.float32))
    velocities.fill(0.0); regions.fill(0); roles.fill(0)
    segments.from_numpy(np.asarray(((0, 1, -1), (0, 2, -1)), np.int32))
    assembler.install_registered_topology(((0, 1), (0, 2)), vertex_count=3)
    assembler.install_explicit_endpoint_aliases(())
    key = (0, 0, 0, 1)
    assembler.raw_route_valid[key] = 1; assembler.raw_route_kind[key] = 0
    assembler.raw_route_target[key] = (2, 0, 0); assembler.raw_route_region[key] = 0
    assembler.raw_route_primitive[key] = (0, 1, -1); assembler.raw_route_weights[key] = (1, 0, 0)
    assembler.raw_route_anchor_m[key] = (0, 0, 0); assembler.raw_route_normal[key] = (0, 1, 0)
    assembler.raw_route_nominal_sample_m[key] = (0, .1, 0); assembler.raw_route_actual_sample_m[key] = (0, .1, 0)
    assembler.raw_route_sample_valid[key] = 1; assembler.raw_route_generation[key] = 23
    assembler.face_raw_count[2, 0, 0] = (0, 1, 0)
    scan = dict(inactive_axis=2, cell_face_x_m=audit_case["face_x"], cell_face_y_m=audit_case["face_y"], cell_face_z_m=audit_case["face_z"], cell_center_x_m=audit_case["center_x"], cell_center_y_m=audit_case["center_y"], cell_center_z_m=audit_case["center_z"], projection_segment_indices=segments, projection_segment_count=2, marker_position_m=positions, marker_velocity_mps=velocities, marker_region_id=regions, projection_vertex_count=3)
    assembler.scan_registered_active_faces_device(**scan)
    assembler.certify_active_raw_routes_device(expected_generation=23, support_available=1, support_anisotropic=0, strict_support_radius_xyz_m=(2, 2, 2), marker_normal_m=normals, marker_role=roles, **scan)
    assert assembler.audit_valid[2, 0, 0, 1] == (0 if concave else 1)


@pytest.mark.parametrize("mutation", (None, "role", "position", "velocity"))
def test_registered_segment_audit_explicit_cap_alias_cross_region(
    audit_case: dict[str, object], mutation: str | None
) -> None:
    """An exact declared cap alias alone may cross regions; bad alias data fails."""

    assembler = audit_case["assembler"]
    positions = audit_case["positions"]; normals = audit_case["normals"]
    velocities = audit_case["velocities"]; regions = audit_case["regions"]; roles = audit_case["roles"]; segments = audit_case["segments"]
    assembler.clear_device_transaction()
    positions.from_numpy(np.asarray(((0,0,0),(-1,0,0),(0,0,0),(1,-1,0)), np.float32))
    normals.from_numpy(np.asarray(((0,1,0),(0,1,0),(1,1,0),(1,1,0)), np.float32))
    velocities.fill(0.0); regions.from_numpy(np.asarray((0,0,3,3), np.int32)); roles.from_numpy(np.asarray((0,0,2,0), np.int32))
    if mutation == "role": roles[2] = 99
    if mutation == "position": positions[2] = (.0001, 0, 0)
    if mutation == "velocity": velocities[2] = (.001, 0, 0)
    segments.from_numpy(np.asarray(((0,1,-1),(2,3,-1)), np.int32))
    assembler.install_registered_topology(((0,1),(2,3)), vertex_count=4)
    assembler.install_explicit_endpoint_aliases(((0,2),), expected_role_pairs=((0,2),))
    key = (0,0,0,1)
    assembler.raw_route_valid[key] = 1; assembler.raw_route_kind[key] = 0; assembler.raw_route_target[key] = (2,0,0); assembler.raw_route_region[key] = 0
    assembler.raw_route_primitive[key] = (0,1,-1); assembler.raw_route_weights[key] = (.5,.5,0); assembler.raw_route_anchor_m[key] = (-.5,0,0); assembler.raw_route_normal[key] = (0,1,0)
    assembler.raw_route_nominal_sample_m[key] = (-.5,.1,0); assembler.raw_route_actual_sample_m[key] = (-.5,.1,0); assembler.raw_route_sample_valid[key] = 1; assembler.raw_route_generation[key] = 29
    assembler.face_raw_count[2,0,0] = (0,1,0)
    scan = dict(inactive_axis=2, cell_face_x_m=audit_case["face_x"], cell_face_y_m=audit_case["face_y"], cell_face_z_m=audit_case["face_z"], cell_center_x_m=audit_case["center_x"], cell_center_y_m=audit_case["center_y"], cell_center_z_m=audit_case["center_z"], projection_segment_indices=segments, projection_segment_count=2, marker_position_m=positions, marker_velocity_mps=velocities, marker_region_id=regions, projection_vertex_count=4)
    assembler.scan_registered_active_faces_device(**scan)
    assembler.certify_active_raw_routes_device(expected_generation=29, support_available=1, support_anisotropic=0, strict_support_radius_xyz_m=(2,2,2), marker_normal_m=normals, marker_role=roles, **scan)
    assert assembler.audit_valid[2,0,0,1] == (1 if mutation is None else 0)


def _one_source_with_radius(case, scan, *, anisotropic, radius=.3):
    assembler = case["assembler"]
    assembler.raw_route_valid.fill(0)
    assembler.raw_route_valid[2, 0, 0, 1] = 1
    assembler.face_raw_count.fill(0)
    assembler.face_raw_count[2, 0, 0] = (0, 1, 0)
    assembler.scan_registered_active_faces_device(**scan)
    assembler.certify_active_raw_routes_device(
        expected_generation=19, support_available=1, support_anisotropic=int(anisotropic),
        strict_support_radius_xyz_m=(radius, radius, radius),
        marker_normal_m=case["normals"], marker_role=case["roles"], **scan,
    )


@pytest.mark.parametrize("anisotropic", (False, True))
@pytest.mark.parametrize("mirrored", (False, True))
def test_registered_segment_audit_accepts_local_part_of_long_segment(audit_case, anisotropic, mirrored):
    """A local intercept is valid even when both finite endpoints are outside."""
    scan = _prepare(audit_case, mirrored=mirrored, reverse_storage=mirrored)
    _one_source_with_radius(audit_case, scan, anisotropic=anisotropic, radius=.2)
    assembler = audit_case["assembler"]
    assert assembler.audit_valid[2, 0, 0, 1] == 1
    assert assembler.audit_rejection_count[None] == 0


@pytest.mark.parametrize("anisotropic", (False, True))
def test_registered_segment_audit_local_convex_corner_with_long_incident_edges(audit_case, anisotropic):
    """The local corner cone must not depend on lengths of its remote ends."""
    scan = _prepare(audit_case)
    assembler = audit_case["assembler"]
    audit_case["positions"].from_numpy(np.asarray(((.4, 0, 0), (-.6, 0, 0), (.4, -1, 0), (4, 0, 0)), np.float32))
    audit_case["normals"].from_numpy(np.asarray(((1, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)), np.float32))
    audit_case["segments"].from_numpy(np.asarray(((0, 1, -1), (0, 2, -1)), np.int32))
    assembler.install_registered_topology(((0, 1), (0, 2)), vertex_count=3)
    scan["projection_vertex_count"] = 3
    key = (2, 0, 0, 1)
    assembler.raw_route_primitive[key] = (0, 1, -1)
    assembler.raw_route_weights[key] = (1, 0, 0)
    assembler.raw_route_anchor_m[key] = (.4, 0, 0)
    assembler.raw_route_nominal_sample_m[key] = (.4, .1, 0)
    assembler.raw_route_actual_sample_m[key] = (.4, .1, 0)
    _one_source_with_radius(audit_case, scan, anisotropic=anisotropic)
    assert assembler.owner_vertex[2, 0, 0, 1] == 0
    assert assembler.audit_valid[2, 0, 0, 1] == 1


@pytest.mark.parametrize("anisotropic", (False, True))
def test_registered_segment_audit_rejects_outside_connector_between_clipped_edges(audit_case, anisotropic):
    """Two intersecting local fragments cannot connect through a remote vertex."""
    scan = _prepare(audit_case)
    assembler = audit_case["assembler"]
    audit_case["positions"].from_numpy(np.asarray(((0, -.1, 0), (2, 0, 0), (0, .1, 0), (4, 0, 0)), np.float32))
    first_normal = np.asarray((-.1, 2, 0), np.float32)
    first_normal /= np.linalg.norm(first_normal)
    audit_case["normals"].from_numpy(np.asarray((first_normal, (0, 1, 0), (.1, 2, 0), (0, 1, 0)), np.float32))
    audit_case["segments"].from_numpy(np.asarray(((0, 1, -1), (1, 2, -1)), np.int32))
    assembler.install_registered_topology(((0, 1), (1, 2)), vertex_count=3)
    scan["projection_vertex_count"] = 3
    key = (2, 0, 0, 1)
    anchor = np.asarray((.5, -.075, 0), np.float32)
    sample = anchor + np.float32(.1) * first_normal
    assembler.raw_route_primitive[key] = (0, 1, -1)
    assembler.raw_route_weights[key] = (.75, .25, 0)
    assembler.raw_route_anchor_m[key] = anchor
    assembler.raw_route_normal[key] = first_normal
    assembler.raw_route_nominal_sample_m[key] = sample
    assembler.raw_route_actual_sample_m[key] = sample
    _one_source_with_radius(audit_case, scan, anisotropic=anisotropic)
    assert assembler.owner_segment_index[2, 0, 0, 1] == 1
    assert assembler.audit_valid[2, 0, 0, 1] == 0


@pytest.mark.parametrize("anisotropic", (False, True))
def test_registered_segment_audit_rejects_support_boundary_contact(audit_case, anisotropic):
    """A finite segment merely touching the open support does not become local."""
    scan = _prepare(audit_case)
    _one_source_with_radius(audit_case, scan, anisotropic=anisotropic, radius=.1)
    assert audit_case["assembler"].audit_valid[2, 0, 0, 1] == 0
