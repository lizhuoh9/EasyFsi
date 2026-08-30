"""CPU-kernel RED/negative contracts for final-owner-aware MAC selection.

The fixture evaluates the real nearest-owner and normal-side certificate.
It does not fabricate raw authors or publish a canonical boundary ledger.
"""

import itertools

import numpy as np
import pytest
import taichi as ti

from simulation_core.coupling.hibm_mpm import component_face_segment_assembly as assembly
from simulation_core.coupling.hibm_mpm.core import HibmMpmIbBoundaryConditions


@ti.kernel
def _select(boundary: ti.template(), source: ti.types.vector(3, ti.i32), axis: ti.i32,
            anchor: ti.types.vector(3, ti.f32), sample: ti.types.vector(3, ti.f32),
            obstacle: ti.template(), fx: ti.template(), fy: ti.template(), fz: ti.template(),
            cx: ti.template(), cy: ti.template(), cz: ti.template(), permission: ti.template(),
            result: ti.template()):
    valid, storage, alpha, failure, pair_valid, pair_storage, pair_alpha = (
        boundary._select_canonical_component_face_storage_device(
            source, axis, anchor, sample, obstacle, 1, fx, fy, fz, cx, cy, cz,
            4, 4, 4, candidate_permission=permission,
        )
    )
    result[0] = valid
    result[1], result[2], result[3] = storage.x, storage.y, storage.z
    result[4] = failure


@pytest.fixture(scope="module")
def candidate_case():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1,
            opt_level=1, advanced_optimization=False, offline_cache=False)
    initializer = assembly.init_taichi
    assembly.init_taichi = lambda runtime=None: None
    try:
        assembler = assembly.RegisteredComponentFaceSegmentAssembler(grid_nodes=(4, 4, 4), marker_capacity=6)
    finally:
        assembly.init_taichi = initializer
    coordinates = [ti.field(ti.f32, shape=5) for _ in range(3)] + [ti.field(ti.f32, shape=4) for _ in range(3)]
    for field in coordinates[:3]:
        field.from_numpy(np.linspace(0, 1, 5, dtype=np.float32))
    for field in coordinates[3:]:
        field.from_numpy(np.arange(4, dtype=np.float32) / 4 + 0.125)
    fields = {name: ti.Vector.field(3, dtype=dtype, shape=6) for name, dtype in (
        ("positions", ti.f32), ("velocities", ti.f32), ("normals", ti.f32), ("segments", ti.i32),
    )}
    fields.update(regions=ti.field(ti.i32, shape=6), roles=ti.field(ti.i32, shape=6),
                  obstacle=ti.field(ti.i32, shape=(4, 4, 4)), result=ti.field(ti.i32, shape=5))
    yield {**fields, "assembler": assembler, "coordinates": coordinates,
           "boundary": object.__new__(HibmMpmIbBoundaryConditions)}
    ti.reset()


def _configure(case, inactive_axis, reflect_normal, reflect_component, reverse_storage,
               *, shape="curved", shift=0.0):
    points = np.asarray([[0.375, 0.65625, 0.125], [0.375, 0.65625, 0.5625],
                         [0.375, 0.59375, 0.59375], [0.375, 0.59375, 0.9375]], dtype=np.float64)
    if shape == "curved":
        # A nonzero normal component makes this an eligible production raw
        # lane too; the example does not rely on a tangential zero-normal lane.
        points[3, 1] -= (points[3, 2] - points[2, 2]) / 16
    elif shape == "plane":
        points[:, 1] = 0.59375
    elif shape == "slanted":
        points[:, 1] = 0.59375 + (points[:, 2] - 0.625) / 8
    elif shape == "inside":
        points[:, 1] = 0.65625
    points[:, 1] += shift
    normals = np.zeros_like(points)
    for edge in range(3):
        chord = points[edge + 1] - points[edge]
        outward = np.asarray([0, chord[2], -chord[1]])
        outward /= np.linalg.norm(outward)
        normals[edge:edge + 2] += outward
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    source_point = np.asarray([0.375, 0.625, 0.625])
    # The source really projects to the upper segment. Both lower/upper MAC
    # faces have positive source-ray progress, but the lower final owner is
    # the bulging segment and places that face inside the surface.
    chord = points[3] - points[2]
    weight = np.dot(source_point - points[2], chord) / np.dot(chord, chord)
    anchor = points[2] + weight * chord
    sample = anchor + 0.25 * normals[3]
    source = np.asarray([1, 2, 2], dtype=np.int32)
    if reflect_normal:
        points[:, 1] = 1 - points[:, 1]
        normals[:, 1] *= -1
        anchor[1], sample[1] = 1 - anchor[1], 1 - sample[1]
        source[1] = 1
    if reflect_component:
        points[:, 2] = 1 - points[:, 2]
        normals[:, 2] *= -1
        anchor[2], sample[2] = 1 - anchor[2], 1 - sample[2]
        source[2] = 1
    permutation = np.roll(np.arange(3), inactive_axis)
    source = source[permutation]
    component = (inactive_axis + 2) % 3
    for name, values in (("positions", points), ("normals", normals)):
        padded = np.zeros((6, 3), dtype=np.float32)
        padded[:4] = values[:, permutation]
        case[name].from_numpy(padded)
    case["velocities"].fill(0.125)
    edges = [(0, 1), (1, 2), (2, 3)]
    if reverse_storage:
        edges = [(second, first) for first, second in reversed(edges)]
    segments = np.full((6, 3), -1, dtype=np.int32)
    segments[:3, :2] = edges
    case["segments"].from_numpy(segments)
    case["regions"].fill(101)
    case["roles"].fill(101)
    case["obstacle"].fill(0)
    assembler = case["assembler"]
    assembler.clear_device_transaction()
    assembler.install_registered_topology(edges, vertex_count=4)
    # The public registry canonicalizes order; this low-level permutation
    # fixture installs matching adjacency for the deliberately permuted table.
    for vertex in range(4):
        incident = [index for index, edge in enumerate(edges) if vertex in edge]
        assembler.registered_vertex_degree[vertex] = len(incident)
        assembler.registered_vertex_adjacency[vertex] = tuple((incident + [-1, -1])[:2])
    assembler.install_explicit_endpoint_aliases(())
    for offset in range(2):
        face = source.copy()
        face[component] += offset
        assembler.face_raw_count[tuple(face)][component] = 1
    coordinates = case["coordinates"]
    scan = dict(zip(("cell_face_x_m", "cell_face_y_m", "cell_face_z_m",
                     "cell_center_x_m", "cell_center_y_m", "cell_center_z_m"), coordinates))
    scan.update(inactive_axis=inactive_axis, projection_segment_indices=case["segments"],
                projection_segment_count=3, marker_position_m=case["positions"],
                marker_velocity_mps=case["velocities"], marker_region_id=case["regions"], projection_vertex_count=4)
    assembler.scan_registered_active_faces_device(**scan)
    assembler._prepare_registered_audit_geometry_kernel(
        inactive_axis, case["segments"], 3, case["positions"], case["velocities"],
        case["normals"], case["regions"], case["roles"], 4,
    )
    assembler._certify_registered_owners_kernel(
        inactive_axis, 1, (0.5, 0.5, 0.5), *coordinates, case["segments"], 3,
        case["positions"], case["velocities"], case["regions"],
    )
    return source, component, anchor[permutation], sample[permutation]


@pytest.mark.parametrize("inactive_axis,reflect_normal,reflect_component,reverse_storage", tuple(
    itertools.product(range(3), (False, True), (False, True), (False, True))
))
@pytest.mark.parametrize("shift", (-1 / 128, 0.0, 1 / 128))
def test_curved_source_outside_but_final_owner_inside_selects_only_legal_mac_face(
    candidate_case, inactive_axis, reflect_normal, reflect_component, reverse_storage, shift,
):
    case = candidate_case
    source, component, anchor, sample = _configure(
        case, inactive_axis, reflect_normal, reflect_component, reverse_storage, shift=shift,
    )
    assembler = case["assembler"]
    expected = source.copy()
    expected[component] += 0 if reflect_component else 1
    bad = source.copy()
    bad[component] += 1 if reflect_component else 0
    expected_key = (*map(int, expected), component)
    bad_key = (*map(int, bad), component)
    assert assembler._audited_owner_valid[expected_key] == 1
    assert assembler._audited_owner_failure[bad_key] == 3
    assert tuple(assembler.owner_segment[expected_key]) == (2, 3)
    assert tuple(assembler.owner_segment[bad_key]) == (0, 1)
    _select(case["boundary"], source, component, anchor, sample, case["obstacle"],
            *case["coordinates"], assembler._audited_owner_valid, case["result"])
    result = case["result"].to_numpy()
    assert result[0] == 1
    np.testing.assert_array_equal(result[1:4], expected)


@pytest.mark.parametrize("shape", ("plane", "slanted", "inside"))
@pytest.mark.parametrize("inactive_axis,reflect_normal", tuple(itertools.product(range(3), (False, True))))
def test_planar_symmetry_and_genuine_inside_negative(candidate_case, shape, inactive_axis, reflect_normal):
    case = candidate_case
    source, component, anchor, sample = _configure(case, inactive_axis, reflect_normal, False, False, shape=shape)
    obstacle_before = case["obstacle"].to_numpy().copy()
    assembler = case["assembler"]
    _select(case["boundary"], source, component, anchor, sample, case["obstacle"],
            *case["coordinates"], assembler._audited_owner_valid, case["result"])
    result = case["result"].to_numpy()
    assert bool(result[0]) is (shape != "inside")
    if shape == "inside":
        assert result[4] == 3  # No final-owner-admissible candidate, not a ray error.
    np.testing.assert_array_equal(case["obstacle"].to_numpy(), obstacle_before)


@pytest.mark.parametrize("inactive_axis", range(3))
def test_candidate_prepass_reuses_final_audit_without_creating_authors(candidate_case, inactive_axis):
    case = candidate_case
    source, component, _, _ = _configure(case, inactive_axis, False, False, False)
    assembler = case["assembler"]
    keys = []
    expected = []
    for offset in range(2):
        face = source.copy()
        face[component] += offset
        key = (*map(int, face), component)
        keys.append(key)
        expected.append(int(assembler._audited_owner_valid[key]))
    assembler.clear_device_transaction()
    for key in keys:
        assembler.candidate_face_requested[key[:3]][component] = 1
    coordinates = dict(zip(("cell_face_x_m", "cell_face_y_m", "cell_face_z_m",
                            "cell_center_x_m", "cell_center_y_m", "cell_center_z_m"), case["coordinates"]))
    assembler.prepare_candidate_owner_geometry(
        **coordinates, inactive_axis=inactive_axis, support_available=1, support_anisotropic=1,
        strict_support_radius_xyz_m=(0.5, 0.5, 0.5),
        projection_segment_indices=case["segments"], projection_segment_count=3,
        marker_position_m=case["positions"], marker_velocity_mps=case["velocities"],
        marker_normal_m=case["normals"], marker_region_id=case["regions"], marker_role=case["roles"],
        projection_vertex_count=4,
    )
    assert [int(assembler.candidate_owner_permission[key]) for key in keys] == expected
    assert not np.any(assembler.face_raw_count.to_numpy())
    assert not np.any(assembler.raw_route_valid.to_numpy())
