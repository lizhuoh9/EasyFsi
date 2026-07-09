import math

import numpy as np
import pytest
import taichi as ti

# CRITICAL: a CUDA production run may be using the GPU concurrently in this
# environment. Never ti.init(arch=ti.cuda) here -- CPU backend only. Module
# scope, once: the module under test never calls ti.init itself (see its
# docstring), so this is the single place Taichi gets initialized for the
# whole test file.
ti.init(arch=ti.cpu)

from simulation_core.coupling import marker_seeding
from simulation_core.coupling.marker_seeding import (
    resample_polyline_markers_by_arc_length,
    seed_markers_from_tri_surface,
    tri_surface_needs_reseed,
    update_markers_from_vertices,
)


# ---------------------------------------------------------------------
# Shared helpers (pure numpy; independent of the module's own formulas
# wherever the test is meant to check those formulas against ground truth)
# ---------------------------------------------------------------------
def _right_triangle():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    return vertices, triangles


def _triangle_soup(rng, triangle_count, coordinate_scale=1.0, spread=0.4):
    """Disjoint triangles (no shared vertices) so per-triangle deformation
    in a test cannot couple into any other triangle's edges."""
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    triangles = np.empty((triangle_count, 3), dtype=np.int64)
    for t in range(triangle_count):
        center = rng.uniform(-coordinate_scale, coordinate_scale, size=3)
        offsets = rng.uniform(-spread, spread, size=(3, 3))
        vertices[3 * t : 3 * t + 3] = center[None, :] + offsets
        triangles[t] = [3 * t, 3 * t + 1, 3 * t + 2]
    return vertices, triangles


def _triangle_areas(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _triangle_longest_edges(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    return np.maximum(np.maximum(e0, e1), e2)


def _expected_level(longest_edge, min_cell, spacing_safety):
    threshold = spacing_safety * min_cell
    if longest_edge <= 0.0:
        return 0
    ratio = longest_edge / threshold
    if ratio <= 1.0:
        return 0
    return int(math.ceil(math.log2(ratio)))


# ---------------------------------------------------------------------
# Taichi gotcha regression: this file's own module (marker_seeding.py)
# must never gain `from __future__ import annotations`, which breaks
# @ti.kernel argument extraction at import time.
# ---------------------------------------------------------------------
def test_module_has_no_future_annotations_import():
    source = open(marker_seeding.__file__, encoding="utf-8").read()
    assert "from __future__ import annotations" not in source


# ---------------------------------------------------------------------
# Area exactness
# ---------------------------------------------------------------------
def test_area_exactness_random_irregular_mesh():
    rng = np.random.default_rng(42)
    triangle_count = 7
    vertices, triangles = _triangle_soup(rng, triangle_count, coordinate_scale=2.0, spread=0.6)
    rest_areas_ground_truth = _triangle_areas(vertices, triangles)
    assert np.all(rest_areas_ground_truth > 1.0e-6), "test mesh must be non-degenerate"

    cell = (0.35, 0.35, 0.35)
    seed = seed_markers_from_tri_surface(vertices, triangles, cell, spacing_safety=0.75)
    data = seed.to_numpy()

    per_triangle_marker_area_sum = np.bincount(
        data["parent_triangle"], weights=data["areas_m2"], minlength=triangle_count
    )
    np.testing.assert_allclose(
        per_triangle_marker_area_sum, data["rest_area_m2"], rtol=1e-12, atol=1e-20
    )
    np.testing.assert_allclose(
        per_triangle_marker_area_sum.sum(), data["rest_area_m2"].sum(), rtol=1e-12
    )


def test_degenerate_triangle_single_marker_zero_area_zero_normal():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0e-20, 0.0, 0.0], [0.0, 1.0e-20, 0.0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    cell = (0.75, 0.75, 0.75)

    seed = seed_markers_from_tri_surface(vertices, triangles, cell)
    data = seed.to_numpy()

    assert seed.marker_count == 1
    assert int(data["subdivision_level"][0]) == 0
    assert data["areas_m2"][0] == 0.0
    assert data["rest_area_m2"][0] == 0.0
    np.testing.assert_array_equal(data["normals"][0], np.array([0.0, 0.0, 0.0], dtype=np.float32))


# ---------------------------------------------------------------------
# Spacing bound
# ---------------------------------------------------------------------
def test_spacing_bound_barycentric_reconstruction_pure_math():
    """Independent (module-free) proof that a level-L sub-triangle's edges
    are the parent's edges scaled by exactly 1/2**L, reconstructed via the
    same barycentric lattice-point formula the kernel uses."""
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([3.0, 0.0, 0.0])
    v2 = np.array([0.0, 2.0, 0.0])
    parent_longest_edge = max(
        np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)
    )

    def lattice_point(a, b, s):
        return ((s - a - b) / s) * v0 + (a / s) * v1 + (b / s) * v2

    for level in range(0, 5):
        s = 2**level
        p0 = lattice_point(0, 0, s)
        p1 = lattice_point(1, 0, s)
        p2 = lattice_point(0, 1, s)
        sub_longest_edge = max(
            np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)
        )
        assert sub_longest_edge == pytest.approx(parent_longest_edge / s, rel=1e-12)


@pytest.mark.parametrize(
    "cell",
    [
        (0.75, 0.75, 0.75),
        (0.4, 0.9, 0.6),
        (1.2, 0.3, 0.5),
        (0.05, 5.0, 5.0),
    ],
)
def test_spacing_bound_holds_for_seeded_fields(cell):
    vertices, triangles = _right_triangle()
    spacing_safety = 0.75
    seed = seed_markers_from_tri_surface(vertices, triangles, cell, spacing_safety=spacing_safety)
    level = int(seed.subdivision_level.to_numpy()[0])

    longest_edge = float(_triangle_longest_edges(vertices, triangles)[0])
    min_cell = float(min(cell))
    sub_edge = longest_edge / (2**level)
    threshold = spacing_safety * min_cell
    assert sub_edge <= threshold * (1.0 + 1.0e-9)
    assert level == _expected_level(longest_edge, min_cell, spacing_safety)


def test_marker_count_scales_4x_per_halved_cell():
    vertices, triangles = _right_triangle()
    cell_coarse = (0.75, 0.75, 0.75)
    cell_fine = (0.375, 0.375, 0.375)

    seed_coarse = seed_markers_from_tri_surface(vertices, triangles, cell_coarse)
    seed_fine = seed_markers_from_tri_surface(vertices, triangles, cell_fine)

    level_coarse = int(seed_coarse.subdivision_level.to_numpy()[0])
    level_fine = int(seed_fine.subdivision_level.to_numpy()[0])
    assert level_fine == level_coarse + 1
    assert seed_fine.marker_count == 4 * seed_coarse.marker_count


def test_max_level_exceeded_raises_actionable_error():
    vertices, triangles = _right_triangle()
    tiny_cell = (1.0e-6, 1.0e-6, 1.0e-6)
    with pytest.raises(ValueError, match="max_level"):
        seed_markers_from_tri_surface(vertices, triangles, tiny_cell, max_level=3)


# ---------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------
def test_normals_unit_and_vertex_order_flip():
    vertices, triangles = _right_triangle()
    cell = (0.75, 0.75, 0.75)

    seed = seed_markers_from_tri_surface(vertices, triangles, cell)
    normals = seed.normals.to_numpy()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)

    flipped_triangles = np.array([[0, 2, 1]], dtype=np.int64)
    seed_flipped = seed_markers_from_tri_surface(vertices, flipped_triangles, cell)
    normals_flipped = seed_flipped.normals.to_numpy()

    np.testing.assert_allclose(normals_flipped, -normals, atol=1e-6)


# ---------------------------------------------------------------------
# Update exactness under an affine map
# ---------------------------------------------------------------------
def test_update_exactness_under_affine_map():
    rng = np.random.default_rng(7)
    triangle_count = 4
    vertices, triangles = _triangle_soup(rng, triangle_count, coordinate_scale=1.0, spread=0.4)
    cell = (0.3, 0.3, 0.3)

    seed = seed_markers_from_tri_surface(vertices, triangles, cell)
    rest_data = seed.to_numpy()
    rest_positions_f64 = rest_data["positions_m"].astype(np.float64)

    # A generic (non-rigid: scale + shear + rotation-ish) affine map.
    linear_map = np.array(
        [
            [1.3, 0.15, -0.05],
            [-0.1, 0.85, 0.2],
            [0.05, -0.1, 1.15],
        ]
    )
    translation = np.array([0.4, -0.25, 0.1])

    deformed_vertices = (linear_map @ vertices.T).T + translation
    seed.vertices_m.from_numpy(deformed_vertices.astype(np.float32))
    update_markers_from_vertices(seed)
    updated = seed.to_numpy()

    # Position: barycentric combination commutes with any affine map.
    expected_positions = (linear_map @ rest_positions_f64.T).T + translation
    np.testing.assert_allclose(
        updated["positions_m"].astype(np.float64), expected_positions, atol=1e-6
    )

    # Area: ratio of deformed/rest triangle area times the fixed rest marker area.
    deformed_areas = _triangle_areas(deformed_vertices, triangles)
    rest_areas = rest_data["rest_area_m2"]
    ratio_per_triangle = deformed_areas / rest_areas
    expected_marker_area = (
        ratio_per_triangle[updated["parent_triangle"]] * rest_data["rest_marker_area_m2"]
    )
    np.testing.assert_allclose(
        updated["areas_m2"], expected_marker_area, rtol=1e-5, atol=1e-15
    )

    # Normal: unit cross product of the deformed parent triangle.
    dv0 = deformed_vertices[triangles[:, 0]]
    dv1 = deformed_vertices[triangles[:, 1]]
    dv2 = deformed_vertices[triangles[:, 2]]
    tri_normal = np.cross(dv1 - dv0, dv2 - dv0)
    tri_normal = tri_normal / np.linalg.norm(tri_normal, axis=1, keepdims=True)
    expected_normals = tri_normal[updated["parent_triangle"]]
    np.testing.assert_allclose(
        updated["normals"].astype(np.float64), expected_normals, atol=1e-6
    )


# ---------------------------------------------------------------------
# Reseed check
# ---------------------------------------------------------------------
def test_rigid_motion_does_not_trigger_reseed():
    vertices, triangles = _right_triangle()
    cell = (0.75, 0.75, 0.75)
    seed = seed_markers_from_tri_surface(vertices, triangles, cell)

    theta = math.radians(30.0)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([5.0, -3.0, 2.0])
    rotated_vertices = (rotation @ vertices.T).T + translation
    seed.vertices_m.from_numpy(rotated_vertices.astype(np.float32))

    assert tri_surface_needs_reseed(seed, cell) is False


def test_large_stretch_triggers_reseed():
    vertices, triangles = _right_triangle()
    cell = (0.75, 0.75, 0.75)
    seed = seed_markers_from_tri_surface(vertices, triangles, cell)

    stretched_vertices = vertices * 3.0
    seed.vertices_m.from_numpy(stretched_vertices.astype(np.float32))

    assert tri_surface_needs_reseed(seed, cell) is True


# ---------------------------------------------------------------------
# Polyline resample (host-only utility)
# ---------------------------------------------------------------------
def test_resample_polyline_open_line():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    areas = np.array([0.1, 0.2, 0.15])

    out_pos, out_norm, out_area = resample_polyline_markers_by_arc_length(
        positions, normals, areas, target_spacing_m=0.4, closed=False
    )

    assert out_pos.shape[0] >= 2
    np.testing.assert_allclose(out_area.sum(), areas.sum(), rtol=1e-9)
    spacings = np.linalg.norm(np.diff(out_pos, axis=0), axis=1)
    assert np.all(spacings <= 0.4 + 1e-9)
    np.testing.assert_allclose(np.linalg.norm(out_norm, axis=1), 1.0, atol=1e-9)
    # endpoints preserved exactly for an open polyline
    np.testing.assert_allclose(out_pos[0], positions[0], atol=1e-12)
    np.testing.assert_allclose(out_pos[-1], positions[-1], atol=1e-12)


def test_resample_polyline_quarter_circle():
    theta = np.linspace(0.0, math.pi / 2.0, 25)
    radius = 2.0
    positions = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1
    )
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    areas = np.full(positions.shape[0], 0.05)

    out_pos, out_norm, out_area = resample_polyline_markers_by_arc_length(
        positions, normals, areas, target_spacing_m=0.3, closed=False
    )

    np.testing.assert_allclose(out_area.sum(), areas.sum(), rtol=1e-9)
    chord_spacings = np.linalg.norm(np.diff(out_pos, axis=0), axis=1)
    assert np.all(chord_spacings <= 0.3 + 1e-6)
    np.testing.assert_allclose(np.linalg.norm(out_norm, axis=1), 1.0, atol=1e-9)


def test_resample_polyline_closed_loop():
    theta = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    radius = 1.5
    positions = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1
    )
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    areas = np.full(positions.shape[0], 0.02)

    out_pos, out_norm, out_area = resample_polyline_markers_by_arc_length(
        positions, normals, areas, target_spacing_m=0.5, closed=True
    )

    np.testing.assert_allclose(out_area.sum(), areas.sum(), rtol=1e-9)
    looped = np.vstack([out_pos, out_pos[:1]])
    spacings = np.linalg.norm(np.diff(looped, axis=0), axis=1)
    assert np.all(spacings <= 0.5 + 1e-6)
    np.testing.assert_allclose(np.linalg.norm(out_norm, axis=1), 1.0, atol=1e-9)


def test_resample_polyline_bad_inputs_raise():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    areas = np.array([0.1, 0.1])

    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions[:1], normals[:1], areas[:1], 0.1)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions, normals[:1], areas, 0.1)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions, normals, areas[:1], 0.1)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions, normals, -areas, 0.1)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions, normals, areas, 0.0)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(positions, normals, areas, -1.0)
    with pytest.raises(ValueError):
        resample_polyline_markers_by_arc_length(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), normals, areas, 0.1
        )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
def test_determinism_bit_identical_across_runs():
    rng = np.random.default_rng(123)
    vertices, triangles = _triangle_soup(rng, 5, coordinate_scale=1.5, spread=0.5)
    cell = (0.4, 0.4, 0.4)

    seed_a = seed_markers_from_tri_surface(vertices, triangles, cell)
    seed_b = seed_markers_from_tri_surface(vertices, triangles, cell)

    assert seed_a.marker_count == seed_b.marker_count
    data_a = seed_a.to_numpy()
    data_b = seed_b.to_numpy()
    for key in data_a:
        assert np.array_equal(data_a[key], data_b[key]), f"mismatch in field {key!r}"

    # Determinism also holds across a subsequent update call.
    rng2 = np.random.default_rng(999)
    linear_map = np.eye(3) + 0.1 * rng2.standard_normal((3, 3))
    translation = rng2.standard_normal(3)
    deformed = (linear_map @ vertices.T).T + translation
    seed_a.vertices_m.from_numpy(deformed.astype(np.float32))
    seed_b.vertices_m.from_numpy(deformed.astype(np.float32))
    update_markers_from_vertices(seed_a)
    update_markers_from_vertices(seed_b)

    updated_a = seed_a.to_numpy()
    updated_b = seed_b.to_numpy()
    for key in ("positions_m", "normals", "areas_m2"):
        assert np.array_equal(updated_a[key], updated_b[key]), f"mismatch in field {key!r}"
