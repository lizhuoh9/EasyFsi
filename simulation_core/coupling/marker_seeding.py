"""Grid-adaptive, deformation-following surface-marker seeding for HIBM-MPM FSI.

A marker is ``(position, outward unit normal, area weight)`` sampled on a
triangle surface mesh via closed-form triangular-lattice subdivision. Each
triangle is refined independently until every sub-triangle edge fits inside
the local fluid grid spacing (times a safety factor). Every marker keeps a
fixed barycentric binding -- parent triangle id + 3 barycentric weights --
captured once at seed time; every simulation step,
``update_markers_from_vertices`` recomputes position/normal/area straight
from the CURRENT deformed parent-triangle vertices in a single
``@ti.kernel`` parallel over markers. There are no host round trips on the
per-step hot path, and no incremental error accumulation: each call is a
fresh, from-scratch recompute from the authoritative vertex buffer, not an
update-in-place, so nothing can drift over many steps. The barycentric
reconstruction is exact for per-triangle affine deformation (any affine map
commutes with a barycentric combination, because barycentric weights sum to
1).

Area bookkeeping is exact by construction. Subdividing a flat triangle at
level L (S = 2**L segments per edge) into a uniform barycentric lattice
always yields S*S == 4**L sub-triangles, and EVERY one of them -- UP- or
DOWN-pointing -- is a similar copy of the parent triangle scaled by 1/S.
This is a standard property of barycentric subdivision: the map from the
reference simplex (in (i, j) lattice space) to the physical triangle
(v0, v1, v2) is affine/linear, so it has exactly one constant Jacobian
everywhere. In reference space, all S*S sub-triangles of a uniform
triangular lattice have identical area (by the regular tiling's symmetry),
and a single constant Jacobian scales all of them into world space by the
same factor. Consequently ``rest_area_m2[t] / (S*S)`` exactly partitions
the parent rest area with zero residual, independent of the parent
triangle's shape.

Degenerate triangles (rest area < 1e-30 m^2) are seeded at level 0 with a
single marker of zero area and a zero-vector normal (not a unit vector --
a degenerate triangle has no well-defined normal). Their area stays pinned
at zero even if later deformation "opens up" the triangle, because the
fixed-at-seed-time ``rest_marker_area_m2`` denominator is zero; the normal,
by contrast, is recomputed fresh from the live cross product on every
``update_markers_from_vertices`` call and so DOES recover a well-defined
direction if the triangle stops being degenerate. This asymmetry is
intentional: area is a fixed force-integration weight that must never
silently reappear from a division guard, while the normal has no similar
correctness hazard.

TAICHI GOTCHA (project memory): this module contains ``@ti.kernel``
functions and therefore MUST NOT add a postponed-evaluation-of-annotations
``__future__`` import at module scope (PEP 563) -- that breaks Taichi's
kernel argument extraction at import time.

Taichi initialization: this module never calls ``ti.init`` (or the
project's GPU-only ``simulation_core.diagnostics.runtime.init_taichi``)
itself. The caller is responsible for initializing Taichi before
constructing a ``TriSurfaceMarkerSeedFields`` -- production code
initializes CUDA/GPU via ``init_taichi``; tests in this repository call
``ti.init(arch=ti.cpu)`` once at module scope (CPU backend, same kernels,
GPU-ready later).
"""

import math

import numpy as np
import taichi as ti

# Hard ceiling on subdivision level, independent of the caller's max_level.
# 2**level is computed as a plain Python/host int for the loop bound and as
# an i32 bit-shift on the device; staying comfortably below the i32 shift
# range (31) keeps both host and device arithmetic unambiguous. In practice
# the per-triangle inner lattice loop (O(S^2) serial work on a single GPU
# thread) is already impractical well before this ceiling is reached.
_MAX_SUPPORTED_SUBDIVISION_LEVEL = 30

# Degenerate-triangle classification threshold, in m^2 (host decision).
_DEGENERATE_REST_AREA_M2 = 1.0e-30

# Guard floor used for denominators inside device kernels (distinct from
# the degenerate-triangle classification threshold above: this is purely a
# div-by-zero guard, chosen far below any physically meaningful area/length
# so it never perturbs a genuine non-degenerate computation).
_DEVICE_DIVISION_GUARD = 1.0e-300


@ti.data_oriented
class TriSurfaceMarkerSeedFields:
    """Structure-of-arrays Taichi fields for deformation-following markers.

    Built for a FIXED marker count ``n``, triangle count ``T``, and vertex
    count ``V`` -- this class does not support resizing after construction.
    Use ``seed_markers_from_tri_surface`` to size and populate an instance
    from a rest triangle mesh.

    Marker fields (dtype chosen to match the solver's existing marker
    representation, ``HibmMpmSurfaceMarkers`` in
    ``simulation_core/coupling/hibm_mpm/core.py``, which stores
    ``x_gamma_m``/``n_gamma`` as ``ti.f32`` vectors):
        positions_m           (n, 3) ti.f32  -- world-space marker position
        normals               (n, 3) ti.f32  -- outward unit normal
        areas_m2              (n,)   ti.f64  -- CURRENT area weight (force
                                                 integration: total force =
                                                 sum(traction * area), so
                                                 this is kept in f64 for
                                                 exact bookkeeping)
        rest_marker_area_m2   (n,)   ti.f64  -- area weight fixed at seed
                                                 time (S*S-way equal
                                                 partition of the parent's
                                                 rest area); used by the
                                                 runtime update as the
                                                 deformation-ratio base
        parent_triangle       (n,)   ti.i32  -- index into the triangle
                                                 arrays below
        barycentric           (n, 3) ti.f64  -- fixed barycentric binding
                                                 (kept in f64: this is the
                                                 coefficient set that must
                                                 reconstruct the marker's
                                                 position exactly every
                                                 step)
        region_id             (n,)   ti.i32  -- copied from the parent
                                                 triangle at seed time

    Per-triangle rest data:
        rest_area_m2        (T,) ti.f64 -- authoritative parent rest area,
                                            computed device-side from the
                                            SAME (f32-stored) vertex buffer
                                            used by the runtime update, so
                                            the deformation ratio is exactly
                                            1.0 when vertices are unchanged
        subdivision_level    (T,) ti.i32
        tri_vertex_ids        (T, 3) ti.i32 -- indices into vertices_m
        tri_marker_offset    (T,) ti.i32  -- exclusive prefix-sum offset
                                             into the marker arrays
        tri_region_id        (T,) ti.i32
        tri_degenerate       (T,) ti.i32  -- 1 if rest area < 1e-30 m^2

    Vertex buffer (caller writes CURRENT deformed vertices here before
    calling ``update_markers_from_vertices``; seeded with the rest mesh):
        vertices_m (V, 3) ti.f32
    """

    def __init__(self, marker_count: int, triangle_count: int, vertex_count: int) -> None:
        if int(marker_count) <= 0:
            raise ValueError("marker_count must be positive")
        if int(triangle_count) <= 0:
            raise ValueError("triangle_count must be positive")
        if int(vertex_count) <= 0:
            raise ValueError("vertex_count must be positive")
        self.marker_count = int(marker_count)
        self.triangle_count = int(triangle_count)
        self.vertex_count = int(vertex_count)

        # --- markers (structure of arrays) ---
        self.positions_m = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_count)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_count)
        self.areas_m2 = ti.field(dtype=ti.f64, shape=self.marker_count)
        self.rest_marker_area_m2 = ti.field(dtype=ti.f64, shape=self.marker_count)
        self.parent_triangle = ti.field(dtype=ti.i32, shape=self.marker_count)
        self.barycentric = ti.Vector.field(3, dtype=ti.f64, shape=self.marker_count)
        self.region_id = ti.field(dtype=ti.i32, shape=self.marker_count)

        # --- per-triangle rest data ---
        self.rest_area_m2 = ti.field(dtype=ti.f64, shape=self.triangle_count)
        self.subdivision_level = ti.field(dtype=ti.i32, shape=self.triangle_count)
        self.tri_vertex_ids = ti.Vector.field(3, dtype=ti.i32, shape=self.triangle_count)
        self.tri_marker_offset = ti.field(dtype=ti.i32, shape=self.triangle_count)
        self.tri_region_id = ti.field(dtype=ti.i32, shape=self.triangle_count)
        self.tri_degenerate = ti.field(dtype=ti.i32, shape=self.triangle_count)

        # --- vertex buffer (caller writes CURRENT deformed vertices here) ---
        self.vertices_m = ti.Vector.field(3, dtype=ti.f32, shape=self.vertex_count)

        # --- scalar reseed flag (0-d field; one host readback per call) ---
        self._reseed_flag = ti.field(dtype=ti.i32, shape=())

    # ------------------------------------------------------------------
    # Host inspection helpers
    # ------------------------------------------------------------------
    def to_numpy(self) -> dict:
        """Snapshot every field as numpy arrays (host round trip; tests only)."""
        return {
            "positions_m": self.positions_m.to_numpy(),
            "normals": self.normals.to_numpy(),
            "areas_m2": self.areas_m2.to_numpy(),
            "rest_marker_area_m2": self.rest_marker_area_m2.to_numpy(),
            "parent_triangle": self.parent_triangle.to_numpy(),
            "barycentric": self.barycentric.to_numpy(),
            "region_id": self.region_id.to_numpy(),
            "rest_area_m2": self.rest_area_m2.to_numpy(),
            "subdivision_level": self.subdivision_level.to_numpy(),
            "tri_vertex_ids": self.tri_vertex_ids.to_numpy(),
            "tri_marker_offset": self.tri_marker_offset.to_numpy(),
            "tri_region_id": self.tri_region_id.to_numpy(),
            "tri_degenerate": self.tri_degenerate.to_numpy(),
            "vertices_m": self.vertices_m.to_numpy(),
        }

    # ------------------------------------------------------------------
    # Closed-form lattice-subdivision helpers (no recursion).
    #
    # For a level-L subdivision (S = 2**L segments per edge), lattice point
    # P(i, j) for i, j >= 0, i + j <= S has barycentric coordinates
    # ((S - i - j) / S, i / S, j / S) with respect to (v0, v1, v2).
    #
    #   UP(i, j)   (valid for i + j <= S - 1): corners P(i,j), P(i+1,j),
    #                                          P(i,j+1)
    #   DOWN(i, j) (valid for i + j <= S - 2): corners P(i+1,j), P(i,j+1),
    #                                          P(i+1,j+1)
    #
    # Each sub-triangle's centroid barycentric coordinate is the mean of
    # its 3 corners' barycentric coordinates. The two funcs below sum the
    # three UN-normalized (S - a - b, a, b) corner tuples first and divide
    # once by (3*S) at the end, rather than normalizing each corner before
    # averaging: since i, j, S are small integers exactly representable in
    # f64, the additions are exact and only the final division rounds.
    # ------------------------------------------------------------------
    @ti.func
    def _up_centroid_barycentric(self, i, j, s):
        s_f = ti.cast(s, ti.f64)
        i_f = ti.cast(i, ti.f64)
        j_f = ti.cast(j, ti.f64)
        p0 = ti.Vector([s_f - i_f - j_f, i_f, j_f], dt=ti.f64)
        p1 = ti.Vector([s_f - (i_f + 1.0) - j_f, i_f + 1.0, j_f], dt=ti.f64)
        p2 = ti.Vector([s_f - i_f - (j_f + 1.0), i_f, j_f + 1.0], dt=ti.f64)
        return (p0 + p1 + p2) / (3.0 * s_f)

    @ti.func
    def _down_centroid_barycentric(self, i, j, s):
        s_f = ti.cast(s, ti.f64)
        i_f = ti.cast(i, ti.f64)
        j_f = ti.cast(j, ti.f64)
        p0 = ti.Vector([s_f - (i_f + 1.0) - j_f, i_f + 1.0, j_f], dt=ti.f64)
        p1 = ti.Vector([s_f - i_f - (j_f + 1.0), i_f, j_f + 1.0], dt=ti.f64)
        p2 = ti.Vector([s_f - (i_f + 1.0) - (j_f + 1.0), i_f + 1.0, j_f + 1.0], dt=ti.f64)
        return (p0 + p1 + p2) / (3.0 * s_f)

    @ti.func
    def _write_marker(self, idx, parent, v0, v1, v2, bary, normal_store, marker_area, region):
        position = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        self.positions_m[idx] = ti.cast(position, ti.f32)
        self.normals[idx] = ti.cast(normal_store, ti.f32)
        self.areas_m2[idx] = marker_area
        self.rest_marker_area_m2[idx] = marker_area
        self.parent_triangle[idx] = parent
        self.barycentric[idx] = bary
        self.region_id[idx] = region

    # ------------------------------------------------------------------
    # Emission kernel (build-time; parallel over triangles).
    # ------------------------------------------------------------------
    @ti.kernel
    def _emit_markers_kernel(self, triangle_count: ti.i32):
        for t in range(triangle_count):
            ids = self.tri_vertex_ids[t]
            v0 = ti.cast(self.vertices_m[ids[0]], ti.f64)
            v1 = ti.cast(self.vertices_m[ids[1]], ti.f64)
            v2 = ti.cast(self.vertices_m[ids[2]], ti.f64)
            region = self.tri_region_id[t]
            level = self.subdivision_level[t]
            s = 1 << level
            offset = self.tri_marker_offset[t]

            edge_cross = (v1 - v0).cross(v2 - v0)
            cross_norm = edge_cross.norm()
            normal_store = edge_cross / ti.max(cross_norm, _DEVICE_DIVISION_GUARD)
            rest_area = 0.5 * cross_norm
            if self.tri_degenerate[t] != 0:
                normal_store = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
                rest_area = 0.0
            self.rest_area_m2[t] = rest_area

            s_f = ti.cast(s, ti.f64)
            marker_area = rest_area / (s_f * s_f)

            k = 0
            for i in range(s):
                for j in range(s - i):
                    bary_up = self._up_centroid_barycentric(i, j, s)
                    self._write_marker(
                        offset + k, t, v0, v1, v2, bary_up, normal_store, marker_area, region
                    )
                    k += 1
                    if j <= s - 2 - i:
                        bary_down = self._down_centroid_barycentric(i, j, s)
                        self._write_marker(
                            offset + k,
                            t,
                            v0,
                            v1,
                            v2,
                            bary_down,
                            normal_store,
                            marker_area,
                            region,
                        )
                        k += 1

    # ------------------------------------------------------------------
    # Runtime update kernel (tier-2 hot path; parallel over markers).
    # ------------------------------------------------------------------
    @ti.kernel
    def _update_markers_kernel(self, marker_count: ti.i32):
        for m in range(marker_count):
            t = self.parent_triangle[m]
            ids = self.tri_vertex_ids[t]
            v0 = ti.cast(self.vertices_m[ids[0]], ti.f64)
            v1 = ti.cast(self.vertices_m[ids[1]], ti.f64)
            v2 = ti.cast(self.vertices_m[ids[2]], ti.f64)
            bary = self.barycentric[m]

            position = bary[0] * v0 + bary[1] * v1 + bary[2] * v2

            edge_cross = (v1 - v0).cross(v2 - v0)
            cross_norm = edge_cross.norm()
            normal = edge_cross / ti.max(cross_norm, _DEVICE_DIVISION_GUARD)
            deformed_area = 0.5 * cross_norm

            rest_area = self.rest_area_m2[t]
            ratio = deformed_area / ti.max(rest_area, _DEVICE_DIVISION_GUARD)
            area = ratio * self.rest_marker_area_m2[m]

            self.positions_m[m] = ti.cast(position, ti.f32)
            self.normals[m] = ti.cast(normal, ti.f32)
            self.areas_m2[m] = area

    # ------------------------------------------------------------------
    # Reseed check kernel (cold path; parallel over triangles).
    # ------------------------------------------------------------------
    @ti.kernel
    def _reseed_check_kernel(
        self,
        triangle_count: ti.i32,
        min_cell_m: ti.f64,
        spacing_safety: ti.f64,
        growth_margin: ti.f64,
    ):
        self._reseed_flag[None] = 0
        threshold = spacing_safety * min_cell_m * growth_margin
        for t in range(triangle_count):
            # Rest-degenerate triangles always retain one zero-area marker;
            # reseeding cannot refine them, so they must not create a
            # permanent cold-path trigger merely because they are collinear.
            if self.tri_degenerate[t] == 0:
                ids = self.tri_vertex_ids[t]
                v0 = ti.cast(self.vertices_m[ids[0]], ti.f64)
                v1 = ti.cast(self.vertices_m[ids[1]], ti.f64)
                v2 = ti.cast(self.vertices_m[ids[2]], ti.f64)
                e0 = (v1 - v0).norm()
                e1 = (v2 - v1).norm()
                e2 = (v0 - v2).norm()
                longest_edge = ti.max(e0, ti.max(e1, e2))
                level = self.subdivision_level[t]
                s_f = ti.cast(1 << level, ti.f64)
                sub_edge = longest_edge / s_f
                if sub_edge > threshold:
                    ti.atomic_max(self._reseed_flag[None], 1)


def _validate_cell_size_xyz_m(cell_size_xyz_m) -> np.ndarray:
    cell = np.asarray(cell_size_xyz_m, dtype=np.float64).reshape(-1)
    if cell.shape[0] != 3:
        raise ValueError("cell_size_xyz_m must have exactly 3 components")
    if not np.all(np.isfinite(cell)):
        raise ValueError("cell_size_xyz_m must contain finite values")
    if np.any(cell <= 0.0):
        raise ValueError("cell_size_xyz_m must be strictly positive")
    return cell


def seed_markers_from_tri_surface(
    vertices_m: np.ndarray,
    triangles: np.ndarray,
    cell_size_xyz_m,
    *,
    spacing_safety: float = 0.75,
    triangle_region_ids: np.ndarray | None = None,
    max_level: int = 12,
) -> TriSurfaceMarkerSeedFields:
    """Seed deformation-following surface markers on a rest triangle mesh.

    Host orchestration (numpy, one-off) picks a subdivision level per
    triangle from its longest rest edge and the local grid spacing, then a
    single ``@ti.kernel`` (parallel over triangles) emits every marker via
    closed-form triangular-lattice subdivision -- no recursion, no
    per-marker host round trips.

    Args:
        vertices_m: (V, 3) rest vertex positions, meters.
        triangles: (T, 3) int vertex indices into vertices_m.
        cell_size_xyz_m: 3 positive fluid grid cell sizes (dx, dy, dz).
        spacing_safety: fraction of min(cell_size_xyz_m) that a sub-triangle
            edge must fit inside (default 0.75).
        triangle_region_ids: optional (T,) int region id per triangle,
            copied onto every marker seeded from that triangle. Defaults to
            all zeros.
        max_level: subdivision-level ceiling; a triangle that would need a
            finer level raises ValueError (actionable: usually a mesh/grid
            unit or scale mismatch).

    Returns:
        A populated TriSurfaceMarkerSeedFields.
    """
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices_m must have shape (V, 3); got {vertices.shape}")
    if vertices.shape[0] == 0:
        raise ValueError("vertices_m must contain at least one vertex")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices_m must contain finite values")

    tris_in = np.asarray(triangles)
    if tris_in.ndim != 2 or tris_in.shape[1] != 3:
        raise ValueError(f"triangles must have shape (T, 3); got {tris_in.shape}")
    if tris_in.shape[0] == 0:
        raise ValueError("triangles must contain at least one triangle")
    tris = tris_in.astype(np.int64)
    if np.any(tris < 0) or np.any(tris >= vertices.shape[0]):
        raise ValueError("triangles references vertex indices out of range for vertices_m")

    cell = _validate_cell_size_xyz_m(cell_size_xyz_m)

    spacing_safety_f = float(spacing_safety)
    if not (math.isfinite(spacing_safety_f) and spacing_safety_f > 0.0):
        raise ValueError("spacing_safety must be a finite positive number")

    max_level_int = int(max_level)
    if max_level_int <= 0:
        raise ValueError("max_level must be positive")
    if max_level_int > _MAX_SUPPORTED_SUBDIVISION_LEVEL:
        raise ValueError(
            f"max_level={max_level_int} exceeds supported ceiling "
            f"{_MAX_SUPPORTED_SUBDIVISION_LEVEL} (the per-triangle lattice "
            "loop, 4**level markers on a single GPU thread, becomes "
            "impractical well before this)"
        )

    triangle_count = int(tris.shape[0])
    if triangle_region_ids is None:
        region_ids = np.zeros(triangle_count, dtype=np.int32)
    else:
        region_ids = np.asarray(triangle_region_ids).astype(np.int32).reshape(-1)
        if region_ids.shape[0] != triangle_count:
            raise ValueError("triangle_region_ids must have shape (T,) matching triangles")

    tri_v0 = vertices[tris[:, 0]]
    tri_v1 = vertices[tris[:, 1]]
    tri_v2 = vertices[tris[:, 2]]
    edge0 = np.linalg.norm(tri_v1 - tri_v0, axis=1)
    edge1 = np.linalg.norm(tri_v2 - tri_v1, axis=1)
    edge2 = np.linalg.norm(tri_v0 - tri_v2, axis=1)
    longest_edge = np.maximum(np.maximum(edge0, edge1), edge2)
    cross = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
    rest_area_host = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate = rest_area_host < _DEGENERATE_REST_AREA_M2

    min_cell = float(np.min(cell))
    threshold = spacing_safety_f * min_cell

    ratio = np.zeros(triangle_count, dtype=np.float64)
    positive_edges = longest_edge > 0.0
    ratio[positive_edges] = longest_edge[positive_edges] / threshold

    level = np.zeros(triangle_count, dtype=np.float64)
    needs_split = ratio > 1.0
    level[needs_split] = np.ceil(np.log2(ratio[needs_split]))
    level = np.maximum(level, 0.0)
    level[degenerate] = 0.0
    level_int64 = level.astype(np.int64)

    if np.any(level_int64 > max_level_int):
        worst = int(np.argmax(level_int64))
        raise ValueError(
            f"triangle {worst} requires subdivision level {int(level_int64[worst])}, "
            f"exceeding max_level={max_level_int} "
            f"(longest edge={longest_edge[worst]:.6g} m, target sub-edge <= "
            f"{threshold:.6g} m from spacing_safety={spacing_safety_f:g} * "
            f"min(cell_size_xyz_m)={min_cell:.6g} m). Check units and mesh/grid "
            "scale mismatch, or relax spacing_safety/max_level."
        )
    level_int32 = level_int64.astype(np.int32)

    marker_counts = np.int64(4) ** level_int64
    total_markers = int(np.sum(marker_counts))
    if total_markers <= 0:
        raise ValueError("computed zero total markers; check triangles/cell_size_xyz_m inputs")

    offsets = np.zeros(triangle_count, dtype=np.int64)
    if triangle_count > 1:
        offsets[1:] = np.cumsum(marker_counts)[:-1]

    seed = TriSurfaceMarkerSeedFields(
        marker_count=total_markers,
        triangle_count=triangle_count,
        vertex_count=int(vertices.shape[0]),
    )

    seed.vertices_m.from_numpy(vertices.astype(np.float32))
    seed.tri_vertex_ids.from_numpy(tris.astype(np.int32))
    seed.subdivision_level.from_numpy(level_int32)
    seed.tri_marker_offset.from_numpy(offsets.astype(np.int32))
    seed.tri_region_id.from_numpy(region_ids)
    seed.tri_degenerate.from_numpy(degenerate.astype(np.int32))

    seed._emit_markers_kernel(triangle_count)
    return seed


def update_markers_from_vertices(seed: TriSurfaceMarkerSeedFields) -> None:
    """Recompute every marker's position/normal/area from seed.vertices_m.

    Caller must have already written the CURRENT deformed vertex positions
    into ``seed.vertices_m`` (e.g. via ``.from_numpy(...)`` or another
    ``@ti.kernel``). Device-resident: one ``@ti.kernel`` parallel over
    markers, zero host round trips.
    """
    if not isinstance(seed, TriSurfaceMarkerSeedFields):
        raise TypeError("seed must be a TriSurfaceMarkerSeedFields instance")
    seed._update_markers_kernel(seed.marker_count)


def tri_surface_needs_reseed(
    seed: TriSurfaceMarkerSeedFields,
    cell_size_xyz_m,
    *,
    spacing_safety: float = 0.75,
    growth_margin: float = 1.0,
) -> bool:
    """Return True if any triangle's CURRENT deformed spacing violates the bound.

    Cold path: a single ``@ti.kernel`` scan over triangles (reading
    ``seed.vertices_m`` directly, independent of the marker fields) plus one
    scalar host readback of the 0-d flag field.

    Args:
        seed: a populated TriSurfaceMarkerSeedFields with CURRENT deformed
            vertices already written into seed.vertices_m.
        cell_size_xyz_m: 3 positive fluid grid cell sizes (dx, dy, dz).
        spacing_safety: same meaning as in seed_markers_from_tri_surface.
        growth_margin: extra multiplicative slack on the allowed threshold
            (>1 tolerates more stretch before flagging; default 1.0 uses
            exactly the seeding-time threshold).
    """
    if not isinstance(seed, TriSurfaceMarkerSeedFields):
        raise TypeError("seed must be a TriSurfaceMarkerSeedFields instance")
    cell = _validate_cell_size_xyz_m(cell_size_xyz_m)

    spacing_safety_f = float(spacing_safety)
    if not (math.isfinite(spacing_safety_f) and spacing_safety_f > 0.0):
        raise ValueError("spacing_safety must be a finite positive number")
    growth_margin_f = float(growth_margin)
    if not (math.isfinite(growth_margin_f) and growth_margin_f > 0.0):
        raise ValueError("growth_margin must be a finite positive number")

    min_cell = float(np.min(cell))
    seed._reseed_check_kernel(seed.triangle_count, min_cell, spacing_safety_f, growth_margin_f)
    return bool(seed._reseed_flag[None])


# ----------------------------------------------------------------------
# Host-only utility (numpy, cold path): resample a small ordered curve of
# markers (e.g. a 2D beam cross-section's boundary polyline, Turek-Hron
# style) by arc length. No Taichi involved.
# ----------------------------------------------------------------------
def resample_polyline_markers_by_arc_length(
    positions_m: np.ndarray,
    normals: np.ndarray,
    areas_m2: np.ndarray,
    target_spacing_m: float,
    *,
    closed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample an ordered marker polyline to near-uniform arc-length spacing.

    Args:
        positions_m: (k, 3) ordered station positions, meters.
        normals: (k, 3) station normals (need not be pre-normalized).
        areas_m2: (k,) station area weights.
        target_spacing_m: desired arc-length spacing between stations.
        closed: if True, the polyline is treated as a closed loop (an
            implicit edge connects the last station back to the first) and
            the output stations are placed periodically around the loop
            (no duplicated closing point). If False, the output stations
            span from the first to the last input station inclusive.

    Returns:
        (positions, normals, areas) resampled arrays, all with the same
        first-axis length ``m``. Output areas are uniform and their sum
        exactly equals ``sum(areas_m2)`` (residual assignment on the last
        station absorbs floating-point rounding from the division).
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    normals_in = np.asarray(normals, dtype=np.float64)
    areas = np.asarray(areas_m2, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape (k, 3); got {positions.shape}")
    station_count_in = positions.shape[0]
    if station_count_in < 2:
        raise ValueError("positions_m must contain at least 2 stations")
    if normals_in.shape != positions.shape:
        raise ValueError("normals must have the same shape as positions_m")
    if areas.shape != (station_count_in,):
        raise ValueError("areas_m2 must have shape (k,) matching positions_m")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must contain finite values")
    if not np.all(np.isfinite(normals_in)):
        raise ValueError("normals must contain finite values")
    if not np.all(np.isfinite(areas)):
        raise ValueError("areas_m2 must contain finite values")
    if np.any(areas < 0.0):
        raise ValueError("areas_m2 must be non-negative")
    target_spacing = float(target_spacing_m)
    if not (math.isfinite(target_spacing) and target_spacing > 0.0):
        raise ValueError("target_spacing_m must be finite and positive")

    total_input_area = float(np.sum(areas))

    normal_norms = np.linalg.norm(normals_in, axis=1)
    safe_norms = np.where(normal_norms > 1.0e-12, normal_norms, 1.0)
    unit_normals_in = normals_in / safe_norms[:, None]

    if closed:
        loop_positions = np.vstack([positions, positions[:1]])
        loop_unit_normals = np.vstack([unit_normals_in, unit_normals_in[:1]])
    else:
        loop_positions = positions
        loop_unit_normals = unit_normals_in

    segment_vectors = np.diff(loop_positions, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("polyline has zero total arc length; cannot resample")

    if closed:
        station_count = max(2, int(math.ceil(total_length / target_spacing)))
        target_lengths = (np.arange(station_count, dtype=np.float64) / station_count) * total_length
    else:
        station_count = max(2, int(math.ceil(total_length / target_spacing)) + 1)
        target_lengths = np.linspace(0.0, total_length, station_count)

    out_positions = np.empty((station_count, 3), dtype=np.float64)
    out_normals = np.empty((station_count, 3), dtype=np.float64)
    for axis in range(3):
        out_positions[:, axis] = np.interp(target_lengths, cumulative, loop_positions[:, axis])
        out_normals[:, axis] = np.interp(target_lengths, cumulative, loop_unit_normals[:, axis])

    out_normal_norms = np.linalg.norm(out_normals, axis=1)
    safe_out_norms = np.where(out_normal_norms > 1.0e-12, out_normal_norms, 1.0)
    out_normals = out_normals / safe_out_norms[:, None]

    if total_input_area > 0.0:
        uniform_area = total_input_area / station_count
        out_areas = np.full(station_count, uniform_area, dtype=np.float64)
        out_areas[-1] = total_input_area - np.sum(out_areas[:-1])
    else:
        out_areas = np.zeros(station_count, dtype=np.float64)

    return out_positions, out_normals, out_areas
