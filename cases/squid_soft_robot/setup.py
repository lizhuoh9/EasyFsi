from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from simulation_core import (
    CartesianGrid,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    GradedGridSpec,
    RefinementRegion,
    SurfaceMesh,
    TaichiRuntimeConfig,
    TriSurfaceRegionDiagnostics,
    build_graded_grid,
)

from .cli import PRESSURE_SOLVER_CHOICES
from .source_config import (
    _face_ids_for_region,
    _selection_ids_as_int_tuple,
    source_config_requests_region14_aperture_carve,
    source_config_solid_obstacle_particle_region_ids,
    source_config_volume_particle_cache_path,
)
from .spec import SquidReducedSpec, required_tuple3


def _surface_mesh_path(config: dict[str, object]) -> Path:
    mesh_path = config.get("surface_mesh_cache_path") or config.get("mesh_path")
    if not isinstance(mesh_path, str) or not mesh_path:
        raise ValueError("source config does not contain a mesh path")
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _cell_indices_for_points(
    points_m: np.ndarray,
    grid: CartesianGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must be an Nx3 array")
    faces = (
        np.asarray(grid.cell_faces_x_m, dtype=np.float64),
        np.asarray(grid.cell_faces_y_m, dtype=np.float64),
        np.asarray(grid.cell_faces_z_m, dtype=np.float64),
    )
    nodes = tuple(int(value) for value in grid.grid_nodes)
    i = np.searchsorted(faces[0], points[:, 0], side="right") - 1
    j = np.searchsorted(faces[1], points[:, 1], side="right") - 1
    k = np.searchsorted(faces[2], points[:, 2], side="right") - 1
    tolerance = 1.0e-12
    on_x_max = np.isclose(points[:, 0], faces[0][-1], rtol=0.0, atol=tolerance)
    on_y_max = np.isclose(points[:, 1], faces[1][-1], rtol=0.0, atol=tolerance)
    on_z_max = np.isclose(points[:, 2], faces[2][-1], rtol=0.0, atol=tolerance)
    i[on_x_max] = nodes[0] - 1
    j[on_y_max] = nodes[1] - 1
    k[on_z_max] = nodes[2] - 1
    valid = (
        (i >= 0)
        & (i < nodes[0])
        & (j >= 0)
        & (j < nodes[1])
        & (k >= 0)
        & (k < nodes[2])
    )
    return i.astype(np.int64), j.astype(np.int64), k.astype(np.int64), valid


def _surface_region_seed_mask(
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    region_ids: Sequence[int],
    radius_cells: int = 1,
    normal_probe_distance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    import trimesh

    nodes = tuple(int(value) for value in grid.grid_nodes)
    seed = np.zeros(nodes, dtype=bool)
    unique_region_ids = tuple(sorted({int(value) for value in region_ids}))
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in unique_region_ids:
        face_ids = _face_ids_for_region(dict(config), region_id)
        region_face_counts[str(region_id)] = len(face_ids)
        selected_face_ids.extend(face_ids)
    radius = max(0, int(radius_cells))
    if not selected_face_ids:
        return seed, {
            "fluid_active_mask_surface_seed_region_ids": unique_region_ids,
            "fluid_active_mask_surface_seed_face_count": 0,
            "fluid_active_mask_surface_seed_point_count": 0,
            "fluid_active_mask_surface_seed_point_in_grid_count": 0,
            "fluid_active_mask_surface_seed_cell_count": 0,
            "fluid_active_mask_surface_seed_radius_cells": radius,
            "fluid_active_mask_surface_seed_normal_probe_distance_m": max(
                0.0,
                float(normal_probe_distance_m),
            ),
            "fluid_active_mask_surface_seed_normal_probe_point_count": 0,
            "fluid_active_mask_surface_seed_region_face_counts": region_face_counts,
        }
    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    points = np.concatenate(
        (
            centroids,
            tri[:, 0, :],
            tri[:, 1, :],
            tri[:, 2, :],
        ),
        axis=0,
    )
    normal_probe_point_count = 0
    normal_probe_distance = max(0.0, float(normal_probe_distance_m))
    if normal_probe_distance > 0.0:
        raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
        normal_norms = np.linalg.norm(raw_normals, axis=1)
        valid_normals = normal_norms > 1.0e-30
        if np.any(valid_normals):
            unit_normals = np.zeros_like(raw_normals)
            unit_normals[valid_normals] = (
                raw_normals[valid_normals]
                / normal_norms[valid_normals, None]
            )
            normal_probe_points = (
                centroids[valid_normals]
                + unit_normals[valid_normals] * normal_probe_distance
            )
            normal_probe_point_count = int(normal_probe_points.shape[0])
            points = np.concatenate((points, normal_probe_points), axis=0)
    i, j, k, valid = _cell_indices_for_points(points, grid)
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                seed[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return seed, {
        "fluid_active_mask_surface_seed_region_ids": unique_region_ids,
        "fluid_active_mask_surface_seed_face_count": int(len(selected_face_ids)),
        "fluid_active_mask_surface_seed_point_count": int(points.shape[0]),
        "fluid_active_mask_surface_seed_point_in_grid_count": int(np.count_nonzero(valid)),
        "fluid_active_mask_surface_seed_cell_count": int(np.count_nonzero(seed)),
        "fluid_active_mask_surface_seed_radius_cells": radius,
        "fluid_active_mask_surface_seed_normal_probe_distance_m": normal_probe_distance,
        "fluid_active_mask_surface_seed_normal_probe_point_count": normal_probe_point_count,
        "fluid_active_mask_surface_seed_region_face_counts": region_face_counts,
    }


def _clear_surface_region_normal_probe_obstacle_cells(
    obstacle: np.ndarray,
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    region_ids: Sequence[int],
    normal_probe_distance_m: float,
    radius_cells: int = 0,
) -> dict[str, object]:
    import trimesh

    if obstacle.shape != tuple(int(value) for value in grid.grid_nodes):
        raise ValueError("obstacle shape must match grid.grid_nodes")
    unique_region_ids = tuple(sorted({int(value) for value in region_ids}))
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in unique_region_ids:
        face_ids = _face_ids_for_region(dict(config), region_id)
        region_face_counts[str(region_id)] = len(face_ids)
        selected_face_ids.extend(face_ids)
    probe_distance = max(0.0, float(normal_probe_distance_m))
    radius = max(0, int(radius_cells))
    if not selected_face_ids or probe_distance <= 0.0:
        return {
            "fluid_active_mask_surface_probe_clear_region_ids": unique_region_ids,
            "fluid_active_mask_surface_probe_clear_face_count": int(
                len(selected_face_ids)
            ),
            "fluid_active_mask_surface_probe_clear_point_count": 0,
            "fluid_active_mask_surface_probe_clear_cell_count": 0,
            "fluid_active_mask_surface_probe_clear_cells_ijk": (),
            "fluid_active_mask_surface_probe_clear_radius_cells": radius,
            "fluid_active_mask_surface_probe_clear_distance_m": probe_distance,
            "fluid_active_mask_surface_probe_clear_region_face_counts": (
                region_face_counts
            ),
        }

    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    normal_norms = np.linalg.norm(raw_normals, axis=1)
    valid_normals = normal_norms > 1.0e-30
    points = np.empty((0, 3), dtype=np.float64)
    if np.any(valid_normals):
        unit_normals = (
            raw_normals[valid_normals]
            / normal_norms[valid_normals, None]
        )
        points = centroids[valid_normals] + unit_normals * probe_distance

    nodes = tuple(int(value) for value in grid.grid_nodes)
    i, j, k, valid = _cell_indices_for_points(points, grid)
    clear_mask = np.zeros(obstacle.shape, dtype=bool)
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                clear_mask[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    cleared_mask = obstacle & clear_mask
    cleared = int(np.count_nonzero(cleared_mask))
    cleared_cells_ijk = tuple(
        tuple(int(value) for value in cell)
        for cell in np.argwhere(cleared_mask)
    )
    obstacle[clear_mask] = False
    return {
        "fluid_active_mask_surface_probe_clear_region_ids": unique_region_ids,
        "fluid_active_mask_surface_probe_clear_face_count": int(len(selected_face_ids)),
        "fluid_active_mask_surface_probe_clear_point_count": int(points.shape[0]),
        "fluid_active_mask_surface_probe_clear_cell_count": cleared,
        "fluid_active_mask_surface_probe_clear_cells_ijk": cleared_cells_ijk,
        "fluid_active_mask_surface_probe_clear_radius_cells": radius,
        "fluid_active_mask_surface_probe_clear_distance_m": probe_distance,
        "fluid_active_mask_surface_probe_clear_region_face_counts": region_face_counts,
    }


def _solid_band_protection_mask_from_cells(
    shape: Sequence[int],
    cells_ijk: Sequence[Sequence[int]],
    *,
    radius_cells: int = 0,
) -> np.ndarray:
    mask = np.zeros(tuple(int(value) for value in shape), dtype=np.int32)
    radius = max(0, int(radius_cells))
    offsets = tuple(range(-radius, radius + 1))
    nx, ny, nz = mask.shape
    for cell in cells_ijk:
        if len(cell) != 3:
            continue
        ci, cj, ck = (int(value) for value in cell)
        for di in offsets:
            i = ci + di
            if i < 0 or i >= nx:
                continue
            for dj in offsets:
                j = cj + dj
                if j < 0 or j >= ny:
                    continue
                for dk in offsets:
                    k = ck + dk
                    if 0 <= k < nz:
                        mask[i, j, k] = 1
    return mask


def _mark_particle_obstacle_cells(
    *,
    grid: CartesianGrid,
    particle_positions_m: np.ndarray,
    particle_region_ids: np.ndarray,
    obstacle_region_ids: Sequence[int],
    dilation_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    nodes = tuple(int(value) for value in grid.grid_nodes)
    obstacle = np.zeros(nodes, dtype=bool)
    obstacle_regions = {int(value) for value in obstacle_region_ids}
    selected = np.isin(
        np.asarray(particle_region_ids, dtype=np.int32),
        np.asarray(sorted(obstacle_regions), dtype=np.int32),
    )
    i, j, k, valid = _cell_indices_for_points(
        np.asarray(particle_positions_m, dtype=np.float64)[selected],
        grid,
    )
    selected_valid_count = int(np.count_nonzero(valid))
    radius = max(0, int(dilation_cells))
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                obstacle[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return obstacle, {
        "particle_obstacle_region_ids": tuple(sorted(obstacle_regions)),
        "selected_particle_count": int(np.count_nonzero(selected)),
        "selected_particle_in_grid_count": selected_valid_count,
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "particle_stamp_dilation_cells": radius,
        "particle_stamp_method": "volume_particle_cell_stamp",
    }


def _mark_surface_obstacle_cells(
    *,
    config: Mapping[str, object],
    grid: CartesianGrid,
    surface_region_ids: Sequence[int],
    dilation_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    import trimesh

    nodes = tuple(int(value) for value in grid.grid_nodes)
    obstacle = np.zeros(nodes, dtype=bool)
    mesh_path = _surface_mesh_path(dict(config))
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    selected_face_ids: list[int] = []
    region_face_counts: dict[str, int] = {}
    for region_id in surface_region_ids:
        face_ids = _face_ids_for_region(dict(config), int(region_id))
        region_face_counts[str(int(region_id))] = len(face_ids)
        selected_face_ids.extend(face_ids)
    if not selected_face_ids:
        return obstacle, {
            "surface_obstacle_region_ids": tuple(int(value) for value in surface_region_ids),
            "selected_surface_face_count": 0,
            "selected_surface_point_count": 0,
            "selected_surface_point_in_grid_count": 0,
            "raw_solid_obstacle_cell_count": 0,
            "surface_stamp_dilation_cells": max(0, int(dilation_cells)),
            "surface_stamp_method": "surface_triangle_centroid_and_vertex_cell_stamp",
            "surface_region_face_counts": region_face_counts,
        }
    selected = faces[np.asarray(selected_face_ids, dtype=np.int64)]
    tri = vertices[selected]
    centroids = np.mean(tri, axis=1)
    points = np.concatenate(
        (
            centroids,
            tri[:, 0, :],
            tri[:, 1, :],
            tri[:, 2, :],
        ),
        axis=0,
    )
    i, j, k, valid = _cell_indices_for_points(points, grid)
    radius = max(0, int(dilation_cells))
    offsets = tuple(range(-radius, radius + 1))
    for di in offsets:
        ii = i[valid] + di
        valid_i = (ii >= 0) & (ii < nodes[0])
        for dj in offsets:
            jj = j[valid] + dj
            valid_j = (jj >= 0) & (jj < nodes[1])
            for dk in offsets:
                kk = k[valid] + dk
                valid_k = (kk >= 0) & (kk < nodes[2])
                valid_offset = valid_i & valid_j & valid_k
                obstacle[
                    ii[valid_offset],
                    jj[valid_offset],
                    kk[valid_offset],
                ] = True
    return obstacle, {
        "surface_obstacle_region_ids": tuple(int(value) for value in surface_region_ids),
        "surface_mesh_path": str(mesh_path),
        "selected_surface_face_count": int(len(selected_face_ids)),
        "selected_surface_point_count": int(points.shape[0]),
        "selected_surface_point_in_grid_count": int(np.count_nonzero(valid)),
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "surface_stamp_dilation_cells": radius,
        "surface_stamp_method": "surface_triangle_centroid_and_vertex_cell_stamp",
        "surface_region_face_counts": region_face_counts,
    }


def _apply_region14_opening_carve_to_obstacle(
    obstacle: np.ndarray,
    grid: CartesianGrid,
    *,
    aperture_geometry: Mapping[str, object],
    radius_cells: int,
    depth_cells: int,
) -> int:
    if not bool(aperture_geometry.get("available", False)):
        return 0
    center = aperture_geometry.get("area_weighted_centroid_m", ())
    if not isinstance(center, Sequence) or len(center) < 3:
        return 0
    radius_m = float(aperture_geometry.get("vertex_radius_p95_m", 0.0))
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        return 0
    center_x = float(center[0])
    center_y = float(center[1])
    center_z = float(center[2])
    max_xy_spacing_m = max(
        max(float(value) for value in grid.cell_widths_x_m),
        max(float(value) for value in grid.cell_widths_y_m),
    )
    max_z_spacing_m = max(float(value) for value in grid.cell_widths_z_m)
    carve_radius_m = radius_m + max(0, int(radius_cells)) * max_xy_spacing_m
    carve_half_depth_m = max(1, int(depth_cells)) * max_z_spacing_m
    x = np.asarray(grid.cell_centers_x_m, dtype=np.float64)
    y = np.asarray(grid.cell_centers_y_m, dtype=np.float64)
    z = np.asarray(grid.cell_centers_z_m, dtype=np.float64)
    radial = (
        (x[:, None, None] - center_x) ** 2
        + (y[None, :, None] - center_y) ** 2
    ) <= carve_radius_m * carve_radius_m
    axial = np.abs(z[None, None, :] - center_z) <= carve_half_depth_m
    carve = radial & axial
    carved = int(np.count_nonzero(obstacle & carve))
    obstacle[carve] = False
    return carved


def _z_min_connected_active_mask(
    candidate_fluid_mask: np.ndarray,
    *,
    seed_radius_cells: int,
) -> tuple[np.ndarray, int]:
    active = np.asarray(candidate_fluid_mask, dtype=bool)
    seed = np.zeros(active.shape, dtype=bool)
    nz = active.shape[2]
    seed_depth = max(1, int(seed_radius_cells))
    seed_depth = min(seed_depth, nz)
    seed[:, :, :seed_depth] = True
    return _connected_active_mask(active, seed)


def _connected_active_mask(
    candidate_fluid_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    active = np.asarray(candidate_fluid_mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    if seed.shape != active.shape:
        raise ValueError("seed_mask shape must match candidate_fluid_mask")
    visited = np.zeros(active.shape, dtype=bool)
    nx, ny, nz = active.shape
    stack: list[tuple[int, int, int]] = []
    seed_indices = np.argwhere(active & seed)
    for i, j, k in seed_indices:
        index = (int(i), int(j), int(k))
        visited[index] = True
        stack.append(index)
    seed_count = len(stack)
    while stack:
        i, j, k = stack.pop()
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if (
                0 <= ni < nx
                and 0 <= nj < ny
                and 0 <= nk < nz
                and active[ni, nj, nk]
                and not visited[ni, nj, nk]
            ):
                visited[ni, nj, nk] = True
                stack.append((ni, nj, nk))
    return visited, seed_count


def _component_count(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros(active.shape, dtype=bool)
    nx, ny, nz = active.shape
    count = 0
    for seed_index in zip(*np.nonzero(active), strict=False):
        if visited[seed_index]:
            continue
        count += 1
        stack = [tuple(int(value) for value in seed_index)]
        visited[seed_index] = True
        while stack:
            i, j, k = stack.pop()
            for ni, nj, nk in (
                (i - 1, j, k),
                (i + 1, j, k),
                (i, j - 1, k),
                (i, j + 1, k),
                (i, j, k - 1),
                (i, j, k + 1),
            ):
                if (
                    0 <= ni < nx
                    and 0 <= nj < ny
                    and 0 <= nk < nz
                    and active[ni, nj, nk]
                    and not visited[ni, nj, nk]
                ):
                    visited[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
    return count


def _mask_bbox_ijk(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return None
    mins = tuple(int(value) for value in np.min(indices, axis=0))
    maxs = tuple(int(value) for value in np.max(indices, axis=0))
    return mins, maxs


def _minimum_obstacle_carve_path(
    *,
    obstacle: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int] | None:
    obstacle_bool = np.asarray(obstacle, dtype=bool)
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if obstacle_bool.shape != source.shape or source.shape != target.shape:
        raise ValueError("obstacle, source_mask, and target_mask shapes must match")
    nx, ny, nz = obstacle_bool.shape
    total = nx * ny * nz
    source_indices = np.flatnonzero(source.reshape(-1))
    if source_indices.size == 0 or not np.any(target):
        return None
    target_flat = target.reshape(-1)
    obstacle_flat = obstacle_bool.reshape(-1)
    max_distance = np.iinfo(np.int32).max
    distance = np.full(total, max_distance, dtype=np.int32)
    previous = np.full(total, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for flat_index in source_indices:
        index = int(flat_index)
        distance[index] = 0
        queue.append(index)

    def unravel(index: int) -> tuple[int, int, int]:
        i = index // (ny * nz)
        rem = index - i * ny * nz
        j = rem // nz
        k = rem - j * nz
        return int(i), int(j), int(k)

    def ravel(i: int, j: int, k: int) -> int:
        return int((i * ny + j) * nz + k)

    end_index = -1
    while queue:
        index = queue.popleft()
        if target_flat[index]:
            end_index = int(index)
            break
        i, j, k = unravel(index)
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            neighbor = ravel(ni, nj, nk)
            step_cost = 1 if obstacle_flat[neighbor] else 0
            candidate_distance = int(distance[index]) + step_cost
            if candidate_distance >= int(distance[neighbor]):
                continue
            distance[neighbor] = candidate_distance
            previous[neighbor] = index
            if step_cost == 0:
                queue.appendleft(neighbor)
            else:
                queue.append(neighbor)
    if end_index < 0:
        return None
    path_indices: list[int] = []
    cursor = end_index
    while cursor >= 0:
        path_indices.append(int(cursor))
        cursor = int(previous[cursor])
    path_indices.reverse()
    return [unravel(index) for index in path_indices], int(distance[end_index])


def _connect_surface_seed_components_to_zmin(
    obstacle: np.ndarray,
    *,
    boundary_seed: np.ndarray,
    surface_seed: np.ndarray,
    max_carve_cells: int,
) -> dict[str, object]:
    if obstacle.shape != boundary_seed.shape or boundary_seed.shape != surface_seed.shape:
        raise ValueError("obstacle, boundary_seed, and surface_seed shapes must match")
    max_carve = max(0, int(max_carve_cells))
    carved_mask = np.zeros(obstacle.shape, dtype=bool)
    total_carved = 0
    connected_paths = 0
    skipped_max_carve = False
    path_reports: list[dict[str, object]] = []

    candidate_fluid = ~np.asarray(obstacle, dtype=bool)
    zmin_active, _ = _connected_active_mask(candidate_fluid, boundary_seed)
    surface_active, _ = _connected_active_mask(candidate_fluid, surface_seed)
    target = surface_active & ~zmin_active
    initial_unreachable_cells = int(np.count_nonzero(target))
    initial_unreachable_components = _component_count(target)

    while np.any(target):
        path_result = _minimum_obstacle_carve_path(
            obstacle=obstacle,
            source_mask=zmin_active,
            target_mask=target,
        )
        if path_result is None:
            break
        path, obstacle_cost = path_result
        if obstacle_cost <= 0:
            break
        if total_carved + obstacle_cost > max_carve:
            skipped_max_carve = True
            break
        path_carved = 0
        for i, j, k in path:
            if bool(obstacle[i, j, k]):
                obstacle[i, j, k] = False
                carved_mask[i, j, k] = True
                path_carved += 1
        if path_carved <= 0:
            break
        total_carved += path_carved
        connected_paths += 1
        path_reports.append(
            {
                "path_cell_count": int(len(path)),
                "carved_cell_count": int(path_carved),
                "obstacle_cost": int(obstacle_cost),
                "start_ijk": tuple(int(value) for value in path[0]),
                "end_ijk": tuple(int(value) for value in path[-1]),
            }
        )
        candidate_fluid = ~np.asarray(obstacle, dtype=bool)
        zmin_active, _ = _connected_active_mask(candidate_fluid, boundary_seed)
        surface_active, _ = _connected_active_mask(candidate_fluid, surface_seed)
        target = surface_active & ~zmin_active

    final_unreachable_cells = int(np.count_nonzero(target))
    final_unreachable_components = _component_count(target)
    bbox = _mask_bbox_ijk(carved_mask)
    return {
        "enabled": True,
        "max_carve_cells": int(max_carve),
        "initial_unreachable_surface_seed_cell_count": initial_unreachable_cells,
        "initial_unreachable_surface_seed_component_count": initial_unreachable_components,
        "final_unreachable_surface_seed_cell_count": final_unreachable_cells,
        "final_unreachable_surface_seed_component_count": final_unreachable_components,
        "connected_path_count": int(connected_paths),
        "carved_cell_count": int(total_carved),
        "skipped_by_max_carve_limit": bool(skipped_max_carve),
        "carved_bbox_ijk": None if bbox is None else bbox,
        "paths": tuple(path_reports),
    }


def build_source_config_fluid_obstacle_mask(
    *,
    config: Mapping[str, object],
    source_config_path: Path,
    grid: CartesianGrid,
    aperture_geometry: Mapping[str, object],
    connect_surface_seeds_to_zmin: bool = False,
    surface_seed_zmin_connection_max_carve_cells: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    analysis = config.get("analysis_settings", {})
    if not isinstance(analysis, Mapping):
        raise ValueError("source_config analysis_settings must be a mapping")
    mode = str(analysis.get("fluid_active_mask_mode", ""))
    if mode not in {"ibamr_like_connected_component", "fsi_connected_component"}:
        raise ValueError(
            "source_config fluid active mask mode must be "
            "ibamr_like_connected_component or fsi_connected_component"
        )
    seed_sides = tuple(
        str(value)
        for value in analysis.get("fluid_active_mask_seed_boundary_sides", ("z_min",))
    )
    if "z_min" not in seed_sides:
        raise ValueError("refactored squid source-config active mask currently requires z_min seeding")
    surface_only_region_ids = tuple(
        sorted(
            set(
                _selection_ids_as_int_tuple(
                    analysis.get("solid_obstacle_surface_only_region_ids", ()),
                )
            )
        )
    )
    cache_path: Path | None = None
    available_region_ids: tuple[int, ...] = ()
    obstacle_region_ids: tuple[int, ...] = ()
    obstacle_report: dict[str, object]
    if surface_only_region_ids:
        obstacle, obstacle_report = _mark_surface_obstacle_cells(
            config=config,
            grid=grid,
            surface_region_ids=surface_only_region_ids,
            dilation_cells=int(analysis.get("solid_obstacle_mask_dilation_cells", 0) or 0),
        )
        obstacle_region_ids = surface_only_region_ids
    else:
        cache_path = source_config_volume_particle_cache_path(source_config_path)
        particle_cache = np.load(cache_path)
        positions = np.asarray(particle_cache["particle_rest_positions_m"], dtype=np.float64)
        region_ids = np.asarray(particle_cache["particle_region_ids"], dtype=np.int32)
        available_region_ids = tuple(
            int(value) for value in np.unique(region_ids).astype(np.int32).tolist()
        )
        obstacle_region_ids = source_config_solid_obstacle_particle_region_ids(
            config,
            available_region_ids,
        )
        obstacle, obstacle_report = _mark_particle_obstacle_cells(
            grid=grid,
            particle_positions_m=positions,
            particle_region_ids=region_ids,
            obstacle_region_ids=obstacle_region_ids,
            dilation_cells=int(analysis.get("solid_obstacle_mask_dilation_cells", 0) or 0),
        )
    carved_count = 0
    if source_config_requests_region14_aperture_carve(config):
        carved_count = _apply_region14_opening_carve_to_obstacle(
            obstacle,
            grid,
            aperture_geometry=aperture_geometry,
            radius_cells=int(
                analysis.get("solid_obstacle_opening_carve_radius_cells", 1) or 1
            ),
            depth_cells=int(
                analysis.get("solid_obstacle_opening_carve_depth_cells", 2) or 2
            ),
        )
    seed_radius_cells = int(analysis.get("fluid_active_mask_seed_radius_cells", 1) or 1)
    surface_seed_normal_probe_cells = max(
        0,
        int(analysis.get("fluid_active_mask_surface_seed_normal_probe_cells", 1) or 0),
    )
    surface_seed_normal_probe_distance_m = 0.0
    if surface_seed_normal_probe_cells > 0:
        min_cell_spacing_m = min(
            min(float(value) for value in grid.cell_widths_x_m),
            min(float(value) for value in grid.cell_widths_y_m),
            min(float(value) for value in grid.cell_widths_z_m),
        )
        surface_seed_normal_probe_distance_m = (
            float(surface_seed_normal_probe_cells) * min_cell_spacing_m
        )
    clear_region_ids = _selection_ids_as_int_tuple(
        analysis.get("fluid_active_mask_surface_probe_clear_region_ids", ()),
    )
    if (
        not clear_region_ids
        and bool(
            analysis.get(
                "fluid_active_mask_clear_primary_fsi_surface_probe_obstacles",
                True,
            )
        )
    ):
        moving_surface_ids = _selection_ids_as_int_tuple(
            analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
        )
        if moving_surface_ids:
            clear_region_ids = (int(moving_surface_ids[0]),)
    if clear_region_ids:
        surface_probe_clear_report = _clear_surface_region_normal_probe_obstacle_cells(
            obstacle,
            config=config,
            grid=grid,
            region_ids=clear_region_ids,
            normal_probe_distance_m=surface_seed_normal_probe_distance_m,
            radius_cells=int(
                analysis.get(
                    "fluid_active_mask_surface_probe_clear_radius_cells",
                    0,
                )
                or 0
            ),
        )
    else:
        surface_probe_clear_report = {
            "fluid_active_mask_surface_probe_clear_region_ids": (),
            "fluid_active_mask_surface_probe_clear_face_count": 0,
            "fluid_active_mask_surface_probe_clear_point_count": 0,
            "fluid_active_mask_surface_probe_clear_cell_count": 0,
            "fluid_active_mask_surface_probe_clear_cells_ijk": (),
            "fluid_active_mask_surface_probe_clear_radius_cells": 0,
            "fluid_active_mask_surface_probe_clear_distance_m": (
                surface_seed_normal_probe_distance_m
            ),
            "fluid_active_mask_surface_probe_clear_region_face_counts": {},
        }
    candidate_fluid = ~obstacle
    boundary_seed = np.zeros(candidate_fluid.shape, dtype=bool)
    if "z_min" in seed_sides:
        seed_depth = min(max(1, seed_radius_cells), candidate_fluid.shape[2])
        boundary_seed[:, :, :seed_depth] = True
    seed_region_ids = set(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_region_ids", ()),
        )
    )
    seed_region_ids.update(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_region_id", ()),
        )
    )
    seed_region_ids.update(
        _selection_ids_as_int_tuple(
            analysis.get("fluid_active_mask_seed_surface_region_ids", ()),
        )
    )
    if bool(analysis.get("fluid_active_mask_seed_fsi_contact_surfaces", True)):
        seed_region_ids.update(
            _selection_ids_as_int_tuple(
                analysis.get("solid_obstacle_moving_fsi_contact_surface_region_ids", ()),
            )
        )
    surface_seed_report: dict[str, object]
    if seed_region_ids:
        surface_seed, surface_seed_report = _surface_region_seed_mask(
            config=config,
            grid=grid,
            region_ids=tuple(sorted(seed_region_ids)),
            radius_cells=seed_radius_cells,
            normal_probe_distance_m=surface_seed_normal_probe_distance_m,
        )
    else:
        surface_seed = np.zeros(candidate_fluid.shape, dtype=bool)
        surface_seed_report = {
            "fluid_active_mask_surface_seed_region_ids": (),
            "fluid_active_mask_surface_seed_face_count": 0,
            "fluid_active_mask_surface_seed_point_count": 0,
            "fluid_active_mask_surface_seed_point_in_grid_count": 0,
            "fluid_active_mask_surface_seed_cell_count": 0,
            "fluid_active_mask_surface_seed_radius_cells": seed_radius_cells,
            "fluid_active_mask_surface_seed_normal_probe_distance_m": (
                surface_seed_normal_probe_distance_m
            ),
            "fluid_active_mask_surface_seed_normal_probe_point_count": 0,
            "fluid_active_mask_surface_seed_region_face_counts": {},
        }
    surface_seed_zmin_connection_report: dict[str, object] = {
        "enabled": False,
        "reason": "not_requested",
        "max_carve_cells": int(max(0, surface_seed_zmin_connection_max_carve_cells)),
    }
    if connect_surface_seeds_to_zmin:
        surface_seed_zmin_connection_report = _connect_surface_seed_components_to_zmin(
            obstacle,
            boundary_seed=boundary_seed,
            surface_seed=surface_seed,
            max_carve_cells=surface_seed_zmin_connection_max_carve_cells,
        )
    candidate_fluid = ~obstacle
    active_water, seed_cell_count = _connected_active_mask(
        candidate_fluid,
        boundary_seed | surface_seed,
    )
    final_obstacle = ~active_water
    report = {
        "enabled": True,
        "method": "source_config_cad_obstacle_z_min_connected_component",
        "mode": mode,
        "source_config_path": str(source_config_path),
        "volume_particle_cache_path": None if cache_path is None else str(cache_path),
        "grid_nodes": tuple(int(value) for value in grid.grid_nodes),
        "available_particle_region_ids": available_region_ids,
        "obstacle_region_ids": obstacle_region_ids,
        "solid_obstacle_opening_carved_cell_count": int(carved_count),
        "fluid_active_mask_seed_boundary_sides": seed_sides,
        "fluid_active_mask_seed_radius_cells": seed_radius_cells,
        "fluid_active_mask_boundary_seed_cell_count": int(
            np.count_nonzero(boundary_seed & candidate_fluid)
        ),
        "fluid_active_mask_surface_seed_candidate_cell_count": int(
            np.count_nonzero(surface_seed & candidate_fluid)
        ),
        "fluid_active_mask_seed_cell_count": int(seed_cell_count),
        "raw_solid_obstacle_cell_count": int(np.count_nonzero(obstacle)),
        "candidate_fluid_cell_count": int(np.count_nonzero(candidate_fluid)),
        "fluid_active_cell_count": int(np.count_nonzero(active_water)),
        "fluid_inactive_cell_count": int(final_obstacle.size - np.count_nonzero(active_water)),
        "final_obstacle_cell_count": int(np.count_nonzero(final_obstacle)),
        "host_device_transfer_policy": "one_time_initial_obstacle_from_numpy_before_steps",
        **obstacle_report,
        **surface_probe_clear_report,
        **surface_seed_report,
        "fluid_active_mask_surface_seed_zmin_connection": (
            surface_seed_zmin_connection_report
        ),
    }
    return final_obstacle.astype(np.int32), report


def _active_water_mask_for_points(
    points_m: np.ndarray,
    *,
    grid: CartesianGrid,
    obstacle_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(obstacle_mask, dtype=np.int32)
    if mask.shape != tuple(grid.grid_nodes):
        raise ValueError(
            f"obstacle_mask shape {mask.shape!r} does not match grid_nodes {tuple(grid.grid_nodes)!r}"
        )
    i, j, k, valid = _cell_indices_for_points(points_m, grid)
    active = np.zeros(valid.shape, dtype=bool)
    active[valid] = mask[i[valid], j[valid], k[valid]] == 0
    return active


def _orient_normals_to_active_water_mask(
    centroids_m: np.ndarray,
    normals: np.ndarray,
    region_ids: np.ndarray,
    *,
    grid: CartesianGrid,
    obstacle_mask: np.ndarray,
    probe_distance_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    plus_active = _active_water_mask_for_points(
        centroids_m + normals * probe_distance_m,
        grid=grid,
        obstacle_mask=obstacle_mask,
    )
    minus_active = _active_water_mask_for_points(
        centroids_m - normals * probe_distance_m,
        grid=grid,
        obstacle_mask=obstacle_mask,
    )
    flip = (~plus_active) & minus_active
    oriented = np.array(normals, copy=True)
    oriented[flip] *= -1.0
    final_active = plus_active | flip
    both_active = plus_active & minus_active
    neither_active = (~plus_active) & (~minus_active)
    by_region: dict[str, dict[str, int]] = {}
    for region in sorted({int(value) for value in region_ids.tolist()}):
        mask = region_ids == region
        by_region[str(region)] = {
            "face_count": int(np.count_nonzero(mask)),
            "plus_active_count": int(np.count_nonzero(plus_active & mask)),
            "minus_active_count": int(np.count_nonzero(minus_active & mask)),
            "flipped_count": int(np.count_nonzero(flip & mask)),
            "both_active_count": int(np.count_nonzero(both_active & mask)),
            "neither_active_count": int(np.count_nonzero(neither_active & mask)),
            "final_active_count": int(np.count_nonzero(final_active & mask)),
        }
    return oriented, {
        "method": "source_config_active_water_mask_probe_orientation",
        "probe_distance_m": float(probe_distance_m),
        "flipped_count": int(np.count_nonzero(flip)),
        "plus_active_count": int(np.count_nonzero(plus_active)),
        "minus_active_count": int(np.count_nonzero(minus_active)),
        "both_active_count": int(np.count_nonzero(both_active)),
        "neither_active_count": int(np.count_nonzero(neither_active)),
        "final_active_count": int(np.count_nonzero(final_active)),
        "face_count": int(len(region_ids)),
        "by_region": by_region,
    }


def nozzle_taper_geometry(spec: SquidReducedSpec) -> tuple[float, float, float]:
    taper_end_z_m = float(spec.chamber_z_min_m)
    taper_length_m = max(0.0, float(spec.nozzle_taper_length_m))
    taper_start_z_m = max(float(spec.downstream_z_m), taper_end_z_m - taper_length_m)
    inlet_radius_m = (
        float(spec.nozzle_taper_inlet_radius_m)
        if spec.nozzle_taper_inlet_radius_m is not None
        else float(spec.chamber_radius_m)
    )
    return (taper_start_z_m, taper_end_z_m, inlet_radius_m)


def nozzle_radius_at_z_m(spec: SquidReducedSpec, z_m: float) -> float:
    base_radius_m = float(spec.nozzle_radius_m)
    if not bool(spec.nozzle_taper_enabled):
        return base_radius_m
    taper_start_z_m, taper_end_z_m, inlet_radius_m = nozzle_taper_geometry(spec)
    if taper_end_z_m <= taper_start_z_m or z_m < taper_start_z_m or z_m >= taper_end_z_m:
        return base_radius_m
    fraction = (float(z_m) - taper_start_z_m) / max(taper_end_z_m - taper_start_z_m, 1.0e-12)
    return base_radius_m + (inlet_radius_m - base_radius_m) * min(max(fraction, 0.0), 1.0)


def reduced_water_geometry_report(spec: SquidReducedSpec) -> dict[str, object]:
    taper_start_z_m, taper_end_z_m, inlet_radius_m = nozzle_taper_geometry(spec)
    mid_z_m = 0.5 * (taper_start_z_m + taper_end_z_m)
    return {
        "nozzle_taper_enabled": bool(spec.nozzle_taper_enabled),
        "nozzle_taper_start_z_m": float(taper_start_z_m),
        "nozzle_taper_end_z_m": float(taper_end_z_m),
        "nozzle_taper_length_m": float(max(0.0, taper_end_z_m - taper_start_z_m)),
        "nozzle_taper_inlet_radius_m": float(inlet_radius_m),
        "nozzle_throat_radius_m": float(spec.nozzle_radius_m),
        "nozzle_radius_at_taper_start_m": float(nozzle_radius_at_z_m(spec, taper_start_z_m)),
        "nozzle_radius_at_taper_mid_m": float(nozzle_radius_at_z_m(spec, mid_z_m)),
        "nozzle_radius_at_taper_end_m": float(inlet_radius_m)
        if bool(spec.nozzle_taper_enabled) and taper_end_z_m > taper_start_z_m
        else float(spec.nozzle_radius_m),
        "outlet_plume_radius_m": float(spec.outlet_plume_radius_m),
        "downstream_farfield_open_enabled": bool(spec.downstream_farfield_open_enabled),
    }


def _reduced_water_mask(points_m: np.ndarray, spec: SquidReducedSpec) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    rx = points[:, 0] - float(spec.monitor_center_x_m)
    ry = points[:, 1] - float(spec.monitor_center_y_m)
    radius = np.sqrt(rx * rx + ry * ry)
    z = points[:, 2]
    nozzle_radius = np.asarray(
        [nozzle_radius_at_z_m(spec, float(value)) for value in z],
        dtype=np.float64,
    )
    chamber = (
        (radius <= float(spec.chamber_radius_m))
        & (z >= float(spec.chamber_z_min_m))
        & (z <= float(spec.chamber_z_max_m))
    )
    nozzle = (
        (radius <= nozzle_radius)
        & (z >= float(spec.downstream_z_m))
        & (z <= float(spec.nozzle_z_max_m))
    )
    outlet_plume = (
        (radius <= float(spec.outlet_plume_radius_m))
        & (z >= float(spec.fluid_bounds_min_m[2]))
        & (z < float(spec.downstream_z_m))
    )
    downstream_farfield = np.zeros_like(z, dtype=bool)
    if bool(spec.downstream_farfield_open_enabled):
        downstream_farfield = z <= float(spec.downstream_farfield_open_z_max_m)
    return chamber | nozzle | outlet_plume | downstream_farfield


def _orient_normals_to_reduced_water(
    centroids_m: np.ndarray,
    normals: np.ndarray,
    region_ids: np.ndarray,
    spec: SquidReducedSpec,
    probe_distance_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    plus_active = _reduced_water_mask(centroids_m + normals * probe_distance_m, spec)
    minus_active = _reduced_water_mask(centroids_m - normals * probe_distance_m, spec)
    flip = (~plus_active) & minus_active
    oriented = np.array(normals, copy=True)
    oriented[flip] *= -1.0
    final_active = plus_active | flip
    both_active = plus_active & minus_active
    neither_active = (~plus_active) & (~minus_active)
    by_region: dict[str, dict[str, int]] = {}
    for region in sorted({int(value) for value in region_ids.tolist()}):
        mask = region_ids == region
        by_region[str(region)] = {
            "face_count": int(np.count_nonzero(mask)),
            "plus_active_count": int(np.count_nonzero(plus_active & mask)),
            "minus_active_count": int(np.count_nonzero(minus_active & mask)),
            "flipped_count": int(np.count_nonzero(flip & mask)),
            "both_active_count": int(np.count_nonzero(both_active & mask)),
            "neither_active_count": int(np.count_nonzero(neither_active & mask)),
            "final_active_count": int(np.count_nonzero(final_active & mask)),
        }
    return oriented, {
        "method": "reduced_water_side_probe_orientation",
        "probe_distance_m": float(probe_distance_m),
        "flipped_count": int(np.count_nonzero(flip)),
        "plus_active_count": int(np.count_nonzero(plus_active)),
        "minus_active_count": int(np.count_nonzero(minus_active)),
        "both_active_count": int(np.count_nonzero(both_active)),
        "neither_active_count": int(np.count_nonzero(neither_active)),
        "final_active_count": int(np.count_nonzero(final_active)),
        "face_count": int(len(region_ids)),
        "by_region": by_region,
    }


def _area_weighted_normal_by_region(
    normals: np.ndarray,
    areas_m2: np.ndarray,
    region_ids: np.ndarray,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for region in sorted({int(value) for value in region_ids.tolist()}):
        mask = region_ids == region
        if not np.any(mask):
            continue
        weighted_normal = np.sum(normals[mask] * areas_m2[mask, None], axis=0)
        norm = float(np.linalg.norm(weighted_normal))
        if norm <= 1.0e-30:
            continue
        result[str(region)] = [
            float(value) for value in (weighted_normal / norm).tolist()
        ]
    return result


def build_tri_surface_diagnostics(
    config: dict[str, object],
    runtime: TaichiRuntimeConfig,
    *,
    spec: SquidReducedSpec | None = None,
    probe_distance_m: float | None = None,
    water_obstacle_mask: np.ndarray | None = None,
    water_grid: CartesianGrid | None = None,
    region_ids: tuple[int, ...] = (7, 8),
    solid_region_ids: tuple[int, ...] = (7, 8, 5),
) -> tuple[
    TriSurfaceRegionDiagnostics,
    dict[str, object],
    SurfaceMesh,
    np.ndarray,
    TriSurfaceRegionDiagnostics,
]:
    import trimesh

    mesh_path = _surface_mesh_path(config)
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh must contain triangular faces")

    def build_region_subset(active_region_ids: tuple[int, ...], label: str):
        selected_face_ids: list[int] = []
        selected_region_ids: list[int] = []
        region_face_counts: dict[str, int] = {}
        for region_id in active_region_ids:
            ids = _face_ids_for_region(config, region_id)
            region_face_counts[str(region_id)] = len(ids)
            selected_face_ids.extend(ids)
            selected_region_ids.extend([region_id] * len(ids))
        if not selected_face_ids:
            raise ValueError(f"no selected {label} region faces found")
        if max(selected_face_ids) >= len(faces) or min(selected_face_ids) < 0:
            raise ValueError(f"selected {label} face IDs are outside the surface mesh")

        selected_faces = faces[np.asarray(selected_face_ids, dtype=np.int64)]
        tri = vertices[selected_faces]
        area_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        doubled_area = np.linalg.norm(area_normals, axis=1)
        valid = doubled_area > 1.0e-20
        if not np.all(valid):
            selected_faces = selected_faces[valid]
            tri = tri[valid]
            area_normals = area_normals[valid]
            doubled_area = doubled_area[valid]
            selected_region_ids = [
                region for region, keep in zip(selected_region_ids, valid.tolist(), strict=True) if keep
            ]
        centroids = np.mean(tri, axis=1)
        areas = 0.5 * doubled_area
        normals = area_normals / doubled_area[:, None]
        region_array = np.asarray(selected_region_ids, dtype=np.int32)
        return selected_faces, centroids, areas, normals, region_array, region_face_counts

    selected_faces, centroids, areas, normals, region_array, region_face_counts = (
        build_region_subset(region_ids, "FSI diagnostic")
    )
    normal_orientation: dict[str, object] = {
        "method": "mesh_face_winding",
        "probe_distance_m": None,
        "flipped_count": 0,
        "face_count": int(len(region_array)),
    }
    if (
        water_obstacle_mask is not None
        and water_grid is not None
        and probe_distance_m is not None
    ):
        normals, normal_orientation = _orient_normals_to_active_water_mask(
            centroids,
            normals,
            region_array,
            grid=water_grid,
            obstacle_mask=water_obstacle_mask,
            probe_distance_m=float(probe_distance_m),
        )
    elif spec is not None and probe_distance_m is not None:
        normals, normal_orientation = _orient_normals_to_reduced_water(
            centroids,
            normals,
            region_array,
            spec,
            float(probe_distance_m),
        )
    solid_faces, solid_centroids, solid_areas, solid_normals, solid_region_array, solid_region_face_counts = (
        build_region_subset(solid_region_ids, "solid MPM")
    )
    solid_normal_orientation: dict[str, object] = {
        "method": "mesh_face_winding",
        "probe_distance_m": None,
        "flipped_count": 0,
        "face_count": int(len(solid_region_array)),
    }
    if (
        water_obstacle_mask is not None
        and water_grid is not None
        and probe_distance_m is not None
    ):
        solid_normals, solid_normal_orientation = _orient_normals_to_active_water_mask(
            solid_centroids,
            solid_normals,
            solid_region_array,
            grid=water_grid,
            obstacle_mask=water_obstacle_mask,
            probe_distance_m=float(probe_distance_m),
        )
    elif spec is not None and probe_distance_m is not None:
        solid_normals, solid_normal_orientation = _orient_normals_to_reduced_water(
            solid_centroids,
            solid_normals,
            solid_region_array,
            spec,
            float(probe_distance_m),
        )
    unique_vertex_ids, inverse_vertex_ids = np.unique(
        solid_faces.reshape(-1),
        return_inverse=True,
    )
    tri_surface_mesh = SurfaceMesh(
        vertices=vertices[unique_vertex_ids],
        faces=inverse_vertex_ids.reshape((-1, 3)).astype(np.int32),
    )

    diagnostics = TriSurfaceRegionDiagnostics(face_capacity=int(len(areas)), runtime=runtime)
    diagnostics.load_faces(
        centroid_m=centroids.astype(np.float32),
        normal=normals.astype(np.float32),
        area_m2=areas.astype(np.float32),
        region_id=region_array,
    )
    # S2-A11c: the solid subset (FSI regions + the fixed rim) gets its own
    # diagnostics object so the layered solid path can bind the rim
    # constraint (fixed_region_id) AND represent the rim as markers whose
    # velocity-Dirichlet rows seal the membrane-edge annulus.
    solid_diagnostics = TriSurfaceRegionDiagnostics(
        face_capacity=int(len(solid_areas)),
        runtime=runtime,
    )
    solid_diagnostics.load_faces(
        centroid_m=solid_centroids.astype(np.float32),
        normal=solid_normals.astype(np.float32),
        area_m2=solid_areas.astype(np.float32),
        region_id=solid_region_array,
    )
    metadata = {
        "mesh_path": str(mesh_path),
        "mesh_scale_to_m": mesh_scale_to_m,
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "diagnostic_face_count": int(len(areas)),
        "region_face_counts": region_face_counts,
        "diagnostic_area_m2_by_region": {
            str(region): float(np.sum(areas[region_array == region])) for region in region_ids
        },
        "diagnostic_area_weighted_normal_by_region": _area_weighted_normal_by_region(
            normals,
            areas,
            region_array,
        ),
        "solid_region_face_counts": solid_region_face_counts,
        "solid_area_m2_by_region": {
            str(region): float(np.sum(solid_areas[solid_region_array == region]))
            for region in solid_region_ids
        },
        "solid_area_weighted_normal_by_region": _area_weighted_normal_by_region(
            solid_normals,
            solid_areas,
            solid_region_array,
        ),
        "solid_surface_vertex_count": int(tri_surface_mesh.vertex_count),
        "solid_surface_face_count": int(tri_surface_mesh.face_count),
        "solid_surface_edge_note": "deduplicated from FSI triangles plus fixed rim triangles for TriMooneyShellMpmState",
        "centroid_bounds_min_m": [float(value) for value in np.min(centroids, axis=0)],
        "centroid_bounds_max_m": [float(value) for value in np.max(centroids, axis=0)],
        "solid_centroid_bounds_min_m": [float(value) for value in np.min(solid_centroids, axis=0)],
        "solid_centroid_bounds_max_m": [float(value) for value in np.max(solid_centroids, axis=0)],
        "normal_orientation": normal_orientation,
        "solid_normal_orientation": solid_normal_orientation,
    }
    return diagnostics, metadata, tri_surface_mesh, solid_region_array, solid_diagnostics


def compute_region_geometry_stats(
    config: dict[str, object],
    region_id: int,
) -> dict[str, object]:
    import trimesh

    face_ids = _face_ids_for_region(config, region_id)
    if not face_ids:
        return {
            "region_id": int(region_id),
            "available": False,
            "face_count": 0,
        }
    mesh_path = _surface_mesh_path(config)
    mesh_scale_to_m = float(config.get("mesh_scale_to_m", 1.0))
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale_to_m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max(face_ids) >= len(faces) or min(face_ids) < 0:
        raise ValueError(f"region {region_id} face IDs are outside the surface mesh")
    selected_faces = faces[np.asarray(face_ids, dtype=np.int64)]
    tri = vertices[selected_faces]
    area_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    doubled_area = np.linalg.norm(area_normals, axis=1)
    valid = doubled_area > 1.0e-20
    tri = tri[valid]
    area_normals = area_normals[valid]
    doubled_area = doubled_area[valid]
    if tri.size == 0:
        return {
            "region_id": int(region_id),
            "available": False,
            "face_count": int(len(face_ids)),
            "valid_face_count": 0,
        }
    areas = 0.5 * doubled_area
    centroids = np.mean(tri, axis=1)
    vertices_flat = tri.reshape((-1, 3))
    area_total = float(np.sum(areas))
    if area_total > 0.0:
        area_weighted_centroid = np.sum(centroids * areas[:, None], axis=0) / area_total
    else:
        area_weighted_centroid = np.mean(centroids, axis=0)
    xy_center = area_weighted_centroid[:2]
    vertex_radius = np.linalg.norm(vertices_flat[:, :2] - xy_center[None, :], axis=1)
    centroid_radius = np.linalg.norm(centroids[:, :2] - xy_center[None, :], axis=1)
    normals = area_normals / doubled_area[:, None]
    area_weighted_normal = np.sum(normals * areas[:, None], axis=0)
    normal_norm = float(np.linalg.norm(area_weighted_normal))
    if normal_norm > 0.0:
        area_weighted_normal = area_weighted_normal / normal_norm
    return {
        "region_id": int(region_id),
        "available": True,
        "mesh_path": str(mesh_path),
        "mesh_scale_to_m": mesh_scale_to_m,
        "face_count": int(len(face_ids)),
        "valid_face_count": int(len(areas)),
        "area_m2": area_total,
        "area_weighted_centroid_m": [float(value) for value in area_weighted_centroid],
        "vertex_bounds_min_m": [float(value) for value in np.min(vertices_flat, axis=0)],
        "vertex_bounds_max_m": [float(value) for value in np.max(vertices_flat, axis=0)],
        "centroid_bounds_min_m": [float(value) for value in np.min(centroids, axis=0)],
        "centroid_bounds_max_m": [float(value) for value in np.max(centroids, axis=0)],
        "vertex_radius_min_m": float(np.min(vertex_radius)),
        "vertex_radius_mean_m": float(np.mean(vertex_radius)),
        "vertex_radius_p95_m": float(np.percentile(vertex_radius, 95.0)),
        "vertex_radius_max_m": float(np.max(vertex_radius)),
        "centroid_radius_max_m": float(np.max(centroid_radius)),
        "area_weighted_normal": [float(value) for value in area_weighted_normal],
    }


def spec_with_region14_aperture(
    spec: SquidReducedSpec,
    aperture_stats: dict[str, object],
    *,
    open_downstream_farfield: bool = False,
) -> SquidReducedSpec:
    if not bool(aperture_stats.get("available", False)):
        return spec
    center = aperture_stats.get("area_weighted_centroid_m", [])
    radius = float(aperture_stats.get("vertex_radius_p95_m", spec.nozzle_radius_m))
    if not isinstance(center, list | tuple) or len(center) < 2 or radius <= 0.0:
        return spec
    aperture_z = float(center[2]) if len(center) >= 3 else float(spec.lip_z_m)
    return replace(
        spec,
        monitor_center_x_m=float(center[0]),
        monitor_center_y_m=float(center[1]),
        nozzle_radius_m=radius,
        outlet_plume_radius_m=radius,
        monitor_radius_m=radius,
        downstream_farfield_open_enabled=bool(open_downstream_farfield),
        downstream_farfield_open_z_max_m=aperture_z,
    )


def spec_with_nozzle_taper(
    spec: SquidReducedSpec,
    *,
    taper_length_m: float | None = None,
    inlet_radius_m: float | None = None,
) -> SquidReducedSpec:
    length_m = (
        min(
            float(spec.nozzle_length_m),
            max(float(spec.chamber_z_min_m) - float(spec.downstream_z_m), 0.0),
        )
        if taper_length_m is None
        else float(taper_length_m)
    )
    if length_m <= 0.0:
        raise ValueError("nozzle taper length must be positive")
    inlet_radius = (
        float(spec.chamber_radius_m)
        if inlet_radius_m is None
        else float(inlet_radius_m)
    )
    if inlet_radius <= float(spec.nozzle_radius_m):
        raise ValueError("nozzle taper inlet radius must exceed the throat radius")
    return replace(
        spec,
        nozzle_taper_enabled=True,
        nozzle_taper_length_m=length_m,
        nozzle_taper_inlet_radius_m=inlet_radius,
    )


def spec_with_nozzle_graded_grid(
    spec: SquidReducedSpec,
    *,
    target_spacing_m: float | None = None,
    farfield_spacing_m: float = 3.0e-3,
    max_growth_ratio: float = 1.2,
    max_cells: int | None = None,
    extra_refinement_regions: Sequence[RefinementRegion] = (),
) -> SquidReducedSpec:
    target_spacing = (
        float(target_spacing_m)
        if target_spacing_m is not None
        else float(spec.nozzle_radius_m) / 5.0
    )
    if target_spacing <= 0.0:
        raise ValueError("graded grid target spacing must be positive")
    if farfield_spacing_m <= 0.0:
        raise ValueError("graded grid farfield spacing must be positive")
    if max_growth_ratio <= 1.0:
        raise ValueError("graded grid max growth ratio must be greater than 1")

    taper_start_z_m, taper_end_z_m, taper_inlet_radius_m = nozzle_taper_geometry(spec)
    radius_m = (
        max(float(spec.nozzle_radius_m), float(taper_inlet_radius_m))
        if bool(spec.nozzle_taper_enabled) and taper_end_z_m > taper_start_z_m
        else float(spec.nozzle_radius_m)
    )
    bounds_min = spec.fluid_bounds_min_m
    bounds_max = spec.fluid_bounds_max_m
    region_bounds_min = (
        max(float(bounds_min[0]), float(spec.monitor_center_x_m) - radius_m),
        max(float(bounds_min[1]), float(spec.monitor_center_y_m) - radius_m),
        max(float(bounds_min[2]), min(float(spec.downstream_z_m), float(spec.nozzle_z_max_m))),
    )
    region_bounds_max = (
        min(float(bounds_max[0]), float(spec.monitor_center_x_m) + radius_m),
        min(float(bounds_max[1]), float(spec.monitor_center_y_m) + radius_m),
        min(float(bounds_max[2]), max(float(spec.downstream_z_m), float(spec.nozzle_z_max_m))),
    )
    if any(hi <= lo for lo, hi in zip(region_bounds_min, region_bounds_max, strict=True)):
        raise ValueError("graded nozzle refinement region does not overlap the fluid domain")
    refinement_regions = (
        RefinementRegion(
            bounds_min_m=region_bounds_min,
            bounds_max_m=region_bounds_max,
            target_spacing_m=target_spacing,
        ),
    ) + tuple(extra_refinement_regions)

    graded_grid = GradedGridSpec(
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        farfield_spacing_m=float(farfield_spacing_m),
        max_growth_ratio=float(max_growth_ratio),
        max_cells=max_cells,
        refinement_regions=refinement_regions,
    )
    grid = build_graded_grid(graded_grid)
    return replace(
        spec,
        grid_nodes=grid.grid_nodes,
        cartesian_grid=None,
        graded_grid=graded_grid,
    )


def tail_refinement_region_from_geometry(
    spec: SquidReducedSpec,
    tail_geometry: dict[str, object],
    *,
    target_spacing_m: float,
    padding_m: float,
) -> RefinementRegion | None:
    if not bool(tail_geometry.get("available", False)):
        return None
    target_spacing = float(target_spacing_m)
    if target_spacing <= 0.0:
        raise ValueError("tail refinement target spacing must be positive")
    padding = float(padding_m)
    if padding < 0.0:
        raise ValueError("tail refinement padding must be non-negative")
    raw_min = required_tuple3(
        tail_geometry.get("vertex_bounds_min_m"),
        field="tail refinement vertex_bounds_min_m",
    )
    raw_max = required_tuple3(
        tail_geometry.get("vertex_bounds_max_m"),
        field="tail refinement vertex_bounds_max_m",
    )
    bounds_min = tuple(
        max(float(domain_min), float(raw_value) - padding)
        for domain_min, raw_value in zip(spec.fluid_bounds_min_m, raw_min, strict=True)
    )
    bounds_max = tuple(
        min(float(domain_max), float(raw_value) + padding)
        for domain_max, raw_value in zip(spec.fluid_bounds_max_m, raw_max, strict=True)
    )
    if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("tail refinement region does not overlap the fluid domain")
    return RefinementRegion(
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        target_spacing_m=target_spacing,
    )


def refinement_region_summary(region: RefinementRegion | None) -> dict[str, object] | None:
    if region is None:
        return None
    return {
        "bounds_min_m": [float(value) for value in region.bounds_min_m],
        "bounds_max_m": [float(value) for value in region.bounds_max_m],
        "target_spacing_m": [float(value) for value in region.target_spacing_m],
    }


def cartesian_grid_for_spec(spec: SquidReducedSpec) -> CartesianGrid:
    if spec.graded_grid is not None:
        return build_graded_grid(spec.graded_grid)
    if spec.cartesian_grid is not None:
        return spec.cartesian_grid
    return CartesianGrid.uniform(
        bounds_min_m=spec.fluid_bounds_min_m,
        bounds_max_m=spec.fluid_bounds_max_m,
        grid_nodes=spec.grid_nodes,
    )


def cartesian_grid_axis_min_spacing_m(grid: CartesianGrid) -> tuple[float, float, float]:
    return (
        float(min(grid.cell_widths_x_m)),
        float(min(grid.cell_widths_y_m)),
        float(min(grid.cell_widths_z_m)),
    )


def cartesian_grid_axis_max_spacing_m(grid: CartesianGrid) -> tuple[float, float, float]:
    return (
        float(max(grid.cell_widths_x_m)),
        float(max(grid.cell_widths_y_m)),
        float(max(grid.cell_widths_z_m)),
    )


def solid_mpm_bounds_padding_distance_m(
    *,
    fluid_grid_axis_max_spacing_m: Sequence[float],
    estimated_solid_particle_spacing_m: float,
) -> float:
    fluid_axis_spacing = tuple(float(value) for value in fluid_grid_axis_max_spacing_m)
    if len(fluid_axis_spacing) != 3:
        raise ValueError("fluid_grid_axis_max_spacing_m must contain exactly 3 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in fluid_axis_spacing):
        raise ValueError("fluid_grid_axis_max_spacing_m entries must be finite and positive")
    solid_spacing = float(estimated_solid_particle_spacing_m)
    if not math.isfinite(solid_spacing) or solid_spacing <= 0.0:
        raise ValueError("estimated_solid_particle_spacing_m must be finite and positive")
    return 3.0 * max(max(fluid_axis_spacing), solid_spacing)


def cartesian_grid_uniform_spacing_m(grid: CartesianGrid) -> tuple[float, float, float] | None:
    if not grid.is_uniform:
        return None
    return tuple(float(value) for value in grid.uniform_spacing_m)


def _count_axis_centers_in_bounds(
    centers: Sequence[float],
    lower: float,
    upper: float,
) -> int:
    return sum(1 for value in centers if lower <= float(value) <= upper)


def _axis_width_range_in_bounds(
    centers: Sequence[float],
    widths: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    selected = [
        float(width)
        for center, width in zip(centers, widths, strict=True)
        if lower <= float(center) <= upper
    ]
    if not selected:
        return None
    return (min(selected), max(selected))


def _max_adjacent_spacing_ratio(widths: Sequence[float]) -> float:
    ratios = []
    for left, right in zip(widths, widths[1:], strict=False):
        left_value = float(left)
        right_value = float(right)
        if left_value > 0.0 and right_value > 0.0:
            ratios.append(max(left_value / right_value, right_value / left_value))
    return max(ratios, default=1.0)


def fluid_grid_resolution_report(spec: SquidReducedSpec) -> dict[str, object]:
    grid = cartesian_grid_for_spec(spec)
    radius_m = float(spec.nozzle_radius_m)
    nozzle_bounds_min = (
        float(spec.monitor_center_x_m) - radius_m,
        float(spec.monitor_center_y_m) - radius_m,
        min(float(spec.downstream_z_m), float(spec.nozzle_z_max_m)),
    )
    nozzle_bounds_max = (
        float(spec.monitor_center_x_m) + radius_m,
        float(spec.monitor_center_y_m) + radius_m,
        max(float(spec.downstream_z_m), float(spec.nozzle_z_max_m)),
    )
    axes = (
        (grid.cell_centers_x_m, grid.cell_widths_x_m, nozzle_bounds_min[0], nozzle_bounds_max[0]),
        (grid.cell_centers_y_m, grid.cell_widths_y_m, nozzle_bounds_min[1], nozzle_bounds_max[1]),
        (grid.cell_centers_z_m, grid.cell_widths_z_m, nozzle_bounds_min[2], nozzle_bounds_max[2]),
    )
    nozzle_cells = tuple(
        _count_axis_centers_in_bounds(centers, lower, upper)
        for centers, _widths, lower, upper in axes
    )
    nozzle_width_ranges = tuple(
        _axis_width_range_in_bounds(centers, widths, lower, upper)
        for centers, widths, lower, upper in axes
    )
    min_widths = tuple(
        None if width_range is None else width_range[0]
        for width_range in nozzle_width_ranges
    )
    max_widths = tuple(
        None if width_range is None else width_range[1]
        for width_range in nozzle_width_ranges
    )
    target_spacing = None
    if spec.graded_grid is not None and spec.graded_grid.refinement_regions:
        target_spacing = spec.graded_grid.refinement_regions[0].target_spacing_m
    return {
        "graded_enabled": spec.graded_grid is not None,
        "grid_nodes": [int(value) for value in grid.grid_nodes],
        "bounds_min_m": [float(value) for value in grid.bounds_min_m],
        "bounds_max_m": [float(value) for value in grid.bounds_max_m],
        "nozzle_bounds_min_m": [float(value) for value in nozzle_bounds_min],
        "nozzle_bounds_max_m": [float(value) for value in nozzle_bounds_max],
        "nozzle_cells_x": int(nozzle_cells[0]),
        "nozzle_cells_y": int(nozzle_cells[1]),
        "nozzle_cells_z": int(nozzle_cells[2]),
        "nozzle_diameter_cells_min": int(min(nozzle_cells[0], nozzle_cells[1])),
        "nozzle_resolves_diameter_10_cells": min(nozzle_cells[0], nozzle_cells[1]) >= 10,
        "nozzle_min_cell_width_m": [
            None if value is None else float(value) for value in min_widths
        ],
        "nozzle_max_cell_width_m": [
            None if value is None else float(value) for value in max_widths
        ],
        "global_min_cell_width_m": [
            float(value) for value in cartesian_grid_axis_min_spacing_m(grid)
        ],
        "global_max_cell_width_m": [
            float(value) for value in cartesian_grid_axis_max_spacing_m(grid)
        ],
        "max_adjacent_spacing_ratio": [
            float(_max_adjacent_spacing_ratio(grid.cell_widths_x_m)),
            float(_max_adjacent_spacing_ratio(grid.cell_widths_y_m)),
            float(_max_adjacent_spacing_ratio(grid.cell_widths_z_m)),
        ],
        "graded_grid_target_spacing_m": (
            None if target_spacing is None else [float(value) for value in target_spacing]
        ),
    }


def solid_mpm_bounds_from_surface_metadata(
    metadata: Mapping[str, object],
    *,
    fallback_bounds_min_m: Sequence[float],
    fallback_bounds_max_m: Sequence[float],
    padding_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fallback_min = required_tuple3(
        fallback_bounds_min_m,
        field="fallback_bounds_min_m",
    )
    fallback_max = required_tuple3(
        fallback_bounds_max_m,
        field="fallback_bounds_max_m",
    )
    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_m must be a finite non-negative number")
    surface_min = required_tuple3(
        metadata.get("solid_centroid_bounds_min_m", fallback_min),
        field="metadata.solid_centroid_bounds_min_m",
    )
    surface_max = required_tuple3(
        metadata.get("solid_centroid_bounds_max_m", fallback_max),
        field="metadata.solid_centroid_bounds_max_m",
    )
    bounds_min = tuple(
        min(domain_min, solid_min - padding)
        for domain_min, solid_min in zip(fallback_min, surface_min, strict=True)
    )
    bounds_max = tuple(
        max(domain_max, solid_max + padding)
        for domain_max, solid_max in zip(fallback_max, surface_max, strict=True)
    )
    if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("solid MPM bounds must have positive extent")
    return bounds_min, bounds_max


def resolve_pressure_solver(
    pressure_solver: str,
    *,
    graded_grid_enabled: bool,
    fsi_coupling_mode: str | None = None,
) -> str:
    solver_name = str(pressure_solver)
    if solver_name == "auto":
        if str(fsi_coupling_mode) == FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
            return "fv_cg"
        return "fv_cg" if graded_grid_enabled else "fv_multigrid"
    if solver_name not in PRESSURE_SOLVER_CHOICES:
        raise ValueError(f"unsupported pressure solver: {pressure_solver!r}")
    if graded_grid_enabled and solver_name not in {"fv_jacobi", "fv_multigrid", "fv_cg"}:
        raise ValueError("--use-graded-grid requires an FV pressure solver")
    return solver_name


def effective_fluid_substeps_for_grid(
    spec: SquidReducedSpec,
    requested_substeps: int,
    *,
    grid: CartesianGrid | None = None,
) -> int:
    requested = max(1, int(requested_substeps))
    if spec.graded_grid is None:
        return requested
    grid_for_spacing = cartesian_grid_for_spec(spec) if grid is None else grid
    min_spacing_m = min(cartesian_grid_axis_min_spacing_m(grid_for_spacing))
    farfield_spacing_m = max(float(value) for value in spec.graded_grid.farfield_spacing_m)
    # Resolve the finest graded cells at a half-farfield CFL; the full-step
    # ratio was not enough for the projected-IBM divergence guard.
    fine_cell_spacing_ratio = int(math.ceil(farfield_spacing_m / max(min_spacing_m, 1.0e-12)))
    reference_dt_s = float(spec.base_dt_s) if spec.base_dt_s is not None else float(spec.dt_s)
    dt_scale = float(spec.dt_s) / max(reference_dt_s, 1.0e-12)
    graded_substeps = max(1, int(math.ceil(2 * fine_cell_spacing_ratio * dt_scale)))
    return max(requested, graded_substeps)


def pressure_projection_budget_report(
    *,
    fluid_substeps: int,
    ibm_correction_iterations: int,
    fsi_coupling_iterations: int,
    projection_iterations: int,
    fsi_coupling_enabled: bool,
) -> dict[str, object]:
    substeps = max(1, int(fluid_substeps))
    correction_iterations = max(1, int(ibm_correction_iterations))
    coupling_iterations = max(1, int(fsi_coupling_iterations))
    pressure_iterations = max(1, int(projection_iterations))
    trial_evaluations = coupling_iterations if bool(fsi_coupling_enabled) else 0
    accepted_evaluations = 1
    project_calls_per_fluid_evaluation = substeps * correction_iterations
    trial_project_calls = trial_evaluations * project_calls_per_fluid_evaluation
    accepted_project_calls = accepted_evaluations * project_calls_per_fluid_evaluation
    total_project_calls = trial_project_calls + accepted_project_calls
    return {
        "fluid_substeps": substeps,
        "ibm_correction_iterations": correction_iterations,
        "fsi_coupling_enabled": bool(fsi_coupling_enabled),
        "fsi_coupling_trial_evaluations_per_physical_step_max": trial_evaluations,
        "accepted_fluid_step_evaluations_per_physical_step": accepted_evaluations,
        "fluid_step_evaluations_per_physical_step_max": (
            trial_evaluations + accepted_evaluations
        ),
        "pressure_project_calls_per_fluid_evaluation": project_calls_per_fluid_evaluation,
        "trial_pressure_project_calls_per_step_max": trial_project_calls,
        "full_report_pressure_project_calls_per_step": accepted_project_calls,
        "pressure_project_calls_per_physical_step_max": total_project_calls,
        "projection_iterations_per_project_call_budget": pressure_iterations,
        "cg_iteration_budget_per_physical_step_max": (
            total_project_calls * pressure_iterations
        ),
        "note": (
            "Budget only: this reports the current algorithmic projection-count upper "
            "bound and does not change pressure, velocity, flow, IBM force, or FSI "
            "coupling physics."
        ),
    }


def resolve_divergence_cleanup_iterations(
    value: int,
    *,
    graded_grid_enabled: bool,
    value_was_explicit: bool = True,
) -> int:
    iterations = max(0, int(value))
    if graded_grid_enabled and iterations > 0:
        if not bool(value_was_explicit):
            return 0
        raise ValueError(
            "--use-graded-grid requires --divergence-cleanup-iterations 0 until "
            "non-uniform cleanup operators are implemented"
        )
    return iterations


def reduced_active_water_connectivity(
    spec: SquidReducedSpec,
    obstacle_cell_count: int,
    obstacle_mask: np.ndarray | None = None,
) -> dict[str, object]:
    total_cells = int(spec.grid_nodes[0] * spec.grid_nodes[1] * spec.grid_nodes[2])
    active_cells = total_cells - int(obstacle_cell_count)
    if obstacle_mask is None:
        return {
            "method": "latest_core_reduced_chamber_nozzle_obstacle_seeded_from_z_min_analytic_fallback",
            "component_count": 1,
            "seed_boundary": "z_min",
            "active_cell_count": active_cells,
            "inactive_cell_count": int(obstacle_cell_count),
            "z_min_connected_active_cell_count": active_cells,
            "trapped_active_cell_count": 0,
            "connectivity_passed": active_cells > 0,
            "limitation": "No obstacle mask was supplied, so connectivity fell back to the legacy analytic count.",
        }
    mask = np.asarray(obstacle_mask, dtype=np.int32)
    if mask.shape != tuple(spec.grid_nodes):
        raise ValueError(
            f"obstacle_mask shape {mask.shape!r} does not match grid_nodes {tuple(spec.grid_nodes)!r}"
        )
    active = mask == 0
    active_cells = int(np.count_nonzero(active))
    inactive_cells = int(mask.size - active_cells)
    visited = np.zeros(active.shape, dtype=bool)
    component_count = 0
    z_min_connected_active_cells = 0
    trapped_active_cells = 0
    nx, ny, nz = active.shape
    for seed_index in zip(*np.nonzero(active), strict=False):
        if visited[seed_index]:
            continue
        component_count += 1
        stack = [tuple(int(value) for value in seed_index)]
        visited[seed_index] = True
        component_size = 0
        touches_z_min = False
        while stack:
            i, j, k = stack.pop()
            component_size += 1
            touches_z_min = touches_z_min or k == 0
            for ni, nj, nk in (
                (i - 1, j, k),
                (i + 1, j, k),
                (i, j - 1, k),
                (i, j + 1, k),
                (i, j, k - 1),
                (i, j, k + 1),
            ):
                if (
                    0 <= ni < nx
                    and 0 <= nj < ny
                    and 0 <= nk < nz
                    and active[ni, nj, nk]
                    and not visited[ni, nj, nk]
                ):
                    visited[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
        if touches_z_min:
            z_min_connected_active_cells += component_size
        else:
            trapped_active_cells += component_size
    return {
        "method": "latest_core_obstacle_flood_fill_from_z_min",
        "component_count": component_count,
        "seed_boundary": "z_min",
        "active_cell_count": active_cells,
        "inactive_cell_count": inactive_cells,
        "z_min_connected_active_cell_count": z_min_connected_active_cells,
        "trapped_active_cell_count": trapped_active_cells,
        "connectivity_passed": active_cells > 0 and trapped_active_cells == 0,
    }
