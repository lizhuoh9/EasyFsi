from dataclasses import dataclass

import numpy as np
import taichi as ti

from .geometry import SurfaceMesh, UvSphereResolution
from .runtime import TaichiRuntimeConfig, init_taichi


@dataclass(frozen=True)
class UvMooneyShellMpmReport:
    particle_count: int
    edge_count: int
    active_grid_nodes: int
    grid_out_of_bounds_particle_count: int
    mean_radial_stretch: float
    max_radial_stretch_error: float
    max_edge_strain: float
    max_speed_mps: float
    total_mass_kg: float
    internal_force_rms_n: float
    net_internal_force_relative_error: float
    transfer_relative_error: float


@dataclass(frozen=True)
class TriMooneyShellMpmReport:
    particle_count: int
    face_count: int
    edge_count: int
    active_grid_nodes: int
    grid_out_of_bounds_particle_count: int
    particle_spacing_m: float
    grid_spacing_m: tuple[float, float, float]
    mean_radial_stretch: float
    max_radial_stretch_error: float
    max_edge_strain: float
    max_speed_mps: float
    total_mass_kg: float
    total_area_m2: float
    primary_mean_displacement_m: tuple[float, float, float]
    primary_mean_velocity_mps: tuple[float, float, float]
    secondary_mean_displacement_m: tuple[float, float, float]
    secondary_mean_velocity_mps: tuple[float, float, float]
    particle_momentum_kg_mps: tuple[float, float, float]
    grid_momentum_kg_mps: tuple[float, float, float]
    total_force_n: tuple[float, float, float]
    internal_force_rms_n: float
    net_internal_force_relative_error: float
    transfer_relative_error: float
    primary_particle_count: int = 0
    secondary_particle_count: int = 0


def _unique_undirected_edges(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for ia, ib, ic in np.asarray(faces, dtype=np.int32):
        for a, b in ((ia, ib), (ib, ic), (ic, ia)):
            edge = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            edges.add(edge)
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edges), dtype=np.int32)


def _mesh_bounds_with_padding(
    vertices: np.ndarray,
    bounds_padding_fraction: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    extent = bounds_max - bounds_min
    max_extent = float(np.max(extent))
    if max_extent <= 0.0:
        max_extent = 1.0
    padding = max(bounds_padding_fraction * max_extent, 1.0e-6 * max_extent)
    padded_min = bounds_min - padding
    padded_max = bounds_max + padding
    return tuple(float(v) for v in padded_min), tuple(float(v) for v in padded_max)


def _validate_flip_blend(flip_blend: float) -> float:
    value = float(flip_blend)
    if not 0.0 <= value <= 1.0:
        raise ValueError("flip_blend must be in [0, 1]")
    return value


def _vector3(value: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    return (float(value[0]), float(value[1]), float(value[2]))


def _raise_if_all_particles_out_of_bounds(
    particle_count: int,
    grid_out_of_bounds_particle_count: int,
) -> None:
    if particle_count > 0 and grid_out_of_bounds_particle_count == particle_count:
        raise RuntimeError(
            f"all {particle_count} MPM particles are outside the background grid; "
            "the solid has left the simulation domain"
        )


def _raise_if_out_of_bounds_exceeds_tolerance(
    particle_count: int,
    grid_out_of_bounds_particle_count: int,
    tolerance: int,
) -> None:
    if grid_out_of_bounds_particle_count > int(tolerance):
        raise RuntimeError(
            f"{grid_out_of_bounds_particle_count} of {particle_count} MPM particles "
            "are outside the background grid; refusing to advance a partial solid"
        )


def _raise_if_required_shell_region_empty(
    *,
    primary_count: int,
    secondary_count: int,
) -> None:
    if primary_count <= 0:
        raise RuntimeError("primary shell region has no in-grid MPM particles")
    if secondary_count <= 0:
        raise RuntimeError("secondary shell region has no in-grid MPM particles")


@ti.data_oriented
class TriMooneyShellMpmState:
    """Paper-style Mooney membrane MPM on an explicit triangle surface mesh."""

    def __init__(
        self,
        mesh: SurfaceMesh,
        *,
        thickness_m: float,
        density_kgm3: float,
        c1_pa: float,
        c2_pa: float,
        membrane_force_scale: float = 1.0,
        grid_nodes: tuple[int, int, int] = (36, 36, 36),
        bounds_padding_fraction: float = 0.5,
        face_region_id: np.ndarray | None = None,
        primary_region_id: int,
        secondary_region_id: int,
        fixed_region_id: int = -1,
        primary_thickness_m: float | None = None,
        secondary_thickness_m: float | None = None,
        out_of_bounds_particle_tolerance: int = 0,
        require_nonempty_region_counts: bool | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ):
        init_taichi(runtime)
        if mesh.vertex_count <= 0:
            raise ValueError("mesh must contain at least one vertex")
        if mesh.face_count <= 0:
            raise ValueError("mesh must contain at least one triangle")
        if thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if density_kgm3 <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        if c1_pa <= 0.0 or c2_pa < 0.0:
            raise ValueError("Mooney constants must be non-negative with c1 > 0")
        if membrane_force_scale <= 0.0:
            raise ValueError("membrane_force_scale must be positive")
        if min(grid_nodes) < 4:
            raise ValueError("grid_nodes must be at least 4 in each direction")
        if bounds_padding_fraction <= 0.0:
            raise ValueError("bounds_padding_fraction must be positive")
        if int(out_of_bounds_particle_tolerance) < 0:
            raise ValueError("out_of_bounds_particle_tolerance must be non-negative")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        edges = _unique_undirected_edges(faces)
        regions = np.zeros(mesh.face_count, dtype=np.int32)
        if face_region_id is not None:
            regions = np.asarray(face_region_id, dtype=np.int32)
            if regions.shape != (mesh.face_count,):
                raise ValueError("face_region_id must have shape (face_count,)")
        region_counts_required = (
            bool(
                np.any(regions == int(primary_region_id))
                or np.any(regions == int(secondary_region_id))
            )
            if require_nonempty_region_counts is None
            else bool(require_nonempty_region_counts)
        )
        bounds_min, bounds_max = _mesh_bounds_with_padding(vertices, bounds_padding_fraction)

        self.particle_count = int(mesh.vertex_count)
        self.face_count = int(mesh.face_count)
        self.edge_count = int(edges.shape[0])
        self.thickness_m = float(thickness_m)
        self.density_kgm3 = float(density_kgm3)
        self.c1_pa = float(c1_pa)
        self.c2_pa = float(c2_pa)
        self.membrane_force_scale = float(membrane_force_scale)
        self.primary_region_id = int(primary_region_id)
        self.secondary_region_id = int(secondary_region_id)
        self.fixed_region_id = int(fixed_region_id)
        self.out_of_bounds_particle_tolerance = int(out_of_bounds_particle_tolerance)
        self.require_nonempty_region_counts = region_counts_required
        self.primary_thickness_m = float(thickness_m if primary_thickness_m is None else primary_thickness_m)
        self.secondary_thickness_m = float(thickness_m if secondary_thickness_m is None else secondary_thickness_m)
        self.grid_nodes = tuple(int(value) for value in grid_nodes)
        self.nx, self.ny, self.nz = self.grid_nodes
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.dx = (
            (self.bounds_max[0] - self.bounds_min[0]) / (self.nx - 1),
            (self.bounds_max[1] - self.bounds_min[1]) / (self.ny - 1),
            (self.bounds_max[2] - self.bounds_min[2]) / (self.nz - 1),
        )
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.rest_x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.saved_x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.saved_u = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.saved_v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.mass_kg = ti.field(dtype=ti.f32, shape=self.particle_count)
        self.area_weight_m2 = ti.field(dtype=ti.f32, shape=self.particle_count)
        self.surface_normal = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.internal_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.external_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.vertex_region_id = ti.field(dtype=ti.i32, shape=self.particle_count)
        self.fixed_particle = ti.field(dtype=ti.i32, shape=self.particle_count)
        self.rest_center_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.face_indices = ti.Vector.field(3, dtype=ti.i32, shape=self.face_count)
        self.face_region_id = ti.field(dtype=ti.i32, shape=self.face_count)
        self.edge_indices = ti.Vector.field(2, dtype=ti.i32, shape=max(self.edge_count, 1))

        self.grid_mass_kg = ti.field(dtype=ti.f32, shape=self.grid_nodes)
        self.grid_velocity_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)
        self.grid_velocity_before_force_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)
        self.grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)

        self.report_active_grid_nodes = ti.field(dtype=ti.i32, shape=())
        self.report_grid_out_of_bounds_particle_count = ti.field(dtype=ti.i32, shape=())
        self.report_total_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.report_total_area_m2 = ti.field(dtype=ti.f32, shape=())
        self.report_radial_stretch_sum = ti.field(dtype=ti.f32, shape=())
        self.report_radial_stretch_count = ti.field(dtype=ti.i32, shape=())
        self.report_max_radial_stretch_error = ti.field(dtype=ti.f32, shape=())
        self.report_max_edge_strain = ti.field(dtype=ti.f32, shape=())
        self.report_max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.report_internal_force_sum_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_internal_force_square_sum_n2 = ti.field(dtype=ti.f32, shape=())
        self.report_particle_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_transfer_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_total_force_n = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_current_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_rest_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_center_count = ti.field(dtype=ti.i32, shape=())
        self.report_momentum_square_sum = ti.field(dtype=ti.f64, shape=())
        self.report_force_impulse_square_sum = ti.field(dtype=ti.f64, shape=())
        self.report_transfer_relative_error = ti.field(dtype=ti.f64, shape=())
        self.report_primary_displacement_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_velocity_sum_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_displacement_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_velocity_sum_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_count = ti.field(dtype=ti.i32, shape=())
        self.report_secondary_count = ti.field(dtype=ti.i32, shape=())
        self.report_float_snapshot = ti.Vector.field(32, dtype=ti.f64, shape=())
        self.report_count_snapshot = ti.Vector.field(5, dtype=ti.i32, shape=())
        self.report_host_snapshot = ti.field(dtype=ti.f64, shape=37)
        self.primary_area_m2 = ti.field(dtype=ti.f32, shape=())
        self.secondary_area_m2 = ti.field(dtype=ti.f32, shape=())
        self.primary_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.secondary_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.last_report_host_reads = 0

        self.x.from_numpy(vertices)
        self.rest_x.from_numpy(vertices)
        self.saved_x.from_numpy(vertices)
        self.face_indices.from_numpy(faces)
        self.face_region_id.from_numpy(regions)
        if self.edge_count > 0:
            self.edge_indices.from_numpy(edges)
        self._init_mass_kernel(
            float(thickness_m),
            float(density_kgm3),
            int(self.primary_region_id),
            int(self.secondary_region_id),
            int(self.fixed_region_id),
            float(self.primary_thickness_m),
            float(self.secondary_thickness_m),
        )
        self._update_rest_center()

    @ti.kernel
    def _update_rest_center_kernel(self):
        self.rest_center_m[None] = ti.Vector([0.0, 0.0, 0.0])
        for p in range(self.particle_count):
            self._atomic_add_vector(self.rest_center_m, self.rest_x[p])

    @ti.kernel
    def _normalize_rest_center_kernel(self):
        self.rest_center_m[None] = self.rest_center_m[None] / ti.cast(self.particle_count, ti.f32)

    def _update_rest_center(self) -> None:
        self._update_rest_center_kernel()
        self._normalize_rest_center_kernel()

    @ti.func
    def _clear_reports(self):
        self.report_active_grid_nodes[None] = 0
        self.report_grid_out_of_bounds_particle_count[None] = 0
        self.report_total_mass_kg[None] = 0.0
        self.report_total_area_m2[None] = 0.0
        self.report_radial_stretch_sum[None] = 0.0
        self.report_radial_stretch_count[None] = 0
        self.report_max_radial_stretch_error[None] = 0.0
        self.report_max_edge_strain[None] = 0.0
        self.report_max_speed_mps[None] = 0.0
        self.report_internal_force_sum_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_internal_force_square_sum_n2[None] = 0.0
        self.report_particle_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_transfer_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_total_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_current_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_rest_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_center_count[None] = 0
        self.report_momentum_square_sum[None] = 0.0
        self.report_force_impulse_square_sum[None] = 0.0
        self.report_transfer_relative_error[None] = 0.0
        self.report_primary_displacement_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_velocity_sum_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_displacement_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_velocity_sum_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_count[None] = 0
        self.report_secondary_count[None] = 0
        self.report_float_snapshot[None] = ti.Vector(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.report_count_snapshot[None] = ti.Vector([0, 0, 0, 0, 0])

    @ti.func
    def _pack_report_snapshot(self):
        self.report_float_snapshot[None] = ti.Vector(
            [
                self.report_internal_force_square_sum_n2[None],
                self.report_internal_force_sum_n[None].x,
                self.report_internal_force_sum_n[None].y,
                self.report_internal_force_sum_n[None].z,
                self.report_total_area_m2[None],
                self.report_radial_stretch_sum[None],
                self.report_max_radial_stretch_error[None],
                self.report_max_edge_strain[None],
                self.report_max_speed_mps[None],
                self.report_total_mass_kg[None],
                self.report_primary_displacement_sum_m[None].x,
                self.report_primary_displacement_sum_m[None].y,
                self.report_primary_displacement_sum_m[None].z,
                self.report_primary_velocity_sum_mps[None].x,
                self.report_primary_velocity_sum_mps[None].y,
                self.report_primary_velocity_sum_mps[None].z,
                self.report_secondary_displacement_sum_m[None].x,
                self.report_secondary_displacement_sum_m[None].y,
                self.report_secondary_displacement_sum_m[None].z,
                self.report_secondary_velocity_sum_mps[None].x,
                self.report_secondary_velocity_sum_mps[None].y,
                self.report_secondary_velocity_sum_mps[None].z,
                self.report_particle_momentum_kg_mps[None].x,
                self.report_particle_momentum_kg_mps[None].y,
                self.report_particle_momentum_kg_mps[None].z,
                self.report_grid_momentum_kg_mps[None].x,
                self.report_grid_momentum_kg_mps[None].y,
                self.report_grid_momentum_kg_mps[None].z,
                self.report_total_force_n[None].x,
                self.report_total_force_n[None].y,
                self.report_total_force_n[None].z,
                self.report_transfer_relative_error[None],
            ]
        )
        self.report_count_snapshot[None] = ti.Vector(
            [
                self.report_active_grid_nodes[None],
                self.report_grid_out_of_bounds_particle_count[None],
                self.report_radial_stretch_count[None],
                self.report_primary_count[None],
                self.report_secondary_count[None],
            ]
        )
        packed_values = self.report_float_snapshot[None]
        packed_counts = self.report_count_snapshot[None]
        for snapshot_index in ti.static(range(32)):
            self.report_host_snapshot[snapshot_index] = packed_values[snapshot_index]
        for snapshot_index in ti.static(range(5)):
            self.report_host_snapshot[32 + snapshot_index] = ti.cast(
                packed_counts[snapshot_index],
                ti.f64,
            )

    @ti.func
    def _scalar_is_safe(self, value):
        return value == value and ti.abs(value) < 1.0e30

    @ti.func
    def _vector_is_safe(self, value):
        return (
            self._scalar_is_safe(value.x)
            and self._scalar_is_safe(value.y)
            and self._scalar_is_safe(value.z)
        )

    @ti.func
    def _atomic_add_vector(self, field, value):
        if self._vector_is_safe(value):
            ti.atomic_add(field[None].x, value.x)
            ti.atomic_add(field[None].y, value.y)
            ti.atomic_add(field[None].z, value.z)

    @ti.func
    def _atomic_add_particle_force(self, index, value):
        if self._vector_is_safe(value):
            ti.atomic_add(self.internal_force_n[index].x, value.x)
            ti.atomic_add(self.internal_force_n[index].y, value.y)
            ti.atomic_add(self.internal_force_n[index].z, value.z)

    @ti.func
    def _atomic_add_particle_external_force(self, index, value):
        if self._vector_is_safe(value):
            ti.atomic_add(self.external_force_n[index].x, value.x)
            ti.atomic_add(self.external_force_n[index].y, value.y)
            ti.atomic_add(self.external_force_n[index].z, value.z)

    @ti.func
    def _atomic_add_particle_surface_normal(self, index, value):
        ti.atomic_add(self.surface_normal[index].x, value.x)
        ti.atomic_add(self.surface_normal[index].y, value.y)
        ti.atomic_add(self.surface_normal[index].z, value.z)

    @ti.func
    def _atomic_add_particle_surface_area(self, index, value):
        ti.atomic_add(self.area_weight_m2[index], value)

    @ti.func
    def _accumulate_current_face_surface_normal(self, ia, ib, ic):
        area_vector = (self.x[ib] - self.x[ia]).cross(self.x[ic] - self.x[ia])
        area_vector_norm = area_vector.norm()
        if area_vector_norm > 1.0e-12:
            self._atomic_add_particle_surface_normal(ia, area_vector)
            self._atomic_add_particle_surface_normal(ib, area_vector)
            self._atomic_add_particle_surface_normal(ic, area_vector)
            area_share = 0.5 * area_vector_norm / 3.0
            self._atomic_add_particle_surface_area(ia, area_share)
            self._atomic_add_particle_surface_area(ib, area_share)
            self._atomic_add_particle_surface_area(ic, area_share)

    @ti.func
    def _update_particle_surface_normals(self):
        for p in range(self.particle_count):
            self.surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
            self.area_weight_m2[p] = 0.0
        for f in range(self.face_count):
            face = self.face_indices[f]
            self._accumulate_current_face_surface_normal(face.x, face.y, face.z)
        for p in range(self.particle_count):
            normal = self.surface_normal[p]
            norm = normal.norm()
            if norm > 1.0e-12:
                self.surface_normal[p] = normal / norm

    @ti.func
    def _accumulate_face_area_and_mass(self, ia, ib, ic, region, density_kgm3, thickness_m):
        ab = self.rest_x[ib] - self.rest_x[ia]
        ac = self.rest_x[ic] - self.rest_x[ia]
        area = 0.5 * ab.cross(ac).norm()
        share = area / 3.0
        ti.atomic_add(self.area_weight_m2[ia], share)
        ti.atomic_add(self.area_weight_m2[ib], share)
        ti.atomic_add(self.area_weight_m2[ic], share)
        mass_share = density_kgm3 * thickness_m * share
        ti.atomic_add(self.mass_kg[ia], mass_share)
        ti.atomic_add(self.mass_kg[ib], mass_share)
        ti.atomic_add(self.mass_kg[ic], mass_share)
        ti.atomic_min(self.vertex_region_id[ia], region)
        ti.atomic_min(self.vertex_region_id[ib], region)
        ti.atomic_min(self.vertex_region_id[ic], region)

    @ti.kernel
    def _init_mass_kernel(
        self,
        thickness_m: ti.f32,
        density_kgm3: ti.f32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        fixed_region_id: ti.i32,
        primary_thickness_m: ti.f32,
        secondary_thickness_m: ti.f32,
    ):
        self.primary_area_m2[None] = 0.0
        self.secondary_area_m2[None] = 0.0
        self.primary_mass_kg[None] = 0.0
        self.secondary_mass_kg[None] = 0.0
        for p in range(self.particle_count):
            self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            self.u[p] = ti.Vector([0.0, 0.0, 0.0])
            self.saved_u[p] = ti.Vector([0.0, 0.0, 0.0])
            self.saved_v[p] = ti.Vector([0.0, 0.0, 0.0])
            self.mass_kg[p] = 0.0
            self.area_weight_m2[p] = 0.0
            self.vertex_region_id[p] = 2147483647
            self.fixed_particle[p] = 0
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
        for f in range(self.face_count):
            face = self.face_indices[f]
            region = self.face_region_id[f]
            local_thickness = thickness_m
            if region == primary_region_id:
                local_thickness = primary_thickness_m
            elif region == secondary_region_id:
                local_thickness = secondary_thickness_m
            elif fixed_region_id >= 0 and region == fixed_region_id:
                local_thickness = primary_thickness_m
            self._accumulate_face_area_and_mass(
                face.x,
                face.y,
                face.z,
                region,
                density_kgm3,
                local_thickness,
            )
            rest_a = self.rest_x[face.x]
            rest_b = self.rest_x[face.y]
            rest_c = self.rest_x[face.z]
            area = 0.5 * (rest_b - rest_a).cross(rest_c - rest_a).norm()
            mass = density_kgm3 * local_thickness * area
            if fixed_region_id >= 0 and region == fixed_region_id:
                self.fixed_particle[face.x] = 1
                self.fixed_particle[face.y] = 1
                self.fixed_particle[face.z] = 1
            if region == primary_region_id:
                ti.atomic_add(self.primary_area_m2[None], area)
                ti.atomic_add(self.primary_mass_kg[None], mass)
            elif region == secondary_region_id:
                ti.atomic_add(self.secondary_area_m2[None], area)
                ti.atomic_add(self.secondary_mass_kg[None], mass)
        self._update_particle_surface_normals()

    @ti.func
    def _accumulate_edge_strain_stat(self, ia, ib):
        rest_delta = self.rest_x[ib] - self.rest_x[ia]
        current_delta = self.x[ib] - self.x[ia]
        rest_length_m = rest_delta.norm()
        current_length_m = current_delta.norm()
        if rest_length_m > 1.0e-12 and current_length_m > 1.0e-12:
            stretch = ti.max(current_length_m / rest_length_m, 1.0e-6)
            ti.atomic_max(self.report_max_edge_strain[None], ti.abs(stretch - 1.0))

    @ti.func
    def _limit_vector_norm(self, value, max_norm):
        limited = value
        norm = value.norm()
        if norm != norm or norm > 1.0e30:
            limited = ti.Vector([0.0, 0.0, 0.0])
        elif max_norm > 0.0 and norm > max_norm:
            limited = value * (max_norm / norm)
        return limited

    @ti.func
    def _accumulate_mooney_face(self, ia, ib, ic, pressure_pa, thickness_m):
        rest_a = self.rest_x[ia]
        rest_b = self.rest_x[ib]
        rest_c = self.rest_x[ic]
        a = self.x[ia]
        b = self.x[ib]
        c = self.x[ic]
        rest_area_vec = (rest_b - rest_a).cross(rest_c - rest_a)
        area_vec = (b - a).cross(c - a)
        rest_area_m2 = 0.5 * rest_area_vec.norm()
        rest_area_vec_norm = rest_area_vec.norm()
        area_vec_norm = area_vec.norm()
        area_m2 = 0.5 * area_vec_norm
        if rest_area_m2 > 1.0e-12 and rest_area_vec_norm > 1.0e-12:
            rest_normal = rest_area_vec / rest_area_vec_norm
            rest_edge0 = rest_b - rest_a
            rest_edge0_len = rest_edge0.norm()
            if rest_edge0_len > 1.0e-12:
                rest_t0 = rest_edge0 / rest_edge0_len
                rest_t1 = rest_normal.cross(rest_t0)
                rest_ac = rest_c - rest_a
                rest_xc = rest_ac.dot(rest_t0)
                rest_yc = rest_ac.dot(rest_t1)
                if ti.abs(rest_yc) > 1.0e-12:
                    inv00 = 1.0 / rest_edge0_len
                    inv01 = -rest_xc / (rest_edge0_len * rest_yc)
                    inv11 = 1.0 / rest_yc
                    edge_current0 = b - a
                    edge_current1 = c - a
                    f0 = edge_current0 * inv00
                    f1 = edge_current0 * inv01 + edge_current1 * inv11
                    c00 = f0.dot(f0)
                    c01 = f0.dot(f1)
                    c11 = f1.dot(f1)
                    c_cap = 1.0e6
                    c00 = ti.min(c00, c_cap)
                    c01 = ti.min(ti.max(c01, -c_cap), c_cap)
                    c11 = ti.min(c11, c_cap)
                    det_c = ti.max(c00 * c11 - c01 * c01, 1.0e-6)
                    inv_det_c = 1.0 / det_c
                    inv_c00 = c11 * inv_det_c
                    inv_c01 = -c01 * inv_det_c
                    inv_c11 = c00 * inv_det_c
                    trace_c = c00 + c11
                    s00 = self.c1_pa * (1.0 - inv_c00 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c00 + inv_det_c - trace_c * inv_det_c * inv_c00
                    )
                    s01 = self.c1_pa * (-inv_c01 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c01 - trace_c * inv_det_c * inv_c01
                    )
                    s11 = self.c1_pa * (1.0 - inv_c11 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c11 + inv_det_c - trace_c * inv_det_c * inv_c11
                    )
                    p0 = self.membrane_force_scale * 2.0 * (f0 * s00 + f1 * s01)
                    p1 = self.membrane_force_scale * 2.0 * (f0 * s01 + f1 * s11)
                    rest_volume_m3 = thickness_m * rest_area_m2
                    grad_edge0 = rest_volume_m3 * (p0 * inv00 + p1 * inv01)
                    grad_edge1 = rest_volume_m3 * (p1 * inv11)
                    force_cap_n = (
                        self.membrane_force_scale
                        * (self.c1_pa + self.c2_pa)
                        * thickness_m
                        * ti.sqrt(rest_area_m2)
                        * 100.0
                    )
                    self._atomic_add_particle_force(
                        ia,
                        self._limit_vector_norm(grad_edge0 + grad_edge1, force_cap_n),
                    )
                    self._atomic_add_particle_force(
                        ib,
                        self._limit_vector_norm(-grad_edge0, force_cap_n),
                    )
                    self._atomic_add_particle_force(
                        ic,
                        self._limit_vector_norm(-grad_edge1, force_cap_n),
                    )
            normal = rest_normal
            if area_vec_norm > 1.0e-12:
                normal = area_vec / area_vec_norm
            pressure_force = pressure_pa * area_m2 / 3.0 * normal
            self._atomic_add_particle_external_force(ia, pressure_force)
            self._atomic_add_particle_external_force(ib, pressure_force)
            self._atomic_add_particle_external_force(ic, pressure_force)

    @ti.func
    def _compute_surface_forces(self, pressure_pa):
        for p in range(self.particle_count):
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
        for f in range(self.face_count):
            face = self.face_indices[f]
            self._accumulate_mooney_face(face.x, face.y, face.z, pressure_pa, self.thickness_m)
        for e in range(self.edge_count):
            edge = self.edge_indices[e]
            self._accumulate_edge_strain_stat(edge.x, edge.y)

    @ti.func
    def _apply_region_load(self, ia, ib, ic, force_n):
        share = force_n / 3.0
        self._atomic_add_particle_external_force(ia, share)
        self._atomic_add_particle_external_force(ib, share)
        self._atomic_add_particle_external_force(ic, share)

    @ti.func
    def _compute_region_surface_forces(
        self,
        primary_region_id,
        secondary_region_id,
        primary_area_load_region_id,
        primary_area_load_npm2,
        primary_interface_reaction_n,
        secondary_interface_reaction_n,
        preserve_existing_external_force,
    ):
        for p in range(self.particle_count):
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            if preserve_existing_external_force == 0:
                self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
        for f in range(self.face_count):
            face = self.face_indices[f]
            region = self.face_region_id[f]
            rest_a = self.rest_x[face.x]
            rest_b = self.rest_x[face.y]
            rest_c = self.rest_x[face.z]
            area = 0.5 * (rest_b - rest_a).cross(rest_c - rest_a).norm()
            thickness_m = self.thickness_m
            if region == primary_region_id:
                thickness_m = self.primary_thickness_m
            elif region == secondary_region_id:
                thickness_m = self.secondary_thickness_m
            self._accumulate_mooney_face(face.x, face.y, face.z, 0.0, thickness_m)
            face_external_force = ti.Vector([0.0, 0.0, 0.0])
            if region == primary_area_load_region_id:
                face_external_force += primary_area_load_npm2 * area
            if region == primary_region_id:
                reaction = primary_interface_reaction_n * area / ti.max(self.primary_area_m2[None], 1.0e-12)
                face_external_force += reaction
            elif region == secondary_region_id:
                reaction = secondary_interface_reaction_n * area / ti.max(self.secondary_area_m2[None], 1.0e-12)
                face_external_force += reaction
            if face_external_force.norm() > 0.0:
                self._apply_region_load(face.x, face.y, face.z, face_external_force)
        for e in range(self.edge_count):
            edge = self.edge_indices[e]
            self._accumulate_edge_strain_stat(edge.x, edge.y)

    @ti.kernel
    def _add_region_area_load_kernel(
        self,
        region_id: ti.i32,
        area_load_x_npm2: ti.f32,
        area_load_y_npm2: ti.f32,
        area_load_z_npm2: ti.f32,
    ):
        area_load = ti.Vector(
            [area_load_x_npm2, area_load_y_npm2, area_load_z_npm2]
        )
        for f in range(self.face_count):
            if self.face_region_id[f] == region_id:
                face = self.face_indices[f]
                rest_a = self.rest_x[face.x]
                rest_b = self.rest_x[face.y]
                rest_c = self.rest_x[face.z]
                area = 0.5 * (rest_b - rest_a).cross(rest_c - rest_a).norm()
                self._apply_region_load(face.x, face.y, face.z, area_load * area)

    def add_region_area_load(
        self,
        *,
        region_id: int,
        area_load_npm2: tuple[float, float, float],
    ) -> None:
        area_load = _vector3(area_load_npm2, "area_load_npm2")
        self._add_region_area_load_kernel(
            int(region_id),
            float(area_load[0]),
            float(area_load[1]),
            float(area_load[2]),
        )

    @ti.func
    def _particle_grid_coordinate(self, p):
        return ti.Vector(
            [
                (self.x[p].x - self.bounds_min[0]) / self.dx[0],
                (self.x[p].y - self.bounds_min[1]) / self.dx[1],
                (self.x[p].z - self.bounds_min[2]) / self.dx[2],
            ]
        )

    @ti.func
    def _particle_grid_out_of_bounds(self, coord):
        out_of_bounds = 0
        if self._vector_is_safe(coord) == 0:
            out_of_bounds = 1
        if coord.x < 0.0 or coord.x > ti.cast(self.nx - 1, ti.f32):
            out_of_bounds = 1
        if coord.y < 0.0 or coord.y > ti.cast(self.ny - 1, ti.f32):
            out_of_bounds = 1
        if coord.z < 0.0 or coord.z > ti.cast(self.nz - 1, ti.f32):
            out_of_bounds = 1
        return out_of_bounds

    @ti.func
    def _scatter_particle(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        momentum = self.mass_kg[p] * self.v[p]
        total_force = self.internal_force_n[p] + self.external_force_n[p]
        if self.fixed_particle[p] == 0:
            for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
                weight = (
                    (fx.x if ox == 1 else 1.0 - fx.x)
                    * (fx.y if oy == 1 else 1.0 - fx.y)
                    * (fx.z if oz == 1 else 1.0 - fx.z)
                )
                node = (base.x + ox, base.y + oy, base.z + oz)
                ti.atomic_add(self.grid_mass_kg[node], weight * self.mass_kg[p])
                ti.atomic_add(self.grid_velocity_mps[node].x, weight * momentum.x)
                ti.atomic_add(self.grid_velocity_mps[node].y, weight * momentum.y)
                ti.atomic_add(self.grid_velocity_mps[node].z, weight * momentum.z)
                ti.atomic_add(self.grid_force_n[node].x, weight * total_force.x)
                ti.atomic_add(self.grid_force_n[node].y, weight * total_force.y)
                ti.atomic_add(self.grid_force_n[node].z, weight * total_force.z)

    @ti.func
    def _interpolate_grid_velocity(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        velocity = ti.Vector([0.0, 0.0, 0.0])
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (fx.x if ox == 1 else 1.0 - fx.x)
                * (fx.y if oy == 1 else 1.0 - fx.y)
                * (fx.z if oz == 1 else 1.0 - fx.z)
            )
            node = (base.x + ox, base.y + oy, base.z + oz)
            velocity += weight * self.grid_velocity_mps[node]
        return velocity

    @ti.func
    def _interpolate_grid_velocity_delta(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        delta = ti.Vector([0.0, 0.0, 0.0])
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (fx.x if ox == 1 else 1.0 - fx.x)
                * (fx.y if oy == 1 else 1.0 - fx.y)
                * (fx.z if oz == 1 else 1.0 - fx.z)
            )
            node = (base.x + ox, base.y + oy, base.z + oz)
            delta += weight * (
                self.grid_velocity_mps[node] - self.grid_velocity_before_force_mps[node]
            )
        return delta

    @ti.kernel
    def _step_kernel(
        self,
        dt_s: ti.f32,
        pressure_pa: ti.f32,
        velocity_damping: ti.f32,
        flip_blend: ti.f32,
        body_acceleration_x_mps2: ti.f32,
        body_acceleration_y_mps2: ti.f32,
        body_acceleration_z_mps2: ti.f32,
    ):
        for i, j, k in self.grid_mass_kg:
            self.grid_mass_kg[i, j, k] = 0.0
            self.grid_velocity_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_velocity_before_force_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_force_n[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self._clear_reports()
        self._compute_surface_forces(pressure_pa)
        body_acceleration = ti.Vector(
            [
                body_acceleration_x_mps2,
                body_acceleration_y_mps2,
                body_acceleration_z_mps2,
            ]
        )

        for p in range(self.particle_count):
            self.external_force_n[p] += self.mass_kg[p] * body_acceleration
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) != 0:
                ti.atomic_add(self.report_grid_out_of_bounds_particle_count[None], 1)
            else:
                self._scatter_particle(p)
            self._atomic_add_vector(self.report_internal_force_sum_n, self.internal_force_n[p])
            ti.atomic_add(
                self.report_internal_force_square_sum_n2[None],
                self.internal_force_n[p].dot(self.internal_force_n[p]),
            )
            particle_momentum = self.mass_kg[p] * self.v[p]
            total_force = self.internal_force_n[p] + self.external_force_n[p]
            if self.fixed_particle[p] != 0:
                particle_momentum = ti.Vector([0.0, 0.0, 0.0])
                total_force = ti.Vector([0.0, 0.0, 0.0])
            self._atomic_add_vector(self.report_particle_momentum_kg_mps, particle_momentum)
            self._atomic_add_vector(self.report_total_force_n, total_force)
            ti.atomic_add(self.report_total_mass_kg[None], self.mass_kg[p])
            ti.atomic_add(self.report_total_area_m2[None], self.area_weight_m2[p])
            ti.atomic_add(
                self.report_momentum_square_sum[None],
                particle_momentum.dot(particle_momentum),
            )
            force_impulse = dt_s * total_force
            ti.atomic_add(
                self.report_force_impulse_square_sum[None],
                force_impulse.dot(force_impulse),
            )

        for i, j, k in self.grid_mass_kg:
            mass = self.grid_mass_kg[i, j, k]
            if mass > 1.0e-20:
                velocity = self.grid_velocity_mps[i, j, k] / mass
                self.grid_velocity_before_force_mps[i, j, k] = velocity
                velocity += dt_s * self.grid_force_n[i, j, k] / mass
                transfer_velocity = velocity
                velocity *= velocity_damping
                self.grid_velocity_mps[i, j, k] = velocity
                self._atomic_add_vector(
                    self.report_transfer_grid_momentum_kg_mps,
                    mass * transfer_velocity,
                )
                self._atomic_add_vector(self.report_grid_momentum_kg_mps, mass * velocity)
                ti.atomic_add(self.report_active_grid_nodes[None], 1)
                ti.atomic_max(self.report_max_speed_mps[None], velocity.norm())

        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                pic_v = self._interpolate_grid_velocity(p)
                flip_v = self.v[p] + self._interpolate_grid_velocity_delta(p)
                new_v = (1.0 - flip_blend) * pic_v + flip_blend * flip_v
                self.v[p] = new_v
                delta_x = dt_s * new_v
                if self.fixed_particle[p] == 0:
                    self.x[p] += delta_x
                    self.u[p] += delta_x
                else:
                    self.x[p] = self.rest_x[p]
                    self.u[p] = ti.Vector([0.0, 0.0, 0.0])
                    self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            elif self.fixed_particle[p] != 0:
                self.x[p] = self.rest_x[p]
                self.u[p] = ti.Vector([0.0, 0.0, 0.0])
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            else:
                delta_x = dt_s * self.v[p]
                self.x[p] += delta_x
                self.u[p] += delta_x
                ti.atomic_max(self.report_max_speed_mps[None], self.v[p].norm())
            report_coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(report_coord) == 0:
                self._atomic_add_vector(self.report_current_center_sum_m, self.x[p])
                self._atomic_add_vector(self.report_radial_rest_center_sum_m, self.rest_x[p])
                ti.atomic_add(self.report_radial_center_count[None], 1)
                region = self.vertex_region_id[p]
                if region == self.primary_region_id:
                    self._atomic_add_vector(self.report_primary_displacement_sum_m, self.u[p])
                    self._atomic_add_vector(self.report_primary_velocity_sum_mps, self.v[p])
                    ti.atomic_add(self.report_primary_count[None], 1)
                elif region == self.secondary_region_id:
                    self._atomic_add_vector(self.report_secondary_displacement_sum_m, self.u[p])
                    self._atomic_add_vector(self.report_secondary_velocity_sum_mps, self.v[p])
                    ti.atomic_add(self.report_secondary_count[None], 1)

        radial_center_count = ti.max(ti.cast(self.report_radial_center_count[None], ti.f32), 1.0)
        current_center = self.report_current_center_sum_m[None] / radial_center_count
        rest_center = self.report_radial_rest_center_sum_m[None] / radial_center_count
        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                rest_radius = (self.rest_x[p] - rest_center).norm()
                current_radius = (self.x[p] - current_center).norm()
                radial_stretch = current_radius / ti.max(rest_radius, 1.0e-12)
                ti.atomic_add(self.report_radial_stretch_sum[None], radial_stretch)
                ti.atomic_add(self.report_radial_stretch_count[None], 1)
                ti.atomic_max(
                    self.report_max_radial_stretch_error[None],
                    ti.abs(radial_stretch - 1.0),
                )

        expected_grid_momentum = (
            self.report_particle_momentum_kg_mps[None]
            + dt_s * self.report_total_force_n[None]
        )
        denominator = ti.max(
            expected_grid_momentum.norm(),
            ti.sqrt(self.report_momentum_square_sum[None]),
            ti.sqrt(self.report_force_impulse_square_sum[None]),
            1.0e-20,
        )
        self.report_transfer_relative_error[None] = (
            self.report_transfer_grid_momentum_kg_mps[None] - expected_grid_momentum
        ).norm() / denominator
        self._update_particle_surface_normals()
        self._pack_report_snapshot()

    @ti.kernel
    def _step_region_kernel(
        self,
        dt_s: ti.f32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_area_load_region_id: ti.i32,
        primary_area_load_x_npm2: ti.f32,
        primary_area_load_y_npm2: ti.f32,
        primary_area_load_z_npm2: ti.f32,
        primary_interface_reaction_x_n: ti.f32,
        primary_interface_reaction_y_n: ti.f32,
        primary_interface_reaction_z_n: ti.f32,
        secondary_interface_reaction_x_n: ti.f32,
        secondary_interface_reaction_y_n: ti.f32,
        secondary_interface_reaction_z_n: ti.f32,
        velocity_damping: ti.f32,
        flip_blend: ti.f32,
        body_acceleration_x_mps2: ti.f32,
        body_acceleration_y_mps2: ti.f32,
        body_acceleration_z_mps2: ti.f32,
        preserve_existing_external_force: ti.i32,
    ):
        for i, j, k in self.grid_mass_kg:
            self.grid_mass_kg[i, j, k] = 0.0
            self.grid_velocity_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_velocity_before_force_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_force_n[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self._clear_reports()
        primary_area_load = ti.Vector(
            [primary_area_load_x_npm2, primary_area_load_y_npm2, primary_area_load_z_npm2]
        )
        primary_reaction = ti.Vector(
            [primary_interface_reaction_x_n, primary_interface_reaction_y_n, primary_interface_reaction_z_n]
        )
        secondary_reaction = ti.Vector(
            [secondary_interface_reaction_x_n, secondary_interface_reaction_y_n, secondary_interface_reaction_z_n]
        )
        self._compute_region_surface_forces(
            primary_region_id,
            secondary_region_id,
            primary_area_load_region_id,
            primary_area_load,
            primary_reaction,
            secondary_reaction,
            preserve_existing_external_force,
        )
        body_acceleration = ti.Vector(
            [
                body_acceleration_x_mps2,
                body_acceleration_y_mps2,
                body_acceleration_z_mps2,
            ]
        )

        for p in range(self.particle_count):
            self.external_force_n[p] += self.mass_kg[p] * body_acceleration
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) != 0:
                ti.atomic_add(self.report_grid_out_of_bounds_particle_count[None], 1)
            else:
                self._scatter_particle(p)
            self._atomic_add_vector(self.report_internal_force_sum_n, self.internal_force_n[p])
            ti.atomic_add(
                self.report_internal_force_square_sum_n2[None],
                self.internal_force_n[p].dot(self.internal_force_n[p]),
            )
            particle_momentum = self.mass_kg[p] * self.v[p]
            total_force = self.internal_force_n[p] + self.external_force_n[p]
            if self.fixed_particle[p] != 0:
                particle_momentum = ti.Vector([0.0, 0.0, 0.0])
                total_force = ti.Vector([0.0, 0.0, 0.0])
            self._atomic_add_vector(self.report_particle_momentum_kg_mps, particle_momentum)
            self._atomic_add_vector(self.report_total_force_n, total_force)
            ti.atomic_add(self.report_total_mass_kg[None], self.mass_kg[p])
            ti.atomic_add(self.report_total_area_m2[None], self.area_weight_m2[p])
            ti.atomic_add(
                self.report_momentum_square_sum[None],
                particle_momentum.dot(particle_momentum),
            )
            force_impulse = dt_s * total_force
            ti.atomic_add(
                self.report_force_impulse_square_sum[None],
                force_impulse.dot(force_impulse),
            )

        for i, j, k in self.grid_mass_kg:
            mass = self.grid_mass_kg[i, j, k]
            if mass > 1.0e-20:
                velocity = self.grid_velocity_mps[i, j, k] / mass
                self.grid_velocity_before_force_mps[i, j, k] = velocity
                velocity += dt_s * self.grid_force_n[i, j, k] / mass
                transfer_velocity = velocity
                velocity *= velocity_damping
                self.grid_velocity_mps[i, j, k] = velocity
                self._atomic_add_vector(
                    self.report_transfer_grid_momentum_kg_mps,
                    mass * transfer_velocity,
                )
                self._atomic_add_vector(self.report_grid_momentum_kg_mps, mass * velocity)
                ti.atomic_add(self.report_active_grid_nodes[None], 1)
                ti.atomic_max(self.report_max_speed_mps[None], velocity.norm())

        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                pic_v = self._interpolate_grid_velocity(p)
                flip_v = self.v[p] + self._interpolate_grid_velocity_delta(p)
                new_v = (1.0 - flip_blend) * pic_v + flip_blend * flip_v
                self.v[p] = new_v
                delta_x = dt_s * new_v
                if self.fixed_particle[p] == 0:
                    self.x[p] += delta_x
                    self.u[p] += delta_x
                else:
                    self.x[p] = self.rest_x[p]
                    self.u[p] = ti.Vector([0.0, 0.0, 0.0])
                    self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            elif self.fixed_particle[p] != 0:
                self.x[p] = self.rest_x[p]
                self.u[p] = ti.Vector([0.0, 0.0, 0.0])
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            else:
                delta_x = dt_s * self.v[p]
                self.x[p] += delta_x
                self.u[p] += delta_x
                ti.atomic_max(self.report_max_speed_mps[None], self.v[p].norm())
            report_coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(report_coord) == 0:
                self._atomic_add_vector(self.report_current_center_sum_m, self.x[p])
                self._atomic_add_vector(self.report_radial_rest_center_sum_m, self.rest_x[p])
                ti.atomic_add(self.report_radial_center_count[None], 1)
                region = self.vertex_region_id[p]
                if region == primary_region_id:
                    self._atomic_add_vector(self.report_primary_displacement_sum_m, self.u[p])
                    self._atomic_add_vector(self.report_primary_velocity_sum_mps, self.v[p])
                    ti.atomic_add(self.report_primary_count[None], 1)
                elif region == secondary_region_id:
                    self._atomic_add_vector(self.report_secondary_displacement_sum_m, self.u[p])
                    self._atomic_add_vector(self.report_secondary_velocity_sum_mps, self.v[p])
                    ti.atomic_add(self.report_secondary_count[None], 1)

        radial_center_count = ti.max(ti.cast(self.report_radial_center_count[None], ti.f32), 1.0)
        current_center = self.report_current_center_sum_m[None] / radial_center_count
        rest_center = self.report_radial_rest_center_sum_m[None] / radial_center_count
        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                rest_radius = (self.rest_x[p] - rest_center).norm()
                current_radius = (self.x[p] - current_center).norm()
                radial_stretch = current_radius / ti.max(rest_radius, 1.0e-12)
                ti.atomic_add(self.report_radial_stretch_sum[None], radial_stretch)
                ti.atomic_add(self.report_radial_stretch_count[None], 1)
                ti.atomic_max(
                    self.report_max_radial_stretch_error[None],
                    ti.abs(radial_stretch - 1.0),
                )

        expected_grid_momentum = (
            self.report_particle_momentum_kg_mps[None]
            + dt_s * self.report_total_force_n[None]
        )
        denominator = ti.max(
            expected_grid_momentum.norm(),
            ti.sqrt(self.report_momentum_square_sum[None]),
            ti.sqrt(self.report_force_impulse_square_sum[None]),
            1.0e-20,
        )
        self.report_transfer_relative_error[None] = (
            self.report_transfer_grid_momentum_kg_mps[None] - expected_grid_momentum
        ).norm() / denominator
        self._update_particle_surface_normals()
        self._pack_report_snapshot()

    def step(
        self,
        *,
        dt_s: float,
        pressure_pa: float,
        velocity_damping: float = 1.0,
        flip_blend: float = 0.95,
        body_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_report: bool = True,
    ) -> TriMooneyShellMpmReport | None:
        body_acceleration = _vector3(body_acceleration_mps2, "body_acceleration_mps2")
        self._step_kernel(
            float(dt_s),
            float(pressure_pa),
            float(velocity_damping),
            _validate_flip_blend(flip_blend),
            float(body_acceleration[0]),
            float(body_acceleration[1]),
            float(body_acceleration[2]),
        )
        if not read_report:
            self.last_report_host_reads = 0
            return None
        return self.report()

    def advance_region_loads(
        self,
        *,
        dt_s: float,
        primary_region_id: int,
        secondary_region_id: int,
        primary_area_load_npm2: tuple[float, float, float],
        primary_interface_reaction_n: tuple[float, float, float],
        secondary_interface_reaction_n: tuple[float, float, float],
        primary_area_load_region_id: int | None = None,
        velocity_damping: float = 1.0,
        flip_blend: float = 0.95,
        body_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_report: bool = True,
    ) -> TriMooneyShellMpmReport | None:
        """Advance two region-specific external loads without case-specific axes."""
        primary_area_load = _vector3(primary_area_load_npm2, "primary_area_load_npm2")
        primary_reaction = _vector3(primary_interface_reaction_n, "primary_interface_reaction_n")
        secondary_reaction = _vector3(secondary_interface_reaction_n, "secondary_interface_reaction_n")
        body_acceleration = _vector3(body_acceleration_mps2, "body_acceleration_mps2")
        area_load_region_id = (
            int(primary_region_id)
            if primary_area_load_region_id is None
            else int(primary_area_load_region_id)
        )
        self._step_region_kernel(
            float(dt_s),
            int(primary_region_id),
            int(secondary_region_id),
            area_load_region_id,
            float(primary_area_load[0]),
            float(primary_area_load[1]),
            float(primary_area_load[2]),
            float(primary_reaction[0]),
            float(primary_reaction[1]),
            float(primary_reaction[2]),
            float(secondary_reaction[0]),
            float(secondary_reaction[1]),
            float(secondary_reaction[2]),
            float(velocity_damping),
            _validate_flip_blend(flip_blend),
            float(body_acceleration[0]),
            float(body_acceleration[1]),
            float(body_acceleration[2]),
            0,
        )
        if not read_report:
            self.last_report_host_reads = 0
            return None
        return self.report()

    def advance_with_external_forces(
        self,
        *,
        dt_s: float,
        primary_region_id: int,
        secondary_region_id: int,
        velocity_damping: float = 1.0,
        flip_blend: float = 0.95,
        body_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_report: bool = True,
    ) -> TriMooneyShellMpmReport | None:
        """Advance shell dynamics using preloaded external_force_n as MPM load."""
        body_acceleration = _vector3(body_acceleration_mps2, "body_acceleration_mps2")
        self._step_region_kernel(
            float(dt_s),
            int(primary_region_id),
            int(secondary_region_id),
            int(primary_region_id),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(velocity_damping),
            _validate_flip_blend(flip_blend),
            float(body_acceleration[0]),
            float(body_acceleration[1]),
            float(body_acceleration[2]),
            1,
        )
        if not read_report:
            self.last_report_host_reads = 0
            return None
        return self.report()

    @ti.kernel
    def _save_state_kernel(self):
        for p in range(self.particle_count):
            self.saved_x[p] = self.x[p]
            self.saved_u[p] = self.u[p]
            self.saved_v[p] = self.v[p]

    @ti.kernel
    def _restore_state_kernel(self):
        for p in range(self.particle_count):
            self.x[p] = self.saved_x[p]
            self.u[p] = self.saved_u[p]
            self.v[p] = self.saved_v[p]
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
        self._update_particle_surface_normals()

    def save_state(self) -> None:
        self._save_state_kernel()

    def restore_state(self) -> None:
        self._restore_state_kernel()

    def report(self) -> TriMooneyShellMpmReport:
        snapshot = self.report_host_snapshot.to_numpy()
        values = snapshot[:32]
        counts = snapshot[32:37]
        self.last_report_host_reads = 1
        _raise_if_out_of_bounds_exceeds_tolerance(
            int(self.particle_count),
            int(counts[1]),
            self.out_of_bounds_particle_tolerance,
        )
        force_l2 = float(values[0]) ** 0.5
        total_area_m2 = float(values[4])
        particle_spacing_m = 0.0
        if total_area_m2 > 0.0 and self.particle_count > 0:
            particle_spacing_m = (total_area_m2 / self.particle_count) ** 0.5
        primary_particle_count = int(counts[3])
        secondary_particle_count = int(counts[4])
        if self.require_nonempty_region_counts:
            _raise_if_required_shell_region_empty(
                primary_count=primary_particle_count,
                secondary_count=secondary_particle_count,
            )
        primary_count = max(primary_particle_count, 1)
        secondary_count = max(secondary_particle_count, 1)
        return TriMooneyShellMpmReport(
            particle_count=self.particle_count,
            face_count=self.face_count,
            edge_count=self.edge_count,
            active_grid_nodes=int(counts[0]),
            grid_out_of_bounds_particle_count=int(counts[1]),
            particle_spacing_m=particle_spacing_m,
            grid_spacing_m=self.dx,
            mean_radial_stretch=float(values[5]) / max(int(counts[2]), 1),
            max_radial_stretch_error=float(values[6]),
            max_edge_strain=float(values[7]),
            max_speed_mps=float(values[8]),
            total_mass_kg=float(values[9]),
            total_area_m2=total_area_m2,
            primary_mean_displacement_m=(
                float(values[10]) / primary_count,
                float(values[11]) / primary_count,
                float(values[12]) / primary_count,
            ),
            primary_mean_velocity_mps=(
                float(values[13]) / primary_count,
                float(values[14]) / primary_count,
                float(values[15]) / primary_count,
            ),
            secondary_mean_displacement_m=(
                float(values[16]) / secondary_count,
                float(values[17]) / secondary_count,
                float(values[18]) / secondary_count,
            ),
            secondary_mean_velocity_mps=(
                float(values[19]) / secondary_count,
                float(values[20]) / secondary_count,
                float(values[21]) / secondary_count,
            ),
            particle_momentum_kg_mps=(
                float(values[22]),
                float(values[23]),
                float(values[24]),
            ),
            grid_momentum_kg_mps=(
                float(values[25]),
                float(values[26]),
                float(values[27]),
            ),
            total_force_n=(
                float(values[28]),
                float(values[29]),
                float(values[30]),
            ),
            internal_force_rms_n=(force_l2 * force_l2 / max(self.particle_count, 1)) ** 0.5,
            net_internal_force_relative_error=(
                (
                    float(values[1]) ** 2
                    + float(values[2]) ** 2
                    + float(values[3]) ** 2
                )
                ** 0.5
                / max(force_l2, 1.0e-20)
            ),
            transfer_relative_error=float(values[31]),
            primary_particle_count=primary_particle_count,
            secondary_particle_count=secondary_particle_count,
        )


@ti.data_oriented
class UvMooneyShellMpmState:
    """Paper-style UV-sphere membrane MPM with 2D Mooney membrane forces."""

    def __init__(
        self,
        resolution: UvSphereResolution,
        *,
        radius_m: float,
        thickness_m: float,
        density_kgm3: float,
        c1_pa: float,
        c2_pa: float,
        membrane_force_scale: float = 1.0,
        grid_nodes: tuple[int, int, int] = (36, 36, 36),
        bounds_scale: float = 2.0,
        runtime: TaichiRuntimeConfig | None = None,
    ):
        init_taichi(runtime)
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if density_kgm3 <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        if c1_pa <= 0.0 or c2_pa < 0.0:
            raise ValueError("Mooney constants must be non-negative with c1 > 0")
        if membrane_force_scale <= 0.0:
            raise ValueError("membrane_force_scale must be positive")
        if min(grid_nodes) < 4:
            raise ValueError("grid_nodes must be at least 4 in each direction")
        if bounds_scale <= 1.0:
            raise ValueError("bounds_scale must be greater than 1")

        self.latitude_bands = int(resolution.latitude_bands)
        self.longitude_segments = int(resolution.longitude_segments)
        self.particle_count = int(resolution.vertex_count)
        self.radius_m = float(radius_m)
        self.thickness_m = float(thickness_m)
        self.density_kgm3 = float(density_kgm3)
        self.c1_pa = float(c1_pa)
        self.c2_pa = float(c2_pa)
        self.membrane_force_scale = float(membrane_force_scale)
        self.grid_nodes = tuple(int(value) for value in grid_nodes)
        self.nx, self.ny, self.nz = self.grid_nodes
        bound = bounds_scale * self.radius_m
        self.bounds_min = (-bound, -bound, -bound)
        self.bounds_max = (bound, bound, bound)
        self.dx = (
            (self.bounds_max[0] - self.bounds_min[0]) / (self.nx - 1),
            (self.bounds_max[1] - self.bounds_min[1]) / (self.ny - 1),
            (self.bounds_max[2] - self.bounds_min[2]) / (self.nz - 1),
        )

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.rest_x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.mass_kg = ti.field(dtype=ti.f32, shape=self.particle_count)
        self.area_weight_m2 = ti.field(dtype=ti.f32, shape=self.particle_count)
        self.surface_normal = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.internal_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.external_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.rest_center_m = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.grid_mass_kg = ti.field(dtype=ti.f32, shape=self.grid_nodes)
        self.grid_velocity_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)
        self.grid_velocity_before_force_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)
        self.grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)

        self.report_edge_count = ti.field(dtype=ti.i32, shape=())
        self.report_active_grid_nodes = ti.field(dtype=ti.i32, shape=())
        self.report_grid_out_of_bounds_particle_count = ti.field(dtype=ti.i32, shape=())
        self.report_total_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.report_radial_stretch_sum = ti.field(dtype=ti.f32, shape=())
        self.report_radial_stretch_count = ti.field(dtype=ti.i32, shape=())
        self.report_max_radial_stretch_error = ti.field(dtype=ti.f32, shape=())
        self.report_max_edge_strain = ti.field(dtype=ti.f32, shape=())
        self.report_max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.report_internal_force_sum_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_internal_force_square_sum_n2 = ti.field(dtype=ti.f32, shape=())
        self.report_particle_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_transfer_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_total_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_current_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_rest_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_center_count = ti.field(dtype=ti.i32, shape=())
        self.report_momentum_square_sum = ti.field(dtype=ti.f32, shape=())
        self.report_force_impulse_square_sum = ti.field(dtype=ti.f32, shape=())
        self.report_transfer_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_float_snapshot = ti.Vector.field(10, dtype=ti.f32, shape=())
        self.report_count_snapshot = ti.Vector.field(4, dtype=ti.i32, shape=())
        self.report_host_snapshot = ti.field(dtype=ti.f32, shape=14)
        self.last_report_host_reads = 0

        self._init_kernel(
            float(radius_m),
            float(thickness_m),
            float(density_kgm3),
        )
        self._update_rest_center()

    @ti.func
    def _ring_index(self, ring, segment):
        return 1 + (ring - 1) * self.longitude_segments + segment % self.longitude_segments

    @ti.func
    def _uv_vertex(self, index, radius_m):
        point = ti.Vector([0.0, 0.0, radius_m])
        if index == self.particle_count - 1:
            point = ti.Vector([0.0, 0.0, -radius_m])
        elif index > 0:
            local = index - 1
            ring = local // self.longitude_segments + 1
            segment = local % self.longitude_segments
            theta = ti.math.pi * ti.cast(ring, ti.f32) / ti.cast(self.latitude_bands, ti.f32)
            phi = (
                2.0
                * ti.math.pi
                * ti.cast(segment, ti.f32)
                / ti.cast(self.longitude_segments, ti.f32)
            )
            point = ti.Vector(
                [
                    radius_m * ti.sin(theta) * ti.cos(phi),
                    radius_m * ti.sin(theta) * ti.sin(phi),
                    radius_m * ti.cos(theta),
                ]
            )
        return point

    @ti.kernel
    def _init_kernel(self, radius_m: ti.f32, thickness_m: ti.f32, density_kgm3: ti.f32):
        for p in range(self.particle_count):
            point = self._uv_vertex(p, radius_m)
            self.x[p] = point
            self.rest_x[p] = point
            self.v[p] = ti.Vector([0.0, 0.0, 0.0])
            self.mass_kg[p] = 0.0
            self.area_weight_m2[p] = 0.0
            self.surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
        top_index = 0
        bottom_index = self.particle_count - 1
        last_ring = self.latitude_bands - 1
        for j in range(self.longitude_segments):
            self._accumulate_face_area(top_index, self._ring_index(1, j), self._ring_index(1, j + 1))
            self._accumulate_face_area(bottom_index, self._ring_index(last_ring, j + 1), self._ring_index(last_ring, j))
        for ring, j in ti.ndrange(self.latitude_bands - 2, self.longitude_segments):
            current_ring = ring + 1
            next_ring = current_ring + 1
            a = self._ring_index(current_ring, j)
            b = self._ring_index(current_ring, j + 1)
            c = self._ring_index(next_ring, j + 1)
            d = self._ring_index(next_ring, j)
            self._accumulate_face_area(a, d, c)
            self._accumulate_face_area(a, c, b)
        for p in range(self.particle_count):
            self.mass_kg[p] = density_kgm3 * thickness_m * self.area_weight_m2[p]
        self._update_particle_surface_normals()

    @ti.kernel
    def _update_rest_center_kernel(self):
        self.rest_center_m[None] = ti.Vector([0.0, 0.0, 0.0])
        for p in range(self.particle_count):
            self._atomic_add_vector(self.rest_center_m, self.rest_x[p])

    @ti.kernel
    def _normalize_rest_center_kernel(self):
        self.rest_center_m[None] = self.rest_center_m[None] / ti.cast(self.particle_count, ti.f32)

    def _update_rest_center(self) -> None:
        self._update_rest_center_kernel()
        self._normalize_rest_center_kernel()

    @ti.func
    def _accumulate_face_area(self, ia, ib, ic):
        ab = self.rest_x[ib] - self.rest_x[ia]
        ac = self.rest_x[ic] - self.rest_x[ia]
        area = 0.5 * ab.cross(ac).norm()
        share = area / 3.0
        ti.atomic_add(self.area_weight_m2[ia], share)
        ti.atomic_add(self.area_weight_m2[ib], share)
        ti.atomic_add(self.area_weight_m2[ic], share)

    @ti.func
    def _atomic_add_particle_surface_normal(self, index, value):
        ti.atomic_add(self.surface_normal[index].x, value.x)
        ti.atomic_add(self.surface_normal[index].y, value.y)
        ti.atomic_add(self.surface_normal[index].z, value.z)

    @ti.func
    def _atomic_add_particle_surface_area(self, index, value):
        ti.atomic_add(self.area_weight_m2[index], value)

    @ti.func
    def _accumulate_current_face_surface_normal(self, ia, ib, ic):
        area_vector = (self.x[ib] - self.x[ia]).cross(self.x[ic] - self.x[ia])
        area_vector_norm = area_vector.norm()
        if area_vector_norm > 1.0e-12:
            self._atomic_add_particle_surface_normal(ia, area_vector)
            self._atomic_add_particle_surface_normal(ib, area_vector)
            self._atomic_add_particle_surface_normal(ic, area_vector)
            area_share = 0.5 * area_vector_norm / 3.0
            self._atomic_add_particle_surface_area(ia, area_share)
            self._atomic_add_particle_surface_area(ib, area_share)
            self._atomic_add_particle_surface_area(ic, area_share)

    @ti.func
    def _update_particle_surface_normals(self):
        for p in range(self.particle_count):
            self.surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
            self.area_weight_m2[p] = 0.0
        top_index = 0
        bottom_index = self.particle_count - 1
        last_ring = self.latitude_bands - 1
        for j in range(self.longitude_segments):
            self._accumulate_current_face_surface_normal(
                top_index,
                self._ring_index(1, j),
                self._ring_index(1, j + 1),
            )
            self._accumulate_current_face_surface_normal(
                bottom_index,
                self._ring_index(last_ring, j + 1),
                self._ring_index(last_ring, j),
            )
        for ring, j in ti.ndrange(self.latitude_bands - 2, self.longitude_segments):
            current_ring = ring + 1
            next_ring = current_ring + 1
            a = self._ring_index(current_ring, j)
            b = self._ring_index(current_ring, j + 1)
            c = self._ring_index(next_ring, j + 1)
            d = self._ring_index(next_ring, j)
            self._accumulate_current_face_surface_normal(a, d, c)
            self._accumulate_current_face_surface_normal(a, c, b)
        for p in range(self.particle_count):
            normal = self.surface_normal[p]
            norm = normal.norm()
            if norm > 1.0e-12:
                self.surface_normal[p] = normal / norm

    @ti.func
    def _clear_reports(self):
        self.report_edge_count[None] = 0
        self.report_active_grid_nodes[None] = 0
        self.report_grid_out_of_bounds_particle_count[None] = 0
        self.report_total_mass_kg[None] = 0.0
        self.report_radial_stretch_sum[None] = 0.0
        self.report_radial_stretch_count[None] = 0
        self.report_max_radial_stretch_error[None] = 0.0
        self.report_max_edge_strain[None] = 0.0
        self.report_max_speed_mps[None] = 0.0
        self.report_internal_force_sum_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_internal_force_square_sum_n2[None] = 0.0
        self.report_particle_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_transfer_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_total_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_current_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_rest_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_center_count[None] = 0
        self.report_momentum_square_sum[None] = 0.0
        self.report_force_impulse_square_sum[None] = 0.0
        self.report_transfer_relative_error[None] = 0.0
        self.report_float_snapshot[None] = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.report_count_snapshot[None] = ti.Vector([0, 0, 0, 0])

    @ti.func
    def _pack_report_snapshot(self):
        self.report_float_snapshot[None] = ti.Vector(
            [
                self.report_internal_force_square_sum_n2[None],
                self.report_internal_force_sum_n[None].x,
                self.report_internal_force_sum_n[None].y,
                self.report_internal_force_sum_n[None].z,
                self.report_radial_stretch_sum[None],
                self.report_max_radial_stretch_error[None],
                self.report_max_edge_strain[None],
                self.report_max_speed_mps[None],
                self.report_total_mass_kg[None],
                self.report_transfer_relative_error[None],
            ]
        )
        self.report_count_snapshot[None] = ti.Vector(
            [
                self.report_edge_count[None],
                self.report_active_grid_nodes[None],
                self.report_grid_out_of_bounds_particle_count[None],
                self.report_radial_stretch_count[None],
            ]
        )
        packed_values = self.report_float_snapshot[None]
        packed_counts = self.report_count_snapshot[None]
        for snapshot_index in ti.static(range(10)):
            self.report_host_snapshot[snapshot_index] = packed_values[snapshot_index]
        for snapshot_index in ti.static(range(4)):
            self.report_host_snapshot[10 + snapshot_index] = ti.cast(
                packed_counts[snapshot_index],
                ti.f32,
            )

    @ti.func
    def _atomic_add_vector(self, field, value):
        ti.atomic_add(field[None].x, value.x)
        ti.atomic_add(field[None].y, value.y)
        ti.atomic_add(field[None].z, value.z)

    @ti.func
    def _atomic_add_particle_force(self, index, value):
        ti.atomic_add(self.internal_force_n[index].x, value.x)
        ti.atomic_add(self.internal_force_n[index].y, value.y)
        ti.atomic_add(self.internal_force_n[index].z, value.z)

    @ti.func
    def _atomic_add_particle_external_force(self, index, value):
        ti.atomic_add(self.external_force_n[index].x, value.x)
        ti.atomic_add(self.external_force_n[index].y, value.y)
        ti.atomic_add(self.external_force_n[index].z, value.z)

    @ti.func
    def _accumulate_edge_strain_stat(self, ia, ib):
        rest_delta = self.rest_x[ib] - self.rest_x[ia]
        current_delta = self.x[ib] - self.x[ia]
        rest_length_m = rest_delta.norm()
        current_length_m = current_delta.norm()
        if rest_length_m > 1.0e-12 and current_length_m > 1.0e-12:
            stretch = ti.max(current_length_m / rest_length_m, 1.0e-6)
            ti.atomic_max(self.report_max_edge_strain[None], ti.abs(stretch - 1.0))
            ti.atomic_add(self.report_edge_count[None], 1)

    @ti.func
    def _accumulate_mooney_face(self, ia, ib, ic, pressure_pa):
        rest_a = self.rest_x[ia]
        rest_b = self.rest_x[ib]
        rest_c = self.rest_x[ic]
        a = self.x[ia]
        b = self.x[ib]
        c = self.x[ic]
        rest_area_vec = (rest_b - rest_a).cross(rest_c - rest_a)
        area_vec = (b - a).cross(c - a)
        rest_area_m2 = 0.5 * rest_area_vec.norm()
        rest_area_vec_norm = rest_area_vec.norm()
        area_vec_norm = area_vec.norm()
        area_m2 = 0.5 * area_vec_norm
        if rest_area_m2 > 1.0e-12 and rest_area_vec_norm > 1.0e-12:
            rest_normal = rest_area_vec / rest_area_vec_norm
            rest_edge0 = rest_b - rest_a
            rest_edge0_len = rest_edge0.norm()
            if rest_edge0_len > 1.0e-12:
                rest_t0 = rest_edge0 / rest_edge0_len
                rest_t1 = rest_normal.cross(rest_t0)
                rest_ac = rest_c - rest_a
                rest_xc = rest_ac.dot(rest_t0)
                rest_yc = rest_ac.dot(rest_t1)
                if ti.abs(rest_yc) > 1.0e-12:
                    inv00 = 1.0 / rest_edge0_len
                    inv01 = -rest_xc / (rest_edge0_len * rest_yc)
                    inv11 = 1.0 / rest_yc
                    edge_current0 = b - a
                    edge_current1 = c - a
                    f0 = edge_current0 * inv00
                    f1 = edge_current0 * inv01 + edge_current1 * inv11
                    c00 = f0.dot(f0)
                    c01 = f0.dot(f1)
                    c11 = f1.dot(f1)
                    det_c = ti.max(c00 * c11 - c01 * c01, 1.0e-12)
                    inv_det_c = 1.0 / det_c
                    inv_c00 = c11 * inv_det_c
                    inv_c01 = -c01 * inv_det_c
                    inv_c11 = c00 * inv_det_c
                    trace_c = c00 + c11
                    s00 = self.c1_pa * (1.0 - inv_c00 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c00 + inv_det_c - trace_c * inv_det_c * inv_c00
                    )
                    s01 = self.c1_pa * (-inv_c01 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c01 - trace_c * inv_det_c * inv_c01
                    )
                    s11 = self.c1_pa * (1.0 - inv_c11 * inv_det_c) + self.c2_pa * (
                        det_c * inv_c11 + inv_det_c - trace_c * inv_det_c * inv_c11
                    )
                    p0 = self.membrane_force_scale * 2.0 * (f0 * s00 + f1 * s01)
                    p1 = self.membrane_force_scale * 2.0 * (f0 * s01 + f1 * s11)
                    rest_volume_m3 = self.thickness_m * rest_area_m2
                    grad_edge0 = rest_volume_m3 * (p0 * inv00 + p1 * inv01)
                    grad_edge1 = rest_volume_m3 * (p1 * inv11)
                    self._atomic_add_particle_force(ia, grad_edge0 + grad_edge1)
                    self._atomic_add_particle_force(ib, -grad_edge0)
                    self._atomic_add_particle_force(ic, -grad_edge1)
            normal = rest_normal
            if area_vec_norm > 1.0e-12:
                normal = area_vec / area_vec_norm
            pressure_force = pressure_pa * area_m2 / 3.0 * normal
            self._atomic_add_particle_external_force(ia, pressure_force)
            self._atomic_add_particle_external_force(ib, pressure_force)
            self._atomic_add_particle_external_force(ic, pressure_force)

    @ti.func
    def _compute_surface_forces(self, pressure_pa):
        for p in range(self.particle_count):
            self.internal_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])

        top_index = 0
        bottom_index = self.particle_count - 1
        last_ring = self.latitude_bands - 1
        for j in range(self.longitude_segments):
            self._accumulate_mooney_face(
                top_index,
                self._ring_index(1, j),
                self._ring_index(1, j + 1),
                pressure_pa,
            )
            self._accumulate_mooney_face(
                bottom_index,
                self._ring_index(last_ring, j + 1),
                self._ring_index(last_ring, j),
                pressure_pa,
            )
            self._accumulate_edge_strain_stat(top_index, self._ring_index(1, j))
            self._accumulate_edge_strain_stat(bottom_index, self._ring_index(last_ring, j))

        for ring, j in ti.ndrange(self.latitude_bands - 2, self.longitude_segments):
            current_ring = ring + 1
            next_ring = current_ring + 1
            a = self._ring_index(current_ring, j)
            b = self._ring_index(current_ring, j + 1)
            c = self._ring_index(next_ring, j + 1)
            d = self._ring_index(next_ring, j)
            self._accumulate_mooney_face(a, d, c, pressure_pa)
            self._accumulate_mooney_face(a, c, b, pressure_pa)

        for ring, j in ti.ndrange(self.latitude_bands - 1, self.longitude_segments):
            current_ring = ring + 1
            self._accumulate_edge_strain_stat(
                self._ring_index(current_ring, j),
                self._ring_index(current_ring, j + 1),
            )

        for ring, j in ti.ndrange(self.latitude_bands - 2, self.longitude_segments):
            current_ring = ring + 1
            next_ring = current_ring + 1
            self._accumulate_edge_strain_stat(
                self._ring_index(current_ring, j),
                self._ring_index(next_ring, j),
            )
            self._accumulate_edge_strain_stat(
                self._ring_index(current_ring, j),
                self._ring_index(next_ring, j + 1),
            )

    @ti.func
    def _particle_grid_coordinate(self, p):
        return ti.Vector(
            [
                (self.x[p].x - self.bounds_min[0]) / self.dx[0],
                (self.x[p].y - self.bounds_min[1]) / self.dx[1],
                (self.x[p].z - self.bounds_min[2]) / self.dx[2],
            ]
        )

    @ti.func
    def _particle_grid_out_of_bounds(self, coord):
        out_of_bounds = 0
        if coord.x < 0.0 or coord.x > ti.cast(self.nx - 1, ti.f32):
            out_of_bounds = 1
        if coord.y < 0.0 or coord.y > ti.cast(self.ny - 1, ti.f32):
            out_of_bounds = 1
        if coord.z < 0.0 or coord.z > ti.cast(self.nz - 1, ti.f32):
            out_of_bounds = 1
        return out_of_bounds

    @ti.func
    def _scatter_particle(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        momentum = self.mass_kg[p] * self.v[p]
        total_force = self.internal_force_n[p] + self.external_force_n[p]
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (fx.x if ox == 1 else 1.0 - fx.x)
                * (fx.y if oy == 1 else 1.0 - fx.y)
                * (fx.z if oz == 1 else 1.0 - fx.z)
            )
            node = (base.x + ox, base.y + oy, base.z + oz)
            ti.atomic_add(self.grid_mass_kg[node], weight * self.mass_kg[p])
            ti.atomic_add(self.grid_velocity_mps[node].x, weight * momentum.x)
            ti.atomic_add(self.grid_velocity_mps[node].y, weight * momentum.y)
            ti.atomic_add(self.grid_velocity_mps[node].z, weight * momentum.z)
            ti.atomic_add(self.grid_force_n[node].x, weight * total_force.x)
            ti.atomic_add(self.grid_force_n[node].y, weight * total_force.y)
            ti.atomic_add(self.grid_force_n[node].z, weight * total_force.z)

    @ti.func
    def _interpolate_grid_velocity(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        velocity = ti.Vector([0.0, 0.0, 0.0])
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (fx.x if ox == 1 else 1.0 - fx.x)
                * (fx.y if oy == 1 else 1.0 - fx.y)
                * (fx.z if oz == 1 else 1.0 - fx.z)
            )
            node = (base.x + ox, base.y + oy, base.z + oz)
            velocity += weight * self.grid_velocity_mps[node]
        return velocity

    @ti.func
    def _interpolate_grid_velocity_delta(self, p):
        coord = self._particle_grid_coordinate(p)
        base = ti.Vector(
            [
                ti.min(ti.max(ti.cast(ti.floor(coord.x), ti.i32), 0), self.nx - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.y), ti.i32), 0), self.ny - 2),
                ti.min(ti.max(ti.cast(ti.floor(coord.z), ti.i32), 0), self.nz - 2),
            ]
        )
        fx = ti.Vector(
            [
                ti.min(ti.max(coord.x - ti.cast(base.x, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.y - ti.cast(base.y, ti.f32), 0.0), 1.0),
                ti.min(ti.max(coord.z - ti.cast(base.z, ti.f32), 0.0), 1.0),
            ]
        )
        delta = ti.Vector([0.0, 0.0, 0.0])
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (fx.x if ox == 1 else 1.0 - fx.x)
                * (fx.y if oy == 1 else 1.0 - fx.y)
                * (fx.z if oz == 1 else 1.0 - fx.z)
            )
            node = (base.x + ox, base.y + oy, base.z + oz)
            delta += weight * (
                self.grid_velocity_mps[node] - self.grid_velocity_before_force_mps[node]
            )
        return delta

    @ti.kernel
    def _step_kernel(
        self,
        dt_s: ti.f32,
        pressure_pa: ti.f32,
        velocity_damping: ti.f32,
        flip_blend: ti.f32,
        body_acceleration_x_mps2: ti.f32,
        body_acceleration_y_mps2: ti.f32,
        body_acceleration_z_mps2: ti.f32,
    ):
        for i, j, k in self.grid_mass_kg:
            self.grid_mass_kg[i, j, k] = 0.0
            self.grid_velocity_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_velocity_before_force_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_force_n[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self._clear_reports()
        self._compute_surface_forces(pressure_pa)
        body_acceleration = ti.Vector(
            [
                body_acceleration_x_mps2,
                body_acceleration_y_mps2,
                body_acceleration_z_mps2,
            ]
        )

        for p in range(self.particle_count):
            self.external_force_n[p] += self.mass_kg[p] * body_acceleration
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) != 0:
                ti.atomic_add(self.report_grid_out_of_bounds_particle_count[None], 1)
            else:
                self._scatter_particle(p)
            self._atomic_add_vector(self.report_internal_force_sum_n, self.internal_force_n[p])
            ti.atomic_add(
                self.report_internal_force_square_sum_n2[None],
                self.internal_force_n[p].dot(self.internal_force_n[p]),
            )
            self._atomic_add_vector(
                self.report_particle_momentum_kg_mps,
                self.mass_kg[p] * self.v[p],
            )
            self._atomic_add_vector(
                self.report_total_force_n,
                self.internal_force_n[p] + self.external_force_n[p],
            )
            total_force = self.internal_force_n[p] + self.external_force_n[p]
            ti.atomic_add(self.report_total_mass_kg[None], self.mass_kg[p])
            ti.atomic_add(
                self.report_momentum_square_sum[None],
                (self.mass_kg[p] * self.v[p]).dot(self.mass_kg[p] * self.v[p]),
            )
            force_impulse = dt_s * total_force
            ti.atomic_add(
                self.report_force_impulse_square_sum[None],
                force_impulse.dot(force_impulse),
            )

        for i, j, k in self.grid_mass_kg:
            mass = self.grid_mass_kg[i, j, k]
            if mass > 1.0e-20:
                velocity = self.grid_velocity_mps[i, j, k] / mass
                self.grid_velocity_before_force_mps[i, j, k] = velocity
                velocity += dt_s * self.grid_force_n[i, j, k] / mass
                transfer_velocity = velocity
                velocity *= velocity_damping
                self.grid_velocity_mps[i, j, k] = velocity
                self._atomic_add_vector(
                    self.report_transfer_grid_momentum_kg_mps,
                    mass * transfer_velocity,
                )
                self._atomic_add_vector(self.report_grid_momentum_kg_mps, mass * velocity)
                ti.atomic_add(self.report_active_grid_nodes[None], 1)
                ti.atomic_max(self.report_max_speed_mps[None], velocity.norm())

        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                pic_v = self._interpolate_grid_velocity(p)
                flip_v = self.v[p] + self._interpolate_grid_velocity_delta(p)
                new_v = (1.0 - flip_blend) * pic_v + flip_blend * flip_v
                self.v[p] = new_v
                self.x[p] += dt_s * new_v
                self._atomic_add_vector(self.report_current_center_sum_m, self.x[p])
                self._atomic_add_vector(self.report_radial_rest_center_sum_m, self.rest_x[p])
                ti.atomic_add(self.report_radial_center_count[None], 1)
            else:
                self.x[p] += dt_s * self.v[p]
                ti.atomic_max(self.report_max_speed_mps[None], self.v[p].norm())

        radial_center_count = ti.max(ti.cast(self.report_radial_center_count[None], ti.f32), 1.0)
        current_center = self.report_current_center_sum_m[None] / radial_center_count
        rest_center = self.report_radial_rest_center_sum_m[None] / radial_center_count
        for p in range(self.particle_count):
            coord = self._particle_grid_coordinate(p)
            if self._particle_grid_out_of_bounds(coord) == 0:
                rest_radius = (self.rest_x[p] - rest_center).norm()
                current_radius = (self.x[p] - current_center).norm()
                radial_stretch = current_radius / ti.max(rest_radius, 1.0e-12)
                ti.atomic_add(self.report_radial_stretch_sum[None], radial_stretch)
                ti.atomic_add(self.report_radial_stretch_count[None], 1)
                ti.atomic_max(
                    self.report_max_radial_stretch_error[None],
                    ti.abs(radial_stretch - 1.0),
                )

        expected_grid_momentum = (
            self.report_particle_momentum_kg_mps[None]
            + dt_s * self.report_total_force_n[None]
        )
        denominator = ti.max(
            expected_grid_momentum.norm(),
            ti.sqrt(self.report_momentum_square_sum[None]),
            ti.sqrt(self.report_force_impulse_square_sum[None]),
            1.0e-20,
        )
        self.report_transfer_relative_error[None] = (
            self.report_transfer_grid_momentum_kg_mps[None] - expected_grid_momentum
        ).norm() / denominator
        self._update_particle_surface_normals()
        self._pack_report_snapshot()

    def step(
        self,
        *,
        dt_s: float,
        pressure_pa: float,
        velocity_damping: float = 1.0,
        flip_blend: float = 0.95,
        body_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> UvMooneyShellMpmReport:
        body_acceleration = _vector3(body_acceleration_mps2, "body_acceleration_mps2")
        self._step_kernel(
            float(dt_s),
            float(pressure_pa),
            float(velocity_damping),
            _validate_flip_blend(flip_blend),
            float(body_acceleration[0]),
            float(body_acceleration[1]),
            float(body_acceleration[2]),
        )
        return self.report()

    def report(self) -> UvMooneyShellMpmReport:
        snapshot = self.report_host_snapshot.to_numpy()
        values = snapshot[:10]
        counts = snapshot[10:14]
        self.last_report_host_reads = 1
        _raise_if_all_particles_out_of_bounds(int(self.particle_count), int(counts[2]))
        force_l2 = float(values[0]) ** 0.5
        return UvMooneyShellMpmReport(
            particle_count=self.particle_count,
            edge_count=int(counts[0]),
            active_grid_nodes=int(counts[1]),
            grid_out_of_bounds_particle_count=int(counts[2]),
            mean_radial_stretch=float(values[4]) / max(int(counts[3]), 1),
            max_radial_stretch_error=float(values[5]),
            max_edge_strain=float(values[6]),
            max_speed_mps=float(values[7]),
            total_mass_kg=float(values[8]),
            internal_force_rms_n=(force_l2 * force_l2 / max(self.particle_count, 1)) ** 0.5,
            net_internal_force_relative_error=(
                (float(values[1]) ** 2 + float(values[2]) ** 2 + float(values[3]) ** 2) ** 0.5
                / max(force_l2, 1.0e-20)
            ),
            transfer_relative_error=float(values[9]),
        )
