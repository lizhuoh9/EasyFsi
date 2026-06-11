from dataclasses import dataclass
import math

import numpy as np
import taichi as ti

from .runtime import TaichiRuntimeConfig, init_taichi


@dataclass(frozen=True)
class TriSurfaceDiagnosticReport:
    face_count: int
    pressure_traction_force_n: tuple[float, float, float]
    pressure_traction_abs_force_n: float
    pressure_traction_face_count: int | None
    pressure_traction_area_m2: float
    projected_ibm_residual_mps: float
    projected_ibm_residual_l2_mps: float
    projected_ibm_sample_count: int
    invalid_probe_count: int
    valid_probe_fraction: float
    invalid_probe_area_m2: float = 0.0
    invalid_probe_volume_source_m3s: float = 0.0
    force_sample_count: int | None = None
    force_invalid_probe_count: int | None = None
    force_valid_probe_count: int | None = None
    force_valid_probe_fraction: float = math.nan
    grid_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    primary_fluid_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    secondary_fluid_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    constraint_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    primary_pressure_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    secondary_pressure_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    primary_constraint_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    secondary_constraint_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    viscous_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    fluid_stress_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    primary_viscous_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    secondary_viscous_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    primary_fluid_stress_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    secondary_fluid_stress_traction_force_n: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    volume_source_m3s: float = math.nan
    primary_volume_source_m3s: float = math.nan
    secondary_volume_source_m3s: float = math.nan
    active_force_cells: int | None = None


@dataclass(frozen=True)
class TriSurfaceForcePairReport:
    primary_fluid_force_n: tuple[float, float, float]
    secondary_fluid_force_n: tuple[float, float, float]
    force_sample_count: int = 0
    force_invalid_probe_count: int = 0
    force_valid_probe_count: int = 0
    force_valid_probe_fraction: float = math.nan
    invalid_probe_count: int = 0
    valid_probe_fraction: float = math.nan
    invalid_probe_area_m2: float = 0.0
    invalid_probe_volume_source_m3s: float = 0.0


@ti.data_oriented
class TriSurfaceRegionDiagnostics:
    """GPU diagnostics for arbitrary triangle surface regions.

    This class intentionally handles only reusable region-level diagnostics:
    pressure traction, projected no-slip residual, and probe validity. Case
    selection, mesh loading, and monitor interpretation stay outside the core.
    """

    def __init__(self, face_capacity: int, runtime: TaichiRuntimeConfig | None = None):
        init_taichi(runtime)
        if face_capacity <= 0:
            raise ValueError("face_capacity must be positive")
        self.face_capacity = int(face_capacity)
        self.face_count = 0

        self.centroid_m = ti.Vector.field(3, dtype=ti.f32, shape=self.face_capacity)
        self.rest_centroid_m = ti.Vector.field(3, dtype=ti.f32, shape=self.face_capacity)
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=self.face_capacity)
        self.area_m2 = ti.field(dtype=ti.f32, shape=self.face_capacity)
        self.region_id = ti.field(dtype=ti.i32, shape=self.face_capacity)

        self.report_pressure_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_pressure_abs_force_n = ti.field(dtype=ti.f32, shape=())
        self.report_pressure_area_m2 = ti.field(dtype=ti.f32, shape=())
        self.report_pressure_face_count = ti.field(dtype=ti.i32, shape=())
        self.report_residual_square_sum = ti.field(dtype=ti.f32, shape=())
        self.report_residual_max = ti.field(dtype=ti.f32, shape=())
        self.report_sample_count = ti.field(dtype=ti.i32, shape=())
        self.report_invalid_probe_count = ti.field(dtype=ti.i32, shape=())
        self.report_invalid_probe_area_m2 = ti.field(dtype=ti.f32, shape=())
        self.report_invalid_probe_volume_source_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_fluid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_fluid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_constraint_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_pressure_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_pressure_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_constraint_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_constraint_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_viscous_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_fluid_stress_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_viscous_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_viscous_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_fluid_stress_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_fluid_stress_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_volume_source_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_primary_volume_source_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_secondary_volume_source_m3s = ti.field(dtype=ti.f32, shape=())
        self.report_active_force_cells = ti.field(dtype=ti.i32, shape=())
        self.report_float_snapshot_a = ti.Vector.field(32, dtype=ti.f32, shape=())
        self.report_float_snapshot_b = ti.Vector.field(22, dtype=ti.f32, shape=())
        self.report_force_pair_snapshot = ti.Vector.field(10, dtype=ti.f32, shape=())
        self.report_count_snapshot = ti.Vector.field(4, dtype=ti.i32, shape=())
        self.report_force_cell_index_capacity = max(1, self.face_capacity * 8)
        self.report_force_cell_index_count = ti.field(dtype=ti.i32, shape=())
        self.report_force_cell_indices = ti.field(
            dtype=ti.i64,
            shape=self.report_force_cell_index_capacity,
        )
        self.report_force_cell_force_n = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=self.report_force_cell_index_capacity,
        )
        self.force_impulse_n_s = ti.Vector.field(6, dtype=ti.f64, shape=())
        self.last_report_host_reads = 0
        self.last_force_impulse_host_reads = 0

    def load_faces(
        self,
        centroid_m: np.ndarray,
        normal: np.ndarray,
        area_m2: np.ndarray,
        region_id: np.ndarray,
    ) -> None:
        centroids = np.asarray(centroid_m, dtype=np.float32)
        normals = np.asarray(normal, dtype=np.float32)
        areas = np.asarray(area_m2, dtype=np.float32)
        regions = np.asarray(region_id, dtype=np.int32)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError("centroid_m must have shape (n, 3)")
        if normals.shape != centroids.shape:
            raise ValueError("normal must have the same shape as centroid_m")
        if areas.shape != (centroids.shape[0],):
            raise ValueError("area_m2 must have shape (n,)")
        if regions.shape != (centroids.shape[0],):
            raise ValueError("region_id must have shape (n,)")
        if centroids.shape[0] > self.face_capacity:
            raise ValueError("face count exceeds face_capacity")
        if not np.all(np.isfinite(centroids)):
            raise ValueError("centroid_m must contain finite values")
        if not np.all(np.isfinite(normals)):
            raise ValueError("normal must contain finite values")
        if not np.all(np.isfinite(areas)):
            raise ValueError("area_m2 must contain finite values")
        if np.any(areas < 0.0):
            raise ValueError("area_m2 must be non-negative")
        normal_norms = np.linalg.norm(normals, axis=1)
        if np.any(normal_norms <= 1.0e-12):
            raise ValueError("normal vectors must be non-degenerate")
        normals = normals / normal_norms[:, None]

        self.face_count = int(centroids.shape[0])
        for i in range(self.face_count):
            self.centroid_m[i] = (
                float(centroids[i, 0]),
                float(centroids[i, 1]),
                float(centroids[i, 2]),
            )
            self.rest_centroid_m[i] = (
                float(centroids[i, 0]),
                float(centroids[i, 1]),
                float(centroids[i, 2]),
            )
            self.normal[i] = (
                float(normals[i, 0]),
                float(normals[i, 1]),
                float(normals[i, 2]),
            )
            self.area_m2[i] = float(areas[i])
            self.region_id[i] = int(regions[i])

    @staticmethod
    def _velocity_tuple(
        velocity_mps: tuple[float, float, float],
        name: str,
    ) -> tuple[float, float, float]:
        if len(velocity_mps) != 3:
            raise ValueError(f"{name} must contain exactly 3 components")
        return (float(velocity_mps[0]), float(velocity_mps[1]), float(velocity_mps[2]))

    @staticmethod
    def _non_negative_float(value: object, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return result

    @ti.kernel
    def _update_region_offsets_kernel(
        self,
        face_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_offset_x_m: ti.f32,
        primary_offset_y_m: ti.f32,
        primary_offset_z_m: ti.f32,
        secondary_offset_x_m: ti.f32,
        secondary_offset_y_m: ti.f32,
        secondary_offset_z_m: ti.f32,
    ):
        for face in range(face_count):
            region = self.region_id[face]
            offset = ti.Vector([0.0, 0.0, 0.0])
            if region == primary_region_id:
                offset = ti.Vector([primary_offset_x_m, primary_offset_y_m, primary_offset_z_m])
            elif region == secondary_region_id:
                offset = ti.Vector([secondary_offset_x_m, secondary_offset_y_m, secondary_offset_z_m])
            rest = self.rest_centroid_m[face]
            self.centroid_m[face] = rest + offset

    def update_region_offsets(
        self,
        *,
        primary_region_id: int,
        secondary_region_id: int,
        primary_offset_m: tuple[float, float, float],
        secondary_offset_m: tuple[float, float, float],
    ) -> None:
        primary_offset = self._velocity_tuple(primary_offset_m, "primary_offset_m")
        secondary_offset = self._velocity_tuple(secondary_offset_m, "secondary_offset_m")
        self._update_region_offsets_kernel(
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            float(primary_offset[0]),
            float(primary_offset[1]),
            float(primary_offset[2]),
            float(secondary_offset[0]),
            float(secondary_offset[1]),
            float(secondary_offset[2]),
        )

    @ti.func
    def _sample_fluid_velocity_trilinear(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx,
        ny,
        nz,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = ti.Vector([0.0, 0.0, 0.0])
        fluid_weight = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * velocity_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _sample_pressure_trilinear(
        self,
        pressure_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx,
        ny,
        nz,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        value = 0.0
        fluid_weight = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            if obstacle_field[i0 + oi, j0 + oj, k0 + ok] == 0:
                value += weight * pressure_field[i0 + oi, j0 + oj, k0 + ok]
                fluid_weight += weight
        if fluid_weight > 1.0e-12:
            value /= fluid_weight
        return value, fluid_weight

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value,
        faces: ti.template(),
        centers: ti.template(),
        count: ti.i32,
    ):
        coordinate = 0.0
        if value <= centers[0]:
            half_width = ti.max(centers[0] - faces[0], 1.0e-18)
            coordinate = -0.5 * (centers[0] - value) / half_width
        elif value >= centers[count - 1]:
            half_width = ti.max(faces[count] - centers[count - 1], 1.0e-18)
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (value - centers[count - 1]) / half_width
        else:
            lower = 0
            upper = count - 1
            while upper - lower > 1:
                middle = (lower + upper) // 2
                if value >= centers[middle]:
                    lower = middle
                else:
                    upper = middle
            upper = ti.min(lower + 1, count - 1)
            distance = ti.max(centers[upper] - centers[lower], 1.0e-18)
            coordinate = ti.cast(lower, ti.f32) + (value - centers[lower]) / distance
        return coordinate

    @ti.func
    def _grid_coordinate_from_fields(
        self,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        return ti.Vector(
            [
                self._axis_grid_coordinate_device(position.x, cell_face_x_m, cell_center_x_m, nx),
                self._axis_grid_coordinate_device(position.y, cell_face_y_m, cell_center_y_m, ny),
                self._axis_grid_coordinate_device(position.z, cell_face_z_m, cell_center_z_m, nz),
            ]
        )

    @ti.func
    def _local_normal_probe_distance_m(
        self,
        normal,
        gx,
        gy,
        gz,
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        i0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        j0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        k0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        tx = ti.min(ti.max(gx - ti.cast(i0, ti.f32), 0.0), 1.0)
        ty = ti.min(ti.max(gy - ti.cast(j0, ti.f32), 0.0), 1.0)
        tz = ti.min(ti.max(gz - ti.cast(k0, ti.f32), 0.0), 1.0)
        local_width_x_m = 0.0
        local_width_y_m = 0.0
        local_width_z_m = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            wx = 1.0 - tx if oi == 0 else tx
            wy = 1.0 - ty if oj == 0 else ty
            wz = 1.0 - tz if ok == 0 else tz
            weight = wx * wy * wz
            local_width_x_m += weight * cell_width_x_m[i0 + oi]
            local_width_y_m += weight * cell_width_y_m[j0 + oj]
            local_width_z_m += weight * cell_width_z_m[k0 + ok]
        unit_normal = normal / ti.max(normal.norm(), 1.0e-12)
        inverse_metric_square = (
            (unit_normal.x / ti.max(local_width_x_m, 1.0e-18))
            * (unit_normal.x / ti.max(local_width_x_m, 1.0e-18))
            + (unit_normal.y / ti.max(local_width_y_m, 1.0e-18))
            * (unit_normal.y / ti.max(local_width_y_m, 1.0e-18))
            + (unit_normal.z / ti.max(local_width_z_m, 1.0e-18))
            * (unit_normal.z / ti.max(local_width_z_m, 1.0e-18))
        )
        return 1.0 / ti.sqrt(ti.max(inverse_metric_square, 1.0e-18))

    @ti.func
    def _clamp_position_to_bounds(
        self,
        position,
        bounds_min_x: ti.f32,
        bounds_min_y: ti.f32,
        bounds_min_z: ti.f32,
        bounds_max_x: ti.f32,
        bounds_max_y: ti.f32,
        bounds_max_z: ti.f32,
    ):
        return ti.Vector(
            [
                ti.min(ti.max(position.x, bounds_min_x), bounds_max_x),
                ti.min(ti.max(position.y, bounds_min_y), bounds_max_y),
                ti.min(ti.max(position.z, bounds_min_z), bounds_max_z),
            ]
        )

    @ti.func
    def _sample_velocity_gradient(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx,
        ny,
        nz,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
    ):
        ix0 = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        iy0 = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        iz0 = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        dx = ti.max(cell_center_x_m[ix1] - cell_center_x_m[ix0], 1.0e-18)
        dy = ti.max(cell_center_y_m[iy1] - cell_center_y_m[iy0], 1.0e-18)
        dz = ti.max(cell_center_z_m[iz1] - cell_center_z_m[iz0], 1.0e-18)
        vx0, wx0 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, ti.cast(ix0, ti.f32), gy, gz, nx, ny, nz
        )
        vx1, wx1 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, ti.cast(ix1, ti.f32), gy, gz, nx, ny, nz
        )
        vy0, wy0 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, gx, ti.cast(iy0, ti.f32), gz, nx, ny, nz
        )
        vy1, wy1 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, gx, ti.cast(iy1, ti.f32), gz, nx, ny, nz
        )
        vz0, wz0 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, gx, gy, ti.cast(iz0, ti.f32), nx, ny, nz
        )
        vz1, wz1 = self._sample_fluid_velocity_trilinear(
            velocity_field, obstacle_field, gx, gy, ti.cast(iz1, ti.f32), nx, ny, nz
        )
        dvdx = ti.Vector([0.0, 0.0, 0.0])
        dvdy = ti.Vector([0.0, 0.0, 0.0])
        dvdz = ti.Vector([0.0, 0.0, 0.0])
        if wx0 > 1.0e-12 and wx1 > 1.0e-12:
            dvdx = (vx1 - vx0) / dx
        if wy0 > 1.0e-12 and wy1 > 1.0e-12:
            dvdy = (vy1 - vy0) / dy
        if wz0 > 1.0e-12 and wz1 > 1.0e-12:
            dvdz = (vz1 - vz0) / dz
        return ti.Matrix(
            [
                [dvdx.x, dvdy.x, dvdz.x],
                [dvdx.y, dvdy.y, dvdz.y],
                [dvdx.z, dvdy.z, dvdz.z],
            ]
        )

    @ti.func
    def _atomic_add_vector(self, field, value):
        ti.atomic_add(field[None].x, value.x)
        ti.atomic_add(field[None].y, value.y)
        ti.atomic_add(field[None].z, value.z)

    @ti.func
    def _append_report_force_cell_index(self, linear_index: ti.i64, force_n):
        slot = ti.atomic_add(self.report_force_cell_index_count[None], 1)
        if slot < ti.static(self.report_force_cell_index_capacity):
            self.report_force_cell_indices[slot] = linear_index + 1
            self.report_force_cell_force_n[slot] = force_n

    @ti.func
    def _pack_report_only_active_force_cell_count(self):
        active_count = 0
        stored_count = ti.min(
            self.report_force_cell_index_count[None],
            ti.static(self.report_force_cell_index_capacity),
        )
        for index in range(stored_count):
            linear_index = self.report_force_cell_indices[index]
            duplicate = False
            for previous in range(index):
                if self.report_force_cell_indices[previous] == linear_index:
                    duplicate = True
            if linear_index != 0 and not duplicate:
                force_sum = ti.Vector([0.0, 0.0, 0.0])
                for candidate in range(stored_count):
                    if self.report_force_cell_indices[candidate] == linear_index:
                        force_sum += self.report_force_cell_force_n[candidate]
                if force_sum.norm() > 0.0:
                    active_count += 1
        self.report_active_force_cells[None] = active_count

    @ti.func
    def _pack_full_report_snapshot(self):
        self.report_float_snapshot_a[None] = ti.Vector(
            [
                self.report_pressure_force_n[None].x,
                self.report_pressure_force_n[None].y,
                self.report_pressure_force_n[None].z,
                self.report_pressure_abs_force_n[None],
                self.report_pressure_area_m2[None],
                self.report_residual_square_sum[None],
                self.report_residual_max[None],
                self.report_invalid_probe_area_m2[None],
                self.report_invalid_probe_volume_source_m3s[None],
                self.report_grid_force_n[None].x,
                self.report_grid_force_n[None].y,
                self.report_grid_force_n[None].z,
                self.report_primary_fluid_force_n[None].x,
                self.report_primary_fluid_force_n[None].y,
                self.report_primary_fluid_force_n[None].z,
                self.report_secondary_fluid_force_n[None].x,
                self.report_secondary_fluid_force_n[None].y,
                self.report_secondary_fluid_force_n[None].z,
                self.report_constraint_force_n[None].x,
                self.report_constraint_force_n[None].y,
                self.report_constraint_force_n[None].z,
                self.report_primary_pressure_force_n[None].x,
                self.report_primary_pressure_force_n[None].y,
                self.report_primary_pressure_force_n[None].z,
                self.report_secondary_pressure_force_n[None].x,
                self.report_secondary_pressure_force_n[None].y,
                self.report_secondary_pressure_force_n[None].z,
                self.report_primary_constraint_force_n[None].x,
                self.report_primary_constraint_force_n[None].y,
                self.report_primary_constraint_force_n[None].z,
                self.report_secondary_constraint_force_n[None].x,
                self.report_secondary_constraint_force_n[None].y,
            ]
        )
        self.report_float_snapshot_b[None] = ti.Vector(
            [
                self.report_secondary_constraint_force_n[None].z,
                self.report_viscous_force_n[None].x,
                self.report_viscous_force_n[None].y,
                self.report_viscous_force_n[None].z,
                self.report_fluid_stress_force_n[None].x,
                self.report_fluid_stress_force_n[None].y,
                self.report_fluid_stress_force_n[None].z,
                self.report_primary_viscous_force_n[None].x,
                self.report_primary_viscous_force_n[None].y,
                self.report_primary_viscous_force_n[None].z,
                self.report_secondary_viscous_force_n[None].x,
                self.report_secondary_viscous_force_n[None].y,
                self.report_secondary_viscous_force_n[None].z,
                self.report_primary_fluid_stress_force_n[None].x,
                self.report_primary_fluid_stress_force_n[None].y,
                self.report_primary_fluid_stress_force_n[None].z,
                self.report_secondary_fluid_stress_force_n[None].x,
                self.report_secondary_fluid_stress_force_n[None].y,
                self.report_secondary_fluid_stress_force_n[None].z,
                self.report_volume_source_m3s[None],
                self.report_primary_volume_source_m3s[None],
                self.report_secondary_volume_source_m3s[None],
            ]
        )
        self.report_count_snapshot[None] = ti.Vector(
            [
                self.report_pressure_face_count[None],
                self.report_sample_count[None],
                self.report_invalid_probe_count[None],
                self.report_active_force_cells[None],
            ]
        )

    @ti.func
    def _pack_force_pair_snapshot(self):
        self.report_force_pair_snapshot[None] = ti.Vector(
            [
                self.report_primary_fluid_force_n[None].x,
                self.report_primary_fluid_force_n[None].y,
                self.report_primary_fluid_force_n[None].z,
                self.report_secondary_fluid_force_n[None].x,
                self.report_secondary_fluid_force_n[None].y,
                self.report_secondary_fluid_force_n[None].z,
                ti.cast(self.report_sample_count[None], ti.f32),
                ti.cast(self.report_invalid_probe_count[None], ti.f32),
                self.report_invalid_probe_area_m2[None],
                self.report_invalid_probe_volume_source_m3s[None],
            ]
        )

    @ti.func
    def _pack_report_snapshot(self):
        self._pack_full_report_snapshot()
        self._pack_force_pair_snapshot()

    @ti.kernel
    def reset_force_impulse_accumulator(self):
        self.force_impulse_n_s[None] = ti.Vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dt=ti.f64
        )

    @ti.kernel
    def accumulate_force_impulse(self, dt_s: ti.f64):
        self.force_impulse_n_s[None][0] += (
            ti.cast(self.report_primary_fluid_force_n[None].x, ti.f64) * dt_s
        )
        self.force_impulse_n_s[None][1] += (
            ti.cast(self.report_primary_fluid_force_n[None].y, ti.f64) * dt_s
        )
        self.force_impulse_n_s[None][2] += (
            ti.cast(self.report_primary_fluid_force_n[None].z, ti.f64) * dt_s
        )
        self.force_impulse_n_s[None][3] += (
            ti.cast(self.report_secondary_fluid_force_n[None].x, ti.f64) * dt_s
        )
        self.force_impulse_n_s[None][4] += (
            ti.cast(self.report_secondary_fluid_force_n[None].y, ti.f64) * dt_s
        )
        self.force_impulse_n_s[None][5] += (
            ti.cast(self.report_secondary_fluid_force_n[None].z, ti.f64) * dt_s
        )

    @ti.func
    def _scatter_force_to_grid(
        self,
        force_field,
        obstacle_field,
        force_n,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight

        actual_force = ti.Vector([0.0, 0.0, 0.0])
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    node_force = force_n * renormalized_weight
                    cell_volume_m3 = cell_width_x_m[ii] * cell_width_y_m[jj] * cell_width_z_m[kk]
                    force_density = node_force / ti.max(cell_volume_m3, 1.0e-18)
                    ti.atomic_add(force_field[ii, jj, kk].x, force_density.x)
                    ti.atomic_add(force_field[ii, jj, kk].y, force_density.y)
                    ti.atomic_add(force_field[ii, jj, kk].z, force_density.z)
                    actual_force += node_force
                    self._atomic_add_vector(self.report_grid_force_n, node_force)
        return actual_force

    @ti.func
    def _scatter_force_to_grid_report_only(
        self,
        obstacle_field,
        force_n,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight

        actual_force = ti.Vector([0.0, 0.0, 0.0])
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    node_force = force_n * renormalized_weight
                    actual_force += node_force
                    self._atomic_add_vector(self.report_grid_force_n, node_force)
                    if node_force.norm() > 0.0:
                        linear_index = (
                            (ti.cast(ii, ti.i64) * ti.cast(ny, ti.i64) + ti.cast(jj, ti.i64))
                            * ti.cast(nz, ti.i64)
                            + ti.cast(kk, ti.i64)
                        )
                        self._append_report_force_cell_index(linear_index, node_force)
        return actual_force

    @ti.func
    def _scatter_volume_source_to_grid(
        self,
        volume_source_field,
        obstacle_field,
        source_m3s,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight

        actual_source_m3s = 0.0
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    node_source_m3s = source_m3s * renormalized_weight
                    cell_volume_m3 = cell_width_x_m[ii] * cell_width_y_m[jj] * cell_width_z_m[kk]
                    ti.atomic_add(
                        volume_source_field[ii, jj, kk],
                        node_source_m3s / ti.max(cell_volume_m3, 1.0e-18),
                    )
                    actual_source_m3s += node_source_m3s
        return actual_source_m3s

    @ti.func
    def _scatter_volume_source_to_grid_report_only(
        self,
        obstacle_field,
        source_m3s,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight

        actual_source_m3s = 0.0
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    actual_source_m3s += source_m3s * renormalized_weight
        return actual_source_m3s

    @ti.func
    def _scatter_velocity_constraint_to_grid(
        self,
        target_sum_field,
        weight_field,
        obstacle_field,
        target_velocity,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    ti.atomic_add(target_sum_field[ii, jj, kk].x, target_velocity.x * renormalized_weight)
                    ti.atomic_add(target_sum_field[ii, jj, kk].y, target_velocity.y * renormalized_weight)
                    ti.atomic_add(target_sum_field[ii, jj, kk].z, target_velocity.z * renormalized_weight)
                    ti.atomic_add(weight_field[ii, jj, kk], renormalized_weight)
        return valid_weight_sum

    @ti.func
    def _scatter_velocity_constraint_region_to_grid(
        self,
        target_sum_field,
        weight_field,
        obstacle_field,
        target_velocity,
        position,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx,
        ny,
        nz,
    ):
        grid_coordinate = self._grid_coordinate_from_fields(
            position,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            nx,
            ny,
            nz,
        )
        gx = grid_coordinate.x
        gy = grid_coordinate.y
        gz = grid_coordinate.z
        base_i = ti.min(ti.max(ti.floor(gx, ti.i32), 0), nx - 2)
        base_j = ti.min(ti.max(ti.floor(gy, ti.i32), 0), ny - 2)
        base_k = ti.min(ti.max(ti.floor(gz, ti.i32), 0), nz - 2)
        fx = ti.min(ti.max(gx - ti.cast(base_i, ti.f32), 0.0), 1.0)
        fy = ti.min(ti.max(gy - ti.cast(base_j, ti.f32), 0.0), 1.0)
        fz = ti.min(ti.max(gz - ti.cast(base_k, ti.f32), 0.0), 1.0)
        valid_weight_sum = 0.0
        for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
            ii = base_i + oi
            jj = base_j + oj
            kk = base_k + ok
            wx = 1.0 - fx if oi == 0 else fx
            wy = 1.0 - fy if oj == 0 else fy
            wz = 1.0 - fz if ok == 0 else fz
            weight = wx * wy * wz
            if obstacle_field[ii, jj, kk] == 0:
                valid_weight_sum += weight
        if valid_weight_sum > 0.0:
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                ii = base_i + oi
                jj = base_j + oj
                kk = base_k + ok
                wx = 1.0 - fx if oi == 0 else fx
                wy = 1.0 - fy if oj == 0 else fy
                wz = 1.0 - fz if ok == 0 else fz
                weight = wx * wy * wz
                if obstacle_field[ii, jj, kk] == 0:
                    renormalized_weight = weight / valid_weight_sum
                    ti.atomic_add(target_sum_field[ii, jj, kk].x, target_velocity.x * renormalized_weight)
                    ti.atomic_add(target_sum_field[ii, jj, kk].y, target_velocity.y * renormalized_weight)
                    ti.atomic_add(target_sum_field[ii, jj, kk].z, target_velocity.z * renormalized_weight)
                    ti.atomic_add(weight_field[ii, jj, kk], renormalized_weight)

    @ti.kernel
    def _spread_fsi_force_kernel(
        self,
        velocity_field: ti.template(),
        pressure_field: ti.template(),
        force_field: ti.template(),
        volume_source_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        face_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_target_velocity_mps: ti.types.vector(3, ti.f32),
        secondary_target_velocity_mps: ti.types.vector(3, ti.f32),
        probe_distance_m: ti.f32,
        density_kgm3: ti.f32,
        viscosity_pa_s: ti.f32,
        dt_s: ti.f32,
        constraint_force_scale: ti.f32,
        constraint_force_solid_mobility_ratio: ti.f32,
        primary_constraint_force_solid_mobility_ratio: ti.f32,
        secondary_constraint_force_solid_mobility_ratio: ti.f32,
        primary_velocity_target_solid_mobility_ratio: ti.f32,
        secondary_velocity_target_solid_mobility_ratio: ti.f32,
        primary_interface_impedance_force_n: ti.types.vector(3, ti.f32),
        secondary_interface_impedance_force_n: ti.types.vector(3, ti.f32),
        primary_interface_area_m2: ti.f32,
        secondary_interface_area_m2: ti.f32,
        bounds_min_x: ti.f32,
        bounds_min_y: ti.f32,
        bounds_min_z: ti.f32,
        bounds_max_x: ti.f32,
        bounds_max_y: ti.f32,
        bounds_max_z: ti.f32,
        dx: ti.f32,
        dy: ti.f32,
        dz: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        read_full_report: ti.i32,
        write_fields: ti.i32,
        read_force_pair_report: ti.i32,
    ):
        self.report_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_pressure_abs_force_n[None] = 0.0
        self.report_pressure_area_m2[None] = 0.0
        self.report_pressure_face_count[None] = 0
        self.report_residual_square_sum[None] = 0.0
        self.report_residual_max[None] = 0.0
        self.report_sample_count[None] = 0
        self.report_invalid_probe_count[None] = 0
        self.report_invalid_probe_area_m2[None] = 0.0
        self.report_invalid_probe_volume_source_m3s[None] = 0.0
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_volume_source_m3s[None] = 0.0
        self.report_primary_volume_source_m3s[None] = 0.0
        self.report_secondary_volume_source_m3s[None] = 0.0
        self.report_active_force_cells[None] = 0
        self.report_force_cell_index_count[None] = 0
        for face in range(face_count):
            region = self.region_id[face]
            if region == primary_region_id or region == secondary_region_id:
                n = self.normal[face]
                area = self.area_m2[face]
                boundary_velocity = secondary_target_velocity_mps
                region_constraint_force_solid_mobility_ratio = (
                    secondary_constraint_force_solid_mobility_ratio
                )
                region_velocity_target_solid_mobility_ratio = (
                    secondary_velocity_target_solid_mobility_ratio
                )
                if region == primary_region_id:
                    boundary_velocity = primary_target_velocity_mps
                    region_constraint_force_solid_mobility_ratio = (
                        primary_constraint_force_solid_mobility_ratio
                    )
                    region_velocity_target_solid_mobility_ratio = (
                        primary_velocity_target_solid_mobility_ratio
                    )
                interface_impedance_force_on_solid = secondary_interface_impedance_force_n
                interface_area_m2 = secondary_interface_area_m2
                if region == primary_region_id:
                    interface_impedance_force_on_solid = primary_interface_impedance_force_n
                    interface_area_m2 = primary_interface_area_m2

                local_probe_distance_m = 0.0
                if probe_distance_m <= 0.0:
                    centroid_grid_coordinate = self._grid_coordinate_from_fields(
                        self.centroid_m[face],
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    local_probe_distance_m = self._local_normal_probe_distance_m(
                        n,
                        centroid_grid_coordinate.x,
                        centroid_grid_coordinate.y,
                        centroid_grid_coordinate.z,
                        cell_width_x_m,
                        cell_width_y_m,
                        cell_width_z_m,
                        nx,
                        ny,
                        nz,
                    )
                effective_probe_distance_m = probe_distance_m
                if effective_probe_distance_m <= 0.0:
                    effective_probe_distance_m = local_probe_distance_m
                probe = self.centroid_m[face] + n * effective_probe_distance_m
                in_bounds = (
                    probe.x >= bounds_min_x
                    and probe.x <= bounds_max_x
                    and probe.y >= bounds_min_y
                    and probe.y <= bounds_max_y
                    and probe.z >= bounds_min_z
                    and probe.z <= bounds_max_z
                )
                sample_probe = probe
                if not in_bounds:
                    ti.atomic_add(self.report_invalid_probe_count[None], 1)
                    ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                    ti.atomic_add(
                        self.report_invalid_probe_volume_source_m3s[None],
                        boundary_velocity.dot(n) * area,
                    )
                sample_in_bounds = in_bounds
                if sample_in_bounds:
                    grid_coordinate = self._grid_coordinate_from_fields(
                        sample_probe,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    gx = grid_coordinate.x
                    gy = grid_coordinate.y
                    gz = grid_coordinate.z
                    sampled, _sampled_fluid_weight = self._sample_fluid_velocity_trilinear(
                        velocity_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                    )
                    sample_has_fluid_support = _sampled_fluid_weight > 1.0e-12
                    if in_bounds and _sampled_fluid_weight <= 1.0e-12:
                        ti.atomic_add(self.report_invalid_probe_count[None], 1)
                        ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                        ti.atomic_add(
                            self.report_invalid_probe_volume_source_m3s[None],
                            boundary_velocity.dot(n) * area,
                        )
                    sample_area = area if sample_has_fluid_support else 0.0
                    sample_count = ti.cast(sample_has_fluid_support, ti.i32)
                    sampled_pressure, _pressure_fluid_weight = self._sample_pressure_trilinear(
                        pressure_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                    )
                    velocity_gradient = self._sample_velocity_gradient(
                        velocity_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
                    pressure_force_on_solid = -sampled_pressure * n * sample_area
                    viscous_stress = viscosity_pa_s * (velocity_gradient + velocity_gradient.transpose())
                    viscous_force_on_solid = (viscous_stress @ n) * sample_area
                    fluid_stress_force_on_solid = pressure_force_on_solid + viscous_force_on_solid
                    velocity_target_mobility_scale = (
                        1.0 / (1.0 + region_velocity_target_solid_mobility_ratio)
                    )
                    effective_boundary_velocity = sampled + (
                        boundary_velocity - sampled
                    ) * velocity_target_mobility_scale
                    residual_vec = effective_boundary_velocity - sampled
                    residual = residual_vec.norm()
                    control_volume_m3 = sample_area * effective_probe_distance_m
                    constraint_force_mobility_scale = (
                        1.0 / (1.0 + region_constraint_force_solid_mobility_ratio)
                    )
                    constraint_force = (
                        density_kgm3
                        * control_volume_m3
                        * residual_vec
                        / ti.max(dt_s, 1.0e-12)
                        * constraint_force_scale
                        * constraint_force_mobility_scale
                    )
                    stress_action_on_fluid = -fluid_stress_force_on_solid
                    interface_impedance_action_on_fluid = ti.Vector([0.0, 0.0, 0.0])
                    if interface_area_m2 > 1.0e-30:
                        interface_impedance_action_on_fluid = (
                            -interface_impedance_force_on_solid
                            * sample_area
                            / interface_area_m2
                        )
                    total_fluid_force = constraint_force + stress_action_on_fluid
                    total_fluid_force += interface_impedance_action_on_fluid
                    if read_full_report != 0:
                        ti.atomic_add(self.report_pressure_force_n[None].x, pressure_force_on_solid.x)
                        ti.atomic_add(self.report_pressure_force_n[None].y, pressure_force_on_solid.y)
                        ti.atomic_add(self.report_pressure_force_n[None].z, pressure_force_on_solid.z)
                        self._atomic_add_vector(self.report_viscous_force_n, viscous_force_on_solid)
                        self._atomic_add_vector(self.report_fluid_stress_force_n, fluid_stress_force_on_solid)
                        ti.atomic_add(self.report_pressure_abs_force_n[None], ti.abs(sampled_pressure) * sample_area)
                        ti.atomic_add(self.report_pressure_area_m2[None], sample_area)
                        ti.atomic_add(self.report_pressure_face_count[None], sample_count)
                        if region == primary_region_id:
                            self._atomic_add_vector(self.report_primary_pressure_force_n, pressure_force_on_solid)
                            self._atomic_add_vector(self.report_primary_viscous_force_n, viscous_force_on_solid)
                            self._atomic_add_vector(
                                self.report_primary_fluid_stress_force_n,
                                fluid_stress_force_on_solid,
                            )
                        else:
                            self._atomic_add_vector(self.report_secondary_pressure_force_n, pressure_force_on_solid)
                            self._atomic_add_vector(self.report_secondary_viscous_force_n, viscous_force_on_solid)
                            self._atomic_add_vector(
                                self.report_secondary_fluid_stress_force_n,
                                fluid_stress_force_on_solid,
                            )
                    actual_fluid_force = ti.Vector([0.0, 0.0, 0.0])
                    actual_volume_source_m3s = 0.0
                    if write_fields != 0:
                        actual_fluid_force = self._scatter_force_to_grid(
                            force_field,
                            obstacle_field,
                            total_fluid_force,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            cell_width_x_m,
                            cell_width_y_m,
                            cell_width_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        actual_volume_source_m3s = self._scatter_volume_source_to_grid(
                            volume_source_field,
                            obstacle_field,
                            effective_boundary_velocity.dot(n) * sample_area,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            cell_width_x_m,
                            cell_width_y_m,
                            cell_width_z_m,
                            nx,
                            ny,
                            nz,
                        )
                    else:
                        actual_fluid_force = self._scatter_force_to_grid_report_only(
                            obstacle_field,
                            total_fluid_force,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        actual_volume_source_m3s = self._scatter_volume_source_to_grid_report_only(
                            obstacle_field,
                            effective_boundary_velocity.dot(n) * sample_area,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            nx,
                            ny,
                            nz,
                        )
                    if read_full_report != 0:
                        ti.atomic_add(self.report_volume_source_m3s[None], actual_volume_source_m3s)
                    if region == primary_region_id:
                        self._atomic_add_vector(self.report_primary_fluid_force_n, actual_fluid_force)
                        if read_full_report != 0:
                            ti.atomic_add(
                                self.report_primary_volume_source_m3s[None],
                                actual_volume_source_m3s,
                            )
                    else:
                        self._atomic_add_vector(self.report_secondary_fluid_force_n, actual_fluid_force)
                        if read_full_report != 0:
                            ti.atomic_add(
                                self.report_secondary_volume_source_m3s[None],
                                actual_volume_source_m3s,
                            )
                    if read_full_report != 0:
                        self._atomic_add_vector(self.report_constraint_force_n, constraint_force)
                        if region == primary_region_id:
                            self._atomic_add_vector(self.report_primary_constraint_force_n, constraint_force)
                        else:
                            self._atomic_add_vector(self.report_secondary_constraint_force_n, constraint_force)
                        ti.atomic_add(
                            self.report_residual_square_sum[None],
                            residual * residual * ti.cast(sample_has_fluid_support, ti.f32),
                        )
                        ti.atomic_max(
                            self.report_residual_max[None],
                            residual if sample_has_fluid_support else 0.0,
                        )
                    ti.atomic_add(self.report_sample_count[None], sample_count)

        if read_full_report != 0 and write_fields == 0:
            self._pack_report_only_active_force_cell_count()
        if read_full_report != 0:
            self._pack_full_report_snapshot()
        if read_force_pair_report != 0:
            self._pack_force_pair_snapshot()

    @ti.kernel
    def _pack_active_force_cells_from_field_kernel(self, force_field: ti.template()):
        self.report_active_force_cells[None] = 0
        for i, j, k in force_field:
            if force_field[i, j, k].norm() > 0.0:
                ti.atomic_add(self.report_active_force_cells[None], 1)
        self._pack_full_report_snapshot()

    @ti.kernel
    def _spread_fsi_velocity_constraint_kernel(
        self,
        target_sum_field: ti.template(),
        weight_field: ti.template(),
        primary_target_sum_field: ti.template(),
        primary_weight_field: ti.template(),
        secondary_target_sum_field: ti.template(),
        secondary_weight_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        face_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_target_velocity_mps: ti.types.vector(3, ti.f32),
        secondary_target_velocity_mps: ti.types.vector(3, ti.f32),
        probe_distance_m: ti.f32,
        bounds_min_x: ti.f32,
        bounds_min_y: ti.f32,
        bounds_min_z: ti.f32,
        bounds_max_x: ti.f32,
        bounds_max_y: ti.f32,
        bounds_max_z: ti.f32,
        dx: ti.f32,
        dy: ti.f32,
        dz: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        read_full_report: ti.i32,
        write_region_fields: ti.i32,
    ):
        if read_full_report != 0:
            self.report_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_pressure_abs_force_n[None] = 0.0
            self.report_pressure_area_m2[None] = 0.0
            self.report_pressure_face_count[None] = 0
            self.report_residual_square_sum[None] = 0.0
            self.report_residual_max[None] = 0.0
            self.report_sample_count[None] = 0
            self.report_invalid_probe_count[None] = 0
            self.report_invalid_probe_area_m2[None] = 0.0
            self.report_invalid_probe_volume_source_m3s[None] = 0.0
            self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_primary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_secondary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_primary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_secondary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_primary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_secondary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_primary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_secondary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_primary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_secondary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
            self.report_volume_source_m3s[None] = 0.0
            self.report_primary_volume_source_m3s[None] = 0.0
            self.report_secondary_volume_source_m3s[None] = 0.0
            self.report_active_force_cells[None] = 0

        for face in range(face_count):
            region = self.region_id[face]
            if region == primary_region_id or region == secondary_region_id:
                n = self.normal[face]
                area = self.area_m2[face]
                boundary_velocity = secondary_target_velocity_mps
                if region == primary_region_id:
                    boundary_velocity = primary_target_velocity_mps
                effective_probe_distance_m = probe_distance_m
                if effective_probe_distance_m <= 0.0:
                    centroid_grid_coordinate = self._grid_coordinate_from_fields(
                        self.centroid_m[face],
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    effective_probe_distance_m = self._local_normal_probe_distance_m(
                        n,
                        centroid_grid_coordinate.x,
                        centroid_grid_coordinate.y,
                        centroid_grid_coordinate.z,
                        cell_width_x_m,
                        cell_width_y_m,
                        cell_width_z_m,
                        nx,
                        ny,
                        nz,
                    )
                probe = self.centroid_m[face] + n * effective_probe_distance_m
                in_bounds = (
                    probe.x >= bounds_min_x
                    and probe.x <= bounds_max_x
                    and probe.y >= bounds_min_y
                    and probe.y <= bounds_max_y
                    and probe.z >= bounds_min_z
                    and probe.z <= bounds_max_z
                )
                sample_probe = probe
                if read_full_report != 0 and not in_bounds:
                    ti.atomic_add(self.report_invalid_probe_count[None], 1)
                    ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                    ti.atomic_add(
                        self.report_invalid_probe_volume_source_m3s[None],
                        boundary_velocity.dot(n) * area,
                    )
                sample_in_bounds = in_bounds
                if sample_in_bounds:
                    sampled_fluid_weight = self._scatter_velocity_constraint_to_grid(
                        target_sum_field,
                        weight_field,
                        obstacle_field,
                        boundary_velocity,
                        sample_probe,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    if sampled_fluid_weight > 1.0e-12:
                        if read_full_report != 0:
                            ti.atomic_add(self.report_sample_count[None], 1)
                        if write_region_fields != 0:
                            if region == primary_region_id:
                                self._scatter_velocity_constraint_region_to_grid(
                                    primary_target_sum_field,
                                    primary_weight_field,
                                    obstacle_field,
                                    boundary_velocity,
                                    sample_probe,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                            else:
                                self._scatter_velocity_constraint_region_to_grid(
                                    secondary_target_sum_field,
                                    secondary_weight_field,
                                    obstacle_field,
                                    boundary_velocity,
                                    sample_probe,
                                    cell_face_x_m,
                                    cell_face_y_m,
                                    cell_face_z_m,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                    nx,
                                    ny,
                                    nz,
                                )
                    elif read_full_report != 0:
                        ti.atomic_add(self.report_invalid_probe_count[None], 1)
                        ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                        ti.atomic_add(
                            self.report_invalid_probe_volume_source_m3s[None],
                            boundary_velocity.dot(n) * area,
                        )
        if read_full_report != 0:
            self._pack_report_snapshot()

    @ti.kernel
    def _diagnose_from_fields_kernel(
        self,
        velocity_field: ti.template(),
        pressure_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        face_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_target_velocity_mps: ti.types.vector(3, ti.f32),
        secondary_target_velocity_mps: ti.types.vector(3, ti.f32),
        probe_distance_m: ti.f32,
        bounds_min_x: ti.f32,
        bounds_min_y: ti.f32,
        bounds_min_z: ti.f32,
        bounds_max_x: ti.f32,
        bounds_max_y: ti.f32,
        bounds_max_z: ti.f32,
        dx: ti.f32,
        dy: ti.f32,
        dz: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
    ):
        self.report_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_pressure_abs_force_n[None] = 0.0
        self.report_pressure_area_m2[None] = 0.0
        self.report_pressure_face_count[None] = 0
        self.report_residual_square_sum[None] = 0.0
        self.report_residual_max[None] = 0.0
        self.report_sample_count[None] = 0
        self.report_invalid_probe_count[None] = 0
        self.report_invalid_probe_area_m2[None] = 0.0
        self.report_invalid_probe_volume_source_m3s[None] = 0.0
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_fluid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_pressure_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_constraint_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_viscous_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_fluid_stress_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_volume_source_m3s[None] = 0.0
        self.report_primary_volume_source_m3s[None] = 0.0
        self.report_secondary_volume_source_m3s[None] = 0.0
        self.report_active_force_cells[None] = 0

        for face in range(face_count):
            region = self.region_id[face]
            if region == primary_region_id or region == secondary_region_id:
                n = self.normal[face]
                area = self.area_m2[face]
                boundary_velocity = secondary_target_velocity_mps
                if region == primary_region_id:
                    boundary_velocity = primary_target_velocity_mps

                effective_probe_distance_m = probe_distance_m
                if effective_probe_distance_m <= 0.0:
                    centroid_grid_coordinate = self._grid_coordinate_from_fields(
                        self.centroid_m[face],
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    effective_probe_distance_m = self._local_normal_probe_distance_m(
                        n,
                        centroid_grid_coordinate.x,
                        centroid_grid_coordinate.y,
                        centroid_grid_coordinate.z,
                        cell_width_x_m,
                        cell_width_y_m,
                        cell_width_z_m,
                        nx,
                        ny,
                        nz,
                    )
                probe = self.centroid_m[face] + n * effective_probe_distance_m
                in_bounds = (
                    probe.x >= bounds_min_x
                    and probe.x <= bounds_max_x
                    and probe.y >= bounds_min_y
                    and probe.y <= bounds_max_y
                    and probe.z >= bounds_min_z
                    and probe.z <= bounds_max_z
                )
                sample_probe = probe
                if not in_bounds:
                    ti.atomic_add(self.report_invalid_probe_count[None], 1)
                    ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                    ti.atomic_add(
                        self.report_invalid_probe_volume_source_m3s[None],
                        boundary_velocity.dot(n) * area,
                    )
                sample_in_bounds = in_bounds
                if sample_in_bounds:
                    grid_coordinate = self._grid_coordinate_from_fields(
                        sample_probe,
                        cell_face_x_m,
                        cell_face_y_m,
                        cell_face_z_m,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                        nx,
                        ny,
                        nz,
                    )
                    gx = grid_coordinate.x
                    gy = grid_coordinate.y
                    gz = grid_coordinate.z
                    sampled_velocity, _sampled_fluid_weight = self._sample_fluid_velocity_trilinear(
                        velocity_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                    )
                    sample_has_fluid_support = _sampled_fluid_weight > 1.0e-12
                    if in_bounds and _sampled_fluid_weight <= 1.0e-12:
                        ti.atomic_add(self.report_invalid_probe_count[None], 1)
                        ti.atomic_add(self.report_invalid_probe_area_m2[None], area)
                        ti.atomic_add(
                            self.report_invalid_probe_volume_source_m3s[None],
                            boundary_velocity.dot(n) * area,
                        )
                    sample_area = area if sample_has_fluid_support else 0.0
                    sample_count = ti.cast(sample_has_fluid_support, ti.i32)
                    sampled_pressure, _pressure_fluid_weight = self._sample_pressure_trilinear(
                        pressure_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                    )
                    pressure_force_on_solid = -sampled_pressure * n * sample_area
                    velocity_gradient = self._sample_velocity_gradient(
                        velocity_field,
                        obstacle_field,
                        gx,
                        gy,
                        gz,
                        nx,
                        ny,
                        nz,
                        cell_center_x_m,
                        cell_center_y_m,
                        cell_center_z_m,
                    )
                    viscous_stress = viscosity_pa_s * (
                        velocity_gradient + velocity_gradient.transpose()
                    )
                    viscous_force_on_solid = (viscous_stress @ n) * sample_area
                    fluid_stress_force_on_solid = pressure_force_on_solid + viscous_force_on_solid
                    residual = (sampled_velocity - boundary_velocity).norm()
                    ti.atomic_add(
                        self.report_pressure_force_n[None].x,
                        pressure_force_on_solid.x,
                    )
                    ti.atomic_add(
                        self.report_pressure_force_n[None].y,
                        pressure_force_on_solid.y,
                    )
                    ti.atomic_add(
                        self.report_pressure_force_n[None].z,
                        pressure_force_on_solid.z,
                    )
                    if region == primary_region_id:
                        ti.atomic_add(
                            self.report_primary_pressure_force_n[None].x,
                            pressure_force_on_solid.x,
                        )
                        ti.atomic_add(
                            self.report_primary_pressure_force_n[None].y,
                            pressure_force_on_solid.y,
                        )
                        ti.atomic_add(
                            self.report_primary_pressure_force_n[None].z,
                            pressure_force_on_solid.z,
                        )
                        self._atomic_add_vector(
                            self.report_primary_viscous_force_n,
                            viscous_force_on_solid,
                        )
                        self._atomic_add_vector(
                            self.report_primary_fluid_stress_force_n,
                            fluid_stress_force_on_solid,
                        )
                    else:
                        ti.atomic_add(
                            self.report_secondary_pressure_force_n[None].x,
                            pressure_force_on_solid.x,
                        )
                        ti.atomic_add(
                            self.report_secondary_pressure_force_n[None].y,
                            pressure_force_on_solid.y,
                        )
                        ti.atomic_add(
                            self.report_secondary_pressure_force_n[None].z,
                            pressure_force_on_solid.z,
                        )
                        self._atomic_add_vector(
                            self.report_secondary_viscous_force_n,
                            viscous_force_on_solid,
                        )
                        self._atomic_add_vector(
                            self.report_secondary_fluid_stress_force_n,
                            fluid_stress_force_on_solid,
                        )
                    self._atomic_add_vector(self.report_viscous_force_n, viscous_force_on_solid)
                    self._atomic_add_vector(
                        self.report_fluid_stress_force_n,
                        fluid_stress_force_on_solid,
                    )
                    ti.atomic_add(self.report_pressure_abs_force_n[None], ti.abs(sampled_pressure) * sample_area)
                    ti.atomic_add(self.report_pressure_area_m2[None], sample_area)
                    ti.atomic_add(self.report_pressure_face_count[None], sample_count)
                    ti.atomic_add(
                        self.report_residual_square_sum[None],
                        residual * residual * ti.cast(sample_has_fluid_support, ti.f32),
                    )
                    ti.atomic_max(
                        self.report_residual_max[None],
                        residual if sample_has_fluid_support else 0.0,
                    )
                    ti.atomic_add(self.report_sample_count[None], sample_count)
        self._pack_report_snapshot()

    @ti.kernel
    def _spread_pressure_interface_matrix_terms_kernel(
        self,
        diagonal_field: ti.template(),
        rhs_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        face_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_pressure_robin_impedance_ns_m: ti.f32,
        secondary_pressure_robin_impedance_ns_m: ti.f32,
        primary_pressure_robin_reference_pa: ti.f32,
        secondary_pressure_robin_reference_pa: ti.f32,
        primary_interface_area_m2: ti.f32,
        secondary_interface_area_m2: ti.f32,
        density_kgm3: ti.f32,
        dt_s: ti.f32,
        probe_distance_m: ti.f32,
        bounds_min_x: ti.f32,
        bounds_min_y: ti.f32,
        bounds_min_z: ti.f32,
        bounds_max_x: ti.f32,
        bounds_max_y: ti.f32,
        bounds_max_z: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        for face in range(face_count):
            region = self.region_id[face]
            if region == primary_region_id or region == secondary_region_id:
                impedance_ns_m = secondary_pressure_robin_impedance_ns_m
                reference_pa = secondary_pressure_robin_reference_pa
                interface_area_m2 = secondary_interface_area_m2
                if region == primary_region_id:
                    impedance_ns_m = primary_pressure_robin_impedance_ns_m
                    reference_pa = primary_pressure_robin_reference_pa
                    interface_area_m2 = primary_interface_area_m2
                if impedance_ns_m > 0.0 and interface_area_m2 > 0.0:
                    n = self.normal[face]
                    sample_area_m2 = self.area_m2[face]
                    sample_probe = self.centroid_m[face] + n * probe_distance_m
                    in_bounds = (
                        sample_probe.x >= bounds_min_x
                        and sample_probe.x <= bounds_max_x
                        and sample_probe.y >= bounds_min_y
                        and sample_probe.y <= bounds_max_y
                        and sample_probe.z >= bounds_min_z
                        and sample_probe.z <= bounds_max_z
                    )
                    if in_bounds:
                        pressure_admittance_m3_s_pa = (
                            interface_area_m2 * sample_area_m2 / impedance_ns_m
                        )
                        diagonal_integral = (
                            density_kgm3
                            / ti.max(dt_s, 1.0e-12)
                            * pressure_admittance_m3_s_pa
                        )
                        self._scatter_volume_source_to_grid(
                            diagonal_field,
                            obstacle_field,
                            diagonal_integral,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            cell_width_x_m,
                            cell_width_y_m,
                            cell_width_z_m,
                            nx,
                            ny,
                            nz,
                        )
                        self._scatter_volume_source_to_grid(
                            rhs_field,
                            obstacle_field,
                            diagonal_integral * reference_pa,
                            sample_probe,
                            cell_face_x_m,
                            cell_face_y_m,
                            cell_face_z_m,
                            cell_center_x_m,
                            cell_center_y_m,
                            cell_center_z_m,
                            cell_width_x_m,
                            cell_width_y_m,
                            cell_width_z_m,
                            nx,
                            ny,
                            nz,
                        )

    @staticmethod
    def _grid_field_tuple(grid_fields) -> tuple[object, ...]:
        required_fields = (
            "cell_face_x_m",
            "cell_face_y_m",
            "cell_face_z_m",
            "cell_center_x_m",
            "cell_center_y_m",
            "cell_center_z_m",
            "cell_width_x_m",
            "cell_width_y_m",
            "cell_width_z_m",
        )
        missing = [name for name in required_fields if not hasattr(grid_fields, name)]
        if missing:
            raise ValueError(
                "grid_fields must expose CartesianGrid device fields: " + ", ".join(missing)
            )
        return tuple(getattr(grid_fields, name) for name in required_fields)

    def spread_pressure_interface_matrix_terms(
        self,
        diagonal_field,
        rhs_field,
        obstacle_field,
        *,
        grid_fields,
        primary_region_id: int,
        secondary_region_id: int,
        primary_pressure_robin_impedance_ns_m: float,
        secondary_pressure_robin_impedance_ns_m: float,
        primary_pressure_robin_reference_pa: float,
        secondary_pressure_robin_reference_pa: float,
        primary_interface_area_m2: float,
        secondary_interface_area_m2: float,
        density_kgm3: float,
        dt_s: float,
        probe_distance_m: float,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        spacing_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
    ) -> None:
        primary_impedance = self._non_negative_float(
            primary_pressure_robin_impedance_ns_m,
            "primary_pressure_robin_impedance_ns_m",
        )
        secondary_impedance = self._non_negative_float(
            secondary_pressure_robin_impedance_ns_m,
            "secondary_pressure_robin_impedance_ns_m",
        )
        primary_reference = float(primary_pressure_robin_reference_pa)
        secondary_reference = float(secondary_pressure_robin_reference_pa)
        if not math.isfinite(primary_reference):
            raise ValueError("primary_pressure_robin_reference_pa must be finite")
        if not math.isfinite(secondary_reference):
            raise ValueError("secondary_pressure_robin_reference_pa must be finite")
        primary_area_m2 = float(primary_interface_area_m2)
        secondary_area_m2 = float(secondary_interface_area_m2)
        if not math.isfinite(primary_area_m2) or primary_area_m2 < 0.0:
            raise ValueError("primary_interface_area_m2 must be a finite non-negative number")
        if not math.isfinite(secondary_area_m2) or secondary_area_m2 < 0.0:
            raise ValueError("secondary_interface_area_m2 must be a finite non-negative number")
        if primary_impedance > 0.0 and primary_area_m2 <= 0.0:
            raise ValueError(
                "primary_interface_area_m2 must be positive when primary pressure Robin impedance is nonzero"
            )
        if secondary_impedance > 0.0 and secondary_area_m2 <= 0.0:
            raise ValueError(
                "secondary_interface_area_m2 must be positive when secondary pressure Robin impedance is nonzero"
            )
        density = self._non_negative_float(density_kgm3, "density_kgm3")
        if density <= 0.0:
            raise ValueError("density_kgm3 must be positive")
        dt = self._non_negative_float(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        probe_distance = self._non_negative_float(probe_distance_m, "probe_distance_m")
        bounds_min = self._velocity_tuple(bounds_min_m, "bounds_min_m")
        bounds_max = self._velocity_tuple(bounds_max_m, "bounds_max_m")
        grid_nodes_tuple = tuple(int(v) for v in grid_nodes)
        if len(grid_nodes_tuple) != 3 or any(v <= 1 for v in grid_nodes_tuple):
            raise ValueError("grid_nodes must contain three values greater than one")
        spacing_tuple = self._velocity_tuple(spacing_m, "spacing_m")
        if any(value <= 0.0 for value in spacing_tuple):
            raise ValueError("spacing_m must be positive")
        grid_field_tuple = self._grid_field_tuple(grid_fields)
        self._spread_pressure_interface_matrix_terms_kernel(
            diagonal_field,
            rhs_field,
            obstacle_field,
            *grid_field_tuple,
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            float(primary_impedance),
            float(secondary_impedance),
            float(primary_reference),
            float(secondary_reference),
            float(primary_area_m2),
            float(secondary_area_m2),
            float(density),
            float(dt),
            float(probe_distance),
            float(bounds_min[0]),
            float(bounds_min[1]),
            float(bounds_min[2]),
            float(bounds_max[0]),
            float(bounds_max[1]),
            float(bounds_max[2]),
            int(grid_nodes_tuple[0]),
            int(grid_nodes_tuple[1]),
            int(grid_nodes_tuple[2]),
        )

    def spread_fsi_forces(
        self,
        velocity_field,
        pressure_field,
        force_field,
        volume_source_field,
        obstacle_field,
        *,
        grid_fields,
        primary_region_id: int,
        secondary_region_id: int,
        primary_velocity_mps: tuple[float, float, float],
        secondary_velocity_mps: tuple[float, float, float],
        probe_distance_m: float,
        density_kgm3: float,
        viscosity_pa_s: float,
        dt_s: float,
        constraint_force_scale: float,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        spacing_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
        constraint_force_solid_mobility_ratio: float = 0.0,
        primary_constraint_force_solid_mobility_ratio: float | None = None,
        secondary_constraint_force_solid_mobility_ratio: float | None = None,
        velocity_target_solid_mobility_ratio: float = 0.0,
        primary_velocity_target_solid_mobility_ratio: float | None = None,
        secondary_velocity_target_solid_mobility_ratio: float | None = None,
        primary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        secondary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        primary_interface_area_m2: float = 0.0,
        secondary_interface_area_m2: float = 0.0,
        read_full_report: bool = True,
        read_force_pair_report: bool = True,
    ) -> TriSurfaceDiagnosticReport | TriSurfaceForcePairReport | None:
        primary_velocity = self._velocity_tuple(primary_velocity_mps, "primary_velocity_mps")
        secondary_velocity = self._velocity_tuple(secondary_velocity_mps, "secondary_velocity_mps")
        primary_impedance_force = self._velocity_tuple(
            primary_interface_impedance_force_n,
            "primary_interface_impedance_force_n",
        )
        secondary_impedance_force = self._velocity_tuple(
            secondary_interface_impedance_force_n,
            "secondary_interface_impedance_force_n",
        )
        primary_area_m2 = float(primary_interface_area_m2)
        secondary_area_m2 = float(secondary_interface_area_m2)
        force_solid_mobility_ratio = self._non_negative_float(
            constraint_force_solid_mobility_ratio,
            "constraint_force_solid_mobility_ratio",
        )
        primary_force_solid_mobility_ratio = (
            force_solid_mobility_ratio
            if primary_constraint_force_solid_mobility_ratio is None
            else self._non_negative_float(
                primary_constraint_force_solid_mobility_ratio,
                "primary_constraint_force_solid_mobility_ratio",
            )
        )
        secondary_force_solid_mobility_ratio = (
            force_solid_mobility_ratio
            if secondary_constraint_force_solid_mobility_ratio is None
            else self._non_negative_float(
                secondary_constraint_force_solid_mobility_ratio,
                "secondary_constraint_force_solid_mobility_ratio",
            )
        )
        velocity_solid_mobility_ratio = self._non_negative_float(
            velocity_target_solid_mobility_ratio,
            "velocity_target_solid_mobility_ratio",
        )
        primary_velocity_solid_mobility_ratio = (
            velocity_solid_mobility_ratio
            if primary_velocity_target_solid_mobility_ratio is None
            else self._non_negative_float(
                primary_velocity_target_solid_mobility_ratio,
                "primary_velocity_target_solid_mobility_ratio",
            )
        )
        secondary_velocity_solid_mobility_ratio = (
            velocity_solid_mobility_ratio
            if secondary_velocity_target_solid_mobility_ratio is None
            else self._non_negative_float(
                secondary_velocity_target_solid_mobility_ratio,
                "secondary_velocity_target_solid_mobility_ratio",
            )
        )
        if not math.isfinite(primary_area_m2) or primary_area_m2 < 0.0:
            raise ValueError("primary_interface_area_m2 must be a finite non-negative number")
        if not math.isfinite(secondary_area_m2) or secondary_area_m2 < 0.0:
            raise ValueError("secondary_interface_area_m2 must be a finite non-negative number")
        if any(component != 0.0 for component in primary_impedance_force) and primary_area_m2 <= 0.0:
            raise ValueError("primary_interface_area_m2 must be positive when primary impedance force is nonzero")
        if any(component != 0.0 for component in secondary_impedance_force) and secondary_area_m2 <= 0.0:
            raise ValueError("secondary_interface_area_m2 must be positive when secondary impedance force is nonzero")
        grid_field_tuple = self._grid_field_tuple(grid_fields)
        self._spread_fsi_force_kernel(
            velocity_field,
            pressure_field,
            force_field,
            volume_source_field,
            obstacle_field,
            *grid_field_tuple,
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            ti.Vector(primary_velocity),
            ti.Vector(secondary_velocity),
            float(probe_distance_m),
            float(density_kgm3),
            float(viscosity_pa_s),
            float(dt_s),
            float(constraint_force_scale),
            force_solid_mobility_ratio,
            primary_force_solid_mobility_ratio,
            secondary_force_solid_mobility_ratio,
            primary_velocity_solid_mobility_ratio,
            secondary_velocity_solid_mobility_ratio,
            ti.Vector(primary_impedance_force),
            ti.Vector(secondary_impedance_force),
            float(primary_area_m2),
            float(secondary_area_m2),
            float(bounds_min_m[0]),
            float(bounds_min_m[1]),
            float(bounds_min_m[2]),
            float(bounds_max_m[0]),
            float(bounds_max_m[1]),
            float(bounds_max_m[2]),
            float(spacing_m[0]),
            float(spacing_m[1]),
            float(spacing_m[2]),
            int(grid_nodes[0]),
            int(grid_nodes[1]),
            int(grid_nodes[2]),
            int(bool(read_full_report)),
            1,
            int(bool(read_force_pair_report)),
        )
        if not read_full_report:
            if not read_force_pair_report:
                self.last_report_host_reads = 0
                return None
            return self.force_pair_report()
        self._pack_active_force_cells_from_field_kernel(force_field)
        return self.report(stress_fields_computed=True, force_fields_computed=True)

    def diagnose_fsi_forces_from_fields(
        self,
        velocity_field,
        pressure_field,
        force_field,
        volume_source_field,
        obstacle_field,
        *,
        grid_fields,
        primary_region_id: int,
        secondary_region_id: int,
        primary_velocity_mps: tuple[float, float, float],
        secondary_velocity_mps: tuple[float, float, float],
        probe_distance_m: float,
        density_kgm3: float,
        viscosity_pa_s: float,
        dt_s: float,
        constraint_force_scale: float,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        spacing_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
        constraint_force_solid_mobility_ratio: float = 0.0,
        primary_constraint_force_solid_mobility_ratio: float | None = None,
        secondary_constraint_force_solid_mobility_ratio: float | None = None,
        velocity_target_solid_mobility_ratio: float = 0.0,
        primary_velocity_target_solid_mobility_ratio: float | None = None,
        secondary_velocity_target_solid_mobility_ratio: float | None = None,
        primary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        secondary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        primary_interface_area_m2: float = 0.0,
        secondary_interface_area_m2: float = 0.0,
    ) -> TriSurfaceDiagnosticReport:
        primary_velocity = self._velocity_tuple(primary_velocity_mps, "primary_velocity_mps")
        secondary_velocity = self._velocity_tuple(secondary_velocity_mps, "secondary_velocity_mps")
        primary_impedance_force = self._velocity_tuple(
            primary_interface_impedance_force_n,
            "primary_interface_impedance_force_n",
        )
        secondary_impedance_force = self._velocity_tuple(
            secondary_interface_impedance_force_n,
            "secondary_interface_impedance_force_n",
        )
        primary_area_m2 = float(primary_interface_area_m2)
        secondary_area_m2 = float(secondary_interface_area_m2)
        force_solid_mobility_ratio = self._non_negative_float(
            constraint_force_solid_mobility_ratio,
            "constraint_force_solid_mobility_ratio",
        )
        primary_force_solid_mobility_ratio = (
            force_solid_mobility_ratio
            if primary_constraint_force_solid_mobility_ratio is None
            else self._non_negative_float(
                primary_constraint_force_solid_mobility_ratio,
                "primary_constraint_force_solid_mobility_ratio",
            )
        )
        secondary_force_solid_mobility_ratio = (
            force_solid_mobility_ratio
            if secondary_constraint_force_solid_mobility_ratio is None
            else self._non_negative_float(
                secondary_constraint_force_solid_mobility_ratio,
                "secondary_constraint_force_solid_mobility_ratio",
            )
        )
        velocity_solid_mobility_ratio = self._non_negative_float(
            velocity_target_solid_mobility_ratio,
            "velocity_target_solid_mobility_ratio",
        )
        primary_velocity_solid_mobility_ratio = (
            velocity_solid_mobility_ratio
            if primary_velocity_target_solid_mobility_ratio is None
            else self._non_negative_float(
                primary_velocity_target_solid_mobility_ratio,
                "primary_velocity_target_solid_mobility_ratio",
            )
        )
        secondary_velocity_solid_mobility_ratio = (
            velocity_solid_mobility_ratio
            if secondary_velocity_target_solid_mobility_ratio is None
            else self._non_negative_float(
                secondary_velocity_target_solid_mobility_ratio,
                "secondary_velocity_target_solid_mobility_ratio",
            )
        )
        if not math.isfinite(primary_area_m2) or primary_area_m2 < 0.0:
            raise ValueError("primary_interface_area_m2 must be a finite non-negative number")
        if not math.isfinite(secondary_area_m2) or secondary_area_m2 < 0.0:
            raise ValueError("secondary_interface_area_m2 must be a finite non-negative number")
        if any(component != 0.0 for component in primary_impedance_force) and primary_area_m2 <= 0.0:
            raise ValueError("primary_interface_area_m2 must be positive when primary impedance force is nonzero")
        if any(component != 0.0 for component in secondary_impedance_force) and secondary_area_m2 <= 0.0:
            raise ValueError("secondary_interface_area_m2 must be positive when secondary impedance force is nonzero")
        grid_field_tuple = self._grid_field_tuple(grid_fields)
        self._spread_fsi_force_kernel(
            velocity_field,
            pressure_field,
            force_field,
            volume_source_field,
            obstacle_field,
            *grid_field_tuple,
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            ti.Vector(primary_velocity),
            ti.Vector(secondary_velocity),
            float(probe_distance_m),
            float(density_kgm3),
            float(viscosity_pa_s),
            float(dt_s),
            float(constraint_force_scale),
            force_solid_mobility_ratio,
            primary_force_solid_mobility_ratio,
            secondary_force_solid_mobility_ratio,
            primary_velocity_solid_mobility_ratio,
            secondary_velocity_solid_mobility_ratio,
            ti.Vector(primary_impedance_force),
            ti.Vector(secondary_impedance_force),
            float(primary_area_m2),
            float(secondary_area_m2),
            float(bounds_min_m[0]),
            float(bounds_min_m[1]),
            float(bounds_min_m[2]),
            float(bounds_max_m[0]),
            float(bounds_max_m[1]),
            float(bounds_max_m[2]),
            float(spacing_m[0]),
            float(spacing_m[1]),
            float(spacing_m[2]),
            int(grid_nodes[0]),
            int(grid_nodes[1]),
            int(grid_nodes[2]),
            1,
            0,
            0,
        )
        return self.report(stress_fields_computed=True, force_fields_computed=True)

    def spread_fsi_velocity_constraints(
        self,
        target_sum_field,
        weight_field,
        *,
        grid_fields,
        primary_region_id: int,
        secondary_region_id: int,
        primary_velocity_mps: tuple[float, float, float],
        secondary_velocity_mps: tuple[float, float, float],
        probe_distance_m: float,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        spacing_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
        read_full_report: bool = True,
    ) -> TriSurfaceDiagnosticReport | None:
        primary_velocity = self._velocity_tuple(primary_velocity_mps, "primary_velocity_mps")
        secondary_velocity = self._velocity_tuple(secondary_velocity_mps, "secondary_velocity_mps")
        grid_field_tuple = self._grid_field_tuple(grid_fields)
        obstacle_field = getattr(grid_fields, "obstacle", None)
        if obstacle_field is None:
            raise ValueError("grid_fields must expose obstacle for velocity-constraint diagnostics")
        primary_target_sum_field = getattr(
            grid_fields,
            "velocity_constraint_primary_sum",
            target_sum_field,
        )
        primary_weight_field = getattr(
            grid_fields,
            "velocity_constraint_primary_weight",
            weight_field,
        )
        secondary_target_sum_field = getattr(
            grid_fields,
            "velocity_constraint_secondary_sum",
            target_sum_field,
        )
        secondary_weight_field = getattr(
            grid_fields,
            "velocity_constraint_secondary_weight",
            weight_field,
        )
        write_region_fields = int(
            hasattr(grid_fields, "velocity_constraint_primary_sum")
            and hasattr(grid_fields, "velocity_constraint_primary_weight")
            and hasattr(grid_fields, "velocity_constraint_secondary_sum")
            and hasattr(grid_fields, "velocity_constraint_secondary_weight")
        )
        self._spread_fsi_velocity_constraint_kernel(
            target_sum_field,
            weight_field,
            primary_target_sum_field,
            primary_weight_field,
            secondary_target_sum_field,
            secondary_weight_field,
            obstacle_field,
            *grid_field_tuple,
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            ti.Vector(primary_velocity),
            ti.Vector(secondary_velocity),
            float(probe_distance_m),
            float(bounds_min_m[0]),
            float(bounds_min_m[1]),
            float(bounds_min_m[2]),
            float(bounds_max_m[0]),
            float(bounds_max_m[1]),
            float(bounds_max_m[2]),
            float(spacing_m[0]),
            float(spacing_m[1]),
            float(spacing_m[2]),
            int(grid_nodes[0]),
            int(grid_nodes[1]),
            int(grid_nodes[2]),
            int(bool(read_full_report)),
            write_region_fields,
        )
        if not read_full_report:
            self.last_report_host_reads = 0
            return None
        return self.report(stress_fields_computed=False, force_fields_computed=False)

    def diagnose_from_fields(
        self,
        velocity_field,
        pressure_field,
        *,
        grid_fields,
        primary_region_id: int,
        secondary_region_id: int,
        primary_velocity_mps: tuple[float, float, float],
        secondary_velocity_mps: tuple[float, float, float],
        probe_distance_m: float,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        spacing_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
        viscosity_pa_s: float,
    ) -> TriSurfaceDiagnosticReport:
        primary_velocity = self._velocity_tuple(primary_velocity_mps, "primary_velocity_mps")
        secondary_velocity = self._velocity_tuple(secondary_velocity_mps, "secondary_velocity_mps")
        grid_field_tuple = self._grid_field_tuple(grid_fields)
        obstacle_field = getattr(grid_fields, "obstacle", None)
        if obstacle_field is None:
            raise ValueError("grid_fields must expose obstacle for velocity-gradient diagnostics")
        self._diagnose_from_fields_kernel(
            velocity_field,
            pressure_field,
            obstacle_field,
            *grid_field_tuple,
            int(self.face_count),
            int(primary_region_id),
            int(secondary_region_id),
            ti.Vector(primary_velocity),
            ti.Vector(secondary_velocity),
            float(probe_distance_m),
            float(bounds_min_m[0]),
            float(bounds_min_m[1]),
            float(bounds_min_m[2]),
            float(bounds_max_m[0]),
            float(bounds_max_m[1]),
            float(bounds_max_m[2]),
            float(spacing_m[0]),
            float(spacing_m[1]),
            float(spacing_m[2]),
            int(grid_nodes[0]),
            int(grid_nodes[1]),
            int(grid_nodes[2]),
            float(viscosity_pa_s),
        )
        return self.report(stress_fields_computed=True, force_fields_computed=False)

    def force_pair_report(self) -> TriSurfaceForcePairReport:
        values = self.report_force_pair_snapshot[None]
        self.last_report_host_reads = 1
        sample_count = int(float(values[6]))
        invalid_count = int(float(values[7]))
        total_probe_count = sample_count + invalid_count
        valid_fraction = sample_count / max(total_probe_count, 1)
        return TriSurfaceForcePairReport(
            primary_fluid_force_n=(
                float(values[0]),
                float(values[1]),
                float(values[2]),
            ),
            secondary_fluid_force_n=(
                float(values[3]),
                float(values[4]),
                float(values[5]),
            ),
            force_sample_count=sample_count,
            force_invalid_probe_count=invalid_count,
            force_valid_probe_count=sample_count,
            force_valid_probe_fraction=valid_fraction,
            invalid_probe_count=invalid_count,
            valid_probe_fraction=valid_fraction,
            invalid_probe_area_m2=float(values[8]),
            invalid_probe_volume_source_m3s=float(values[9]),
        )

    def force_impulse_report(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        values = self.force_impulse_n_s[None]
        self.last_force_impulse_host_reads = 1
        return (
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5])),
        )

    @staticmethod
    def _field_vector3_tuple(field) -> tuple[float, float, float]:
        vector = field[None]
        return (float(vector[0]), float(vector[1]), float(vector[2]))

    @staticmethod
    def _nan_vector3() -> tuple[float, float, float]:
        return (math.nan, math.nan, math.nan)

    def report(self, *, stress_fields_computed: bool, force_fields_computed: bool) -> TriSurfaceDiagnosticReport:
        values_a = self.report_float_snapshot_a[None]
        values_b = self.report_float_snapshot_b[None]
        counts = self.report_count_snapshot[None]
        self.last_report_host_reads = 3

        def vector_from_a(offset: int) -> tuple[float, float, float]:
            return (
                float(values_a[offset]),
                float(values_a[offset + 1]),
                float(values_a[offset + 2]),
            )

        def vector_from_b(offset: int) -> tuple[float, float, float]:
            return (
                float(values_b[offset]),
                float(values_b[offset + 1]),
                float(values_b[offset + 2]),
            )

        sample_count = int(counts[1])
        invalid_count = int(counts[2])
        total_probe_count = sample_count + invalid_count
        valid_fraction = sample_count / max(total_probe_count, 1)
        residual_l2 = math.nan
        if stress_fields_computed and sample_count > 0:
            residual_l2 = (float(values_a[5]) / sample_count) ** 0.5
        elif stress_fields_computed:
            residual_l2 = 0.0
        nan_vector = self._nan_vector3()
        pressure_force = (
            vector_from_a(0)
            if stress_fields_computed
            else nan_vector
        )
        pressure_abs_force_n = (
            float(values_a[3])
            if stress_fields_computed
            else math.nan
        )
        pressure_face_count = (
            int(counts[0])
            if stress_fields_computed
            else None
        )
        pressure_area_m2 = (
            float(values_a[4])
            if stress_fields_computed
            else math.nan
        )
        residual_max = (
            float(values_a[6])
            if stress_fields_computed
            else math.nan
        )
        primary_pressure_force = (
            vector_from_a(21)
            if stress_fields_computed
            else nan_vector
        )
        secondary_pressure_force = (
            vector_from_a(24)
            if stress_fields_computed
            else nan_vector
        )
        viscous_force = (
            vector_from_b(1)
            if stress_fields_computed
            else nan_vector
        )
        fluid_stress_force = (
            vector_from_b(4)
            if stress_fields_computed
            else nan_vector
        )
        primary_viscous_force = (
            vector_from_b(7)
            if stress_fields_computed
            else nan_vector
        )
        secondary_viscous_force = (
            vector_from_b(10)
            if stress_fields_computed
            else nan_vector
        )
        primary_fluid_stress_force = (
            vector_from_b(13)
            if stress_fields_computed
            else nan_vector
        )
        secondary_fluid_stress_force = (
            vector_from_b(16)
            if stress_fields_computed
            else nan_vector
        )
        force_sample_count = sample_count if force_fields_computed else None
        force_invalid_count = invalid_count if force_fields_computed else None
        force_valid_count = sample_count if force_fields_computed else None
        force_valid_fraction = valid_fraction if force_fields_computed else math.nan
        grid_force = (
            vector_from_a(9)
            if force_fields_computed
            else nan_vector
        )
        primary_fluid_force = (
            vector_from_a(12)
            if force_fields_computed
            else nan_vector
        )
        secondary_fluid_force = (
            vector_from_a(15)
            if force_fields_computed
            else nan_vector
        )
        constraint_force = (
            vector_from_a(18)
            if force_fields_computed
            else nan_vector
        )
        primary_constraint_force = (
            vector_from_a(27)
            if force_fields_computed
            else nan_vector
        )
        secondary_constraint_force = (
            (float(values_a[30]), float(values_a[31]), float(values_b[0]))
            if force_fields_computed
            else nan_vector
        )
        volume_source_m3s = (
            float(values_b[19])
            if force_fields_computed
            else math.nan
        )
        primary_volume_source_m3s = (
            float(values_b[20])
            if force_fields_computed
            else math.nan
        )
        secondary_volume_source_m3s = (
            float(values_b[21])
            if force_fields_computed
            else math.nan
        )
        active_force_cells = (
            int(counts[3])
            if force_fields_computed
            else None
        )
        return TriSurfaceDiagnosticReport(
            face_count=int(self.face_count),
            pressure_traction_force_n=pressure_force,
            pressure_traction_abs_force_n=pressure_abs_force_n,
            pressure_traction_face_count=pressure_face_count,
            pressure_traction_area_m2=pressure_area_m2,
            projected_ibm_residual_mps=residual_max,
            projected_ibm_residual_l2_mps=residual_l2,
            projected_ibm_sample_count=sample_count,
            invalid_probe_count=invalid_count,
            valid_probe_fraction=valid_fraction,
            invalid_probe_area_m2=float(values_a[7]),
            invalid_probe_volume_source_m3s=float(values_a[8]),
            force_sample_count=force_sample_count,
            force_invalid_probe_count=force_invalid_count,
            force_valid_probe_count=force_valid_count,
            force_valid_probe_fraction=force_valid_fraction,
            grid_force_n=grid_force,
            primary_fluid_force_n=primary_fluid_force,
            secondary_fluid_force_n=secondary_fluid_force,
            constraint_force_n=constraint_force,
            primary_pressure_traction_force_n=primary_pressure_force,
            secondary_pressure_traction_force_n=secondary_pressure_force,
            primary_constraint_force_n=primary_constraint_force,
            secondary_constraint_force_n=secondary_constraint_force,
            viscous_traction_force_n=viscous_force,
            fluid_stress_traction_force_n=fluid_stress_force,
            primary_viscous_traction_force_n=primary_viscous_force,
            secondary_viscous_traction_force_n=secondary_viscous_force,
            primary_fluid_stress_traction_force_n=primary_fluid_stress_force,
            secondary_fluid_stress_traction_force_n=secondary_fluid_stress_force,
            volume_source_m3s=volume_source_m3s,
            primary_volume_source_m3s=primary_volume_source_m3s,
            secondary_volume_source_m3s=secondary_volume_source_m3s,
            active_force_cells=active_force_cells,
        )
