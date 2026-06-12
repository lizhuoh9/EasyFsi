from dataclasses import dataclass

import taichi as ti

from .runtime import TaichiRuntimeConfig, init_taichi


MIN_DEFORMATION_SINGULAR_VALUE = 1.0e-2
MAX_DEFORMATION_SINGULAR_VALUE = 1.0e2


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


@dataclass(frozen=True)
class NeoHookeanMpmReport:
    particle_count: int
    active_grid_nodes: int
    grid_out_of_bounds_particle_count: int
    particle_spacing_m: float
    grid_spacing_m: tuple[float, float, float]
    total_mass_kg: float
    total_volume_m3: float
    primary_mean_displacement_m: tuple[float, float, float]
    primary_mean_velocity_mps: tuple[float, float, float]
    secondary_mean_displacement_m: tuple[float, float, float]
    secondary_mean_velocity_mps: tuple[float, float, float]
    particle_momentum_kg_mps: tuple[float, float, float]
    grid_momentum_kg_mps: tuple[float, float, float]
    external_force_n: tuple[float, float, float]
    transfer_relative_error: float
    max_speed_mps: float
    max_abs_j: float
    deformation_clamp_count: int
    mean_radial_stretch: float
    max_radial_stretch_error: float


@ti.data_oriented
class NeoHookeanMpmState:
    """GPU Neo-Hookean APIC/MLS-MPM particle state."""

    def __init__(
        self,
        particle_capacity: int,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
        runtime: TaichiRuntimeConfig | None = None,
    ):
        init_taichi(runtime)
        if particle_capacity <= 0:
            raise ValueError("particle_capacity must be positive")
        if min(grid_nodes) < 4:
            raise ValueError("grid_nodes must be at least 4 in each direction")
        self.particle_capacity = int(particle_capacity)
        self.particle_count = 0
        self.grid_nodes = tuple(int(value) for value in grid_nodes)
        self.nx, self.ny, self.nz = self.grid_nodes
        self.bounds_min = tuple(float(value) for value in bounds_min_m)
        self.bounds_max = tuple(float(value) for value in bounds_max_m)
        self.dx = (
            (self.bounds_max[0] - self.bounds_min[0]) / self.nx,
            (self.bounds_max[1] - self.bounds_min[1]) / self.ny,
            (self.bounds_max[2] - self.bounds_min[2]) / self.nz,
        )
        if min(self.dx) <= 0.0:
            raise ValueError("bounds must define a positive grid spacing")

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.rest_x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.particle_capacity)
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.particle_capacity)
        self.saved_x = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.saved_v = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.saved_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.particle_capacity)
        self.saved_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.particle_capacity)
        self.mass_kg = ti.field(dtype=ti.f32, shape=self.particle_capacity)
        self.volume_m3 = ti.field(dtype=ti.f32, shape=self.particle_capacity)
        self.area_weight_m2 = ti.field(dtype=ti.f32, shape=self.particle_capacity)
        self.rest_area_weight_m2 = ti.field(dtype=ti.f32, shape=self.particle_capacity)
        self.region_id = ti.field(dtype=ti.i32, shape=self.particle_capacity)
        self.fixed_particle = ti.field(dtype=ti.i32, shape=self.particle_capacity)
        self.surface_normal = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.rest_surface_normal = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.external_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_capacity)
        self.rest_center_m = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.grid_mass_kg = ti.field(dtype=ti.f32, shape=self.grid_nodes)
        self.grid_velocity_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)
        self.grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_nodes)

        self.report_total_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.report_total_volume_m3 = ti.field(dtype=ti.f32, shape=())
        self.report_particle_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_transfer_grid_momentum_kg_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_external_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_current_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_rest_center_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_radial_center_count = ti.field(dtype=ti.i32, shape=())
        self.report_particle_momentum_square_sum = ti.field(dtype=ti.f32, shape=())
        self.report_transfer_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.report_max_abs_j = ti.field(dtype=ti.f32, shape=())
        self.report_deformation_clamp_count = ti.field(dtype=ti.i32, shape=())
        self.report_radial_stretch_sum = ti.field(dtype=ti.f32, shape=())
        self.report_radial_stretch_count = ti.field(dtype=ti.i32, shape=())
        self.report_max_radial_stretch_error = ti.field(dtype=ti.f32, shape=())
        self.report_active_grid_nodes = ti.field(dtype=ti.i32, shape=())
        self.report_grid_out_of_bounds_particle_count = ti.field(dtype=ti.i32, shape=())
        self.report_primary_displacement_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_velocity_sum_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_displacement_sum_m = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_velocity_sum_mps = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_count = ti.field(dtype=ti.i32, shape=())
        self.report_secondary_count = ti.field(dtype=ti.i32, shape=())
        self.report_float_snapshot = ti.Vector.field(28, dtype=ti.f32, shape=())
        self.report_count_snapshot = ti.Vector.field(6, dtype=ti.i32, shape=())
        self.report_host_snapshot = ti.field(dtype=ti.f32, shape=34)
        self.primary_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.secondary_mass_kg = ti.field(dtype=ti.f32, shape=())
        self.last_report_host_reads = 0

    @ti.func
    def _identity(self):
        return ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def _initialize_box_kernel(
        self,
        nxp: ti.i32,
        nyp: ti.i32,
        nzp: ti.i32,
        min_x: ti.f32,
        min_y: ti.f32,
        min_z: ti.f32,
        max_x: ti.f32,
        max_y: ti.f32,
        max_z: ti.f32,
        density_kgm3: ti.f32,
    ):
        particle_count = nxp * nyp * nzp
        volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        particle_volume = volume / ti.max(ti.cast(particle_count, ti.f32), 1.0)
        for p in range(self.particle_capacity):
            if p < particle_count:
                ix = p % nxp
                iy = (p // nxp) % nyp
                iz = p // (nxp * nyp)
                x = min_x + (ti.cast(ix, ti.f32) + 0.5) * (max_x - min_x) / ti.cast(nxp, ti.f32)
                y = min_y + (ti.cast(iy, ti.f32) + 0.5) * (max_y - min_y) / ti.cast(nyp, ti.f32)
                z = min_z + (ti.cast(iz, ti.f32) + 0.5) * (max_z - min_z) / ti.cast(nzp, ti.f32)
                self.x[p] = ti.Vector([x, y, z])
                self.rest_x[p] = ti.Vector([x, y, z])
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.F[p] = self._identity()
                self.volume_m3[p] = particle_volume
                self.area_weight_m2[p] = 0.0
                self.rest_area_weight_m2[p] = 0.0
                self.region_id[p] = 0
                self.fixed_particle[p] = 0
                self.surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
                self.rest_surface_normal[p] = ti.Vector([0.0, 0.0, 0.0])
                self.mass_kg[p] = density_kgm3 * particle_volume
                self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _update_rest_center_kernel(self, particle_count: ti.i32):
        self.rest_center_m[None] = ti.Vector([0.0, 0.0, 0.0])
        for p in range(self.particle_capacity):
            if p < particle_count:
                ti.atomic_add(self.rest_center_m[None].x, self.rest_x[p].x)
                ti.atomic_add(self.rest_center_m[None].y, self.rest_x[p].y)
                ti.atomic_add(self.rest_center_m[None].z, self.rest_x[p].z)

    @ti.kernel
    def _normalize_rest_center_kernel(self, particle_count: ti.i32):
        if particle_count > 0:
            self.rest_center_m[None] = self.rest_center_m[None] / ti.cast(particle_count, ti.f32)

    def _update_rest_center(self) -> None:
        self._update_rest_center_kernel(int(self.particle_count))
        self._normalize_rest_center_kernel(int(self.particle_count))

    def initialize_box(
        self,
        *,
        particle_counts: tuple[int, int, int],
        box_min_m: tuple[float, float, float],
        box_max_m: tuple[float, float, float],
        density_kgm3: float,
    ) -> None:
        nxp, nyp, nzp = (int(value) for value in particle_counts)
        if min(nxp, nyp, nzp) <= 0:
            raise ValueError("particle_counts must be positive")
        particle_count = nxp * nyp * nzp
        if particle_count > self.particle_capacity:
            raise ValueError("particle count exceeds capacity")
        self.particle_count = particle_count
        self._initialize_box_kernel(
            nxp,
            nyp,
            nzp,
            float(box_min_m[0]),
            float(box_min_m[1]),
            float(box_min_m[2]),
            float(box_max_m[0]),
            float(box_max_m[1]),
            float(box_max_m[2]),
            float(density_kgm3),
        )
        self._update_rest_center()

    @ti.kernel
    def _initialize_layered_tri_surface_kernel(
        self,
        centroid_m: ti.template(),
        normal: ti.template(),
        area_m2: ti.template(),
        face_region_id: ti.template(),
        face_count: ti.i32,
        layer_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        fixed_region_id: ti.i32,
        density_kgm3: ti.f32,
        primary_thickness_m: ti.f32,
        secondary_thickness_m: ti.f32,
    ):
        self.primary_mass_kg[None] = 0.0
        self.secondary_mass_kg[None] = 0.0
        for p in range(self.particle_capacity):
            if p < face_count * layer_count:
                face = p // layer_count
                layer = p - face * layer_count
                region = face_region_id[face]
                thickness = 0.0
                if region == primary_region_id:
                    thickness = primary_thickness_m
                elif region == secondary_region_id:
                    thickness = secondary_thickness_m
                elif fixed_region_id >= 0 and region == fixed_region_id:
                    thickness = primary_thickness_m
                layer_fraction = (
                    (ti.cast(layer, ti.f32) + 0.5) / ti.cast(layer_count, ti.f32)
                    - 0.5
                )
                area_weight = area_m2[face] / ti.cast(layer_count, ti.f32)
                volume = area_weight * thickness
                mass = density_kgm3 * volume
                position = centroid_m[face] + normal[face] * (layer_fraction * thickness)
                self.x[p] = position
                self.rest_x[p] = position
                self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.F[p] = self._identity()
                self.volume_m3[p] = volume
                self.area_weight_m2[p] = area_weight
                self.rest_area_weight_m2[p] = area_weight
                self.mass_kg[p] = mass
                self.region_id[p] = region
                fixed = 0
                if fixed_region_id >= 0 and region == fixed_region_id:
                    fixed = 1
                self.fixed_particle[p] = fixed
                self.surface_normal[p] = normal[face]
                self.rest_surface_normal[p] = normal[face]
                self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])
                if region == primary_region_id:
                    ti.atomic_add(self.primary_mass_kg[None], mass)
                elif region == secondary_region_id:
                    ti.atomic_add(self.secondary_mass_kg[None], mass)

    def initialize_layered_tri_surface(
        self,
        tri_surface,
        *,
        layer_count: int,
        primary_region_id: int,
        secondary_region_id: int,
        fixed_region_id: int = -1,
        density_kgm3: float,
        primary_thickness_m: float,
        secondary_thickness_m: float,
    ) -> None:
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if tri_surface.face_count <= 0:
            raise ValueError("tri_surface must contain faces")
        active_regions = set(
            int(region)
            for region in tri_surface.region_id.to_numpy()[: int(tri_surface.face_count)]
        )
        supported_regions = {int(primary_region_id), int(secondary_region_id)}
        fixed_region_clause = ""
        if int(fixed_region_id) >= 0:
            supported_regions.add(int(fixed_region_id))
            fixed_region_clause = f" and fixed_region_id={int(fixed_region_id)}"
        unsupported_regions = sorted(active_regions - supported_regions)
        if unsupported_regions:
            raise ValueError(
                "unsupported layered tri-surface region IDs for "
                f"NeoHookeanMpmState: {unsupported_regions}; expected only "
                f"primary_region_id={int(primary_region_id)} and "
                f"secondary_region_id={int(secondary_region_id)}"
                f"{fixed_region_clause}"
            )
        particle_count = int(tri_surface.face_count) * int(layer_count)
        if particle_count > self.particle_capacity:
            raise ValueError("particle count exceeds capacity")
        self.particle_count = particle_count
        self._initialize_layered_tri_surface_kernel(
            tri_surface.centroid_m,
            tri_surface.normal,
            tri_surface.area_m2,
            tri_surface.region_id,
            int(tri_surface.face_count),
            int(layer_count),
            int(primary_region_id),
            int(secondary_region_id),
            int(fixed_region_id),
            float(density_kgm3),
            float(primary_thickness_m),
            float(secondary_thickness_m),
        )
        self._update_rest_center()

    @ti.kernel
    def _set_layered_region_loads_kernel(
        self,
        particle_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
        primary_area_load_x_npm2: ti.f32,
        primary_area_load_y_npm2: ti.f32,
        primary_area_load_z_npm2: ti.f32,
        primary_interface_reaction_x_n: ti.f32,
        primary_interface_reaction_y_n: ti.f32,
        primary_interface_reaction_z_n: ti.f32,
        secondary_interface_reaction_x_n: ti.f32,
        secondary_interface_reaction_y_n: ti.f32,
        secondary_interface_reaction_z_n: ti.f32,
    ):
        primary_area_load = ti.Vector(
            [primary_area_load_x_npm2, primary_area_load_y_npm2, primary_area_load_z_npm2]
        )
        primary_reaction = ti.Vector(
            [primary_interface_reaction_x_n, primary_interface_reaction_y_n, primary_interface_reaction_z_n]
        )
        secondary_reaction = ti.Vector(
            [secondary_interface_reaction_x_n, secondary_interface_reaction_y_n, secondary_interface_reaction_z_n]
        )
        for p in range(self.particle_capacity):
            if p < particle_count:
                region = self.region_id[p]
                force = ti.Vector([0.0, 0.0, 0.0])
                if region == primary_region_id:
                    force += primary_area_load * self.area_weight_m2[p]
                    force += primary_reaction * self.mass_kg[p] / ti.max(self.primary_mass_kg[None], 1.0e-12)
                elif region == secondary_region_id:
                    force += secondary_reaction * self.mass_kg[p] / ti.max(self.secondary_mass_kg[None], 1.0e-12)
                self.external_force_n[p] = force

    def set_layered_region_loads(
        self,
        *,
        primary_region_id: int,
        secondary_region_id: int,
        primary_area_load_npm2: tuple[float, float, float],
        primary_interface_reaction_n: tuple[float, float, float],
        secondary_interface_reaction_n: tuple[float, float, float],
    ) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before setting loads")
        primary_area_load = _vector3(primary_area_load_npm2, "primary_area_load_npm2")
        primary_reaction = _vector3(primary_interface_reaction_n, "primary_interface_reaction_n")
        secondary_reaction = _vector3(secondary_interface_reaction_n, "secondary_interface_reaction_n")
        self._set_layered_region_loads_kernel(
            int(self.particle_count),
            int(primary_region_id),
            int(secondary_region_id),
            float(primary_area_load[0]),
            float(primary_area_load[1]),
            float(primary_area_load[2]),
            float(primary_reaction[0]),
            float(primary_reaction[1]),
            float(primary_reaction[2]),
            float(secondary_reaction[0]),
            float(secondary_reaction[1]),
            float(secondary_reaction[2]),
        )

    @ti.kernel
    def _set_region_normal_pressure_kernel(
        self,
        particle_count: ti.i32,
        region_id: ti.i32,
        pressure_pa: ti.f32,
    ):
        for p in range(self.particle_capacity):
            if p < particle_count:
                if self.region_id[p] == region_id:
                    self.external_force_n[p] = -pressure_pa * self.area_weight_m2[p] * self.surface_normal[p]
                else:
                    self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])

    def set_region_normal_pressure(
        self,
        *,
        region_id: int,
        pressure_pa: float,
    ) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before setting loads")
        self._set_region_normal_pressure_kernel(
            int(self.particle_count),
            int(region_id),
            float(pressure_pa),
        )

    @ti.kernel
    def _add_region_normal_pressure_kernel(
        self,
        particle_count: ti.i32,
        region_id: ti.i32,
        pressure_pa: ti.f32,
    ):
        for p in range(self.particle_capacity):
            if p < particle_count and self.region_id[p] == region_id:
                self.external_force_n[p] += (
                    -pressure_pa * self.area_weight_m2[p] * self.surface_normal[p]
                )

    def add_region_normal_pressure(
        self,
        *,
        region_id: int,
        pressure_pa: float,
    ) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before setting loads")
        self._add_region_normal_pressure_kernel(
            int(self.particle_count),
            int(region_id),
            float(pressure_pa),
        )

    @ti.kernel
    def _add_region_area_load_kernel(
        self,
        particle_count: ti.i32,
        region_id: ti.i32,
        area_load_x_npm2: ti.f32,
        area_load_y_npm2: ti.f32,
        area_load_z_npm2: ti.f32,
    ):
        area_load = ti.Vector(
            [area_load_x_npm2, area_load_y_npm2, area_load_z_npm2]
        )
        for p in range(self.particle_capacity):
            if p < particle_count and self.region_id[p] == region_id:
                self.external_force_n[p] += area_load * self.area_weight_m2[p]

    def add_region_area_load(
        self,
        *,
        region_id: int,
        area_load_npm2: tuple[float, float, float],
    ) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before setting loads")
        area_load = _vector3(area_load_npm2, "area_load_npm2")
        self._add_region_area_load_kernel(
            int(self.particle_count),
            int(region_id),
            float(area_load[0]),
            float(area_load[1]),
            float(area_load[2]),
        )

    @ti.kernel
    def _set_uniform_velocity_kernel(self, vx: ti.f32, vy: ti.f32, vz: ti.f32, particle_count: ti.i32):
        velocity = ti.Vector([vx, vy, vz])
        for p in range(self.particle_capacity):
            if p < particle_count:
                self.v[p] = velocity

    def set_uniform_velocity(self, velocity_mps: tuple[float, float, float]) -> None:
        self._set_uniform_velocity_kernel(
            float(velocity_mps[0]),
            float(velocity_mps[1]),
            float(velocity_mps[2]),
            int(self.particle_count),
        )

    @ti.kernel
    def _set_uniform_external_force_kernel(self, fx: ti.f32, fy: ti.f32, fz: ti.f32, particle_count: ti.i32):
        force = ti.Vector([fx, fy, fz])
        for p in range(self.particle_capacity):
            if p < particle_count:
                self.external_force_n[p] = force

    def set_uniform_external_force(self, force_n: tuple[float, float, float]) -> None:
        self._set_uniform_external_force_kernel(
            float(force_n[0]),
            float(force_n[1]),
            float(force_n[2]),
            int(self.particle_count),
        )

    @ti.func
    def _clear_report_func(self):
        self.report_total_mass_kg[None] = 0.0
        self.report_total_volume_m3[None] = 0.0
        self.report_particle_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_transfer_grid_momentum_kg_mps[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_external_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_current_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_rest_center_sum_m[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_radial_center_count[None] = 0
        self.report_particle_momentum_square_sum[None] = 0.0
        self.report_transfer_relative_error[None] = 0.0
        self.report_max_speed_mps[None] = 0.0
        self.report_max_abs_j[None] = 0.0
        self.report_deformation_clamp_count[None] = 0
        self.report_radial_stretch_sum[None] = 0.0
        self.report_radial_stretch_count[None] = 0
        self.report_max_radial_stretch_error[None] = 0.0
        self.report_active_grid_nodes[None] = 0
        self.report_grid_out_of_bounds_particle_count[None] = 0
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
            ]
        )
        self.report_count_snapshot[None] = ti.Vector([0, 0, 0, 0, 0, 0])

    @ti.func
    def _atomic_add_vector(self, field, value):
        ti.atomic_add(field[None].x, value.x)
        ti.atomic_add(field[None].y, value.y)
        ti.atomic_add(field[None].z, value.z)

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
    def _particle_grid_stencil_out_of_bounds(self, coord):
        out_of_bounds = 0
        if coord.x < 0.5 or coord.x >= ti.cast(self.nx, ti.f32) - 1.5:
            out_of_bounds = 1
        if coord.y < 0.5 or coord.y >= ti.cast(self.ny, ti.f32) - 1.5:
            out_of_bounds = 1
        if coord.z < 0.5 or coord.z >= ti.cast(self.nz, ti.f32) - 1.5:
            out_of_bounds = 1
        return out_of_bounds

    @ti.func
    def _update_surface_geometry_from_deformation(self, p):
        rest_area = self.rest_area_weight_m2[p]
        rest_normal = self.rest_surface_normal[p]
        if rest_area > 0.0 and rest_normal.norm() > 1.0e-12:
            Fp = self.F[p]
            J = Fp.determinant()
            if J > 1.0e-12:
                area_vector = J * (Fp.inverse().transpose() @ rest_normal)
                area_scale = area_vector.norm()
                if area_scale > 1.0e-12:
                    self.surface_normal[p] = area_vector / area_scale
                    self.area_weight_m2[p] = rest_area * area_scale

    @ti.func
    def _weights(self, fx):
        return (
            0.5 * (1.5 - fx) * (1.5 - fx),
            0.75 - (fx - 1.0) * (fx - 1.0),
            0.5 * (fx - 0.5) * (fx - 0.5),
        )

    @ti.kernel
    def _step_kernel(
        self,
        particle_count: ti.i32,
        dt_s: ti.f32,
        mu_pa: ti.f32,
        lambda_pa: ti.f32,
        velocity_damping: ti.f32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        for i, j, k in self.grid_mass_kg:
            self.grid_mass_kg[i, j, k] = 0.0
            self.grid_velocity_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_force_n[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        self._clear_report_func()

        for p in range(self.particle_capacity):
            if p < particle_count:
                Xp = self._particle_grid_coordinate(p)
                base = ti.cast(ti.floor(Xp - 0.5), ti.i32)
                fx = Xp - ti.cast(base, ti.f32)
                wx = self._weights(fx.x)
                wy = self._weights(fx.y)
                wz = self._weights(fx.z)
                raw_Fp = self.F[p]
                raw_J = raw_Fp.determinant()
                U, sig, V = ti.svd(raw_Fp)
                clamped = 0
                for axis in ti.static(range(3)):
                    raw_sigma = sig[axis, axis]
                    raw_sigma_abs = ti.abs(raw_sigma)
                    clamped_sigma = ti.min(
                        ti.max(raw_sigma_abs, MIN_DEFORMATION_SINGULAR_VALUE),
                        MAX_DEFORMATION_SINGULAR_VALUE,
                    )
                    if ti.abs(clamped_sigma - raw_sigma) > 1.0e-6:
                        clamped = 1
                    sig[axis, axis] = clamped_sigma
                if (U @ V.transpose()).determinant() < 0.0:
                    clamped = 1
                    for row in ti.static(range(3)):
                        U[row, 2] = -U[row, 2]
                Fp = U @ sig @ V.transpose()
                if Fp.determinant() <= 0.0:
                    clamped = 1
                    Fp[0, 2] = -Fp[0, 2]
                    Fp[1, 2] = -Fp[1, 2]
                    Fp[2, 2] = -Fp[2, 2]
                if raw_J <= 0.0:
                    clamped = 1
                if clamped == 1:
                    ti.atomic_add(self.report_deformation_clamp_count[None], 1)
                    self.F[p] = Fp
                J = Fp.determinant()
                FinvT = Fp.inverse().transpose()
                P = mu_pa * (Fp - FinvT) + lambda_pa * ti.log(J) * FinvT
                inv_dx2 = ti.Matrix(
                    [
                        [4.0 / (self.dx[0] * self.dx[0]), 0.0, 0.0],
                        [0.0, 4.0 / (self.dx[1] * self.dx[1]), 0.0],
                        [0.0, 0.0, 4.0 / (self.dx[2] * self.dx[2])],
                    ]
                )
                stress = -dt_s * self.volume_m3[p] * ((P @ Fp.transpose()) @ inv_dx2)
                affine = stress + self.mass_kg[p] * self.C[p]
                particle_momentum = self.mass_kg[p] * self.v[p]
                particle_external_force = self.external_force_n[p]
                if self.fixed_particle[p] != 0:
                    # Fixed particles anchor the grid with their mass but
                    # carry zero momentum, zero affine/stress contribution,
                    # and an inert external force.
                    particle_momentum = ti.Vector([0.0, 0.0, 0.0])
                    affine = ti.Matrix.zero(ti.f32, 3, 3)
                    particle_external_force = ti.Vector([0.0, 0.0, 0.0])
                self._atomic_add_vector(self.report_particle_momentum_kg_mps, particle_momentum)
                self._atomic_add_vector(self.report_external_force_n, particle_external_force)
                ti.atomic_add(self.report_total_mass_kg[None], self.mass_kg[p])
                ti.atomic_add(self.report_total_volume_m3[None], self.volume_m3[p])
                ti.atomic_add(
                    self.report_particle_momentum_square_sum[None],
                    particle_momentum.dot(particle_momentum),
                )
                ti.atomic_max(self.report_max_abs_j[None], ti.abs(J))

                if self._particle_grid_stencil_out_of_bounds(Xp) != 0:
                    ti.atomic_add(self.report_grid_out_of_bounds_particle_count[None], 1)
                else:
                    for ox, oy, oz in ti.static(ti.ndrange(3, 3, 3)):
                        node = base + ti.Vector([ox, oy, oz])
                        weight = wx[ox] * wy[oy] * wz[oz]
                        dpos = ti.Vector(
                            [
                                (ti.cast(ox, ti.f32) - fx.x) * self.dx[0],
                                (ti.cast(oy, ti.f32) - fx.y) * self.dx[1],
                                (ti.cast(oz, ti.f32) - fx.z) * self.dx[2],
                            ]
                        )
                        momentum = weight * (particle_momentum + affine @ dpos)
                        force = weight * particle_external_force
                        ti.atomic_add(self.grid_mass_kg[node], weight * self.mass_kg[p])
                        ti.atomic_add(self.grid_velocity_mps[node].x, momentum.x)
                        ti.atomic_add(self.grid_velocity_mps[node].y, momentum.y)
                        ti.atomic_add(self.grid_velocity_mps[node].z, momentum.z)
                        ti.atomic_add(self.grid_force_n[node].x, force.x)
                        ti.atomic_add(self.grid_force_n[node].y, force.y)
                        ti.atomic_add(self.grid_force_n[node].z, force.z)

        for i, j, k in self.grid_mass_kg:
            mass = self.grid_mass_kg[i, j, k]
            if mass > 1.0e-20:
                velocity = self.grid_velocity_mps[i, j, k] / mass
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

        for p in range(self.particle_capacity):
            if p < particle_count:
                Xp = self._particle_grid_coordinate(p)
                if self._particle_grid_stencil_out_of_bounds(Xp) == 0:
                    base = ti.cast(ti.floor(Xp - 0.5), ti.i32)
                    fx = Xp - ti.cast(base, ti.f32)
                    wx = self._weights(fx.x)
                    wy = self._weights(fx.y)
                    wz = self._weights(fx.z)
                    new_v = ti.Vector([0.0, 0.0, 0.0])
                    new_C = ti.Matrix.zero(ti.f32, 3, 3)
                    for ox, oy, oz in ti.static(ti.ndrange(3, 3, 3)):
                        node = base + ti.Vector([ox, oy, oz])
                        weight = wx[ox] * wy[oy] * wz[oz]
                        dpos = ti.Vector(
                            [
                                (ti.cast(ox, ti.f32) - fx.x) * self.dx[0],
                                (ti.cast(oy, ti.f32) - fx.y) * self.dx[1],
                                (ti.cast(oz, ti.f32) - fx.z) * self.dx[2],
                            ]
                        )
                        grid_v = self.grid_velocity_mps[node]
                        new_v += weight * grid_v
                        new_C += weight * (
                            grid_v.outer_product(
                                ti.Vector(
                                    [
                                        4.0 * dpos.x / (self.dx[0] * self.dx[0]),
                                        4.0 * dpos.y / (self.dx[1] * self.dx[1]),
                                        4.0 * dpos.z / (self.dx[2] * self.dx[2]),
                                    ]
                                )
                            )
                        )
                    if self.fixed_particle[p] == 0:
                        self.v[p] = new_v
                        self.C[p] = new_C
                        self.x[p] += dt_s * new_v
                        self.F[p] = (self._identity() + dt_s * new_C) @ self.F[p]
                if self.fixed_particle[p] != 0:
                    # Fixed particles stay frozen: zero velocity, frozen
                    # position, rest-identity deformation, no affine state.
                    self.v[p] = ti.Vector([0.0, 0.0, 0.0])
                    self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                    self.F[p] = self._identity()
                self._update_surface_geometry_from_deformation(p)
                report_coord = self._particle_grid_coordinate(p)
                if self._particle_grid_stencil_out_of_bounds(report_coord) == 0:
                    if self.region_id[p] == primary_region_id:
                        self._atomic_add_vector(self.report_primary_displacement_sum_m, self.x[p] - self.rest_x[p])
                        self._atomic_add_vector(self.report_primary_velocity_sum_mps, self.v[p])
                        ti.atomic_add(self.report_primary_count[None], 1)
                    elif self.region_id[p] == secondary_region_id:
                        self._atomic_add_vector(self.report_secondary_displacement_sum_m, self.x[p] - self.rest_x[p])
                        self._atomic_add_vector(self.report_secondary_velocity_sum_mps, self.v[p])
                        ti.atomic_add(self.report_secondary_count[None], 1)

        for p in range(self.particle_capacity):
            if p < particle_count:
                coord = self._particle_grid_coordinate(p)
                if self._particle_grid_stencil_out_of_bounds(coord) == 0:
                    self._atomic_add_vector(self.report_current_center_sum_m, self.x[p])
                    self._atomic_add_vector(self.report_radial_rest_center_sum_m, self.rest_x[p])
                    ti.atomic_add(self.report_radial_center_count[None], 1)
        radial_center_count = ti.max(ti.cast(self.report_radial_center_count[None], ti.f32), 1.0)
        current_center = self.report_current_center_sum_m[None] / radial_center_count
        rest_center = self.report_radial_rest_center_sum_m[None] / radial_center_count
        for p in range(self.particle_capacity):
            if p < particle_count:
                coord = self._particle_grid_coordinate(p)
                if self._particle_grid_stencil_out_of_bounds(coord) == 0:
                    rest_radius = (self.rest_x[p] - rest_center).norm()
                    if rest_radius > 1.0e-12:
                        radial_stretch = (self.x[p] - current_center).norm() / rest_radius
                        ti.atomic_add(self.report_radial_stretch_sum[None], radial_stretch)
                        ti.atomic_add(self.report_radial_stretch_count[None], 1)
                        ti.atomic_max(
                            self.report_max_radial_stretch_error[None],
                            ti.abs(radial_stretch - 1.0),
                        )

        expected_grid_momentum = self.report_particle_momentum_kg_mps[None] + dt_s * self.report_external_force_n[None]
        denominator = ti.max(
            expected_grid_momentum.norm(),
            ti.sqrt(self.report_particle_momentum_square_sum[None]),
            1.0e-20,
        )
        self.report_transfer_relative_error[None] = (
            self.report_transfer_grid_momentum_kg_mps[None] - expected_grid_momentum
        ).norm() / denominator
        self.report_float_snapshot[None] = ti.Vector(
            [
                self.report_total_mass_kg[None],
                self.report_total_volume_m3[None],
                self.report_particle_momentum_kg_mps[None].x,
                self.report_particle_momentum_kg_mps[None].y,
                self.report_particle_momentum_kg_mps[None].z,
                self.report_grid_momentum_kg_mps[None].x,
                self.report_grid_momentum_kg_mps[None].y,
                self.report_grid_momentum_kg_mps[None].z,
                self.report_external_force_n[None].x,
                self.report_external_force_n[None].y,
                self.report_external_force_n[None].z,
                self.report_transfer_relative_error[None],
                self.report_max_speed_mps[None],
                self.report_max_abs_j[None],
                self.report_radial_stretch_sum[None],
                self.report_max_radial_stretch_error[None],
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
            ]
        )
        self.report_count_snapshot[None] = ti.Vector(
            [
                self.report_active_grid_nodes[None],
                self.report_grid_out_of_bounds_particle_count[None],
                self.report_deformation_clamp_count[None],
                self.report_radial_stretch_count[None],
                self.report_primary_count[None],
                self.report_secondary_count[None],
            ]
        )
        packed_values = self.report_float_snapshot[None]
        packed_counts = self.report_count_snapshot[None]
        for snapshot_index in ti.static(range(28)):
            self.report_host_snapshot[snapshot_index] = packed_values[snapshot_index]
        for snapshot_index in ti.static(range(6)):
            self.report_host_snapshot[28 + snapshot_index] = ti.cast(
                packed_counts[snapshot_index],
                ti.f32,
            )

    def step(
        self,
        *,
        dt_s: float,
        mu_pa: float,
        lambda_pa: float,
        primary_region_id: int,
        secondary_region_id: int,
        velocity_damping: float = 1.0,
        read_report: bool = True,
    ) -> NeoHookeanMpmReport | None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before stepping")
        self._step_kernel(
            int(self.particle_count),
            float(dt_s),
            float(mu_pa),
            float(lambda_pa),
            float(velocity_damping),
            int(primary_region_id),
            int(secondary_region_id),
        )
        if not read_report:
            self.last_report_host_reads = 0
            return None
        return self.report()

    @ti.kernel
    def _save_state_kernel(self, particle_count: ti.i32):
        for p in range(particle_count):
            self.saved_x[p] = self.x[p]
            self.saved_v[p] = self.v[p]
            self.saved_C[p] = self.C[p]
            self.saved_F[p] = self.F[p]

    @ti.kernel
    def _restore_state_kernel(self, particle_count: ti.i32):
        for p in range(particle_count):
            self.x[p] = self.saved_x[p]
            self.v[p] = self.saved_v[p]
            self.C[p] = self.saved_C[p]
            self.F[p] = self.saved_F[p]
            self._update_surface_geometry_from_deformation(p)
            self.external_force_n[p] = ti.Vector([0.0, 0.0, 0.0])

    def save_state(self) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before saving state")
        self._save_state_kernel(int(self.particle_count))

    def restore_state(self) -> None:
        if self.particle_count <= 0:
            raise ValueError("initialize particles before restoring state")
        self._restore_state_kernel(int(self.particle_count))

    def report(self) -> NeoHookeanMpmReport:
        snapshot = self.report_host_snapshot.to_numpy()
        values = snapshot[:28]
        counts = snapshot[28:34]
        self.last_report_host_reads = 1
        _raise_if_all_particles_out_of_bounds(int(self.particle_count), int(counts[1]))
        total_volume_m3 = float(values[1])
        particle_spacing_m = 0.0
        if total_volume_m3 > 0.0 and self.particle_count > 0:
            particle_spacing_m = (total_volume_m3 / self.particle_count) ** (1.0 / 3.0)
        primary_count = max(int(counts[4]), 1)
        secondary_count = max(int(counts[5]), 1)
        radial_stretch_count = max(int(counts[3]), 1)
        return NeoHookeanMpmReport(
            particle_count=int(self.particle_count),
            active_grid_nodes=int(counts[0]),
            grid_out_of_bounds_particle_count=int(counts[1]),
            particle_spacing_m=particle_spacing_m,
            grid_spacing_m=self.dx,
            total_mass_kg=float(values[0]),
            total_volume_m3=total_volume_m3,
            primary_mean_displacement_m=(
                float(values[16]) / primary_count,
                float(values[17]) / primary_count,
                float(values[18]) / primary_count,
            ),
            primary_mean_velocity_mps=(
                float(values[19]) / primary_count,
                float(values[20]) / primary_count,
                float(values[21]) / primary_count,
            ),
            secondary_mean_displacement_m=(
                float(values[22]) / secondary_count,
                float(values[23]) / secondary_count,
                float(values[24]) / secondary_count,
            ),
            secondary_mean_velocity_mps=(
                float(values[25]) / secondary_count,
                float(values[26]) / secondary_count,
                float(values[27]) / secondary_count,
            ),
            particle_momentum_kg_mps=(float(values[2]), float(values[3]), float(values[4])),
            grid_momentum_kg_mps=(float(values[5]), float(values[6]), float(values[7])),
            external_force_n=(float(values[8]), float(values[9]), float(values[10])),
            transfer_relative_error=float(values[11]),
            max_speed_mps=float(values[12]),
            max_abs_j=float(values[13]),
            deformation_clamp_count=int(counts[2]),
            mean_radial_stretch=float(values[14]) / radial_stretch_count,
            max_radial_stretch_error=float(values[15]),
        )
