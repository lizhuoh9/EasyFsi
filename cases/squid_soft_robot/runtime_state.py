from collections.abc import Sequence

import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.runtime import init_taichi

from .history import divergence_sample_report_fields
from .setup import cartesian_grid_axis_min_spacing_m, nozzle_taper_geometry
from .source_config import _vector3
from .spec import SquidReducedSpec


@ti.data_oriented
class ReducedSquidFSI:
    def __init__(
        self,
        spec: SquidReducedSpec,
        runtime: TaichiRuntimeConfig,
    ):
        init_taichi(runtime)
        self.spec = spec
        self.fluid = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=spec.fluid_bounds_min_m,
                bounds_max_m=spec.fluid_bounds_max_m,
                grid_nodes=spec.grid_nodes,
                density_kgm3=spec.water_density_kgm3,
                viscosity_pa_s=spec.water_viscosity_pa_s,
                dt_s=spec.dt_s,
                cartesian_grid=spec.cartesian_grid,
                graded_grid=spec.graded_grid,
            ),
            runtime=runtime,
        )

        self.time_s = ti.field(dtype=ti.f32, shape=())
        self.pressure_load_pa = ti.field(dtype=ti.f32, shape=())
        self.hydraulic_pressure_pa = ti.field(dtype=ti.f32, shape=())
        self.main_w_m = ti.field(dtype=ti.f32, shape=())
        self.main_v_mps = ti.field(dtype=ti.f32, shape=())
        self.tail_w_m = ti.field(dtype=ti.f32, shape=())
        self.tail_v_mps = ti.field(dtype=ti.f32, shape=())
        self.primary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.secondary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.volume_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.nozzle_velocity_z_mps = ti.field(dtype=ti.f32, shape=())
        self.max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.lip_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.outlet_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.downstream_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.lip_sample_count = ti.field(dtype=ti.i32, shape=())
        self.outlet_sample_count = ti.field(dtype=ti.i32, shape=())
        self.downstream_sample_count = ti.field(dtype=ti.i32, shape=())
        self.sample_report_float_snapshot = ti.Vector.field(15, dtype=ti.f32, shape=())
        self.sample_report_count_snapshot = ti.Vector.field(3, dtype=ti.i32, shape=())
        self.sample_report_host_snapshot = ti.field(dtype=ti.f64, shape=18)
        self.saved_time_s = ti.field(dtype=ti.f32, shape=())
        self.saved_pressure_load_pa = ti.field(dtype=ti.f32, shape=())
        self.saved_hydraulic_pressure_pa = ti.field(dtype=ti.f32, shape=())
        self.saved_main_w_m = ti.field(dtype=ti.f32, shape=())
        self.saved_main_v_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_tail_w_m = ti.field(dtype=ti.f32, shape=())
        self.saved_tail_v_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_primary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.saved_secondary_interface_reaction_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.saved_volume_flux_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_nozzle_velocity_z_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_max_speed_mps = ti.field(dtype=ti.f32, shape=())
        self.saved_lip_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_outlet_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_downstream_flow_z_m3s = ti.field(dtype=ti.f32, shape=())
        self.saved_lip_sample_count = ti.field(dtype=ti.i32, shape=())
        self.saved_outlet_sample_count = ti.field(dtype=ti.i32, shape=())
        self.saved_downstream_sample_count = ti.field(dtype=ti.i32, shape=())
        self.last_sample_report_host_reads = 0

        self._reset_kernel()

    @ti.kernel
    def _reset_kernel(self):
        self.time_s[None] = 0.0
        self.pressure_load_pa[None] = 0.0
        self.hydraulic_pressure_pa[None] = 0.0
        self.main_w_m[None] = 0.0
        self.main_v_mps[None] = 0.0
        self.tail_w_m[None] = 0.0
        self.tail_v_mps[None] = 0.0
        self.primary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.secondary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.volume_flux_m3s[None] = 0.0
        self.nozzle_velocity_z_mps[None] = 0.0
        self.max_speed_mps[None] = 0.0
        self.lip_flow_z_m3s[None] = 0.0
        self.outlet_flow_z_m3s[None] = 0.0
        self.downstream_flow_z_m3s[None] = 0.0
        self.lip_sample_count[None] = 0
        self.outlet_sample_count[None] = 0
        self.downstream_sample_count[None] = 0
        self.sample_report_float_snapshot[None] = ti.Vector(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.sample_report_count_snapshot[None] = ti.Vector([0, 0, 0])
        for index in ti.static(range(18)):
            self.sample_report_host_snapshot[index] = 0.0
        self.saved_time_s[None] = 0.0
        self.saved_pressure_load_pa[None] = 0.0
        self.saved_hydraulic_pressure_pa[None] = 0.0
        self.saved_main_w_m[None] = 0.0
        self.saved_main_v_mps[None] = 0.0
        self.saved_tail_w_m[None] = 0.0
        self.saved_tail_v_mps[None] = 0.0
        self.saved_primary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.saved_secondary_interface_reaction_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.saved_volume_flux_m3s[None] = 0.0
        self.saved_nozzle_velocity_z_mps[None] = 0.0
        self.saved_max_speed_mps[None] = 0.0
        self.saved_lip_flow_z_m3s[None] = 0.0
        self.saved_outlet_flow_z_m3s[None] = 0.0
        self.saved_downstream_flow_z_m3s[None] = 0.0
        self.saved_lip_sample_count[None] = 0
        self.saved_outlet_sample_count[None] = 0
        self.saved_downstream_sample_count[None] = 0

    @ti.kernel
    def save_reduced_state_kernel(self):
        self.saved_time_s[None] = self.time_s[None]
        self.saved_pressure_load_pa[None] = self.pressure_load_pa[None]
        self.saved_hydraulic_pressure_pa[None] = self.hydraulic_pressure_pa[None]
        self.saved_main_w_m[None] = self.main_w_m[None]
        self.saved_main_v_mps[None] = self.main_v_mps[None]
        self.saved_tail_w_m[None] = self.tail_w_m[None]
        self.saved_tail_v_mps[None] = self.tail_v_mps[None]
        self.saved_primary_interface_reaction_force_n[None] = self.primary_interface_reaction_force_n[None]
        self.saved_secondary_interface_reaction_force_n[None] = self.secondary_interface_reaction_force_n[None]
        self.saved_volume_flux_m3s[None] = self.volume_flux_m3s[None]
        self.saved_nozzle_velocity_z_mps[None] = self.nozzle_velocity_z_mps[None]
        self.saved_max_speed_mps[None] = self.max_speed_mps[None]
        self.saved_lip_flow_z_m3s[None] = self.lip_flow_z_m3s[None]
        self.saved_outlet_flow_z_m3s[None] = self.outlet_flow_z_m3s[None]
        self.saved_downstream_flow_z_m3s[None] = self.downstream_flow_z_m3s[None]
        self.saved_lip_sample_count[None] = self.lip_sample_count[None]
        self.saved_outlet_sample_count[None] = self.outlet_sample_count[None]
        self.saved_downstream_sample_count[None] = self.downstream_sample_count[None]

    @ti.kernel
    def restore_reduced_state_kernel(self):
        self.time_s[None] = self.saved_time_s[None]
        self.pressure_load_pa[None] = self.saved_pressure_load_pa[None]
        self.hydraulic_pressure_pa[None] = self.saved_hydraulic_pressure_pa[None]
        self.main_w_m[None] = self.saved_main_w_m[None]
        self.main_v_mps[None] = self.saved_main_v_mps[None]
        self.tail_w_m[None] = self.saved_tail_w_m[None]
        self.tail_v_mps[None] = self.saved_tail_v_mps[None]
        self.primary_interface_reaction_force_n[None] = self.saved_primary_interface_reaction_force_n[None]
        self.secondary_interface_reaction_force_n[None] = self.saved_secondary_interface_reaction_force_n[None]
        self.volume_flux_m3s[None] = self.saved_volume_flux_m3s[None]
        self.nozzle_velocity_z_mps[None] = self.saved_nozzle_velocity_z_mps[None]
        self.max_speed_mps[None] = self.saved_max_speed_mps[None]
        self.lip_flow_z_m3s[None] = self.saved_lip_flow_z_m3s[None]
        self.outlet_flow_z_m3s[None] = self.saved_outlet_flow_z_m3s[None]
        self.downstream_flow_z_m3s[None] = self.saved_downstream_flow_z_m3s[None]
        self.lip_sample_count[None] = self.saved_lip_sample_count[None]
        self.outlet_sample_count[None] = self.saved_outlet_sample_count[None]
        self.downstream_sample_count[None] = self.saved_downstream_sample_count[None]

    def save_reduced_state(self) -> None:
        self.save_reduced_state_kernel()

    def restore_reduced_state(self) -> None:
        self.restore_reduced_state_kernel()

    @ti.kernel
    def set_interface_reaction_kernel(
        self,
        primary_force_n: ti.types.vector(3, ti.f32),
        secondary_force_n: ti.types.vector(3, ti.f32),
    ):
        self.primary_interface_reaction_force_n[None] = primary_force_n
        self.secondary_interface_reaction_force_n[None] = secondary_force_n

    def set_interface_reaction(
        self,
        *,
        primary_force_n: Sequence[float],
        secondary_force_n: Sequence[float],
    ) -> None:
        primary = _vector3(primary_force_n, name="primary_force_n")
        secondary = _vector3(secondary_force_n, name="secondary_force_n")
        self.set_interface_reaction_kernel(
            ti.Vector(primary),
            ti.Vector(secondary),
        )

    @ti.kernel
    def set_structure_state_kernel(
        self,
        time_s: ti.f32,
        pressure_pa: ti.f32,
        hydraulic_pressure_pa: ti.f32,
        main_displacement_z_m: ti.f32,
        main_velocity_z_mps: ti.f32,
        tail_displacement_z_m: ti.f32,
        tail_velocity_z_mps: ti.f32,
        volume_flux_m3s: ti.f32,
        nozzle_velocity_z_mps: ti.f32,
    ):
        self.time_s[None] = time_s
        self.pressure_load_pa[None] = pressure_pa
        self.hydraulic_pressure_pa[None] = hydraulic_pressure_pa
        self.main_w_m[None] = main_displacement_z_m
        self.main_v_mps[None] = main_velocity_z_mps
        self.tail_w_m[None] = tail_displacement_z_m
        self.tail_v_mps[None] = tail_velocity_z_mps
        self.volume_flux_m3s[None] = volume_flux_m3s
        self.nozzle_velocity_z_mps[None] = nozzle_velocity_z_mps

    def set_structure_state(
        self,
        *,
        time_s: float,
        pressure_pa: float,
        hydraulic_pressure_pa: float,
        main_displacement_z_m: float,
        main_velocity_z_mps: float,
        tail_displacement_z_m: float,
        tail_velocity_z_mps: float,
        volume_flux_m3s: float,
        nozzle_velocity_z_mps: float,
    ) -> None:
        self.set_structure_state_kernel(
            float(time_s),
            float(pressure_pa),
            float(hydraulic_pressure_pa),
            float(main_displacement_z_m),
            float(main_velocity_z_mps),
            float(tail_displacement_z_m),
            float(tail_velocity_z_mps),
            float(volume_flux_m3s),
            float(nozzle_velocity_z_mps),
        )

    @ti.func
    def _cell_disk_intersects_axisymmetric_region(
        self,
        rx: ti.f32,
        ry: ti.f32,
        half_width_x_m: ti.f32,
        half_width_y_m: ti.f32,
        radius_m: ti.f32,
    ) -> ti.i32:
        closest_x = ti.max(ti.abs(rx) - half_width_x_m, 0.0)
        closest_y = ti.max(ti.abs(ry) - half_width_y_m, 0.0)
        intersects = closest_x * closest_x + closest_y * closest_y <= radius_m * radius_m
        return ti.cast(intersects, ti.i32)

    @ti.func
    def _cell_z_interval_intersects(
        self,
        cell_min_z_m: ti.f32,
        cell_max_z_m: ti.f32,
        lower_z_m: ti.f32,
        upper_z_m: ti.f32,
    ) -> ti.i32:
        return ti.cast(cell_max_z_m >= lower_z_m and cell_min_z_m <= upper_z_m, ti.i32)

    @ti.func
    def _conservative_taper_radius_m(
        self,
        cell_min_z_m: ti.f32,
        cell_max_z_m: ti.f32,
        nozzle_radius_m: ti.f32,
        nozzle_taper_enabled: ti.i32,
        nozzle_taper_start_z_m: ti.f32,
        nozzle_taper_end_z_m: ti.f32,
        nozzle_taper_inlet_radius_m: ti.f32,
    ) -> ti.f32:
        radius_m = nozzle_radius_m
        if (
            nozzle_taper_enabled == 1
            and cell_max_z_m >= nozzle_taper_start_z_m
            and cell_min_z_m <= nozzle_taper_end_z_m
        ):
            overlap_hi_z_m = ti.min(cell_max_z_m, nozzle_taper_end_z_m)
            fraction = (overlap_hi_z_m - nozzle_taper_start_z_m) / ti.max(
                nozzle_taper_end_z_m - nozzle_taper_start_z_m,
                1.0e-12,
            )
            radius_m = nozzle_radius_m + (
                nozzle_taper_inlet_radius_m - nozzle_radius_m
            ) * ti.min(ti.max(fraction, 0.0), 1.0)
        return radius_m

    @ti.kernel
    def mark_reduced_squid_water_domain_kernel(
        self,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        bounds_min_z: ti.f32,
        center_x_m: ti.f32,
        center_y_m: ti.f32,
        chamber_radius_m: ti.f32,
        chamber_z_min_m: ti.f32,
        chamber_z_max_m: ti.f32,
        nozzle_radius_m: ti.f32,
        nozzle_z_max_m: ti.f32,
        downstream_z_m: ti.f32,
        outlet_plume_radius_m: ti.f32,
        nozzle_taper_enabled: ti.i32,
        nozzle_taper_start_z_m: ti.f32,
        nozzle_taper_end_z_m: ti.f32,
        nozzle_taper_inlet_radius_m: ti.f32,
        downstream_farfield_open_enabled: ti.i32,
        downstream_farfield_open_z_max_m: ti.f32,
        preserve_existing_obstacles: ti.i32,
    ):
        for i, j, k in self.fluid.obstacle:
            existing_obstacle = self.fluid.obstacle[i, j, k] != 0
            x = cell_center_x_m[i]
            y = cell_center_y_m[j]
            z = cell_center_z_m[k]
            rx = x - center_x_m
            ry = y - center_y_m
            half_width_x_m = 0.5 * cell_width_x_m[i]
            half_width_y_m = 0.5 * cell_width_y_m[j]
            half_width_z_m = 0.5 * cell_width_z_m[k]
            cell_min_z_m = z - half_width_z_m
            cell_max_z_m = z + half_width_z_m
            chamber = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    chamber_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    chamber_z_min_m,
                    chamber_z_max_m,
                ) == 1
            )
            local_nozzle_radius_m = nozzle_radius_m
            local_nozzle_radius_m = self._conservative_taper_radius_m(
                cell_min_z_m,
                cell_max_z_m,
                nozzle_radius_m,
                nozzle_taper_enabled,
                nozzle_taper_start_z_m,
                nozzle_taper_end_z_m,
                nozzle_taper_inlet_radius_m,
            )
            nozzle = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    local_nozzle_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    downstream_z_m,
                    nozzle_z_max_m,
                ) == 1
            )
            outlet_plume = (
                self._cell_disk_intersects_axisymmetric_region(
                    rx,
                    ry,
                    half_width_x_m,
                    half_width_y_m,
                    outlet_plume_radius_m,
                ) == 1
                and self._cell_z_interval_intersects(
                    cell_min_z_m,
                    cell_max_z_m,
                    bounds_min_z,
                    downstream_z_m,
                ) == 1
            )
            downstream_farfield = (
                downstream_farfield_open_enabled == 1
                and cell_min_z_m <= downstream_farfield_open_z_max_m
            )
            reduced_water = chamber or nozzle or outlet_plume or downstream_farfield
            if preserve_existing_obstacles == 1:
                self.fluid.obstacle[i, j, k] = 0 if reduced_water and not existing_obstacle else 1
            else:
                self.fluid.obstacle[i, j, k] = 0 if reduced_water else 1

    def _apply_reduced_squid_water_domain(
        self,
        *,
        preserve_existing_obstacles: bool,
    ) -> None:
        spec = self.spec
        taper_start_z_m, taper_end_z_m, taper_inlet_radius_m = nozzle_taper_geometry(spec)
        self.mark_reduced_squid_water_domain_kernel(
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.cell_width_x_m,
            self.fluid.cell_width_y_m,
            self.fluid.cell_width_z_m,
            float(spec.fluid_bounds_min_m[2]),
            float(spec.monitor_center_x_m),
            float(spec.monitor_center_y_m),
            float(spec.chamber_radius_m),
            float(spec.chamber_z_min_m),
            float(spec.chamber_z_max_m),
            float(spec.nozzle_radius_m),
            float(spec.nozzle_z_max_m),
            float(spec.downstream_z_m),
            float(spec.outlet_plume_radius_m),
            1 if spec.nozzle_taper_enabled else 0,
            float(taper_start_z_m),
            float(taper_end_z_m),
            float(taper_inlet_radius_m),
            1 if spec.downstream_farfield_open_enabled else 0,
            float(spec.downstream_farfield_open_z_max_m),
            1 if preserve_existing_obstacles else 0,
        )

    def mark_reduced_squid_water_domain(self) -> None:
        self._apply_reduced_squid_water_domain(preserve_existing_obstacles=False)

    def intersect_current_obstacles_with_reduced_squid_water_domain(self) -> None:
        self._apply_reduced_squid_water_domain(preserve_existing_obstacles=True)

    @ti.func
    def _section_area_fraction(
        self,
        rx,
        ry,
        half_width_x_m,
        half_width_y_m,
        radius_m,
    ):
        hits = 0
        for sx, sy in ti.ndrange(8, 8):
            sample_x = rx + (-half_width_x_m + (ti.cast(sx, ti.f32) + 0.5) * half_width_x_m / 4.0)
            sample_y = ry + (-half_width_y_m + (ti.cast(sy, ti.f32) + 0.5) * half_width_y_m / 4.0)
            if sample_x * sample_x + sample_y * sample_y <= radius_m * radius_m:
                hits += 1
        return ti.cast(hits, ti.f32) / 64.0

    @ti.func
    def _accumulate_section(
        self,
        velocity_z,
        z,
        target_z,
        dz,
        rx,
        ry,
        radius_m,
        cell_area_m2,
        cell_width_x_m,
        cell_width_y_m,
        section_id,
    ):
        if ti.abs(z - target_z) <= 0.5 * dz:
            area_fraction = self._section_area_fraction(
                rx,
                ry,
                0.5 * cell_width_x_m,
                0.5 * cell_width_y_m,
                radius_m,
            )
            section_area_m2 = cell_area_m2 * area_fraction
            section_flux_m3s = velocity_z * section_area_m2
            if section_id == 0:
                ti.atomic_add(self.lip_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.lip_sample_count[None], 1)
            elif section_id == 1:
                ti.atomic_add(self.outlet_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.outlet_sample_count[None], 1)
            else:
                ti.atomic_add(self.downstream_flow_z_m3s[None], section_flux_m3s)
                if area_fraction > 0.0:
                    ti.atomic_add(self.downstream_sample_count[None], 1)

    @ti.kernel
    def sample_sections_kernel(
        self,
        velocity: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_x_m: ti.f32,
        center_y_m: ti.f32,
        lip_radius_m: ti.f32,
        outlet_radius_m: ti.f32,
        downstream_radius_m: ti.f32,
        lip_z_m: ti.f32,
        outlet_z_m: ti.f32,
        downstream_z_m: ti.f32,
    ):
        self.lip_flow_z_m3s[None] = 0.0
        self.outlet_flow_z_m3s[None] = 0.0
        self.downstream_flow_z_m3s[None] = 0.0
        self.lip_sample_count[None] = 0
        self.outlet_sample_count[None] = 0
        self.downstream_sample_count[None] = 0
        self.max_speed_mps[None] = 0.0
        for i, j, k in velocity:
            if self.fluid.obstacle[i, j, k] == 0:
                x = cell_center_x_m[i]
                y = cell_center_y_m[j]
                z = cell_center_z_m[k]
                rx = x - center_x_m
                ry = y - center_y_m
                vz = velocity[i, j, k].z
                dz = cell_width_z_m[k]
                cell_width_x = cell_width_x_m[i]
                cell_width_y = cell_width_y_m[j]
                cell_area_m2 = cell_width_x_m[i] * cell_width_y_m[j]
                self._accumulate_section(
                    vz,
                    z,
                    lip_z_m,
                    dz,
                    rx,
                    ry,
                    lip_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    0,
                )
                self._accumulate_section(
                    vz,
                    z,
                    outlet_z_m,
                    dz,
                    rx,
                    ry,
                    outlet_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    1,
                )
                self._accumulate_section(
                    vz,
                    z,
                    downstream_z_m,
                    dz,
                    rx,
                    ry,
                    downstream_radius_m,
                    cell_area_m2,
                    cell_width_x,
                    cell_width_y,
                    2,
                )
                ti.atomic_max(self.max_speed_mps[None], velocity[i, j, k].norm())
        self.sample_report_float_snapshot[None] = ti.Vector(
            [
                self.time_s[None],
                self.pressure_load_pa[None],
                self.hydraulic_pressure_pa[None],
                self.main_w_m[None],
                self.main_v_mps[None],
                self.tail_w_m[None],
                self.tail_v_mps[None],
                self.primary_interface_reaction_force_n[None].z,
                self.secondary_interface_reaction_force_n[None].z,
                self.volume_flux_m3s[None],
                self.nozzle_velocity_z_mps[None],
                self.lip_flow_z_m3s[None],
                self.outlet_flow_z_m3s[None],
                self.downstream_flow_z_m3s[None],
                self.max_speed_mps[None],
            ]
        )
        self.sample_report_count_snapshot[None] = ti.Vector(
            [
                self.lip_sample_count[None],
                self.outlet_sample_count[None],
                self.downstream_sample_count[None],
            ]
        )
        self.sample_report_host_snapshot[0] = ti.cast(self.time_s[None], ti.f64)
        self.sample_report_host_snapshot[1] = ti.cast(self.pressure_load_pa[None], ti.f64)
        self.sample_report_host_snapshot[2] = ti.cast(self.hydraulic_pressure_pa[None], ti.f64)
        self.sample_report_host_snapshot[3] = ti.cast(self.main_w_m[None], ti.f64)
        self.sample_report_host_snapshot[4] = ti.cast(self.main_v_mps[None], ti.f64)
        self.sample_report_host_snapshot[5] = ti.cast(self.tail_w_m[None], ti.f64)
        self.sample_report_host_snapshot[6] = ti.cast(self.tail_v_mps[None], ti.f64)
        self.sample_report_host_snapshot[7] = ti.cast(
            self.primary_interface_reaction_force_n[None].z,
            ti.f64,
        )
        self.sample_report_host_snapshot[8] = ti.cast(
            self.secondary_interface_reaction_force_n[None].z,
            ti.f64,
        )
        self.sample_report_host_snapshot[9] = ti.cast(self.volume_flux_m3s[None], ti.f64)
        self.sample_report_host_snapshot[10] = ti.cast(self.nozzle_velocity_z_mps[None], ti.f64)
        self.sample_report_host_snapshot[11] = ti.cast(self.lip_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[12] = ti.cast(self.outlet_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[13] = ti.cast(self.downstream_flow_z_m3s[None], ti.f64)
        self.sample_report_host_snapshot[14] = ti.cast(self.max_speed_mps[None], ti.f64)
        self.sample_report_host_snapshot[15] = ti.cast(self.lip_sample_count[None], ti.f64)
        self.sample_report_host_snapshot[16] = ti.cast(self.outlet_sample_count[None], ti.f64)
        self.sample_report_host_snapshot[17] = ti.cast(self.downstream_sample_count[None], ti.f64)

    def project_and_sample(
        self,
        projection_iterations: int,
        pressure_outlet_zmin: bool,
    ) -> dict[str, object]:
        divergence = self.fluid.project(
            iterations=projection_iterations,
            pressure_outlet_zmin=pressure_outlet_zmin,
        )
        return self.sample_after_projection(divergence)

    def sample_after_projection(
        self,
        divergence: dict[str, float],
        *,
        dt_s: float | None = None,
    ) -> dict[str, object]:
        spec = self.spec
        self.sample_sections_kernel(
            self.fluid.velocity,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.cell_width_x_m,
            self.fluid.cell_width_y_m,
            self.fluid.cell_width_z_m,
            float(spec.monitor_center_x_m),
            float(spec.monitor_center_y_m),
            float(spec.monitor_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.lip_z_m),
            float(spec.outlet_z_m),
            float(spec.downstream_z_m),
        )
        h = min(cartesian_grid_axis_min_spacing_m(self.fluid.grid))
        sample_values = self.sample_report_float_snapshot[None]
        sample_counts = self.sample_report_count_snapshot[None]
        self.last_sample_report_host_reads = 1
        max_speed = float(sample_values[14])
        cfl_dt_s = float(self.spec.dt_s) if dt_s is None else float(dt_s)
        return {
            "time_s": float(sample_values[0]),
            "pressure_load_pa": float(sample_values[1]),
            "hydraulic_pressure_pa": float(sample_values[2]),
            "main_displacement_z_m": float(sample_values[3]),
            "main_velocity_z_mps": float(sample_values[4]),
            "tail_displacement_z_m": float(sample_values[5]),
            "tail_velocity_z_mps": float(sample_values[6]),
            "main_interface_reaction_z_n": float(sample_values[7]),
            "tail_interface_reaction_z_n": float(sample_values[8]),
            "volume_flux_m3s": float(sample_values[9]),
            "nozzle_velocity_z_mps": float(sample_values[10]),
            "lip_flow_z_m3s": float(sample_values[11]),
            "outlet_flow_z_m3s": float(sample_values[12]),
            "downstream_flow_z_m3s": float(sample_values[13]),
            "lip_flow_negative_z_m3s": -float(sample_values[11]),
            "outlet_flow_negative_z_m3s": -float(sample_values[12]),
            "downstream_flow_negative_z_m3s": -float(sample_values[13]),
            "lip_sample_count": int(sample_counts[0]),
            "outlet_sample_count": int(sample_counts[1]),
            "downstream_sample_count": int(sample_counts[2]),
            "max_fluid_speed_mps": max_speed,
            "cfl": max_speed * cfl_dt_s / max(h, 1.0e-12),
            **divergence_sample_report_fields(divergence),
        }

    def sample_cfl_report(
        self,
        *,
        dt_s: float | None = None,
    ) -> dict[str, float]:
        spec = self.spec
        self.sample_sections_kernel(
            self.fluid.velocity,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.cell_width_x_m,
            self.fluid.cell_width_y_m,
            self.fluid.cell_width_z_m,
            float(spec.monitor_center_x_m),
            float(spec.monitor_center_y_m),
            float(spec.monitor_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.outlet_plume_radius_m),
            float(spec.lip_z_m),
            float(spec.outlet_z_m),
            float(spec.downstream_z_m),
        )
        h = min(cartesian_grid_axis_min_spacing_m(self.fluid.grid))
        sample_values = self.sample_report_float_snapshot[None]
        self.last_sample_report_host_reads = 1
        max_speed = float(sample_values[14])
        cfl_dt_s = float(self.spec.dt_s) if dt_s is None else float(dt_s)
        return {
            "max_fluid_speed_mps": max_speed,
            "cfl": max_speed * cfl_dt_s / max(h, 1.0e-12),
        }
