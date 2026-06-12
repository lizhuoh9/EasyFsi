import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import taichi as ti

from .runtime import TaichiRuntimeConfig, init_taichi


FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED = "legacy_projected_reduced"
FSI_COUPLING_MODE_HIBM_MPM_SHARP = "hibm_mpm_sharp"
FSI_COUPLING_MODE_CHOICES = (
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
)


@dataclass(frozen=True)
class HibmMpmPaperRequirement:
    requirement: str
    paper_section: str
    paper_mechanism: str
    current_status: str
    required_solver_work: str


@dataclass(frozen=True)
class HibmMpmSurfaceMarkerForceReport:
    primary_marker_force_n: tuple[float, float, float]
    secondary_marker_force_n: tuple[float, float, float]
    total_marker_force_n: tuple[float, float, float]
    primary_marker_count: int
    secondary_marker_count: int
    total_marker_count: int
    fluid_reaction_force_n: tuple[float, float, float]
    action_reaction_residual_n: float


@dataclass(frozen=True)
class HibmMpmFluidStressSampleReport:
    valid_marker_count: int
    invalid_marker_count: int
    max_abs_traction_pa: float
    two_sided_pressure_marker_count: int = 0
    viscous_gradient_invalid_marker_count: int = 0
    far_pressure_closed_marker_count: int = 0
    far_pressure_closed_extended_marker_count: int = 0
    far_pressure_anchor_closed_marker_count: int = 0
    closure_gradient_missing_marker_count: int = 0


@dataclass(frozen=True)
class HibmMpmNoSlipResidualReport:
    valid_marker_count: int
    invalid_marker_count: int
    max_no_slip_residual_mps: float
    l2_no_slip_residual_mps: float


@dataclass(frozen=True)
class HibmMpmMpmForceScatterReport:
    active_marker_count: int
    invalid_marker_count: int
    active_particle_count: int
    total_marker_force_n: tuple[float, float, float]
    total_mpm_external_force_n: tuple[float, float, float]
    action_reaction_residual_n: float


@dataclass(frozen=True)
class HibmMpmExternalForceClearReport:
    cleared_particle_count: int
    max_abs_external_force_before_n: float


@dataclass(frozen=True)
class HibmMpmSurfaceUpdateReport:
    updated_marker_count: int
    invalid_marker_count: int
    max_marker_displacement_m: float
    max_marker_speed_mps: float
    geometry_updated_marker_count: int = 0
    geometry_invalid_marker_count: int = 0
    max_marker_normal_change: float = 0.0
    max_marker_area_change_m2: float = 0.0


@dataclass(frozen=True)
class HibmMpmIbNodeSearchReport:
    near_boundary_node_count: int
    external_ib_node_count: int
    internal_node_count: int
    invalid_projection_count: int


@dataclass(frozen=True)
class HibmMpmIbBoundaryConditionReport:
    no_slip_dirichlet_count: int
    pressure_neumann_count: int
    inactive_internal_node_count: int


@dataclass(frozen=True)
class HibmMpmVelocityDirichletBoundaryReport:
    active_velocity_dirichlet_rows: int
    inactive_obstacle_rows: int
    max_abs_velocity_mps: float
    invalid_reconstruction_row_count: int = 0
    invalid_no_fluid_sample_row_count: int = 0
    invalid_nonpositive_gap_row_count: int = 0
    invalid_node_behind_boundary_row_count: int = 0
    invalid_node_beyond_interior_row_count: int = 0
    narrow_gap_boundary_velocity_row_count: int = 0
    relocated_row_count: int = 0
    relocation_merged_row_count: int = 0
    relocation_blocked_row_count: int = 0
    min_projection_weight: float = 0.0
    max_projection_weight: float = 0.0


@dataclass(frozen=True)
class HibmMpmPressureNeumannMatrixReport:
    active_pressure_neumann_rows: int
    rhs_integral: float
    max_abs_rhs: float
    skipped_velocity_dirichlet_row_count: int = 0
    skipped_obstacle_owner_row_count: int = 0
    active_pressure_neumann_marker_count: int = 0
    max_pressure_neumann_rows_per_marker: int = 0
    invalid_reconstruction_row_count: int = 0
    min_reconstruction_gap_m: float = 0.0
    max_reconstruction_gap_m: float = 0.0
    max_transmissibility_m: float = 0.0
    max_raw_transmissibility_m: float = 0.0
    max_transmissibility_limit_m: float = 0.0
    transmissibility_capped_row_count: int = 0
    max_diagonal_per_m2: float = 0.0


@dataclass(frozen=True)
class HibmMpmPressureNeumannGradientReport:
    active_marker_count: int
    max_abs_gradient_pa_per_m: float


@dataclass(frozen=True)
class HibmMpmSharpFluidToMpmLoadReport:
    ib_node_search: HibmMpmIbNodeSearchReport
    internal_obstacle_cell_count: int
    solid_band_nonprojectable_cell_count: int
    pressure_disconnected_nonprojectable_cell_count: int
    boundary_conditions: HibmMpmIbBoundaryConditionReport
    pressure_neumann_gradient: HibmMpmPressureNeumannGradientReport | None
    velocity_dirichlet: HibmMpmVelocityDirichletBoundaryReport
    pressure_neumann: HibmMpmPressureNeumannMatrixReport
    fluid_predictor_applied: bool
    fluid_projection: dict[str, Any]
    no_slip_residual: HibmMpmNoSlipResidualReport
    fluid_stress: HibmMpmFluidStressSampleReport
    marker_forces: HibmMpmSurfaceMarkerForceReport
    mpm_external_force_clear: HibmMpmExternalForceClearReport
    mpm_force_scatter: HibmMpmMpmForceScatterReport


@dataclass(frozen=True)
class HibmMpmSharpMpmStepReport:
    fluid_to_mpm_loads: HibmMpmSharpFluidToMpmLoadReport
    mpm: Any
    surface_feedback: HibmMpmSurfaceUpdateReport
    next_ib_node_search: HibmMpmIbNodeSearchReport
    next_internal_obstacle_cell_count: int
    next_solid_band_nonprojectable_cell_count: int
    next_pressure_disconnected_nonprojectable_cell_count: int
    next_boundary_conditions: HibmMpmIbBoundaryConditionReport
    next_velocity_dirichlet: HibmMpmVelocityDirichletBoundaryReport
    next_pressure_neumann: HibmMpmPressureNeumannMatrixReport
    next_pressure_neumann_gradient: HibmMpmPressureNeumannGradientReport | None = None


@dataclass(frozen=True)
class HibmMpmSharpNeoHookeanStepReport(HibmMpmSharpMpmStepReport):
    pass


def _vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    try:
        vector = tuple(float(component) for component in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 3 components") from exc
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if any(not math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain only finite values")
    return (vector[0], vector[1], vector[2])


def _normalize_vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = _vector3(value, name=name)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must contain non-zero vectors")
    return tuple(component / norm for component in vector)


@ti.data_oriented
class HibmMpmSurfaceMarkers:
    def __init__(
        self,
        marker_capacity: int,
        projection_triangle_capacity: int | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.marker_capacity = int(marker_capacity)
        if projection_triangle_capacity is None:
            projection_triangle_capacity = self.marker_capacity
        if int(projection_triangle_capacity) <= 0:
            raise ValueError("projection_triangle_capacity must be positive")
        self.projection_triangle_capacity = int(projection_triangle_capacity)
        self.marker_count = 0
        self.projection_triangle_count = 0

        self.x_gamma_m = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.v_gamma_mps = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.n_gamma = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.A_gamma_m2 = ti.field(dtype=ti.f32, shape=self.marker_capacity)
        self.region_id = ti.field(dtype=ti.i32, shape=self.marker_capacity)
        self.t_gamma_pa = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.F_gamma_n = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_capacity)
        self.projection_triangle_indices = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.projection_triangle_capacity,
        )
        # S2-A6: per-marker pressure-Neumann anchor cell. The pressure
        # matrix row assembly records, for every marker that received at
        # least one row, the (i, j, k) of the row-owning fluid cell (a
        # solve-participating, non-obstacle cell by construction);
        # (-1, -1, -1) means "no row / unset". The stress sampler may use
        # it as a last-resort water-side pressure probe for closure-region
        # markers whose normal walks are fully sealed by band obstacles.
        self.marker_pressure_anchor_cell = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.marker_capacity,
        )

        self.report_primary_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_secondary_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_total_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_primary_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_secondary_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_total_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_stress_valid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_stress_invalid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_stress_max_abs_traction_pa = ti.field(dtype=ti.f32, shape=())
        self.report_stress_two_sided_pressure_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_viscous_gradient_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_closed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_closed_extended_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_far_pressure_anchor_closed_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_stress_closure_gradient_missing_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_no_slip_valid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_no_slip_invalid_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_no_slip_max_residual_mps = ti.field(dtype=ti.f32, shape=())
        self.report_no_slip_sum_residual2_mps2 = ti.field(dtype=ti.f64, shape=())
        self.report_mpm_scatter_marker_force_n = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(),
        )
        self.report_mpm_scatter_external_force_n = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(),
        )
        self.report_mpm_scatter_active_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_scatter_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_scatter_active_particle_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_external_force_clear_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_mpm_external_force_clear_max_abs_n = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_updated_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_max_displacement_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_max_speed_mps = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_geometry_updated_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_geometry_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_feedback_max_normal_change = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_feedback_max_area_change_m2 = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_surface_field_load_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_surface_field_load_invalid_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_max_abs = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_projection_triangle_invalid_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # Taichi zero-initializes fields, and (0, 0, 0) is a real cell
        # index: establish the unset sentinel before anyone can sample.
        self.reset_pressure_anchor_cells()

    @ti.kernel
    def _reset_pressure_anchor_cells_kernel(self):
        for marker in self.marker_pressure_anchor_cell:
            self.marker_pressure_anchor_cell[marker] = ti.Vector([-1, -1, -1])

    def reset_pressure_anchor_cells(self) -> None:
        """Reset every marker's pressure-Neumann anchor cell to unset.

        Runs over the full capacity so markers that never receive a
        pressure matrix row (including slots beyond marker_count) keep the
        (-1, -1, -1) sentinel and stay on the invalid path in the sampler.
        """
        self._reset_pressure_anchor_cells_kernel()

    def load_markers(
        self,
        *,
        positions_m: Sequence[Sequence[float]],
        velocities_mps: Sequence[Sequence[float]],
        normals: Sequence[Sequence[float]],
        areas_m2: Sequence[float],
        region_ids: Sequence[int],
    ) -> None:
        count = len(positions_m)
        if count > self.marker_capacity:
            raise ValueError("marker count exceeds marker_capacity")
        if not (
            len(velocities_mps)
            == len(normals)
            == len(areas_m2)
            == len(region_ids)
            == count
        ):
            raise ValueError("marker inputs must have matching lengths")
        self.marker_count = int(count)
        self.projection_triangle_count = 0
        for marker in range(count):
            position = _vector3(positions_m[marker], name="positions_m")
            velocity = _vector3(velocities_mps[marker], name="velocities_mps")
            normal = _normalize_vector3(normals[marker], name="normals")
            area = float(areas_m2[marker])
            if not math.isfinite(area) or area < 0.0:
                raise ValueError("areas_m2 must contain finite non-negative values")
            self.x_gamma_m[marker] = position
            self.v_gamma_mps[marker] = velocity
            self.n_gamma[marker] = normal
            self.A_gamma_m2[marker] = area
            self.region_id[marker] = int(region_ids[marker])
            self.t_gamma_pa[marker] = (0.0, 0.0, 0.0)
            self.F_gamma_n[marker] = (0.0, 0.0, 0.0)

    def set_projection_triangles(
        self,
        triangle_indices: Sequence[Sequence[int]],
    ) -> int:
        count = len(triangle_indices)
        if count > self.projection_triangle_capacity:
            raise ValueError(
                "projection triangle count exceeds projection_triangle_capacity"
            )
        marker_count = int(self.marker_count)
        for triangle_index, triangle in enumerate(triangle_indices):
            if len(triangle) != 3:
                raise ValueError("projection triangles must contain three indices")
            ia, ib, ic = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
            if (
                ia < 0
                or ib < 0
                or ic < 0
                or ia >= marker_count
                or ib >= marker_count
                or ic >= marker_count
            ):
                raise ValueError("projection triangle index out of marker range")
            self.projection_triangle_indices[triangle_index] = (ia, ib, ic)
        self.projection_triangle_count = int(count)
        return self.projection_triangle_count

    @ti.kernel
    def _load_projection_triangles_from_field_kernel(
        self,
        triangle_indices: ti.template(),
        triangle_count: ti.i32,
        marker_count: ti.i32,
    ):
        self.report_projection_triangle_invalid_count[None] = 0
        for triangle_index in range(triangle_count):
            triangle = triangle_indices[triangle_index]
            invalid = (
                triangle.x < 0
                or triangle.y < 0
                or triangle.z < 0
                or triangle.x >= marker_count
                or triangle.y >= marker_count
                or triangle.z >= marker_count
            )
            if invalid:
                self.report_projection_triangle_invalid_count[None] += 1
            self.projection_triangle_indices[triangle_index] = triangle

    def load_projection_triangles_from_field(
        self,
        triangle_indices,
        *,
        triangle_count: int,
    ) -> int:
        count = int(triangle_count)
        if count < 0:
            raise ValueError("triangle_count must be non-negative")
        if count > self.projection_triangle_capacity:
            raise ValueError(
                "projection triangle count exceeds projection_triangle_capacity"
            )
        self._load_projection_triangles_from_field_kernel(
            triangle_indices,
            count,
            int(self.marker_count),
        )
        invalid_count = int(self.report_projection_triangle_invalid_count[None])
        if invalid_count > 0:
            raise ValueError("projection triangle index out of marker range")
        self.projection_triangle_count = count
        return self.projection_triangle_count

    @ti.kernel
    def _load_markers_from_surface_fields_kernel(
        self,
        surface_position_m: ti.template(),
        surface_normal: ti.template(),
        surface_area_m2: ti.template(),
        surface_region_id: ti.template(),
        marker_count: ti.i32,
        initial_velocity_x_mps: ti.f32,
        initial_velocity_y_mps: ti.f32,
        initial_velocity_z_mps: ti.f32,
    ):
        self.report_surface_field_load_marker_count[None] = marker_count
        self.report_surface_field_load_invalid_marker_count[None] = 0
        initial_velocity = ti.Vector(
            [
                initial_velocity_x_mps,
                initial_velocity_y_mps,
                initial_velocity_z_mps,
            ]
        )
        for marker in range(marker_count):
            normal = surface_normal[marker]
            normal_norm = normal.norm()
            area = surface_area_m2[marker]
            if normal_norm <= 1.0e-12 or area < 0.0:
                self.report_surface_field_load_invalid_marker_count[None] += 1
            self.x_gamma_m[marker] = surface_position_m[marker]
            self.v_gamma_mps[marker] = initial_velocity
            self.n_gamma[marker] = normal / ti.max(normal_norm, 1.0e-12)
            self.A_gamma_m2[marker] = ti.max(area, 0.0)
            self.region_id[marker] = surface_region_id[marker]
            self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self.F_gamma_n[marker] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _load_markers_from_surface_velocity_fields_kernel(
        self,
        surface_position_m: ti.template(),
        surface_velocity_mps: ti.template(),
        surface_normal: ti.template(),
        surface_area_m2: ti.template(),
        surface_region_id: ti.template(),
        marker_count: ti.i32,
    ):
        self.report_surface_field_load_marker_count[None] = marker_count
        self.report_surface_field_load_invalid_marker_count[None] = 0
        for marker in range(marker_count):
            normal = surface_normal[marker]
            normal_norm = normal.norm()
            area = surface_area_m2[marker]
            if normal_norm <= 1.0e-12 or area < 0.0:
                self.report_surface_field_load_invalid_marker_count[None] += 1
            velocity = surface_velocity_mps[marker]
            self.x_gamma_m[marker] = surface_position_m[marker]
            self.v_gamma_mps[marker] = velocity
            self.n_gamma[marker] = normal / ti.max(normal_norm, 1.0e-12)
            self.A_gamma_m2[marker] = ti.max(area, 0.0)
            self.region_id[marker] = surface_region_id[marker]
            self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
            self.F_gamma_n[marker] = ti.Vector([0.0, 0.0, 0.0])

    def load_markers_from_surface_fields(
        self,
        surface_position_m,
        surface_normal,
        surface_area_m2,
        surface_region_id,
        *,
        marker_count: int,
        initial_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
        surface_velocity_mps=None,
    ) -> int:
        count = int(marker_count)
        if count < 0:
            raise ValueError("marker_count must be non-negative")
        if count > self.marker_capacity:
            raise ValueError("marker_count exceeds marker_capacity")
        self.marker_count = count
        self.projection_triangle_count = 0
        if surface_velocity_mps is None:
            initial_velocity = _vector3(
                initial_velocity_mps,
                name="initial_velocity_mps",
            )
            self._load_markers_from_surface_fields_kernel(
                surface_position_m,
                surface_normal,
                surface_area_m2,
                surface_region_id,
                count,
                float(initial_velocity[0]),
                float(initial_velocity[1]),
                float(initial_velocity[2]),
            )
        else:
            self._load_markers_from_surface_velocity_fields_kernel(
                surface_position_m,
                surface_velocity_mps,
                surface_normal,
                surface_area_m2,
                surface_region_id,
                count,
            )
        invalid_count = int(self.report_surface_field_load_invalid_marker_count[None])
        if invalid_count > 0:
            raise ValueError(
                "surface fields must contain non-negative areas and non-zero normals"
            )
        return count

    @ti.kernel
    def _update_pressure_neumann_gradient_from_fluid_predictor_kernel(
        self,
        marker_pressure_neumann_gradient_pa_per_m_field: ti.template(),
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        marker_count: ti.i32,
        density_kgm3: ti.f32,
        dt_s: ti.f32,
        probe_distance_m: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_gradient_marker_count[None] = 0
        self.report_pressure_neumann_gradient_max_abs[None] = 0.0
        for marker in range(marker_count):
            sample_position = (
                self.x_gamma_m[marker] + self.n_gamma[marker] * probe_distance_m
            )
            grid_coordinate = self._grid_coordinate_from_fields(
                sample_position,
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
            predictor_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                velocity_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            normal_gradient = 0.0
            if fluid_weight > 1.0e-12:
                normal_gradient = (
                    density_kgm3
                    * (predictor_velocity - self.v_gamma_mps[marker]).dot(
                        self.n_gamma[marker]
                    )
                    / dt_s
                )
                self.report_pressure_neumann_gradient_marker_count[None] += 1
                ti.atomic_max(
                    self.report_pressure_neumann_gradient_max_abs[None],
                    ti.abs(normal_gradient),
                )
            marker_pressure_neumann_gradient_pa_per_m_field[marker] = normal_gradient

    def update_pressure_neumann_gradient_from_fluid_predictor(
        self,
        marker_pressure_neumann_gradient_pa_per_m_field,
        *,
        velocity_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        density_kgm3: float,
        dt_s: float,
        probe_distance_m: float,
    ) -> HibmMpmPressureNeumannGradientReport:
        density = float(density_kgm3)
        dt = float(dt_s)
        probe_distance = float(probe_distance_m)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("density_kgm3 must be finite and positive")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not math.isfinite(probe_distance) or probe_distance <= 0.0:
            raise ValueError("probe_distance_m must be finite and positive")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._update_pressure_neumann_gradient_from_fluid_predictor_kernel(
            marker_pressure_neumann_gradient_pa_per_m_field,
            velocity_field,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(self.marker_count),
            density,
            dt,
            probe_distance,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        return HibmMpmPressureNeumannGradientReport(
            active_marker_count=int(
                self.report_pressure_neumann_gradient_marker_count[None]
            ),
            max_abs_gradient_pa_per_m=float(
                self.report_pressure_neumann_gradient_max_abs[None]
            ),
        )

    def set_marker_tractions_pa(
        self,
        tractions_pa: Sequence[Sequence[float]],
    ) -> None:
        if len(tractions_pa) != self.marker_count:
            raise ValueError("tractions_pa must match marker_count")
        for marker in range(self.marker_count):
            self.t_gamma_pa[marker] = _vector3(tractions_pa[marker], name="tractions_pa")

    @ti.kernel
    def _compute_marker_forces_kernel(self, marker_count: ti.i32):
        for marker in range(marker_count):
            self.F_gamma_n[marker] = self.t_gamma_pa[marker] * self.A_gamma_m2[marker]

    def compute_marker_forces(self) -> None:
        self._compute_marker_forces_kernel(int(self.marker_count))

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value: ti.f32,
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
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (
                value - centers[count - 1]
            ) / half_width
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
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        return ti.Vector(
            [
                self._axis_grid_coordinate_device(
                    position.x,
                    cell_face_x_m,
                    cell_center_x_m,
                    nx,
                ),
                self._axis_grid_coordinate_device(
                    position.y,
                    cell_face_y_m,
                    cell_center_y_m,
                    ny,
                ),
                self._axis_grid_coordinate_device(
                    position.z,
                    cell_face_z_m,
                    cell_center_z_m,
                    nz,
                ),
            ]
        )

    @ti.func
    def _sample_pressure_trilinear(
        self,
        pressure_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
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
    def _sample_fluid_velocity_trilinear(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
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
    def _sample_velocity_gradient(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
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
            velocity_field,
            obstacle_field,
            ti.cast(ix0, ti.f32),
            gy,
            gz,
            nx,
            ny,
            nz,
        )
        vx1, wx1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            ti.cast(ix1, ti.f32),
            gy,
            gz,
            nx,
            ny,
            nz,
        )
        vy0, wy0 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            ti.cast(iy0, ti.f32),
            gz,
            nx,
            ny,
            nz,
        )
        vy1, wy1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            ti.cast(iy1, ti.f32),
            gz,
            nx,
            ny,
            nz,
        )
        vz0, wz0 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            gy,
            ti.cast(iz0, ti.f32),
            nx,
            ny,
            nz,
        )
        vz1, wz1 = self._sample_fluid_velocity_trilinear(
            velocity_field,
            obstacle_field,
            gx,
            gy,
            ti.cast(iz1, ti.f32),
            nx,
            ny,
            nz,
        )
        dvdx = ti.Vector([0.0, 0.0, 0.0])
        dvdy = ti.Vector([0.0, 0.0, 0.0])
        dvdz = ti.Vector([0.0, 0.0, 0.0])
        gradient_valid = 1
        if wx0 > 1.0e-12 and wx1 > 1.0e-12:
            dvdx = (vx1 - vx0) / dx
        else:
            gradient_valid = 0
        if wy0 > 1.0e-12 and wy1 > 1.0e-12:
            dvdy = (vy1 - vy0) / dy
        else:
            gradient_valid = 0
        if wz0 > 1.0e-12 and wz1 > 1.0e-12:
            dvdz = (vz1 - vz0) / dz
        else:
            gradient_valid = 0
        return (
            ti.Matrix(
                [
                    [dvdx.x, dvdy.x, dvdz.x],
                    [dvdx.y, dvdy.y, dvdz.y],
                    [dvdx.z, dvdy.z, dvdz.z],
                ]
            ),
            gradient_valid,
        )

    @ti.kernel
    def _sample_fluid_stress_to_marker_tractions_kernel(
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
        marker_pressure_anchor_cell: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        viscosity_pa_s: ti.f32,
        two_sided_pressure: ti.i32,
        far_pressure_region_id: ti.i32,
        far_pressure_pa: ti.f32,
        far_pressure_inside_probe_max_multiplier: ti.f32,
        use_pressure_anchor_fallback: ti.i32,
    ):
        self.report_stress_valid_marker_count[None] = 0
        self.report_stress_invalid_marker_count[None] = 0
        self.report_stress_max_abs_traction_pa[None] = 0.0
        self.report_stress_two_sided_pressure_marker_count[None] = 0
        self.report_stress_viscous_gradient_invalid_marker_count[None] = 0
        self.report_stress_far_pressure_closed_marker_count[None] = 0
        self.report_stress_far_pressure_closed_extended_marker_count[None] = 0
        self.report_stress_far_pressure_anchor_closed_marker_count[None] = 0
        self.report_stress_closure_gradient_missing_marker_count[None] = 0
        for marker in range(marker_count):
            position = self.x_gamma_m[marker]
            normal = self.n_gamma[marker]
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
            i_near = ti.min(ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0), nx - 1)
            j_near = ti.min(ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0), ny - 1)
            k_near = ti.min(ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0), nz - 1)
            pressure, pressure_weight = self._sample_pressure_trilinear(
                pressure_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            pressure_traction = -pressure * normal
            pressure_sample_valid = pressure_weight > 1.0e-12
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            gradient_valid = 1
            if two_sided_pressure != 0:
                normal_spacing_inv = (
                    ti.abs(normal.x) / cell_width_x_m[i_near]
                    + ti.abs(normal.y) / cell_width_y_m[j_near]
                    + ti.abs(normal.z) / cell_width_z_m[k_near]
                )
                probe_distance_m = 1.0 / ti.max(normal_spacing_inv, 1.0e-12)
                outside_pressure = 0.0
                inside_pressure = 0.0
                outside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                inside_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                # S2-A4: per-side "found" is split into pressure-found and
                # gradient-found. In far-pressure closure regions the squid
                # band obstacle slabs leave only 1-3 cell wide water gaps
                # behind the membranes at production grids: trilinear
                # pressure is samplable inside such a gap, but the one-sided
                # viscous gradient stencil (complete fluid neighbor pairs on
                # all three axes) almost never is, and the merged acceptance
                # gate rejected the whole candidate - silently dropping the
                # O(1e3 Pa) pressure drive over an O(0.1 Pa) viscous term.
                # Branch decisions (two-sided / closure / mirrored closure)
                # key on the pressure flags; an unfound gradient side simply
                # keeps its zero matrix. Outside closure regions both flags
                # are only ever set together, preserving the original merged
                # gate bit for bit.
                outside_pressure_found = 0
                outside_gradient_found = 0
                inside_pressure_found = 0
                inside_gradient_found = 0
                inside_found_extended = 0
                closure_region_marker = 0
                if (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                ):
                    closure_region_marker = 1
                # S2-A5: runtime range(5) instead of ti.static. The body only
                # consumes probe_index through ti.cast(..., ti.f32) (exact for
                # 0..4), the per-thread serial loop runs the iterations in the
                # same order with the same carried flag state, so every float
                # is produced by the identical expression tree in the
                # identical order - bitwise-preserving - while the JIT stops
                # cloning the inlined trilinear/gradient sampling bodies five
                # times per side.
                for probe_index in range(5):
                    probe_distance = probe_distance_m * (
                        1.0 + 0.5 * ti.cast(probe_index, ti.f32)
                    )
                    # Walk on while the side pressure is missing; in closure
                    # regions additionally walk on to fill a still-missing
                    # gradient from a farther candidate (the gradient update
                    # below never overwrites an already-found pressure).
                    if outside_pressure_found == 0 or (
                        closure_region_marker == 1 and outside_gradient_found == 0
                    ):
                        outside_position = position + normal * probe_distance
                        outside_coordinate = self._grid_coordinate_from_fields(
                            outside_position,
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
                        sample_pressure, sample_weight = self._sample_pressure_trilinear(
                            pressure_field,
                            obstacle_field,
                            outside_coordinate.x,
                            outside_coordinate.y,
                            outside_coordinate.z,
                            nx,
                            ny,
                            nz,
                        )
                        sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        sample_gradient_valid = 1
                        if viscosity_pa_s > 0.0:
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient(
                                    velocity_field,
                                    obstacle_field,
                                    outside_coordinate.x,
                                    outside_coordinate.y,
                                    outside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                        # S2-A5 single-copy acceptance: the accept decision is
                        # split from the stores so each store exists once in
                        # the IR. Closure regions keep the S2-A4 decoupled
                        # guards (pressure needs trilinear fluid weight; the
                        # gradient additionally needs its complete one-sided
                        # stencil, so a farther candidate can still supply
                        # it). Outside closure regions the original merged
                        # gate applies atomically: pressure and gradient are
                        # accepted together from the same candidate or not at
                        # all, so a nearer pressure can never pair with a
                        # farther gradient - bit for bit the original gate.
                        candidate_merged_ok = 0
                        if sample_weight > 1.0e-12 and sample_gradient_valid == 1:
                            candidate_merged_ok = 1
                        accept_pressure = 0
                        accept_gradient = 0
                        if closure_region_marker == 1:
                            if sample_weight > 1.0e-12 and outside_pressure_found == 0:
                                accept_pressure = 1
                            if candidate_merged_ok == 1 and outside_gradient_found == 0:
                                accept_gradient = 1
                        else:
                            if candidate_merged_ok == 1:
                                accept_pressure = 1
                                accept_gradient = 1
                        if accept_pressure == 1:
                            outside_pressure = sample_pressure
                            outside_pressure_found = 1
                        if accept_gradient == 1:
                            outside_gradient = sample_gradient
                            outside_gradient_found = 1
                    if inside_pressure_found == 0 or (
                        closure_region_marker == 1 and inside_gradient_found == 0
                    ):
                        inside_position = position - normal * probe_distance
                        inside_coordinate = self._grid_coordinate_from_fields(
                            inside_position,
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
                        sample_pressure, sample_weight = self._sample_pressure_trilinear(
                            pressure_field,
                            obstacle_field,
                            inside_coordinate.x,
                            inside_coordinate.y,
                            inside_coordinate.z,
                            nx,
                            ny,
                            nz,
                        )
                        sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        sample_gradient_valid = 1
                        if viscosity_pa_s > 0.0:
                            sample_gradient, sample_gradient_valid = (
                                self._sample_velocity_gradient(
                                    velocity_field,
                                    obstacle_field,
                                    inside_coordinate.x,
                                    inside_coordinate.y,
                                    inside_coordinate.z,
                                    nx,
                                    ny,
                                    nz,
                                    cell_center_x_m,
                                    cell_center_y_m,
                                    cell_center_z_m,
                                )
                            )
                        # S2-A5 single-copy acceptance, symmetric to the
                        # outside (+n) walk above.
                        candidate_merged_ok = 0
                        if sample_weight > 1.0e-12 and sample_gradient_valid == 1:
                            candidate_merged_ok = 1
                        accept_pressure = 0
                        accept_gradient = 0
                        if closure_region_marker == 1:
                            if sample_weight > 1.0e-12 and inside_pressure_found == 0:
                                accept_pressure = 1
                            if candidate_merged_ok == 1 and inside_gradient_found == 0:
                                accept_gradient = 1
                        else:
                            if candidate_merged_ok == 1:
                                accept_pressure = 1
                                accept_gradient = 1
                        if accept_pressure == 1:
                            inside_pressure = sample_pressure
                            inside_pressure_found = 1
                        if accept_gradient == 1:
                            inside_gradient = sample_gradient
                            inside_gradient_found = 1
                # S2-A3 extended inside (-n) walk: opt-in, far-pressure
                # closure regions only. It runs strictly after the standard
                # ladder above failed to find inside water within 3x, and
                # extends the reach uniformly to the requested multiplier.
                # The outside (+n) walk is intentionally never extended.
                # The outside_pressure_found == 0 gate keeps
                # mirrored-orientation markers (water on +n, structurally dry
                # -n) on the mirrored closure branch: without it the extended
                # walk could tunnel through a thin dry band to unrelated deep
                # water and silently replace the known far pressure with a
                # spurious two-sided sample, dropping the drive on exactly
                # the markers the closure exists for. S2-A4: the gate keys on
                # the pressure flag (not the gradient flag) so a mirrored
                # marker whose outside water has a broken gradient stencil
                # stays mirror-protected as well.
                if (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                    and inside_pressure_found == 0
                    and outside_pressure_found == 0
                    and far_pressure_inside_probe_max_multiplier > 3.0
                ):
                    # S2-A5: runtime loop, same bitwise-preservation argument
                    # as the standard walk above.
                    for probe_index in range(5):
                        probe_distance = probe_distance_m * (
                            3.0
                            + (far_pressure_inside_probe_max_multiplier - 3.0)
                            * (ti.cast(probe_index, ti.f32) + 1.0)
                            / 5.0
                        )
                        # The extension only runs for closure-region markers
                        # (see the entry gate above), so the S2-A4 decoupled
                        # acceptance applies unconditionally here.
                        if inside_pressure_found == 0 or inside_gradient_found == 0:
                            inside_position = position - normal * probe_distance
                            inside_coordinate = self._grid_coordinate_from_fields(
                                inside_position,
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
                            sample_pressure, sample_weight = self._sample_pressure_trilinear(
                                pressure_field,
                                obstacle_field,
                                inside_coordinate.x,
                                inside_coordinate.y,
                                inside_coordinate.z,
                                nx,
                                ny,
                                nz,
                            )
                            sample_gradient = ti.Matrix.zero(ti.f32, 3, 3)
                            sample_gradient_valid = 1
                            if viscosity_pa_s > 0.0:
                                sample_gradient, sample_gradient_valid = (
                                    self._sample_velocity_gradient(
                                        velocity_field,
                                        obstacle_field,
                                        inside_coordinate.x,
                                        inside_coordinate.y,
                                        inside_coordinate.z,
                                        nx,
                                        ny,
                                        nz,
                                        cell_center_x_m,
                                        cell_center_y_m,
                                        cell_center_z_m,
                                    )
                                )
                            if sample_weight > 1.0e-12 and inside_pressure_found == 0:
                                inside_pressure = sample_pressure
                                inside_pressure_found = 1
                                inside_found_extended = 1
                            if (
                                sample_weight > 1.0e-12
                                and sample_gradient_valid == 1
                                and inside_gradient_found == 0
                            ):
                                inside_gradient = sample_gradient
                                inside_gradient_found = 1
                if outside_pressure_found == 1 and inside_pressure_found == 1:
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    ti.atomic_add(
                        self.report_stress_two_sided_pressure_marker_count[None],
                        1,
                    )
                elif (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                    and inside_pressure_found == 1
                    and outside_pressure_found == 0
                ):
                    outside_pressure = far_pressure_pa
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    ti.atomic_add(
                        self.report_stress_far_pressure_closed_marker_count[None],
                        1,
                    )
                    if inside_found_extended == 1:
                        ti.atomic_add(
                            self.report_stress_far_pressure_closed_extended_marker_count[None],
                            1,
                        )
                elif (
                    far_pressure_region_id != -1
                    and self.region_id[marker] == far_pressure_region_id
                    and outside_pressure_found == 1
                    and inside_pressure_found == 0
                ):
                    # Mirrored closure: the structurally dry side sits on the
                    # inside (-n) walk because of the CAD winding. The same
                    # covariant formula applies with the known far pressure
                    # substituted on the dry side; inside_gradient is provably
                    # zero here (it is only written when inside_gradient_found
                    # is set, and a gradient can only be found at a candidate
                    # whose fluid weight also sets the pressure flag, which is
                    # 0 on this branch).
                    inside_pressure = far_pressure_pa
                    pressure_traction = (inside_pressure - outside_pressure) * normal
                    pressure_sample_valid = True
                    gradient = outside_gradient - inside_gradient
                    ti.atomic_add(
                        self.report_stress_far_pressure_closed_marker_count[None],
                        1,
                    )
                else:
                    pressure_sample_valid = False
                    # S2-A6 anchor fallback: both closure branches missed
                    # because the normal walk never sampled even a pressure
                    # weight on either side (the band obstacle slabs seal
                    # the whole 12x reach in some cavity columns). If the
                    # pressure-Neumann row assembly anchored this marker to
                    # a row-owning fluid cell, read that cell-center
                    # pressure directly (no interpolation - the anchor
                    # participates in the pressure solve by construction,
                    # never a band/stale cell) as the water side and the
                    # known far pressure as the dry side. The water side
                    # follows from the anchor center's normal projection:
                    # anchor on -n means water inside, so the covariant
                    # two-sided formula gives (p_anchor - p_far) * n;
                    # otherwise the mirrored orientation gives
                    # (p_far - p_anchor) * n. The viscous gradient stays
                    # the zero matrix: no walk candidate had fluid weight,
                    # so no one-sided gradient was ever found either.
                    if (
                        use_pressure_anchor_fallback != 0
                        and closure_region_marker == 1
                        and marker_pressure_anchor_cell[marker].x >= 0
                    ):
                        anchor_cell = marker_pressure_anchor_cell[marker]
                        anchor_pressure = pressure_field[
                            anchor_cell.x,
                            anchor_cell.y,
                            anchor_cell.z,
                        ]
                        anchor_center = ti.Vector(
                            [
                                cell_center_x_m[anchor_cell.x],
                                cell_center_y_m[anchor_cell.y],
                                cell_center_z_m[anchor_cell.z],
                            ]
                        )
                        inside_pressure = far_pressure_pa
                        outside_pressure = anchor_pressure
                        if (anchor_center - position).dot(normal) < 0.0:
                            inside_pressure = anchor_pressure
                            outside_pressure = far_pressure_pa
                        pressure_traction = (
                            inside_pressure - outside_pressure
                        ) * normal
                        pressure_sample_valid = True
                        gradient = ti.Matrix.zero(ti.f32, 3, 3)
                        ti.atomic_add(
                            self.report_stress_far_pressure_closed_marker_count[
                                None
                            ],
                            1,
                        )
                        ti.atomic_add(
                            self.report_stress_far_pressure_anchor_closed_marker_count[
                                None
                            ],
                            1,
                        )
                # S2-A4 diagnostic: the marker closes on pressure while at
                # least one pressure-found side never completed a viscous
                # gradient stencil (that side's gradient contribution is
                # exactly zero). In two-sided mode the outer gradient_valid
                # stays 1, so final validity equals pressure_sample_valid and
                # counting here counts exactly the finally-valid markers.
                if closure_region_marker == 1 and pressure_sample_valid:
                    if (
                        outside_pressure_found == 1
                        and outside_gradient_found == 0
                    ) or (
                        inside_pressure_found == 1
                        and inside_gradient_found == 0
                    ):
                        ti.atomic_add(
                            self.report_stress_closure_gradient_missing_marker_count[
                                None
                            ],
                            1,
                        )
            else:
                gradient, gradient_valid = self._sample_velocity_gradient(
                    velocity_field,
                    obstacle_field,
                    grid_coordinate.x,
                    grid_coordinate.y,
                    grid_coordinate.z,
                    nx,
                    ny,
                    nz,
                    cell_center_x_m,
                    cell_center_y_m,
                    cell_center_z_m,
                )
            stress_sample_valid = pressure_sample_valid
            if viscosity_pa_s > 0.0 and gradient_valid == 0:
                stress_sample_valid = False
                ti.atomic_add(
                    self.report_stress_viscous_gradient_invalid_marker_count[None],
                    1,
                )
            if stress_sample_valid:
                viscous_stress = viscosity_pa_s * (gradient + gradient.transpose())
                traction = pressure_traction + viscous_stress @ normal
                self.t_gamma_pa[marker] = traction
                self.report_stress_valid_marker_count[None] += 1
                ti.atomic_max(
                    self.report_stress_max_abs_traction_pa[None],
                    traction.norm(),
                )
            else:
                self.t_gamma_pa[marker] = ti.Vector([0.0, 0.0, 0.0])
                self.report_stress_invalid_marker_count[None] += 1

    def sample_fluid_stress_to_marker_tractions(
        self,
        velocity_field,
        pressure_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        grid_nodes: tuple[int, int, int],
        *,
        viscosity_pa_s: float,
        two_sided_pressure: bool = False,
        far_pressure_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        use_pressure_anchor_fallback: bool = False,
    ) -> HibmMpmFluidStressSampleReport:
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        viscosity = float(viscosity_pa_s)
        if not math.isfinite(viscosity) or viscosity < 0.0:
            raise ValueError("viscosity_pa_s must be a finite non-negative number")
        far_region_id = int(far_pressure_region_id)
        far_pressure = float(far_pressure_pa)
        if not math.isfinite(far_pressure):
            raise ValueError("far_pressure_pa must be a finite number")
        far_inside_probe_max = float(far_pressure_inside_probe_max_multiplier)
        if not math.isfinite(far_inside_probe_max) or far_inside_probe_max < 3.0:
            raise ValueError(
                "far_pressure_inside_probe_max_multiplier must be finite and >= 3.0"
            )
        self._sample_fluid_stress_to_marker_tractions_kernel(
            velocity_field,
            pressure_field,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            cell_width_x_m,
            cell_width_y_m,
            cell_width_z_m,
            self.marker_pressure_anchor_cell,
            int(self.marker_count),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
            viscosity,
            1 if bool(two_sided_pressure) else 0,
            far_region_id,
            far_pressure,
            far_inside_probe_max,
            1 if bool(use_pressure_anchor_fallback) else 0,
        )
        return HibmMpmFluidStressSampleReport(
            valid_marker_count=int(self.report_stress_valid_marker_count[None]),
            invalid_marker_count=int(self.report_stress_invalid_marker_count[None]),
            max_abs_traction_pa=float(
                self.report_stress_max_abs_traction_pa[None]
            ),
            two_sided_pressure_marker_count=int(
                self.report_stress_two_sided_pressure_marker_count[None]
            ),
            viscous_gradient_invalid_marker_count=int(
                self.report_stress_viscous_gradient_invalid_marker_count[None]
            ),
            far_pressure_closed_marker_count=int(
                self.report_stress_far_pressure_closed_marker_count[None]
            ),
            far_pressure_closed_extended_marker_count=int(
                self.report_stress_far_pressure_closed_extended_marker_count[None]
            ),
            far_pressure_anchor_closed_marker_count=int(
                self.report_stress_far_pressure_anchor_closed_marker_count[None]
            ),
            closure_gradient_missing_marker_count=int(
                self.report_stress_closure_gradient_missing_marker_count[None]
            ),
        )

    @ti.kernel
    def _sample_no_slip_residual_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_no_slip_valid_marker_count[None] = 0
        self.report_no_slip_invalid_marker_count[None] = 0
        self.report_no_slip_max_residual_mps[None] = 0.0
        self.report_no_slip_sum_residual2_mps2[None] = ti.cast(0.0, ti.f64)
        for marker in range(marker_count):
            grid_coordinate = self._grid_coordinate_from_fields(
                self.x_gamma_m[marker],
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
            fluid_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                velocity_field,
                obstacle_field,
                grid_coordinate.x,
                grid_coordinate.y,
                grid_coordinate.z,
                nx,
                ny,
                nz,
            )
            if fluid_weight > 1.0e-12:
                residual = fluid_velocity - self.v_gamma_mps[marker]
                residual_norm = residual.norm()
                self.report_no_slip_valid_marker_count[None] += 1
                ti.atomic_max(
                    self.report_no_slip_max_residual_mps[None],
                    residual_norm,
                )
                self.report_no_slip_sum_residual2_mps2[None] += ti.cast(
                    residual_norm * residual_norm,
                    ti.f64,
                )
            else:
                self.report_no_slip_invalid_marker_count[None] += 1

    def sample_no_slip_residual(
        self,
        velocity_field,
        obstacle_field,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
    ) -> HibmMpmNoSlipResidualReport:
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._sample_no_slip_residual_kernel(
            velocity_field,
            obstacle_field,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(self.marker_count),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        valid_count = int(self.report_no_slip_valid_marker_count[None])
        sum_residual2 = float(self.report_no_slip_sum_residual2_mps2[None])
        l2_residual = (
            math.sqrt(sum_residual2 / float(valid_count)) if valid_count else 0.0
        )
        return HibmMpmNoSlipResidualReport(
            valid_marker_count=valid_count,
            invalid_marker_count=int(self.report_no_slip_invalid_marker_count[None]),
            max_no_slip_residual_mps=float(
                self.report_no_slip_max_residual_mps[None]
            ),
            l2_no_slip_residual_mps=l2_residual,
        )

    @ti.func
    def _marker_particle_shape_weight(
        self,
        marker_position_m,
        particle_position_m,
        support_radius_m: ti.f32,
    ):
        relative = marker_position_m - particle_position_m
        wx = ti.max(1.0 - ti.abs(relative.x) / support_radius_m, 0.0)
        wy = ti.max(1.0 - ti.abs(relative.y) / support_radius_m, 0.0)
        wz = ti.max(1.0 - ti.abs(relative.z) / support_radius_m, 0.0)
        return wx * wy * wz

    @ti.kernel
    def _clear_mpm_external_forces_kernel(
        self,
        external_force_n: ti.template(),
        particle_count: ti.i32,
    ):
        self.report_mpm_external_force_clear_count[None] = 0
        self.report_mpm_external_force_clear_max_abs_n[None] = 0.0
        for particle in range(particle_count):
            force = external_force_n[particle]
            force_norm = force.norm()
            if force_norm > 0.0:
                ti.atomic_add(self.report_mpm_external_force_clear_count[None], 1)
                ti.atomic_max(
                    self.report_mpm_external_force_clear_max_abs_n[None],
                    force_norm,
                )
            external_force_n[particle] = ti.Vector([0.0, 0.0, 0.0])

    def clear_mpm_external_forces(
        self,
        external_force_n,
        *,
        particle_count: int,
    ) -> HibmMpmExternalForceClearReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        self._clear_mpm_external_forces_kernel(external_force_n, particles)
        return HibmMpmExternalForceClearReport(
            cleared_particle_count=int(
                self.report_mpm_external_force_clear_count[None]
            ),
            max_abs_external_force_before_n=float(
                self.report_mpm_external_force_clear_max_abs_n[None]
            ),
        )

    @ti.kernel
    def _scatter_marker_forces_to_mpm_particles_kernel(
        self,
        external_force_n: ti.template(),
        particle_position_m: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
    ):
        self.report_mpm_scatter_marker_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_mpm_scatter_external_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_mpm_scatter_active_marker_count[None] = 0
        self.report_mpm_scatter_invalid_marker_count[None] = 0
        self.report_mpm_scatter_active_particle_count[None] = 0
        for marker in range(marker_count):
            marker_position = self.x_gamma_m[marker]
            marker_force = self.F_gamma_n[marker]
            weight_sum = 0.0
            for particle in range(particle_count):
                weight_sum += self._marker_particle_shape_weight(
                    marker_position,
                    particle_position_m[particle],
                    support_radius_m,
                )
            if weight_sum > 1.0e-12:
                self.report_mpm_scatter_active_marker_count[None] += 1
                self.report_mpm_scatter_marker_force_n[None] += marker_force
                for particle in range(particle_count):
                    weight = self._marker_particle_shape_weight(
                        marker_position,
                        particle_position_m[particle],
                        support_radius_m,
                    )
                    if weight > 0.0:
                        force_contribution = marker_force * (weight / weight_sum)
                        external_force_n[particle] += force_contribution
                        self.report_mpm_scatter_external_force_n[None] += (
                            force_contribution
                        )
                        self.report_mpm_scatter_active_particle_count[None] += 1
            else:
                self.report_mpm_scatter_invalid_marker_count[None] += 1

    def scatter_marker_forces_to_mpm_particles(
        self,
        external_force_n,
        particle_position_m,
        *,
        particle_count: int,
        support_radius_m: float,
    ) -> HibmMpmMpmForceScatterReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        self._scatter_marker_forces_to_mpm_particles_kernel(
            external_force_n,
            particle_position_m,
            int(self.marker_count),
            particles,
            support_radius,
        )
        total_marker_force = self._vector_field_tuple(
            self.report_mpm_scatter_marker_force_n
        )
        total_external_force = self._vector_field_tuple(
            self.report_mpm_scatter_external_force_n
        )
        residual = math.sqrt(
            sum(
                (marker_component - external_component)
                * (marker_component - external_component)
                for marker_component, external_component in zip(
                    total_marker_force,
                    total_external_force,
                    strict=True,
                )
            )
        )
        return HibmMpmMpmForceScatterReport(
            active_marker_count=int(self.report_mpm_scatter_active_marker_count[None]),
            invalid_marker_count=int(
                self.report_mpm_scatter_invalid_marker_count[None]
            ),
            active_particle_count=int(
                self.report_mpm_scatter_active_particle_count[None]
            ),
            total_marker_force_n=total_marker_force,
            total_mpm_external_force_n=total_external_force,
            action_reaction_residual_n=residual,
        )

    @ti.kernel
    def _update_surface_feedback_from_mpm_particles_kernel(
        self,
        particle_position_m: ti.template(),
        particle_velocity_mps: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
        dt_s: ti.f32,
    ):
        self.report_surface_feedback_updated_marker_count[None] = 0
        self.report_surface_feedback_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_displacement_m[None] = 0.0
        self.report_surface_feedback_max_speed_mps[None] = 0.0
        for marker in range(marker_count):
            old_position = self.x_gamma_m[marker]
            velocity_sum = ti.Vector([0.0, 0.0, 0.0])
            weight_sum = 0.0
            for particle in range(particle_count):
                weight = self._marker_particle_shape_weight(
                    old_position,
                    particle_position_m[particle],
                    support_radius_m,
                )
                if weight > 0.0:
                    velocity_sum += weight * particle_velocity_mps[particle]
                    weight_sum += weight
            if weight_sum > 1.0e-12:
                new_velocity = velocity_sum / weight_sum
                new_position = old_position + dt_s * new_velocity
                self.x_gamma_m[marker] = new_position
                self.v_gamma_mps[marker] = new_velocity
                self.report_surface_feedback_updated_marker_count[None] += 1
                ti.atomic_max(
                    self.report_surface_feedback_max_displacement_m[None],
                    (new_position - old_position).norm(),
                )
                ti.atomic_max(
                    self.report_surface_feedback_max_speed_mps[None],
                    new_velocity.norm(),
                )
            else:
                self.report_surface_feedback_invalid_marker_count[None] += 1

    def update_surface_feedback_from_mpm_particles(
        self,
        particle_position_m,
        particle_velocity_mps,
        *,
        particle_count: int,
        support_radius_m: float,
        dt_s: float,
    ) -> HibmMpmSurfaceUpdateReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        feedback_dt = float(dt_s)
        if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
            raise ValueError("dt_s must be a finite positive number")
        self._update_surface_feedback_from_mpm_particles_kernel(
            particle_position_m,
            particle_velocity_mps,
            int(self.marker_count),
            particles,
            support_radius,
            feedback_dt,
        )
        return HibmMpmSurfaceUpdateReport(
            updated_marker_count=int(
                self.report_surface_feedback_updated_marker_count[None]
            ),
            invalid_marker_count=int(
                self.report_surface_feedback_invalid_marker_count[None]
            ),
            max_marker_displacement_m=float(
                self.report_surface_feedback_max_displacement_m[None]
            ),
            max_marker_speed_mps=float(
                self.report_surface_feedback_max_speed_mps[None]
            ),
        )

    @ti.kernel
    def _update_surface_feedback_from_mpm_surface_particles_kernel(
        self,
        particle_position_m: ti.template(),
        particle_velocity_mps: ti.template(),
        particle_normal: ti.template(),
        particle_area_m2: ti.template(),
        marker_count: ti.i32,
        particle_count: ti.i32,
        support_radius_m: ti.f32,
        dt_s: ti.f32,
    ):
        self.report_surface_feedback_updated_marker_count[None] = 0
        self.report_surface_feedback_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_displacement_m[None] = 0.0
        self.report_surface_feedback_max_speed_mps[None] = 0.0
        self.report_surface_feedback_geometry_updated_marker_count[None] = 0
        self.report_surface_feedback_geometry_invalid_marker_count[None] = 0
        self.report_surface_feedback_max_normal_change[None] = 0.0
        self.report_surface_feedback_max_area_change_m2[None] = 0.0
        for marker in range(marker_count):
            old_position = self.x_gamma_m[marker]
            old_normal = self.n_gamma[marker]
            old_area = self.A_gamma_m2[marker]
            velocity_sum = ti.Vector([0.0, 0.0, 0.0])
            normal_sum = ti.Vector([0.0, 0.0, 0.0])
            area_sum = 0.0
            weight_sum = 0.0
            geometry_weight_sum = 0.0
            for particle in range(particle_count):
                weight = self._marker_particle_shape_weight(
                    old_position,
                    particle_position_m[particle],
                    support_radius_m,
                )
                if weight > 0.0:
                    velocity_sum += weight * particle_velocity_mps[particle]
                    weight_sum += weight
                    particle_surface_normal = particle_normal[particle]
                    particle_surface_area = particle_area_m2[particle]
                    if (
                        particle_surface_normal.norm() > 1.0e-12
                        and particle_surface_area > 0.0
                    ):
                        normal_sum += weight * particle_surface_normal
                        area_sum += weight * particle_surface_area
                        geometry_weight_sum += weight
            if weight_sum > 1.0e-12:
                new_velocity = velocity_sum / weight_sum
                new_position = old_position + dt_s * new_velocity
                self.x_gamma_m[marker] = new_position
                self.v_gamma_mps[marker] = new_velocity
                self.report_surface_feedback_updated_marker_count[None] += 1
                ti.atomic_max(
                    self.report_surface_feedback_max_displacement_m[None],
                    (new_position - old_position).norm(),
                )
                ti.atomic_max(
                    self.report_surface_feedback_max_speed_mps[None],
                    new_velocity.norm(),
                )
                if geometry_weight_sum > 1.0e-12 and normal_sum.norm() > 1.0e-12:
                    new_normal = normal_sum.normalized()
                    new_area = area_sum / geometry_weight_sum
                    self.n_gamma[marker] = new_normal
                    self.A_gamma_m2[marker] = new_area
                    self.report_surface_feedback_geometry_updated_marker_count[
                        None
                    ] += 1
                    ti.atomic_max(
                        self.report_surface_feedback_max_normal_change[None],
                        (new_normal - old_normal).norm(),
                    )
                    ti.atomic_max(
                        self.report_surface_feedback_max_area_change_m2[None],
                        ti.abs(new_area - old_area),
                    )
                else:
                    self.report_surface_feedback_geometry_invalid_marker_count[
                        None
                    ] += 1
            else:
                self.report_surface_feedback_invalid_marker_count[None] += 1
                self.report_surface_feedback_geometry_invalid_marker_count[None] += 1

    def update_surface_feedback_from_mpm_surface_particles(
        self,
        particle_position_m,
        particle_velocity_mps,
        particle_normal,
        particle_area_m2,
        *,
        particle_count: int,
        support_radius_m: float,
        dt_s: float,
    ) -> HibmMpmSurfaceUpdateReport:
        particles = int(particle_count)
        if particles <= 0:
            raise ValueError("particle_count must be positive")
        support_radius = float(support_radius_m)
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius_m must be a finite positive number")
        feedback_dt = float(dt_s)
        if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
            raise ValueError("dt_s must be a finite positive number")
        self._update_surface_feedback_from_mpm_surface_particles_kernel(
            particle_position_m,
            particle_velocity_mps,
            particle_normal,
            particle_area_m2,
            int(self.marker_count),
            particles,
            support_radius,
            feedback_dt,
        )
        return HibmMpmSurfaceUpdateReport(
            updated_marker_count=int(
                self.report_surface_feedback_updated_marker_count[None]
            ),
            invalid_marker_count=int(
                self.report_surface_feedback_invalid_marker_count[None]
            ),
            max_marker_displacement_m=float(
                self.report_surface_feedback_max_displacement_m[None]
            ),
            max_marker_speed_mps=float(
                self.report_surface_feedback_max_speed_mps[None]
            ),
            geometry_updated_marker_count=int(
                self.report_surface_feedback_geometry_updated_marker_count[None]
            ),
            geometry_invalid_marker_count=int(
                self.report_surface_feedback_geometry_invalid_marker_count[None]
            ),
            max_marker_normal_change=float(
                self.report_surface_feedback_max_normal_change[None]
            ),
            max_marker_area_change_m2=float(
                self.report_surface_feedback_max_area_change_m2[None]
            ),
        )

    @ti.kernel
    def _aggregate_region_forces_kernel(
        self,
        marker_count: ti.i32,
        primary_region_id: ti.i32,
        secondary_region_id: ti.i32,
    ):
        self.report_primary_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_secondary_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_total_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_primary_marker_count[None] = 0
        self.report_secondary_marker_count[None] = 0
        self.report_total_marker_count[None] = 0
        for marker in range(marker_count):
            force = self.F_gamma_n[marker]
            self.report_total_force_n[None] += force
            self.report_total_marker_count[None] += 1
            if self.region_id[marker] == primary_region_id:
                self.report_primary_force_n[None] += force
                self.report_primary_marker_count[None] += 1
            if self.region_id[marker] == secondary_region_id:
                self.report_secondary_force_n[None] += force
                self.report_secondary_marker_count[None] += 1

    def aggregate_region_forces(
        self,
        *,
        primary_region_id: int,
        secondary_region_id: int,
    ) -> HibmMpmSurfaceMarkerForceReport:
        self._aggregate_region_forces_kernel(
            int(self.marker_count),
            int(primary_region_id),
            int(secondary_region_id),
        )
        primary = self._vector_field_tuple(self.report_primary_force_n)
        secondary = self._vector_field_tuple(self.report_secondary_force_n)
        total = self._vector_field_tuple(self.report_total_force_n)
        primary_count = int(self.report_primary_marker_count[None])
        secondary_count = int(self.report_secondary_marker_count[None])
        total_count = int(self.report_total_marker_count[None])
        fluid_reaction = tuple(-component for component in total)
        residual = math.sqrt(
            sum(
                (total_component + reaction_component)
                * (total_component + reaction_component)
                for total_component, reaction_component in zip(
                    total,
                    fluid_reaction,
                    strict=True,
                )
            )
        )
        return HibmMpmSurfaceMarkerForceReport(
            primary_marker_force_n=primary,
            secondary_marker_force_n=secondary,
            total_marker_force_n=total,
            primary_marker_count=primary_count,
            secondary_marker_count=secondary_count,
            total_marker_count=total_count,
            fluid_reaction_force_n=fluid_reaction,
            action_reaction_residual_n=residual,
        )

    def marker_force_n(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.F_gamma_n, marker_index)

    def marker_traction_pa(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.t_gamma_pa, marker_index)

    def marker_normal(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.n_gamma, marker_index)

    def marker_velocity_mps(self, marker_index: int) -> tuple[float, float, float]:
        return self._vector_value_tuple(self.v_gamma_mps, marker_index)

    def marker_region_id(self, marker_index: int) -> int:
        return int(self.region_id[int(marker_index)])

    @staticmethod
    def _vector_value_tuple(field, index: int) -> tuple[float, float, float]:
        value = field[int(index)]
        return (float(value[0]), float(value[1]), float(value[2]))

    @staticmethod
    def _vector_field_tuple(field) -> tuple[float, float, float]:
        value = field[None]
        return (float(value[0]), float(value[1]), float(value[2]))


@ti.data_oriented
class HibmMpmIbNodeSearch:
    _NODE_NONE = 0
    _NODE_EXTERNAL_IB = 1
    _NODE_INTERNAL = 2
    _F32_EPSILON = 1.1920928955078125e-7

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        bounds_min_m: Sequence[float],
        bounds_max_m: Sequence[float],
        marker_capacity: int,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value <= 0 for value in nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        bounds_min = _vector3(bounds_min_m, name="bounds_min_m")
        bounds_max = _vector3(bounds_max_m, name="bounds_max_m")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.grid_nodes = nodes
        self.bounds_min_m = bounds_min
        self.bounds_max_m = bounds_max
        self.spacing_m = tuple(
            (hi - lo) / float(count)
            for lo, hi, count in zip(bounds_min, bounds_max, nodes, strict=True)
        )
        self.marker_capacity = int(marker_capacity)

        self.node_kind_code = ti.field(dtype=ti.i32, shape=nodes)
        self.nearest_marker = ti.field(dtype=ti.i32, shape=nodes)
        self.node_signed_distance_m = ti.field(dtype=ti.f32, shape=nodes)
        self.node_boundary_point_m = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        self.node_interior_fluid_point_m = ti.Vector.field(3, dtype=ti.f32, shape=nodes)
        self.node_projection_marker_indices = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=nodes,
        )
        self.node_projection_marker_weights = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )

        self.report_near_boundary_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_external_ib_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_internal_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_invalid_projection_count = ti.field(dtype=ti.i32, shape=())

    def _default_sign_tolerance_m(self) -> float:
        min_spacing_m = max(min(self.spacing_m), 1.0e-12)
        max_extent_m = max(
            hi - lo for lo, hi in zip(self.bounds_min_m, self.bounds_max_m, strict=True)
        )
        max_coordinate_m = max(
            max(abs(value) for value in self.bounds_min_m),
            max(abs(value) for value in self.bounds_max_m),
        )
        coordinate_scale_m = max(max_extent_m, max_coordinate_m, min_spacing_m, 1.0)
        return max(
            1.0e-12,
            1.0e-6 * min_spacing_m,
            2.0 * self._F32_EPSILON * coordinate_scale_m,
        )

    @ti.func
    def _closest_point_on_triangle(self, position, a, b, c):
        ab = b - a
        ac = c - a
        face_normal = ab.cross(ac)
        normal_norm = face_normal.norm()
        closest = a
        valid = 0
        if normal_norm > 1.0e-12:
            valid = 1
            face_normal = face_normal / normal_norm
            ap = position - a
            d1 = ab.dot(ap)
            d2 = ac.dot(ap)
            if d1 <= 0.0 and d2 <= 0.0:
                closest = a
            else:
                bp = position - b
                d3 = ab.dot(bp)
                d4 = ac.dot(bp)
                if d3 >= 0.0 and d4 <= d3:
                    closest = b
                else:
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        edge_fraction = d1 / ti.max(d1 - d3, 1.0e-30)
                        closest = a + edge_fraction * ab
                    else:
                        cp = position - c
                        d5 = ab.dot(cp)
                        d6 = ac.dot(cp)
                        if d6 >= 0.0 and d5 <= d6:
                            closest = c
                        else:
                            vb = d5 * d2 - d1 * d6
                            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                edge_fraction = d2 / ti.max(d2 - d6, 1.0e-30)
                                closest = a + edge_fraction * ac
                            else:
                                va = d3 * d6 - d5 * d4
                                if (
                                    va <= 0.0
                                    and d4 - d3 >= 0.0
                                    and d5 - d6 >= 0.0
                                ):
                                    edge_fraction = (d4 - d3) / ti.max(
                                        d4 - d3 + d5 - d6,
                                        1.0e-30,
                                    )
                                    closest = b + edge_fraction * (c - b)
                                else:
                                    denom = ti.max(va + vb + vc, 1.0e-30)
                                    v = vb / denom
                                    w = vc / denom
                                    closest = a + ab * v + ac * w
        return closest, face_normal, valid

    @ti.func
    def _oriented_triangle_normal(self, face_normal, marker_normals, ia, ib, ic):
        normal = face_normal
        average_normal = marker_normals[ia] + marker_normals[ib] + marker_normals[ic]
        average_norm = average_normal.norm()
        if average_norm > 1.0e-12:
            average_normal = average_normal / average_norm
            if normal.dot(average_normal) < 0.0:
                normal = -normal
        return normal

    @ti.func
    def _barycentric_weights_on_triangle(self, point, a, b, c):
        ab = b - a
        ac = c - a
        ap = point - a
        d00 = ab.dot(ab)
        d01 = ab.dot(ac)
        d11 = ac.dot(ac)
        d20 = ap.dot(ab)
        d21 = ap.dot(ac)
        denom = d00 * d11 - d01 * d01
        weights = ti.Vector([1.0, 0.0, 0.0])
        if ti.abs(denom) > 1.0e-30:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            u = ti.min(ti.max(u, 0.0), 1.0)
            v = ti.min(ti.max(v, 0.0), 1.0)
            w = ti.min(ti.max(w, 0.0), 1.0)
            total = ti.max(u + v + w, 1.0e-30)
            weights = ti.Vector([u / total, v / total, w / total])
        return weights

    @ti.func
    def _nearest_triangle_marker(self, position, marker_positions_m, ia, ib, ic):
        nearest = ia
        offset_a = position - marker_positions_m[ia]
        nearest_distance2 = offset_a.dot(offset_a)
        offset_b = position - marker_positions_m[ib]
        distance_b2 = offset_b.dot(offset_b)
        if distance_b2 < nearest_distance2:
            nearest_distance2 = distance_b2
            nearest = ib
        offset_c = position - marker_positions_m[ic]
        distance_c2 = offset_c.dot(offset_c)
        if distance_c2 < nearest_distance2:
            nearest = ic
        return nearest

    @ti.kernel
    def _search_and_classify_kernel(
        self,
        marker_positions_m: ti.template(),
        marker_normals: ti.template(),
        projection_triangle_indices: ti.template(),
        marker_count: ti.i32,
        projection_triangle_count: ti.i32,
        search_radius_m: ti.f32,
        interior_probe_distance_m: ti.f32,
        sign_tolerance_m: ti.f32,
        classify_far_internal_nodes: ti.i32,
        bounds_min_x_m: ti.f32,
        bounds_min_y_m: ti.f32,
        bounds_min_z_m: ti.f32,
        spacing_x_m: ti.f32,
        spacing_y_m: ti.f32,
        spacing_z_m: ti.f32,
    ):
        self.report_near_boundary_node_count[None] = 0
        self.report_external_ib_node_count[None] = 0
        self.report_internal_node_count[None] = 0
        self.report_invalid_projection_count[None] = 0
        for node in ti.grouped(self.node_kind_code):
            position = ti.Vector(
                [
                    bounds_min_x_m + (ti.cast(node[0], ti.f32) + 0.5) * spacing_x_m,
                    bounds_min_y_m + (ti.cast(node[1], ti.f32) + 0.5) * spacing_y_m,
                    bounds_min_z_m + (ti.cast(node[2], ti.f32) + 0.5) * spacing_z_m,
                ]
            )
            self.node_kind_code[node] = self._NODE_NONE
            self.nearest_marker[node] = -1
            self.node_signed_distance_m[node] = 0.0
            self.node_boundary_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_interior_fluid_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_projection_marker_indices[node] = ti.Vector([-1, -1, -1])
            self.node_projection_marker_weights[node] = ti.Vector([0.0, 0.0, 0.0])

            nearest = -1
            nearest_distance = 1.0e30
            nearest_signed_distance = 0.0
            nearest_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_projection_indices = ti.Vector([-1, -1, -1])
            nearest_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_external = -1
            nearest_external_distance = 1.0e30
            nearest_external_signed_distance = 0.0
            nearest_external_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_external_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_external_projection_indices = ti.Vector([-1, -1, -1])
            nearest_external_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_global = -1
            nearest_global_distance = 1.0e30
            nearest_global_signed_distance = 0.0
            nearest_global_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_global_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_global_projection_indices = ti.Vector([-1, -1, -1])
            nearest_global_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            external_seen = 0
            global_external_seen = 0
            if projection_triangle_count > 0:
                for triangle_index in range(projection_triangle_count):
                    triangle = projection_triangle_indices[triangle_index]
                    ia = triangle.x
                    ib = triangle.y
                    ic = triangle.z
                    closest, face_normal, valid = self._closest_point_on_triangle(
                        position,
                        marker_positions_m[ia],
                        marker_positions_m[ib],
                        marker_positions_m[ic],
                    )
                    if valid != 0:
                        normal = self._oriented_triangle_normal(
                            face_normal,
                            marker_normals,
                            ia,
                            ib,
                            ic,
                        )
                        offset = position - closest
                        distance = offset.norm()
                        signed_distance = offset.dot(normal)
                        projection_weights = self._barycentric_weights_on_triangle(
                            closest,
                            marker_positions_m[ia],
                            marker_positions_m[ib],
                            marker_positions_m[ic],
                        )
                        projection_indices = ti.Vector([ia, ib, ic])
                        marker = self._nearest_triangle_marker(
                            position,
                            marker_positions_m,
                            ia,
                            ib,
                            ic,
                        )
                        if classify_far_internal_nodes != 0:
                            if signed_distance > sign_tolerance_m:
                                global_external_seen = 1
                            if distance < nearest_global_distance:
                                nearest_global_distance = distance
                                nearest_global = marker
                                nearest_global_signed_distance = signed_distance
                                nearest_global_boundary_point = closest
                                nearest_global_normal = normal
                                nearest_global_projection_indices = projection_indices
                                nearest_global_projection_weights = projection_weights
                        if distance < search_radius_m:
                            if signed_distance > sign_tolerance_m:
                                external_seen = 1
                                if distance < nearest_external_distance:
                                    nearest_external_distance = distance
                                    nearest_external = marker
                                    nearest_external_signed_distance = signed_distance
                                    nearest_external_boundary_point = closest
                                    nearest_external_normal = normal
                                    nearest_external_projection_indices = (
                                        projection_indices
                                    )
                                    nearest_external_projection_weights = (
                                        projection_weights
                                    )
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest = marker
                                nearest_signed_distance = signed_distance
                                nearest_boundary_point = closest
                                nearest_normal = normal
                                nearest_projection_indices = projection_indices
                                nearest_projection_weights = projection_weights
            else:
                for marker in range(marker_count):
                    offset = position - marker_positions_m[marker]
                    distance = offset.norm()
                    signed_distance = offset.dot(marker_normals[marker])
                    if classify_far_internal_nodes != 0:
                        if signed_distance > sign_tolerance_m:
                            global_external_seen = 1
                        if distance < nearest_global_distance:
                            nearest_global_distance = distance
                            nearest_global = marker
                            nearest_global_signed_distance = signed_distance
                            nearest_global_boundary_point = marker_positions_m[marker]
                            nearest_global_normal = marker_normals[marker]
                            nearest_global_projection_indices = ti.Vector(
                                [marker, -1, -1]
                            )
                            nearest_global_projection_weights = ti.Vector(
                                [1.0, 0.0, 0.0]
                            )
                    if distance < search_radius_m:
                        if signed_distance > sign_tolerance_m:
                            external_seen = 1
                            if distance < nearest_external_distance:
                                nearest_external_distance = distance
                                nearest_external = marker
                                nearest_external_signed_distance = signed_distance
                                nearest_external_boundary_point = marker_positions_m[
                                    marker
                                ]
                                nearest_external_normal = marker_normals[marker]
                                nearest_external_projection_indices = ti.Vector(
                                    [marker, -1, -1]
                                )
                                nearest_external_projection_weights = ti.Vector(
                                    [1.0, 0.0, 0.0]
                                )
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest = marker
                            nearest_signed_distance = signed_distance
                            nearest_boundary_point = marker_positions_m[marker]
                            nearest_normal = marker_normals[marker]
                            nearest_projection_indices = ti.Vector([marker, -1, -1])
                            nearest_projection_weights = ti.Vector([1.0, 0.0, 0.0])

            if nearest >= 0:
                self.report_near_boundary_node_count[None] += 1
                selected = nearest
                selected_signed_distance = nearest_signed_distance
                boundary_point = nearest_boundary_point
                normal = nearest_normal
                selected_projection_indices = nearest_projection_indices
                selected_projection_weights = nearest_projection_weights
                if external_seen == 1 and nearest_external >= 0:
                    selected = nearest_external
                    selected_signed_distance = nearest_external_signed_distance
                    boundary_point = nearest_external_boundary_point
                    normal = nearest_external_normal
                    selected_projection_indices = nearest_external_projection_indices
                    selected_projection_weights = nearest_external_projection_weights
                self.nearest_marker[node] = selected
                self.node_signed_distance_m[node] = selected_signed_distance
                normal_distance = (position - boundary_point).dot(normal)
                interior_distance = interior_probe_distance_m
                if normal_distance > sign_tolerance_m:
                    interior_distance = normal_distance + interior_probe_distance_m
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = selected_projection_indices
                self.node_projection_marker_weights[node] = selected_projection_weights
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_distance
                )
                if ti.abs(selected_signed_distance) <= sign_tolerance_m:
                    self.report_invalid_projection_count[None] += 1
                if external_seen == 1:
                    self.node_kind_code[node] = self._NODE_EXTERNAL_IB
                    self.report_external_ib_node_count[None] += 1
                else:
                    self.node_kind_code[node] = self._NODE_INTERNAL
                    self.report_internal_node_count[None] += 1
            elif (
                classify_far_internal_nodes != 0
                and nearest_global >= 0
                and global_external_seen == 0
            ):
                self.node_kind_code[node] = self._NODE_INTERNAL
                self.nearest_marker[node] = nearest_global
                self.node_signed_distance_m[node] = nearest_global_signed_distance
                boundary_point = nearest_global_boundary_point
                normal = nearest_global_normal
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = (
                    nearest_global_projection_indices
                )
                self.node_projection_marker_weights[node] = (
                    nearest_global_projection_weights
                )
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_probe_distance_m
                )
                self.report_internal_node_count[None] += 1

    @ti.kernel
    def _search_and_classify_grid_fields_kernel(
        self,
        marker_positions_m: ti.template(),
        marker_normals: ti.template(),
        projection_triangle_indices: ti.template(),
        marker_count: ti.i32,
        projection_triangle_count: ti.i32,
        search_radius_m: ti.f32,
        interior_probe_distance_m: ti.f32,
        sign_tolerance_m: ti.f32,
        classify_far_internal_nodes: ti.i32,
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
    ):
        self.report_near_boundary_node_count[None] = 0
        self.report_external_ib_node_count[None] = 0
        self.report_internal_node_count[None] = 0
        self.report_invalid_projection_count[None] = 0
        for node in ti.grouped(self.node_kind_code):
            position = ti.Vector(
                [
                    cell_center_x_m[node[0]],
                    cell_center_y_m[node[1]],
                    cell_center_z_m[node[2]],
                ]
            )
            self.node_kind_code[node] = self._NODE_NONE
            self.nearest_marker[node] = -1
            self.node_signed_distance_m[node] = 0.0
            self.node_boundary_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_interior_fluid_point_m[node] = ti.Vector([0.0, 0.0, 0.0])
            self.node_projection_marker_indices[node] = ti.Vector([-1, -1, -1])
            self.node_projection_marker_weights[node] = ti.Vector([0.0, 0.0, 0.0])

            nearest = -1
            nearest_distance = 1.0e30
            nearest_signed_distance = 0.0
            nearest_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_projection_indices = ti.Vector([-1, -1, -1])
            nearest_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_external = -1
            nearest_external_distance = 1.0e30
            nearest_external_signed_distance = 0.0
            nearest_external_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_external_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_external_projection_indices = ti.Vector([-1, -1, -1])
            nearest_external_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            nearest_global = -1
            nearest_global_distance = 1.0e30
            nearest_global_signed_distance = 0.0
            nearest_global_boundary_point = ti.Vector([0.0, 0.0, 0.0])
            nearest_global_normal = ti.Vector([0.0, 0.0, 1.0])
            nearest_global_projection_indices = ti.Vector([-1, -1, -1])
            nearest_global_projection_weights = ti.Vector([0.0, 0.0, 0.0])
            external_seen = 0
            global_external_seen = 0
            if projection_triangle_count > 0:
                for triangle_index in range(projection_triangle_count):
                    triangle = projection_triangle_indices[triangle_index]
                    ia = triangle.x
                    ib = triangle.y
                    ic = triangle.z
                    closest, face_normal, valid = self._closest_point_on_triangle(
                        position,
                        marker_positions_m[ia],
                        marker_positions_m[ib],
                        marker_positions_m[ic],
                    )
                    if valid != 0:
                        normal = self._oriented_triangle_normal(
                            face_normal,
                            marker_normals,
                            ia,
                            ib,
                            ic,
                        )
                        offset = position - closest
                        distance = offset.norm()
                        signed_distance = offset.dot(normal)
                        projection_weights = self._barycentric_weights_on_triangle(
                            closest,
                            marker_positions_m[ia],
                            marker_positions_m[ib],
                            marker_positions_m[ic],
                        )
                        projection_indices = ti.Vector([ia, ib, ic])
                        marker = self._nearest_triangle_marker(
                            position,
                            marker_positions_m,
                            ia,
                            ib,
                            ic,
                        )
                        if classify_far_internal_nodes != 0:
                            if signed_distance > sign_tolerance_m:
                                global_external_seen = 1
                            if distance < nearest_global_distance:
                                nearest_global_distance = distance
                                nearest_global = marker
                                nearest_global_signed_distance = signed_distance
                                nearest_global_boundary_point = closest
                                nearest_global_normal = normal
                                nearest_global_projection_indices = projection_indices
                                nearest_global_projection_weights = projection_weights
                        if distance < search_radius_m:
                            if signed_distance > sign_tolerance_m:
                                external_seen = 1
                                if distance < nearest_external_distance:
                                    nearest_external_distance = distance
                                    nearest_external = marker
                                    nearest_external_signed_distance = signed_distance
                                    nearest_external_boundary_point = closest
                                    nearest_external_normal = normal
                                    nearest_external_projection_indices = (
                                        projection_indices
                                    )
                                    nearest_external_projection_weights = (
                                        projection_weights
                                    )
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest = marker
                                nearest_signed_distance = signed_distance
                                nearest_boundary_point = closest
                                nearest_normal = normal
                                nearest_projection_indices = projection_indices
                                nearest_projection_weights = projection_weights
            else:
                for marker in range(marker_count):
                    offset = position - marker_positions_m[marker]
                    distance = offset.norm()
                    signed_distance = offset.dot(marker_normals[marker])
                    if classify_far_internal_nodes != 0:
                        if signed_distance > sign_tolerance_m:
                            global_external_seen = 1
                        if distance < nearest_global_distance:
                            nearest_global_distance = distance
                            nearest_global = marker
                            nearest_global_signed_distance = signed_distance
                            nearest_global_boundary_point = marker_positions_m[marker]
                            nearest_global_normal = marker_normals[marker]
                            nearest_global_projection_indices = ti.Vector(
                                [marker, -1, -1]
                            )
                            nearest_global_projection_weights = ti.Vector(
                                [1.0, 0.0, 0.0]
                            )
                    if distance < search_radius_m:
                        if signed_distance > sign_tolerance_m:
                            external_seen = 1
                            if distance < nearest_external_distance:
                                nearest_external_distance = distance
                                nearest_external = marker
                                nearest_external_signed_distance = signed_distance
                                nearest_external_boundary_point = marker_positions_m[
                                    marker
                                ]
                                nearest_external_normal = marker_normals[marker]
                                nearest_external_projection_indices = ti.Vector(
                                    [marker, -1, -1]
                                )
                                nearest_external_projection_weights = ti.Vector(
                                    [1.0, 0.0, 0.0]
                                )
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest = marker
                            nearest_signed_distance = signed_distance
                            nearest_boundary_point = marker_positions_m[marker]
                            nearest_normal = marker_normals[marker]
                            nearest_projection_indices = ti.Vector([marker, -1, -1])
                            nearest_projection_weights = ti.Vector([1.0, 0.0, 0.0])

            if nearest >= 0:
                self.report_near_boundary_node_count[None] += 1
                selected = nearest
                selected_signed_distance = nearest_signed_distance
                boundary_point = nearest_boundary_point
                normal = nearest_normal
                selected_projection_indices = nearest_projection_indices
                selected_projection_weights = nearest_projection_weights
                if external_seen == 1 and nearest_external >= 0:
                    selected = nearest_external
                    selected_signed_distance = nearest_external_signed_distance
                    boundary_point = nearest_external_boundary_point
                    normal = nearest_external_normal
                    selected_projection_indices = nearest_external_projection_indices
                    selected_projection_weights = nearest_external_projection_weights
                self.nearest_marker[node] = selected
                self.node_signed_distance_m[node] = selected_signed_distance
                normal_distance = (position - boundary_point).dot(normal)
                interior_distance = interior_probe_distance_m
                if normal_distance > sign_tolerance_m:
                    interior_distance = normal_distance + interior_probe_distance_m
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = selected_projection_indices
                self.node_projection_marker_weights[node] = selected_projection_weights
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_distance
                )
                if ti.abs(selected_signed_distance) <= sign_tolerance_m:
                    self.report_invalid_projection_count[None] += 1
                if external_seen == 1:
                    self.node_kind_code[node] = self._NODE_EXTERNAL_IB
                    self.report_external_ib_node_count[None] += 1
                else:
                    self.node_kind_code[node] = self._NODE_INTERNAL
                    self.report_internal_node_count[None] += 1
            elif (
                classify_far_internal_nodes != 0
                and nearest_global >= 0
                and global_external_seen == 0
            ):
                self.node_kind_code[node] = self._NODE_INTERNAL
                self.nearest_marker[node] = nearest_global
                self.node_signed_distance_m[node] = nearest_global_signed_distance
                boundary_point = nearest_global_boundary_point
                normal = nearest_global_normal
                self.node_boundary_point_m[node] = boundary_point
                self.node_projection_marker_indices[node] = (
                    nearest_global_projection_indices
                )
                self.node_projection_marker_weights[node] = (
                    nearest_global_projection_weights
                )
                self.node_interior_fluid_point_m[node] = (
                    boundary_point + normal * interior_probe_distance_m
                )
                self.report_internal_node_count[None] += 1

    def _validate_search_inputs(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None,
    ) -> tuple[float, float, float]:
        if int(markers.marker_count) > self.marker_capacity:
            raise ValueError("markers.marker_count exceeds marker_capacity")
        search_radius = float(search_radius_m)
        if not math.isfinite(search_radius) or search_radius <= 0.0:
            raise ValueError("search_radius_m must be a finite positive number")
        probe_distance = float(interior_probe_distance_m)
        if not math.isfinite(probe_distance) or probe_distance <= 0.0:
            raise ValueError(
                "interior_probe_distance_m must be a finite positive number"
            )
        sign_tolerance = (
            self._default_sign_tolerance_m()
            if sign_tolerance_m is None
            else float(sign_tolerance_m)
        )
        if not math.isfinite(sign_tolerance) or sign_tolerance < 0.0:
            raise ValueError("sign_tolerance_m must be a finite non-negative number")
        return search_radius, probe_distance, sign_tolerance

    def search_and_classify(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None = None,
        classify_far_internal_nodes: bool = False,
    ) -> HibmMpmIbNodeSearchReport:
        search_radius, probe_distance, sign_tolerance = self._validate_search_inputs(
            markers,
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            sign_tolerance_m=sign_tolerance_m,
        )
        self._search_and_classify_kernel(
            markers.x_gamma_m,
            markers.n_gamma,
            markers.projection_triangle_indices,
            int(markers.marker_count),
            int(markers.projection_triangle_count),
            search_radius,
            probe_distance,
            sign_tolerance,
            1 if bool(classify_far_internal_nodes) else 0,
            float(self.bounds_min_m[0]),
            float(self.bounds_min_m[1]),
            float(self.bounds_min_m[2]),
            float(self.spacing_m[0]),
            float(self.spacing_m[1]),
            float(self.spacing_m[2]),
        )
        return HibmMpmIbNodeSearchReport(
            near_boundary_node_count=int(self.report_near_boundary_node_count[None]),
            external_ib_node_count=int(self.report_external_ib_node_count[None]),
            internal_node_count=int(self.report_internal_node_count[None]),
            invalid_projection_count=int(self.report_invalid_projection_count[None]),
        )

    def search_and_classify_grid_fields(
        self,
        markers: HibmMpmSurfaceMarkers,
        *,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        search_radius_m: float,
        interior_probe_distance_m: float,
        sign_tolerance_m: float | None = None,
        classify_far_internal_nodes: bool = False,
    ) -> HibmMpmIbNodeSearchReport:
        search_radius, probe_distance, sign_tolerance = self._validate_search_inputs(
            markers,
            search_radius_m=search_radius_m,
            interior_probe_distance_m=interior_probe_distance_m,
            sign_tolerance_m=sign_tolerance_m,
        )
        self._search_and_classify_grid_fields_kernel(
            markers.x_gamma_m,
            markers.n_gamma,
            markers.projection_triangle_indices,
            int(markers.marker_count),
            int(markers.projection_triangle_count),
            search_radius,
            probe_distance,
            sign_tolerance,
            1 if bool(classify_far_internal_nodes) else 0,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
        )
        return HibmMpmIbNodeSearchReport(
            near_boundary_node_count=int(self.report_near_boundary_node_count[None]),
            external_ib_node_count=int(self.report_external_ib_node_count[None]),
            internal_node_count=int(self.report_internal_node_count[None]),
            invalid_projection_count=int(self.report_invalid_projection_count[None]),
        )

    def node_kind(self, node_index: tuple[int, int, int]) -> str:
        code = int(self.node_kind_code[self._node_index(node_index)])
        if code == self._NODE_EXTERNAL_IB:
            return "external_ib"
        if code == self._NODE_INTERNAL:
            return "internal"
        return "none"

    def nearest_marker_index(self, node_index: tuple[int, int, int]) -> int:
        return int(self.nearest_marker[self._node_index(node_index)])

    def signed_distance_m(self, node_index: tuple[int, int, int]) -> float:
        return float(self.node_signed_distance_m[self._node_index(node_index)])

    def boundary_point_m(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(self.node_boundary_point_m, node_index)

    def interior_fluid_point_m(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.node_interior_fluid_point_m,
            node_index,
        )

    def _node_index(self, node_index: tuple[int, int, int]) -> tuple[int, int, int]:
        index = tuple(int(value) for value in node_index)
        if len(index) != 3 or any(
            value < 0 or value >= limit
            for value, limit in zip(index, self.grid_nodes, strict=True)
        ):
            raise IndexError("node_index out of range")
        return index

    def _vector_field_value_tuple(
        self,
        field,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        value = field[self._node_index(node_index)]
        return (float(value[0]), float(value[1]), float(value[2]))


@ti.data_oriented
class HibmMpmIbBoundaryConditions:
    _NODE_EXTERNAL_IB = HibmMpmIbNodeSearch._NODE_EXTERNAL_IB
    _NODE_INTERNAL = HibmMpmIbNodeSearch._NODE_INTERNAL

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        marker_capacity: int,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        init_taichi(runtime)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value <= 0 for value in nodes):
            raise ValueError("grid_nodes must contain three positive integers")
        if int(marker_capacity) <= 0:
            raise ValueError("marker_capacity must be positive")
        self.grid_nodes = nodes
        self.marker_capacity = int(marker_capacity)

        self.active_ib_node = ti.field(dtype=ti.i32, shape=nodes)
        self.velocity_dirichlet_mps_field = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )
        self.pressure_neumann_normal_field = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=nodes,
        )
        self.pressure_neumann_gradient_field = ti.field(dtype=ti.f32, shape=nodes)
        self.marker_pressure_neumann_gradient_field = ti.field(
            dtype=ti.f32,
            shape=self.marker_capacity,
        )
        self.marker_pressure_neumann_row_count = ti.field(
            dtype=ti.i32,
            shape=self.marker_capacity,
        )

        self.report_no_slip_dirichlet_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_neumann_count = ti.field(dtype=ti.i32, shape=())
        self.report_inactive_internal_node_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_neumann_matrix_rows = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_neumann_rhs_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_neumann_max_abs_rhs = ti.field(dtype=ti.f32, shape=())
        self.report_pressure_neumann_invalid_reconstruction_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_min_reconstruction_gap_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_reconstruction_gap_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_transmissibility_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_raw_transmissibility_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_max_transmissibility_limit_m = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_transmissibility_capped_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_max_diagonal_per_m2 = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_pressure_neumann_skipped_velocity_dirichlet_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_active_marker_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_max_rows_per_marker = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_node_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_gradient_max_abs = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_obstacle_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_max_abs_velocity = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_reconstruction_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_narrow_gap_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_relocated_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_relocation_merged_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_neumann_skipped_obstacle_owner_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_min_projection_weight = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_max_projection_weight = ti.field(
            dtype=ti.f32,
            shape=(),
        )

    @ti.kernel
    def _build_from_search_kernel(
        self,
        node_kind_code: ti.template(),
        nearest_marker: ti.template(),
        projection_marker_indices: ti.template(),
        projection_marker_weights: ti.template(),
        marker_velocities_mps: ti.template(),
        marker_normals: ti.template(),
        marker_pressure_neumann_gradient_pa_per_m: ti.template(),
        marker_count: ti.i32,
    ):
        self.report_no_slip_dirichlet_count[None] = 0
        self.report_pressure_neumann_count[None] = 0
        self.report_inactive_internal_node_count[None] = 0
        for node in ti.grouped(self.active_ib_node):
            self.active_ib_node[node] = 0
            self.velocity_dirichlet_mps_field[node] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure_neumann_normal_field[node] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure_neumann_gradient_field[node] = 0.0

            if node_kind_code[node] == self._NODE_EXTERNAL_IB:
                marker = nearest_marker[node]
                if 0 <= marker < marker_count:
                    target_velocity = ti.Vector([0.0, 0.0, 0.0])
                    target_normal = ti.Vector([0.0, 0.0, 0.0])
                    target_gradient = 0.0
                    total_weight = 0.0
                    indices = projection_marker_indices[node]
                    weights = projection_marker_weights[node]
                    for slot in ti.static(range(3)):
                        projection_marker = indices[slot]
                        weight = weights[slot]
                        if (
                            0 <= projection_marker < marker_count
                            and weight > 0.0
                        ):
                            target_velocity += (
                                weight * marker_velocities_mps[projection_marker]
                            )
                            target_normal += weight * marker_normals[projection_marker]
                            target_gradient += (
                                weight
                                * marker_pressure_neumann_gradient_pa_per_m[
                                    projection_marker
                                ]
                            )
                            total_weight += weight
                    if total_weight > 1.0e-12:
                        inv_weight = 1.0 / total_weight
                        target_velocity *= inv_weight
                        target_normal *= inv_weight
                        target_gradient *= inv_weight
                        normal_norm = target_normal.norm()
                        if normal_norm > 1.0e-12:
                            target_normal = target_normal / normal_norm
                        else:
                            target_normal = marker_normals[marker]
                    else:
                        target_velocity = marker_velocities_mps[marker]
                        target_normal = marker_normals[marker]
                        target_gradient = marker_pressure_neumann_gradient_pa_per_m[
                            marker
                        ]
                    self.active_ib_node[node] = 1
                    self.velocity_dirichlet_mps_field[node] = target_velocity
                    self.pressure_neumann_normal_field[node] = target_normal
                    self.pressure_neumann_gradient_field[node] = target_gradient
                    self.report_no_slip_dirichlet_count[None] += 1
                    self.report_pressure_neumann_count[None] += 1
            elif node_kind_code[node] == self._NODE_INTERNAL:
                self.report_inactive_internal_node_count[None] += 1

    def build_from_search(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        marker_pressure_neumann_gradient_pa_per_m: Sequence[float],
    ) -> HibmMpmIbBoundaryConditionReport:
        self._validate_search_and_markers(search, markers)
        if len(marker_pressure_neumann_gradient_pa_per_m) != markers.marker_count:
            raise ValueError(
                "marker_pressure_neumann_gradient_pa_per_m must match marker_count"
            )
        for marker, gradient in enumerate(marker_pressure_neumann_gradient_pa_per_m):
            value = float(gradient)
            if not math.isfinite(value):
                raise ValueError(
                    "marker_pressure_neumann_gradient_pa_per_m must be finite"
                )
            self.marker_pressure_neumann_gradient_field[marker] = value

        return self.build_from_search_device_fields(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_field
            ),
        )

    def build_from_search_device_fields(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        marker_pressure_neumann_gradient_pa_per_m_field,
    ) -> HibmMpmIbBoundaryConditionReport:
        self._validate_search_and_markers(search, markers)
        self._build_from_search_kernel(
            search.node_kind_code,
            search.nearest_marker,
            search.node_projection_marker_indices,
            search.node_projection_marker_weights,
            markers.v_gamma_mps,
            markers.n_gamma,
            marker_pressure_neumann_gradient_pa_per_m_field,
            int(markers.marker_count),
        )
        return HibmMpmIbBoundaryConditionReport(
            no_slip_dirichlet_count=int(self.report_no_slip_dirichlet_count[None]),
            pressure_neumann_count=int(self.report_pressure_neumann_count[None]),
            inactive_internal_node_count=int(
                self.report_inactive_internal_node_count[None]
            ),
        )

    def _validate_search_and_markers(
        self,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
    ) -> None:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        if int(markers.marker_count) > self.marker_capacity:
            raise ValueError("markers.marker_count exceeds marker_capacity")
        if int(search.marker_capacity) > self.marker_capacity:
            raise ValueError("search.marker_capacity exceeds marker_capacity")

    @ti.kernel
    def _assemble_velocity_dirichlet_boundary_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        obstacle_field: ti.template(),
    ):
        self.report_velocity_dirichlet_boundary_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        self.report_velocity_dirichlet_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_invalid_reconstruction_rows[None] = 0
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None] = 0
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None] = 0
        self.report_velocity_dirichlet_min_projection_weight[None] = 1.0e30
        self.report_velocity_dirichlet_max_projection_weight[None] = 0.0
        for node in ti.grouped(velocity_dirichlet_active):
            velocity_dirichlet_active[node] = 0
            velocity_dirichlet_value_mps[node] = ti.Vector([0.0, 0.0, 0.0])
            if self.active_ib_node[node] == 1:
                if obstacle_field[node] == 0:
                    target_velocity = self.velocity_dirichlet_mps_field[node]
                    velocity_dirichlet_active[node] = 1
                    velocity_dirichlet_value_mps[node] = target_velocity
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_rows[None],
                        1,
                    )
                    ti.atomic_max(
                        self.report_velocity_dirichlet_max_abs_velocity[None],
                        ti.max(
                            ti.max(ti.abs(target_velocity.x), ti.abs(target_velocity.y)),
                            ti.abs(target_velocity.z),
                        ),
                    )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    def assemble_velocity_dirichlet_boundary_rows(
        self,
        velocity_dirichlet_active,
        velocity_dirichlet_value_mps,
        obstacle_field,
    ) -> HibmMpmVelocityDirichletBoundaryReport:
        self._assemble_velocity_dirichlet_boundary_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            obstacle_field,
        )
        return HibmMpmVelocityDirichletBoundaryReport(
            active_velocity_dirichlet_rows=int(
                self.report_velocity_dirichlet_boundary_rows[None]
            ),
            inactive_obstacle_rows=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_max_abs_velocity[None]
            ),
        )

    @ti.func
    def _axis_grid_coordinate_device(
        self,
        value: ti.f32,
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
            coordinate = ti.cast(count - 1, ti.f32) + 0.5 * (
                value - centers[count - 1]
            ) / half_width
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
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        return ti.Vector(
            [
                self._axis_grid_coordinate_device(
                    position.x,
                    cell_face_x_m,
                    cell_center_x_m,
                    nx,
                ),
                self._axis_grid_coordinate_device(
                    position.y,
                    cell_face_y_m,
                    cell_center_y_m,
                    ny,
                ),
                self._axis_grid_coordinate_device(
                    position.z,
                    cell_face_z_m,
                    cell_center_z_m,
                    nz,
                ),
            ]
        )

    @ti.func
    def _sample_fluid_velocity_trilinear(
        self,
        velocity_field,
        obstacle_field,
        gx,
        gy,
        gz,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
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
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
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
    def _walk_interior_velocity_sample(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        boundary_point,
        walk_normal,
        base_distance,
        step_m,
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        sample_velocity = ti.Vector([0.0, 0.0, 0.0])
        sample_weight = 0.0
        sample_distance = base_distance
        found = 0
        for step_index in ti.static(range(5)):
            if found == 0:
                candidate_distance = base_distance + step_m * ti.cast(
                    step_index, ti.f32
                )
                candidate_point = boundary_point + walk_normal * candidate_distance
                candidate_coordinate = self._grid_coordinate_from_fields(
                    candidate_point,
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
                velocity_value, weight = self._sample_fluid_velocity_trilinear(
                    velocity_field,
                    obstacle_field,
                    candidate_coordinate.x,
                    candidate_coordinate.y,
                    candidate_coordinate.z,
                    nx,
                    ny,
                    nz,
                )
                if weight > 1.0e-12:
                    sample_velocity = velocity_value
                    sample_weight = weight
                    sample_distance = candidate_distance
                    found = 1
        return sample_velocity, sample_weight, sample_distance

    @ti.kernel
    def _assemble_velocity_dirichlet_reconstructed_boundary_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        velocity_dirichlet_projection_weight: ti.template(),
        obstacle_field: ti.template(),
        velocity_field: ti.template(),
        node_boundary_point_m: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_velocity_dirichlet_boundary_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        self.report_velocity_dirichlet_max_abs_velocity[None] = 0.0
        self.report_velocity_dirichlet_invalid_reconstruction_rows[None] = 0
        self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None] = 0
        self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None] = 0
        self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None] = 0
        self.report_velocity_dirichlet_narrow_gap_rows[None] = 0
        self.report_velocity_dirichlet_min_projection_weight[None] = 1.0e30
        self.report_velocity_dirichlet_max_projection_weight[None] = 0.0
        for node in ti.grouped(velocity_dirichlet_active):
            velocity_dirichlet_active[node] = 0
            velocity_dirichlet_value_mps[node] = ti.Vector([0.0, 0.0, 0.0])
            velocity_dirichlet_projection_weight[node] = 0.0
            if self.active_ib_node[node] == 1:
                if obstacle_field[node] == 0:
                    boundary_velocity = self.velocity_dirichlet_mps_field[node]
                    normal = self.pressure_neumann_normal_field[node]
                    boundary_point = node_boundary_point_m[node]
                    interior_point = node_interior_fluid_point_m[node]
                    node_position = ti.Vector(
                        [
                            cell_center_x_m[node[0]],
                            cell_center_y_m[node[1]],
                            cell_center_z_m[node[2]],
                        ]
                    )
                    normal_denominator = (interior_point - boundary_point).dot(normal)
                    normal_distance = (node_position - boundary_point).dot(normal)
                    node_width_x = cell_face_x_m[node[0] + 1] - cell_face_x_m[node[0]]
                    node_width_y = cell_face_y_m[node[1] + 1] - cell_face_y_m[node[1]]
                    node_width_z = cell_face_z_m[node[2] + 1] - cell_face_z_m[node[2]]
                    walk_step_m = 0.5 / ti.max(
                        ti.abs(normal.x) / ti.max(node_width_x, 1.0e-12)
                        + ti.abs(normal.y) / ti.max(node_width_y, 1.0e-12)
                        + ti.abs(normal.z) / ti.max(node_width_z, 1.0e-12),
                        1.0e-12,
                    )
                    interior_velocity, fluid_weight, sample_denominator = (
                        self._walk_interior_velocity_sample(
                            velocity_field,
                            obstacle_field,
                            boundary_point,
                            normal,
                            normal_denominator,
                            walk_step_m,
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
                    )
                    target_velocity = boundary_velocity
                    reconstruction_alpha = 0.0
                    if (
                        fluid_weight > 1.0e-12
                        and sample_denominator > 1.0e-12
                        and normal_distance >= 0.0
                        and normal_distance <= sample_denominator
                    ):
                        alpha = ti.min(
                            ti.max(normal_distance / sample_denominator, 0.0),
                            1.0,
                        )
                        reconstruction_alpha = alpha
                        target_velocity = (
                            boundary_velocity
                            + (interior_velocity - boundary_velocity) * alpha
                        )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = target_velocity
                        velocity_dirichlet_projection_weight[node] = (
                            reconstruction_alpha
                        )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(target_velocity.x),
                                    ti.abs(target_velocity.y),
                                ),
                                ti.abs(target_velocity.z),
                            ),
                        )
                    elif (
                        fluid_weight <= 1.0e-12
                        and normal_denominator > 1.0e-12
                        and normal_distance >= 0.0
                    ):
                        ti.atomic_add(
                            self.report_velocity_dirichlet_narrow_gap_rows[None],
                            1,
                        )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = boundary_velocity
                        velocity_dirichlet_projection_weight[node] = 0.0
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            0.0,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(boundary_velocity.x),
                                    ti.abs(boundary_velocity.y),
                                ),
                                ti.abs(boundary_velocity.z),
                            ),
                        )
                    else:
                        ti.atomic_add(
                            self.report_velocity_dirichlet_invalid_reconstruction_rows[
                                None
                            ],
                            1,
                        )
                        if fluid_weight <= 1.0e-12:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_denominator <= 1.0e-12:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_distance < 0.0:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[
                                    None
                                ],
                                1,
                            )
                        elif normal_distance > normal_denominator:
                            ti.atomic_add(
                                self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[
                                    None
                                ],
                                1,
                            )
                        velocity_dirichlet_active[node] = 1
                        velocity_dirichlet_value_mps[node] = boundary_velocity
                        velocity_dirichlet_projection_weight[node] = 0.0
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            0.0,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(boundary_velocity.x),
                                    ti.abs(boundary_velocity.y),
                                ),
                                ti.abs(boundary_velocity.z),
                            ),
                        )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    @ti.kernel
    def _relocate_masked_velocity_dirichlet_rows_kernel(
        self,
        velocity_dirichlet_active: ti.template(),
        velocity_dirichlet_value_mps: ti.template(),
        velocity_dirichlet_projection_weight: ti.template(),
        obstacle_field: ti.template(),
        velocity_field: ti.template(),
        node_boundary_point_m: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_velocity_dirichlet_relocated_rows[None] = 0
        self.report_velocity_dirichlet_relocation_merged_rows[None] = 0
        self.report_velocity_dirichlet_obstacle_rows[None] = 0
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] == 1 and obstacle_field[node] != 0:
                normal = self.pressure_neumann_normal_field[node]
                boundary_point = node_boundary_point_m[node]
                boundary_velocity = self.velocity_dirichlet_mps_field[node]
                node_position = ti.Vector(
                    [
                        cell_center_x_m[node[0]],
                        cell_center_y_m[node[1]],
                        cell_center_z_m[node[2]],
                    ]
                )
                side = (node_position - boundary_point).dot(normal)
                walk_normal = normal
                if side < 0.0:
                    walk_normal = -normal
                node_width_x = cell_face_x_m[node[0] + 1] - cell_face_x_m[node[0]]
                node_width_y = cell_face_y_m[node[1] + 1] - cell_face_y_m[node[1]]
                node_width_z = cell_face_z_m[node[2] + 1] - cell_face_z_m[node[2]]
                walk_step_m = 0.5 / ti.max(
                    ti.abs(walk_normal.x) / ti.max(node_width_x, 1.0e-12)
                    + ti.abs(walk_normal.y) / ti.max(node_width_y, 1.0e-12)
                    + ti.abs(walk_normal.z) / ti.max(node_width_z, 1.0e-12),
                    1.0e-12,
                )
                node_distance = ti.abs(side)
                target_i = -1
                target_j = -1
                target_k = -1
                target_distance = 0.0
                for step_index in ti.static(range(8)):
                    if target_i < 0:
                        candidate_distance = node_distance + walk_step_m * ti.cast(
                            step_index + 1, ti.f32
                        )
                        candidate_point = (
                            boundary_point + walk_normal * candidate_distance
                        )
                        candidate_coordinate = self._grid_coordinate_from_fields(
                            candidate_point,
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
                        candidate_i = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.x + 0.5, ti.i32),
                                0,
                            ),
                            nx - 1,
                        )
                        candidate_j = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.y + 0.5, ti.i32),
                                0,
                            ),
                            ny - 1,
                        )
                        candidate_k = ti.min(
                            ti.max(
                                ti.floor(candidate_coordinate.z + 0.5, ti.i32),
                                0,
                            ),
                            nz - 1,
                        )
                        if obstacle_field[candidate_i, candidate_j, candidate_k] == 0:
                            candidate_center = ti.Vector(
                                [
                                    cell_center_x_m[candidate_i],
                                    cell_center_y_m[candidate_j],
                                    cell_center_z_m[candidate_k],
                                ]
                            )
                            candidate_center_distance = (
                                candidate_center - boundary_point
                            ).dot(walk_normal)
                            if candidate_center_distance > 1.0e-12:
                                target_i = candidate_i
                                target_j = candidate_j
                                target_k = candidate_k
                                target_distance = candidate_center_distance
                if target_i >= 0:
                    previous = ti.atomic_or(
                        velocity_dirichlet_active[target_i, target_j, target_k],
                        1,
                    )
                    if previous == 0:
                        sample_velocity, sample_weight, sample_distance = (
                            self._walk_interior_velocity_sample(
                                velocity_field,
                                obstacle_field,
                                boundary_point,
                                walk_normal,
                                target_distance + 2.0 * walk_step_m,
                                walk_step_m,
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
                        )
                        target_velocity = boundary_velocity
                        reconstruction_alpha = 0.0
                        if (
                            sample_weight > 1.0e-12
                            and sample_distance > target_distance
                        ):
                            reconstruction_alpha = ti.min(
                                ti.max(target_distance / sample_distance, 0.0),
                                1.0,
                            )
                            target_velocity = (
                                boundary_velocity
                                + (sample_velocity - boundary_velocity)
                                * reconstruction_alpha
                            )
                        velocity_dirichlet_value_mps[
                            target_i, target_j, target_k
                        ] = target_velocity
                        velocity_dirichlet_projection_weight[
                            target_i, target_j, target_k
                        ] = reconstruction_alpha
                        ti.atomic_add(
                            self.report_velocity_dirichlet_relocated_rows[None],
                            1,
                        )
                        ti.atomic_add(
                            self.report_velocity_dirichlet_boundary_rows[None],
                            1,
                        )
                        ti.atomic_min(
                            self.report_velocity_dirichlet_min_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_projection_weight[None],
                            reconstruction_alpha,
                        )
                        ti.atomic_max(
                            self.report_velocity_dirichlet_max_abs_velocity[None],
                            ti.max(
                                ti.max(
                                    ti.abs(target_velocity.x),
                                    ti.abs(target_velocity.y),
                                ),
                                ti.abs(target_velocity.z),
                            ),
                        )
                    else:
                        ti.atomic_add(
                            self.report_velocity_dirichlet_relocation_merged_rows[
                                None
                            ],
                            1,
                        )
                else:
                    ti.atomic_add(
                        self.report_velocity_dirichlet_obstacle_rows[None],
                        1,
                    )

    def assemble_velocity_dirichlet_reconstructed_boundary_rows(
        self,
        velocity_dirichlet_active,
        velocity_dirichlet_value_mps,
        velocity_dirichlet_projection_weight,
        obstacle_field,
        velocity_field,
        search: HibmMpmIbNodeSearch,
        *,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
    ) -> HibmMpmVelocityDirichletBoundaryReport:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._assemble_velocity_dirichlet_reconstructed_boundary_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            velocity_dirichlet_projection_weight,
            obstacle_field,
            velocity_field,
            search.node_boundary_point_m,
            search.node_interior_fluid_point_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        self._relocate_masked_velocity_dirichlet_rows_kernel(
            velocity_dirichlet_active,
            velocity_dirichlet_value_mps,
            velocity_dirichlet_projection_weight,
            obstacle_field,
            velocity_field,
            search.node_boundary_point_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        active_rows = int(self.report_velocity_dirichlet_boundary_rows[None])
        min_projection_weight = 0.0
        if active_rows > 0:
            min_projection_weight = float(
                self.report_velocity_dirichlet_min_projection_weight[None]
            )
        return HibmMpmVelocityDirichletBoundaryReport(
            active_velocity_dirichlet_rows=active_rows,
            inactive_obstacle_rows=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            max_abs_velocity_mps=float(
                self.report_velocity_dirichlet_max_abs_velocity[None]
            ),
            invalid_reconstruction_row_count=int(
                self.report_velocity_dirichlet_invalid_reconstruction_rows[None]
            ),
            invalid_no_fluid_sample_row_count=int(
                self.report_velocity_dirichlet_invalid_no_fluid_sample_rows[None]
            ),
            invalid_nonpositive_gap_row_count=int(
                self.report_velocity_dirichlet_invalid_nonpositive_gap_rows[None]
            ),
            invalid_node_behind_boundary_row_count=int(
                self.report_velocity_dirichlet_invalid_node_behind_boundary_rows[None]
            ),
            invalid_node_beyond_interior_row_count=int(
                self.report_velocity_dirichlet_invalid_node_beyond_interior_rows[None]
            ),
            narrow_gap_boundary_velocity_row_count=int(
                self.report_velocity_dirichlet_narrow_gap_rows[None]
            ),
            relocated_row_count=int(
                self.report_velocity_dirichlet_relocated_rows[None]
            ),
            relocation_merged_row_count=int(
                self.report_velocity_dirichlet_relocation_merged_rows[None]
            ),
            relocation_blocked_row_count=int(
                self.report_velocity_dirichlet_obstacle_rows[None]
            ),
            min_projection_weight=min_projection_weight,
            max_projection_weight=float(
                self.report_velocity_dirichlet_max_projection_weight[None]
            ),
        )

    @ti.kernel
    def _clear_pressure_neumann_rows_by_marker_kernel(
        self,
        marker_count: ti.i32,
    ):
        for marker in range(marker_count):
            self.marker_pressure_neumann_row_count[marker] = 0

    @ti.kernel
    def _summarize_pressure_neumann_rows_by_marker_kernel(
        self,
        marker_count: ti.i32,
    ):
        self.report_pressure_neumann_active_marker_count[None] = 0
        self.report_pressure_neumann_max_rows_per_marker[None] = 0
        for marker in range(marker_count):
            row_count = self.marker_pressure_neumann_row_count[marker]
            if row_count > 0:
                ti.atomic_add(
                    self.report_pressure_neumann_active_marker_count[None],
                    1,
                )
                ti.atomic_max(
                    self.report_pressure_neumann_max_rows_per_marker[None],
                    row_count,
                )

    @ti.kernel
    def _assemble_pressure_neumann_matrix_rows_kernel(
        self,
        pressure_matrix_diagonal: ti.template(),
        pressure_matrix_rhs: ti.template(),
        pressure_coupling_active: ti.template(),
        pressure_coupling_neighbor: ti.template(),
        pressure_coupling_coefficient: ti.template(),
        obstacle_field: ti.template(),
        velocity_dirichlet_active: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        node_boundary_point_m: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        nearest_marker: ti.template(),
        marker_pressure_anchor_cell: ti.template(),
        marker_count: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_matrix_rows[None] = 0
        self.report_pressure_neumann_rhs_integral[None] = ti.cast(0.0, ti.f64)
        self.report_pressure_neumann_max_abs_rhs[None] = 0.0
        self.report_pressure_neumann_invalid_reconstruction_rows[None] = 0
        self.report_pressure_neumann_min_reconstruction_gap_m[None] = 1.0e30
        self.report_pressure_neumann_max_reconstruction_gap_m[None] = 0.0
        self.report_pressure_neumann_max_transmissibility_m[None] = 0.0
        self.report_pressure_neumann_max_raw_transmissibility_m[None] = 0.0
        self.report_pressure_neumann_max_transmissibility_limit_m[None] = 0.0
        self.report_pressure_neumann_transmissibility_capped_rows[None] = 0
        self.report_pressure_neumann_max_diagonal_per_m2[None] = 0.0
        self.report_pressure_neumann_skipped_velocity_dirichlet_rows[None] = 0
        self.report_pressure_neumann_skipped_obstacle_owner_rows[None] = 0
        for node in ti.grouped(self.active_ib_node):
            if self.active_ib_node[node] == 1 and obstacle_field[node] != 0:
                ti.atomic_add(
                    self.report_pressure_neumann_skipped_obstacle_owner_rows[None],
                    1,
                )
            if (
                self.active_ib_node[node] == 1
                and obstacle_field[node] == 0
                and velocity_dirichlet_active[node] != 0
            ):
                ti.atomic_add(
                    self.report_pressure_neumann_skipped_velocity_dirichlet_rows[None],
                    1,
                )
            if (
                self.active_ib_node[node] == 1
                and obstacle_field[node] == 0
                and velocity_dirichlet_active[node] == 0
            ):
                marker = nearest_marker[node]
                if 0 <= marker < marker_count:
                    volume_m3 = (
                        cell_width_x_m[node[0]]
                        * cell_width_y_m[node[1]]
                        * cell_width_z_m[node[2]]
                    )
                    if volume_m3 > 0.0:
                        normal = self.pressure_neumann_normal_field[node]
                        boundary_point = node_boundary_point_m[node]
                        interior_point = node_interior_fluid_point_m[node]
                        node_position = ti.Vector(
                            [
                                cell_center_x_m[node[0]],
                                cell_center_y_m[node[1]],
                                cell_center_z_m[node[2]],
                            ]
                        )
                        grid_coordinate = self._grid_coordinate_from_fields(
                            interior_point,
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
                        neighbor_i = ti.min(
                            ti.max(ti.floor(grid_coordinate.x + 0.5, ti.i32), 0),
                            nx - 1,
                        )
                        neighbor_j = ti.min(
                            ti.max(ti.floor(grid_coordinate.y + 0.5, ti.i32), 0),
                            ny - 1,
                        )
                        neighbor_k = ti.min(
                            ti.max(ti.floor(grid_coordinate.z + 0.5, ti.i32), 0),
                            nz - 1,
                        )
                        neighbor_position = ti.Vector(
                            [
                                cell_center_x_m[neighbor_i],
                                cell_center_y_m[neighbor_j],
                                cell_center_z_m[neighbor_k],
                            ]
                        )
                        node_distance = (node_position - boundary_point).dot(normal)
                        neighbor_distance = (
                            neighbor_position - boundary_point
                        ).dot(normal)
                        normal_denominator = (interior_point - boundary_point).dot(
                            normal
                        )
                        reconstruction_gap = ti.abs(neighbor_distance - node_distance)
                        node_spacing_inv = (
                            ti.abs(normal.x) / cell_width_x_m[node[0]]
                            + ti.abs(normal.y) / cell_width_y_m[node[1]]
                            + ti.abs(normal.z) / cell_width_z_m[node[2]]
                        )
                        neighbor_spacing_inv = (
                            ti.abs(normal.x) / cell_width_x_m[neighbor_i]
                            + ti.abs(normal.y) / cell_width_y_m[neighbor_j]
                            + ti.abs(normal.z) / cell_width_z_m[neighbor_k]
                        )
                        node_normal_width = 1.0 / ti.max(
                            node_spacing_inv,
                            1.0e-12,
                        )
                        neighbor_normal_width = 1.0 / ti.max(
                            neighbor_spacing_inv,
                            1.0e-12,
                        )
                        min_normal_width = ti.min(
                            node_normal_width,
                            neighbor_normal_width,
                        )
                        reconstruction_gap_floor = ti.max(
                            1.0e-12,
                            1.0e-3 * min_normal_width,
                        )
                        row_reconstructable = 0
                        if (
                            reconstruction_gap > reconstruction_gap_floor
                            and normal_denominator > 1.0e-12
                            and node_distance >= 0.0
                            and node_distance <= normal_denominator
                            and obstacle_field[neighbor_i, neighbor_j, neighbor_k] == 0
                        ):
                            row_reconstructable = 1
                        else:
                            abs_x = ti.abs(normal.x)
                            abs_y = ti.abs(normal.y)
                            abs_z = ti.abs(normal.z)
                            fallback_i = node[0]
                            fallback_j = node[1]
                            fallback_k = node[2]
                            if abs_x >= abs_y and abs_x >= abs_z:
                                step = 1
                                if normal.x < 0.0:
                                    step = -1
                                fallback_i = node[0] + step
                                if fallback_i < 0 or fallback_i >= nx:
                                    fallback_i = node[0] - step
                                fallback_i = ti.min(ti.max(fallback_i, 0), nx - 1)
                                if obstacle_field[fallback_i, fallback_j, fallback_k] != 0:
                                    alternate_i = ti.min(ti.max(node[0] - step, 0), nx - 1)
                                    if obstacle_field[alternate_i, fallback_j, fallback_k] == 0:
                                        fallback_i = alternate_i
                            elif abs_y >= abs_x and abs_y >= abs_z:
                                step = 1
                                if normal.y < 0.0:
                                    step = -1
                                fallback_j = node[1] + step
                                if fallback_j < 0 or fallback_j >= ny:
                                    fallback_j = node[1] - step
                                fallback_j = ti.min(ti.max(fallback_j, 0), ny - 1)
                                if obstacle_field[fallback_i, fallback_j, fallback_k] != 0:
                                    alternate_j = ti.min(ti.max(node[1] - step, 0), ny - 1)
                                    if obstacle_field[fallback_i, alternate_j, fallback_k] == 0:
                                        fallback_j = alternate_j
                            else:
                                step = 1
                                if normal.z < 0.0:
                                    step = -1
                                fallback_k = node[2] + step
                                if fallback_k < 0 or fallback_k >= nz:
                                    fallback_k = node[2] - step
                                fallback_k = ti.min(ti.max(fallback_k, 0), nz - 1)
                                if obstacle_field[fallback_i, fallback_j, fallback_k] != 0:
                                    alternate_k = ti.min(ti.max(node[2] - step, 0), nz - 1)
                                    if obstacle_field[fallback_i, fallback_j, alternate_k] == 0:
                                        fallback_k = alternate_k
                            neighbor_i = fallback_i
                            neighbor_j = fallback_j
                            neighbor_k = fallback_k
                            neighbor_position = ti.Vector(
                                [
                                    cell_center_x_m[neighbor_i],
                                    cell_center_y_m[neighbor_j],
                                    cell_center_z_m[neighbor_k],
                                ]
                            )
                            neighbor_distance = (
                                neighbor_position - boundary_point
                            ).dot(normal)
                            reconstruction_gap = ti.abs(
                                neighbor_distance - node_distance
                            )
                            neighbor_spacing_inv = (
                                ti.abs(normal.x) / cell_width_x_m[neighbor_i]
                                + ti.abs(normal.y) / cell_width_y_m[neighbor_j]
                                + ti.abs(normal.z) / cell_width_z_m[neighbor_k]
                            )
                            neighbor_normal_width = 1.0 / ti.max(
                                neighbor_spacing_inv,
                                1.0e-12,
                            )
                            min_normal_width = ti.min(
                                node_normal_width,
                                neighbor_normal_width,
                            )
                            reconstruction_gap_floor = ti.max(
                                1.0e-12,
                                1.0e-3 * min_normal_width,
                            )
                            if (
                                reconstruction_gap > reconstruction_gap_floor
                                and obstacle_field[neighbor_i, neighbor_j, neighbor_k] == 0
                                and (
                                    neighbor_i != node[0]
                                    or neighbor_j != node[1]
                                    or neighbor_k != node[2]
                                )
                            ):
                                row_reconstructable = 1
                        if row_reconstructable != 0:
                            neighbor_volume_m3 = (
                                cell_width_x_m[neighbor_i]
                                * cell_width_y_m[neighbor_j]
                                * cell_width_z_m[neighbor_k]
                            )
                            interface_area_m2 = 0.5 * (
                                volume_m3 / ti.max(node_normal_width, 1.0e-12)
                                + neighbor_volume_m3
                                / ti.max(neighbor_normal_width, 1.0e-12)
                            )
                            raw_transmissibility = (
                                interface_area_m2 / reconstruction_gap
                            )
                            transmissibility_limit = (
                                20.0
                                * interface_area_m2
                                / ti.max(min_normal_width, 1.0e-12)
                            )
                            transmissibility = ti.min(
                                raw_transmissibility,
                                transmissibility_limit,
                            )
                            if raw_transmissibility > transmissibility_limit:
                                ti.atomic_add(
                                    self.report_pressure_neumann_transmissibility_capped_rows[
                                        None
                                    ],
                                    1,
                                )
                            node_coefficient = transmissibility / volume_m3
                            neighbor_coefficient = (
                                transmissibility / neighbor_volume_m3
                            )
                            pressure_jump = (
                                self.pressure_neumann_gradient_field[node]
                                * (node_distance - neighbor_distance)
                            )
                            node_rhs_density = node_coefficient * pressure_jump
                            neighbor_rhs_density = (
                                -neighbor_coefficient * pressure_jump
                            )
                            ti.atomic_add(
                                pressure_matrix_diagonal[node],
                                node_coefficient,
                            )
                            ti.atomic_add(
                                pressure_matrix_diagonal[
                                    neighbor_i,
                                    neighbor_j,
                                    neighbor_k,
                                ],
                                neighbor_coefficient,
                            )
                            ti.atomic_add(pressure_matrix_rhs[node], node_rhs_density)
                            ti.atomic_add(
                                pressure_matrix_rhs[
                                    neighbor_i,
                                    neighbor_j,
                                    neighbor_k,
                                ],
                                neighbor_rhs_density,
                            )
                            pressure_coupling_active[node] = 1
                            pressure_coupling_neighbor[node] = ti.Vector(
                                [neighbor_i, neighbor_j, neighbor_k]
                            )
                            pressure_coupling_coefficient[node] = transmissibility
                            ti.atomic_add(
                                self.report_pressure_neumann_matrix_rows[None],
                                1,
                            )
                            previous_marker_row_count = ti.atomic_add(
                                self.marker_pressure_neumann_row_count[marker],
                                1,
                            )
                            if previous_marker_row_count == 0:
                                # S2-A6: anchor the marker to its first
                                # row-owning fluid cell. All acceptance
                                # paths (direct interior-point walk, the
                                # bounds-relocated axial fallback and the
                                # obstacle-alternate axial fallback)
                                # converge on this row-write block, so
                                # every row writer is covered; the 0 -> 1
                                # transition of the per-marker row counter
                                # elects exactly one writer per marker,
                                # keeping the 3-component store tear-free
                                # without a second guard field. The node is
                                # obstacle-free and receives diagonal/rhs
                                # terms above, i.e. it participates in the
                                # pressure solve by construction.
                                marker_pressure_anchor_cell[marker] = ti.Vector(
                                    [node[0], node[1], node[2]]
                                )
                            self.report_pressure_neumann_rhs_integral[None] += (
                                ti.cast(node_rhs_density * volume_m3, ti.f64)
                                + ti.cast(
                                    neighbor_rhs_density * neighbor_volume_m3,
                                    ti.f64,
                                )
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_abs_rhs[None],
                                ti.max(
                                    ti.abs(node_rhs_density),
                                    ti.abs(neighbor_rhs_density),
                                ),
                            )
                            ti.atomic_min(
                                self.report_pressure_neumann_min_reconstruction_gap_m[
                                    None
                                ],
                                reconstruction_gap,
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_reconstruction_gap_m[
                                    None
                                ],
                                reconstruction_gap,
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_transmissibility_m[
                                    None
                                ],
                                transmissibility,
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_raw_transmissibility_m[
                                    None
                                ],
                                raw_transmissibility,
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_transmissibility_limit_m[
                                    None
                                ],
                                transmissibility_limit,
                            )
                            ti.atomic_max(
                                self.report_pressure_neumann_max_diagonal_per_m2[None],
                                ti.max(node_coefficient, neighbor_coefficient),
                            )
                        else:
                            ti.atomic_add(
                                self.report_pressure_neumann_invalid_reconstruction_rows[
                                    None
                                ],
                                1,
                            )
                    else:
                        ti.atomic_add(
                            self.report_pressure_neumann_invalid_reconstruction_rows[
                                None
                            ],
                            1,
                        )
                else:
                    ti.atomic_add(
                        self.report_pressure_neumann_invalid_reconstruction_rows[None],
                        1,
                    )

    def assemble_pressure_neumann_matrix_rows(
        self,
        pressure_matrix_diagonal,
        pressure_matrix_rhs,
        pressure_coupling_active,
        pressure_coupling_neighbor,
        pressure_coupling_coefficient,
        obstacle_field,
        velocity_dirichlet_active,
        cell_width_x_m,
        cell_width_y_m,
        cell_width_z_m,
        search: HibmMpmIbNodeSearch,
        markers: HibmMpmSurfaceMarkers,
        *,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
    ) -> HibmMpmPressureNeumannMatrixReport:
        self._validate_search_and_markers(search, markers)
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._clear_pressure_neumann_rows_by_marker_kernel(
            int(markers.marker_count),
        )
        # S2-A6: full-capacity sentinel reset before assembly so markers
        # that fail to produce a row (invalid reconstruction paths) keep
        # (-1, -1, -1) and stay on the sampler's invalid path.
        markers.reset_pressure_anchor_cells()
        self._assemble_pressure_neumann_matrix_rows_kernel(
            pressure_matrix_diagonal,
            pressure_matrix_rhs,
            pressure_coupling_active,
            pressure_coupling_neighbor,
            pressure_coupling_coefficient,
            obstacle_field,
            velocity_dirichlet_active,
            cell_width_x_m,
            cell_width_y_m,
            cell_width_z_m,
            search.node_boundary_point_m,
            search.node_interior_fluid_point_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            search.nearest_marker,
            markers.marker_pressure_anchor_cell,
            int(markers.marker_count),
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        self._summarize_pressure_neumann_rows_by_marker_kernel(
            int(markers.marker_count),
        )
        active_rows = int(self.report_pressure_neumann_matrix_rows[None])
        min_reconstruction_gap_m = 0.0
        if active_rows > 0:
            min_reconstruction_gap_m = float(
                self.report_pressure_neumann_min_reconstruction_gap_m[None]
            )
        return HibmMpmPressureNeumannMatrixReport(
            active_pressure_neumann_rows=active_rows,
            rhs_integral=float(self.report_pressure_neumann_rhs_integral[None]),
            max_abs_rhs=float(self.report_pressure_neumann_max_abs_rhs[None]),
            skipped_velocity_dirichlet_row_count=int(
                self.report_pressure_neumann_skipped_velocity_dirichlet_rows[None]
            ),
            skipped_obstacle_owner_row_count=int(
                self.report_pressure_neumann_skipped_obstacle_owner_rows[None]
            ),
            active_pressure_neumann_marker_count=int(
                self.report_pressure_neumann_active_marker_count[None]
            ),
            max_pressure_neumann_rows_per_marker=int(
                self.report_pressure_neumann_max_rows_per_marker[None]
            ),
            invalid_reconstruction_row_count=int(
                self.report_pressure_neumann_invalid_reconstruction_rows[None]
            ),
            min_reconstruction_gap_m=min_reconstruction_gap_m,
            max_reconstruction_gap_m=float(
                self.report_pressure_neumann_max_reconstruction_gap_m[None]
            ),
            max_transmissibility_m=float(
                self.report_pressure_neumann_max_transmissibility_m[None]
            ),
            max_raw_transmissibility_m=float(
                self.report_pressure_neumann_max_raw_transmissibility_m[None]
            ),
            max_transmissibility_limit_m=float(
                self.report_pressure_neumann_max_transmissibility_limit_m[None]
            ),
            transmissibility_capped_row_count=int(
                self.report_pressure_neumann_transmissibility_capped_rows[None]
            ),
            max_diagonal_per_m2=float(
                self.report_pressure_neumann_max_diagonal_per_m2[None]
            ),
        )

    def is_active(self, node_index: tuple[int, int, int]) -> bool:
        return bool(int(self.active_ib_node[self._node_index(node_index)]))

    def velocity_dirichlet_mps(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.velocity_dirichlet_mps_field,
            node_index,
        )

    def pressure_neumann_normal(
        self,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return self._vector_field_value_tuple(
            self.pressure_neumann_normal_field,
            node_index,
        )

    def pressure_neumann_gradient_pa_per_m(
        self,
        node_index: tuple[int, int, int],
    ) -> float:
        return float(
            self.pressure_neumann_gradient_field[self._node_index(node_index)]
        )

    @ti.kernel
    def _update_pressure_neumann_gradient_from_ib_nodes_kernel(
        self,
        velocity_field: ti.template(),
        obstacle_field: ti.template(),
        node_interior_fluid_point_m: ti.template(),
        cell_face_x_m: ti.template(),
        cell_face_y_m: ti.template(),
        cell_face_z_m: ti.template(),
        cell_center_x_m: ti.template(),
        cell_center_y_m: ti.template(),
        cell_center_z_m: ti.template(),
        density_kgm3: ti.f32,
        dt_s: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        self.report_pressure_neumann_gradient_node_count[None] = 0
        self.report_pressure_neumann_gradient_max_abs[None] = 0.0
        for node in ti.grouped(self.active_ib_node):
            self.pressure_neumann_gradient_field[node] = 0.0
            if self.active_ib_node[node] == 1 and obstacle_field[node] == 0:
                interior_point = node_interior_fluid_point_m[node]
                grid_coordinate = self._grid_coordinate_from_fields(
                    interior_point,
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
                predictor_velocity, fluid_weight = self._sample_fluid_velocity_trilinear(
                    velocity_field,
                    obstacle_field,
                    grid_coordinate.x,
                    grid_coordinate.y,
                    grid_coordinate.z,
                    nx,
                    ny,
                    nz,
                )
                if fluid_weight > 1.0e-12:
                    normal = self.pressure_neumann_normal_field[node]
                    boundary_velocity = self.velocity_dirichlet_mps_field[node]
                    normal_gradient = (
                        density_kgm3
                        * (predictor_velocity - boundary_velocity).dot(normal)
                        / dt_s
                    )
                    self.pressure_neumann_gradient_field[node] = normal_gradient
                    ti.atomic_add(
                        self.report_pressure_neumann_gradient_node_count[None],
                        1,
                    )
                    ti.atomic_max(
                        self.report_pressure_neumann_gradient_max_abs[None],
                        ti.abs(normal_gradient),
                    )

    def update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
        self,
        *,
        velocity_field,
        obstacle_field,
        search: HibmMpmIbNodeSearch,
        cell_face_x_m,
        cell_face_y_m,
        cell_face_z_m,
        cell_center_x_m,
        cell_center_y_m,
        cell_center_z_m,
        grid_nodes: tuple[int, int, int],
        density_kgm3: float,
        dt_s: float,
    ) -> HibmMpmPressureNeumannGradientReport:
        if tuple(search.grid_nodes) != self.grid_nodes:
            raise ValueError("search.grid_nodes must match boundary grid_nodes")
        density = float(density_kgm3)
        dt = float(dt_s)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("density_kgm3 must be finite and positive")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        nodes = tuple(int(value) for value in grid_nodes)
        if len(nodes) != 3 or any(value < 2 for value in nodes):
            raise ValueError("grid_nodes must contain three values >= 2")
        self._update_pressure_neumann_gradient_from_ib_nodes_kernel(
            velocity_field,
            obstacle_field,
            search.node_interior_fluid_point_m,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            density,
            dt,
            int(nodes[0]),
            int(nodes[1]),
            int(nodes[2]),
        )
        return HibmMpmPressureNeumannGradientReport(
            active_marker_count=int(
                self.report_pressure_neumann_gradient_node_count[None]
            ),
            max_abs_gradient_pa_per_m=float(
                self.report_pressure_neumann_gradient_max_abs[None]
            ),
        )

    def _node_index(self, node_index: tuple[int, int, int]) -> tuple[int, int, int]:
        index = tuple(int(value) for value in node_index)
        if len(index) != 3 or any(
            value < 0 or value >= limit
            for value, limit in zip(index, self.grid_nodes, strict=True)
        ):
            raise IndexError("node_index out of range")
        return index

    def _vector_field_value_tuple(
        self,
        field,
        node_index: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        value = field[self._node_index(node_index)]
        return (float(value[0]), float(value[1]), float(value[2]))


class HibmMpmSharpCouplingState:
    """Generic sharp HIBM-MPM coupling state owned by simulation_core.

    The state bundles marker fields, IB node search fields, boundary-condition
    fields, and the per-marker pressure-Neumann gradient field so case runners
    can select the solver path without owning HIBM internals.
    """

    def __init__(
        self,
        *,
        grid_nodes: tuple[int, int, int],
        bounds_min_m: Sequence[float],
        bounds_max_m: Sequence[float],
        marker_capacity: int,
        projection_triangle_capacity: int | None = None,
        runtime: TaichiRuntimeConfig | None = None,
    ) -> None:
        self.markers = HibmMpmSurfaceMarkers(
            marker_capacity=marker_capacity,
            projection_triangle_capacity=projection_triangle_capacity,
            runtime=runtime,
        )
        self.ib_search = HibmMpmIbNodeSearch(
            grid_nodes=grid_nodes,
            bounds_min_m=bounds_min_m,
            bounds_max_m=bounds_max_m,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        self.ib_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=grid_nodes,
            marker_capacity=marker_capacity,
            runtime=runtime,
        )
        self.marker_pressure_neumann_gradient_pa_per_m = (
            self.ib_boundary.marker_pressure_neumann_gradient_field
        )

    def load_markers_from_surface_fields(
        self,
        surface_position_m,
        surface_normal,
        surface_area_m2,
        surface_region_id,
        *,
        marker_count: int,
        initial_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
        surface_velocity_mps=None,
        projection_triangle_indices=None,
        projection_triangle_count: int | None = None,
    ) -> int:
        loaded_marker_count = self.markers.load_markers_from_surface_fields(
            surface_position_m,
            surface_normal,
            surface_area_m2,
            surface_region_id,
            marker_count=int(marker_count),
            initial_velocity_mps=initial_velocity_mps,
            surface_velocity_mps=surface_velocity_mps,
        )
        if projection_triangle_indices is not None:
            if projection_triangle_count is None:
                raise ValueError(
                    "projection_triangle_count is required with projection_triangle_indices"
                )
            self.markers.load_projection_triangles_from_field(
                projection_triangle_indices,
                triangle_count=int(projection_triangle_count),
            )
        return loaded_marker_count

    def advance_mpm_step(
        self,
        *,
        fluid: Any,
        mpm_external_force_n,
        mpm_particle_position_m,
        mpm_particle_velocity_mps,
        mpm_particle_normal,
        mpm_particle_area_m2,
        mpm_particle_count: int,
        solid_step: Callable[[], Any],
        search_radius_m: float,
        interior_probe_distance_m: float,
        mpm_support_radius_m: float,
        primary_region_id: int = 0,
        secondary_region_id: int = 0,
        far_pressure_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        fluid_dt_s: float | None = None,
        fluid_substeps: int = 1,
        projection_iterations: int = 40,
        run_fluid_predictor: bool = True,
        fluid_advection_scheme: str = "euler",
        pressure_neumann_density_kgm3: float | None = None,
        pressure_neumann_dt_s: float | None = None,
        pressure_outlet_zmin: bool = False,
        reset_pressure: bool = True,
        pressure_solver: str = "fv_cg",
        pressure_solve_failure_policy: str = "raise",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        surface_feedback_dt_s: float | None = None,
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        classify_far_internal_nodes: bool = False,
        diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    ) -> HibmMpmSharpMpmStepReport:
        return advance_hibm_mpm_sharp_mpm_step(
            fluid=fluid,
            markers=self.markers,
            ib_search=self.ib_search,
            ib_boundary=self.ib_boundary,
            mpm_external_force_n=mpm_external_force_n,
            mpm_particle_position_m=mpm_particle_position_m,
            mpm_particle_velocity_mps=mpm_particle_velocity_mps,
            mpm_particle_normal=mpm_particle_normal,
            mpm_particle_area_m2=mpm_particle_area_m2,
            mpm_particle_count=int(mpm_particle_count),
            solid_step=solid_step,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_pa_per_m
            ),
            search_radius_m=float(search_radius_m),
            interior_probe_distance_m=float(interior_probe_distance_m),
            mpm_support_radius_m=float(mpm_support_radius_m),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_pa=float(far_pressure_pa),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            fluid_dt_s=fluid_dt_s,
            fluid_substeps=int(fluid_substeps),
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=bool(run_fluid_predictor),
            fluid_advection_scheme=str(fluid_advection_scheme),
            pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
            pressure_neumann_dt_s=pressure_neumann_dt_s,
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            reset_pressure=bool(reset_pressure),
            pressure_solver=str(pressure_solver),
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=float(cg_tolerance),
            cg_preconditioner=str(cg_preconditioner),
            surface_feedback_dt_s=surface_feedback_dt_s,
            divergence_cleanup_iterations=int(divergence_cleanup_iterations),
            divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
            classify_far_internal_nodes=bool(classify_far_internal_nodes),
            diagnostic_disable_pressure_neumann_matrix_rows=bool(
                diagnostic_disable_pressure_neumann_matrix_rows
            ),
        )

    def advance_neo_hookean_step(
        self,
        *,
        fluid: Any,
        solid: Any,
        search_radius_m: float,
        interior_probe_distance_m: float,
        mpm_support_radius_m: float,
        solid_dt_s: float,
        mu_pa: float,
        lambda_pa: float,
        primary_region_id: int,
        secondary_region_id: int,
        far_pressure_region_id: int = -1,
        far_pressure_pa: float = 0.0,
        far_pressure_inside_probe_max_multiplier: float = 3.0,
        fluid_dt_s: float | None = None,
        fluid_substeps: int = 1,
        projection_iterations: int = 40,
        run_fluid_predictor: bool = True,
        fluid_advection_scheme: str = "euler",
        pressure_neumann_density_kgm3: float | None = None,
        pressure_neumann_dt_s: float | None = None,
        pressure_outlet_zmin: bool = False,
        reset_pressure: bool = True,
        pressure_solver: str = "fv_cg",
        pressure_solve_failure_policy: str = "raise",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        read_mpm_report: bool = True,
        solid_external_loads: Callable[[], None] | None = None,
        classify_far_internal_nodes: bool = False,
        diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
    ) -> HibmMpmSharpNeoHookeanStepReport:
        return advance_hibm_mpm_sharp_neo_hookean_step(
            fluid=fluid,
            markers=self.markers,
            ib_search=self.ib_search,
            ib_boundary=self.ib_boundary,
            solid=solid,
            marker_pressure_neumann_gradient_pa_per_m_field=(
                self.marker_pressure_neumann_gradient_pa_per_m
            ),
            search_radius_m=float(search_radius_m),
            interior_probe_distance_m=float(interior_probe_distance_m),
            mpm_support_radius_m=float(mpm_support_radius_m),
            solid_dt_s=float(solid_dt_s),
            mu_pa=float(mu_pa),
            lambda_pa=float(lambda_pa),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            far_pressure_region_id=int(far_pressure_region_id),
            far_pressure_pa=float(far_pressure_pa),
            far_pressure_inside_probe_max_multiplier=float(
                far_pressure_inside_probe_max_multiplier
            ),
            fluid_dt_s=fluid_dt_s,
            fluid_substeps=int(fluid_substeps),
            projection_iterations=int(projection_iterations),
            run_fluid_predictor=bool(run_fluid_predictor),
            fluid_advection_scheme=str(fluid_advection_scheme),
            pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
            pressure_neumann_dt_s=pressure_neumann_dt_s,
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            reset_pressure=bool(reset_pressure),
            pressure_solver=str(pressure_solver),
            pressure_solve_failure_policy=str(pressure_solve_failure_policy),
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=float(cg_tolerance),
            cg_preconditioner=str(cg_preconditioner),
            divergence_cleanup_iterations=int(divergence_cleanup_iterations),
            divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
            read_mpm_report=bool(read_mpm_report),
            solid_external_loads=solid_external_loads,
            classify_far_internal_nodes=bool(classify_far_internal_nodes),
            diagnostic_disable_pressure_neumann_matrix_rows=bool(
                diagnostic_disable_pressure_neumann_matrix_rows
            ),
        )


def assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    mpm_external_force_n,
    mpm_particle_position_m,
    mpm_particle_count: int,
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    primary_region_id: int = 0,
    secondary_region_id: int = 0,
    far_pressure_region_id: int = -1,
    far_pressure_pa: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = True,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    classify_far_internal_nodes: bool = False,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
) -> HibmMpmSharpFluidToMpmLoadReport:
    """Run the sharp-interface fluid solve up to marker traction MPM loading.

    This is the generic HIBM-MPM coupling field path: it assembles no-slip and
    pressure Neumann boundary rows from surface markers, projects the fluid with
    those rows, samples full fluid stress to marker tractions, clears any stale
    MPM external force, and scatters only marker forces into MPM particles.
    """
    particles = int(mpm_particle_count)
    if particles <= 0:
        raise ValueError("mpm_particle_count must be positive")
    iterations = int(projection_iterations)
    if iterations <= 0:
        raise ValueError("projection_iterations must be positive")
    substeps = int(fluid_substeps)
    if substeps <= 0:
        raise ValueError("fluid_substeps must be positive")
    advection_scheme = str(fluid_advection_scheme)
    pressure_solve_failure_policy_name = str(pressure_solve_failure_policy)
    if pressure_solve_failure_policy_name not in {"raise", "report"}:
        raise ValueError("pressure_solve_failure_policy must be 'raise' or 'report'")
    fluid_substep_dt = None
    if dt_s is not None:
        fluid_substep_dt = float(dt_s) / float(substeps)
    else:
        fluid_dt = getattr(fluid, "dt", None)
        if fluid_dt is not None:
            fluid_substep_dt = float(fluid_dt) / float(substeps)
    pressure_neumann_density = None
    pressure_neumann_dt = None
    if pressure_neumann_density_kgm3 is not None:
        pressure_neumann_density = float(pressure_neumann_density_kgm3)
        if (
            not math.isfinite(pressure_neumann_density)
            or pressure_neumann_density <= 0.0
        ):
            raise ValueError("pressure_neumann_density_kgm3 must be finite and positive")
        if pressure_neumann_dt_s is None:
            raise ValueError(
                "pressure_neumann_dt_s must be provided when updating pressure "
                "Neumann from the fluid predictor"
            )
        pressure_neumann_dt = float(pressure_neumann_dt_s)
        if not math.isfinite(pressure_neumann_dt) or pressure_neumann_dt <= 0.0:
            raise ValueError("pressure_neumann_dt_s must be finite and positive")
        pressure_neumann_dt = pressure_neumann_dt / float(substeps)

    def assemble_velocity_dirichlet_rows() -> HibmMpmVelocityDirichletBoundaryReport:
        fluid.clear_velocity_dirichlet_boundary_rows()
        return ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )

    def combine_projection_reports(
        projection_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not projection_reports:
            return {
                "fluid_substeps": substeps,
                "fluid_advection_scheme": advection_scheme,
            }
        combined = dict(projection_reports[-1])
        combined["fluid_substeps"] = substeps
        combined["fluid_advection_scheme"] = advection_scheme
        sum_keys = (
            "cg_project_calls",
            "cg_iterations_total",
            "cg_host_residual_checks",
            "cg_mean_host_reads",
            "cg_mean_projection_count",
            "cg_breakdown_count",
            "hibm_post_dirichlet_consistency_projection_count",
        )
        for key in sum_keys:
            if any(key in report for report in projection_reports):
                combined[key] = sum(int(report.get(key, 0)) for report in projection_reports)
        max_keys = (
            "cg_iterations_max",
            "cg_initial_relative_residual_max",
            "cg_relative_residual_max",
        )
        for key in max_keys:
            if any(key in report for report in projection_reports):
                combined[key] = max(
                    float(report.get(key, 0.0)) for report in projection_reports
                )
        if any("cg_converged_all" in report for report in projection_reports):
            combined["cg_converged_all"] = all(
                bool(report.get("cg_converged_all", True))
                for report in projection_reports
            )
        if any("pressure_solve_failed" in report for report in projection_reports):
            failed = any(
                bool(report.get("pressure_solve_failed", False))
                for report in projection_reports
            )
            combined["pressure_solve_failed"] = failed
            combined["pressure_solve_failure_action"] = (
                "reported" if failed else "none"
            )
        if any(
            "hibm_post_dirichlet_consistency_projection_applied" in report
            for report in projection_reports
        ):
            combined["hibm_post_dirichlet_consistency_projection_applied"] = any(
                bool(
                    report.get(
                        "hibm_post_dirichlet_consistency_projection_applied",
                        False,
                    )
                )
                for report in projection_reports
            )
        breakdown = ""
        for report in projection_reports:
            if report.get("cg_breakdown"):
                breakdown = str(report["cg_breakdown"])
        if breakdown:
            combined["cg_breakdown"] = breakdown
        return combined

    ib_report = ib_search.search_and_classify_grid_fields(
        markers,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
    )
    internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
        ib_search.node_kind_code,
        internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
    )
    boundary_report = ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
    )
    velocity_report = assemble_velocity_dirichlet_rows()
    solid_band_nonprojectable_cell_count = 0
    for _band_pass in range(8):
        band_increment = fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
        if int(band_increment) <= 0:
            break
        solid_band_nonprojectable_cell_count += int(band_increment)
        velocity_report = assemble_velocity_dirichlet_rows()
    pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    pressure_gradient_report = None
    pressure_report = HibmMpmPressureNeumannMatrixReport(
        active_pressure_neumann_rows=0,
        rhs_integral=0.0,
        max_abs_rhs=0.0,
    )
    projection_reports: list[dict[str, Any]] = []
    for _ in range(substeps):
        velocity_report = assemble_velocity_dirichlet_rows()
        pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
        if bool(run_fluid_predictor):
            fluid.predict(
                dt_s=fluid_substep_dt,
                advection_scheme=advection_scheme,
            )
        if pressure_neumann_density is not None and pressure_neumann_dt is not None:
            markers.update_pressure_neumann_gradient_from_fluid_predictor(
                marker_pressure_neumann_gradient_pa_per_m_field,
                velocity_field=fluid.velocity,
                obstacle_field=fluid.obstacle,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                density_kgm3=pressure_neumann_density,
                dt_s=pressure_neumann_dt,
                probe_distance_m=float(interior_probe_distance_m),
            )
            boundary_report = ib_boundary.build_from_search_device_fields(
                ib_search,
                markers,
                marker_pressure_neumann_gradient_pa_per_m_field=(
                    marker_pressure_neumann_gradient_pa_per_m_field
                ),
            )
            velocity_report = assemble_velocity_dirichlet_rows()
            pressure_disconnected_nonprojectable_cell_count = (
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )
            pressure_gradient_report = (
                ib_boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
                    velocity_field=fluid.velocity,
                    obstacle_field=fluid.obstacle,
                    search=ib_search,
                    cell_face_x_m=fluid.cell_face_x_m,
                    cell_face_y_m=fluid.cell_face_y_m,
                    cell_face_z_m=fluid.cell_face_z_m,
                    cell_center_x_m=fluid.cell_center_x_m,
                    cell_center_y_m=fluid.cell_center_y_m,
                    cell_center_z_m=fluid.cell_center_z_m,
                    grid_nodes=fluid.grid.grid_nodes,
                    density_kgm3=pressure_neumann_density,
                    dt_s=pressure_neumann_dt,
                )
            )

        fluid.clear_pressure_interface_matrix_terms()
        if diagnostic_disable_pressure_neumann_matrix_rows:
            pressure_report = HibmMpmPressureNeumannMatrixReport(
                active_pressure_neumann_rows=0,
                rhs_integral=0.0,
                max_abs_rhs=0.0,
            )
        else:
            pressure_report = ib_boundary.assemble_pressure_neumann_matrix_rows(
                fluid.pressure_interface_matrix_diagonal,
                fluid.pressure_interface_matrix_rhs,
                fluid.pressure_interface_coupling_active,
                fluid.pressure_interface_coupling_neighbor,
                fluid.pressure_interface_coupling_coefficient,
                fluid.obstacle,
                fluid.velocity_dirichlet_boundary_active,
                fluid.cell_width_x_m,
                fluid.cell_width_y_m,
                fluid.cell_width_z_m,
                ib_search,
                markers,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
            )
        requested_pressure_solver = str(pressure_solver)
        effective_pressure_solver = requested_pressure_solver
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        if (
            int(pressure_report.active_pressure_neumann_rows) > 0
            and effective_pressure_solver != "fv_cg"
        ):
            effective_pressure_solver = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = "hibm_pressure_neumann_interface_rows"
        project_report = dict(
            fluid.project(
                iterations=iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                dt_s=fluid_substep_dt,
                preserve_velocity_constraints=False,
                reset_pressure=bool(reset_pressure),
                pressure_solver=effective_pressure_solver,
                multigrid_cycles=multigrid_cycles,
                cg_tolerance=float(cg_tolerance),
                cg_preconditioner=str(cg_preconditioner),
                pressure_solve_failure_policy=pressure_solve_failure_policy_name,
                divergence_cleanup_iterations=int(divergence_cleanup_iterations),
                divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
                read_report=True,
            )
        )
        project_report.update(
            {
                "pressure_solver_requested": requested_pressure_solver,
                "pressure_solver": effective_pressure_solver,
                "pressure_solver_forced_to_fv_cg": pressure_solver_forced_to_fv_cg,
                "pressure_solver_force_reason": pressure_solver_force_reason,
                "pressure_interface_neumann_active_rows": int(
                    pressure_report.active_pressure_neumann_rows
                ),
            }
        )
        projection_reports.append(project_report)
    projection_report = combine_projection_reports(projection_reports)
    velocity_report = assemble_velocity_dirichlet_rows()
    pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    if int(velocity_report.active_velocity_dirichlet_rows) > 0:
        requested_pressure_solver = str(pressure_solver)
        effective_pressure_solver = requested_pressure_solver
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        if (
            int(pressure_report.active_pressure_neumann_rows) > 0
            and effective_pressure_solver != "fv_cg"
        ):
            effective_pressure_solver = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = "hibm_pressure_neumann_interface_rows"
        consistency_project_report = dict(
            fluid.project(
                iterations=iterations,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                dt_s=fluid_substep_dt,
                preserve_velocity_constraints=False,
                reset_pressure=bool(reset_pressure),
                pressure_solver=effective_pressure_solver,
                multigrid_cycles=multigrid_cycles,
                cg_tolerance=float(cg_tolerance),
                cg_preconditioner=str(cg_preconditioner),
                pressure_solve_failure_policy=pressure_solve_failure_policy_name,
                divergence_cleanup_iterations=int(divergence_cleanup_iterations),
                divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
                read_report=True,
            )
        )
        consistency_project_report.update(
            {
                "pressure_solver_requested": requested_pressure_solver,
                "pressure_solver": effective_pressure_solver,
                "pressure_solver_forced_to_fv_cg": pressure_solver_forced_to_fv_cg,
                "pressure_solver_force_reason": pressure_solver_force_reason,
                "pressure_interface_neumann_active_rows": int(
                    pressure_report.active_pressure_neumann_rows
                ),
                "hibm_projection_stage": "post_dirichlet_reconstruction_consistency",
                "hibm_post_dirichlet_consistency_projection_applied": True,
                "hibm_post_dirichlet_consistency_projection_count": 1,
            }
        )
        projection_reports.append(consistency_project_report)
        projection_report = combine_projection_reports(projection_reports)
    fluid.compute_divergence(pressure_outlet_zmin=bool(pressure_outlet_zmin))
    (
        final_raw_stats,
        final_stats,
        final_interior_raw_stats,
        final_interior_stats,
        velocity_dirichlet_near_raw_stats,
        velocity_dirichlet_near_stats,
        velocity_dirichlet_far_raw_stats,
        velocity_dirichlet_far_stats,
        pressure_correctable_raw_stats,
        pressure_correctable_stats,
        pressure_fixed_raw_stats,
        pressure_fixed_stats,
        interior_pressure_correctable_raw_stats,
        interior_pressure_correctable_stats,
        interior_pressure_fixed_raw_stats,
        interior_pressure_fixed_stats,
    ) = fluid.final_and_dirichlet_partition_report_stats(
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
    )
    projection_report = {
        **projection_report,
        "l2": final_stats["l2"],
        "max_abs": final_stats["max_abs"],
        "raw_l2": final_raw_stats["l2"],
        "raw_max_abs": final_raw_stats["max_abs"],
        "interior_l2": final_interior_stats["l2"],
        "interior_max_abs": final_interior_stats["max_abs"],
        "interior_raw_l2": final_interior_raw_stats["l2"],
        "interior_raw_max_abs": final_interior_raw_stats["max_abs"],
        "unreached_l2": float(fluid.last_unreached_divergence_stats["l2"]),
        "unreached_max_abs": float(
            fluid.last_unreached_divergence_stats["max_abs"]
        ),
        "unreached_cell_count": int(
            fluid.last_unreached_divergence_stats["count"]
        ),
        "post_constraint_l2": final_stats["l2"],
        "post_constraint_max_abs": final_stats["max_abs"],
        "post_constraint_raw_l2": final_raw_stats["l2"],
        "post_constraint_raw_max_abs": final_raw_stats["max_abs"],
        "velocity_dirichlet_near_l2": velocity_dirichlet_near_stats["l2"],
        "velocity_dirichlet_near_max_abs": velocity_dirichlet_near_stats["max_abs"],
        "velocity_dirichlet_near_raw_l2": velocity_dirichlet_near_raw_stats["l2"],
        "velocity_dirichlet_near_raw_max_abs": (
            velocity_dirichlet_near_raw_stats["max_abs"]
        ),
        "velocity_dirichlet_far_l2": velocity_dirichlet_far_stats["l2"],
        "velocity_dirichlet_far_max_abs": velocity_dirichlet_far_stats["max_abs"],
        "velocity_dirichlet_far_raw_l2": velocity_dirichlet_far_raw_stats["l2"],
        "velocity_dirichlet_far_raw_max_abs": (
            velocity_dirichlet_far_raw_stats["max_abs"]
        ),
        "pressure_correctable_l2": pressure_correctable_stats["l2"],
        "pressure_correctable_max_abs": pressure_correctable_stats["max_abs"],
        "pressure_correctable_cell_count": pressure_correctable_stats["count"],
        "pressure_correctable_raw_l2": pressure_correctable_raw_stats["l2"],
        "pressure_correctable_raw_max_abs": (
            pressure_correctable_raw_stats["max_abs"]
        ),
        "pressure_fixed_l2": pressure_fixed_stats["l2"],
        "pressure_fixed_max_abs": pressure_fixed_stats["max_abs"],
        "pressure_fixed_cell_count": pressure_fixed_stats["count"],
        "pressure_fixed_raw_l2": pressure_fixed_raw_stats["l2"],
        "pressure_fixed_raw_max_abs": pressure_fixed_raw_stats["max_abs"],
        "interior_pressure_correctable_l2": (
            interior_pressure_correctable_stats["l2"]
        ),
        "interior_pressure_correctable_max_abs": (
            interior_pressure_correctable_stats["max_abs"]
        ),
        "interior_pressure_correctable_cell_count": (
            interior_pressure_correctable_stats["count"]
        ),
        "interior_pressure_correctable_raw_l2": (
            interior_pressure_correctable_raw_stats["l2"]
        ),
        "interior_pressure_correctable_raw_max_abs": (
            interior_pressure_correctable_raw_stats["max_abs"]
        ),
        "interior_pressure_fixed_l2": interior_pressure_fixed_stats["l2"],
        "interior_pressure_fixed_max_abs": (
            interior_pressure_fixed_stats["max_abs"]
        ),
        "interior_pressure_fixed_cell_count": interior_pressure_fixed_stats["count"],
        "interior_pressure_fixed_raw_l2": interior_pressure_fixed_raw_stats["l2"],
        "interior_pressure_fixed_raw_max_abs": (
            interior_pressure_fixed_raw_stats["max_abs"]
        ),
    }
    no_slip_report = markers.sample_no_slip_residual(
        fluid.velocity,
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.grid.grid_nodes,
    )
    stress_report = markers.sample_fluid_stress_to_marker_tractions(
        fluid.velocity,
        fluid.pressure,
        fluid.obstacle,
        fluid.cell_face_x_m,
        fluid.cell_face_y_m,
        fluid.cell_face_z_m,
        fluid.cell_center_x_m,
        fluid.cell_center_y_m,
        fluid.cell_center_z_m,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        fluid.grid.grid_nodes,
        viscosity_pa_s=fluid.mu,
        two_sided_pressure=True,
        far_pressure_region_id=int(far_pressure_region_id),
        far_pressure_pa=float(far_pressure_pa),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        # S2-A6: the anchor fallback rides the closure opt-in. It is only
        # armed when this very call also assembled the pressure-Neumann
        # rows (anchors are captured there, strictly before this sampling
        # and before the projection that solved fluid.pressure); with the
        # diagnostic row disable the anchors would be stale, so keep the
        # fallback off.
        use_pressure_anchor_fallback=(
            int(far_pressure_region_id) != -1
            and not bool(diagnostic_disable_pressure_neumann_matrix_rows)
        ),
    )
    markers.compute_marker_forces()
    marker_force_report = markers.aggregate_region_forces(
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
    )
    clear_report = markers.clear_mpm_external_forces(
        mpm_external_force_n,
        particle_count=particles,
    )
    scatter_report = markers.scatter_marker_forces_to_mpm_particles(
        mpm_external_force_n,
        mpm_particle_position_m,
        particle_count=particles,
        support_radius_m=float(mpm_support_radius_m),
    )
    return HibmMpmSharpFluidToMpmLoadReport(
        ib_node_search=ib_report,
        internal_obstacle_cell_count=internal_obstacle_cell_count,
        solid_band_nonprojectable_cell_count=solid_band_nonprojectable_cell_count,
        pressure_disconnected_nonprojectable_cell_count=(
            pressure_disconnected_nonprojectable_cell_count
        ),
        boundary_conditions=boundary_report,
        pressure_neumann_gradient=pressure_gradient_report,
        velocity_dirichlet=velocity_report,
        pressure_neumann=pressure_report,
        fluid_predictor_applied=bool(run_fluid_predictor),
        fluid_projection=projection_report,
        no_slip_residual=no_slip_report,
        fluid_stress=stress_report,
        marker_forces=marker_force_report,
        mpm_external_force_clear=clear_report,
        mpm_force_scatter=scatter_report,
    )


def advance_hibm_mpm_sharp_mpm_step(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    mpm_external_force_n,
    mpm_particle_position_m,
    mpm_particle_velocity_mps,
    mpm_particle_normal,
    mpm_particle_area_m2,
    mpm_particle_count: int,
    solid_step: Callable[[], Any],
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    primary_region_id: int = 0,
    secondary_region_id: int = 0,
    far_pressure_region_id: int = -1,
    far_pressure_pa: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    fluid_dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = True,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    surface_feedback_dt_s: float | None = None,
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    classify_far_internal_nodes: bool = False,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
) -> HibmMpmSharpMpmStepReport:
    if not callable(solid_step):
        raise ValueError("solid_step must be callable")
    particles = int(mpm_particle_count)
    if surface_feedback_dt_s is None:
        feedback_dt = float(fluid_dt_s) if fluid_dt_s is not None else float(fluid.dt)
    else:
        feedback_dt = float(surface_feedback_dt_s)
    if not math.isfinite(feedback_dt) or feedback_dt <= 0.0:
        raise ValueError("surface_feedback_dt_s must be finite and positive")
    pressure_neumann_density = None
    pressure_neumann_dt = None
    if pressure_neumann_density_kgm3 is not None:
        pressure_neumann_density = float(pressure_neumann_density_kgm3)
        if (
            not math.isfinite(pressure_neumann_density)
            or pressure_neumann_density <= 0.0
        ):
            raise ValueError("pressure_neumann_density_kgm3 must be finite and positive")
        if pressure_neumann_dt_s is None:
            raise ValueError(
                "pressure_neumann_dt_s must be provided when updating pressure "
                "Neumann from the fluid predictor"
            )
        pressure_neumann_dt = float(pressure_neumann_dt_s)
        if not math.isfinite(pressure_neumann_dt) or pressure_neumann_dt <= 0.0:
            raise ValueError("pressure_neumann_dt_s must be finite and positive")
    load_report = assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
        fluid=fluid,
        markers=markers,
        ib_search=ib_search,
        ib_boundary=ib_boundary,
        mpm_external_force_n=mpm_external_force_n,
        mpm_particle_position_m=mpm_particle_position_m,
        mpm_particle_count=particles,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        mpm_support_radius_m=float(mpm_support_radius_m),
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
        far_pressure_region_id=int(far_pressure_region_id),
        far_pressure_pa=float(far_pressure_pa),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        dt_s=fluid_dt_s,
        fluid_substeps=int(fluid_substeps),
        projection_iterations=int(projection_iterations),
        run_fluid_predictor=bool(run_fluid_predictor),
        fluid_advection_scheme=str(fluid_advection_scheme),
        pressure_neumann_density_kgm3=pressure_neumann_density,
        pressure_neumann_dt_s=pressure_neumann_dt,
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
        reset_pressure=bool(reset_pressure),
        pressure_solver=str(pressure_solver),
        pressure_solve_failure_policy=str(pressure_solve_failure_policy),
        multigrid_cycles=multigrid_cycles,
        cg_tolerance=float(cg_tolerance),
        cg_preconditioner=str(cg_preconditioner),
        divergence_cleanup_iterations=int(divergence_cleanup_iterations),
        divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
        diagnostic_disable_pressure_neumann_matrix_rows=bool(
            diagnostic_disable_pressure_neumann_matrix_rows
        ),
    )
    mpm_report = solid_step()
    feedback_report = markers.update_surface_feedback_from_mpm_surface_particles(
        mpm_particle_position_m,
        mpm_particle_velocity_mps,
        mpm_particle_normal,
        mpm_particle_area_m2,
        particle_count=particles,
        support_radius_m=float(mpm_support_radius_m),
        dt_s=feedback_dt,
    )
    next_ib_report = ib_search.search_and_classify_grid_fields(
        markers,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
    )
    next_pressure_neumann_gradient_report = None
    if pressure_neumann_density is not None and pressure_neumann_dt is not None:
        next_pressure_neumann_gradient_report = (
            ib_boundary.update_pressure_neumann_gradient_from_fluid_predictor_ib_nodes(
                velocity_field=fluid.velocity,
                obstacle_field=fluid.obstacle,
                search=ib_search,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
                density_kgm3=pressure_neumann_density,
                dt_s=pressure_neumann_dt,
            )
        )
    next_internal_obstacle_cell_count = fluid.apply_hibm_internal_obstacles(
        ib_search.node_kind_code,
        internal_node_code=HibmMpmIbNodeSearch._NODE_INTERNAL,
    )
    next_boundary_report = ib_boundary.build_from_search_device_fields(
        ib_search,
        markers,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
    )
    fluid.clear_velocity_dirichlet_boundary_rows()
    fluid.clear_pressure_interface_matrix_terms()
    next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
        fluid.velocity_dirichlet_boundary_active,
        fluid.velocity_dirichlet_boundary_value_mps,
        fluid.velocity_dirichlet_boundary_projection_weight,
        fluid.obstacle,
        fluid.velocity,
        ib_search,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
    )
    next_solid_band_nonprojectable_cell_count = (
        fluid.mark_hibm_solid_band_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    next_pressure_disconnected_nonprojectable_cell_count = (
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
        )
    )
    if (
        int(next_solid_band_nonprojectable_cell_count) > 0
        or int(next_pressure_disconnected_nonprojectable_cell_count) > 0
    ):
        next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            ib_search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
        )
        for _next_band_pass in range(8):
            next_band_increment = (
                fluid.mark_hibm_solid_band_nonprojectable_cells(
                    pressure_outlet_zmin=bool(pressure_outlet_zmin),
                )
            )
            if int(next_band_increment) <= 0:
                break
            next_solid_band_nonprojectable_cell_count = int(
                next_solid_band_nonprojectable_cell_count
            ) + int(next_band_increment)
            next_velocity_report = ib_boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
                fluid.velocity_dirichlet_boundary_active,
                fluid.velocity_dirichlet_boundary_value_mps,
                fluid.velocity_dirichlet_boundary_projection_weight,
                fluid.obstacle,
                fluid.velocity,
                ib_search,
                cell_face_x_m=fluid.cell_face_x_m,
                cell_face_y_m=fluid.cell_face_y_m,
                cell_face_z_m=fluid.cell_face_z_m,
                cell_center_x_m=fluid.cell_center_x_m,
                cell_center_y_m=fluid.cell_center_y_m,
                cell_center_z_m=fluid.cell_center_z_m,
                grid_nodes=fluid.grid.grid_nodes,
            )
        next_pressure_disconnected_nonprojectable_cell_count = (
            fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
            )
        )
    next_pressure_report = ib_boundary.assemble_pressure_neumann_matrix_rows(
        fluid.pressure_interface_matrix_diagonal,
        fluid.pressure_interface_matrix_rhs,
        fluid.pressure_interface_coupling_active,
        fluid.pressure_interface_coupling_neighbor,
        fluid.pressure_interface_coupling_coefficient,
        fluid.obstacle,
        fluid.velocity_dirichlet_boundary_active,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        ib_search,
        markers,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
    )
    return HibmMpmSharpMpmStepReport(
        fluid_to_mpm_loads=load_report,
        mpm=mpm_report,
        surface_feedback=feedback_report,
        next_ib_node_search=next_ib_report,
        next_internal_obstacle_cell_count=next_internal_obstacle_cell_count,
        next_solid_band_nonprojectable_cell_count=next_solid_band_nonprojectable_cell_count,
        next_pressure_disconnected_nonprojectable_cell_count=(
            next_pressure_disconnected_nonprojectable_cell_count
        ),
        next_boundary_conditions=next_boundary_report,
        next_velocity_dirichlet=next_velocity_report,
        next_pressure_neumann=next_pressure_report,
        next_pressure_neumann_gradient=next_pressure_neumann_gradient_report,
    )


def hibm_mpm_sharp_step_summary(
    report: HibmMpmSharpMpmStepReport,
) -> dict[str, Any]:
    load = report.fluid_to_mpm_loads
    marker_forces = load.marker_forces
    scatter = load.mpm_force_scatter
    feedback = report.surface_feedback
    pressure_gradient = load.pressure_neumann_gradient
    next_gradient = report.next_pressure_neumann_gradient
    return {
        "hibm_coupling_scheme": "explicit_loose",
        "hibm_added_mass_stability_status": "unmeasured",
        "hibm_added_mass_stability_measured": False,
        "hibm_added_mass_stabilization": "none",
        "hibm_semi_implicit_coupling_enabled": False,
        "hibm_semi_implicit_coupling_matrix_active": False,
        "hibm_ib_node_count": load.ib_node_search.near_boundary_node_count,
        "hibm_ib_external_node_count": load.ib_node_search.external_ib_node_count,
        "hibm_ib_internal_node_count": load.ib_node_search.internal_node_count,
        "hibm_internal_obstacle_cell_count": load.internal_obstacle_cell_count,
        "hibm_solid_band_nonprojectable_cell_count": (
            load.solid_band_nonprojectable_cell_count
        ),
        "hibm_pressure_disconnected_nonprojectable_cell_count": (
            load.pressure_disconnected_nonprojectable_cell_count
        ),
        "hibm_ib_invalid_projection_count": (
            load.ib_node_search.invalid_projection_count
        ),
        "hibm_boundary_no_slip_count": (
            load.boundary_conditions.no_slip_dirichlet_count
        ),
        "hibm_boundary_pressure_neumann_count": (
            load.boundary_conditions.pressure_neumann_count
        ),
        "hibm_velocity_dirichlet_active_rows": (
            load.velocity_dirichlet.active_velocity_dirichlet_rows
        ),
        "hibm_velocity_dirichlet_invalid_reconstruction_count": (
            load.velocity_dirichlet.invalid_reconstruction_row_count
        ),
        "hibm_velocity_dirichlet_invalid_no_fluid_sample_count": (
            load.velocity_dirichlet.invalid_no_fluid_sample_row_count
        ),
        "hibm_velocity_dirichlet_invalid_nonpositive_gap_count": (
            load.velocity_dirichlet.invalid_nonpositive_gap_row_count
        ),
        "hibm_velocity_dirichlet_invalid_node_behind_boundary_count": (
            load.velocity_dirichlet.invalid_node_behind_boundary_row_count
        ),
        "hibm_velocity_dirichlet_invalid_node_beyond_interior_count": (
            load.velocity_dirichlet.invalid_node_beyond_interior_row_count
        ),
        "hibm_velocity_dirichlet_narrow_gap_count": (
            load.velocity_dirichlet.narrow_gap_boundary_velocity_row_count
        ),
        "hibm_velocity_dirichlet_relocated_rows": (
            load.velocity_dirichlet.relocated_row_count
        ),
        "hibm_velocity_dirichlet_relocation_merged_rows": (
            load.velocity_dirichlet.relocation_merged_row_count
        ),
        "hibm_velocity_dirichlet_relocation_blocked_rows": (
            load.velocity_dirichlet.relocation_blocked_row_count
        ),
        "hibm_velocity_dirichlet_min_projection_weight": (
            load.velocity_dirichlet.min_projection_weight
        ),
        "hibm_velocity_dirichlet_max_projection_weight": (
            load.velocity_dirichlet.max_projection_weight
        ),
        "hibm_pressure_neumann_active_rows": (
            load.pressure_neumann.active_pressure_neumann_rows
        ),
        "hibm_pressure_neumann_skipped_velocity_dirichlet_count": (
            load.pressure_neumann.skipped_velocity_dirichlet_row_count
        ),
        "hibm_pressure_neumann_skipped_obstacle_owner_count": (
            load.pressure_neumann.skipped_obstacle_owner_row_count
        ),
        "hibm_pressure_neumann_rhs_integral": load.pressure_neumann.rhs_integral,
        "hibm_pressure_neumann_max_abs_rhs": load.pressure_neumann.max_abs_rhs,
        "hibm_pressure_neumann_invalid_reconstruction_count": (
            load.pressure_neumann.invalid_reconstruction_row_count
        ),
        "hibm_pressure_neumann_min_reconstruction_gap_m": (
            load.pressure_neumann.min_reconstruction_gap_m
        ),
        "hibm_pressure_neumann_max_reconstruction_gap_m": (
            load.pressure_neumann.max_reconstruction_gap_m
        ),
        "hibm_pressure_neumann_max_transmissibility_m": (
            load.pressure_neumann.max_transmissibility_m
        ),
        "hibm_pressure_neumann_max_raw_transmissibility_m": (
            load.pressure_neumann.max_raw_transmissibility_m
        ),
        "hibm_pressure_neumann_max_transmissibility_limit_m": (
            load.pressure_neumann.max_transmissibility_limit_m
        ),
        "hibm_pressure_neumann_transmissibility_capped_row_count": (
            load.pressure_neumann.transmissibility_capped_row_count
        ),
        "hibm_pressure_neumann_max_diagonal_per_m2": (
            load.pressure_neumann.max_diagonal_per_m2
        ),
        "hibm_pressure_neumann_active_marker_count": (
            load.pressure_neumann.active_pressure_neumann_marker_count
        ),
        "hibm_pressure_neumann_max_rows_per_marker": (
            load.pressure_neumann.max_pressure_neumann_rows_per_marker
        ),
        "hibm_pressure_neumann_gradient_available": pressure_gradient is not None,
        "hibm_pressure_neumann_gradient_active_marker_count": (
            0 if pressure_gradient is None else pressure_gradient.active_marker_count
        ),
        "hibm_pressure_neumann_gradient_max_abs_pa_per_m": (
            0.0
            if pressure_gradient is None
            else pressure_gradient.max_abs_gradient_pa_per_m
        ),
        "hibm_velocity_dirichlet_near_divergence_l2": load.fluid_projection.get(
            "velocity_dirichlet_near_l2",
            0.0,
        ),
        "hibm_velocity_dirichlet_near_divergence_max_abs": load.fluid_projection.get(
            "velocity_dirichlet_near_max_abs",
            0.0,
        ),
        "hibm_velocity_dirichlet_far_divergence_l2": load.fluid_projection.get(
            "velocity_dirichlet_far_l2",
            0.0,
        ),
        "hibm_velocity_dirichlet_far_divergence_max_abs": load.fluid_projection.get(
            "velocity_dirichlet_far_max_abs",
            0.0,
        ),
        "hibm_pressure_correctable_divergence_l2": load.fluid_projection.get(
            "pressure_correctable_l2",
            0.0,
        ),
        "hibm_pressure_correctable_divergence_max_abs": load.fluid_projection.get(
            "pressure_correctable_max_abs",
            0.0,
        ),
        "hibm_pressure_correctable_divergence_cell_count": load.fluid_projection.get(
            "pressure_correctable_cell_count",
            0,
        ),
        "hibm_pressure_fixed_divergence_l2": load.fluid_projection.get(
            "pressure_fixed_l2",
            0.0,
        ),
        "hibm_pressure_fixed_divergence_max_abs": load.fluid_projection.get(
            "pressure_fixed_max_abs",
            0.0,
        ),
        "hibm_pressure_fixed_divergence_cell_count": load.fluid_projection.get(
            "pressure_fixed_cell_count",
            0,
        ),
        "hibm_interior_pressure_correctable_divergence_l2": (
            load.fluid_projection.get("interior_pressure_correctable_l2", 0.0)
        ),
        "hibm_interior_pressure_correctable_divergence_max_abs": (
            load.fluid_projection.get("interior_pressure_correctable_max_abs", 0.0)
        ),
        "hibm_interior_pressure_correctable_divergence_cell_count": (
            load.fluid_projection.get("interior_pressure_correctable_cell_count", 0)
        ),
        "hibm_interior_pressure_fixed_divergence_l2": load.fluid_projection.get(
            "interior_pressure_fixed_l2",
            0.0,
        ),
        "hibm_interior_pressure_fixed_divergence_max_abs": load.fluid_projection.get(
            "interior_pressure_fixed_max_abs",
            0.0,
        ),
        "hibm_interior_pressure_fixed_divergence_cell_count": load.fluid_projection.get(
            "interior_pressure_fixed_cell_count",
            0,
        ),
        "hibm_fluid_predictor_applied": load.fluid_predictor_applied,
        "hibm_no_slip_residual_valid_marker_count": (
            load.no_slip_residual.valid_marker_count
        ),
        "hibm_no_slip_residual_invalid_marker_count": (
            load.no_slip_residual.invalid_marker_count
        ),
        "hibm_no_slip_residual_max_mps": (
            load.no_slip_residual.max_no_slip_residual_mps
        ),
        "hibm_no_slip_residual_l2_mps": (
            load.no_slip_residual.l2_no_slip_residual_mps
        ),
        "hibm_full_stress_valid_marker_count": load.fluid_stress.valid_marker_count,
        "hibm_full_stress_invalid_marker_count": (
            load.fluid_stress.invalid_marker_count
        ),
        "hibm_full_stress_viscous_gradient_invalid_marker_count": (
            load.fluid_stress.viscous_gradient_invalid_marker_count
        ),
        "hibm_full_stress_max_abs_traction_pa": (
            load.fluid_stress.max_abs_traction_pa
        ),
        "hibm_full_stress_far_pressure_closed_marker_count": (
            load.fluid_stress.far_pressure_closed_marker_count
        ),
        "hibm_full_stress_far_pressure_closed_extended_marker_count": (
            load.fluid_stress.far_pressure_closed_extended_marker_count
        ),
        "hibm_full_stress_far_pressure_anchor_closed_marker_count": (
            load.fluid_stress.far_pressure_anchor_closed_marker_count
        ),
        "hibm_full_stress_closure_gradient_missing_marker_count": (
            load.fluid_stress.closure_gradient_missing_marker_count
        ),
        "hibm_marker_primary_count": marker_forces.primary_marker_count,
        "hibm_marker_secondary_count": marker_forces.secondary_marker_count,
        "hibm_marker_total_count": marker_forces.total_marker_count,
        "hibm_marker_primary_force_n": marker_forces.primary_marker_force_n,
        "hibm_marker_secondary_force_n": marker_forces.secondary_marker_force_n,
        "hibm_marker_total_force_n": marker_forces.total_marker_force_n,
        "hibm_marker_fluid_reaction_force_n": marker_forces.fluid_reaction_force_n,
        "hibm_marker_action_reaction_residual_n": (
            marker_forces.action_reaction_residual_n
        ),
        "hibm_mpm_scatter_active_marker_count": scatter.active_marker_count,
        "hibm_mpm_scatter_invalid_marker_count": scatter.invalid_marker_count,
        "hibm_mpm_scatter_active_particle_count": scatter.active_particle_count,
        "hibm_mpm_scatter_total_external_force_n": (
            scatter.total_mpm_external_force_n
        ),
        "hibm_mpm_scatter_action_reaction_residual_n": (
            scatter.action_reaction_residual_n
        ),
        "hibm_surface_updated_marker_count": feedback.updated_marker_count,
        "hibm_surface_invalid_marker_count": feedback.invalid_marker_count,
        "hibm_surface_max_displacement_m": feedback.max_marker_displacement_m,
        "hibm_surface_max_speed_mps": feedback.max_marker_speed_mps,
        "hibm_surface_geometry_updated_marker_count": (
            feedback.geometry_updated_marker_count
        ),
        "hibm_surface_geometry_invalid_marker_count": (
            feedback.geometry_invalid_marker_count
        ),
        "hibm_next_ib_node_count": report.next_ib_node_search.near_boundary_node_count,
        "hibm_next_ib_external_node_count": (
            report.next_ib_node_search.external_ib_node_count
        ),
        "hibm_next_ib_internal_node_count": (
            report.next_ib_node_search.internal_node_count
        ),
        "hibm_next_internal_obstacle_cell_count": (
            report.next_internal_obstacle_cell_count
        ),
        "hibm_next_solid_band_nonprojectable_cell_count": (
            report.next_solid_band_nonprojectable_cell_count
        ),
        "hibm_next_pressure_disconnected_nonprojectable_cell_count": (
            report.next_pressure_disconnected_nonprojectable_cell_count
        ),
        "hibm_next_ib_invalid_projection_count": (
            report.next_ib_node_search.invalid_projection_count
        ),
        "hibm_next_boundary_no_slip_count": (
            report.next_boundary_conditions.no_slip_dirichlet_count
        ),
        "hibm_next_boundary_pressure_neumann_count": (
            report.next_boundary_conditions.pressure_neumann_count
        ),
        "hibm_next_velocity_dirichlet_invalid_reconstruction_count": (
            report.next_velocity_dirichlet.invalid_reconstruction_row_count
        ),
        "hibm_next_velocity_dirichlet_min_projection_weight": (
            report.next_velocity_dirichlet.min_projection_weight
        ),
        "hibm_next_velocity_dirichlet_max_projection_weight": (
            report.next_velocity_dirichlet.max_projection_weight
        ),
        "hibm_next_pressure_neumann_active_rows": (
            report.next_pressure_neumann.active_pressure_neumann_rows
        ),
        "hibm_next_pressure_neumann_skipped_velocity_dirichlet_count": (
            report.next_pressure_neumann.skipped_velocity_dirichlet_row_count
        ),
        "hibm_next_pressure_neumann_invalid_reconstruction_count": (
            report.next_pressure_neumann.invalid_reconstruction_row_count
        ),
        "hibm_next_pressure_neumann_min_reconstruction_gap_m": (
            report.next_pressure_neumann.min_reconstruction_gap_m
        ),
        "hibm_next_pressure_neumann_max_reconstruction_gap_m": (
            report.next_pressure_neumann.max_reconstruction_gap_m
        ),
        "hibm_next_pressure_neumann_max_transmissibility_m": (
            report.next_pressure_neumann.max_transmissibility_m
        ),
        "hibm_next_pressure_neumann_max_raw_transmissibility_m": (
            report.next_pressure_neumann.max_raw_transmissibility_m
        ),
        "hibm_next_pressure_neumann_max_transmissibility_limit_m": (
            report.next_pressure_neumann.max_transmissibility_limit_m
        ),
        "hibm_next_pressure_neumann_transmissibility_capped_row_count": (
            report.next_pressure_neumann.transmissibility_capped_row_count
        ),
        "hibm_next_pressure_neumann_max_diagonal_per_m2": (
            report.next_pressure_neumann.max_diagonal_per_m2
        ),
        "hibm_next_pressure_neumann_gradient_available": next_gradient is not None,
        "hibm_next_pressure_neumann_gradient_active_marker_count": (
            0 if next_gradient is None else next_gradient.active_marker_count
        ),
        "hibm_next_pressure_neumann_gradient_max_abs_pa_per_m": (
            0.0
            if next_gradient is None
            else next_gradient.max_abs_gradient_pa_per_m
        ),
    }


def advance_hibm_mpm_sharp_neo_hookean_step(
    *,
    fluid: Any,
    markers: HibmMpmSurfaceMarkers,
    ib_search: HibmMpmIbNodeSearch,
    ib_boundary: HibmMpmIbBoundaryConditions,
    solid: Any,
    marker_pressure_neumann_gradient_pa_per_m_field,
    search_radius_m: float,
    interior_probe_distance_m: float,
    mpm_support_radius_m: float,
    solid_dt_s: float,
    mu_pa: float,
    lambda_pa: float,
    primary_region_id: int,
    secondary_region_id: int,
    far_pressure_region_id: int = -1,
    far_pressure_pa: float = 0.0,
    far_pressure_inside_probe_max_multiplier: float = 3.0,
    fluid_dt_s: float | None = None,
    fluid_substeps: int = 1,
    projection_iterations: int = 40,
    run_fluid_predictor: bool = True,
    fluid_advection_scheme: str = "euler",
    pressure_neumann_density_kgm3: float | None = None,
    pressure_neumann_dt_s: float | None = None,
    pressure_outlet_zmin: bool = False,
    reset_pressure: bool = True,
    pressure_solver: str = "fv_cg",
    pressure_solve_failure_policy: str = "raise",
    multigrid_cycles: int | None = None,
    cg_tolerance: float = 1.0e-6,
    cg_preconditioner: str = "auto",
    divergence_cleanup_iterations: int = 0,
    divergence_cleanup_relaxation: float = 0.7,
    read_mpm_report: bool = True,
    solid_external_loads: Callable[[], None] | None = None,
    classify_far_internal_nodes: bool = False,
    diagnostic_disable_pressure_neumann_matrix_rows: bool = False,
) -> HibmMpmSharpNeoHookeanStepReport:
    def run_solid_step_with_external_loads() -> Any:
        if solid_external_loads is not None:
            solid_external_loads()
        return solid.step(
            dt_s=float(solid_dt_s),
            mu_pa=float(mu_pa),
            lambda_pa=float(lambda_pa),
            primary_region_id=int(primary_region_id),
            secondary_region_id=int(secondary_region_id),
            velocity_damping=1.0,
            read_report=bool(read_mpm_report),
        )

    report = advance_hibm_mpm_sharp_mpm_step(
        fluid=fluid,
        markers=markers,
        ib_search=ib_search,
        ib_boundary=ib_boundary,
        mpm_external_force_n=solid.external_force_n,
        mpm_particle_position_m=solid.x,
        mpm_particle_velocity_mps=solid.v,
        mpm_particle_normal=solid.surface_normal,
        mpm_particle_area_m2=solid.area_weight_m2,
        mpm_particle_count=solid.particle_count,
        solid_step=run_solid_step_with_external_loads,
        marker_pressure_neumann_gradient_pa_per_m_field=(
            marker_pressure_neumann_gradient_pa_per_m_field
        ),
        search_radius_m=float(search_radius_m),
        interior_probe_distance_m=float(interior_probe_distance_m),
        mpm_support_radius_m=float(mpm_support_radius_m),
        primary_region_id=int(primary_region_id),
        secondary_region_id=int(secondary_region_id),
        far_pressure_region_id=int(far_pressure_region_id),
        far_pressure_pa=float(far_pressure_pa),
        far_pressure_inside_probe_max_multiplier=float(
            far_pressure_inside_probe_max_multiplier
        ),
        fluid_dt_s=fluid_dt_s,
        fluid_substeps=int(fluid_substeps),
        projection_iterations=int(projection_iterations),
        run_fluid_predictor=bool(run_fluid_predictor),
        fluid_advection_scheme=str(fluid_advection_scheme),
        pressure_neumann_density_kgm3=pressure_neumann_density_kgm3,
        pressure_neumann_dt_s=(
            float(solid_dt_s) if pressure_neumann_dt_s is None else pressure_neumann_dt_s
        ),
        pressure_outlet_zmin=bool(pressure_outlet_zmin),
        reset_pressure=bool(reset_pressure),
        pressure_solver=str(pressure_solver),
        pressure_solve_failure_policy=str(pressure_solve_failure_policy),
        multigrid_cycles=multigrid_cycles,
        cg_tolerance=float(cg_tolerance),
        cg_preconditioner=str(cg_preconditioner),
        surface_feedback_dt_s=float(solid_dt_s),
        divergence_cleanup_iterations=int(divergence_cleanup_iterations),
        divergence_cleanup_relaxation=float(divergence_cleanup_relaxation),
        classify_far_internal_nodes=bool(classify_far_internal_nodes),
        diagnostic_disable_pressure_neumann_matrix_rows=bool(
            diagnostic_disable_pressure_neumann_matrix_rows
        ),
    )
    return HibmMpmSharpNeoHookeanStepReport(
        fluid_to_mpm_loads=report.fluid_to_mpm_loads,
        mpm=report.mpm,
        surface_feedback=report.surface_feedback,
        next_ib_node_search=report.next_ib_node_search,
        next_internal_obstacle_cell_count=report.next_internal_obstacle_cell_count,
        next_solid_band_nonprojectable_cell_count=(
            report.next_solid_band_nonprojectable_cell_count
        ),
        next_pressure_disconnected_nonprojectable_cell_count=(
            report.next_pressure_disconnected_nonprojectable_cell_count
        ),
        next_boundary_conditions=report.next_boundary_conditions,
        next_velocity_dirichlet=report.next_velocity_dirichlet,
        next_pressure_neumann=report.next_pressure_neumann,
        next_pressure_neumann_gradient=report.next_pressure_neumann_gradient,
    )


_PAPER_REQUIREMENTS = (
    HibmMpmPaperRequirement(
        requirement="Taichi-resident solver path",
        paper_section="Implementation constraint",
        paper_mechanism=(
            "Solver state for HIBM-MPM search, reconstruction, matrix boundary "
            "terms, stress sampling, and MPM force scatter must remain on the "
            "Taichi side during runtime."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep extending the Taichi field assembly into a complete sharp "
            "step; do not use NumPy loops or host round-trips as the solver "
            "path."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="surface markers",
        paper_section="Sections 3.2-4",
        paper_mechanism=(
            "The solid surface is represented by material points and an "
            "unstructured triangular interface carrying positions, velocities, "
            "normals, and surface areas."
        ),
        current_status="partial",
        required_solver_work=(
            "Build a solver-owned marker field x_gamma, v_gamma, n_gamma, "
            "A_gamma, region_id, traction, and force."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="IB node search",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Fluid grid nodes within a radius comparable to local grid spacing "
            "from interface triangle centroids are marked near-boundary nodes."
        ),
        current_status="partial",
        required_solver_work=(
            "Search fluid grid nodes against the current surface each step and "
            "store IB node counts in solver diagnostics."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="inside/outside classification",
        paper_section="Section 3.4",
        paper_mechanism=(
            "The sign of dot(n_face, x_node - x_face_center) separates external "
            "fluid-side IB nodes from internal solid-side nodes."
        ),
        current_status="partial",
        required_solver_work=(
            "Classify near-boundary nodes with local surface normals and expose "
            "invalid or ambiguous classifications."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="normal reconstruction",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Velocity and pressure values at IB nodes are reconstructed along "
            "the well-defined normal to the body; nearest-surface interpolation "
            "is used only when projection is not unique."
        ),
        current_status="partial",
        required_solver_work=(
            "Find boundary and interior fluid points along each local normal and "
            "record fallback counts."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="velocity Dirichlet no-slip",
        paper_section="Section 2, Equation 11; Section 3.4",
        paper_mechanism=(
            "Fluid velocity on the immersed surface equals the velocity of the "
            "moving or deforming body."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep no-slip Dirichlet rows wired to the fluid projection boundary "
            "condition phase, then connect them to the complete sharp-interface "
            "solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="pressure Neumann matrix rows",
        paper_section="Section 3.4",
        paper_mechanism=(
            "Pressure boundary conditions at IB nodes are Neumann conditions "
            "derived from normal momentum balance."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep HIBM pressure Neumann RHS row assembly wired to FV-CG fields, "
            "then connect it to the full sharp-interface solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="full-stress traction",
        paper_section="Section 2, Equation 12; Section 4",
        paper_mechanism=(
            "The full fluid stress, pressure plus viscous stress, is interpolated "
            "onto the body surface to form traction."
        ),
        current_status="partial",
        required_solver_work=(
            "Keep marker-level sigma_f = -pI + mu(grad v + grad v^T) traction "
            "sampling in the sharp-interface path and finish the complete "
            "HIBM-MPM solve loop."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="per-marker MPM external force",
        paper_section="Sections 3.2-4",
        paper_mechanism=(
            "Surface traction becomes the MPM external force through the "
            "background-grid shape functions."
        ),
        current_status="partial",
        required_solver_work=(
            "Use the sharp load assembly as the primary solid load path and "
            "finish the full HIBM-MPM step without reduced region reaction."
        ),
    ),
    HibmMpmPaperRequirement(
        requirement="surface feedback",
        paper_section="Section 4",
        paper_mechanism=(
            "After the solid solve updates material point positions and "
            "velocities, the next fluid step rebuilds IB boundary conditions "
            "from the new surface."
        ),
        current_status="partial",
        required_solver_work=(
            "Use Taichi surface-field feedback for all production MPM solid "
            "paths and validate the rebuilt next-step IB search and boundary "
            "rows in long runs."
        ),
    ),
)


_SHARP_MISSING = tuple(item.requirement for item in _PAPER_REQUIREMENTS)
_SHARP_CASE_MISSING = (
    "Phase 5 fine-nozzle validation",
)


def hibm_mpm_paper_requirements() -> tuple[dict[str, str], ...]:
    return tuple(asdict(item) for item in _PAPER_REQUIREMENTS)


def fsi_coupling_mode_report(mode: str) -> dict[str, Any]:
    mode_name = str(mode)
    if mode_name == FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED:
        return {
            "mode": mode_name,
            "solver_layer": "simulation_core",
            "implemented": True,
            "core_runner_available": False,
            "case_runner_available": True,
            "phase5_validation_complete": False,
            "legacy": True,
            "paper_hibm_mpm": False,
            "sharp_interface": False,
            "primary_coupling_variable": (
                "projected-IBM velocity residual plus reduced main/tail "
                "region-pair interface reaction"
            ),
            "main_tail_region_reaction_diagnostic_only": True,
            "legacy_projected_reduced": True,
            "not_paper_hibm_mpm": True,
            "missing": list(_SHARP_MISSING),
        }
    if mode_name == FSI_COUPLING_MODE_HIBM_MPM_SHARP:
        return {
            "mode": mode_name,
            "solver_layer": "simulation_core",
            "implemented": True,
            "core_runner_available": True,
            "case_runner_available": True,
            "phase5_validation_complete": False,
            "legacy": False,
            "paper_hibm_mpm": True,
            "sharp_interface": True,
            "primary_coupling_variable": "per-marker HIBM-MPM surface traction",
            "main_tail_region_reaction_diagnostic_only": True,
            "legacy_projected_reduced": False,
            "not_paper_hibm_mpm": False,
            "missing": list(_SHARP_CASE_MISSING),
        }
    choices = ", ".join(FSI_COUPLING_MODE_CHOICES)
    raise ValueError(f"fsi_coupling_mode must be one of: {choices}")


def require_implemented_fsi_coupling_mode(mode: str) -> dict[str, Any]:
    report = fsi_coupling_mode_report(mode)
    if not bool(report["implemented"]):
        missing = ", ".join(str(item) for item in report["missing"])
        raise NotImplementedError(
            f"{mode} is declared but not implemented in simulation_core yet; "
            f"missing solver requirements: {missing}"
        )
    return report
